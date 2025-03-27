import numpy as np
import pandas as pd
from model_training.models import CatchmentModel
import logging
import time
import os
import sys
import json
import torch
import torch.nn as nn
from data_processing import build_sliding_windows_for_subset, standardize_time_series_all, standardize_attributes
import logging 

# Import our custom tqdm that supports logging
try:
    from tqdm_logging import tqdm
except ImportError:
    from tqdm import tqdm

def calculate_river_width(Q: pd.Series) -> pd.Series:
    """
    根据流量计算河道宽度 W = aQ^b
    参考公式：lnW = 2.10 + 0.45lnQ
    
    输入：
        Q: 流量序列 (m³/s)
    输出：
        返回河道宽度序列 (m)
    """
    # 避免log(0)错误
    Q_adj = Q.replace(0, np.nan)
    
    # 使用公式 lnW = 2.10 + 0.45lnQ
    lnW = 2.10 + 0.45 * np.log(Q_adj)
    W = np.exp(lnW)
    
    # 填充可能的NaN值（对应流量为0的情况）
    return W.fillna(0.0)

def compute_temperature_factor(temperature: pd.Series, parameter: str = "TN") -> pd.Series:
    """
    计算温度调整因子 f(t) = α^(t-20)
    
    输入：
        temperature: 温度序列 (°C)
        parameter: 水质参数，"TN"或"TP"
    输出：
        返回温度调整因子序列
    """
    if parameter == "TN":
        alpha = 1.0717  # TN的α值
    else:  # TP
        alpha = 1.06    # TP的α值
    
    return np.power(alpha, temperature - 20)

def compute_nitrogen_concentration_factor(N_concentration: pd.Series) -> pd.Series:
    """
    Calculate the nitrogen concentration adjustment factor f(CN) based on specified values:
    - f(CN) = 7.2 at CN = 0.0001 mg L−1
    - f(CN) = 1 for CN = 1 mg L−1
    - f(CN) = 0.37 for CN = 100 mg L−1
    - Constant at higher concentrations
    
    Performs log-linear interpolation between the specified points.
    
    Parameters:
    -----------
    N_concentration: pd.Series
        Nitrogen concentration series (mg/L)
    
    Returns:
    --------
    pd.Series
        Concentration adjustment factor series
    """
    # Create a copy to avoid modifying the original
    result = pd.Series(index=N_concentration.index, dtype=float)
    
    # Case 1: CN ≤ 0.0001 mg L−1
    mask_lowest = N_concentration <= 0.0001
    result[mask_lowest] = 7.2
    
    # Case 2: 0.0001 < CN ≤ 1 mg L−1
    mask_low = (N_concentration > 0.0001) & (N_concentration <= 1)
    # Log-linear interpolation between (0.0001, 7.2) and (1, 1)
    log_ratio_low = (np.log10(N_concentration[mask_low]) - np.log10(0.0001)) / (np.log10(1) - np.log10(0.0001))
    result[mask_low] = 7.2 - log_ratio_low * (7.2 - 1)
    
    # Case 3: 1 < CN ≤ 100 mg L−1
    mask_mid = (N_concentration > 1) & (N_concentration <= 100)
    # Log-linear interpolation between (1, 1) and (100, 0.37)
    log_ratio_mid = (np.log10(N_concentration[mask_mid]) - np.log10(1)) / (np.log10(100) - np.log10(1))
    result[mask_mid] = 1 - log_ratio_mid * (1 - 0.37)
    
    # Case 4: CN > 100 mg L−1
    mask_high = N_concentration > 100
    result[mask_high] = 0.37
    
    return result

def compute_retainment_factor(v_f: float, Q_up: pd.Series, Q_down: pd.Series, 
                             S_up: pd.Series, S_down: pd.Series,
                             temperature: pd.Series = None,
                             N_concentration: pd.Series = None,
                             parameter: str = "TN") -> pd.Series:
    """
    计算保留系数
    R(Ωj, Ωi) = (1-exp(-v_f·S(Ωj)/(2·Q(Ωj)))) · (1-exp(-v_f·S(Ωi)/(2·Q(Ωi))))
    
    输入：
        v_f: 基础吸收速率参数 (m/yr)
        Q_up: 上游流量序列 (m³/s)
        Q_down: 下游流量序列 (m³/s)
        S_up: 上游河段宽度序列 (m)
        S_down: 下游河段宽度序列 (m)
        temperature: 温度序列 (°C)，若提供则计算温度调整因子
        N_concentration: 氮浓度序列 (mg/L)，若提供且参数为TN，则计算浓度调整因子
        parameter: 水质参数，"TN"或"TP"
    输出：
        返回保留系数序列
    """
    # 避免除以0错误
    Q_up_adj = Q_up.replace(0, np.nan)
    Q_down_adj = Q_down.replace(0, np.nan)
    
    # 应用温度调整因子（如果提供温度数据）
    if temperature is not None:
        temp_factor = compute_temperature_factor(temperature, parameter)
        v_f_adjusted = v_f * temp_factor
    else:
        v_f_adjusted = v_f
    
    # 应用浓度调整因子（如果提供浓度数据且参数为TN）
    if parameter == "TN" and N_concentration is not None:
        conc_factor = compute_nitrogen_concentration_factor(N_concentration)
        v_f_adjusted = v_f_adjusted * conc_factor
    
    # 计算保留系数
    R_up = 1 - np.exp(-v_f_adjusted * S_up / (2 * Q_up_adj))
    R_down = 1 - np.exp(-v_f_adjusted * S_down / (2 * Q_down_adj))
    R = R_up * R_down
    
    # 填充可能的NaN值
    return R.fillna(0.0)
def flow_routing_calculation(df: pd.DataFrame, 
                             iteration: int, 
                             model_func, 
                             river_info: pd.DataFrame, 
                             v_f_TN: float = 35.0,
                             v_f_TP: float = 44.5,
                             attr_dict: dict = None, 
                             model: CatchmentModel = None,
                             target_cols=["TN", "TP"],
                             attr_df: pd.DataFrame = None,
                             E_exist: int = 0,
                             E_exist_path: str = None,
                             E_save: int = 0,
                             E_save_path: str = None) -> pd.DataFrame:
    """
    汇流计算函数
    输入：
        df: 包含日尺度数据的 DataFrame，每行记录一个 COMID 在某日期的数据，
            必须包含 'COMID'、'date'、'Qout'、'temperature_2m_mean'(可选) 等字段
        iteration: 当前迭代次数，用于命名新增加的列
        model_func: 用于预测局部贡献 E 的函数，输入为单个 COMID 的 DataFrame，
                    输出为与日期对齐的 Series
        river_info: 河段信息 DataFrame，必须包含 'COMID' 和 'NextDownID'
        v_f_TN: TN的基础吸收速率参数，默认为35.0 m/yr
        v_f_TP: TP的基础吸收速率参数，默认为44.5 m/yr
        attr_dict: 河段属性字典
        model: 预测模型
        target_cols: 目标列列表，默认为 ["TN", "TP"]
        attr_df: 河段属性 DataFrame，用于识别标记为 'ERA5_exist'=0 的缺失数据河段
        E_exist: 是否从指定路径读取E值，0表示不读取，1表示读取
        E_exist_path: E值读取路径
        E_save: 是否保存计算得到的E值，0表示不保存，1表示保存
        E_save_path: E值保存路径
    输出：
        返回 DataFrame，增加了新列：
            'E_{iteration}_{param}'：局部贡献（预测值）
            'y_up_{iteration}_{param}'：上游汇流贡献
            'y_n_{iteration}_{param}'：汇流总预测值 = E + y_up
            对每个参数param（如TN、TP）分别计算
    """
    #===========================================================================
    # 初始化与准备
    # - 验证模型设备
    # - 复制输入数据框
    # - 转换日期格式
    # - 构建下游河段映射字典
    #===========================================================================
    # 验证模型是否在正确的设备上
    if model and hasattr(model, 'base_model') and hasattr(model.base_model, 'parameters'):
        device = next(model.base_model.parameters()).device
        print(f"===== MODEL DEVICE CHECK =====")
        print(f"Model is on device: {device}")
        print(f"Model type: {type(model.base_model)}")
        print(f"===============================")

    # 复制数据框以避免修改原始数据
    df = df.copy()
    logging.info(f"Flow routing calculation for iteration {iteration} started")
    logging.debug(f"DataFrame head:\n{df.head()}")
    
    # 确保日期列为datetime格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 标识缺失数据的河段
    missing_data_comids = set()
    if attr_df is not None and 'ERA5_exist' in attr_df.columns:
        # 获取 ERA5_exist=0 的河段 ID
        missing_df = attr_df[attr_df['ERA5_exist'] == 0]
        missing_data_comids = set(str(comid) for comid in missing_df['COMID'])
        logging.info(f"标识出 {len(missing_data_comids)} 个缺失数据的河段")
    
    # 从河网信息中构建下游河段映射
    next_down_ids = river_info.set_index('COMID')['NextDownID'].to_dict()
    
    # 递归查找下一个可用的下游河段（跳过缺失的河段）
    def find_next_available_downstream(comid, visited=None):
        """递归查找下一个非缺失的下游河段"""
        if visited is None:
            visited = set()
        
        if comid in visited:  # 检测循环引用
            return 0
        visited.add(comid)
        
        next_down = next_down_ids.get(comid, 0)
        if next_down == 0:  # 已到末端
            return 0
        
        if str(next_down) not in missing_data_comids:  # 下游河段有数据
            return next_down
        
        # 下游河段无数据，继续查找其下游
        return find_next_available_downstream(next_down, visited)
    
    # 绕过缺失数据的河段
    if missing_data_comids:
        # 处理所有河段，查找下一个可用下游
        bypassed_count = 0
        modified_next_down = {}
        
        for comid in list(next_down_ids.keys()):
            if str(comid) in missing_data_comids:
                continue  # 跳过处理缺失数据的河段
            
            next_down = next_down_ids.get(comid, 0)
            if next_down != 0 and str(next_down) in missing_data_comids:
                # 查找下一个可用的非缺失数据的下游
                next_available = find_next_available_downstream(next_down)
                modified_next_down[comid] = next_available
                bypassed_count += 1
        
        # 更新下游映射
        next_down_ids.update(modified_next_down)
        logging.info(f"在河网拓扑中绕过了 {bypassed_count} 个缺失数据的河段")
    
    # 检查是否有温度数据可用
    has_temperature_data = 'temperature_2m_mean' in df.columns
    if has_temperature_data:
        logging.info("Temperature data available, will apply temperature adjustment")
    else:
        logging.info("No temperature data available, using base settling velocities")
    
    # 按河段ID分组并排序，为每个河段创建时间序列
    groups = {comid: group.sort_values("date").copy() for comid, group in df.groupby("COMID")}
    comid_data = {}

    #===========================================================================
    # 第一阶段：批量计算每个河段的局部贡献 E（针对各个水质参数）
    # - 如果E_exist=1，从文件加载E值
    # - 否则，使用模型预测函数计算E值
    # - 如果E_save=1，保存计算得到的E值
    # - 为每个河段初始化y_up和y_n
    # - 监控并管理GPU内存使用
    #===========================================================================
    # 准备E值存储目录
    if E_save == 1 and E_save_path:
        os.makedirs(E_save_path, exist_ok=True)
        logging.info(f"将保存E值到目录: {E_save_path}")
    
    # 第一步：处理E值（加载或计算）
    if E_exist == 1 and E_exist_path:
        logging.info(f"从 {E_exist_path} 读取E值")
        print(f"从 {E_exist_path} 读取E值")
        
        # 构建保存E值的文件名，以迭代次数和参数为基础
        e_values_dict = {}
        
        # 检查指定路径是否为目录
        if os.path.isdir(E_exist_path):
            # 对每个参数，查找对应的E值文件
            for param in target_cols:
                e_file_path = os.path.join(E_exist_path, f"E_{iteration}_{param}.csv")
                if os.path.exists(e_file_path):
                    try:
                        e_df = pd.read_csv(e_file_path)
                        if 'COMID' in e_df.columns and 'Date' in e_df.columns and 'E_value' in e_df.columns:
                            e_df['Date'] = pd.to_datetime(e_df['Date'])
                            e_values_dict[param] = e_df
                            logging.info(f"成功加载参数 {param} 的E值，共 {len(e_df)} 条记录")
                        else:
                            logging.error(f"E值文件 {e_file_path} 格式不正确，应包含COMID、Date和E_value列")
                    except Exception as e:
                        logging.error(f"读取E值文件 {e_file_path} 失败: {str(e)}")
                else:
                    logging.warning(f"找不到参数 {param} 的E值文件: {e_file_path}")
            
            # 检查是否成功加载了所有参数的E值
            if len(e_values_dict) < len(target_cols):
                logging.warning(f"只加载了 {len(e_values_dict)}/{len(target_cols)} 个参数的E值，对缺失参数将使用模型计算")
        else:
            logging.error(f"E值路径 {E_exist_path} 不是一个目录")
        
        # 如果成功加载了E值，则为每个河段设置E值
        if e_values_dict:
            for comid, group in tqdm(groups.items(), desc=f"设置E值 (iteration {iteration})"):
                # 跳过缺失数据的河段
                if str(comid) in missing_data_comids:
                    continue
                
                # 计算河道宽度
                group['width'] = calculate_river_width(group['Qout'])
                
                # 对每个参数，设置E值
                for param in target_cols:
                    if param in e_values_dict:
                        # 从加载的E值中查找对应的记录
                        e_df = e_values_dict[param]
                        comid_e_df = e_df[e_df['COMID'] == comid]
                        
                        if not comid_e_df.empty:
                            # 将E值设置到对应的日期
                            e_series = pd.Series(index=group['date'])
                            for _, row in comid_e_df.iterrows():
                                date_val = row['Date']
                                if date_val in e_series.index:
                                    e_series[date_val] = row['E_value']
                            
                            # 填充缺失的E值为0
                            e_series = e_series.fillna(0.0)
                            
                            # 设置E值、y_up和y_n
                            group[f'E_{param}'] = e_series.values
                        else:
                            # 如果没有找到对应的E值记录，设置为0
                            group[f'E_{param}'] = 0.0
                    else:
                        # 如果没有加载该参数的E值，设置为0
                        group[f'E_{param}'] = 0.0
                    
                    # 初始化y_up和y_n
                    group[f'y_up_{param}'] = 0.0
                    group[f'y_n_{param}'] = 0.0
                
                # 设置索引并保存到数据字典
                group = group.set_index("date")
                comid_data[comid] = group
    else:
        # 没有加载E值，使用模型计算
        logging.info(f"使用模型计算河段E值 (iteration {iteration})")
        
        # 需要保存的E值
        e_values_to_save = {param: [] for param in target_cols} if E_save == 1 else None
        
        logging.info(f"Processing {len(groups)} river segments in batches...")
        
        # 设定批次大小并计算批次数
        batch_size = 1000  # 每批处理1000个COMID
        comid_list = list(groups.keys())
        num_batches = (len(comid_list) + batch_size - 1) // batch_size
        
        # 使用进度条处理所有河段
        with tqdm(total=len(groups), desc=f"Processing river segments for iteration {iteration}") as pbar:
            for batch_idx in range(num_batches):
                batch_start_time = time.time()
                
                # 获取当前批次的河段ID
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(comid_list))
                batch_comids = comid_list[start_idx:end_idx]
                
                # 过滤掉缺失数据的河段
                valid_batch_comids = [comid for comid in batch_comids if str(comid) not in missing_data_comids]
                if len(valid_batch_comids) < len(batch_comids):
                    logging.debug(f"批次 {batch_idx+1}: 过滤掉 {len(batch_comids) - len(valid_batch_comids)} 个缺失数据的河段")
                
                # 批量计算当前批次所有有效河段的E值
                batch_results = model_func(valid_batch_comids, groups, attr_dict, model, target_cols)
                
                # 处理结果并存入数据字典
                for comid in batch_comids:
                    # 跳过缺失数据的河段
                    if str(comid) in missing_data_comids:
                        continue
                        
                    group = groups[comid]
                    E_series = batch_results.get(comid)
                    
                    if E_series is None:
                        logging.warning(f"河段 {comid} 的模型结果为 None，设置为 0")
                        # 为此河段的所有时间设置 E 为 0
                        for param in target_cols:
                            group[f'E_{param}'] = 0.0
                            group[f'y_up_{param}'] = 0.0
                            group[f'y_n_{param}'] = 0.0
                    else:
                        # 检查Qout是否存在异常值
                        if group['Qout'].isnull().values.any():
                            print(f"Qout存在零值，comid={comid}")
                            sys.exit()
                        if (group['Qout']<0).values.any():
                            print(f"Qout存在负值，comid={comid}")
                            ##统计Qout负值的比例
                            print(group['Qout'][group['Qout']<0].count()/group['Qout'].count())
                            sys.exit()

                        # 计算河道宽度
                        group['width'] = calculate_river_width(group['Qout'])
                        
                        # 初始化所有参数的E、y_up和y_n值
                        # 如果结果包含多个目标参数（如TN和TP），分别处理
                        if isinstance(E_series, pd.DataFrame) and len(target_cols) > 1:
                            for param_idx, param in enumerate(target_cols):
                                if param in E_series.columns:
                                    group[f'E_{param}'] = E_series[param].values
                                    group[f'y_up_{param}'] = 0.0
                                    group[f'y_n_{param}'] = 0.0
                                    
                                    # 如果需要保存E值
                                    if E_save == 1:
                                        for date_idx, date_val in enumerate(group['date']):
                                            e_values_to_save[param].append({
                                                'COMID': comid,
                                                'Date': date_val,
                                                'E_value': E_series[param].values[date_idx] if date_idx < len(E_series[param]) else 0.0
                                            })
                        else:
                            # 单一参数情况
                            param = target_cols[0]
                            group[f'E_{param}'] = E_series.values
                            group[f'y_up_{param}'] = 0.0
                            group[f'y_n_{param}'] = 0.0
                            
                            # 如果需要保存E值
                            if E_save == 1:
                                for date_idx, date_val in enumerate(group['date']):
                                    if date_idx < len(E_series):
                                        e_values_to_save[param].append({
                                            'COMID': comid,
                                            'Date': date_val,
                                            'E_value': E_series.values[date_idx]
                                        })
                    
                    group = group.set_index("date")
                    comid_data[comid] = group
                
                # 计算并记录批次处理时间
                batch_time = time.time() - batch_start_time
                
                # 定期记录批次处理信息和内存使用情况
                if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                    logging.info(f"Batch {batch_idx+1}/{num_batches}: Processed {len(valid_batch_comids)} COMIDs in {batch_time:.2f}s")
                    
                    # 监控GPU内存使用
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                        logging.info(f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
                        
                        # 内存占用过高时释放缓存
                        if allocated > 6000:  # 6GB阈值
                            torch.cuda.empty_cache()
                            logging.info("Cleared GPU cache")
                
                # 更新进度条
                pbar.update(len(batch_comids))
        
        # 如果需要保存E值
        if E_save == 1 and E_save_path:
            logging.info(f"保存E值到 {E_save_path}")
            # 为每个参数保存E值
            for param in target_cols:
                if param in e_values_to_save:
                    e_df = pd.DataFrame(e_values_to_save[param])
                    if not e_df.empty:
                        e_file_path = os.path.join(E_save_path, f"E_{iteration}_{param}.csv")
                        e_df.to_csv(e_file_path, index=False)
                        logging.info(f"已保存参数 {param} 的E值，共 {len(e_df)} 条记录，保存至 {e_file_path}")

    #===========================================================================
    # 第二阶段：构建河网拓扑结构
    # - 计算每个河段的入度（上游河段数量）
    # - 初始化用于累积上游贡献的数据结构
    # - 处理头部河段（无上游的河段）
    #===========================================================================
    logging.info("Calculating node indegrees...")
    # 计算入度：若某个河段ID出现在其他河段的NextDownID中，则其入度增加
    indegree = {comid: 0 for comid in comid_data.keys()}
    for comid in comid_data.keys():
        next_down = next_down_ids.get(comid, 0)
        if next_down != 0 and next_down in indegree:
            indegree[next_down] = indegree.get(next_down, 0) + 1

    # 为每个参数创建负荷累加器
    load_acc = {}
    for param in target_cols:
        load_acc[param] = {comid: pd.Series(0.0, index=data.index) for comid, data in comid_data.items()}

    # 找出所有头部河段（入度为0）并初始化其y_n值为局部贡献E
    queue = [comid for comid, deg in indegree.items() if deg == 0]
    logging.info(f"Found {len(queue)} headwater segments")
    for comid in queue:
        data = comid_data[comid]
        for param in target_cols:
            data[f'y_n_{param}'] = data[f'E_{param}']
        comid_data[comid] = data

    #===========================================================================
    # 第三阶段：执行汇流计算（拓扑排序算法）
    # - 按不同水质参数（TN/TP）分别计算
    # - 使用队列按拓扑顺序处理河段
    # - 计算每个河段对下游的贡献

    # - 跟踪入度变化并将处理完成的上游河段的下游加入队列
    #===========================================================================
    logging.info("Starting flow routing calculation...")
    processed_count = 0
    while queue:
        # 从队列中取出下一个要处理的河段
        current = queue.pop(0)
        processed_count += 1
        if processed_count % 1000 == 0:
            logging.info(f"Processed {processed_count} segments so far")
        
        # 获取当前河段数据和下游河段ID
        current_data = comid_data[current]
        next_down = next_down_ids.get(current, 0)
        if next_down == 0 or next_down not in comid_data:
            continue  # 如果没有下游或下游不在数据中，跳过
            
        # 获取下游河段数据并找出共同日期
        down_data = comid_data[next_down]
        common_dates = current_data.index.intersection(down_data.index)
        
        # 检查日期对齐问题
        if len(common_dates) == 0:
            logging.warning(f"Warning: 日期不对齐，COMID {current} 与 COMID {next_down}")
            logging.warning(f"  当前日期: {current_data.index}")
            logging.warning(f"  下游日期: {down_data.index}")
            
        # 计算当前河段对下游的贡献并累加到下游负荷累加器
        if len(common_dates) > 0:
            # 提取共同日期的数据
            Q_current = current_data['Qout'].reindex(common_dates)
            Q_down = down_data['Qout'].reindex(common_dates)
            S_current = current_data['width'].reindex(common_dates)
            S_down = down_data['width'].reindex(common_dates)
            
            # 获取温度数据（如果可用）
            if has_temperature_data:
                temperature = current_data['temperature_2m_mean'].reindex(common_dates)
            else:
                temperature = None
            
            # 为每个参数分别计算贡献
            for param in target_cols:
                # 选择合适的吸收速率参数
                if param == "TN":
                    v_f = v_f_TN
                    # 获取TN浓度数据（如果可用）
                    if f'y_n_{param}' in current_data.columns:
                        N_concentration = current_data[f'y_n_{param}'].reindex(common_dates)
                    else:
                        N_concentration = None
                elif param == "TP":
                    v_f = v_f_TP
                    N_concentration = None
                else:
                    v_f = v_f_TN  # 默认使用TN的参数
                    N_concentration = None
                
                # 提取当前参数的y_n值
                y_n_current = current_data[f'y_n_{param}'].reindex(common_dates)
                
                # 计算保留系数
                R_series = compute_retainment_factor(
                    v_f=v_f, 
                    Q_up=Q_current, 
                    Q_down=Q_down,
                    S_up=S_current,
                    S_down=S_down,
                    temperature=temperature,
                    N_concentration=N_concentration,
                    parameter=param
                )
                
                # 计算贡献并累加到下游负荷累加器
                contribution = y_n_current * R_series * Q_current
                load_acc[param][next_down] = load_acc[param][next_down].add(contribution, fill_value=0.0)
            
        # 减少下游河段的入度
        indegree[next_down] -= 1
         
        # 如果下游河段所有上游都已处理完毕，计算其y_up和y_n并加入队列
        if indegree[next_down] == 0:
            down_data = comid_data[next_down]
            
            # 为每个参数分别计算y_up和y_n
            for param in target_cols:
                # 计算上游贡献浓度
                y_up_down = load_acc[param][next_down] / down_data['Qout'].replace(0, np.nan)
                y_up_down = y_up_down.fillna(0.0)
                
                # 更新下游河段的y_up和y_n
                down_data[f'y_up_{param}'] = y_up_down
                down_data[f'y_n_{param}'] = down_data[f'E_{param}'] + down_data[f'y_up_{param}']
            
            # 更新数据字典
            comid_data[next_down] = down_data
            
            # 将下游河段加入队列
            queue.append(next_down)

    #===========================================================================
    # 第四阶段：结果整合与格式化
    # - 将所有河段的数据合并为单一数据框
    # - 重命名列以包含迭代编号
    # - 返回最终结果
    #===========================================================================
    logging.info("Merging results...")
    # 收集所有河段的数据
    result_list = []
    for comid, data in comid_data.items():
        temp = data.reset_index()
        temp['COMID'] = comid
        result_list.append(temp)
        
    # 合并为单一数据框
    result_df = pd.concat(result_list, ignore_index=True)
    
    # 重命名列以包含迭代编号
    rename_dict = {}
    for param in target_cols:
        rename_dict[f'E_{param}'] = f'E_{iteration}_{param}'
        rename_dict[f'y_up_{param}'] = f'y_up_{iteration}_{param}'
        rename_dict[f'y_n_{param}'] = f'y_n_{iteration}_{param}'

    # Make sure these columns exist before attempting to rename
    missing_cols = [col for col in rename_dict.keys() if col not in result_df.columns]
    if missing_cols:
        logging.warning(f"Warning: Some columns to rename don't exist: {missing_cols}")
        # Only keep the valid columns in the rename dictionary
        rename_dict = {k: v for k, v in rename_dict.items() if k in result_df.columns}   

    result_df = result_df.rename(columns=rename_dict)

    # Add debug log to show column names after renaming
    logging.info(f"Columns after renaming: {result_df.columns.tolist()}")   
    
    logging.info(f"Flow routing calculation for iteration {iteration} complete")
    return result_df