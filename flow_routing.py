import numpy as np
import pandas as pd
from PGRWQI.model_training.models.models import CatchmentModel
import logging
import time
import os
import sys
import json
import torch
import torch.nn as nn
from data_processing import build_sliding_windows_for_subset, standardize_time_series_all, standardize_attributes

# 尝试导入支持日志记录的tqdm，如果不可用则使用标准tqdm
try:
    from tqdm_logging import tqdm
except ImportError:
    from tqdm import tqdm

# ============================================================================
# 河道几何模块: 处理河段长度、宽度等几何属性
# ============================================================================
def get_river_length(comid, attr_dict, river_info=None):
    """
    获取指定COMID的河段长度
    
    参数:
    -----------
    comid : str 或 int
        河段的COMID
    attr_dict : dict
        COMID到属性数组的映射字典
    river_info : pd.DataFrame, 可选
        包含河网信息的DataFrame
        
    返回:
    --------
    float
        河段长度(km)
    """
    comid_str = str(comid)
    
    # 首先尝试从attr_dict获取长度
    if comid_str in attr_dict:
        # 获取该COMID的所有属性值
        attrs = attr_dict[comid_str]
        
        # 检查'lengthkm'是否在属性字典中
        if isinstance(attrs, dict) and 'lengthkm' in attrs:
            return float(attrs['lengthkm'])
    
    # 如果提供了river_info，尝试从中获取长度
    if river_info is not None and 'COMID' in river_info.columns and 'lengthkm' in river_info.columns:
        length_data = river_info[river_info['COMID'] == comid]
        if not length_data.empty:
            return float(length_data['lengthkm'].iloc[0])
    
    # 如果找不到COMID，使用默认长度
    logging.warning(f"未找到COMID {comid}的长度，使用默认值")
    return 1.0  # 默认长度1km
    
def calculate_river_width(Q: pd.Series) -> pd.Series:
    """
    根据流量计算河道宽度 W = aQ^b
    参考公式：lnW = 2.10 + 0.45lnQ
    
    参数:
        Q: 流量序列 (m³/s)
    返回:
        河道宽度序列 (m)
    """
    # 避免log(0)错误
    Q_adj = Q.replace(0, np.nan)
    
    # 使用公式 lnW = 2.10 + 0.45lnQ
    lnW = 2.10 + 0.45 * np.log(Q_adj)
    W = np.exp(lnW)
    
    # 填充可能的NaN值（对应流量为0的情况）
    return W.fillna(0.0)

# ============================================================================
# 环境调整因子模块: 处理温度、浓度等环境因素对水质影响的调整因子
# ============================================================================
def compute_temperature_factor(temperature: pd.Series, parameter: str = "TN") -> pd.Series:
    """
    计算温度调整因子 f(t) = α^(t-20) * (t - 20)
    
    参数:
        temperature: 温度序列 (°C)
        parameter: 水质参数，"TN"或"TP"
    返回:
        温度调整因子序列
    """
    if parameter == "TN":
        alpha = 1.0717  # TN的α值
    else:  # TP
        alpha = 1.06    # TP的α值
    
    # 计算 α^(t-20) * (t - 20) 
    return np.power(alpha, temperature - 20) * (temperature - 20)

def compute_nitrogen_concentration_factor(N_concentration: pd.Series) -> pd.Series:
    """
    计算氮浓度调整因子 f(CN)，基于以下指定值:
    - 当CN = 0.0001 mg/L时，f(CN) = 7.2
    - 当CN = 1 mg/L时，f(CN) = 1
    - 当CN = 100 mg/L时，f(CN) = 0.37
    - 更高浓度时保持不变
    
    在指定点之间使用对数线性插值。
    
    参数:
        N_concentration: 氮浓度序列 (mg/L)
    
    返回:
        浓度调整因子序列
    """
    # 创建副本以避免修改原始数据
    result = pd.Series(index=N_concentration.index, dtype=float)
    
    # 情况1: CN ≤ 0.0001 mg/L
    mask_lowest = N_concentration <= 0.0001
    result[mask_lowest] = 7.2
    
    # 情况2: 0.0001 < CN ≤ 1 mg/L
    mask_low = (N_concentration > 0.0001) & (N_concentration <= 1)
    # (0.0001, 7.2) 和 (1, 1) 之间的对数线性插值
    log_ratio_low = (np.log10(N_concentration[mask_low]) - np.log10(0.0001)) / (np.log10(1) - np.log10(0.0001))
    result[mask_low] = 7.2 - log_ratio_low * (7.2 - 1)
    
    # 情况3: 1 < CN ≤ 100 mg/L
    mask_mid = (N_concentration > 1) & (N_concentration <= 100)
    # (1, 1) 和 (100, 0.37) 之间的对数线性插值
    log_ratio_mid = (np.log10(N_concentration[mask_mid]) - np.log10(1)) / (np.log10(100) - np.log10(1))
    result[mask_mid] = 1 - log_ratio_mid * (1 - 0.37)
    
    # 情况4: CN > 100 mg/L
    mask_high = N_concentration > 100
    result[mask_high] = 0.37
    
    return result

def compute_retainment_factor(v_f: float, Q_up: pd.Series, Q_down: pd.Series, 
                             W_up: pd.Series, W_down: pd.Series,
                             length_up: float, length_down: float,
                             temperature: pd.Series = None,
                             N_concentration: pd.Series = None,
                             parameter: str = "TN",
                             debug_info=None) -> pd.Series:
    """
    计算保留系数，并存储中间结果用于调试
    R(Ωj, Ωi) = (1-exp(-v_f·S(Ωj)/(2·Q(Ωj)))) · (1-exp(-v_f·S(Ωi)/(2·Q(Ωi))))

    参数:
        v_f: 基础吸收速率参数 (m/yr)
        Q_up: 上游流量序列 (m³/s)
        Q_down: 下游流量序列 (m³/s)
        W_up: 上游河段宽度序列 (m)
        W_down: 下游河段宽度序列 (m)
        length_up: 上游河段长度 (km)
        length_down: 下游河段长度 (km)
        temperature: 温度序列 (°C)，若提供则计算温度调整因子
        N_concentration: 氮浓度序列 (mg/L)，若提供且参数为TN，则计算浓度调整因子
        parameter: 水质参数，"TN"或"TP"
        debug_info: 存储中间结果的字典，若为None则不存储

    返回:
        保留系数序列
    """
    # 复制原始流量用于调试
    Q_up_orig = Q_up.copy()
    Q_down_orig = Q_down.copy()
    
    # 避免除以0错误，设置最小流量阈值
    min_flow = 0.001  # 最小流量阈值，1 L/s
    Q_up_adj = Q_up.replace(0, min_flow).clip(lower=min_flow)
    Q_down_adj = Q_down.replace(0, min_flow).clip(lower=min_flow)
    
    # 记录有多少值被裁剪
    up_clipped = (Q_up < min_flow).sum()
    down_clipped = (Q_down < min_flow).sum()
    if up_clipped > 0 or down_clipped > 0:
        logging.info(f"流量裁剪: {up_clipped}个上游和{down_clipped}个下游值 < {min_flow}")
    
    # 单位转换:
    # 1. 长度从km转成m
    length_up_m = length_up * 1000.0  # km -> m
    length_down_m = length_down * 1000.0  # km -> m
    
    # 2. v_f从m/yr转成m/s
    seconds_per_year = 365.25 * 24 * 60 * 60  # 31,557,600秒
    v_f_m_per_second = v_f / seconds_per_year  # m/yr -> m/s
    
    # 存储基础v_f值用于调试
    base_v_f = v_f_m_per_second
    
    # 应用温度调整因子（如果提供温度数据）
    if temperature is not None:
        # 使用已有函数计算温度因子
        temp_factor = compute_temperature_factor(temperature, parameter)
        v_f_adjusted = v_f_m_per_second * temp_factor
    else:
        temp_factor = None
        v_f_adjusted = v_f_m_per_second
    
    # 应用浓度调整因子（如果提供浓度数据且参数为TN）
    if parameter == "TN" and N_concentration is not None:
        # 使用已有函数计算浓度因子
        conc_factor = compute_nitrogen_concentration_factor(N_concentration)
        v_f_adjusted = v_f_adjusted * conc_factor
    else:
        conc_factor = None
    
    # 计算河段面积 (宽度*长度)
    S_up = W_up * length_up_m  # 面积单位：m²
    S_down = W_down * length_down_m  # 面积单位：m²
    
    # 计算指数项（原始值，用于调试）
    exp_up_raw = -v_f_adjusted * S_up / (2 * Q_up_adj)
    exp_down_raw = -v_f_adjusted * S_down / (2 * Q_down_adj)
    
    # 检查极端值并记录
    extreme_up = (exp_up_raw < -50).sum() + (exp_up_raw > 50).sum()
    extreme_down = (exp_down_raw < -50).sum() + (exp_down_raw > 50).sum()
    if extreme_up > 0 or extreme_down > 0:
        logging.warning(f"检测到极端指数值: {extreme_up}个上游, {extreme_down}个下游")
        
    # 裁剪极端值以防止溢出
    exp_up = exp_up_raw.clip(-50, 50)
    exp_down = exp_down_raw.clip(-50, 50)
    
    # 计算保留系数
    R_up = 1 - np.exp(exp_up)
    R_down = 1 - np.exp(exp_down)
    R = R_up * R_down
    
    # 检查NaN和无穷大值
    nan_count = R.isna().sum()
    inf_mask = ~R.isna() & (R.abs() == float('inf'))
    inf_count = inf_mask.sum()
    if nan_count > 0 or inf_count > 0:
        logging.warning(f"保留系数中发现{nan_count}个NaN和{inf_count}个无穷大值")
        
        # 替换无穷大值为有限值
        if inf_count > 0:
            R.loc[inf_mask] = np.sign(R.loc[inf_mask]) * 0.99  # 替换为接近1的值，保持符号
    
    # 如果提供debug_info字典，存储中间结果
    if debug_info is not None and isinstance(debug_info, dict):
        debug_info['Q_up_orig'] = Q_up_orig
        debug_info['Q_down_orig'] = Q_down_orig
        debug_info['Q_up_adj'] = Q_up_adj
        debug_info['Q_down_adj'] = Q_down_adj
        debug_info['W_up'] = W_up
        debug_info['W_down'] = W_down
        debug_info['S_up'] = S_up
        debug_info['S_down'] = S_down
        debug_info['base_v_f'] = base_v_f
        debug_info['v_f_adjusted'] = v_f_adjusted
        debug_info['exp_up_raw'] = exp_up_raw
        debug_info['exp_down_raw'] = exp_down_raw
        debug_info['exp_up'] = exp_up
        debug_info['exp_down'] = exp_down
        debug_info['R_up'] = R_up
        debug_info['R_down'] = R_down
        
        if temp_factor is not None:
            debug_info['temp_factor'] = temp_factor
        if conc_factor is not None:
            debug_info['conc_factor'] = conc_factor
    
    # 填充可能的NaN值
    R_filled = R.fillna(0.0)
    
    return R_filled

# ============================================================================
# 河网拓扑模块: 处理河网拓扑结构相关的功能
# ============================================================================
def build_river_network_topology(river_info, missing_data_comids=None):
    """
    构建河网拓扑结构，处理缺失数据河段的绕过路径
    
    参数:
        river_info: 包含河网信息的DataFrame
        missing_data_comids: 缺失数据的河段ID集合
    
    返回:
        next_down_ids: 更新后的下游河段映射字典
    """
    if missing_data_comids is None:
        missing_data_comids = set()
        
    # 从河网信息中构建下游河段映射
    next_down_ids = river_info.set_index('COMID')['NextDownID'].to_dict()
    
    # 如果有缺失数据的河段，处理绕过路径
    if missing_data_comids:
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
    
    return next_down_ids

def calculate_indegrees(comids, next_down_ids):
    """
    计算河网中每个节点的入度（上游河段数量）
    
    参数:
        comids: 河段ID列表
        next_down_ids: 下游河段映射字典
    
    返回:
        indegree: 每个河段的入度字典
    """
    indegree = {comid: 0 for comid in comids}
    
    for comid in comids:
        next_down = next_down_ids.get(comid, 0)
        if next_down != 0 and next_down in indegree:
            indegree[next_down] = indegree.get(next_down, 0) + 1
            
    return indegree

# ============================================================================
# E值处理模块: 处理局部汇水区贡献的E值
# ============================================================================
def load_e_values(e_exist_path, iteration, target_col):
    """
    从文件加载E值
    
    参数:
        e_exist_path: E值读取路径
        iteration: 当前迭代次数
        target_col: 目标列
    
    返回:
        e_df: 加载的E值DataFrame，如果未找到则返回None
    """
    # 检查指定路径是否为目录
    if not os.path.isdir(e_exist_path):
        logging.error(f"E值路径 {e_exist_path} 不是一个目录")
        return None
    
    # 查找对应的E值文件
    e_file_path = os.path.join(e_exist_path, f"E_{iteration}_{target_col}.csv")
    
    if os.path.exists(e_file_path):
        try:
            e_df = pd.read_csv(e_file_path)
            if 'COMID' in e_df.columns and 'Date' in e_df.columns and 'E_value' in e_df.columns:
                e_df['Date'] = pd.to_datetime(e_df['Date'])
                logging.info(f"成功加载参数 {target_col} 的E值，共 {len(e_df)} 条记录")
                return e_df
            else:
                logging.error(f"E值文件 {e_file_path} 格式不正确，应包含COMID、Date和E_value列")
                return None
        except Exception as e:
            logging.error(f"读取E值文件 {e_file_path} 失败: {str(e)}")
            return None
    else:
        logging.warning(f"找不到参数 {target_col} 的E值文件: {e_file_path}")
        return None
    
def apply_e_values(groups, comid_data, e_df, target_col, missing_data_comids):
    """
    为河段应用已加载的E值
    
    参数:
        groups: 按COMID分组的数据字典
        comid_data: 存储处理结果的字典
        e_df: 加载的E值DataFrame
        target_col: 目标列
        missing_data_comids: 缺失数据的河段ID集合
    
    返回:
        更新后的comid_data字典
    """
    for comid, group in tqdm(groups.items(), desc=f"设置E值"):
        # 跳过缺失数据的河段
        if str(comid) in missing_data_comids:
            continue
        
        # 计算河道宽度
        group['width'] = calculate_river_width(group['Qout'])
        
        # 设置E值
        if e_df is not None:
            # 从加载的E值中查找对应的记录
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
                
                # 设置E值
                group[f'E_{target_col}'] = e_series.values
            else:
                # 如果没有找到对应的E值记录，设置为0
                group[f'E_{target_col}'] = 0.0
        else:
            # 如果没有加载E值，设置为0
            group[f'E_{target_col}'] = 0.0
        
        # 初始化y_up和y_n
        group[f'y_up_{target_col}'] = 0.0
        group[f'y_n_{target_col}'] = 0.0
        
        # 设置索引并保存到数据字典
        group = group.set_index("date")
        comid_data[comid] = group
    
    return comid_data

def calculate_e_values(groups, comid_data, model_func, attr_dict, model, all_target_cols,target_col, missing_data_comids, 
                     iteration, e_save, e_save_path):
    """
    使用模型计算E值
    
    参数:
        groups: 按COMID分组的数据字典
        comid_data: 存储处理结果的字典
        model_func: 预测模型函数
        attr_dict: 河段属性字典
        model: 预测模型
        target_col: 目标列
        missing_data_comids: 缺失数据的河段ID集合
        iteration: 当前迭代次数
        e_save: 是否保存E值
        e_save_path: E值保存路径
    
    返回:
        更新后的comid_data字典
    """
    # 需要保存的E值
    e_values_to_save = [] if e_save == 1 else None
    
    logging.info(f"处理 {len(groups)} 个河段，批量计算...")
    
    # 设定批次大小并计算批次数
    batch_size = 1000  # 每批处理1000个COMID
    comid_list = list(groups.keys())
    num_batches = (len(comid_list) + batch_size - 1) // batch_size
    
    # 使用进度条处理所有河段
    with tqdm(total=len(groups), desc=f"处理河段，迭代 {iteration}") as pbar:
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
            batch_results = model_func(valid_batch_comids, groups, attr_dict, model,all_target_cols = all_target_cols, target_col=target_col)
            
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
                    group[f'E_{target_col}'] = 0.0
                    group[f'y_up_{target_col}'] = 0.0
                    group[f'y_n_{target_col}'] = 0.0
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
                    
                    # 设置E值、y_up和y_n值
                    group[f'E_{target_col}'] = E_series.values
                    group[f'y_up_{target_col}'] = 0.0
                    group[f'y_n_{target_col}'] = 0.0
                    
                    # 如果需要保存E值
                    if e_save == 1:
                        for date_idx, date_val in enumerate(group['date']):
                            if date_idx < len(E_series):
                                e_values_to_save.append({
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
                logging.info(f"批次 {batch_idx+1}/{num_batches}: 处理了 {len(valid_batch_comids)} 个COMID，用时 {batch_time:.2f}秒")
                
                # 监控GPU内存使用
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                    logging.info(f"GPU内存: {allocated:.2f}MB已分配, {reserved:.2f}MB已保留")
                    
                    # 内存占用过高时释放缓存
                    if allocated > 6000:  # 6GB阈值
                        torch.cuda.empty_cache()
                        logging.info("已清理GPU缓存")
            
            # 更新进度条
            pbar.update(len(batch_comids))
    
    # 如果需要保存E值
    if e_save == 1 and e_save_path and e_values_to_save:
        logging.info(f"保存E值到 {e_save_path}")
        # 确保目录存在
        os.makedirs(e_save_path, exist_ok=True)
        
        # 保存E值
        e_df = pd.DataFrame(e_values_to_save)
        if not e_df.empty:
            e_file_path = os.path.join(e_save_path, f"E_{iteration}_{target_col}.csv")
            e_df.to_csv(e_file_path, index=False)
            logging.info(f"已保存参数 {target_col} 的E值，共 {len(e_df)} 条记录，保存至 {e_file_path}")
    
    return comid_data

# ============================================================================
# 汇流计算模块: 执行河网汇流计算的核心功能
# ============================================================================
def process_headwater_segments(comid_data, queue, target_col):
    """
    处理头部河段（入度为0的河段）
    
    参数:
        comid_data: 存储河段数据的字典
        queue: 待处理河段队列
        target_col: 目标列
    
    返回:
        更新后的comid_data
    """
    logging.info(f"处理 {len(queue)} 个头部河段")
    
    for comid in queue:
        data = comid_data[comid]
        data[f'y_n_{target_col}'] = data[f'E_{target_col}']
        comid_data[comid] = data
    
    return comid_data

def execute_flow_routing(comid_data, queue, indegree, next_down_ids, load_acc, 
                        target_col, attr_dict, river_info, v_f_TN, v_f_TP, 
                        store_debug_info=True):
    """
    执行汇流计算的主要逻辑，并存储关键中间结果
    
    参数:
        comid_data: 存储河段数据的字典
        queue: 待处理河段队列
        indegree: 每个河段的入度
        next_down_ids: 下游河段映射
        load_acc: 负荷累加器
        target_col: 目标列
        attr_dict: 河段属性字典
        river_info: 河网信息DataFrame
        v_f_TN: TN的吸收速率参数
        v_f_TP: TP的吸收速率参数
        store_debug_info: 是否存储调试信息
    
    返回:
        更新后的comid_data和调试信息
    """
    logging.info("开始汇流计算...")
    processed_count = 0
    has_temperature_data = any('temperature_2m_mean' in data.columns for data in comid_data.values())
    
    # 存储调试信息的字典
    debug_segments = {}
    
    # 设置调试标志点和周期
    debug_segment_ids = set()  # 特别需要调试的河段ID集合
    debug_interval = 1000      # 每处理多少个河段记录一次详细信息
    problem_segments = []      # 记录出现问题的河段
    
    while queue:
        # 从队列中取出下一个要处理的河段
        current = queue.pop(0)
        processed_count += 1
        
        # 周期性地记录处理进度
        if processed_count % debug_interval == 0:
            logging.info(f"已处理 {processed_count} 个河段")
        
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
            logging.warning(f"警告: 日期不对齐，COMID {current} 与 COMID {next_down}")
            continue
        
        # 调试标志：是否为需要特别关注的河段
        is_debug_segment = current in debug_segment_ids or next_down in debug_segment_ids
        
        # 计算当前河段对下游的贡献
        if len(common_dates) > 0: 
            # 提取共同日期的数据
            Q_current = current_data['Qout'].reindex(common_dates)
            Q_down = down_data['Qout'].reindex(common_dates)
            S_current = current_data['width'].reindex(common_dates)
            S_down = down_data['width'].reindex(common_dates)
            
            # 获取温度数据（如果可用）
            if has_temperature_data and 'temperature_2m_mean' in current_data.columns:
                temperature = current_data['temperature_2m_mean'].reindex(common_dates)
            else:
                temperature = None
            # 初始化调试信息字典
            segment_debug = {} if store_debug_info else None
            
            try:
                # 选择合适的吸收速率参数
                if target_col == "TN":
                    v_f = v_f_TN
                    # 获取TN浓度数据（如果可用）
                    if f'y_n_{target_col}' in current_data.columns:
                        N_concentration = current_data[f'y_n_{target_col}'].reindex(common_dates)
                    else:
                        N_concentration = None
                elif target_col == "TP":
                    v_f = v_f_TP
                    N_concentration = None
                else:
                    v_f = v_f_TN  # 默认使用TN的参数
                    N_concentration = None
                
                # 提取当前参数的y_n值
                y_n_current = current_data[f'y_n_{target_col}'].reindex(common_dates)
                
                # 检查y_n_current是否包含NaN或异常值
                nan_count = y_n_current.isna().sum()
                extreme_count = (~y_n_current.isna() & (y_n_current.abs() > 1e6)).sum()
                if nan_count > 0 or extreme_count > 0:
                    logging.warning(f"COMID {current} 的 y_n_{target_col} 包含 {nan_count} 个NaN和 {extreme_count} 个异常值")
                    if extreme_count > 0:
                        # 裁剪极端值
                        y_n_current = y_n_current.clip(-1e6, 1e6)
                        logging.info(f"已将 COMID {current} 的 y_n_{target_col} 极端值裁剪到 ±1e6 范围内")

                # 获取河段长度
                length_current = get_river_length(current, attr_dict, river_info)
                length_down = get_river_length(next_down, attr_dict, river_info)

                # 计算保留系数，并存储调试信息
                R_series = compute_retainment_factor(
                    v_f=v_f, 
                    Q_up=Q_current, 
                    Q_down=Q_down,
                    W_up=S_current,
                    W_down=S_down,
                    length_up=length_current,
                    length_down=length_down,
                    temperature=temperature,
                    N_concentration=N_concentration,
                    parameter=target_col,
                    debug_info=segment_debug
                )
                
                # 计算贡献并累加到下游负荷累加器
                contribution = y_n_current * R_series * Q_current
                
                # 检查贡献是否包含异常值
                contribution_nan = contribution.isna().sum()
                contribution_extreme = (~contribution.isna() & (contribution.abs() > 1e6)).sum()
                
                if contribution_nan > 0 or contribution_extreme > 0:
                    logging.warning(f"COMID {current} -> {next_down} 的贡献包含 {contribution_nan} 个NaN和 {contribution_extreme} 个异常值")
                    # 裁剪异常贡献值
                    if contribution_extreme > 0:
                        contribution = contribution.clip(-1e6, 1e6)
                        logging.info(f"已将贡献极端值裁剪到 ±1e6 范围内")
                    
                    # 记录问题河段
                    problem_segments.append((current, next_down, target_col))
                
                # 累加贡献
                load_acc[target_col][next_down] = load_acc[target_col][next_down].add(contribution, fill_value=0.0)
                
                # 存储额外的调试列
                if store_debug_info:
                    segment_debug[f'y_n_current_{target_col}'] = y_n_current
                    segment_debug[f'contribution_{target_col}'] = contribution
                    segment_debug[f'R_series_{target_col}'] = R_series
            except Exception as e:
                # 捕获计算过程中的异常
                logging.error(f"处理 COMID {current} -> {next_down} 的 {target_col} 时出错: {str(e)}")
                # 将此河段对添加到问题列表
                problem_segments.append((current, next_down, target_col))
                
                # 使用安全值作为默认贡献
                contribution = pd.Series(0.0, index=common_dates)
                load_acc[target_col][next_down] = load_acc[target_col][next_down].add(contribution, fill_value=0.0)
            
            # 存储调试信息
            if store_debug_info and (is_debug_segment or processed_count % debug_interval == 0):
                segment_debug['common_dates'] = common_dates
                segment_debug['Q_current'] = Q_current
                segment_debug['Q_down'] = Q_down
                segment_debug['S_current'] = S_current
                segment_debug['S_down'] = S_down
                if temperature is not None:
                    segment_debug['temperature'] = temperature
                    
                # 保存调试信息
                debug_key = f"{current}_{next_down}"
                debug_segments[debug_key] = segment_debug
        
        # 减少下游河段的入度
        indegree[next_down] -= 1
         
        # 如果下游河段所有上游都已处理完毕，计算其y_up和y_n并加入队列
        if indegree[next_down] == 0:
            down_data = comid_data[next_down]
            
            try:
                # 计算上游贡献浓度
                y_up_down = load_acc[target_col][next_down] / down_data['Qout'].replace(0, np.nan)
                
                # 检查y_up是否包含异常值
                y_up_nan = y_up_down.isna().sum()
                y_up_extreme = (~y_up_down.isna() & (y_up_down.abs() > 1e6)).sum()
                
                if y_up_nan > 0 or y_up_extreme > 0:
                    logging.warning(f"COMID {next_down} 的 y_up_{target_col} 包含 {y_up_nan} 个NaN和 {y_up_extreme} 个异常值")
                    # 处理异常值
                    if y_up_extreme > 0:
                        y_up_down = y_up_down.clip(-1e6, 1e6)
                        logging.info(f"已将 COMID {next_down} 的 y_up_{target_col} 极端值裁剪到 ±1e6 范围内")
                
                # 填充NaN值
                y_up_down = y_up_down.fillna(0.0)
                
                # 更新下游河段的y_up
                down_data[f'y_up_{target_col}'] = y_up_down
                
                # 计算y_n并检查异常值
                y_n_down = down_data[f'E_{target_col}'] + down_data[f'y_up_{target_col}']
                y_n_extreme = (y_n_down.abs() > 1e6).sum()
                
                if y_n_extreme > 0:
                    logging.warning(f"COMID {next_down} 的 y_n_{target_col} 包含 {y_n_extreme} 个异常值")
                    y_n_down = y_n_down.clip(-1e6, 1e6)
                    logging.info(f"已将 y_n_{target_col} 极端值裁剪到 ±1e6 范围内")
                
                down_data[f'y_n_{target_col}'] = y_n_down
                
                # 存储调试列
                if store_debug_info:
                    down_data[f'debug_load_acc_{target_col}'] = load_acc[target_col][next_down]
                    down_data[f'debug_E_{target_col}'] = down_data[f'E_{target_col}']
            except Exception as e:
                logging.error(f"计算 COMID {next_down} 的 y_up 和 y_n 时出错: {str(e)}")
                # 使用安全值
                down_data[f'y_up_{target_col}'] = 0.0
                down_data[f'y_n_{target_col}'] = down_data[f'E_{target_col}']
                problem_segments.append((next_down, 0, target_col))
            
            # 更新数据字典
            comid_data[next_down] = down_data
            
            # 将下游河段加入队列
            queue.append(next_down)
    
    # 报告问题河段的统计信息
    if problem_segments:
        logging.warning(f"汇流计算中有 {len(problem_segments)} 个问题河段对")
        
        # 统计每个参数的问题数量
        param_problems = {}
        for _, _, param in problem_segments:
            param_problems[param] = param_problems.get(param, 0) + 1
            
        for param, count in param_problems.items():
            logging.warning(f"参数 {param} 有 {count} 个问题河段")
    
    return comid_data, debug_segments

def format_results(comid_data, iteration, target_col):
    """
    将处理结果格式化为DataFrame
    
    参数:
        comid_data: 存储河段数据的字典
        iteration: 当前迭代次数
        target_col: 目标列
    
    返回:
        结果DataFrame
    """
    logging.info("合并结果...")
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
    rename_dict[f'E_{target_col}'] = f'E_{iteration}_{target_col}'
    rename_dict[f'y_up_{target_col}'] = f'y_up_{iteration}_{target_col}'
    rename_dict[f'y_n_{target_col}'] = f'y_n_{iteration}_{target_col}'

    # 确保列存在再重命名
    missing_cols = [col for col in rename_dict.keys() if col not in result_df.columns]
    if missing_cols:
        logging.warning(f"警告: 部分需要重命名的列不存在: {missing_cols}")
        # 只保留有效的列在重命名字典中
        rename_dict = {k: v for k, v in rename_dict.items() if k in result_df.columns}   

    result_df = result_df.rename(columns=rename_dict)

    # 添加调试日志，显示重命名后的列
    logging.info(f"重命名后的列: {result_df.columns.tolist()}")   
    
    return result_df

# ============================================================================
# 主函数: 汇流计算的入口点
# ============================================================================
def flow_routing_calculation(df: pd.DataFrame, 
                            iteration: int, 
                            model_func, 
                            river_info: pd.DataFrame, 
                            v_f_TN: float = 35.0,
                            v_f_TP: float = 44.5,
                            attr_dict: dict = None, 
                            model: CatchmentModel = None,
                            all_target_cols=["TN", "TP"],
                            target_col="TN",
                            attr_df: pd.DataFrame = None,
                            E_exist: int = 0,
                            E_exist_path: str = None,
                            E_save: int = 0,
                            E_save_path: str = None) -> pd.DataFrame:
    """
    汇流计算函数
    
    参数:
        df: 包含日尺度数据的DataFrame，每行记录一个COMID在某日期的数据，
           必须包含'COMID'、'date'、'Qout'、'temperature_2m_mean'(可选)等字段
        iteration: 当前迭代次数，用于命名新增加的列
        model_func: 用于预测局部贡献E的函数，输入为单个COMID的DataFrame，
                   输出为与日期对齐的Series
        river_info: 河段信息DataFrame，必须包含'COMID'和'NextDownID'
        v_f_TN: TN的基础吸收速率参数，默认为35.0 m/yr
        v_f_TP: TP的基础吸收速率参数，默认为44.5 m/yr
        attr_dict: 河段属性字典
        model: 预测模型
        all_target_cols: 所有可能的目标列列表，默认为["TN", "TP"]
        target_col: 主目标变量名称，如 "TN"
        attr_df: 河段属性DataFrame，用于识别标记为'ERA5_exist'=0的缺失数据河段
        E_exist: 是否从指定路径读取E值，0表示不读取，1表示读取
        E_exist_path: E值读取路径
        E_save: 是否保存计算得到的E值，0表示不保存，1表示保存
        E_save_path: E值保存路径
        
    返回:
        DataFrame，增加了新列:
            'E_{iteration}_{target_col}': 局部贡献（预测值）
            'y_up_{iteration}_{target_col}': 上游汇流贡献
            'y_n_{iteration}_{target_col}': 汇流总预测值 = E + y_up
    """
    # =========================================================================
    # 1. 初始化与数据准备
    # =========================================================================
    # 验证模型是否在正确的设备上
    if model and hasattr(model, 'base_model') and hasattr(model.base_model, 'parameters'):
        device = next(model.base_model.parameters()).device
        print(f"===== 模型设备检查 =====")
        print(f"模型在设备: {device}")
        print(f"模型类型: {type(model.base_model)}")
        print(f"======================")

    # 复制数据框以避免修改原始数据
    df = df.copy()
    logging.info(f"迭代 {iteration} 的汇流计算开始")
    
    # 标识缺失数据的河段
    missing_data_comids = set()
    if attr_df is not None and 'ERA5_exist' in attr_df.columns:
        # 获取ERA5_exist=0的河段ID
        missing_df = attr_df[attr_df['ERA5_exist'] == 0]
        missing_data_comids = set(str(comid) for comid in missing_df['COMID'])
        logging.info(f"标识出 {len(missing_data_comids)} 个缺失数据的河段")
    
    # 构建河网拓扑
    next_down_ids = build_river_network_topology(river_info, missing_data_comids)
    
    # 检查是否有温度数据可用
    has_temperature_data = 'temperature_2m_mean' in df.columns
    if has_temperature_data:
        logging.info("温度数据可用，将应用温度调整")
    else:
        logging.info("无温度数据，使用基础沉降速率")
    
    # 按河段ID分组并排序，为每个河段创建时间序列
    groups = {comid: group.sort_values("date").copy() for comid, group in df.groupby("COMID")}
    comid_data = {}
    
    # =========================================================================
    # 2. 处理E值（加载或计算）
    # =========================================================================
    if E_exist == 1 and E_exist_path:
        logging.info(f"从 {E_exist_path} 读取E值")
        print(f"从 {E_exist_path} 读取E值")
        
        # 加载E值
        e_df = load_e_values(E_exist_path, iteration, target_col)
        
        # 如果成功加载了E值，则为每个河段设置E值
        if e_df is not None:
            comid_data = apply_e_values(groups, comid_data, e_df, target_col, missing_data_comids)
        else:
            # 没有加载到E值，使用模型计算
            logging.info(f"未找到E值，使用模型计算河段E值 (迭代 {iteration})")
            comid_data = calculate_e_values(groups=groups, comid_data=comid_data, model_func=model_func, attr_dict=attr_dict, model=model, all_target_cols=all_target_cols,target_col = target_col, 
                                         missing_data_comids=missing_data_comids, iteration=iteration, e_save=E_save, e_save_path=E_save_path)
    else:
        # 没有加载E值，使用模型计算
        logging.info(f"使用模型计算河段E值 (迭代 {iteration})")
        comid_data = calculate_e_values(groups=groups, comid_data=comid_data, model_func=model_func, attr_dict=attr_dict, model=model, all_target_cols=all_target_cols,target_col = target_col, 
                                         missing_data_comids=missing_data_comids, iteration=iteration, e_save=E_save, e_save_path=E_save_path)
   
    
    # =========================================================================
    # 3. 构建河网拓扑结构
    # =========================================================================
    # 计算入度：若某个河段ID出现在其他河段的NextDownID中，则其入度增加
    indegree = calculate_indegrees(comid_data.keys(), next_down_ids)

    # 创建负荷累加器
    load_acc = {target_col: {comid: pd.Series(0.0, index=data.index) for comid, data in comid_data.items()}}

    # 找出所有头部河段（入度为0）并初始化其y_n值为局部贡献E
    queue = [comid for comid, deg in indegree.items() if deg == 0]
    comid_data = process_headwater_segments(comid_data, queue, target_col)
    
    # =========================================================================
    # 4. 执行汇流计算
    # =========================================================================
    comid_data, debug_segments = execute_flow_routing(comid_data, queue, indegree, next_down_ids, load_acc, 
                                     target_col, attr_dict, river_info, v_f_TN, v_f_TP)
    
    # =========================================================================
    # 5. 格式化结果
    # =========================================================================
    result_df = format_results(comid_data, iteration, target_col)
    
    # 只保留新增的列
    new_cols = ['COMID', 'date']
    new_cols.extend([f'E_{iteration}_{target_col}', f'y_up_{iteration}_{target_col}', f'y_n_{iteration}_{target_col}'])
    
    # 只保留存在的列
    existing_cols = [col for col in new_cols if col in result_df.columns]
    result_df = result_df[existing_cols]
    
    logging.info(f"迭代 {iteration} 的汇流计算完成")
    return result_df