"""
flow_routing.py - 河网汇流计算模块

本模块实现了基于物理约束的递归水质模型(PG-RWQ)的河网汇流计算功能。
汇流计算依据河网拓扑结构，考虑上游贡献和本地贡献，计算各河段的水质参数。

主要功能：
1. 读取或计算河段的局部贡献(E值)
2. 基于河网拓扑执行汇流计算
3. 考虑河道物理特性和环境因素
4. 输出水质预测结果

作者: PG-RWQ项目组
版本: 2.0
"""

import os
import pandas as pd
import numpy as np
import logging
import time
import torch
from tqdm import tqdm

# 导入自定义模块
from PGRWQI.flow_routing_modules.geometry import get_river_length, calculate_river_width
from PGRWQI.flow_routing_modules.environment_param import (
    compute_temperature_factor, 
    compute_nitrogen_concentration_factor,
    compute_retainment_factor
)
from PGRWQI.flow_routing_modules.topology import (
    build_river_network_topology,
    calculate_indegrees
)
from PGRWQI.flow_routing_modules.e_values import (
    load_e_values,
    apply_e_values,
    calculate_e_values
)
from PGRWQI.flow_routing_modules.debug_utils import DebugInfoCollector


def process_headwater_segments(comid_data, queue, target_col, debug_collector=None):
    """
    处理头部河段（入度为0的河段）
    
    头部河段是指没有上游河段流入的河段，其水质仅由本地贡献(E值)决定。
    
    参数:
        comid_data: 存储河段数据的字典
        queue: 待处理河段队列
        target_col: 目标列（水质参数，如'TN'或'TP'）
        debug_collector: 调试信息收集器(可选)
    
    返回:
        更新后的comid_data
    """
    logging.info(f"处理 {len(queue)} 个头部河段")
    
    for comid in queue:
        data = comid_data[comid]
        
        # 头部河段的水质预测值等于其局部贡献值
        data[f'y_n_{target_col}'] = data[f'E_{target_col}']
        
        # 保存调试信息
        if debug_collector is not None:
            debug_collector.add_segment_info(
                comid=comid,
                segment_type='headwater',
                data={
                    'dates': data.index.tolist(),
                    f'E_{target_col}': data[f'E_{target_col}'].tolist(),
                    f'y_n_{target_col}': data[f'y_n_{target_col}'].tolist()
                }
            )
        
        comid_data[comid] = data
    
    return comid_data


def execute_flow_routing(comid_data, queue, indegree, next_down_ids, load_acc, 
                        target_col, attr_dict, river_info, v_f_TN, v_f_TP, 
                        check_anomalies=False, debug_collector=None):
    """
    执行汇流计算的主要逻辑
    
    该函数实现了基于拓扑排序的河网汇流计算，从源头河段开始，
    沿着河网结构逐步计算下游河段的水质参数。
    
    参数:
        comid_data: 存储河段数据的字典
        queue: 待处理河段队列
        indegree: 每个河段的入度
        next_down_ids: 下游河段映射
        load_acc: 负荷累加器
        target_col: 目标列(水质参数)
        attr_dict: 河段属性字典
        river_info: 河网信息DataFrame
        v_f_TN: TN的吸收速率参数
        v_f_TP: TP的吸收速率参数
        check_anomalies: 是否检查和处理异常值
        debug_collector: 调试信息收集器(可选)
    
    返回:
        更新后的comid_data
    """
    logging.info("开始汇流计算...")
    processed_count = 0
    has_temperature_data = any('temperature_2m_mean' in data.columns for data in comid_data.values())
    
    while queue:
        # 从队列中取出下一个要处理的河段
        current = queue.pop(0)
        processed_count += 1
        
        # 周期性地记录处理进度
        if processed_count % 1000 == 0:
            logging.info(f"已处理 {processed_count} 个河段")
        
        # 获取当前河段数据和下游河段ID
        current_data = comid_data[current]
        next_down = next_down_ids.get(current, 0)
        
        # 如果没有下游或下游不在数据中，跳过处理
        if next_down == 0 or next_down not in comid_data:
            continue
            
        # 获取下游河段数据并找出共同日期
        down_data = comid_data[next_down]
        common_dates = current_data.index.intersection(down_data.index)
        
        # 检查日期对齐问题
        if len(common_dates) == 0:
            logging.warning(f"警告: 日期不对齐，COMID {current} 与 COMID {next_down}")
            
            # 保存调试信息
            if debug_collector is not None:
                debug_collector.add_warning(
                    'date_mismatch',
                    f"日期不对齐: 上游COMID {current} 与下游COMID {next_down}"
                )
            continue
        
        # 计算当前河段对下游的贡献
        if len(common_dates) > 0: 
            # 提取共同日期的数据
            Q_current = current_data['Qout'].reindex(common_dates)
            Q_down = down_data['Qout'].reindex(common_dates)
            W_current = current_data['width'].reindex(common_dates)
            W_down = down_data['width'].reindex(common_dates)
            
            # 获取温度数据（如果可用）
            temperature = None
            if has_temperature_data and 'temperature_2m_mean' in current_data.columns:
                temperature = current_data['temperature_2m_mean'].reindex(common_dates)
            
            try:
                # 选择合适的吸收速率参数
                if target_col == "TN":
                    v_f = v_f_TN
                    # 获取TN浓度数据（如果可用）
                    N_concentration = None
                    if f'y_n_{target_col}' in current_data.columns:
                        N_concentration = current_data[f'y_n_{target_col}'].reindex(common_dates)
                elif target_col == "TP":
                    v_f = v_f_TP
                    N_concentration = None
                else:
                    v_f = v_f_TN  # 默认使用TN的参数
                    N_concentration = None
                
                # 提取当前参数的y_n值
                y_n_current = current_data[f'y_n_{target_col}'].reindex(common_dates)
                
                # 检查异常值（如果需要）
                if check_anomalies:
                    nan_count = y_n_current.isna().sum()
                    extreme_count = (~y_n_current.isna() & (y_n_current.abs() > 1e6)).sum()
                    if nan_count > 0 or extreme_count > 0:
                        logging.warning(f"COMID {current} 的 y_n_{target_col} 包含 {nan_count} 个NaN和 {extreme_count} 个异常值")
                        if extreme_count > 0:
                            # 裁剪极端值
                            y_n_current = y_n_current.clip(-1e6, 1e6)

                # 获取河段长度
                length_current = get_river_length(current, attr_dict, river_info)
                length_down = get_river_length(next_down, attr_dict, river_info)
                
                # 计算保留系数
                R_series = compute_retainment_factor(
                    v_f=v_f, 
                    Q_up=Q_current, 
                    Q_down=Q_down,
                    W_up=W_current,
                    W_down=W_down,
                    length_up=length_current,
                    length_down=length_down,
                    temperature=temperature,
                    N_concentration=N_concentration,
                    parameter=target_col
                )
                
                # 计算贡献并累加到下游负荷累加器
                contribution = y_n_current * R_series * Q_current
                
                # 检查异常值（如果需要）
                if check_anomalies:
                    contribution_nan = contribution.isna().sum()
                    contribution_extreme = (~contribution.isna() & (contribution.abs() > 1e6)).sum()
                    
                    if contribution_nan > 0 or contribution_extreme > 0:
                        logging.warning(f"COMID {current} -> {next_down} 的贡献包含 {contribution_nan} 个NaN和 {contribution_extreme} 个异常值")
                        # 裁剪异常贡献值
                        if contribution_extreme > 0:
                            contribution = contribution.clip(-1e6, 1e6)
                
                # 保存调试信息
                if debug_collector is not None:
                    debug_dates = common_dates.tolist()  # 转换为Python列表以便序列化
                    debug_collector.add_routing_step(
                        upstream_comid=current,
                        downstream_comid=next_down,
                        step_data={
                            'dates': debug_dates,
                            'Q_upstream': Q_current.tolist(),
                            'Q_downstream': Q_down.tolist(),
                            'width_upstream': W_current.tolist(),
                            'width_downstream': W_down.tolist(),
                            'temperature': temperature.tolist() if temperature is not None else None,
                            'N_concentration': N_concentration.tolist() if N_concentration is not None else None,
                            'y_n_upstream': y_n_current.tolist(),
                            'retention_coefficient': R_series.tolist(),
                            'contribution': contribution.tolist(),
                            'length_upstream': length_current,
                            'length_downstream': length_down,
                            'v_f': v_f
                        }
                    )
                
                # 累加贡献
                load_acc[target_col][next_down] = load_acc[target_col][next_down].add(contribution, fill_value=0.0)
                
            except Exception as e:
                # 捕获计算过程中的异常
                error_msg = f"处理 COMID {current} -> {next_down} 的 {target_col} 时出错: {str(e)}"
                logging.error(error_msg)
                
                # 保存调试信息
                if debug_collector is not None:
                    debug_collector.add_error(
                        'calculation_error',
                        error_msg,
                        {
                            'upstream_comid': current,
                            'downstream_comid': next_down,
                            'target_col': target_col
                        }
                    )
                
                # 使用安全值作为默认贡献
                contribution = pd.Series(0.0, index=common_dates)
                load_acc[target_col][next_down] = load_acc[target_col][next_down].add(contribution, fill_value=0.0)
        
        # 减少下游河段的入度
        indegree[next_down] -= 1
         
        # 如果下游河段所有上游都已处理完毕，计算其y_up和y_n并加入队列
        if indegree[next_down] == 0:
            down_data = comid_data[next_down]
            
            try:
                # 计算上游贡献浓度
                y_up_down = load_acc[target_col][next_down] / down_data['Qout'].replace(0, np.nan)
                
                # 检查异常值（如果需要）
                if check_anomalies:
                    y_up_nan = y_up_down.isna().sum()
                    y_up_extreme = (~y_up_down.isna() & (y_up_down.abs() > 1e6)).sum()
                    
                    if y_up_nan > 0 or y_up_extreme > 0:
                        logging.warning(f"COMID {next_down} 的 y_up_{target_col} 包含 {y_up_nan} 个NaN和 {y_up_extreme} 个异常值")
                        # 处理异常值
                        if y_up_extreme > 0:
                            y_up_down = y_up_down.clip(-1e6, 1e6)
                
                # 填充NaN值
                y_up_down = y_up_down.fillna(0.0)
                
                # 更新下游河段的y_up
                down_data[f'y_up_{target_col}'] = y_up_down
                
                # 计算y_n并检查异常值
                y_n_down = down_data[f'E_{target_col}'] + down_data[f'y_up_{target_col}']
                
                if check_anomalies:
                    y_n_extreme = (y_n_down.abs() > 1e6).sum()
                    
                    if y_n_extreme > 0:
                        logging.warning(f"COMID {next_down} 的 y_n_{target_col} 包含 {y_n_extreme} 个异常值")
                        y_n_down = y_n_down.clip(-1e6, 1e6)
                
                down_data[f'y_n_{target_col}'] = y_n_down
                
                # 保存调试信息
                if debug_collector is not None:
                    debug_collector.add_node_calculation(
                        comid=next_down,
                        node_data={
                            'dates': down_data.index.tolist(),
                            f'E_{target_col}': down_data[f'E_{target_col}'].tolist(),
                            f'y_up_{target_col}': down_data[f'y_up_{target_col}'].tolist(),
                            f'y_n_{target_col}': down_data[f'y_n_{target_col}'].tolist(),
                            'Qout': down_data['Qout'].tolist()
                        }
                    )
                
            except Exception as e:
                error_msg = f"计算 COMID {next_down} 的 y_up 和 y_n 时出错: {str(e)}"
                logging.error(error_msg)
                
                # 保存调试信息
                if debug_collector is not None:
                    debug_collector.add_error(
                        'node_calculation_error',
                        error_msg,
                        {'comid': next_down}
                    )
                
                # 使用安全值
                down_data[f'y_up_{target_col}'] = 0.0
                down_data[f'y_n_{target_col}'] = down_data[f'E_{target_col}']
            
            # 更新数据字典
            comid_data[next_down] = down_data
            
            # 将下游河段加入队列
            queue.append(next_down)
    
    return comid_data


def format_results(comid_data, iteration, target_col, debug_collector=None):
    """
    将处理结果格式化为DataFrame
    
    参数:
        comid_data: 存储河段数据的字典
        iteration: 当前迭代次数
        target_col: 目标列
        debug_collector: 调试信息收集器(可选)
    
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
    
    # 记录调试信息
    if debug_collector is not None:
        debug_collector.add_summary({
            'total_segments': len(comid_data),
            'result_rows': len(result_df),
            'columns': result_df.columns.tolist()
        })
    
    return result_df


def flow_routing_calculation(df: pd.DataFrame, 
                            iteration: int, 
                            model_func, 
                            river_info: pd.DataFrame, 
                            v_f_TN: float = 35.0,
                            v_f_TP: float = 44.5,
                            attr_dict: dict = None, 
                            model = None,
                            all_target_cols=["TN", "TP"],
                            target_col="TN",
                            attr_df: pd.DataFrame = None,
                            E_exist: int = 0,
                            E_exist_path: str = None,
                            E_save: int = 0,
                            E_save_path: str = None,
                            check_anomalies: bool = False,
                            save_debugging_info: bool = False,
                            debug_output_dir: str = "debug_output") -> pd.DataFrame:
    """
    汇流计算主函数
    
    本函数是PG-RWQ流域水质模型中的核心汇流计算过程。
    从上游到下游递归计算河段的水质指标，综合考虑上游贡献和本地贡献。
    
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
        check_anomalies: 是否检查和处理异常值，默认为False
        save_debugging_info: 是否保存调试信息，默认为False
        debug_output_dir: 调试信息输出目录，默认为"debug_output"
        
    返回:
        DataFrame，增加了新列:
            'E_{iteration}_{target_col}': 局部贡献（预测值）
            'y_up_{iteration}_{target_col}': 上游汇流贡献
            'y_n_{iteration}_{target_col}': 汇流总预测值 = E + y_up
    """
    # =========================================================================
    # 1. 初始化与数据准备
    # =========================================================================
    # 复制数据框以避免修改原始数据
    df = df.copy()
    logging.info(f"迭代 {iteration} 的汇流计算开始")
    
    # 初始化调试信息收集器
    debug_collector = None
    if save_debugging_info:
        debug_collector = DebugInfoCollector(
            iteration=iteration,
            target_col=target_col,
            output_dir=debug_output_dir
        )
        debug_collector.add_parameters({
            'iteration': iteration,
            'target_col': target_col,
            'v_f_TN': v_f_TN,
            'v_f_TP': v_f_TP,
            'E_exist': E_exist,
            'E_save': E_save,
            'check_anomalies': check_anomalies
        })
    
    # 标识缺失数据的河段
    missing_data_comids = set()
    if attr_df is not None and 'ERA5_exist' in attr_df.columns:
        # 获取ERA5_exist=0的河段ID
        missing_df = attr_df[attr_df['ERA5_exist'] == 0]
        missing_data_comids = set(str(comid) for comid in missing_df['COMID'])
        logging.info(f"标识出 {len(missing_data_comids)} 个缺失数据的河段")
        
        if debug_collector is not None:
            debug_collector.add_missing_comids(missing_data_comids)
    
    # 构建河网拓扑
    next_down_ids = build_river_network_topology(river_info, missing_data_comids)
    
    # 检查是否有温度数据可用
    has_temperature_data = 'temperature_2m_mean' in df.columns
    if has_temperature_data:
        logging.info("温度数据可用，将应用温度调整")
    else:
        logging.info("无温度数据，使用基础沉降速率")
    
    # 注意：这里不再直接分组，而是从model_func的data_handler获取预分组数据
    # 临时获取分组数据以了解可用的COMID，这个仅仅是为了获取COMID列表
    temp_groups = {comid: group.sort_values("date").copy() for comid, group in df.groupby("COMID")}
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
            comid_data = apply_e_values(
                temp_groups, comid_data, e_df, target_col, missing_data_comids,
                debug_collector
            )
        else:
            # 没有加载到E值，使用模型计算
            logging.info(f"未找到E值，使用模型计算河段E值 (迭代 {iteration})")
            comid_data = calculate_e_values(
                groups_dict=temp_groups, comid_data=comid_data, model_func=model_func,
                target_col=target_col, missing_data_comids=missing_data_comids, iteration=iteration, 
                e_save=E_save, e_save_path=E_save_path, debug_collector=debug_collector
            )
    else:
        # 没有加载E值，使用模型计算
        logging.info(f"使用模型计算河段E值 (迭代 {iteration})")
        comid_data = calculate_e_values(
            groups_dict=temp_groups, comid_data=comid_data, model_func=model_func,
            target_col=target_col, missing_data_comids=missing_data_comids, iteration=iteration, 
            e_save=E_save, e_save_path=E_save_path, debug_collector=debug_collector
        )
    
    # =========================================================================
    # 3. 构建河网拓扑结构
    # =========================================================================
    # 计算入度：若某个河段ID出现在其他河段的NextDownID中，则其入度增加
    indegree = calculate_indegrees(comid_data.keys(), next_down_ids)

    # 创建负荷累加器
    load_acc = {target_col: {comid: pd.Series(0.0, index=data.index) for comid, data in comid_data.items()}}

    # 找出所有头部河段（入度为0）并初始化其y_n值为局部贡献E
    queue = [comid for comid, deg in indegree.items() if deg == 0]
    comid_data = process_headwater_segments(comid_data, queue, target_col, debug_collector)
    
    # =========================================================================
    # 4. 执行汇流计算
    # =========================================================================
    comid_data = execute_flow_routing(
        comid_data, queue, indegree, next_down_ids, load_acc, 
        target_col, attr_dict, river_info, v_f_TN, v_f_TP, 
        check_anomalies=check_anomalies, debug_collector=debug_collector
    )
    
    # =========================================================================
    # 5. 格式化结果
    # =========================================================================
    result_df = format_results(comid_data, iteration, target_col, debug_collector)
    
    # 只保留新增的列
    new_cols = ['COMID', 'date']
    new_cols.extend([f'E_{iteration}_{target_col}', f'y_up_{iteration}_{target_col}', f'y_n_{iteration}_{target_col}'])
    
    # 只保留存在的列
    existing_cols = [col for col in new_cols if col in result_df.columns]
    result_df = result_df[existing_cols]
    
    # 保存调试信息
    if debug_collector is not None:
        debug_collector.save()
        logging.info(f"调试信息已保存到 {debug_output_dir}")
    
    logging.info(f"迭代 {iteration} 的汇流计算完成")
    return result_df