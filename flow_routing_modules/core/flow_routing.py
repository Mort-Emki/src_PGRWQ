import os
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any

# 导入同级模块
from .geometry import get_river_length, calculate_river_width
from .topology import build_river_network_topology, calculate_indegrees
from ..physics.environment_param import compute_retainment_factor
from ..physics.e_values import load_e_values, apply_e_values, calculate_e_values


def execute_flow_routing(comid_data, queue, indegree, next_down_ids, load_acc, 
                        target_col, attr_dict, river_info, v_f_TN, v_f_TP, 
                        check_anomalies=False, enable_debug=True):
    """
    执行汇流计算的主要逻辑 (重构版)
    
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
        enable_debug: 是否启用调试信息记录
    
    返回:
        tuple: (更新后的comid_data, debug_info)
    """
    logging.info("开始汇流计算...")
    processed_count = 0
    has_temperature_data = any('temperature_2m_mean' in data.columns for data in comid_data.values())
    
    # 调试信息存储
    debug_info = {
        'node_calculations': {},  # 键: (comid, date_str)
        'flow_calculations': {},  # 键: (upstream_comid, downstream_comid, date_str)
        'warnings': [],
        'errors': []
    } if enable_debug else None
    
    def add_warning(msg, **details):
        if enable_debug:
            debug_info['warnings'].append({'message': msg, 'details': details})
        logging.warning(msg)
    
    def add_error(msg, **details):
        if enable_debug:
            debug_info['errors'].append({'message': msg, 'details': details})
        logging.error(msg)
    
    while queue:
        current = queue.pop(0)
        processed_count += 1
        
        if processed_count % 1000 == 0:
            logging.info(f"已处理 {processed_count} 个河段")
        
        current_data = comid_data[current]
        next_down = next_down_ids.get(current, 0)
        
        if next_down == 0 or next_down not in comid_data:
            continue
            
        down_data = comid_data[next_down]
        common_dates = current_data.index.intersection(down_data.index)
        
        if len(common_dates) == 0:
            add_warning(f"日期不对齐", upstream=current, downstream=next_down)
            continue
        
        if len(common_dates) > 0: 
            # 提取共同日期的数据
            Q_current = current_data['Qout'].reindex(common_dates)
            Q_down = down_data['Qout'].reindex(common_dates)
            W_current = current_data['width'].reindex(common_dates)
            W_down = down_data['width'].reindex(common_dates)
            
            # 获取温度数据
            temperature = None
            if has_temperature_data and 'temperature_2m_mean' in current_data.columns:
                temperature = current_data['temperature_2m_mean'].reindex(common_dates)
            
            try:
                # 选择吸收速率参数
                if target_col == "TN":
                    v_f = v_f_TN
                    N_concentration = None
                    if f'y_n_{target_col}' in current_data.columns:
                        N_concentration = current_data[f'y_n_{target_col}'].reindex(common_dates)
                elif target_col == "TP":
                    v_f = v_f_TP
                    N_concentration = None
                else:
                    v_f = v_f_TN
                    N_concentration = None
                
                y_n_current = current_data[f'y_n_{target_col}'].reindex(common_dates)
                
                # 异常值检查
                if check_anomalies:
                    nan_count = y_n_current.isna().sum()
                    extreme_count = (~y_n_current.isna() & (y_n_current.abs() > 1e6)).sum()
                    if nan_count > 0 or extreme_count > 0:
                        add_warning(f"COMID {current} 的 y_n_{target_col} 包含异常值", 
                                   nan_count=nan_count, extreme_count=extreme_count)
                        if extreme_count > 0:
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
                
                # 计算贡献
                contribution = y_n_current * R_series * Q_current
                
                # 异常值检查
                if check_anomalies:
                    contribution_nan = contribution.isna().sum()
                    contribution_extreme = (~contribution.isna() & (contribution.abs() > 1e6)).sum()
                    
                    if contribution_nan > 0 or contribution_extreme > 0:
                        add_warning(f"COMID {current} -> {next_down} 的贡献包含异常值",
                                   contribution_nan=contribution_nan, 
                                   contribution_extreme=contribution_extreme)
                        if contribution_extreme > 0:
                            contribution = contribution.clip(-1e6, 1e6)
                
                # 记录调试信息 - 按日期记录流向计算
                if enable_debug:
                    for date in common_dates:
                        date_str = str(date.date()) if hasattr(date, 'date') else str(date)
                        key = (current, next_down, date_str)
                        
                        debug_info['flow_calculations'][key] = {
                            'date': date_str,
                            'upstream_comid': current,
                            'downstream_comid': next_down,
                            'Q_upstream': float(Q_current.loc[date]),
                            'Q_downstream': float(Q_down.loc[date]),
                            'width_upstream': float(W_current.loc[date]),
                            'width_downstream': float(W_down.loc[date]),
                            'temperature': float(temperature.loc[date]) if temperature is not None else None,
                            'N_concentration': float(N_concentration.loc[date]) if N_concentration is not None else None,
                            'y_n_upstream': float(y_n_current.loc[date]),
                            'retention_coefficient': float(R_series.loc[date]),
                            'contribution': float(contribution.loc[date]),
                            'length_upstream': length_current,
                            'length_downstream': length_down,
                            'v_f': v_f,
                            'target_col': target_col
                        }
                
                # 累加贡献
                load_acc[target_col][next_down] = load_acc[target_col][next_down].add(contribution, fill_value=0.0)
                
            except Exception as e:
                error_msg = f"处理 COMID {current} -> {next_down} 的 {target_col} 时出错: {str(e)}"
                add_error(error_msg, upstream=current, downstream=next_down, target_col=target_col)
                
                # 使用安全值
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
                
                # 异常值检查
                if check_anomalies:
                    y_up_nan = y_up_down.isna().sum()
                    y_up_extreme = (~y_up_down.isna() & (y_up_down.abs() > 1e6)).sum()
                    
                    if y_up_nan > 0 or y_up_extreme > 0:
                        add_warning(f"COMID {next_down} 的 y_up_{target_col} 包含异常值",
                                   y_up_nan=y_up_nan, y_up_extreme=y_up_extreme)
                        if y_up_extreme > 0:
                            y_up_down = y_up_down.clip(-1e6, 1e6)
                
                # 填充NaN值
                y_up_down = y_up_down.fillna(0.0)
                
                # 更新下游河段的y_up
                down_data[f'y_up_{target_col}'] = y_up_down
                
                # 计算y_n
                y_n_down = down_data[f'E_{target_col}'] + down_data[f'y_up_{target_col}']
                
                if check_anomalies:
                    y_n_extreme = (y_n_down.abs() > 1e6).sum()
                    
                    if y_n_extreme > 0:
                        add_warning(f"COMID {next_down} 的 y_n_{target_col} 包含异常值",
                                   y_n_extreme=y_n_extreme)
                        y_n_down = y_n_down.clip(-1e6, 1e6)
                
                down_data[f'y_n_{target_col}'] = y_n_down
                
                # 记录调试信息 - 按日期记录节点计算
                if enable_debug:
                    for date in down_data.index:
                        date_str = str(date.date()) if hasattr(date, 'date') else str(date)
                        key = (next_down, date_str)
                        
                        debug_info['node_calculations'][key] = {
                            'date': date_str,
                            'comid': next_down,
                            'E_value': float(down_data.loc[date, f'E_{target_col}']),
                            'y_up_value': float(down_data.loc[date, f'y_up_{target_col}']),
                            'y_n_value': float(down_data.loc[date, f'y_n_{target_col}']),
                            'Qout': float(down_data.loc[date, 'Qout']),
                            'target_col': target_col
                        }
                
            except Exception as e:
                error_msg = f"计算 COMID {next_down} 的 y_up 和 y_n 时出错: {str(e)}"
                add_error(error_msg, comid=next_down, target_col=target_col)
                
                # 使用安全值
                down_data[f'y_up_{target_col}'] = 0.0
                down_data[f'y_n_{target_col}'] = down_data[f'E_{target_col}']
            
            # 更新数据字典
            comid_data[next_down] = down_data
            
            # 将下游河段加入队列
            queue.append(next_down)
    
    # 保存调试信息
    if enable_debug and debug_info is not None:
        save_debug_info(debug_info, target_col)
    
    return comid_data, debug_info


def save_debug_info(debug_info, target_col, output_dir="debug_output"):
    """
    保存调试信息到文件
    
    参数:
        debug_info: 调试信息字典
        target_col: 目标列名
        output_dir: 输出目录
    """
    import json
    import os
    from datetime import datetime
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 转换调试信息为可JSON序列化的格式
    serializable_debug = {
        'metadata': {
            'target_col': target_col,
            'timestamp': timestamp,
            'total_node_calculations': len(debug_info['node_calculations']),
            'total_flow_calculations': len(debug_info['flow_calculations']),
            'total_warnings': len(debug_info['warnings']),
            'total_errors': len(debug_info['errors'])
        },
        'node_calculations': {},
        'flow_calculations': {},
        'warnings': debug_info['warnings'],
        'errors': debug_info['errors']
    }
    
    # 转换节点计算数据
    for (comid, date), data in debug_info['node_calculations'].items():
        key = f"{comid}_{date}"
        serializable_debug['node_calculations'][key] = data
    
    # 转换流向计算数据
    for (uc, dc, date), data in debug_info['flow_calculations'].items():
        key = f"{uc}_{dc}_{date}"
        serializable_debug['flow_calculations'][key] = data
    
    # 保存为JSON文件
    json_file = os.path.join(output_dir, f"debug_info_{target_col}_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_debug, f, indent=2, ensure_ascii=False)
    
    logging.info(f"调试信息已保存到: {json_file}")
    
    # 同时保存一个简化的CSV文件用于快速查看
    save_debug_summary_csv(debug_info, target_col, output_dir, timestamp)
    
    return json_file


def save_debug_summary_csv(debug_info, target_col, output_dir, timestamp):
    """保存调试信息的CSV摘要"""
    import pandas as pd
    import os
    
    # 创建节点计算摘要
    node_data = []
    for (comid, date), data in debug_info['node_calculations'].items():
        node_data.append({
            'comid': comid,
            'date': date,
            'E_value': data['E_value'],
            'y_up_value': data['y_up_value'],
            'y_n_value': data['y_n_value'],
            'Qout': data['Qout']
        })
    
    if node_data:
        node_df = pd.DataFrame(node_data)
        node_csv = os.path.join(output_dir, f"node_calculations_{target_col}_{timestamp}.csv")
        node_df.to_csv(node_csv, index=False)
    
    # 创建流向计算摘要
    flow_data = []
    for (uc, dc, date), data in debug_info['flow_calculations'].items():
        flow_data.append({
            'upstream_comid': uc,
            'downstream_comid': dc,
            'date': date,
            'y_n_upstream': data['y_n_upstream'],
            'retention_coefficient': data['retention_coefficient'],
            'contribution': data['contribution'],
            'Q_upstream': data['Q_upstream']
        })
    
    if flow_data:
        flow_df = pd.DataFrame(flow_data)
        flow_csv = os.path.join(output_dir, f"flow_calculations_{target_col}_{timestamp}.csv")
        flow_df.to_csv(flow_csv, index=False)


def process_headwater_segments(comid_data, queue, target_col):
    """
    处理头部河段（入度为0的河段）
    
    头部河段是指没有上游河段流入的河段，其水质仅由本地贡献(E值)决定。
    """
    logging.info(f"处理 {len(queue)} 个头部河段")
    
    for comid in queue:
        data = comid_data[comid]
        # 头部河段的水质预测值等于其局部贡献值
        data[f'y_n_{target_col}'] = data[f'E_{target_col}']
        comid_data[comid] = data
    
    return comid_data


def format_results(comid_data, iteration, target_col):
    """
    将处理结果格式化为DataFrame
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
                            enable_debug: bool = True) -> pd.DataFrame:
    """
    汇流计算主函数 (重构版)
    """
    # 复制数据框以避免修改原始数据
    df = df.copy()
    logging.info(f"迭代 {iteration} 的汇流计算开始")
    
    # 标识缺失数据的河段
    missing_data_comids = set()
    if attr_df is not None and 'ERA5_exist' in attr_df.columns:
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
    
    # 获取分组数据
    temp_groups = {comid: group.sort_values("date").copy() for comid, group in df.groupby("COMID")}
    comid_data = {}
    
    # 处理E值（加载或计算）
    if E_exist == 1 and E_exist_path:
        logging.info(f"从 {E_exist_path} 读取E值")
        e_df = load_e_values(E_exist_path, iteration, target_col)
        
        if e_df is not None:
            comid_data = apply_e_values(
                temp_groups, comid_data, e_df, target_col, missing_data_comids
            )
        else:
            logging.info(f"未找到E值，使用模型计算河段E值 (迭代 {iteration})")
            comid_data = calculate_e_values(
                groups_dict=temp_groups, comid_data=comid_data, model_func=model_func,
                target_col=target_col, missing_data_comids=missing_data_comids, iteration=iteration, 
                e_save=E_save, e_save_path=E_save_path
            )
    else:
        logging.info(f"使用模型计算河段E值 (迭代 {iteration})")
        comid_data = calculate_e_values(
            groups_dict=temp_groups, comid_data=comid_data, model_func=model_func,
            target_col=target_col, missing_data_comids=missing_data_comids, iteration=iteration, 
            e_save=E_save, e_save_path=E_save_path
        )
    
    # 构建河网拓扑结构
    indegree = calculate_indegrees(comid_data.keys(), next_down_ids)

    # 创建负荷累加器
    load_acc = {target_col: {comid: pd.Series(0.0, index=data.index) for comid, data in comid_data.items()}}

    # 找出所有头部河段（入度为0）并初始化其y_n值为局部贡献E
    queue = [comid for comid, deg in indegree.items() if deg == 0]
    comid_data = process_headwater_segments(comid_data, queue, target_col)
    
    # 执行汇流计算
    comid_data, debug_info = execute_flow_routing(
        comid_data, queue, indegree, next_down_ids, load_acc, 
        target_col, attr_dict, river_info, v_f_TN, v_f_TP, 
        check_anomalies=check_anomalies, enable_debug=enable_debug
    )
    
    # 格式化结果
    result_df = format_results(comid_data, iteration, target_col)
    
    # 只保留新增的列
    new_cols = ['COMID', 'date']
    new_cols.extend([f'E_{iteration}_{target_col}', f'y_up_{iteration}_{target_col}', f'y_n_{iteration}_{target_col}'])
    
    # 只保留存在的列
    existing_cols = [col for col in new_cols if col in result_df.columns]
    result_df = result_df[existing_cols]
    
    logging.info(f"迭代 {iteration} 的汇流计算完成")
    return result_df
