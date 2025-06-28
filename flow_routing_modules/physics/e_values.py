"""
e_values.py - E值(局部贡献)处理模块

本模块提供了处理E值(局部贡献)的功能，包括从文件加载E值、
应用E值到河段和使用模型计算E值。E值代表河段本地汇水区的
水质贡献，是汇流计算的关键输入。

主要功能:
1. 从文件加载E值
2. 应用E值到河段
3. 使用模型计算E值
"""

import os
import pandas as pd
import numpy as np
import logging
import time
import torch
from tqdm import tqdm

from PGRWQI.flow_routing_modules.core.geometry import calculate_river_width


def load_e_values(e_exist_path, iteration, target_col):
    """
    从文件加载E值
    
    参数:
        e_exist_path: E值读取路径
        iteration: 当前迭代次数
        target_col: 目标列(水质参数)
    
    返回:
        pd.DataFrame: 加载的E值DataFrame，如果未找到则返回None
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
    
# def apply_e_values(groups_dict, comid_data, e_df, target_col, missing_data_comids, debug_collector=None):
#     """
#     为河段应用已加载的E值
    
#     参数:
#         groups_dict: 预分组的数据字典（从DataHandler获取）
#         comid_data: 存储处理结果的字典
#         e_df: 加载的E值DataFrame
#         target_col: 目标列(水质参数)
#         missing_data_comids: 缺失数据的河段ID集合
#         debug_collector: 调试信息收集器(可选)
    
#     返回:
#         dict: 更新后的comid_data字典
#     """
#     for comid in tqdm(groups_dict.keys(), desc=f"设置E值"):
#         # 跳过缺失数据的河段
#         if str(comid) in missing_data_comids:
#             continue
        
#         group = groups_dict[comid].copy()
        
#         # 计算河道宽度
#         group['width'] = calculate_river_width(group['Qout'])
        
#         # 设置E值
#         if e_df is not None:
#             # 从加载的E值中查找对应的记录
#             comid_e_df = e_df[e_df['COMID'] == comid]
            
#             if not comid_e_df.empty:
#                 # 将E值设置到对应的日期
#                 e_series = pd.Series(index=group['date'])
#                 for _, row in comid_e_df.iterrows():
#                     date_val = row['Date']
#                     if date_val in e_series.index:
#                         e_series[date_val] = row['E_value']
                
#                 # 填充缺失的E值为0
#                 e_series = e_series.fillna(0.0)
                
#                 # 设置E值
#                 group[f'E_{target_col}'] = e_series.values
#             else:
#                 # 如果没有找到对应的E值记录，设置为0
#                 group[f'E_{target_col}'] = 0.0
#         else:
#             # 如果没有加载E值，设置为0
#             group[f'E_{target_col}'] = 0.0
        
#         # 初始化y_up和y_n
#         group[f'y_up_{target_col}'] = 0.0
#         group[f'y_n_{target_col}'] = 0.0
        
#         # 设置索引并保存到数据字典
#         group = group.set_index("date")
#         comid_data[comid] = group
        
#         # 保存调试信息
#         if debug_collector is not None:
#             debug_collector.add_segment_info(
#                 comid=comid,
#                 segment_type='applied_e',
#                 data={
#                     'dates': group.index.tolist(),
#                     f'E_{target_col}': group[f'E_{target_col}'].tolist(),
#                     'Qout': group['Qout'].tolist(),
#                     'width': group['width'].tolist()
#                 }
#             )
    
#     return comid_data

def apply_e_values(groups_dict, comid_data, e_df, target_col, missing_data_comids, debug_collector=None):
    """
    预分配内存+批量操作
    
    思路：预先分配所有内存，使用NumPy的高级索引和广播
    """
    # print("终极优化E值应用...")
    
    # 如果没有E值数据，使用最快的零初始化
    if e_df is None:
        for comid in groups_dict.keys():
            if str(comid) in missing_data_comids:
                continue
            group = groups_dict[comid].copy()
            # 使用NumPy向量化操作
            qout_values = group['Qout'].values
            group['width'] = calculate_river_width(pd.Series(qout_values))
            
            # 批量设置多个列
            num_rows = len(group)
            group[f'E_{target_col}'] = np.zeros(num_rows)
            group[f'y_up_{target_col}'] = np.zeros(num_rows)
            group[f'y_n_{target_col}'] = np.zeros(num_rows)
            
            group = group.set_index("date")
            comid_data[comid] = group
        return comid_data
    
    # 构建全局索引和预分配内存
    print("构建全局索引...")
    
    # 收集所有unique的(comid, date)对
    all_keys = []
    key_to_group_info = {}
    
    for comid, group in groups_dict.items():
        if str(comid) in missing_data_comids:
            continue
            
        dates = pd.to_datetime(group['date'])
        for i, date in enumerate(dates):
            key = (comid, date)
            all_keys.append(key)
            key_to_group_info[key] = (comid, i)
    
    # 转换为NumPy数组以支持向量化操作
    e_df_copy = e_df.copy()
    e_df_copy['Date'] = pd.to_datetime(e_df_copy['Date'])
    e_df_copy['key'] = list(zip(e_df_copy['COMID'], e_df_copy['Date']))
    
    # 使用pandas的merge进行批量查找（内部优化）
    keys_df = pd.DataFrame({'key': all_keys})
    merged = pd.merge(keys_df, e_df_copy[['key', 'E_value']], on='key', how='left')
    merged['E_value'] = merged['E_value'].fillna(0.0)
    
    # 将结果分配回各个group
    e_results = dict(zip(all_keys, merged['E_value']))
    
    print("批量分配结果...")
    
    # 批量处理所有groups
    for comid in groups_dict.keys():
        if str(comid) in missing_data_comids:
            continue
            
        group = groups_dict[comid].copy()
        
        # 向量化计算
        group['width'] = calculate_river_width(group['Qout'])
        
        # 批量获取E值
        dates = pd.to_datetime(group['date'])
        e_values = np.array([e_results.get((comid, date), 0.0) for date in dates])
        
        # 批量设置所有列
        group[f'E_{target_col}'] = e_values
        group[f'y_up_{target_col}'] = np.zeros(len(group))
        group[f'y_n_{target_col}'] = np.zeros(len(group))
        
        group = group.set_index("date")
        comid_data[comid] = group
    
    return comid_data


def calculate_e_values(groups_dict, comid_data, model_func,target_col, 
                     missing_data_comids, iteration, e_save, e_save_path, debug_collector=None):
    """
    使用模型计算E值
    
    参数:
        groups_dict: 预分组的数据字典（从DataHandler获取）
        comid_data: 存储处理结果的字典
        model_func: 预测模型函数
        attr_dict: 河段属性字典
        model: 预测模型
        all_target_cols: 所有可能的目标列列表
        target_col: 目标列(水质参数)
        missing_data_comids: 缺失数据的河段ID集合
        iteration: 当前迭代次数
        e_save: 是否保存E值(0=不保存，1=保存)
        e_save_path: E值保存路径
        debug_collector: 调试信息收集器(可选)
    
    返回:
        dict: 更新后的comid_data字典
    """
    # 需要保存的E值
    e_values_to_save = [] if e_save == 1 else None
    
    comid_list = list(groups_dict.keys())
    logging.info(f"处理 {len(comid_list)} 个河段，批量计算...")
    
    # 设定批次大小并计算批次数
    batch_size = 1000  # 每批处理1000个COMID
    num_batches = (len(comid_list) + batch_size - 1) // batch_size
    
    # 使用进度条处理所有河段
    with tqdm(total=len(comid_list), desc=f"处理河段，迭代 {iteration}") as pbar:
        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            
            # 获取当前批次的河段ID
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(comid_list))
            batch_comids = comid_list[start_idx:end_idx]
            
            # 过滤掉缺失数据的河段
            valid_batch_comids = [comid for comid in batch_comids if str(comid) not in missing_data_comids]
            
            # 批量计算当前批次所有有效河段的E值
            batch_results = model_func(valid_batch_comids)

            # 处理结果并存入数据字典
            for comid in batch_comids:
                # 跳过缺失数据的河段
                if str(comid) in missing_data_comids:
                    continue
                    
                group = groups_dict[comid].copy()
                E_series = batch_results.get(comid)
                
                if E_series is None:
                    logging.warning(f"河段 {comid} 的模型结果为 None，设置为 0")
                    # 为此河段的所有时间设置 E 为 0
                    group[f'E_{target_col}'] = 0.0
                    group[f'y_up_{target_col}'] = 0.0
                    group[f'y_n_{target_col}'] = 0.0
                else:
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
                
                # 保存调试信息
                if debug_collector is not None:
                    debug_collector.add_segment_info(
                        comid=comid,
                        segment_type='calculated_e',
                        data={
                            'dates': group.index.tolist(),
                            f'E_{target_col}': group[f'E_{target_col}'].tolist(),
                            'Qout': group['Qout'].tolist(),
                            'width': group['width'].tolist()
                        }
                    )
            
            # 更新进度条
            pbar.update(len(batch_comids))
            
            # 定期检查内存使用（每5批次一次）
            if batch_idx % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
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
            
            # 记录调试信息
            if debug_collector is not None:
                debug_collector.add_summary({
                    'e_values_saved': len(e_df),
                    'e_file_path': e_file_path
                })
    
    return comid_data


def visualize_e_values(e_values, iteration, target_col, output_dir="e_visualizations"):
    """
    可视化E值分布
    
    参数:
        e_values: E值DataFrame
        iteration: 迭代次数
        target_col: 目标列(水质参数)
        output_dir: 输出目录
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保E值为数值型
    e_values = pd.to_numeric(e_values['E_value'], errors='coerce')
    
    # 创建直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(e_values, kde=True)
    plt.title(f'E值分布 - 迭代{iteration} {target_col}')
    plt.xlabel('E值')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    output_path = os.path.join(output_dir, f'e_distribution_iteration_{iteration}_{target_col}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建箱线图
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=e_values)
    plt.title(f'E值箱线图 - 迭代{iteration} {target_col}')
    plt.ylabel('E值')
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    output_path = os.path.join(output_dir, f'e_boxplot_iteration_{iteration}_{target_col}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"E值可视化已保存到 {output_dir}")
    
    # 计算并返回统计信息
    stats = {
        'count': len(e_values),
        'mean': e_values.mean(),
        'median': e_values.median(),
        'min': e_values.min(),
        'max': e_values.max(),
        'std': e_values.std(),
        'q25': e_values.quantile(0.25),
        'q75': e_values.quantile(0.75)
    }
    
    return stats