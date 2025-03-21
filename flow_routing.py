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

def compute_retainment_factor(v_f: float, Q_up: pd.Series, Q_down: pd.Series) -> pd.Series:
    """
    计算保留系数
    输入：
        v_f: 吸收速率参数
        Q_up: 上游流量序列（pd.Series）
        Q_down: 下游流量序列（pd.Series）
    输出：
        返回一个 Series，每个元素为对应日期的保留系数
    """
    Q_up_adj = Q_up.replace(0, np.nan)
    Q_down_adj = Q_down.replace(0, np.nan)
    R = (1 - np.exp(-v_f / (2 * Q_up_adj))) * (1 - np.exp(-v_f / (2 * Q_down_adj)))
    return R.fillna(0.0)


def flow_routing_calculation(df: pd.DataFrame, 
                             iteration: int, 
                             model_func, 
                             river_info: pd.DataFrame, 
                             v_f: float = 35.0,
                             attr_dict: dict = None, 
                             model: CatchmentModel = None,
                             target_cols=["TN", "TP"]) -> pd.DataFrame:
    """
    汇流计算函数
    输入：
        df: 包含日尺度数据的 DataFrame，每行记录一个 COMID 在某日期的数据，
            必须包含 'COMID'、'Date'、'Qout' 等字段
        iteration: 当前迭代次数，用于命名新增加的列
        model_func: 用于预测局部贡献 E 的函数，输入为单个 COMID 的 DataFrame，
                    输出为与日期对齐的 Series
        river_info: 河段信息 DataFrame，必须包含 'COMID' 和 'NextDownID'
        v_f: 吸收速率参数
        target_cols: 目标列列表，默认为 ["TN", "TP"]
    输出：
        返回 DataFrame，增加了新列：
            'E_{iteration}'：局部贡献（预测值）
            'y_up_{iteration}'：上游汇流贡献
            'y_n_{iteration}'：汇流总预测值 = E + y_up
    """
        # Debug model device
    if model and hasattr(model, 'base_model') and hasattr(model.base_model, 'parameters'):
        device = next(model.base_model.parameters()).device
        print(f"===== MODEL DEVICE CHECK =====")
        print(f"Model is on device: {device}")
        print(f"Model type: {type(model.base_model)}")
        print(f"===============================")

    df = df.copy()
    logging.info(f"Flow routing calculation for iteration {iteration} started")
    logging.debug(f"DataFrame head:\n{df.head()}")
    
    df['date'] = pd.to_datetime(df['date'])
    
    # 从 river_info 中构造 NextDownID 字典
    next_down_ids = river_info.set_index('COMID')['NextDownID'].to_dict()
    
    # 按 COMID 分组并排序，构造每个河段的时间序列
    groups = {comid: group.sort_values("date").copy() for comid, group in df.groupby("COMID")}
    comid_data = {}

    # 使用批处理方式处理多个COMID
    logging.info(f"Processing {len(groups)} river segments in batches...")
    
    # 按批次处理COMID
    batch_size = 1000  # 每批处理1000个COMID
    comid_list = list(groups.keys())
    num_batches = (len(comid_list) + batch_size - 1) // batch_size
    
    with tqdm(total=len(groups), desc=f"Processing river segments for iteration {iteration}") as pbar:
        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            
            # 获取当前批次的COMID
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(comid_list))
            batch_comids = comid_list[start_idx:end_idx]
            
            # 批量处理当前批次的所有COMID
            batch_results = model_func(batch_comids, groups, attr_dict, model, target_cols)
            
            # 将结果存入comid_data
            for comid in batch_comids:
                group = groups[comid]
                E_series = batch_results[comid]
                
                group['E'] = E_series.values
                group['y_up'] = 0.0
                group['y_n'] = 0.0
                group = group.set_index("date")
                comid_data[comid] = group
            
            # 记录批次处理时间
            batch_time = time.time() - batch_start_time
            
            # 定期记录批次信息
            if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                logging.info(f"Batch {batch_idx+1}/{num_batches}: Processed {len(batch_comids)} COMIDs in {batch_time:.2f}s")
                
                # 定期记录内存使用情况
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                    logging.info(f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
                    
                    # 如果内存占用过高，释放缓存
                    if allocated > 40000:  # 40GB阈值
                        torch.cuda.empty_cache()
                        logging.info("Cleared GPU cache")
            
            # 更新进度条
            pbar.update(len(batch_comids))

    # 计算入度：若某个 COMID 出现在其他河段的 NextDownID 中，则其入度增加
    logging.info("Calculating node indegrees...")
    indegree = {comid: 0 for comid in comid_data.keys()}
    for comid in comid_data.keys():
        next_down = next_down_ids.get(comid, 0)
        if next_down != 0:
            indegree[next_down] = indegree.get(next_down, 0) + 1

    # 初始化负荷累加器，每个 COMID 对应一个与其日期序列对齐的 Series
    load_acc = {comid: pd.Series(0.0, index=data.index) for comid, data in comid_data.items()}

    # 对入度为 0 的头部河段，令 y_n = E
    queue = [comid for comid, deg in indegree.items() if deg == 0]
    logging.info(f"Found {len(queue)} headwater segments")
    for comid in queue:
        data = comid_data[comid]
        data['y_n'] = data['E'] 
        comid_data[comid] = data

    def compute_R_series(Q_up: pd.Series, Q_down: pd.Series) -> pd.Series:
        return compute_retainment_factor(v_f, Q_up, Q_down)

    # 利用队列逐步处理上游，将贡献传递到下游
    logging.info("Starting flow routing calculation...")
    processed_count = 0
    while queue:
        current = queue.pop(0)
        processed_count += 1
        if processed_count % 1000 == 0:
            logging.info(f"Processed {processed_count} segments so far")
            
        current_data = comid_data[current]
        # 修正：直接从 next_down_ids 中获取下游，而不是从当前数据中读取
        next_down = next_down_ids.get(current, 0)
        if next_down == 0:
            continue
        down_data = comid_data[next_down]
        common_dates = current_data.index.intersection(down_data.index)
        if len(common_dates) == 0:
            logging.warning(f"Warning: 日期不对齐，COMID {current} 与 COMID {next_down}")
            logging.warning(f"  当前日期: {current_data.index}")
            logging.warning(f"  下游日期: {down_data.index}")
        if len(common_dates) > 0:
            y_n_current = current_data['y_n'].reindex(common_dates)
            Q_current = current_data['Qout'].reindex(common_dates)
            Q_down = down_data['Qout'].reindex(common_dates)
            R_series = compute_R_series(Q_current, Q_down)
            contribution = y_n_current * R_series * Q_current
            load_acc[next_down] = load_acc[next_down].add(contribution, fill_value=0.0)
        indegree[next_down] -= 1   
        if indegree[next_down] == 0:
            down_data = comid_data[next_down]
            y_up_down = load_acc[next_down] / down_data['Qout'].replace(0, np.nan)
            y_up_down = y_up_down.fillna(0.0)
            down_data['y_up'] = y_up_down
            down_data['y_n'] = down_data['E'] + down_data['y_up']
            comid_data[next_down] = down_data
            queue.append(next_down)

    # 合并所有 COMID 的时间序列为长格式 DataFrame，并重命名新列（带迭代标记）
    logging.info("Merging results...")
    result_list = []
    for comid, data in comid_data.items():
        temp = data.reset_index()
        temp['COMID'] = comid
        result_list.append(temp)
    result_df = pd.concat(result_list, ignore_index=True)
    result_df = result_df.rename(columns={
        'E': f'E_{iteration}',
        'y_up': f'y_up_{iteration}',
        'y_n': f'y_n_{iteration}'
    })
    
    logging.info(f"Flow routing calculation for iteration {iteration} complete")
    return result_df