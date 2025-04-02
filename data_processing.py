import pandas as pd
import numpy as np
from typing import List, Optional
import numba
import time
import logging
import sys

# Import our custom tqdm that supports logging
try:
    from tqdm_logging import tqdm
except ImportError:
    from tqdm import tqdm

def detect_and_handle_anomalies(df, columns_to_check=['Qout'], 
                               check_negative=True, check_outliers=True,
                               fix_negative=False, fix_outliers=False,
                               negative_replacement=0.001,
                               outlier_method='iqr', outlier_threshold=1.5,
                               verbose=True, logger=None):
    """
    检测并可选地修复DataFrame中指定列的异常值。
    
    参数:
    -----------
    df : pandas.DataFrame
        要检查异常值的DataFrame
    columns_to_check : list
        要检查异常值的列名列表
    check_negative : bool
        是否检查负值
    check_outliers : bool
        是否检查异常值
    fix_negative : bool
        是否修复负值
    fix_outliers : bool
        是否修复异常值
    negative_replacement : float
        替换负值时使用的值
    outlier_method : str
        检测异常值的方法 ('iqr', 'zscore', 'percentile')
    outlier_threshold : float
        异常值检测的阈值
    verbose : bool
        是否打印有关检测到的异常值的信息
    logger : logging.Logger or None
        用于记录消息的Logger对象；如果为None，则使用print
    
    返回:
    --------
    pandas.DataFrame
        如果请求修复异常值，则返回修复后的DataFrame
    dict
        包含异常值检测结果的字典
    """
    import numpy as np
    import pandas as pd
    
    # 创建DataFrame的副本，避免修改原始数据
    df_result = df.copy()
    
    # 初始化结果字典
    results = {
        'has_anomalies': False,
        'columns_with_anomalies': [],
        'negative_counts': {},
        'outlier_counts': {},
        'fixed_negative_counts': {},
        'fixed_outlier_counts': {}
    }
    
    # 用于日志记录的函数
    def log_message(message, level='info'):
        if logger:
            if level == 'info':
                logger.info(message)
            elif level == 'warning':
                logger.warning(message)
            elif level == 'error':
                logger.error(message)
        elif verbose:
            print(message)
    
    # 检查每个指定的列
    for column in columns_to_check:
        if column not in df.columns:
            log_message(f"警告: 在DataFrame中未找到列 '{column}'", 'warning')
            continue
        
        # 检查负值
        if check_negative:
            negative_mask = df[column] < 0
            negative_count = negative_mask.sum()
            
            if negative_count > 0:
                results['has_anomalies'] = True
                results['negative_counts'][column] = negative_count
                
                if column not in results['columns_with_anomalies']:
                    results['columns_with_anomalies'].append(column)
                
                # 查找并打印包含负值的COMID
                if negative_count > 0 and 'COMID' in df.columns:
                    comids_with_negative = df.loc[negative_mask, 'COMID'].unique()
                    log_message(f"在列 '{column}' 中发现 {negative_count} 个负值", 'warning')
                    log_message(f"包含负 {column} 值的COMID: {comids_with_negative}", 'warning')
                    log_message(f"负 {column} 值样例:")
                    log_message(df[negative_mask].head().to_string())
                else:
                    log_message(f"在列 '{column}' 中发现 {negative_count} 个负值", 'warning')
                
                # 如果请求修复负值
                if fix_negative:
                    df_result.loc[negative_mask, column] = negative_replacement
                    results['fixed_negative_counts'][column] = negative_count
                    log_message(f"已修复列 '{column}' 中的 {negative_count} 个负值", 'info')
        
        # 检查异常值
        if check_outliers:
            outlier_mask = np.zeros(len(df), dtype=bool)
            
            # 临时去除负值以进行异常值计算（如果适用）
            temp_column = df[column].copy()
            if check_negative and fix_negative:
                temp_column = temp_column.clip(lower=0)
            
            if outlier_method == 'iqr':
                # IQR方法 (四分位距)
                Q1 = temp_column.quantile(0.25)
                Q3 = temp_column.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - outlier_threshold * IQR
                upper_bound = Q3 + outlier_threshold * IQR
                outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            
            elif outlier_method == 'zscore':
                # Z-score方法
                from scipy import stats
                z_scores = np.abs(stats.zscore(temp_column, nan_policy='omit'))
                outlier_mask = z_scores > outlier_threshold
            
            elif outlier_method == 'percentile':
                # 百分位方法
                lower_bound = temp_column.quantile(outlier_threshold / 100)
                upper_bound = temp_column.quantile(1 - outlier_threshold / 100)
                outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                results['has_anomalies'] = True
                results['outlier_counts'][column] = outlier_count
                
                if column not in results['columns_with_anomalies']:
                    results['columns_with_anomalies'].append(column)
                
                # 查找并打印包含异常值的COMID
                if outlier_count > 0 and 'COMID' in df.columns:
                    comids_with_outliers = df.loc[outlier_mask, 'COMID'].unique()
                    log_message(f"在列 '{column}' 中发现 {outlier_count} 个异常值", 'warning')
                    log_message(f"包含异常 {column} 值的COMID: {comids_with_outliers}", 'warning')
                    log_message(f"异常 {column} 值样例:")
                    log_message(df[outlier_mask].head().to_string())
                else:
                    log_message(f"在列 '{column}' 中发现 {outlier_count} 个异常值", 'warning')
                
                # 如果请求修复异常值
                if fix_outliers:
                    # 使用中位数替换异常值
                    median_value = temp_column.median()
                    df_result.loc[outlier_mask, column] = median_value
                    results['fixed_outlier_counts'][column] = outlier_count
                    log_message(f"已修复列 '{column}' 中的 {outlier_count} 个异常值", 'info')
    
    # 汇总结果
    if results['has_anomalies']:
        log_message("数据异常检测结果摘要:", 'info')
        for column in results['columns_with_anomalies']:
            summary = []
            if column in results['negative_counts']:
                summary.append(f"{results['negative_counts'][column]} 个负值")
            if column in results['outlier_counts']:
                summary.append(f"{results['outlier_counts'][column]} 个异常值")
            log_message(f"  列 '{column}': {', '.join(summary)}", 'info')
    else:
        log_message("未检测到异常值。", 'info')
    
    return df_result, results


def check_river_network_consistency(river_info, verbose=True, logger=None):
    """
    检查河网拓扑结构的一致性。
    
    参数:
    -----------
    river_info : pandas.DataFrame
        包含河网拓扑结构的DataFrame，必须包含'COMID'和'NextDownID'列
    verbose : bool
        是否打印检查结果
    logger : logging.Logger or None
        用于记录消息的Logger对象；如果为None，则使用print
        
    返回:
    --------
    dict
        包含检查结果的字典
    """
    import numpy as np
    import pandas as pd
    
    # 用于日志记录的函数
    def log_message(message, level='info'):
        if logger:
            if level == 'info':
                logger.info(message)
            elif level == 'warning':
                logger.warning(message)
            elif level == 'error':
                logger.error(message)
        elif verbose:
            print(message)
    
    # 初始化结果字典
    results = {
        'has_issues': False,
        'missing_comids': [],
        'orphaned_comids': [],
        'cycles': [],
        'multiple_upstreams': {}
    }
    
    if 'COMID' not in river_info.columns or 'NextDownID' not in river_info.columns:
        log_message("错误: river_info必须包含'COMID'和'NextDownID'列", 'error')
        results['has_issues'] = True
        return results
    
    # 检查是否所有NextDownID都存在于COMID中（除了终点河段的NextDownID=0）
    all_comids = set(river_info['COMID'])
    next_down_ids = set(river_info['NextDownID'])
    next_down_ids.discard(0)  # 移除终点河段标识0
    
    missing_comids = next_down_ids - all_comids
    if missing_comids:
        results['has_issues'] = True
        results['missing_comids'] = list(missing_comids)
        log_message(f"警告: 发现 {len(missing_comids)} 个引用的下游河段未在COMID列中找到", 'warning')
        log_message(f"缺失的COMID: {list(missing_comids)[:10]}..." if len(missing_comids) > 10 else f"缺失的COMID: {list(missing_comids)}")
    
    # 检查孤立的河段（没有上游也没有下游）
    next_down_dict = river_info.set_index('COMID')['NextDownID'].to_dict()
    upstream_counts = {}
    
    for comid, next_down in next_down_dict.items():
        if next_down != 0 and next_down in upstream_counts:
            upstream_counts[next_down] = upstream_counts.get(next_down, 0) + 1
        else:
            upstream_counts[next_down] = 1
    
    orphaned_comids = []
    for comid in all_comids:
        if comid not in upstream_counts and next_down_dict.get(comid, 0) == 0:
            orphaned_comids.append(comid)
    
    if orphaned_comids:
        results['has_issues'] = True
        results['orphaned_comids'] = orphaned_comids
        log_message(f"警告: 发现 {len(orphaned_comids)} 个孤立河段（无上游也无下游）", 'warning')
        log_message(f"孤立的COMID: {orphaned_comids[:10]}..." if len(orphaned_comids) > 10 else f"孤立的COMID: {orphaned_comids}")
    
    # 检查循环引用
    def find_cycle(comid, visited=None, path=None):
        if visited is None:
            visited = set()
        if path is None:
            path = []
        
        if comid in path:
            return path[path.index(comid):] + [comid]
        
        if comid in visited or comid not in next_down_dict:
            return None
        
        visited.add(comid)
        path.append(comid)
        
        next_comid = next_down_dict.get(comid, 0)
        if next_comid == 0:
            return None
        
        return find_cycle(next_comid, visited, path)
    
    cycles = []
    for comid in all_comids:
        if comid not in next_down_dict:
            continue
        cycle = find_cycle(comid)
        if cycle:
            cycles.append(cycle)
    
    unique_cycles = []
    cycle_sets = []
    for cycle in cycles:
        cycle_set = set(cycle)
        if cycle_set not in cycle_sets:
            cycle_sets.append(cycle_set)
            unique_cycles.append(cycle)
    
    if unique_cycles:
        results['has_issues'] = True
        results['cycles'] = unique_cycles
        log_message(f"警告: 发现 {len(unique_cycles)} 个循环引用", 'warning')
        for i, cycle in enumerate(unique_cycles[:5]):
            log_message(f"循环 {i+1}: {' -> '.join(map(str, cycle))}", 'warning')
        if len(unique_cycles) > 5:
            log_message(f"... 等 {len(unique_cycles) - 5} 个循环", 'warning')
    
    # 检查具有多个上游的河段
    multiple_upstreams = {comid: count for comid, count in upstream_counts.items() if count > 2 and comid != 0}
    if multiple_upstreams:
        results['has_issues'] = True
        results['multiple_upstreams'] = multiple_upstreams
        log_message(f"信息: 发现 {len(multiple_upstreams)} 个具有3个及以上上游的河段", 'info')
        items = list(multiple_upstreams.items())
        for comid, count in sorted(items[:10], key=lambda x: x[1], reverse=True):
            log_message(f"COMID {comid}: {count} 个上游", 'info')
        if len(multiple_upstreams) > 10:
            log_message(f"... 等 {len(multiple_upstreams) - 10} 个河段", 'info')
    
    # 总结
    if results['has_issues']:
        log_message("河网拓扑结构检查发现问题。", 'warning')
    else:
        log_message("河网拓扑结构检查通过，未发现明显问题。", 'info')
    
    return results

# 定义时间序列数据标准化函数
def standardize_time_series(X_train, X_val):
    from sklearn.preprocessing import StandardScaler
    N_train, T, input_dim = X_train.shape
    scaler = StandardScaler()
    # 将 3D 数据展平为 2D
    X_train_2d = X_train.reshape(-1, input_dim)
    scaler.fit(X_train_2d)
    X_train_scaled = scaler.transform(X_train_2d).reshape(N_train, T, input_dim)
    
    N_val = X_val.shape[0]
    X_val_2d = X_val.reshape(-1, input_dim)
    X_val_scaled = scaler.transform(X_val_2d).reshape(N_val, T, input_dim)
    
    return X_train_scaled, X_val_scaled, scaler


# 对整个时间序列数据进行标准化
def standardize_time_series_all(X):
    from sklearn.preprocessing import StandardScaler
    N, T, input_dim = X.shape
    scaler = StandardScaler()
    # 将 3D 数据展平成 2D
    X_2d = X.reshape(-1, input_dim)
    scaler.fit(X_2d)
    X_scaled_2d = scaler.transform(X_2d)
    # 恢复成原来的形状
    X_scaled = X_scaled_2d.reshape(N, T, input_dim)
    return X_scaled, scaler


# 对属性数据进行标准化（使用所有属性数据计算统计量）
def standardize_attributes(attr_dict):
    from sklearn.preprocessing import StandardScaler
    keys = list(attr_dict.keys())
    attr_matrix = np.vstack([attr_dict[k] for k in keys])
    scaler = StandardScaler()
    scaler.fit(attr_matrix)
    attr_matrix_scaled = scaler.transform(attr_matrix)
    scaled_attr_dict = {k: attr_matrix_scaled[i] for i, k in enumerate(keys)}
    return scaled_attr_dict, scaler

def load_daily_data(csv_path: str) -> pd.DataFrame:
    """
    加载日尺度数据
    输入：
        csv_path: CSV 文件路径，文件中需包含 'COMID'、'Date'、各项驱动特征、流量 (Qout)、TN、TP 等字段
    输出：
        返回一个 DataFrame，每一行记录某个 COMID 在特定日期的数据
    """
    df = pd.read_csv(csv_path)
    return df

def load_river_info(csv_path: str) -> pd.DataFrame:
    """
    加载河段信息数据
    输入：
        csv_path: CSV 文件路径，包含 'COMID'、'NextDownID'、'up1'、'up2'、'up3'、'up4' 等字段
    输出：
        返回一个 DataFrame
    """
    df = pd.read_csv(csv_path)
    return df

def load_river_attributes(csv_path: str) -> pd.DataFrame:
    """
    加载河段属性数据
    输入：
        csv_path: CSV 文件路径，包含 'COMID'、lengthkm、lengthdir、sinuosity、slope、uparea、order_、NextDownID 等属性
    输出：
        返回一个 DataFrame
    """
    df = pd.read_csv(csv_path)
    return df

def merge_datasets(daily_df: pd.DataFrame, info_df: pd.DataFrame, attr_df: pd.DataFrame) -> pd.DataFrame:
    """
    合并日尺度数据、河段信息与属性数据
    输入：
        daily_df: 日尺度数据 DataFrame
        info_df: 河段信息 DataFrame
        attr_df: 河段属性 DataFrame
    输出：
        返回合并后的 DataFrame
    """
    merged = pd.merge(daily_df, info_df, on='COMID', how='left')
    merged = pd.merge(merged, attr_df, on='COMID', how='left')
    return merged


# @numba.njit
# def extract_windows_numba(data: np.ndarray, time_window: int, input_dim: int):
#     n, total_features = data.shape
#     target_dim = total_features - input_dim
#     valid_count = 0
#     # 第一遍扫描：计数有效窗口
#     for i in range(n - time_window + 1):
#         valid = True
#         for j in range(target_dim):
#             if np.isnan(data[i + time_window - 1, input_dim + j]):
#                 valid = False
#                 break
#         if valid:
#             valid_count += 1

#     # 预分配输出数组
#     X_windows = np.empty((valid_count, time_window, input_dim), dtype=data.dtype)
#     Y_windows = np.empty((valid_count, target_dim), dtype=data.dtype)
#     idx = 0
#     for i in range(n - time_window + 1):
#         valid = True
#         for j in range(target_dim):
#             if np.isnan(data[i + time_window - 1, input_dim + j]):
#                 valid = False
#                 break
#         if valid:
#             X_windows[idx, :, :] = data[i:i + time_window, :input_dim]
#             Y_windows[idx, :] = data[i + time_window - 1, input_dim:]
#             idx += 1
#     return X_windows, Y_windows


# def build_sliding_windows_for_subset_3(
#     df: pd.DataFrame,
#     comid_list: List[str],
#     input_cols: Optional[List[str]] = None,
#     target_cols: List[str] = ["TN","TP"],
#     time_window: int = 10,
#     skip_missing_targets: bool = True
# ):
#     """
#     构造滑动窗口数据切片（纯 Python 版本，增加进度条显示）
#     输入：
#         df: 包含日尺度数据的 DataFrame，必须包含 'COMID' 和 'date'
#         comid_list: 要构造数据切片的 COMID 列表
#         input_cols: 输入特征列名列表；若为 None，则除去 {"COMID", "date"} 与 target_cols 后的所有列
#         target_cols: 目标变量列名列表
#         time_window: 时间窗口长度
#         skip_missing_targets: 若为 True，则跳过目标变量包含缺失值的滑窗；若为 False，则保留这些滑窗
#     输出：
#         返回 (X_array, Y_array, COMIDs, Dates)
#             X_array: 形状为 (N, time_window, len(input_cols)) 的数组
#             Y_array: 形状为 (N, len(target_cols)) 的数组，通常取时间窗口最后一时刻的目标值
#             COMIDs: 形状为 (N,) 的数组，每个切片对应的 COMID
#             Dates: 形状为 (N,) 的数组，每个切片最后时刻的日期
#     """
#     sub_df = df[df["COMID"].isin(comid_list)].copy()
#     if input_cols is None:
#         exclude_cols = {"COMID", "date"}.union(target_cols)
#         input_cols = [col for col in df.columns if col not in exclude_cols]
#     X_list, Y_list, comid_track, date_track = [], [], [], []
    
#     # 使用 tqdm 显示每个 COMID 组的处理进度
#     for comid, group_df in tqdm(sub_df.groupby("COMID"), desc="Processing groups (subset_3)"):
#         group_df = group_df.sort_values("date").reset_index(drop=True)
#         needed_cols = input_cols + target_cols
#         sub_data = group_df[needed_cols].values  # shape=(n_rows, len(needed_cols))
        
#         for start_idx in range(len(sub_data) - time_window + 1):
#             window_data = sub_data[start_idx : start_idx + time_window]
#             x_window = window_data[:, :len(input_cols)]
#             y_values = window_data[-1, len(input_cols):]
            
#             # 根据 skip_missing_targets 参数决定是否跳过含有缺失值的滑窗
#             if skip_missing_targets and np.isnan(y_values).any():
#                 continue  # 跳过包含缺失值的滑窗
            
#             X_list.append(x_window)
#             Y_list.append(y_values)
#             comid_track.append(comid)
#             date_track.append(group_df.loc[start_idx + time_window - 1, "date"])
    
#     if not X_list:
#         return None, None, None, None
    
#     X_array = np.array(X_list, dtype=np.float32)
#     Y_array = np.array(Y_list, dtype=np.float32)
#     COMIDs = np.array(comid_track)
#     Dates = np.array(date_track)
#     return X_array, Y_array, COMIDs, Dates

# @numba.njit
# def extract_windows_with_indices_numba(data: np.ndarray, time_window: int, input_dim: int):
#     n, total_features = data.shape
#     target_dim = total_features - input_dim
#     valid_count = 0
#     # 第一遍扫描：计数有效窗口
#     for i in range(n - time_window + 1):
#         valid = True
#         for j in range(target_dim):
#             if np.isnan(data[i + time_window - 1, input_dim + j]):
#                 valid = False
#                 break
#         if valid:
#             valid_count += 1

#     # 预分配输出数组
#     X_windows = np.empty((valid_count, time_window, input_dim), dtype=data.dtype)
#     Y_windows = np.empty((valid_count, target_dim), dtype=data.dtype)
#     valid_indices = np.empty(valid_count, dtype=np.int64)
#     idx = 0
#     for i in range(n - time_window + 1):
#         valid = True
#         for j in range(target_dim):
#             if np.isnan(data[i + time_window - 1, input_dim + j]):
#                 valid = False
#                 break
#         if valid:
#             X_windows[idx, :, :] = data[i:i + time_window, :input_dim]
#             Y_windows[idx, :] = data[i + time_window - 1, input_dim:]
#             valid_indices[idx] = i
#             idx += 1
#     return X_windows, Y_windows, valid_indices

# def build_sliding_windows_for_subset_4(
#     df: pd.DataFrame,
#     comid_list: List[str],
#     input_cols: Optional[List[str]] = None,
#     target_cols: List[str] = ["TN"],
#     time_window: int = 10
# ):
#     """
#     在 df 中，根据 comid_list 指定的河段进行滑窗切片。
#       1. 先筛选 df["COMID"] 在 comid_list 中
#       2. 若未指定 input_cols，使用除 COMID, date, target_cols 外的全部列作为输入
#       3. 每个河段按时间升序，构造 (X, Y, COMID, Date) 切片
#       4. X.shape = (N, time_window, len(input_cols))
#          Y.shape = (N, len(target_cols))
#          COMIDs.shape = (N,)
#          Dates.shape = (N,)

#     参数:
#         df: 已包含 [COMID, date] 及相关特征列(如 flow, temperature_2m_mean 等)
#         comid_list: 需要切片的 COMID 列表(字符串或整数均可，但需和 df["COMID"] 的 dtype 对应)
#         input_cols: 用作时序输入的列。如果未指定，将自动选用 df 的所有列，排除 ["COMID", "date"] + target_cols
#         target_cols: 目标列列表 (如 ["TN", "TP"])
#         time_window: 滑窗大小 (默认10)
#     """
#     # 1. 先将 df 筛选到 comid_list
#     sub_df = df[df["COMID"].isin(comid_list)].copy()
    
#     # 2. 若未指定 input_cols，默认使用所有列，排除 ["COMID", "date"] + target_cols
#     if input_cols is None:
#         exclude_cols = {"COMID", "date"}.union(target_cols)
#         input_cols = [col for col in df.columns if col not in exclude_cols]
    
#     # 3. 分组并做滑窗
#     X_list, Y_list, comid_track, date_track = [], [], [], []
    
#     for comid, group_df in sub_df.groupby("COMID"):
#         group_df = group_df.sort_values("date").reset_index(drop=True)
#         # 确保滑窗只包含 input_cols 和 target_cols
#         needed_cols = input_cols + target_cols
#         sub_data = group_df[needed_cols].values  # shape=(n_rows, len(needed_cols))
        
#         for start_idx in range(len(sub_data) - time_window + 1):
#             window_data = sub_data[start_idx : start_idx + time_window]
            
#             # X 部分
#             x_window = window_data[:, :len(input_cols)]  # 输入特征部分
            
#             # Y 部分
#             y_values = window_data[-1, len(input_cols):]  # 最后一天的目标列值
#             if np.isnan(y_values).any():
#                 continue  # 跳过该滑窗
            
#             # Date 部分
#             date_value = group_df.loc[start_idx + time_window - 1, "date"]  # 滑窗最后一天的日期
            
#             # 添加到结果列表
#             X_list.append(x_window)
#             Y_list.append(y_values)
#             comid_track.append(comid)
#             date_track.append(date_value)
    
#     if not X_list:
#         return None, None, None, None
    
#     X_array = np.array(X_list, dtype=np.float32)  # (N, time_window, len(input_cols))
#     Y_array = np.array(Y_list, dtype=np.float32)  # (N, len(target_cols))
#     COMIDs = np.array(comid_track)                # (N,)  # 保持原类型或转成 str
#     Dates = np.array(date_track)                  # (N,)  # 日期部分
    
#     return X_array, Y_array, COMIDs, Dates



# def build_sliding_windows_for_subset_5(
#     df: pd.DataFrame,
#     comid_list: List[str],
#     input_cols: Optional[List[str]] = None,
#     target_cols: List[str] = ["TN"],
#     time_window: int = 10
# ):
#     """
#     构造滑动窗口数据切片（结合 tqdm 进度条与 Numba 加速，并整合了有效窗口索引的计算）
#     输入：
#         df: 包含日尺度数据的 DataFrame，必须包含 'COMID' 和 'date'
#         comid_list: 要构造数据切片的 COMID 列表
#         input_cols: 输入特征列名列表；若为 None，则除去 {"COMID", "date"} 与 target_cols 后的所有列
#         target_cols: 目标变量列名列表
#         time_window: 时间窗口长度
#     输出：
#         返回 (X_array, Y_array, COMIDs, Dates)
#             X_array: 形状为 (N, time_window, input_dim) 的数组
#             Y_array: 形状为 (N, len(target_cols)) 的数组，通常取时间窗口最后时刻的目标值
#             COMIDs: 形状为 (N,) 的数组，每个切片对应的 COMID
#             Dates: 形状为 (N,) 的数组，每个切片最后时刻的日期
#     """
#     sub_df = df[df["COMID"].isin(comid_list)].copy()
#     if input_cols is None:
#         exclude = {"COMID", "date"}.union(set(target_cols))
#         input_cols = [col for col in df.columns if col not in exclude]
#     X_list, Y_list, comid_track, date_track = [], [], [], []
    
#     for comid, group_df in tqdm(sub_df.groupby("COMID"), desc="Processing groups (subset_4)"):
#         group_df = group_df.sort_values("date").reset_index(drop=True)
#         cols = input_cols + target_cols
#         data_array = group_df[cols].values
#         if data_array.shape[0] < time_window:
#             print(f"警告：COMID {comid} 数据不足，跳过。")
#             continue

#         # 利用 numba 优化函数同时提取滑动窗口数据和对应的起始索引
#         X_windows, Y_windows, valid_indices = extract_windows_with_indices_numba(data_array, time_window, len(input_cols))
#         if valid_indices.size == 0:
#             print(f"警告：COMID {comid} 无有效窗口，跳过。")
#             continue

#         for idx, start_idx in enumerate(valid_indices):
#             X_list.append(X_windows[idx])
#             Y_list.append(Y_windows[idx])
#             comid_track.append(comid)
#             # 直接用有效索引获取对应窗口最后一时刻的日期
#             date_val = pd.to_datetime(group_df.loc[start_idx + time_window - 1, "date"])
#             date_track.append(date_val)
            
#     if not X_list:
#         return None, None, None, None
#     X_array = np.array(X_list, dtype=np.float32)
#     Y_array = np.array(Y_list, dtype=np.float32)
#     COMIDs = np.array(comid_track)
#     Dates = np.array(date_track)
#     return X_array, Y_array, COMIDs, Dates

def build_sliding_windows_for_subset_6(
    df: pd.DataFrame,
    comid_list: List[str],
    input_cols: Optional[List[str]] = None,
    target_col: str = "TN",
    all_target_cols: List[str] = ["TN","TP"],
    time_window: int = 10,
    skip_missing_targets: bool = True
):
    """
    构造滑动窗口数据切片（纯 Python 版本）
    输入：
        df: 包含日尺度数据的 DataFrame，必须包含 'COMID' 和 'date'
        comid_list: 要构造数据切片的 COMID 列表
        input_cols: 输入特征列名列表；若为 None，则除去 {"COMID", "date"} 与 target_cols 后的所有列
        target_cols: 目标变量列名列表
        time_window: 时间窗口长度
        skip_missing_targets: 若为 True，则跳过目标变量包含缺失值的滑窗；若为 False，则保留这些滑窗
    输出：
        返回 (X_array, Y_array, COMIDs, Dates)
            X_array: 形状为 (N, time_window, len(input_cols)) 的数组
            Y_array: 形状为 (N, len(target_cols)) 的数组，通常取时间窗口最后一时刻的目标值
            COMIDs: 形状为 (N,) 的数组，每个切片对应的 COMID
            Dates: 形状为 (N,) 的数组，每个切片最后时刻的日期
    """
    sub_df = df[df["COMID"].isin(comid_list)].copy()
    if input_cols is None:
        exclude_cols = {"COMID", "date"}.union(all_target_cols)
        input_cols = [col for col in df.columns if col not in exclude_cols]
    X_list, Y_list, comid_track, date_track = [], [], [], []
    
    # 移除tqdm，使用普通的for循环
    for comid, group_df in sub_df.groupby("COMID"):
        group_df = group_df.sort_values("date").reset_index(drop=True)
        needed_cols = input_cols + all_target_cols
        sub_data = group_df[needed_cols].values  # shape=(n_rows, len(needed_cols))
        
        for start_idx in range(len(sub_data) - time_window + 1):
            window_data = sub_data[start_idx : start_idx + time_window]
            x_window = window_data[:, :len(input_cols)]
            y_values = window_data[-1, len(input_cols):]
            
            # 根据 skip_missing_targets 参数决定是否跳过含有缺失值的滑窗
            if skip_missing_targets and np.isnan(y_values).any():
                continue  # 跳过包含缺失值的滑窗
            
            X_list.append(x_window)
            Y_list.append(y_values)
            comid_track.append(comid)
            date_track.append(group_df.loc[start_idx + time_window - 1, "date"])
    
    if not X_list:
        return None, None, None, None
    
    X_array = np.array(X_list, dtype=np.float32)
    Y_array = np.array(Y_list, dtype=np.float32)
    COMIDs = np.array(comid_track)
    Dates = np.array(date_track)
    return X_array, Y_array, COMIDs, Dates

def build_sliding_windows_for_subset_7(
    df: pd.DataFrame,
    comid_list: List[str],
    input_cols: Optional[List[str]] = None,
    target_cols: List[str] = ["TN"],
    all_target_cols: List[str] = ["TN","TP"],
    time_window: int = 10,
    skip_missing_targets: bool = True
):
    """
    构造滑动窗口数据切片（带进度条版本）
    输入：
        df: 包含日尺度数据的 DataFrame，必须包含 'COMID' 和 'date'
        comid_list: 要构造数据切片的 COMID 列表
        input_cols: 输入特征列名列表；若为 None，则除去 {"COMID", "date"} 与 target_cols 后的所有列
        target_cols: 目标变量列名列表
        time_window: 时间窗口长度
        skip_missing_targets: 若为 True，则跳过目标变量包含缺失值的滑窗；若为 False，则保留这些滑窗
    输出：
        返回 (X_array, Y_array, COMIDs, Dates)
            X_array: 形状为 (N, time_window, len(input_cols)) 的数组
            Y_array: 形状为 (N, len(target_cols)) 的数组，通常取时间窗口最后一时刻的目标值
            COMIDs: 形状为 (N,) 的数组，每个切片对应的 COMID
            Dates: 形状为 (N,) 的数组，每个切片最后时刻的日期
    """
    sub_df = df[df["COMID"].isin(comid_list)].copy()
    if input_cols is None:
        exclude_cols = {"COMID", "date"}.union(all_target_cols)
        input_cols = [col for col in df.columns if col not in exclude_cols]
    X_list, Y_list, comid_track, date_track = [], [], [], []
    
    # 使用tqdm添加进度条
    from tqdm import tqdm
    
    # 获取COMID分组并创建进度条
    comid_groups = list(sub_df.groupby("COMID"))
    for comid, group_df in tqdm(comid_groups, desc="Processing COMIDs",file=sys.stdout):
        group_df = group_df.sort_values("date").reset_index(drop=True)
        needed_cols = input_cols + all_target_cols
        sub_data = group_df[needed_cols].values  # shape=(n_rows, len(needed_cols))
        
        for start_idx in range(len(sub_data) - time_window + 1):
            window_data = sub_data[start_idx : start_idx + time_window]
            x_window = window_data[:, :len(input_cols)]
            y_values = window_data[-1, len(input_cols):]
            
            # 根据 skip_missing_targets 参数决定是否跳过含有缺失值的滑窗
            if skip_missing_targets and np.isnan(y_values).any():
                continue  # 跳过包含缺失值的滑窗
            
            X_list.append(x_window)
            Y_list.append(y_values)
            comid_track.append(comid)
            date_track.append(group_df.loc[start_idx + time_window - 1, "date"])
    
    if not X_list:
        return None, None, None, None
    
    X_array = np.array(X_list, dtype=np.float32)
    Y_array = np.array(Y_list, dtype=np.float32)
    COMIDs = np.array(comid_track)
    Dates = np.array(date_track)
    return X_array, Y_array, COMIDs, Dates

# 为方便引用，提供别名，_4的意思是第四版,其他地方很多地方用到这个函数名称，遗留问题
build_sliding_windows_for_subset = build_sliding_windows_for_subset_6


##主函数，使用随机数据来测试build_sliding_windows_for_subset_4和build_sliding_windows_for_subset_3，比较速度差距，以及观察结果一致性

def main():
    np.random.seed(42)
    
    # 数据规模设定
    num_comids = 600
    rows_per_comid = 1500
    comid_list = [f"COMID_{i}" for i in range(num_comids)]
    
    # 输入变量数量设为10个
    input_feature_names = [f"feature_{i}" for i in range(1, 21)]
    target_cols = ["TN"]
    
    df_list = []
    for comid in comid_list:
        dates = pd.date_range(start="2020-01-01", periods=rows_per_comid, freq="D")
        # 生成10个输入特征，取值范围可自行调整
        features_data = {name: np.random.rand(rows_per_comid) * 100 for name in input_feature_names}
        # 生成目标变量 "TN"，例如范围在 0 到 10 之间
        TN = np.random.rand(rows_per_comid) * 10
        temp_df = pd.DataFrame({
            "COMID": comid,
            "date": dates,
            "TN": TN
        })
        for name in input_feature_names:
            temp_df[name] = features_data[name]
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    
    input_cols = input_feature_names
    time_window = 10      
    
    # 测试 build_sliding_windows_for_subset_3
    start = time.time()
    X3, Y3, COMIDs3, Dates3 = build_sliding_windows_for_subset_3(df, comid_list, input_cols=input_cols, target_cols=target_cols, time_window=time_window)
    duration3 = time.time() - start
    
    # 测试 build_sliding_windows_for_subset_4
    start = time.time()
    X4, Y4, COMIDs4, Dates4 = build_sliding_windows_for_subset_4(df, comid_list, input_cols=input_cols, target_cols=target_cols, time_window=time_window)
    duration4 = time.time() - start
    
    print("=== build_sliding_windows_for_subset_3 ===")
    if X3 is not None:
        print("X3 shape:", X3.shape)
        print("Y3 shape:", Y3.shape)
        print("COMIDs3 sample:", COMIDs3[:5])
        print("Dates3 sample:", Dates3[:5])
    else:
        print("没有有效的窗口数据。")
    print(f"运行时间: {duration3:.4f}秒\n")
    
    print("=== build_sliding_windows_for_subset_4 ===")
    if X4 is not None:
        print("X4 shape:", X4.shape)
        print("Y4 shape:", Y4.shape)
        print("COMIDs4 sample:", COMIDs4[:5])
        print("Dates4 sample:", Dates4[:5])
    else:
        print("没有有效的窗口数据。")
    print(f"运行时间: {duration4:.4f}秒\n")
    
    if X3 is not None and X4 is not None:
        consistent = (X3.shape == X4.shape and Y3.shape == Y4.shape and len(COMIDs3) == len(COMIDs4) and len(Dates3) == len(Dates4))
        print("结果一致性检查:", consistent)
    else:
        print("无法比较结果一致性，因为至少一个方法没有生成有效窗口。")
    
    ##不仅通过形状观察一致性，还可以通过数据值观察一致性
    if X3 is not None and X4 is not None:
        if consistent:
            print("=== 结果一致性检查 ===")
            for i in range(min(X3.shape[0], X4.shape[0])):
                if not np.allclose(X3[i], X4[i]):
                    print(f"警告：第 {i} 个窗口数据不一致。")
                if not np.allclose(Y3[i], Y4[i]):
                    print(f"警告：第 {i} 个窗口目标值不一致。")
            print("结果一致性检查结束。")
        else:
            print("结果一致性检查失败，因为结果不一致。")

if __name__ == "__main__":
    main()