import pandas as pd
import numpy as np
from typing import List, Optional
from tqdm import tqdm
import numba
import time

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


def build_sliding_windows_for_subset_3(
    df: pd.DataFrame,
    comid_list: List[str],
    input_cols: Optional[List[str]] = None,
    target_cols: List[str] = ["TN","TP"],
    time_window: int = 10,
    skip_missing_targets: bool = True
):
    """
    构造滑动窗口数据切片（纯 Python 版本，增加进度条显示）
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
        exclude_cols = {"COMID", "date"}.union(target_cols)
        input_cols = [col for col in df.columns if col not in exclude_cols]
    X_list, Y_list, comid_track, date_track = [], [], [], []
    
    # 使用 tqdm 显示每个 COMID 组的处理进度
    for comid, group_df in tqdm(sub_df.groupby("COMID"), desc="Processing groups (subset_3)"):
        group_df = group_df.sort_values("date").reset_index(drop=True)
        needed_cols = input_cols + target_cols
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

@numba.njit
def extract_windows_with_indices_numba(data: np.ndarray, time_window: int, input_dim: int):
    n, total_features = data.shape
    target_dim = total_features - input_dim
    valid_count = 0
    # 第一遍扫描：计数有效窗口
    for i in range(n - time_window + 1):
        valid = True
        for j in range(target_dim):
            if np.isnan(data[i + time_window - 1, input_dim + j]):
                valid = False
                break
        if valid:
            valid_count += 1

    # 预分配输出数组
    X_windows = np.empty((valid_count, time_window, input_dim), dtype=data.dtype)
    Y_windows = np.empty((valid_count, target_dim), dtype=data.dtype)
    valid_indices = np.empty(valid_count, dtype=np.int64)
    idx = 0
    for i in range(n - time_window + 1):
        valid = True
        for j in range(target_dim):
            if np.isnan(data[i + time_window - 1, input_dim + j]):
                valid = False
                break
        if valid:
            X_windows[idx, :, :] = data[i:i + time_window, :input_dim]
            Y_windows[idx, :] = data[i + time_window - 1, input_dim:]
            valid_indices[idx] = i
            idx += 1
    return X_windows, Y_windows, valid_indices

def build_sliding_windows_for_subset_4(
    df: pd.DataFrame,
    comid_list: List[str],
    input_cols: Optional[List[str]] = None,
    target_cols: List[str] = ["TN"],
    time_window: int = 10
):
    """
    在 df 中，根据 comid_list 指定的河段进行滑窗切片。
      1. 先筛选 df["COMID"] 在 comid_list 中
      2. 若未指定 input_cols，使用除 COMID, date, target_cols 外的全部列作为输入
      3. 每个河段按时间升序，构造 (X, Y, COMID, Date) 切片
      4. X.shape = (N, time_window, len(input_cols))
         Y.shape = (N, len(target_cols))
         COMIDs.shape = (N,)
         Dates.shape = (N,)

    参数:
        df: 已包含 [COMID, date] 及相关特征列(如 flow, temperature_2m_mean 等)
        comid_list: 需要切片的 COMID 列表(字符串或整数均可，但需和 df["COMID"] 的 dtype 对应)
        input_cols: 用作时序输入的列。如果未指定，将自动选用 df 的所有列，排除 ["COMID", "date"] + target_cols
        target_cols: 目标列列表 (如 ["TN", "TP"])
        time_window: 滑窗大小 (默认10)
    """
    # 1. 先将 df 筛选到 comid_list
    sub_df = df[df["COMID"].isin(comid_list)].copy()
    
    # 2. 若未指定 input_cols，默认使用所有列，排除 ["COMID", "date"] + target_cols
    if input_cols is None:
        exclude_cols = {"COMID", "date"}.union(target_cols)
        input_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 3. 分组并做滑窗
    X_list, Y_list, comid_track, date_track = [], [], [], []
    
    for comid, group_df in sub_df.groupby("COMID"):
        group_df = group_df.sort_values("date").reset_index(drop=True)
        # 确保滑窗只包含 input_cols 和 target_cols
        needed_cols = input_cols + target_cols
        sub_data = group_df[needed_cols].values  # shape=(n_rows, len(needed_cols))
        
        for start_idx in range(len(sub_data) - time_window + 1):
            window_data = sub_data[start_idx : start_idx + time_window]
            
            # X 部分
            x_window = window_data[:, :len(input_cols)]  # 输入特征部分
            
            # Y 部分
            y_values = window_data[-1, len(input_cols):]  # 最后一天的目标列值
            if np.isnan(y_values).any():
                continue  # 跳过该滑窗
            
            # Date 部分
            date_value = group_df.loc[start_idx + time_window - 1, "date"]  # 滑窗最后一天的日期
            
            # 添加到结果列表
            X_list.append(x_window)
            Y_list.append(y_values)
            comid_track.append(comid)
            date_track.append(date_value)
    
    if not X_list:
        return None, None, None, None
    
    X_array = np.array(X_list, dtype=np.float32)  # (N, time_window, len(input_cols))
    Y_array = np.array(Y_list, dtype=np.float32)  # (N, len(target_cols))
    COMIDs = np.array(comid_track)                # (N,)  # 保持原类型或转成 str
    Dates = np.array(date_track)                  # (N,)  # 日期部分
    
    return X_array, Y_array, COMIDs, Dates



def build_sliding_windows_for_subset_5(
    df: pd.DataFrame,
    comid_list: List[str],
    input_cols: Optional[List[str]] = None,
    target_cols: List[str] = ["TN"],
    time_window: int = 10
):
    """
    构造滑动窗口数据切片（结合 tqdm 进度条与 Numba 加速，并整合了有效窗口索引的计算）
    输入：
        df: 包含日尺度数据的 DataFrame，必须包含 'COMID' 和 'date'
        comid_list: 要构造数据切片的 COMID 列表
        input_cols: 输入特征列名列表；若为 None，则除去 {"COMID", "date"} 与 target_cols 后的所有列
        target_cols: 目标变量列名列表
        time_window: 时间窗口长度
    输出：
        返回 (X_array, Y_array, COMIDs, Dates)
            X_array: 形状为 (N, time_window, input_dim) 的数组
            Y_array: 形状为 (N, len(target_cols)) 的数组，通常取时间窗口最后时刻的目标值
            COMIDs: 形状为 (N,) 的数组，每个切片对应的 COMID
            Dates: 形状为 (N,) 的数组，每个切片最后时刻的日期
    """
    sub_df = df[df["COMID"].isin(comid_list)].copy()
    if input_cols is None:
        exclude = {"COMID", "date"}.union(set(target_cols))
        input_cols = [col for col in df.columns if col not in exclude]
    X_list, Y_list, comid_track, date_track = [], [], [], []
    
    for comid, group_df in tqdm(sub_df.groupby("COMID"), desc="Processing groups (subset_4)"):
        group_df = group_df.sort_values("date").reset_index(drop=True)
        cols = input_cols + target_cols
        data_array = group_df[cols].values
        if data_array.shape[0] < time_window:
            print(f"警告：COMID {comid} 数据不足，跳过。")
            continue

        # 利用 numba 优化函数同时提取滑动窗口数据和对应的起始索引
        X_windows, Y_windows, valid_indices = extract_windows_with_indices_numba(data_array, time_window, len(input_cols))
        if valid_indices.size == 0:
            print(f"警告：COMID {comid} 无有效窗口，跳过。")
            continue

        for idx, start_idx in enumerate(valid_indices):
            X_list.append(X_windows[idx])
            Y_list.append(Y_windows[idx])
            comid_track.append(comid)
            # 直接用有效索引获取对应窗口最后一时刻的日期
            date_val = pd.to_datetime(group_df.loc[start_idx + time_window - 1, "date"])
            date_track.append(date_val)
            
    if not X_list:
        return None, None, None, None
    X_array = np.array(X_list, dtype=np.float32)
    Y_array = np.array(Y_list, dtype=np.float32)
    COMIDs = np.array(comid_track)
    Dates = np.array(date_track)
    return X_array, Y_array, COMIDs, Dates


# 为方便引用，提供别名，_4的意思是第四版,其他地方很多地方用到这个函数名称，遗留问题
build_sliding_windows_for_subset = build_sliding_windows_for_subset_3


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