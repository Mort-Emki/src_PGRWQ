import numpy as np
import pandas as pd
from typing import List
from data_processing import build_sliding_windows_for_subset, standardize_time_series_all, standardize_attributes
from model_training.models import CatchmentModel
from flow_routing import flow_routing_calculation ##目录层次问题
from tqdm import tqdm
import numba
import time
import os
import torch

# Import memory monitoring utilities
try:
    from gpu_memory_utils import log_memory_usage, TimingAndMemoryContext, MemoryTracker
except ImportError:
    # Fallback implementation if the module is not available
    def log_memory_usage(prefix=""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            print(f"{prefix}GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
    
    class TimingAndMemoryContext:
        def __init__(self, name="Operation", log_memory=True):
            self.name = name
            self.log_memory = log_memory
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            if self.log_memory and torch.cuda.is_available():
                log_memory_usage(f"[{self.name} START] ")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            if self.log_memory and torch.cuda.is_available():
                log_memory_usage(f"[{self.name} END] ")
            print(f"[TIMING] {self.name} completed in {duration:.2f} seconds")
    
    class MemoryTracker:
        def __init__(self, interval_seconds=5):
            self.interval = interval_seconds
            self.tracking = False
        
        def start(self):
            print("Memory tracking started (simplified version)")
            self.tracking = True
        
        def stop(self):
            self.tracking = False
            print("Memory tracking stopped")
        
        def report(self):
            if torch.cuda.is_available():
                log_memory_usage("[Final Memory Report] ")
            return {}


def iterative_training_procedure(df: pd.DataFrame,
                                 attr_df: pd.DataFrame,
                                 input_features: List[str] = None,
                                 attr_features: List[str] = None,
                                 river_info: pd.DataFrame = None,
                                 target_cols: List[str] = ["TN","TP"],
                                 target_col: str = "TN",
                                 max_iterations: int = 10,
                                 epsilon: float = 0.01,
                                 model_type: str = 'rf',
                                 input_dim: int = None,
                                 hidden_size: int = 64,
                                 num_layers: int = 1,
                                 attr_dim: int = None,
                                 fc_dim: int = 32,
                                 device: str = 'cuda',
                                 comid_wq_list: list = None,
                                 comid_era5_list: list = None,
                                 input_cols: list = None):
    """
    迭代训练过程
    输入：
        df: 日尺度数据 DataFrame，包含 'COMID'、'Date'、target_col、'Qout' 等字段
        attr_dict: 河段属性字典，键为 str(COMID)，值为属性数组（已标准化）
        river_info: 河段信息 DataFrame，包含 'COMID' 和 'NextDownID'
        target_col: 目标变量名称，如 "TP"
        max_iterations: 最大迭代次数
        epsilon: 收敛阈值（残差最大值）
        model_type: 'rf' 或 'lstm'
        input_dim: 模型输入维度（须与 input_cols 长度一致）
        hidden_size, num_layers, attr_dim, fc_dim: 模型参数
        device: 训练设备
        input_cols: 指定用于构造时间序列输入的特征列表（例如，["Feature1", "Feature2", ...]）
    输出：
        返回训练好的模型对象
    """
    # Start memory tracking
    memory_tracker = MemoryTracker(interval_seconds=10)
    memory_tracker.start()
    
    # Initial memory status
    if device == 'cuda' and torch.cuda.is_available():
        log_memory_usage("[Training Start] ")
    
    print('选择头部河段进行初始模型训练。')
    
    # Create a context manager to time and monitor memory for building the attribute dictionary
    with TimingAndMemoryContext("Building Attribute Dictionary"):
        attr_df_head_upstream = attr_df[attr_df['order_'] <= 2]
        df_head_upstream = df[df['COMID'].isin(attr_df_head_upstream['COMID'])]
        
        # 将河段属性数据转换为字典
        attr_dict = {}
        for row in attr_df.itertuples(index=False):
            comid = str(row.COMID)
            row_dict = row._asdict()
            attrs = []
            for attr in attr_features:
                if attr in row_dict:
                    attrs.append(row_dict[attr])
                else:
                    attrs.append(0.0)
            attr_dict[comid] = np.array(attrs, dtype=np.float32)

    # Check for comid lists
    if comid_wq_list is None:
        comid_wq_list = []
    if comid_era5_list is None:
        comid_era5_list = []
    
    with TimingAndMemoryContext("Finding Head Stations"):
        comid_list_head = list(set(df_head_upstream['COMID'].unique().tolist()) & set(comid_wq_list) & set(comid_era5_list))
        if len(comid_list_head) == 0:
            print("警告：comid_wq_list、comid_era5_list 为空，请检查输入。")
            memory_tracker.stop()
            memory_tracker.report()
            return None
        print(f"  选择的头部河段数量：{ len(comid_list_head)}")

    print('构造初始训练数据（滑窗切片）......')
    
    # Build sliding windows with memory monitoring
    with TimingAndMemoryContext("Building Sliding Windows"):
        X_ts_head, Y_head_orig, COMIDs_head, Dates_head = build_sliding_windows_for_subset(
            df, 
            comid_list_head, 
            input_cols=None, 
            target_cols=target_cols, 
            time_window=10
        )
        Y_head = Y_head_orig[:,0]

    print("X_ts_all.shape =", X_ts_head.shape)
    print("Y.shape        =", Y_head.shape)
    print("COMID.shape    =", COMIDs_head.shape)  
    print("Date.shape     =", Dates_head.shape)

    # Save to npz file
    with TimingAndMemoryContext("Saving Training Data"):
        np.savez("upstreams_trainval_mainsrc.npz", X=X_ts_head, Y=Y_head_orig, COMID=COMIDs_head, Date=Dates_head)
        print("训练数据保存成功！")

    # Standardize data
    with TimingAndMemoryContext("Data Standardization"):
        X_ts_head_scaled, ts_scaler = standardize_time_series_all(X_ts_head)
        attr_dict_scaled, attr_scaler = standardize_attributes(attr_dict)

        X_ts_head = X_ts_head_scaled
        attr_dict = attr_dict_scaled

        N, T, input_dim = X_ts_head.shape
        attr_dim = len(next(iter(attr_dict.values())))
        
        # Log memory after standardization
        if device == 'cuda':
            log_memory_usage("[After Standardization] ")

    # Split into train and validation sets
    with TimingAndMemoryContext("Train/Validation Split"):
        N = len(X_ts_head)
        indices = np.random.permutation(N)
        train_size = int(N * 0.8)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]

        X_ts_train = X_ts_head[train_indices]
        comid_arr_train = COMIDs_head[train_indices]
        Y_train = Y_head[train_indices]

        X_ts_val = X_ts_head[valid_indices]
        comid_arr_val = COMIDs_head[valid_indices]
        Y_val = Y_head[valid_indices]

    print("初始模型 A₀ 训练：头部河段训练数据构造完毕。")
    
    # Create model with memory monitoring
    with TimingAndMemoryContext("Model Creation"):
        model = CatchmentModel(model_type=model_type,
                               input_dim=input_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               attr_dim=attr_dim,
                               fc_dim=fc_dim,
                               device=device,
                               memory_check_interval=2)  # Check memory every 2 epochs

    # Train or load model
    model_path = "model_initial_A0_0320_new.pth"
    if not os.path.exists(model_path):
        with TimingAndMemoryContext("Model Training"):
            model.train_model(attr_dict, comid_arr_train, X_ts_train, Y_train, 
                             comid_arr_val, X_ts_val, Y_val, 
                             epochs=100, lr=1e-3, patience=2, batch_size=10000)
        
        with TimingAndMemoryContext("Model Saving"):
            model.save_model(model_path)
            print("模型训练成功！")
    else:
        with TimingAndMemoryContext("Model Loading"):
            model.load_model(model_path)
            print("模型加载成功！")

    def batch_model_func(comid_batch, groups, attr_dict, model, target_cols):
        """
        批量处理多个COMID河段的预测函数
        
        参数:
            comid_batch: 需要处理的COMID列表
            groups: 按COMID分组的数据字典
            attr_dict: 属性字典
            model: 模型对象
            target_cols: 目标列名列表
            
        返回:
            字典，键为COMID，值为预测的Series
        """
        results = {}
        
        # 收集所有COMID的数据
        all_X_ts = []
        all_comids = []
        all_dates = []
        comid_indices = {}
        
        current_idx = 0
        
        # 为每个COMID准备数据
        for comid in comid_batch:
            group = groups[comid]
            group_sorted = group.sort_values("date")
            
            # 保持与原始model_func相同的数据处理逻辑
            X_ts_local, _, _, Dates_local = build_sliding_windows_for_subset(
                df=group, 
                comid_list=[comid], 
                input_cols=None, 
                target_cols=target_cols, 
                time_window=10,
                skip_missing_targets=False
            )
            
            if X_ts_local is None or X_ts_local.shape[0] == 0:
                # 如果没有有效数据，返回全0的Series
                results[comid] = pd.Series(0.0, index=group_sorted["date"])
                continue
            
            # 记录当前COMID数据的索引范围
            end_idx = current_idx + X_ts_local.shape[0]
            comid_indices[comid] = (current_idx, end_idx, Dates_local, group_sorted["date"])
            current_idx = end_idx
            
            # 添加到批处理数组
            all_X_ts.append(X_ts_local)
            all_comids.extend([comid] * X_ts_local.shape[0])
            all_dates.extend(Dates_local)
        
        # 如果批次中没有有效数据，直接返回空结果
        if not all_X_ts:
            return {comid: pd.Series(0.0, index=groups[comid].sort_values("date")["date"]) 
                    for comid in comid_batch}
        
        # 合并所有数据并构建属性矩阵
        X_ts_batch = np.vstack(all_X_ts)
        X_attr_batch = np.zeros((X_ts_batch.shape[0], next(iter(attr_dict.values())).shape[0]), dtype=np.float32)
        
        # 为每个样本设置正确的属性向量
        for i, comid in enumerate(all_comids):
            comid_str = str(comid)
            attr_vec = attr_dict.get(comid_str, np.zeros_like(next(iter(attr_dict.values()))))
            X_attr_batch[i] = attr_vec
        
        # 批量进行预测
        all_preds = model.predict(X_ts_batch, X_attr_batch)
        
        # 将预测结果映射回各个COMID
        for comid in comid_batch:
            if comid not in comid_indices:
                # 如果COMID无有效数据，返回全0
                results[comid] = pd.Series(0.0, index=groups[comid].sort_values("date")["date"])
                continue
                
            start_idx, end_idx, dates, all_dates = comid_indices[comid]
            preds = all_preds[start_idx:end_idx]
            
            # 创建预测Series
            pred_series = pd.Series(preds, index=pd.to_datetime(dates))
            full_series = pd.Series(0.0, index=all_dates)
            full_series.update(pred_series)
            
            results[comid] = full_series
        
        return results 
    
    def initial_model_func(group: pd.DataFrame, attr_dict: dict, model: CatchmentModel):
        with TimingAndMemoryContext("Model Prediction Function", log_memory=False):
            group_sorted = group.sort_values("date")
            X_ts_local, _, _, Dates_local = build_sliding_windows_for_subset(
                df=group, 
                comid_list=[group.iloc[0]['COMID']], 
                input_cols=None, 
                target_cols=target_cols, 
                time_window=10,
                skip_missing_targets=False
            )
            
            if X_ts_local is None:
                print(f"警告：COMID {group.iloc[0]['COMID']} 数据不足，返回 0。")
                return pd.Series(0.0, index=group_sorted["date"])
            
            comid_str = str(group.iloc[0]['COMID'])
            attr_vec = attr_dict.get(comid_str, np.zeros_like(next(iter(attr_dict.values()))))
            X_attr_local = np.tile(attr_vec, (X_ts_local.shape[0], 1))
            
            # Periodic memory check for large predictions
            if device == 'cuda' and X_ts_local.shape[0] > 100:
                log_memory_usage(f"[Prediction for COMID {comid_str}] ")
                
            preds = model.predict(X_ts_local, X_attr_local)
            
            # Create prediction series
            pred_series = pd.Series(preds, index=pd.to_datetime(Dates_local))
            full_series = pd.Series(0.0, index=group_sorted["date"])
            full_series.update(pred_series)
            
            return full_series   
    
    print("初始汇流计算：使用 A₀ 进行预测。")
    
    with TimingAndMemoryContext("Flow Routing Calculation"):
        df_flow = flow_routing_calculation(df = df.copy(), 
                                          iteration=0, 
                                          model_func=batch_model_func, 
                                          river_info=river_info, 
                                          v_f=35.0,
                                          attr_dict=attr_dict,
                                          model=model)
    
    # 迭代更新过程
    for it in range(max_iterations):
        with TimingAndMemoryContext(f"Iteration {it+1}/{max_iterations}"):
            print(f"\n迭代 {it+1}/{max_iterations}")
            col_y_n = f'y_n_{it}'    
            col_y_up = f'y_up_{it}'
            
            # Check memory at iteration start
            if device == 'cuda':
                log_memory_usage(f"[Iteration {it+1} Start] ")
            
            merged = pd.merge(df, df_flow[['COMID', 'Date', col_y_n, col_y_up]], on=['COMID', 'Date'], how='left')
            y_true = merged[target_col].values
            y_pred = merged[col_y_n].values
            residual = y_true - y_pred
            max_resid = np.abs(residual).max()
            print(f"  最大残差: {max_resid:.4f}")
            
            if max_resid < epsilon:
                print("收敛！")
                break
                
            merged["E_label"] = merged[target_col] - merged[col_y_up]
            comid_list_iter = merged["COMID"].unique().tolist()
            
            with TimingAndMemoryContext(f"Building Sliding Windows for Iteration {it+1}"):
                X_ts_iter, _, COMIDs_iter, Dates_iter = build_sliding_windows_for_subset(
                    df, comid_list_iter, input_cols=input_cols, target_cols=[target_col], time_window=5
                )
                
            Y_label_iter = []
            for cid, date_val in zip(COMIDs_iter, Dates_iter):
                subset = merged[(merged["COMID"] == cid) & (merged["Date"] == date_val)]
                if not subset.empty:
                    label_val = subset["E_label"].mean()
                else:
                    label_val = 0.0
                Y_label_iter.append(label_val)
                
            Y_label_iter = np.array(Y_label_iter, dtype=np.float32)
            X_attr_iter = np.vstack([attr_dict.get(str(cid), np.zeros_like(next(iter(attr_dict.values()))))
                                    for cid in COMIDs_iter])
            
            print("  更新模型训练：使用更新后的 E_label。")
            
            with TimingAndMemoryContext(f"Model Training for Iteration {it+1}"):
                model.train_model(X_ts_iter, X_attr_iter, Y_label_iter, epochs=5, lr=1e-3, patience=2, batch_size=32)
            
            # 更新后的模型函数：利用真实切片数据和属性数据进行预测
            def updated_model_func(group: pd.DataFrame):
                group_sorted = group.sort_values("Date")
                X_ts_local, _, _, Dates_local = build_sliding_windows_for_subset(
                    group, [group.iloc[0]['COMID']], input_cols=input_cols, target_cols=[target_col], time_window=5
                )
                if X_ts_local is None:
                    print(f"警告：COMID {group.iloc[0]['COMID']} 数据不足，返回 0。")
                    return pd.Series(0.0, index=group_sorted["Date"])
                    
                comid_str = str(group.iloc[0]['COMID'])
                attr_vec = attr_dict.get(comid_str, np.zeros_like(next(iter(attr_dict.values()))))
                X_attr_local = np.tile(attr_vec, (X_ts_local.shape[0], 1))
                
                # Memory check for large predictions
                if device == 'cuda' and X_ts_local.shape[0] > 100:
                    log_memory_usage(f"[Updated Prediction for COMID {comid_str}] ")
                    
                preds = model.predict(X_ts_local, X_attr_local)
                return pd.Series(preds, index=pd.to_datetime(Dates_local))
            
            with TimingAndMemoryContext(f"Flow Routing for Iteration {it+1}"):
                df_flow = flow_routing_calculation(df.copy(), iteration=it+1, model_func=updated_model_func, 
                                                 river_info=river_info, v_f=35.0)
    
    # Final memory report
    memory_tracker.stop()
    memory_stats = memory_tracker.report()
    
    if device == 'cuda':
        log_memory_usage("[Training Complete] ")
    
    return model