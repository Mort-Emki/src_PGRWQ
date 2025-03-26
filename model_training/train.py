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
import logging 
from logging_utils import setup_logging, restore_stdout_stderr, ensure_dir_exists

# Import memory monitoring utilities
try:
    from gpu_memory_utils import log_memory_usage, TimingAndMemoryContext, MemoryTracker,force_cuda_memory_cleanup
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
                                 input_cols: list = None,
                                 start_iteration: int = 0,
                                 model_version: str = "v1",
                                 flow_results_dir: str = "flow_results",
                                 model_dir: str = "models"):
    """
    PG-RWQ 迭代训练过程
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
        start_iteration: 起始迭代轮数，0表示从头开始，>0表示从指定轮次开始
        model_version: 模型版本号，用于区分不同版本的模型
        flow_results_dir: 汇流结果保存目录
        model_dir: 模型保存目录
    输出：
        返回训练好的模型对象
    """
    #===========================================================================
    # 初始化与内存监控
    # - 启动内存跟踪器
    # - 记录初始内存状态
    # - 确保结果保存目录存在
    #===========================================================================
    # 启动内存跟踪
    memory_tracker = MemoryTracker(interval_seconds=10)
    memory_tracker.start()
    
    # 初始内存状态记录
    if device == 'cuda' and torch.cuda.is_available():
        log_memory_usage("[Training Start] ")
    
    # 创建结果保存目录
    output_dir = ensure_dir_exists(flow_results_dir)
    model_save_dir = ensure_dir_exists(model_dir)
    logging.info(f"Flow routing results will be saved to {output_dir}")
    logging.info(f"Models will be saved to {model_save_dir}")
    
    # 记录训练起始信息
    if start_iteration > 0:
        logging.info(f"Starting from iteration {start_iteration} with model version {model_version}")
        print(f"Starting from iteration {start_iteration} with model version {model_version}")
    else:
        logging.info(f"Starting from initial training (iteration 0) with model version {model_version}")
        print(f"Starting from initial training (iteration 0) with model version {model_version}")
        print('选择头部河段进行初始模型训练。')
    
    # 只有当start_iteration为0时才执行初始模型训练和汇流计算
    if start_iteration == 0:
        #===========================================================================
        # 阶段1: 数据准备
        # - 构建河段属性字典
        # - 选择头部河段
        # - 构造训练数据
        # - 标准化数据
        #===========================================================================
        # 构建河段属性字典
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

        # 检查COMID列表
        if comid_wq_list is None:
            comid_wq_list = []
        if comid_era5_list is None:
            comid_era5_list = []
        
        # 识别头部河段
        with TimingAndMemoryContext("Finding Head Stations"):
            comid_list_head = list(set(df_head_upstream['COMID'].unique().tolist()) & set(comid_wq_list) & set(comid_era5_list))
            if len(comid_list_head) == 0:
                print("警告：comid_wq_list、comid_era5_list 为空，请检查输入。")
                memory_tracker.stop()
                memory_tracker.report()
                return None
            print(f"  选择的头部河段数量：{len(comid_list_head)}")

        # 构造训练数据
        print('构造初始训练数据（滑窗切片）......')
        with TimingAndMemoryContext("Building Sliding Windows"):
            X_ts_head, Y_head_orig, COMIDs_head, Dates_head = build_sliding_windows_for_subset(
                df, 
                comid_list_head, 
                input_cols=None, 
                target_cols=target_cols, 
                time_window=10
            )
            Y_head = Y_head_orig[:,0]

        # 输出数据维度信息
        print("X_ts_all.shape =", X_ts_head.shape)
        print("Y.shape        =", Y_head.shape)
        print("COMID.shape    =", COMIDs_head.shape)  
        print("Date.shape     =", Dates_head.shape)

        # 保存训练数据
        with TimingAndMemoryContext("Saving Training Data"):
            np.savez(f"{output_dir}/upstreams_trainval_{model_version}.npz", 
                    X=X_ts_head, Y=Y_head_orig, COMID=COMIDs_head, Date=Dates_head)
            print("训练数据保存成功！")

        # 标准化数据
        with TimingAndMemoryContext("Data Standardization"):
            X_ts_head_scaled, ts_scaler = standardize_time_series_all(X_ts_head)
            attr_dict_scaled, attr_scaler = standardize_attributes(attr_dict)

            X_ts_head = X_ts_head_scaled
            attr_dict = attr_dict_scaled

            N, T, input_dim = X_ts_head.shape
            attr_dim = len(next(iter(attr_dict.values())))
            
            # 标准化后内存记录
            if device == 'cuda':
                log_memory_usage("[After Standardization] ")

        #===========================================================================
        # 阶段2: 训练/验证集划分
        # - 随机划分训练集和验证集
        # - 准备模型训练
        #===========================================================================
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
        
        #===========================================================================
        # 阶段3: 初始模型训练或加载
        # - 创建模型实例
        # - 训练新模型或加载已有模型
        #===========================================================================
        with TimingAndMemoryContext("Model Creation"):
            model = CatchmentModel(model_type=model_type,
                                input_dim=input_dim,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                attr_dim=attr_dim,
                                fc_dim=fc_dim,
                                device=device,
                                memory_check_interval=2)  # 每2个epoch检查一次内存

        # 训练或加载模型
        # 使用版本号构建模型路径
        model_path = f"{model_save_dir}/model_initial_A0_{model_version}.pth"
        if not os.path.exists(model_path):
            with TimingAndMemoryContext("Model Training"):
                model.train_model(attr_dict, comid_arr_train, X_ts_train, Y_train, 
                                comid_arr_val, X_ts_val, Y_val, 
                                epochs=100, lr=1e-3, patience=2, batch_size=10000)
            
            with TimingAndMemoryContext("Model Saving"):
                model.save_model(model_path)
                print(f"模型训练成功！保存至 {model_path}")
        else:
            with TimingAndMemoryContext("Model Loading"):
                model.load_model(model_path)
                print(f"模型加载成功：{model_path}")

        def batch_model_func(comid_batch, groups, attr_dict, model, target_cols):
            """
            批量处理河段预测函数（带自适应内存管理）
            """
            results = {}
            
            # 首先收集所有必要数据
            all_X_ts = []
            all_comids = []
            all_dates = []
            comid_indices = {}
            
            current_idx = 0
            valid_comids = []
            
            for comid in comid_batch:
                group = groups[comid]
                group_sorted = group.sort_values("date")
                
                X_ts_local, _, _, Dates_local = build_sliding_windows_for_subset(
                    df=group, 
                    comid_list=[comid], 
                    input_cols=None, 
                    target_cols=target_cols, 
                    time_window=10,
                    skip_missing_targets=False
                )
                
                if X_ts_local is None or X_ts_local.shape[0] == 0:
                    results[comid] = pd.Series(0.0, index=group_sorted["date"])
                    continue
                
                end_idx = current_idx + X_ts_local.shape[0]
                comid_indices[comid] = (current_idx, end_idx, Dates_local, group_sorted["date"])
                current_idx = end_idx
                valid_comids.append(comid)
                
                all_X_ts.append(X_ts_local)
                all_comids.extend([comid] * X_ts_local.shape[0])     
                all_dates.extend(Dates_local)
            
            if not all_X_ts:
                return {comid: pd.Series(0.0, index=groups[comid].sort_values("date")["date"]) 
                        for comid in comid_batch}
            
            with TimingAndMemoryContext("GPU Batch Processing"):
                # 堆叠所有数据
                X_ts_batch = np.vstack(all_X_ts)
                
                # 构建属性矩阵
                attr_dim = next(iter(attr_dict.values())).shape[0]
                X_attr_batch = np.zeros((X_ts_batch.shape[0], attr_dim), dtype=np.float32)
                
                for i, comid in enumerate(all_comids):
                    comid_str = str(comid)
                    attr_vec = attr_dict.get(comid_str, np.zeros(attr_dim, dtype=np.float32))
                    X_attr_batch[i] = attr_vec
                
                batch_size = X_ts_batch.shape[0]
                print(f"处理 {len(valid_comids)} 个河段，共 {batch_size} 个预测点")
                
                try:
                    # 该预测函数内置内存安全机制
                    all_preds = model.predict(X_ts_batch, X_attr_batch)
                    
                    # 清理资源
                    del X_ts_batch
                    del X_attr_batch
                    if torch.cuda.is_available():
                        force_cuda_memory_cleanup()
                        
                except Exception as e:
                    print(f"预测过程中出错: {e}")
                    print("尝试逐个处理河段...")
                    
                    # 降级策略：逐个处理河段
                    all_preds = np.zeros(batch_size)
                    for comid in valid_comids:
                        start_idx, end_idx, _, _ = comid_indices[comid]
                        comid_str = str(comid)
                        X_ts_subset = X_ts_batch[start_idx:end_idx]
                        X_attr_subset = np.tile(attr_dict.get(comid_str, np.zeros(attr_dim)), 
                                            (X_ts_subset.shape[0], 1))
                        
                        try:
                            preds_subset = model.predict(X_ts_subset, X_attr_subset)
                            all_preds[start_idx:end_idx] = preds_subset
                        except Exception as e2:
                            print(f"处理河段 {comid} 失败: {e2}")
                            all_preds[start_idx:end_idx] = 0.0
                        
                        # 每处理完一个河段都清理资源
                        if torch.cuda.is_available():
                            force_cuda_memory_cleanup()
            
            # 将预测结果映射回河段
            for comid in valid_comids:
                start_idx, end_idx, dates, all_dates = comid_indices[comid]
                preds = all_preds[start_idx:end_idx]
                
                pred_series = pd.Series(preds, index=pd.to_datetime(dates))
                full_series = pd.Series(0.0, index=all_dates)
                full_series.update(pred_series)
                
                results[comid] = full_series
            
            for comid in comid_batch:
                if comid not in valid_comids and comid not in results:
                    results[comid] = pd.Series(0.0, index=groups[comid].sort_values("date")["date"])
            
            return results
        
        #===========================================================================
        # 阶段4: 初始汇流计算
        # - 使用初始模型A₀进行预测
        # - 执行汇流计算
        # - 保存结果到CSV
        #===========================================================================
        print("初始汇流计算：使用 A₀ 进行预测。")
        
        with TimingAndMemoryContext("Flow Routing Calculation"):
            # 初始迭代使用E_save=1来保存E值
            df_flow = flow_routing_calculation(df = df.copy(), 
                                              iteration=0, 
                                              model_func=batch_model_func, 
                                              river_info=river_info, 
                                              v_f_TN=35.0,
                                              v_f_TP=44.5,
                                              attr_dict=attr_dict,
                                              model=model,
                                              target_cols=target_cols,
                                              attr_df=attr_df,
                                              E_save=1,  # 保存初始E值
                                              E_save_path=f"{output_dir}/E_values_{model_version}")
            
            # 保存初始汇流计算结果
            initial_result_path = os.path.join(output_dir, f"flow_routing_iteration_0_{model_version}.csv")
            df_flow.to_csv(initial_result_path, index=False)
            logging.info(f"初始汇流计算结果已保存至 {initial_result_path}")
            print(f"初始汇流计算结果已保存至 {initial_result_path}")
    else:
        # 如果start_iteration > 0，跳过初始模型训练和初始汇流计算
        # 创建模型实例
        with TimingAndMemoryContext("Model Creation for Continued Training"):
            model = CatchmentModel(model_type=model_type,
                                input_dim=input_dim,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                attr_dim=attr_dim,
                                fc_dim=fc_dim,
                                device=device,
                                memory_check_interval=2)
        
        # 加载最后一轮迭代的模型
        last_iteration = start_iteration - 1
        model_path = f"{model_save_dir}/model_A{last_iteration}_{model_version}.pth"
        if not os.path.exists(model_path):
            logging.error(f"无法找到上一轮迭代的模型: {model_path}")
            print(f"错误：无法找到上一轮迭代的模型 {model_path}")
            memory_tracker.stop()
            memory_tracker.report()
            return None
        
        with TimingAndMemoryContext("Loading Previous Iteration Model"):
            model.load_model(model_path)
            print(f"已加载上一轮迭代模型: {model_path}")
        
        # 加载上一轮的汇流计算结果
        previous_flow_path = os.path.join(output_dir, f"flow_routing_iteration_{last_iteration}_{model_version}.csv")
        if not os.path.exists(previous_flow_path):
            logging.error(f"无法找到上一轮汇流计算结果: {previous_flow_path}")
            print(f"错误：无法找到上一轮汇流计算结果 {previous_flow_path}")
            memory_tracker.stop()
            memory_tracker.report()
            return None
        
        with TimingAndMemoryContext("Loading Previous Flow Routing Results"):
            df_flow = pd.read_csv(previous_flow_path)
            print(f"已加载上一轮汇流计算结果: {previous_flow_path}")
        
        # 构建河段属性字典（仍然需要）
        with TimingAndMemoryContext("Building Attribute Dictionary"):
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
            
            # 标准化属性
            attr_dict_scaled, attr_scaler = standardize_attributes(attr_dict)
            attr_dict = attr_dict_scaled
    
    #===========================================================================
    # 阶段5: 迭代训练与汇流计算
    # - 计算残差并检查收敛性
    # - 构建更新训练数据
    # - 训练更新模型
    # - 执行新一轮汇流计算
    # - 保存每轮迭代结果和模型
    #===========================================================================
    # 从start_iteration开始迭代
    for it in range(start_iteration, max_iterations):
        with TimingAndMemoryContext(f"Iteration {it+1}/{max_iterations}"):
            print(f"\n迭代 {it+1}/{max_iterations}")
            col_y_n = f'y_n_{it}'    
            col_y_up = f'y_up_{it}'
            
            # 每轮迭代开始时检查内存
            if device == 'cuda':
                log_memory_usage(f"[Iteration {it+1} Start] ")
            
            # 计算残差并检查收敛性
            merged = pd.merge(df, df_flow[['COMID', 'Date', col_y_n, col_y_up]], on=['COMID', 'Date'], how='left')
            y_true = merged[target_col].values
            y_pred = merged[col_y_n].values
            residual = y_true - y_pred
            max_resid = np.abs(residual).max()
            print(f"  最大残差: {max_resid:.4f}")
            
            if max_resid < epsilon:
                print("收敛！") 
                break
                
            # 为下一轮训练准备数据
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
            
            # 训练更新模型或加载已有模型
            model_path = f"{model_save_dir}/model_A{it+1}_{model_version}.pth"
            if not os.path.exists(model_path):
                # 训练更新模型
                with TimingAndMemoryContext(f"Model Training for Iteration {it+1}"):
                    model.train_model(X_ts_iter, X_attr_iter, Y_label_iter, epochs=5, lr=1e-3, patience=2, batch_size=32)
                
                # 保存本轮迭代的模型
                with TimingAndMemoryContext(f"Saving Model for Iteration {it+1}"):
                    model.save_model(model_path)
                    print(f"模型已保存至: {model_path}")
            else:
                # 加载已有模型
                with TimingAndMemoryContext(f"Loading Existing Model for Iteration {it+1}"):
                    model.load_model(model_path)
                    print(f"已加载现有模型: {model_path}")
            
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
                
                # 大规模预测的内存检查
                if device == 'cuda' and X_ts_local.shape[0] > 100:
                    log_memory_usage(f"[Updated Prediction for COMID {comid_str}] ")
                    
                preds = model.predict(X_ts_local, X_attr_local)
                return pd.Series(preds, index=pd.to_datetime(Dates_local))
            
            # 执行新一轮汇流计算
            with TimingAndMemoryContext(f"Flow Routing for Iteration {it+1}"):
                df_flow = flow_routing_calculation(df.copy(), iteration=it+1, model_func=updated_model_func, 
                                                river_info=river_info, v_f_TN=35.0, v_f_TP=44.5,
                                                target_cols=target_cols,
                                                E_save=1,  # 保存E值
                                                E_save_path=f"{output_dir}/E_values_{model_version}")
                
                # 保存当前迭代结果
                iter_result_path = os.path.join(output_dir, f"flow_routing_iteration_{it+1}_{model_version}.csv")
                df_flow.to_csv(iter_result_path, index=False)
                logging.info(f"迭代 {it+1} 汇流计算结果已保存至 {iter_result_path}")
                print(f"迭代 {it+1} 汇流计算结果已保存至 {iter_result_path}")
    
    #===========================================================================
    # 阶段6: 完成与清理
    # - 停止内存跟踪
    # - 输出内存使用报告
    # - 返回训练好的模型
    #===========================================================================
    # 生成内存报告
    memory_tracker.stop()
    memory_stats = memory_tracker.report()
    
    if device == 'cuda':
        log_memory_usage("[Training Complete] ")
    
    # 保存最终模型
    final_iter = min(it+1, max_iterations)
    final_model_path = os.path.join(model_save_dir, f"final_model_iteration_{final_iter}_{model_version}.pth")
    model.save_model(final_model_path)
    logging.info(f"最终模型已保存至 {final_model_path}")
    print(f"最终模型已保存至 {final_model_path}")
    
    return model