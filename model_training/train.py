import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import time
import os
import torch
import logging
from tqdm import tqdm

# 引入相关模块
from PGRWQI.flow_routing import flow_routing_calculation 
from PGRWQI.data_processing import build_sliding_windows_for_subset, standardize_time_series_all, standardize_attributes
from PGRWQI.model_training.models import CatchmentModel 
from PGRWQI.logging_utils import setup_logging, restore_stdout_stderr, ensure_dir_exists
from PGRWQI.model_training.gpu_memory_utils import (
    log_memory_usage, 
    TimingAndMemoryContext, 
    MemoryTracker,
    force_cuda_memory_cleanup
)

# ===============================================================================
# 辅助函数
# ===============================================================================

def check_existing_flow_routing_results(
    iteration: int, 
    model_version: str, 
    flow_results_dir: str
) -> Tuple[bool, str]:
    """
    检查是否已存在特定迭代和模型版本的汇流计算结果文件
    
    参数:
        iteration: 迭代次数
        model_version: 模型版本号
        flow_results_dir: 汇流结果保存目录
        
    返回:
        (exists, file_path): 元组，包含是否存在的布尔值和文件路径
    """
    file_path = os.path.join(flow_results_dir, f"flow_routing_iteration_{iteration}_{model_version}.csv")
    exists = os.path.isfile(file_path)
    return exists, file_path

# ===============================================================================
# 数据准备相关函数
# ===============================================================================

def build_attribute_dictionary(
    attr_df: pd.DataFrame, 
    attr_features: List[str]
) -> Dict[str, np.ndarray]:
    """
    构建河段属性字典
    
    参数:
        attr_df: 包含属性数据的DataFrame
        attr_features: 要提取的属性特征列表
        
    返回:
        属性字典，键为COMID字符串，值为属性特征数组
    """
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
    return attr_dict

def prepare_training_data_for_head_segments(
    df: pd.DataFrame,
    attr_df: pd.DataFrame,
    comid_wq_list: List,
    comid_era5_list: List,
    target_cols: List[str],
    output_dir: str,
    model_version: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    为头部河段准备训练数据
    
    参数:
        df: 包含日尺度数据的DataFrame
        attr_df: 包含属性数据的DataFrame
        comid_wq_list: 水质站点COMID列表
        comid_era5_list: ERA5数据覆盖的COMID列表
        target_cols: 目标列列表
        output_dir: 输出目录
        model_version: 模型版本号
        
    返回:
        (X_ts_head, Y_head_orig, COMIDs_head, Dates_head): 训练数据元组
    """
    # 筛选头部河段
    with TimingAndMemoryContext("Finding Head Stations"):
        attr_df_head_upstream = attr_df[attr_df['order_'] <= 2]
        df_head_upstream = df[df['COMID'].isin(attr_df_head_upstream['COMID'])]
        
        comid_list_head = list(set(df_head_upstream['COMID'].unique().tolist()) 
                             & set(comid_wq_list) 
                             & set(comid_era5_list))
        
        if len(comid_list_head) == 0:
            print("警告：comid_wq_list、comid_era5_list 为空，请检查输入。")
            return None, None, None, None
        
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

    # 输出数据维度信息
    print("X_ts_all.shape =", X_ts_head.shape)
    print("Y.shape        =", Y_head_orig.shape)
    print("COMID.shape    =", COMIDs_head.shape)  
    print("Date.shape     =", Dates_head.shape)

    # 保存训练数据
    with TimingAndMemoryContext("Saving Training Data"):
        np.savez(f"{output_dir}/upstreams_trainval_{model_version}.npz", 
                X=X_ts_head, Y=Y_head_orig, COMID=COMIDs_head, Date=Dates_head)
        print("训练数据保存成功！")
        
    return X_ts_head, Y_head_orig, COMIDs_head, Dates_head

def standardize_data(
    X_ts_head: np.ndarray, 
    attr_dict: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Any, Any]:
    """
    标准化时间序列数据和属性数据
    
    参数:
        X_ts_head: 时间序列数据
        attr_dict: 属性字典
        
    返回:
        (X_ts_head_scaled, attr_dict_scaled, ts_scaler, attr_scaler): 标准化后的数据和标量器
    """
    with TimingAndMemoryContext("Data Standardization"):
        X_ts_head_scaled, ts_scaler = standardize_time_series_all(X_ts_head)
        attr_dict_scaled, attr_scaler = standardize_attributes(attr_dict)
        
        # 标准化后内存记录
        if torch.cuda.is_available():
            log_memory_usage("[After Standardization] ")
            
    return X_ts_head_scaled, attr_dict_scaled, ts_scaler, attr_scaler

def split_train_val_data(
    X_ts_head: np.ndarray, 
    Y_head: np.ndarray, 
    COMIDs_head: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将数据划分为训练集和验证集
    
    参数:
        X_ts_head: 时间序列数据
        Y_head: 目标变量
        COMIDs_head: COMID数组
        
    返回:
        (X_ts_train, comid_arr_train, Y_train, X_ts_val, comid_arr_val, Y_val): 划分后的数据
    """
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

    return X_ts_train, comid_arr_train, Y_train, X_ts_val, comid_arr_val, Y_val

# ===============================================================================
# 模型创建和批量预测函数
# ===============================================================================

def create_or_load_model(
    model_type: str,
    input_dim: int,
    hidden_size: int,
    num_layers: int,
    attr_dim: int,
    fc_dim: int,
    device: str,
    model_path: str,
    attr_dict: Dict[str, np.ndarray] = None,
    train_data: Optional[Tuple] = None,
    context_name: str = "Model Operation"
) -> CatchmentModel:
    """
    创建新模型或加载已有模型
    
    参数:
        model_type: 模型类型
        input_dim: 输入维度
        hidden_size: 隐藏层大小
        num_layers: 层数
        attr_dim: 属性维度
        fc_dim: 全连接层维度
        device: 设备
        model_path: 模型路径
        attr_dict: 属性字典(仅训练时需要)
        train_data: 训练数据元组(仅训练时需要)
        context_name: 操作上下文名称
        
    返回:
        创建或加载的模型
    """
    with TimingAndMemoryContext(context_name):
        model = CatchmentModel(
            model_type=model_type,
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attr_dim=attr_dim,
            fc_dim=fc_dim,
            device=device,
            memory_check_interval=2  # 每2个epoch检查一次内存
        )
    
    # 检查是否存在预训练模型
    if os.path.exists(model_path):
        with TimingAndMemoryContext("Model Loading"):
            model.load_model(model_path)
            print(f"模型加载成功：{model_path}")
    elif train_data is not None:
        # 如果没有预训练模型，且提供了训练数据，则训练新模型
        X_ts_train, comid_arr_train, Y_train, X_ts_val, comid_arr_val, Y_val = train_data
        with TimingAndMemoryContext("Model Training"):
            model.train_model(
                attr_dict, comid_arr_train, X_ts_train, Y_train, 
                comid_arr_val, X_ts_val, Y_val, 
                epochs=100, lr=1e-3, patience=2, batch_size=10000
            )
        
        with TimingAndMemoryContext("Model Saving"):
            model.save_model(model_path)
            print(f"模型训练成功！保存至 {model_path}")
    
    return model

def create_batch_model_func(
    df: pd.DataFrame,
    attr_dict: Dict[str, np.ndarray],
    model: CatchmentModel,
    target_cols: List[str]
) -> Callable:
    """
    创建批量模型预测函数
    
    参数:
        df: 包含数据的DataFrame
        attr_dict: 属性字典
        model: 预测模型
        target_cols: 目标列列表
        
    返回:
        批量预测函数
    """
    def batch_model_func(comid_batch, groups, attr_dict_local, model_local, target_cols_local):
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
                target_cols=target_cols_local, 
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
            attr_dim = next(iter(attr_dict_local.values())).shape[0]
            X_attr_batch = np.zeros((X_ts_batch.shape[0], attr_dim), dtype=np.float32)
            
            for i, comid in enumerate(all_comids):
                comid_str = str(comid)
                attr_vec = attr_dict_local.get(comid_str, np.zeros(attr_dim, dtype=np.float32))
                X_attr_batch[i] = attr_vec
            
            batch_size = X_ts_batch.shape[0]
            print(f"处理 {len(valid_comids)} 个河段，共 {batch_size} 个预测点")
            
            try:
                # 该预测函数内置内存安全机制
                all_preds = model_local.predict(X_ts_batch, X_attr_batch)
                
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
                    X_attr_subset = np.tile(attr_dict_local.get(comid_str, np.zeros(attr_dim)), 
                                        (X_ts_subset.shape[0], 1))
                    
                    try:
                        preds_subset = model_local.predict(X_ts_subset, X_attr_subset)
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
    
    return batch_model_func

def create_updated_model_func(
    df: pd.DataFrame,
    attr_dict: Dict[str, np.ndarray],
    model: CatchmentModel,
    target_col: str,
    input_cols: List[str],
    device: str
) -> Callable:
    """
    创建更新的模型预测函数
    
    参数:
        df: 包含数据的DataFrame
        attr_dict: 属性字典
        model: 预测模型
        target_col: 目标列
        input_cols: 输入列列表
        device: 设备
        
    返回:
        更新的预测函数
    """
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
    
    return updated_model_func

# ===============================================================================
# 流程路由计算和相关函数
# ===============================================================================

def perform_flow_routing_calculation(
    df: pd.DataFrame,
    iteration: int,
    model_func: Callable,
    river_info: pd.DataFrame,
    attr_dict: Dict[str, np.ndarray],
    model: CatchmentModel,
    target_cols: List[str],
    attr_df: pd.DataFrame,
    output_dir: str,
    model_version: str,
    exists: bool = False,
    flow_result_path: str = None,
    reuse_existing_flow_results: bool = True
) -> pd.DataFrame:
    """
    执行汇流计算或加载已有结果
    
    参数:
        df: 包含数据的DataFrame
        iteration: 迭代次数
        model_func: 模型预测函数
        river_info: 河网信息
        attr_dict: 属性字典
        model: 预测模型
        target_cols: 目标列列表
        attr_df: 属性DataFrame
        output_dir: 输出目录
        model_version: 模型版本号
        exists: 是否存在已有结果
        flow_result_path: 已有结果路径
        reuse_existing_flow_results: 是否重用已有结果
        
    返回:
        汇流计算结果DataFrame
    """
    if exists and reuse_existing_flow_results:
        # 如果存在且配置为重用，直接加载已有结果
        with TimingAndMemoryContext("Loading Existing Flow Routing Results"):
            print(f"发现已存在的汇流计算结果，加载：{flow_result_path}")
            logging.info(f"Loading existing flow routing results from {flow_result_path}")
            df_flow = pd.read_csv(flow_result_path)
            print(f"成功加载汇流计算结果，共 {len(df_flow)} 条记录")
    else:
        # 如果不存在或配置为不重用，执行汇流计算
        with TimingAndMemoryContext("Flow Routing Calculation"):
            # 初始迭代使用E_save=1来保存E值
            df_flow = flow_routing_calculation(
                df=df.copy(), 
                iteration=iteration, 
                model_func=model_func, 
                river_info=river_info, 
                v_f_TN=35.0,
                v_f_TP=44.5,
                attr_dict=attr_dict,
                model=model,
                target_cols=target_cols,
                attr_df=attr_df,
                E_save=1,  # 保存E值
                E_save_path=f"{output_dir}/E_values_{model_version}"
            )
            
            # 保存汇流计算结果
            result_path = os.path.join(output_dir, f"flow_routing_iteration_{iteration}_{model_version}.csv")
            df_flow.to_csv(result_path, index=False)
            logging.info(f"迭代 {iteration} 汇流计算结果已保存至 {result_path}")
            print(f"迭代 {iteration} 汇流计算结果已保存至 {result_path}")
    
    return df_flow

def prepare_next_iteration_data(
    df: pd.DataFrame,
    df_flow: pd.DataFrame,
    target_col: str,
    col_y_n: str,
    col_y_up: str,
    input_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    准备下一轮迭代的训练数据
    
    参数:
        df: 包含数据的DataFrame
        df_flow: 汇流计算结果DataFrame 
        target_col: 目标列
        col_y_n: y_n列名
        col_y_up: y_up列名
        input_cols: 输入列列表
        
    返回:
        (X_ts_iter, Y_label_iter, X_attr_iter, COMIDs_iter, Dates_iter): 下一轮迭代的训练数据
    """
    # 标准化列名称，确保date列是小写的
    if 'date' in df.columns and 'Date' in df_flow.columns:
        df_flow = df_flow.rename(columns={'Date': 'date'})
    elif 'Date' in df.columns and 'date' in df_flow.columns:
        df = df.rename(columns={'Date': 'date'})
    
    # 检查列是否存在
    required_cols = ['COMID', 'date', col_y_n, col_y_up]
    missing_cols = [col for col in required_cols if col not in df_flow.columns]
    
    if missing_cols:
        print(f"警告: df_flow中缺少列: {missing_cols}")
        print(f"可用列: {df_flow.columns.tolist()}")
        
        # 尝试猜测正确的列名
        corrected_cols = {}
        for col in missing_cols:
            if col.lower() == 'date':
                # 查找任何可能的日期列
                for df_col in df_flow.columns:
                    if df_col.lower() == 'date':
                        corrected_cols[col] = df_col
                        break
            else:
                # 对于其他列，查找类似的名称
                for df_col in df_flow.columns:
                    if col.lower() in df_col.lower():
                        corrected_cols[col] = df_col
                        break
        
        print(f"修正的列映射: {corrected_cols}")
        
        # 用修正的版本替换缺失的列
        for old_col, new_col in corrected_cols.items():
            required_cols[required_cols.index(old_col)] = new_col
    
    try:
        # 使用可能修正的列名进行合并
        merged = pd.merge(
            df, df_flow[required_cols], 
            left_on=['COMID', 'date'], 
            right_on=[required_cols[0], required_cols[1]], 
            how='left'
        )
    except Exception as e:
        print(f"合并失败: {e}")
        print(f"df列: {df.columns.tolist()}")
        print(f"使用的df_flow列: {required_cols}")
        raise
    
    # 为下一轮训练准备数据
    merged["E_label"] = merged[target_col] - merged[col_y_up]
    comid_list_iter = merged["COMID"].unique().tolist()
    
    with TimingAndMemoryContext("Building Sliding Windows for Iteration Data"):
        X_ts_iter, _, COMIDs_iter, Dates_iter = build_sliding_windows_for_subset(
            df, comid_list_iter, input_cols=input_cols, target_cols=[target_col], time_window=5
        )
        
    Y_label_iter = []
    for cid, date_val in zip(COMIDs_iter, Dates_iter):
        subset = merged[(merged["COMID"] == cid) & (merged["date"] == date_val)]
        if not subset.empty:
            label_val = subset["E_label"].mean()
        else:
            label_val = 0.0
        Y_label_iter.append(label_val)
        
    Y_label_iter = np.array(Y_label_iter, dtype=np.float32)
    
    return X_ts_iter, Y_label_iter, COMIDs_iter, Dates_iter

def check_convergence(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    it: int, 
    epsilon: float
) -> Tuple[bool, Dict[str, float]]:
    """
    检查收敛性
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        it: 当前迭代次数
        epsilon: 收敛阈值
        
    返回:
        (converged, stats): 是否收敛和统计信息
    """
    # 检查y_true是否存在有效值
    valid_mask = ~np.isnan(y_true)
    if np.sum(valid_mask) == 0:
        print("警告：没有有效的观测数据，无法评估收敛性")
        return False, None

    # 只使用有效数据计算残差
    valid_y_true = y_true[valid_mask]
    valid_y_pred = y_pred[valid_mask]
    residual = valid_y_true - valid_y_pred

    # 计算误差统计量
    stats = {
        'iteration': it,
        'mae': np.mean(np.abs(residual)),            # 平均绝对误差
        'mse': np.mean(residual ** 2),               # 均方误差
        'rmse': np.sqrt(np.mean(residual ** 2)),     # 均方根误差
        'max_resid': np.max(np.abs(residual)),       # 最大绝对残差
        'valid_data_points': np.sum(valid_mask)      # 有效数据点数量
    }

    # 输出误差信息
    print(f"  迭代 {it+1} 误差统计 (基于 {stats['valid_data_points']} 个有效观测点):")
    print(f"    平均绝对误差 (MAE): {stats['mae']:.4f}")
    print(f"    均方误差 (MSE): {stats['mse']:.4f}")
    print(f"    均方根误差 (RMSE): {stats['rmse']:.4f}")
    print(f"    最大绝对残差: {stats['max_resid']:.4f}")

    # 判断收敛条件 - 根据算法文档描述
    # 方法1: 达到预定精度
    if stats['mae'] < epsilon:
        print(f"收敛! 平均绝对误差 ({stats['mae']:.4f}) 小于阈值 ({epsilon})")
        logging.info(f"迭代 {it+1} 收敛: MAE = {stats['mae']:.4f} < epsilon = {epsilon}")
        return True, stats

    return False, stats

def check_error_trend_convergence(error_history: List[Dict[str, float]]) -> bool:
    """
    检查误差趋势是否收敛
    
    参数:
        error_history: 误差历史记录
        
    返回:
        是否因误差趋势稳定而收敛
    """
    if len(error_history) >= 3:
        # 取最近三轮的平均误差
        recent_errors = [entry['mae'] for entry in error_history[-3:]]
        # 计算误差变化率
        error_changes = []
        for i in range(1, len(recent_errors)):
            prev_error = recent_errors[i-1]
            if prev_error > 0:
                change = (prev_error - recent_errors[i]) / prev_error
                error_changes.append(change)
        
        # 如果连续误差变化率都很小，也认为收敛
        if error_changes and all(abs(change) < 0.01 for change in error_changes):
            print(f"收敛! 误差变化趋于稳定，最近三轮MAE: {recent_errors}")
            logging.info(f"收敛: 误差变化趋于稳定，最近三轮MAE: {recent_errors}")
            return True
    
    return False

# ===============================================================================
# 主训练程序
# ===============================================================================

def iterative_training_procedure(
    df: pd.DataFrame,
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
    model_dir: str = "models",
    reuse_existing_flow_results: bool = True
) -> CatchmentModel:
    """
    PG-RWQ 迭代训练过程
    
    参数:
        df: 日尺度数据 DataFrame，包含 'COMID'、'Date'、target_col、'Qout' 等字段
        attr_df: 河段属性DataFrame
        input_features: 输入特征列表
        attr_features: 属性特征列表
        river_info: 河段信息 DataFrame，包含 'COMID' 和 'NextDownID'
        target_cols: 目标变量列表
        target_col: 主目标变量名称，如 "TN"
        max_iterations: 最大迭代次数
        epsilon: 收敛阈值（残差最大值）
        model_type: 'rf' 或 'lstm'
        input_dim: 模型输入维度（须与 input_cols 长度一致）
        hidden_size, num_layers, attr_dim, fc_dim: 模型参数
        device: 训练设备
        comid_wq_list: 水质站点COMID列表
        comid_era5_list: ERA5覆盖的COMID列表
        input_cols: 时间序列输入特征列表
        start_iteration: 起始迭代轮数，0表示从头开始，>0表示从指定轮次开始
        model_version: 模型版本号
        flow_results_dir: 汇流结果保存目录
        model_dir: 模型保存目录
        reuse_existing_flow_results: 是否重用已存在的汇流计算结果
        
    返回:
        训练好的模型对象
    """
    # ============================================================
    # 初始化与内存监控
    # ============================================================
    # 启动内存跟踪
    memory_tracker = MemoryTracker(interval_seconds=120)
    memory_tracker.start()

    # 初始内存状态记录
    if device == 'cuda' and torch.cuda.is_available():
        log_memory_usage("[Training Start] ", level=0)
    
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
    
    # 初始化误差历史记录
    if not hasattr(iterative_training_procedure, 'error_history'):
        iterative_training_procedure.error_history = []
    
    # ============================================================
    # 只有当 start_iteration 为 0 时才执行初始模型训练和汇流计算
    # ============================================================
    if start_iteration == 0:
        # 构建河段属性字典
        with TimingAndMemoryContext("Building Attribute Dictionary", memory_log_level=1):
            attr_dict = build_attribute_dictionary(attr_df, attr_features)

        # 准备头部河段训练数据
        X_ts_head, Y_head_orig, COMIDs_head, Dates_head = prepare_training_data_for_head_segments(
            df, attr_df, comid_wq_list, comid_era5_list, target_cols, output_dir, model_version
        )
        if X_ts_head is None:
            memory_tracker.stop()
            memory_tracker.report()
            return None
        
        # 取目标列中的第一列作为目标
        Y_head = Y_head_orig[:, 0]
        
        # 标准化数据
        X_ts_head, attr_dict, ts_scaler, attr_scaler = standardize_data(X_ts_head, attr_dict)
        
        # 确定数据维度
        N, T, input_dim = X_ts_head.shape
        attr_dim = len(next(iter(attr_dict.values())))
        
        # 划分训练集和验证集
        train_val_data = split_train_val_data(X_ts_head, Y_head, COMIDs_head)
        
        # 创建或加载初始模型
        initial_model_path = f"{model_save_dir}/model_initial_A0_{model_version}.pth"
        model = create_or_load_model(
            model_type, input_dim, hidden_size, num_layers, attr_dim, fc_dim, 
            device, initial_model_path, attr_dict, train_val_data, "Initial Model Creation"
        )
        
        # 创建批量预测函数
        batch_model_func = create_batch_model_func(df, attr_dict, model, target_cols)
        
        # 执行初始汇流计算（或加载已有结果）
        exists, flow_result_path = check_existing_flow_routing_results(0, model_version, output_dir)
        df_flow = perform_flow_routing_calculation(
            df, 0, batch_model_func, river_info, attr_dict, model, target_cols, 
            attr_df, output_dir, model_version, exists, flow_result_path, reuse_existing_flow_results
        )
    else:
        # 如果 start_iteration > 0，跳过初始模型训练和初始汇流计算
        # 创建模型实例并加载上一轮迭代的模型
        
        # 构建河段属性字典
        with TimingAndMemoryContext("Building Attribute Dictionary"):
            attr_dict = build_attribute_dictionary(attr_df, attr_features)
            # 标准化属性
            attr_dict, attr_scaler = standardize_attributes(attr_dict)
        
        # 加载上一轮迭代的模型
        last_iteration = start_iteration - 1
        model_path = f"{model_save_dir}/model_A{last_iteration}_{model_version}.pth"
        model = create_or_load_model(
            model_type, input_dim, hidden_size, num_layers, attr_dim, fc_dim, 
            device, model_path, context_name="Loading Previous Iteration Model"
        )
        
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
    
    # ============================================================
    # 主迭代训练和汇流计算过程
    # ============================================================
    # 从 start_iteration 开始迭代
    for it in range(start_iteration, max_iterations):
        with TimingAndMemoryContext(f"Iteration {it+1}/{max_iterations}"):
            print(f"\n迭代 {it+1}/{max_iterations}")
            
            # 获取当前迭代的列名
            col_y_n = f'y_n_{it}_{target_col}'
            col_y_up = f'y_up_{it}_{target_col}'
            
            # 显示 df_flow 中的列，用于调试
            print(f"df_flow columns: {df_flow.columns.tolist()}")
            
            # 合并 df 和 df_flow 以进行评估和准备下一轮训练数据
            merged = pd.merge(
                df, df_flow[['COMID', 'date', col_y_n, col_y_up]], 
                on=['COMID', 'date'], 
                how='left'
            )
            
            # 提取 y_true 和 y_pred 进行收敛性检查
            y_true = merged[target_col].values
            y_pred = merged[col_y_n].values
            
            # 检查收敛性
            converged, stats = check_convergence(y_true, y_pred, it, epsilon)
            
            # 如果有统计信息，添加到误差历史记录中
            if stats:
                iterative_training_procedure.error_history.append(stats)
            
            # 检查是否收敛
            if converged:
                break
            
            # 检查误差趋势是否稳定
            if check_error_trend_convergence(iterative_training_procedure.error_history):
                break
            
            # 准备下一轮迭代的训练数据
            merged["E_label"] = merged[target_col] - merged[col_y_up]
            comid_list_iter = merged["COMID"].unique().tolist()
            
            # 构建下一轮迭代的训练数据
            with TimingAndMemoryContext(f"Building Sliding Windows for Iteration {it+1}"):
                X_ts_iter, Y_label_iter, COMIDs_iter, Dates_iter = build_sliding_windows_for_subset(
                    df, comid_list_iter, input_cols=input_cols, target_cols=[target_col], time_window=5
                )
            
            # 构建属性矩阵
            X_attr_iter = np.vstack([
                attr_dict.get(str(cid), np.zeros_like(next(iter(attr_dict.values()))))
                for cid in COMIDs_iter
            ])
            
            print("  更新模型训练：使用更新后的 E_label。")
            
            # 训练更新模型或加载已有模型
            model_path = f"{model_save_dir}/model_A{it+1}_{model_version}.pth"
            if not os.path.exists(model_path):
                # 训练更新模型
                with TimingAndMemoryContext(f"Model Training for Iteration {it+1}"):
                    model.train_model(X_attr_iter, COMIDs_iter, X_ts_iter, Y_label_iter, 
                                     epochs=5, lr=1e-3, patience=2, batch_size=32)
                
                # 保存本轮迭代的模型
                with TimingAndMemoryContext(f"Saving Model for Iteration {it+1}"):
                    model.save_model(model_path)
                    print(f"模型已保存至: {model_path}")
            else:
                # 加载已有模型
                with TimingAndMemoryContext(f"Loading Existing Model for Iteration {it+1}"):
                    model.load_model(model_path)
                    print(f"已加载现有模型: {model_path}")
            
            # 创建更新后的模型预测函数
            updated_model_func = create_updated_model_func(
                df, attr_dict, model, target_col, input_cols, device
            )
            
            # 执行新一轮汇流计算（或加载已有结果）
            exists, flow_result_path = check_existing_flow_routing_results(it+1, model_version, output_dir)
            df_flow = perform_flow_routing_calculation(
                df, it+1, updated_model_func, river_info, attr_dict, model, target_cols, 
                attr_df, output_dir, model_version, exists, flow_result_path, reuse_existing_flow_results
            )
    
    # ============================================================
    # 完成与清理
    # ============================================================
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