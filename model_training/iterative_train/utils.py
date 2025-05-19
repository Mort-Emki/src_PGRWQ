"""
utils.py - 辅助函数模块

提供各种实用函数，如结果处理、路径管理、批量函数创建等。
简化主逻辑中的重复代码，提高可维护性。
"""

import os
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime

from PGRWQI.model_training.gpu_memory_utils import TimingAndMemoryContext
from PGRWQI.logging_utils import ensure_dir_exists


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


def create_predictor(data_handler, model_manager, all_target_cols, target_col):
    """创建预测器实例"""
    from .predictor import CatchmentPredictor
    return CatchmentPredictor(data_handler, model_manager, all_target_cols, target_col)

def save_flow_results(df_flow, iteration, model_version, output_dir):
    """
    保存汇流计算结果
    
    参数:
        df_flow: 汇流计算结果DataFrame
        iteration: 迭代次数
        model_version: 模型版本号
        output_dir: 输出目录
    """
    # 确保目录存在
    ensure_dir_exists(output_dir)
    
    # 保存结果
    result_path = os.path.join(output_dir, f"flow_routing_iteration_{iteration}_{model_version}.csv")
    df_flow.to_csv(result_path, index=False)
    logging.info(f"迭代 {iteration} 汇流计算结果已保存至 {result_path}")


def time_function(func):
    """
    函数执行时间装饰器
    
    参数:
        func: 要计时的函数
        
    返回:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"函数 {func.__name__} 执行时间: {end_time - start_time:.2f} 秒")
        return result
    return wrapper


def split_train_val_data(
    X_ts: np.ndarray, 
    Y: np.ndarray, 
    COMIDs: np.ndarray,
    train_ratio: float = 0.8
) -> Tuple:
    """
    将数据划分为训练集和验证集
    
    参数:
        X_ts: 时间序列数据
        Y: 目标变量
        COMIDs: COMID数组
        train_ratio: 训练集比例
        
    返回:
        (X_ts_train, comid_arr_train, Y_train, X_ts_val, comid_arr_val, Y_val): 划分后的数据
    """
    with TimingAndMemoryContext("训练/验证集划分"):
        N = len(X_ts)
        indices = np.random.permutation(N)
        train_size = int(N * train_ratio)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]

        X_ts_train = X_ts[train_indices]
        comid_arr_train = COMIDs[train_indices]
        Y_train = Y[train_indices]

        X_ts_val = X_ts[valid_indices]
        comid_arr_val = COMIDs[valid_indices]
        Y_val = Y[valid_indices]

    return X_ts_train, comid_arr_train, Y_train, X_ts_val, comid_arr_val, Y_val