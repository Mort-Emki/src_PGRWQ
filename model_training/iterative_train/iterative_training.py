"""
iterative_training.py - PG-RWQ 迭代训练模块

该模块实现 PG-RWQ (Physics-Guided Recursive Water Quality) 模型的迭代训练过程。
整合数据处理、模型训练、汇流计算和评估功能，实现完整的训练循环。
"""

import os
import numpy as np
import pandas as pd
import logging
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

# 导入项目中的函数
from PGRWQI.flow_routing import flow_routing_calculation
from PGRWQI.logging_utils import ensure_dir_exists
from PGRWQI.model_training.gpu_memory_utils import (
    log_memory_usage, 
    TimingAndMemoryContext, 
    MemoryTracker,
    force_cuda_memory_cleanup
)

# 导入自定义组件
from .data_handler import DataHandler
from .model_manager import ModelManager
from .evaluation import ConvergenceChecker, DataValidator
from .utils import (
    check_existing_flow_routing_results,
    create_batch_model_func,
    create_updated_model_func,
    save_flow_results,
    split_train_val_data
)


def iterative_training_procedure(
    df: pd.DataFrame,
    attr_df: pd.DataFrame,
    input_features: List[str] = None,
    attr_features: List[str] = None,
    river_info: pd.DataFrame = None,
    all_target_cols: List[str] = ["TN", "TP"],
    target_col: str = "TN",
    max_iterations: int = 10,
    epsilon: float = 0.01,
    model_type: str = 'rf',
    model_params: Dict[str, Any] = None,
    device: str = 'cuda',
    comid_wq_list: list = None,
    comid_era5_list: list = None,
    input_cols: list = None,
    start_iteration: int = 0,
    model_version: str = "v1",
    flow_results_dir: str = "flow_results",
    model_dir: str = "models",
    reuse_existing_flow_results: bool = True
) -> Any:
    """
    PG-RWQ 迭代训练主程序
    
    参数:
        df: 日尺度数据 DataFrame，包含 'COMID'、'Date'、target_col、'Qout' 等字段
        attr_df: 河段属性DataFrame
        input_features: 输入特征列表
        attr_features: 属性特征列表
        river_info: 河段信息 DataFrame，包含 'COMID' 和 'NextDownID'
        all_target_cols: 所有目标参数列表
        target_col: 主目标参数名称，如 "TN"
        max_iterations: 最大迭代次数
        epsilon: 收敛阈值（残差平均值）
        model_type: 模型类型，如 'lstm', 'rf', 'informer' 等
        model_params: 模型超参数字典，包含特定模型类型所需的所有参数
        device: 训练设备
        comid_wq_list: 水质站点COMID列表
        comid_era5_list: ERA5覆盖的COMID列表
        input_cols: 时间序列输入特征列表（如果与input_features不同）
        start_iteration: 起始迭代轮数，0表示从头开始，>0表示从指定轮次开始
        model_version: 模型版本号
        flow_results_dir: 汇流结果保存目录
        model_dir: 模型保存目录
        reuse_existing_flow_results: 是否重用已存在的汇流计算结果
        
    返回:
        训练好的模型对象
    """
    # ======================================================================
    # 1. 初始化设置
    # ======================================================================
    # 确保model_params不为None
    if model_params is None:
        model_params = {}
    
    # 使用input_features作为input_cols（如果未指定）
    if input_cols is None:
        input_cols = input_features
    
    # 初始化内存监控
    memory_tracker = MemoryTracker(interval_seconds=120)
    memory_tracker.start()
    
    # 记录初始内存状态
    if device == 'cuda' and torch.cuda.is_available():
        log_memory_usage("[训练开始] ", level=0)
    
    # 创建结果目录
    output_dir = ensure_dir_exists(flow_results_dir)
    model_save_dir = ensure_dir_exists(model_dir)
    logging.info(f"汇流计算结果将保存至 {output_dir}")
    logging.info(f"模型将保存至 {model_save_dir}")
    
    # 记录训练开始信息
    if start_iteration > 0:
        logging.info(f"从迭代 {start_iteration} 开始，模型版本 {model_version}")
    else:
        logging.info(f"从初始训练（迭代 0）开始，模型版本 {model_version}")
        logging.info('选择头部河段进行初始模型训练')
    
    # 初始化组件
    data_handler = DataHandler()
    data_handler.initialize(df, attr_df, input_features, attr_features)
    
    model_manager = ModelManager(model_type, device, model_save_dir)
    
    convergence_checker = ConvergenceChecker(epsilon=epsilon)
    
    data_validator = DataValidator()
    
    try:
        # ======================================================================
        # 2. 初始模型训练与首次汇流计算
        # ======================================================================
        if start_iteration == 0:
            # 为头部河段准备标准化训练数据
            X_ts_scaled, attr_dict_scaled, Y_head, COMIDs_head, Dates_head = data_handler.prepare_training_data_for_head_segments(
                comid_wq_list=comid_wq_list,
                comid_era5_list=comid_era5_list,
                all_target_cols=all_target_cols,
                target_col=target_col,
                output_dir=output_dir,
                model_version=model_version
            )
            
            if X_ts_scaled is None:
                logging.error("无法准备头部河段训练数据，终止训练")
                memory_tracker.stop()
                memory_tracker.report()
                return None
            
            # 确定数据维度，更新模型参数
            N, T, input_dim = X_ts_scaled.shape
            attr_dim = next(iter(attr_dict_scaled.values())).shape[0]
            
            # 获取构建参数和训练参数
            build_params = model_params.get('build', {}).copy()
            train_params = model_params.get('train', {}).copy()
            
            # 添加缺失的维度参数
            if 'input_dim' not in build_params:
                build_params['input_dim'] = input_dim
            if 'attr_dim' not in build_params:
                build_params['attr_dim'] = attr_dim
            
            # 划分训练集和验证集
            train_val_data = split_train_val_data(X_ts_scaled, Y_head, COMIDs_head)
            
            # 创建或加载初始模型
            initial_model_path = f"{model_save_dir}/model_initial_A0_{model_version}.pth"
            model = model_manager.create_or_load_model(
                build_params=build_params,
                train_params=train_params,
                model_path=initial_model_path,
                attr_dict=attr_dict_scaled,
                train_data=train_val_data
            )
            
            # 创建批处理函数
            batch_func = create_batch_model_func(
                data_handler=data_handler,
                model_manager=model_manager,
                all_target_cols=all_target_cols,
                target_col=target_col
            )
            
            # 执行初始汇流计算（或加载已有结果）
            exists, flow_result_path = check_existing_flow_routing_results(0, model_version, output_dir)
            if exists and reuse_existing_flow_results:
                # 如果存在且配置为重用，直接加载已有结果
                with TimingAndMemoryContext("加载已有汇流计算结果"):
                    logging.info(f"发现已存在的汇流计算结果，加载：{flow_result_path}")
                    df_flow = pd.read_csv(flow_result_path)
                    logging.info(f"成功加载汇流计算结果，共 {len(df_flow)} 条记录")
            else:
                # 如果不存在或配置为不重用，执行汇流计算
                with TimingAndMemoryContext("执行初始汇流计算"):
                    # 获取标准化后的完整属性字典
                    attr_dict_all = data_handler.get_standardized_attr_dict()
                    
                    # 初始迭代使用E_save=1来保存E值
                    df_flow = flow_routing_calculation(
                        df=df.copy(), 
                        iteration=0, 
                        model_func=batch_func, 
                        river_info=river_info, 
                        v_f_TN=35.0,
                        v_f_TP=44.5,
                        attr_dict=attr_dict_all,
                        model=model,
                        all_target_cols=all_target_cols,
                        target_col=target_col,
                        attr_df=attr_df,
                        E_save=1,  # 保存E值
                        E_save_path=f"{output_dir}/E_values_{model_version}"
                    )
                    
                    # 保存汇流计算结果
                    save_flow_results(df_flow, 0, model_version, output_dir)
        else:
            # ======================================================================
            # 从指定迭代次数开始（加载已有模型和汇流结果）
            # ======================================================================
            # 加载上一轮迭代的模型
            last_iteration = start_iteration - 1
            
            # 获取模型参数
            build_params = model_params.get('build', {}).copy()
            
            # 添加缺失的维度参数
            if 'input_dim' not in build_params and input_features:
                build_params['input_dim'] = len(input_features)
            if 'attr_dim' not in build_params and attr_features:
                build_params['attr_dim'] = len(attr_features)
            
            # 加载上一轮模型
            model_path = f"{model_save_dir}/model_A{last_iteration}_{model_version}.pth"
            
            if not os.path.exists(model_path):
                logging.error(f"无法找到上一轮模型: {model_path}")
                memory_tracker.stop()
                memory_tracker.report()
                return None
                
            model = model_manager.create_or_load_model(
                build_params=build_params,
                train_params={},  # 不需要训练参数，只是加载
                model_path=model_path
            )
            
            # 加载上一轮的汇流计算结果
            previous_flow_path = os.path.join(output_dir, f"flow_routing_iteration_{last_iteration}_{model_version}.csv")
            
            if not os.path.exists(previous_flow_path):
                logging.error(f"无法找到上一轮汇流计算结果: {previous_flow_path}")
                memory_tracker.stop()
                memory_tracker.report()
                return None
            
            with TimingAndMemoryContext("加载上一轮汇流计算结果"):
                df_flow = pd.read_csv(previous_flow_path)
                logging.info(f"已加载上一轮汇流计算结果: {previous_flow_path}")
        
        # ======================================================================
        # 3. 主迭代训练循环
        # ======================================================================
        for it in range(start_iteration, max_iterations):
            with TimingAndMemoryContext(f"迭代 {it+1}/{max_iterations}"):
                logging.info(f"\n迭代 {it+1}/{max_iterations} 开始")
                
                # 获取当前迭代的列名
                col_y_n = f'y_n_{it}_{target_col}'
                col_y_up = f'y_up_{it}_{target_col}'
                
                # 显示df_flow中的列，用于调试
                logging.info(f"df_flow列: {df_flow.columns.tolist()}")
                
                # 合并df和df_flow以进行评估
                merged = pd.merge(
                    df, df_flow[['COMID', 'date', col_y_n, col_y_up]], 
                    on=['COMID', 'date'], 
                    how='left'
                )
                
                # 提取y_true和y_pred进行收敛性检查
                y_true = merged[target_col].values
                y_pred = merged[col_y_n].values
                
                # 检查收敛性
                converged, stats = convergence_checker.check_convergence(y_true, y_pred, it)
                
                # 如果已收敛，跳出迭代循环
                if converged:
                    logging.info(f"迭代 {it+1} 已达到收敛条件，训练结束")
                    break
                
                # ======================================================================
                # 4. 准备下一轮迭代的训练数据
                # ======================================================================
                logging.info("准备下一轮迭代的训练数据")
                # 准备下一轮迭代的训练数据
                X_ts_iter, attr_dict_iter, Y_label_iter, COMIDs_iter, Dates_iter = data_handler.prepare_next_iteration_data(
                    df_flow=df_flow,
                    target_col=target_col,
                    col_y_n=col_y_n,
                    col_y_up=col_y_up
                )
                
                if X_ts_iter is None:
                    logging.error("准备训练数据失败，无法继续迭代")
                    break
                
                # 划分训练集和验证集
                train_val_data = split_train_val_data(X_ts_iter, Y_label_iter, COMIDs_iter)
                
                # ======================================================================
                # 5. 训练/加载迭代模型
                # ======================================================================
                model_path = f"{model_save_dir}/model_A{it+1}_{model_version}.pth"
                
                # 看是否已有模型
                if not os.path.exists(model_path):
                    # 如果没有已有模型，训练新模型
                    logging.info(f"训练迭代 {it+1} 的模型")
                    model = model_manager.create_or_load_model(
                        build_params=build_params,
                        train_params=train_params,
                        model_path=model_path,
                        attr_dict=attr_dict_iter,
                        train_data=train_val_data
                    )
                else:
                    # 如果有已有模型，直接加载
                    logging.info(f"加载已有的迭代 {it+1} 模型: {model_path}")
                    model = model_manager.create_or_load_model(
                        build_params=build_params,
                        train_params={},
                        model_path=model_path
                    )
                
                # ======================================================================
                # 6. 执行新一轮汇流计算
                # ======================================================================
                # 创建更新后的模型预测函数
                updated_model_func = create_updated_model_func(
                    data_handler=data_handler, 
                    model_manager=model_manager,
                    target_col=target_col,
                    device=device
                )
                
                # 执行新一轮汇流计算（或加载已有结果）
                exists, flow_result_path = check_existing_flow_routing_results(it+1, model_version, output_dir)
                
                if exists and reuse_existing_flow_results:
                    # 如果存在且配置为重用，直接加载已有结果
                    with TimingAndMemoryContext(f"加载迭代 {it+1} 已有汇流计算结果"):
                        logging.info(f"发现已存在的汇流计算结果，加载：{flow_result_path}")
                        df_flow = pd.read_csv(flow_result_path)
                        logging.info(f"成功加载汇流计算结果，共 {len(df_flow)} 条记录")
                else:
                    # 如果不存在或配置为不重用，执行汇流计算
                    with TimingAndMemoryContext(f"执行迭代 {it+1} 汇流计算"):
                        # 获取标准化后的完整属性字典
                        attr_dict_all = data_handler.get_standardized_attr_dict()
                        
                        df_flow = flow_routing_calculation(
                            df=df.copy(), 
                            iteration=it+1, 
                            model_func=updated_model_func, 
                            river_info=river_info, 
                            v_f_TN=35.0,
                            v_f_TP=44.5,
                            attr_dict=attr_dict_all,
                            model=model,
                            all_target_cols=all_target_cols,
                            target_col=target_col,
                            attr_df=attr_df,
                            E_save=1,  # 保存E值
                            E_save_path=f"{output_dir}/E_values_{model_version}"
                        )
                        
                        # 保存汇流计算结果
                        save_flow_results(df_flow, it+1, model_version, output_dir)
                
                # ======================================================================
                # 7. 检查数据质量
                # ======================================================================
                # 检查此轮迭代的汇流计算结果的异常值
                logging.info(f"检查迭代 {it+1} 的汇流计算结果质量")
                is_valid_data, abnormal_report = data_validator.check_dataframe_abnormalities(
                    df_flow, it+1, all_target_cols
                )
                
                # 如果数据无效，尝试修复
                if not is_valid_data:
                    logging.warning(f"迭代 {it+1} 的汇流计算结果包含过多异常值，尝试修复...")
                    
                    # 修复异常值
                    df_flow = data_validator.fix_dataframe_abnormalities(
                        df_flow, it+1, all_target_cols
                    )
                    
                    # 保存修复后的结果
                    fixed_path = os.path.join(output_dir, f"flow_routing_iteration_{it+1}_{model_version}_fixed.csv")
                    df_flow.to_csv(fixed_path, index=False)
                    logging.info(f"修复后的结果已保存至 {fixed_path}")
                
                # 验证数据一致性
                is_coherent = data_validator.validate_data_coherence(
                    df, df_flow, input_features, all_target_cols, it+1
                )
                
                if not is_coherent:
                    logging.warning(f"数据一致性检查失败，可能会影响迭代 {it+2} 的训练")
        
        # ======================================================================
        # 8. 完成训练
        # ======================================================================
        # 生成内存报告
        memory_tracker.stop()
        memory_stats = memory_tracker.report()
        
        if device == 'cuda':
            log_memory_usage("[训练完成] ")
        
        # 保存最终模型
        final_iter = min(it+1, max_iterations)
        final_model_path = os.path.join(model_save_dir, f"final_model_iteration_{final_iter}_{model_version}.pth")
        
        if model is not None:
            model.save_model(final_model_path)
            logging.info(f"最终模型已保存至 {final_model_path}")
        
        return model
        
    except Exception as e:
        # 捕获训练过程中的异常
        logging.exception(f"训练过程中发生错误: {str(e)}")
        
        # 清理资源
        if device == 'cuda' and torch.cuda.is_available():
            force_cuda_memory_cleanup()
            
        memory_tracker.stop()
        memory_tracker.report()
        
        return None