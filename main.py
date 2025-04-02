"""
PG-RWQ 水质预测模型训练主程序

本程序是物理约束的递归水质预测模型(PG-RWQ)的主入口。
程序通过 JSON 配置文件加载各种参数，支持不同类型的模型训练，
并在训练过程中监控资源使用情况。

作者: Mortenki
日期: 2025-04-02
版本: 1.0
"""

import os
import time
import sys
import logging
import json
import argparse
import pandas as pd
import numpy as np
import torch
import datetime
import threading
from typing import Dict, Any, List, Optional, Tuple

# 将父目录添加到路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.getLogger().setLevel(logging.DEBUG)

# 导入自定义模块
from PGRWQI.data_processing import load_daily_data, load_river_attributes, detect_and_handle_anomalies, check_river_network_consistency
from PGRWQI.model_training.train import iterative_training_procedure
from PGRWQI.logging_utils import setup_logging, restore_stdout_stderr, ensure_dir_exists
from PGRWQI.tqdm_logging import tqdm
from PGRWQI.model_training.gpu_memory_utils import (
    log_memory_usage, 
    TimingAndMemoryContext, 
    MemoryTracker, 
    periodic_memory_check,
    get_gpu_memory_info,
    set_memory_log_verbosity, 
    set_monitoring_enabled
)

#============================================================================
# 配置文件处理
#============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    从JSON文件加载配置参数
    
    参数:
        config_path: JSON配置文件路径
    
    返回:
        包含配置参数的字典
        
    异常:
        ValueError: 当配置文件格式不正确或缺少必要参数时抛出
    """
    try:
        # 读取JSON文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证必要的配置部分是否存在
        required_sections = ['basic', 'features', 'data', 'models']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"配置文件缺少'{section}'部分")
        
        # 验证模型类型是否存在
        if 'model_type' not in config['basic']:
            raise ValueError("基本配置中缺少'model_type'参数")
        
        # 验证指定的模型类型是否在models配置中
        model_type = config['basic']['model_type']
        if model_type not in config['models']:
            raise ValueError(f"在配置中未找到模型类型'{model_type}'的参数")
        
        logging.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        error_msg = f"加载配置文件{config_path}时出错: {str(e)}"
        logging.error(error_msg)
        raise ValueError(error_msg)

def get_model_params(config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    根据模型类型提取模型特定参数
    
    参数:
        config: 配置字典
        model_type: 模型类型字符串（如'lstm', 'rf'）
    
    返回:
        包含模型参数的字典
        
    异常:
        ValueError: 当指定的模型类型不在配置中时抛出
    """
    # 验证模型类型是否存在
    if model_type not in config['models']:
        raise ValueError(f"在配置中未找到模型类型'{model_type}'")
    
    # 获取模型特定参数
    model_params = config['models'][model_type].copy()
    
    # 根据特征列表添加input_dim和attr_dim
    model_params['input_dim'] = len(config['features']['input_features'])
    model_params['attr_dim'] = len(config['features']['attr_features'])
    
    logging.info(f"已提取'{model_type}'模型的参数，特征维度: {model_params['input_dim']}，属性维度: {model_params['attr_dim']}")
    return model_params

#============================================================================
# 内存监控
#============================================================================

def create_memory_monitor_file(interval_seconds: int = 300, log_dir: str = "logs") -> Optional[threading.Thread]:
    """
    创建GPU内存使用监控文件并启动监控线程
    
    参数:
        interval_seconds: 记录间隔（默认：300秒 = 5分钟）
        log_dir: 日志保存目录
    
    返回:
        监控线程对象，如果创建失败则返回None
    """
    # 使用绝对路径
    original_dir = os.getcwd()
    log_dir = os.path.abspath(log_dir)
    
    # 创建目录（如果不存在）
    try:
        os.makedirs(log_dir, exist_ok=True)
        logging.info(f"已创建/验证GPU内存日志目录: {log_dir}")
    except Exception as e:
        logging.error(f"创建目录{log_dir}时出错: {str(e)}")
        # 使用当前目录作为备选
        log_dir = original_dir
        logging.info(f"改用当前目录保存日志: {log_dir}")
    
    # 创建日志文件（绝对路径）
    log_file = os.path.join(log_dir, "gpu_memory_log.csv")
    
    # 创建或清空文件并写入表头
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,allocated_mb,reserved_mb,max_allocated_mb,percent_used\n")
        logging.info(f"已创建GPU内存日志文件: {log_file}")
    except Exception as e:
        logging.error(f"创建GPU内存日志文件时出错: {str(e)}")
        return None
    
    # 定义监控线程函数
    def _monitor_file():
        """记录GPU内存使用情况到文件的线程函数"""
        while True:
            try:
                # 获取当前时间
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 如果有GPU可用，记录内存使用情况
                if torch.cuda.is_available():
                    info = get_gpu_memory_info()
                    try:
                        # 使用绝对路径打开文件
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"{timestamp},{info['allocated_mb']:.2f},{info['reserved_mb']:.2f},"
                                   f"{info['max_allocated_mb']:.2f},{info['usage_percent']:.2f}\n")
                    except Exception as e:
                        logging.error(f"写入GPU内存日志({log_file})时出错: {str(e)}")
            except Exception as e:
                logging.error(f"GPU内存监控过程中出错: {str(e)}")
            
            # 等待指定时间间隔
            time.sleep(interval_seconds)
    
    # 创建并启动守护线程
    monitor_thread = threading.Thread(target=_monitor_file, daemon=True)
    monitor_thread.start()
    logging.info(f"已启动GPU内存文件记录（间隔: {interval_seconds}秒）")
    return monitor_thread

#============================================================================
# 数据处理模块
#============================================================================

def load_data(data_config: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """
    加载训练所需的各种数据
    
    参数:
        data_config: 包含数据文件路径的配置字典
    
    返回:
        df: 日尺度数据DataFrame
        attr_df: 河段属性DataFrame
        comid_wq_list: 水质站点COMID列表
        comid_era5_list: ERA5覆盖的COMID列表
    """
    # 加载河段属性数据
    with TimingAndMemoryContext("加载河段属性数据"):
        attr_df = load_river_attributes(data_config['river_attributes_csv'])
        logging.info(f"河段属性数据形状: {attr_df.shape}")
    
    # 提取河网信息
    with TimingAndMemoryContext("提取河网信息"):
        river_info = attr_df[['COMID', 'NextDownID', 'lengthkm', 'order_']].copy()
        # 确保NextDownID为数值型；若存在缺失值则填充为0
        river_info['NextDownID'] = pd.to_numeric(
            river_info['NextDownID'], errors='coerce'
        ).fillna(0).astype(int)
        
        # 加载COMID列表
        comid_wq_list = pd.read_csv(
            data_config['comid_wq_list_csv'], header=None
        )[0].tolist()
        logging.info(f"加载了{len(comid_wq_list)}个水质站点COMID")
        
        comid_era5_list = pd.read_csv(
            data_config['comid_era5_list_csv'], header=None
        )[0].tolist()
        logging.info(f"加载了{len(comid_era5_list)}个ERA5覆盖COMID")
    
    # 加载日尺度数据
    with TimingAndMemoryContext("加载日尺度数据"):
        df = load_daily_data(data_config['daily_csv'])
        logging.info(f"日尺度数据形状: {df.shape}")
        
        # 检查异常值
        logging.info("检查数据中的异常值...")
        df, anomaly_results = detect_and_handle_anomalies(
            df, 
            columns_to_check=['Qout'], 
            fix_negative=True,
            negative_replacement=0.001,
            outlier_method='iqr',
            outlier_threshold=3.0,
            verbose=True,
            logger=logging
        )
        
        # 汇报修复结果
        if 'fixed_negative_counts' in anomaly_results:
            if 'Qout' in anomaly_results['fixed_negative_counts']:
                fixed_count = anomaly_results['fixed_negative_counts']['Qout']
                logging.info(f"已修复{fixed_count}个负Qout值（替换为0.001）")
        
        # 检查河网拓扑结构一致性
        with TimingAndMemoryContext("检查河网拓扑结构一致性"):
            network_results = check_river_network_consistency(
                river_info,
                verbose=True,
                logger=logging
            )
            
            # 汇报检查结果
            if network_results['has_issues']:
                logging.warning("河网拓扑结构检查发现问题，请查看详细日志")
    
    return df, attr_df, comid_wq_list, comid_era5_list, river_info

#============================================================================
# 设备检测与初始化
#============================================================================

def initialize_device() -> str:
    """
    检查GPU可用性并初始化计算设备
    
    返回:
        device: 计算设备类型字符串，'cuda'或'cpu'
    """
    with TimingAndMemoryContext("设备初始化"):
        # 检查CUDA是否可用
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"使用设备: {device}")
        
        # 如果使用GPU，记录详细信息
        if device == "cuda":
            # 记录CUDA设备信息
            for i in range(torch.cuda.device_count()):
                device_properties = torch.cuda.get_device_properties(i)
                cuda_info = (
                    f"CUDA设备 {i}: {device_properties.name}\n"
                    f"  总内存: {device_properties.total_memory / (1024**3):.2f} GB\n"
                    f"  CUDA版本: {device_properties.major}.{device_properties.minor}"
                )
                logging.info(cuda_info)
    return device

#============================================================================
# 主程序
#============================================================================

def main():
    """
    PG-RWQ训练流程主函数
    
    处理命令行参数，加载配置，初始化日志和内存监控，
    加载数据，并执行迭代训练过程。
    """

    ## 输出当前路径
    current_path = os.getcwd()
    print(f"当前工作目录: {current_path}")
    #------------------------------------------------------------------------
    # 1. 解析命令行参数
    #------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="PG-RWQ 水质预测模型训练程序")
    parser.add_argument("--config", type=str, default="PGRWQI\\config.json",
                        help="JSON配置文件路径")
    parser.add_argument("--data_dir", type=str, default=None, 
                        help="数据目录路径（覆盖配置中的路径）")
    parser.add_argument("--override_model_type", type=str, ##default='lstm',
                        help="覆盖配置中指定的模型类型")
    args = parser.parse_args()
    
    #------------------------------------------------------------------------
    # 2. 加载配置
    #------------------------------------------------------------------------
    config = load_config(args.config)
    
    # 应用命令行覆盖
    if args.override_model_type:
        config['basic']['model_type'] = args.override_model_type
        logging.info(f"模型类型已被命令行参数覆盖为: {args.override_model_type}")
    
    # 分别获取不同部分的配置
    basic_config = config['basic']
    feature_config = config['features']
    data_config = config['data']
    
    # 获取基于选定模型类型的特定配置
    model_type = basic_config['model_type']
    model_params = get_model_params(config, model_type)
    
    #------------------------------------------------------------------------
    # 3. 设置日志和内存监控
    #------------------------------------------------------------------------
    # 设置日志
    log_dir = ensure_dir_exists(basic_config.get('log_dir', 'logs'))
    logger = setup_logging(log_dir=log_dir)
    
    # 初始化总体内存跟踪
    overall_memory_tracker = MemoryTracker(interval_seconds=30)
    overall_memory_tracker.start()
    
    try:
        #--------------------------------------------------------------------
        # 4. 记录系统信息
        #--------------------------------------------------------------------
        logging.info("=" * 80)
        logging.info("PG-RWQ 水质预测模型训练开始")
        logging.info(f"系统时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Python版本: {sys.version}")
        logging.info(f"PyTorch版本: {torch.__version__}")
        logging.info(f"使用模型类型: {model_type}")
        logging.info("=" * 80)
        
        #--------------------------------------------------------------------
        # 5. 启动GPU内存监控（如果启用）
        #--------------------------------------------------------------------
        if not basic_config.get('disable_monitoring', False) and torch.cuda.is_available():
            # 设置内存日志详细程度
            set_memory_log_verbosity(basic_config.get('memory_log_verbosity', 1))
            
            # 启动周期性内存检查（控制台）
            periodic_monitor = periodic_memory_check(
                interval_seconds=basic_config.get('memory_check_interval', 30)
            )
            
            # 启动基于文件的内存日志记录
            file_monitor = create_memory_monitor_file(
                interval_seconds=basic_config.get('memory_check_interval', 30), 
                log_dir=log_dir
            )
            
            # 初始内存状态
            log_memory_usage("[初始GPU状态] ")
        
        #--------------------------------------------------------------------
        # 6. 设置工作目录
        #--------------------------------------------------------------------
        if args.data_dir:
            with TimingAndMemoryContext("设置工作目录"):
                os.chdir(args.data_dir)
                logging.info(f"工作目录已更改为: {args.data_dir}")
        elif 'data_dir' in basic_config:
            with TimingAndMemoryContext("设置工作目录"):
                os.chdir(basic_config['data_dir'])
                logging.info(f"工作目录已更改为: {basic_config['data_dir']}")
        
        #--------------------------------------------------------------------
        # 7. 提取特征列表
        #--------------------------------------------------------------------
        input_features = feature_config['input_features']
        attr_features = feature_config['attr_features']
        
        # 报告特征维度
        input_dim = len(input_features)
        attr_dim = len(attr_features)
        logging.info(f"输入特征: {len(input_features)}个 (维度: {input_dim})")
        print(f"属性特征: {len(attr_features)}个 (维度: {attr_dim})")
        
        #--------------------------------------------------------------------
        # 8. 检查GPU可用性和初始化设备
        #--------------------------------------------------------------------
        device = initialize_device()
        
        #--------------------------------------------------------------------
        # 9. 加载数据
        #--------------------------------------------------------------------
        df, attr_df, comid_wq_list, comid_era5_list, river_info = load_data(data_config)
        
        #--------------------------------------------------------------------
        # 10. 执行迭代训练流程
        #--------------------------------------------------------------------
        print("=" * 80)
        print(f"开始执行{model_type}模型的迭代训练流程")
        print("=" * 80)
        
        with TimingAndMemoryContext("迭代训练流程"):
            final_model = iterative_training_procedure(
                df=df,
                attr_df=attr_df,
                input_features=input_features,
                attr_features=attr_features,
                river_info=river_info,
                target_cols=basic_config.get('target_cols', ['TN']),
                target_col=basic_config.get('target_col', 'TN'),
                max_iterations=basic_config.get('max_iterations', 5),
                epsilon=basic_config.get('epsilon', 0.01),
                model_type=model_type,
                model_params=model_params,  # 传递模型参数字典
                device=device,
                model_version=basic_config.get('model_version', 'default'),
                comid_wq_list=comid_wq_list,
                comid_era5_list=comid_era5_list,
                input_cols=input_features,
                start_iteration=basic_config.get('start_iteration', 0),
                flow_results_dir=basic_config.get('flow_results_dir', 'flow_results'),
                model_dir=basic_config.get('model_dir', 'models'),
                reuse_existing_flow_results=basic_config.get('reuse_existing_flow_results', True)
            )
        
        #--------------------------------------------------------------------
        # 11. 最终内存报告
        #--------------------------------------------------------------------
        if torch.cuda.is_available():
            log_memory_usage("[训练完成] ")
            
            # 报告GPU内存统计信息
            print("\n===== 最终GPU内存统计 =====")
            print(f"峰值内存使用: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
            print(f"当前已分配内存: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
            print(f"当前保留内存: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
            
            # 清理缓存
            torch.cuda.empty_cache()
            print("GPU缓存已清理")
            print(f"清理缓存后 - 已分配内存: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        
        print("=" * 80)
        print("迭代训练完成，最终模型已训练完毕。")
        print("=" * 80)
    
    except Exception as e:
        logging.exception(f"主程序执行过程中出错: {str(e)}")
        print(f"错误: {str(e)}")
        print("请查看日志了解详细信息")
    
    finally:
        #--------------------------------------------------------------------
        # 12. 获取最终内存报告并清理
        #--------------------------------------------------------------------
        # 获取总体内存报告
        overall_memory_tracker.stop()
        stats = overall_memory_tracker.report()
        
        # 最终清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("最终GPU内存清理完成")
        
        # 确保日志正确刷新，恢复标准输出/错误
        logging.info("训练流程已完成")
        logging.shutdown()
        restore_stdout_stderr()

#============================================================================
# 程序入口
#============================================================================

if __name__ == "__main__":
    main()