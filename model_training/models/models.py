"""
models.py - 水质预测模型基类

该模块定义了完全通用的水质预测模型基类接口，
不包含任何具体模型实现，只提供通用接口和基础功能。
"""

import torch
import numpy as np
import logging
import time
from abc import ABC, abstractmethod

# 尝试导入GPU内存监控工具
try:
    from PGRWQI.model_training.gpu_memory_utils import log_memory_usage, TimingAndMemoryContext
except ImportError:
    # 创建简单的替代函数，以防模块不可用
    def log_memory_usage(prefix=""):
        """记录GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            print(f"{prefix}GPU内存：{allocated:.2f}MB已分配, {reserved:.2f}MB已保留")
    
    class TimingAndMemoryContext:
        """用于计时和内存监控的上下文管理器"""
        def __init__(self, name="操作", log_memory=True):
            self.name = name
            self.log_memory = log_memory
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            if self.log_memory and torch.cuda.is_available():
                log_memory_usage(f"[{self.name} 开始] ")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            if self.log_memory and torch.cuda.is_available():
                log_memory_usage(f"[{self.name} 结束] ")
            print(f"[计时] {self.name} 完成耗时 {duration:.2f} 秒")

class CatchmentModel(ABC):
    """
    区域水质模型基类
    
    提供纯抽象接口和最基本功能，所有具体模型通过继承实现。
    
    属性:
        model_type: 模型类型标识符
        device: 训练设备('cpu'或'cuda')
        memory_check_interval: 内存检查间隔
        base_model: 实际的模型实例
    """
    def __init__(self, model_type='base', device='cpu', memory_check_interval=5):
        """
        初始化水质模型基类
        
        参数:
            model_type: 模型类型标识符
            device: 训练设备('cpu'或'cuda')
            memory_check_interval: 内存检查间隔(单位:epochs)
        """
        self.model_type = model_type
        self.device = device
        self.memory_check_interval = memory_check_interval
        self.base_model = None
        
        # 记录初始内存状态
        if self.device == 'cuda':
            log_memory_usage("[模型初始化] ")

    @abstractmethod
    def train_model(self, attr_dict, comid_arr_train, X_ts_train, Y_train, 
                  comid_arr_val=None, X_ts_val=None, Y_val=None, 
                  epochs=10, lr=1e-3, patience=3, batch_size=32):
        """
        训练模型（抽象方法，子类必须实现）
        
        参数:
            attr_dict: 属性字典
            comid_arr_train: 训练集河段ID数组
            X_ts_train: 训练集时间序列特征
            Y_train: 训练集目标值
            comid_arr_val: 验证集河段ID数组
            X_ts_val: 验证集时间序列特征
            Y_val: 验证集目标值
            epochs: 训练轮数
            lr: 学习率
            patience: 早停耐心值
            batch_size: 批处理大小
        """
        pass

    @abstractmethod
    def predict(self, X_ts, X_attr):
        """
        批量预测（抽象方法，子类必须实现）
        
        参数:
            X_ts: 时间序列特征, 形状为(N, T, D)
            X_attr: 属性特征, 形状为(N, attr_dim)
            
        返回:
            预测结果, 形状为(N,)
        """
        pass

    def predict_single_sample(self, X_ts_single, X_attr_single):
        """
        对单个样本预测
        
        参数:
            X_ts_single: 单样本时间序列数据
            X_attr_single: 单样本属性数据
            
        返回:
            单个预测值
        """
        # 转为批处理形式，然后取第一个结果
        return self.predict(X_ts_single[None, :], X_attr_single[None, :])[0]

    @abstractmethod
    def save_model(self, path):
        """
        保存模型（抽象方法，子类必须实现）
        
        参数:
            path: 保存路径
        """
        pass

    @abstractmethod
    def load_model(self, path):
        """
        加载模型（抽象方法，子类必须实现）
        
        参数:
            path: 模型路径
        """
        pass

    def get_model_info(self):
        """获取模型基本信息"""
        info = {
            "model_type": self.model_type,
            "device": self.device
        }
        return info

    def _calculate_safe_batch_size(self, X_ts, X_attr, memory_fraction=0.25):
        """
        计算安全的批处理大小
        
        参数:
            X_ts: 时间序列输入
            X_attr: 属性输入
            memory_fraction: 使用可用GPU内存的比例
            
        返回:
            批处理大小
        """
        if torch.cuda.is_available():
            # 获取GPU规格
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # 单位: MB
            
            # 使用总内存的一部分进行安全处理
            # 这是为了:
            # 1. 避免内存溢出错误
            # 2. 预留空间应对内存碎片化
            # 3. 考虑系统和框架开销
            # 4. 为中间计算结果预留存储空间
            safe_memory_usage = total_memory * memory_fraction  # MB
            
            # 估计每个样本的内存(单位: MB)
            bytes_per_float = 4  # float32是4字节
            sample_size = X_ts.shape[1] * X_ts.shape[2] + X_attr.shape[1]
            bytes_per_sample = sample_size * bytes_per_float * 3  # 输入, 输出, 梯度
            mb_per_sample = bytes_per_sample / (1024**2)
            
            # 计算安全的批处理大小
            initial_batch_size = int(safe_memory_usage / mb_per_sample)
            batch_size = max(10, min(1000, initial_batch_size))  # 合理范围
            logging.info(f"起始批处理大小: {batch_size} (估计占用 {batch_size * mb_per_sample:.2f}MB)")
        else:
            batch_size = 1000
        
        return batch_size
    

    def _check_nan_in_input(self, X_ts, X_attr=None):
        """
        检查输入数据中是否包含NaN值，但不进行替换
        
        参数:
            X_ts: 时间序列输入, 形状为(N, T, D)
            X_attr: 属性输入, 形状为(N, attr_dim) （可选）
                
        返回:
            tuple: (has_nan, nan_info)
                has_nan: 布尔值，指示输入数据是否包含NaN
                nan_info: 字典，包含NaN值的详细统计信息
        """
        # 初始化结果
        has_nan = False
        nan_info = {
            'has_nan_in_ts': False,
            'has_nan_in_attr': False,
            'ts_nan_count': 0,
            'ts_nan_percent': 0.0,
            'attr_nan_count': 0,
            'attr_nan_percent': 0.0,
            'total_nan_count': 0,
            'total_nan_percent': 0.0
        }
        
        # 检查时间序列数据中的NaN
        ts_nan_mask = np.isnan(X_ts)
        ts_nan_count = np.sum(ts_nan_mask)
        ts_elements = X_ts.size
        
        if ts_nan_count > 0:
            has_nan = True
            nan_info['has_nan_in_ts'] = True
            nan_info['ts_nan_count'] = int(ts_nan_count)
            nan_info['ts_nan_percent'] = (ts_nan_count / ts_elements) * 100
            
            # 可以进一步获取NaN出现的位置
            nan_indices = np.where(ts_nan_mask)
            nan_info['ts_nan_locations'] = list(zip(*nan_indices))
            
            # 在日志中记录NaN信息
            logging.warning(f"时间序列输入包含 {ts_nan_count} 个NaN值 ({nan_info['ts_nan_percent']:.2f}%)")
        
        # 检查属性数据中的NaN（如果提供）
        if X_attr is not None:
            attr_nan_mask = np.isnan(X_attr)
            attr_nan_count = np.sum(attr_nan_mask)
            attr_elements = X_attr.size
            
            if attr_nan_count > 0:
                has_nan = True
                nan_info['has_nan_in_attr'] = True
                nan_info['attr_nan_count'] = int(attr_nan_count)
                nan_info['attr_nan_percent'] = (attr_nan_count / attr_elements) * 100
                
                # 可以进一步获取NaN出现的位置
                attr_nan_indices = np.where(attr_nan_mask)
                nan_info['attr_nan_locations'] = list(zip(*attr_nan_indices))
                
                # 在日志中记录NaN信息
                logging.warning(f"属性输入包含 {attr_nan_count} 个NaN值 ({nan_info['attr_nan_percent']:.2f}%)")
        
        # 计算总体NaN统计信息
        if has_nan:
            total_elements = ts_elements + (X_attr.size if X_attr is not None else 0)
            total_nan_count = ts_nan_count + (attr_nan_count if X_attr is not None else 0)
            nan_info['total_nan_count'] = int(total_nan_count)
            nan_info['total_nan_percent'] = (total_nan_count / total_elements) * 100
            
            logging.warning(f"输入数据总共包含 {total_nan_count} 个NaN值 ({nan_info['total_nan_percent']:.2f}%)")
        
        return has_nan, nan_info
    
    def predict_with_input_check(self, X_ts, X_attr=None, deal_nan=False):
        """
        带输入检查的预测函数，包装了子类实现的predict方法
        
        参数:
            X_ts: 时间序列特征, 形状为(N, T, D)
            X_attr: 属性特征, 形状为(N, attr_dim)
                
        返回:
            预测结果, 形状为(N,)
            """
        # 检查输入数据是否包含NaN值
        has_nan, nan_info = self._check_nan_in_input(X_ts, X_attr)
        
        if (has_nan)&(deal_nan):
            # 清理NaN值
            X_ts = np.nan_to_num(X_ts, nan=0.0)
            if X_attr is not None:
                X_attr = np.nan_to_num(X_attr, nan=0.0)
            logging.info(f"已清理NaN值，总共 {nan_info['total_nan_count']} 个，占比 {nan_info['total_nan_percent']:.2f}%")

        # 预测
        return self.predict(X_ts, X_attr)
    