"""
logging_utils.py - 日志配置工具模块（修复版）

修复tqdm进度条与日志系统的兼容问题
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional

class TqdmCompatibleFormatter(logging.Formatter):
    """与tqdm兼容的日志格式化器"""
    
    def format(self, record):
        # 检查是否是tqdm相关的日志
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            msg = record.msg
            # 如果是进度条相关消息，降低其重要性
            if any(indicator in msg for indicator in ['%|', 'it/s', '\r']):
                # 将进度条相关的ERROR降级为DEBUG
                if record.levelno == logging.ERROR:
                    record.levelno = logging.DEBUG
                    record.levelname = 'DEBUG'
        
        return super().format(record)

class TqdmCompatibleStreamHandler(logging.StreamHandler):
    """与tqdm兼容的流处理器"""
    
    def emit(self, record):
        try:
            # 检查记录是否是进度条相关
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                msg = record.msg
                if any(indicator in msg for indicator in ['%|', 'it/s', '\r']):
                    # 进度条相关消息，只输出到console，不输出到文件
                    return
            
            # 使用tqdm的write方法以保持兼容性
            try:
                from tqdm import tqdm
                tqdm.write(self.format(record), file=self.stream)
            except ImportError:
                # 如果没有tqdm，使用标准输出
                super().emit(record)
        except Exception:
            self.handleError(record)

def ensure_dir_exists(path: str) -> str:
    """确保目录存在，如果不存在则创建
    
    参数:
        path: 目录路径
        
    返回:
        规范化的绝对路径
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

def setup_logging(log_dir: str = "logs", 
                 log_level: str = "INFO",
                 console_level: str = "INFO",
                 file_level: str = "DEBUG",
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True) -> logging.Logger:
    """
    设置日志系统（修复版，解决tqdm兼容问题）
    
    参数:
        log_dir: 日志文件目录
        log_level: 整体日志级别
        console_level: 控制台日志级别
        file_level: 文件日志级别
        enable_file_logging: 是否启用文件日志
        enable_console_logging: 是否启用控制台日志
        
    返回:
        配置好的logger
    """
    # 清除现有的handlers，避免重复
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # 设置根日志级别
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # 创建格式化器
    formatter = TqdmCompatibleFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 简化的控制台格式化器（用于进度条兼容）
    console_formatter = TqdmCompatibleFormatter(
        fmt='%(levelname)s - %(message)s'
    )
    
    # 添加文件处理器
    if enable_file_logging:
        try:
            # 确保日志目录存在
            log_dir = ensure_dir_exists(log_dir)
            
            # 创建日志文件名（包含时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"pgrwqi_{timestamp}.log"
            log_file_path = os.path.join(log_dir, log_filename)
            
            # 创建文件处理器
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            print(f"日志将保存到: {log_file_path}")
            
        except Exception as e:
            print(f"警告: 无法设置文件日志: {e}")
            enable_file_logging = False
    
    # 添加控制台处理器（使用tqdm兼容版本）
    if enable_console_logging:
        console_handler = TqdmCompatibleStreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # 配置特定logger的级别
    # 降低tqdm相关日志的级别
    tqdm_logger = logging.getLogger('tqdm')
    tqdm_logger.setLevel(logging.WARNING)
    
    # 降低一些第三方库的日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # 测试日志设置
    test_message = f"日志系统初始化完成 - 级别: {log_level}"
    if enable_console_logging:
        test_message += " | 控制台: 启用"
    if enable_file_logging:
        test_message += " | 文件: 启用"
    
    root_logger.info(test_message)
    
    return root_logger

# 备份和恢复标准输出/错误流的功能
_original_stdout = None
_original_stderr = None

def backup_stdout_stderr():
    """备份原始的stdout和stderr"""
    global _original_stdout, _original_stderr
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr

def restore_stdout_stderr():
    """恢复原始的stdout和stderr"""
    global _original_stdout, _original_stderr
    if _original_stdout is not None:
        sys.stdout = _original_stdout
    if _original_stderr is not None:
        sys.stderr = _original_stderr

# 自动备份标准流
backup_stdout_stderr()