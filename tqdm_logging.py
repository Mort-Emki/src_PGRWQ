"""
tqdm_logging.py - 自定义 tqdm 模块，支持与日志系统的兼容

修复进度条日志级别问题和重复输出问题
"""

import sys
import logging
from tqdm import tqdm as original_tqdm
from tqdm.auto import tqdm as auto_tqdm

class TqdmLoggingHandler(logging.Handler):
    """自定义日志处理器，与tqdm兼容"""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            # 使用tqdm的write方法，避免干扰进度条
            original_tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)

class TqdmToLogger:
    """将tqdm输出重定向到logger，并设置正确的日志级别"""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buf = ''
    
    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')
        if self.buf:
            # 只有在有实际内容时才记录日志，避免进度条的每次更新都被记录
            # 检查是否是进度条更新（包含百分比和进度条字符）
            if '%|' in self.buf or 'it/s' in self.buf:
                # 进度条更新，使用DEBUG级别而不是ERROR
                self.logger.log(logging.DEBUG, self.buf)
            else:
                # 其他输出，使用指定级别
                self.logger.log(self.level, self.buf)
    
    def flush(self):
        pass

class LoggingTqdm(original_tqdm):
    """支持日志的tqdm版本"""
    
    @staticmethod
    def write(s, file=None, end="\n", nolock=False):
        """重写write方法，避免重复输出到日志"""
        # 获取logger
        logger = logging.getLogger(__name__)
        
        # 检查是否是进度条相关输出
        if '%|' in s or 'it/s' in s:
            # 进度条输出，只输出到控制台，不记录到日志
            if hasattr(sys.stdout, 'write'):
                sys.stdout.write(s + end)
                sys.stdout.flush()
        else:
            # 非进度条输出，正常处理
            original_tqdm.write(s, file=file, end=end, nolock=nolock)
    
    def __init__(self, *args, **kwargs):
        # 设置输出到控制台而不是stderr，避免被日志捕获
        if 'file' not in kwargs:
            kwargs['file'] = sys.stdout
        
        # 禁用动态显示，避免日志污染
        if 'disable' not in kwargs:
            # 检查是否在调试模式
            import os
            if os.environ.get('PYTHONUNBUFFERED') or 'pytest' in sys.modules:
                kwargs['disable'] = False
            else:
                kwargs['disable'] = False
        
        # 设置进度条格式，使其更简洁
        if 'bar_format' not in kwargs:
            kwargs['bar_format'] = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        
        super().__init__(*args, **kwargs)

# 替换默认的tqdm
tqdm = LoggingTqdm

def setup_tqdm_logging():
    """设置tqdm与日志系统的兼容性"""
    # 获取root logger
    root_logger = logging.getLogger()
    
    # 添加自定义处理器
    tqdm_handler = TqdmLoggingHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    tqdm_handler.setFormatter(formatter)
    
    # 设置处理器只处理ERROR及以上级别，避免进度条的INFO消息被处理
    tqdm_handler.setLevel(logging.ERROR)
    
    # 检查是否已经添加了类似的处理器
    handler_exists = any(isinstance(h, TqdmLoggingHandler) for h in root_logger.handlers)
    if not handler_exists:
        root_logger.addHandler(tqdm_handler)
    
    # 重定向tqdm的stderr到logger
    logger = logging.getLogger('tqdm')
    logger.setLevel(logging.WARNING)  # 只记录WARNING及以上级别的tqdm消息
    
    return logger

# 自动设置
_tqdm_logger = setup_tqdm_logging()