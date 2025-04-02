import logging
import os
import sys
import time
from datetime import datetime

class StreamToLogger:
    """
    Custom stream object that redirects writes to a logger instance.
    This allows capturing of stdout/stderr output and sending it to both
    the console and a log file.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.buffer = ""
        self.original_stream = None

    def write(self, buf):
        if self.original_stream:
            self.original_stream.write(buf)
            self.original_stream.flush()
        
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
    
    def flush(self):
        if self.original_stream:
            self.original_stream.flush()

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {str(e)}")
            # Fallback to current directory if we can't create the specified one
            directory = "."
    return directory

def setup_logging(log_dir="logs", log_filename=None):
    """
    Set up logging configuration to write logs to both console and file.
    
    Args:
        log_dir: Directory to store log files
        log_filename: Custom log filename (default: generated from timestamp)
        
    Returns:
        Logger object
    """
    # Ensure logs directory exists
    log_dir = ensure_dir_exists(log_dir) 

    # Generate log filename if not provided
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"pgrwq_training_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_filename)
    
    # Configure root logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler with detailed formatting - use UTF-8 encoding
    file_handler = logging.FileHandler(log_path, 'w', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler with simple formatting
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create stdout/stderr redirectors
    stdout_logger = StreamToLogger(logger, logging.INFO)
    stdout_logger.original_stream = sys.stdout
    
    stderr_logger = StreamToLogger(logger, logging.ERROR)
    stderr_logger.original_stream = sys.stderr
    
    # Redirect standard outputs
    sys.stdout = stdout_logger
    sys.stderr = stderr_logger
    
    # Log initial information
    logger.info(f"Logging initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_path}")
    
    return logger

def restore_stdout_stderr():
    """Restore original stdout and stderr"""
    if hasattr(sys.stdout, 'original_stream') and sys.stdout.original_stream:
        sys.stdout = sys.stdout.original_stream
    
    if hasattr(sys.stderr, 'original_stream') and sys.stderr.original_stream:
        sys.stderr = sys.stderr.original_stream