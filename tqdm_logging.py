from tqdm import tqdm as original_tqdm
import logging

class TqdmLoggingHandler:
    """
    Helper class for tqdm to ensure its output is properly logged.
    """
    def __init__(self, logger=None, level=logging.INFO):
        self.logger = logger or logging.getLogger()
        self.level = level
        self.last_msg = ""

    def write(self, buf):
        buf = buf.strip()
        if buf and buf != self.last_msg:
            self.last_msg = buf
            self.logger.log(self.level, buf)
    
    def flush(self):
        pass

def tqdm(*args, **kwargs):
    """
    Wrapper for tqdm that ensures its output is properly captured in logs.
    Use this instead of the original tqdm when you want progress bars to be logged.
    """
    # Set default arguments for better logging behavior
    kwargs.setdefault('ascii', True)  # Use ASCII characters for better compatibility in logs
    kwargs.setdefault('ncols', 80)    # Fixed width for more consistent log output
    
    # Remove file parameter if it's already set to avoid conflicts
    if 'file' in kwargs:
        del kwargs['file']
    
    # Set miniters to update more frequently for smoother logging
    if 'miniters' not in kwargs:
        kwargs['miniters'] = 10
        
    # Get the logger
    logger = logging.getLogger()
    
    # Create a progress bar that writes to our custom handler
    return original_tqdm(*args, file=TqdmLoggingHandler(logger), **kwargs)