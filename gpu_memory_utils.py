import torch
import time
import threading
from typing import Optional, Callable
import os
import psutil
import numpy as np
import logging

# Global verbosity setting for memory logging
MEMORY_LOG_VERBOSITY = 1  # 0: minimal, 1: normal, 2: verbose
MONITORING_ENABLED = True  # Global switch to enable/disable all monitoring

def set_memory_log_verbosity(level: int):
    """Set the verbosity level for memory logging.
    
    Args:
        level: 0 for minimal logging, 1 for normal, 2 for verbose
    """
    global MEMORY_LOG_VERBOSITY
    MEMORY_LOG_VERBOSITY = level
    logging.info(f"Memory logging verbosity set to {level}")

def set_monitoring_enabled(enabled: bool):
    """Enable or disable all monitoring globally.
    
    Args:
        enabled: True to enable monitoring, False to disable all monitoring
    """
    global MONITORING_ENABLED
    MONITORING_ENABLED = enabled
    status = "enabled" if enabled else "disabled"
    logging.info(f"Performance monitoring {status}")
    print(f"Performance monitoring {status}")

def get_gpu_memory_info():
    """Return GPU memory usage in a human-readable format."""
    if not torch.cuda.is_available():
        return "GPU not available"
    
    # Get memory usage in bytes for the current device
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    
    # Convert to MB for readability
    allocated_mb = allocated / (1024 * 1024)
    reserved_mb = reserved / (1024 * 1024)
    max_allocated_mb = max_allocated / (1024 * 1024)
    
    # Get total memory of the GPU
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_memory_mb = device_props.total_memory / (1024 * 1024)
    
    # Calculate percentage of usage
    usage_percent = (allocated / device_props.total_memory) * 100
    
    return {
        "allocated_mb": allocated_mb,
        "reserved_mb": reserved_mb,
        "max_allocated_mb": max_allocated_mb,
        "total_memory_mb": total_memory_mb,
        "usage_percent": usage_percent
    }

def get_gpu_memory_summary():
    """Return formatted GPU memory summary string."""
    if not torch.cuda.is_available():
        return "GPU not available"
    
    info = get_gpu_memory_info()
    return (f"GPU Memory: {info['allocated_mb']:.1f}MB / {info['total_memory_mb']:.1f}MB "
            f"({info['usage_percent']:.1f}%) | Peak: {info['max_allocated_mb']:.1f}MB")

def get_process_memory_info():
    """Get the current process memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    return memory_mb

def get_system_memory_info():
    """Get system-wide memory usage."""
    memory = psutil.virtual_memory()
    return {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "percent": memory.percent
    }

def log_memory_usage(prefix="", level=1):
    """Log both GPU and system memory usage if verbosity level is high enough.
    
    Args:
        prefix: String prefix for the log message
        level: Minimum verbosity level required to log this message
    """
    # Skip logging if monitoring is disabled or verbosity is lower than required level
    if not MONITORING_ENABLED or MEMORY_LOG_VERBOSITY < level:
        return
    
    sys_memory = get_system_memory_info()
    proc_memory_mb = get_process_memory_info()
    
    system_mem_str = f"System Memory: {sys_memory['used_gb']:.1f}GB / {sys_memory['total_gb']:.1f}GB ({sys_memory['percent']}%)"
    process_mem_str = f"Process Memory: {proc_memory_mb:.1f}MB"
    
    if torch.cuda.is_available():
        gpu_mem_str = get_gpu_memory_summary()
        
        # Only log to file for level >= 1, but print to console at all levels
        if level <= 1:
            logging.info(f"{prefix}[MEMORY] {gpu_mem_str} | {process_mem_str} | {system_mem_str}")
        print(f"{prefix}[MEMORY] {gpu_mem_str} | {process_mem_str} | {system_mem_str}")
    else:
        if level <= 1:
            logging.info(f"{prefix}[MEMORY] {process_mem_str} | {system_mem_str}")
        print(f"{prefix}[MEMORY] {process_mem_str} | {system_mem_str}")

class MemoryTracker:
    """Track memory usage over time with timestamps."""
    
    def __init__(self, interval_seconds=30.0, track_gpu=True, track_system=True):
        self.interval = interval_seconds
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.track_system = track_system
        self.tracking = False
        self.thread = None
        self.history = {
            "timestamps": [],
            "gpu_allocated_mb": [],
            "gpu_reserved_mb": [],
            "system_used_gb": [],
            "process_memory_mb": []
        }
    
    def start(self):
        """Start tracking memory usage in a background thread."""
        # Check if monitoring is globally disabled
        if not MONITORING_ENABLED:
            print("Memory tracking disabled (monitoring is turned off)")
            return
            
        if self.tracking:
            print("Memory tracking already running.")
            return
        
        self.tracking = True
        self.thread = threading.Thread(target=self._track_memory, daemon=True)
        self.thread.start()
        print(f"Memory tracking started (interval: {self.interval}s)")
    
    def stop(self):
        """Stop the memory tracking thread."""
        self.tracking = False
        if self.thread:
            self.thread.join(timeout=self.interval + 1)
            self.thread = None
        print("Memory tracking stopped")
    
    def _track_memory(self):
        """Background thread to periodically record memory usage."""
        start_time = time.time()
        
        while self.tracking:
            current_time = time.time() - start_time
            self.history["timestamps"].append(current_time)
            
            # Track process memory
            self.history["process_memory_mb"].append(get_process_memory_info())
            
            # Track GPU memory if available
            if self.track_gpu:
                mem_info = get_gpu_memory_info()
                self.history["gpu_allocated_mb"].append(mem_info["allocated_mb"])
                self.history["gpu_reserved_mb"].append(mem_info["reserved_mb"])
            else:
                self.history["gpu_allocated_mb"].append(None)
                self.history["gpu_reserved_mb"].append(None)
            
            # Track system memory
            if self.track_system:
                sys_mem = get_system_memory_info()
                self.history["system_used_gb"].append(sys_mem["used_gb"])
            else:
                self.history["system_used_gb"].append(None)
            
            time.sleep(self.interval)
    
    def get_statistics(self):
        """Get summary statistics of the tracked memory usage."""
        stats = {}
        
        if self.track_gpu:
            gpu_allocated = np.array(self.history["gpu_allocated_mb"])
            stats["gpu_mean_mb"] = np.mean(gpu_allocated)
            stats["gpu_max_mb"] = np.max(gpu_allocated)
            stats["gpu_min_mb"] = np.min(gpu_allocated)
            stats["gpu_std_mb"] = np.std(gpu_allocated)
        
        process_mem = np.array(self.history["process_memory_mb"])
        stats["process_mean_mb"] = np.mean(process_mem)
        stats["process_max_mb"] = np.max(process_mem)
        
        return stats
    
    def report(self):
        """Print a summary report of memory usage."""
        stats = self.get_statistics()
        
        print("\n===== Memory Usage Report =====")
        if self.track_gpu and torch.cuda.is_available():
            print(f"GPU Memory (Allocated):")
            print(f"  Max: {stats['gpu_max_mb']:.1f} MB")
            print(f"  Mean: {stats['gpu_mean_mb']:.1f} MB")
            print(f"  Min: {stats['gpu_min_mb']:.1f} MB")
            print(f"  Std Dev: {stats['gpu_std_mb']:.1f} MB")
        
        print(f"Process Memory:")
        print(f"  Max: {stats['process_max_mb']:.1f} MB")
        print(f"  Mean: {stats['process_mean_mb']:.1f} MB")
        print("================================")
        
        return stats

def periodic_memory_check(interval_seconds=300):
    """Start a background thread to log memory status periodically.
    
    Args:
        interval_seconds: Interval between checks in seconds (default: 300s = 5 minutes)
    """
    # If monitoring is disabled, return a dummy thread object
    if not MONITORING_ENABLED:
        logging.info("Periodic memory monitoring disabled (monitoring is turned off)")
        return None
    
    def _check_thread():
        while MONITORING_ENABLED:  # Check the global flag
            log_memory_usage("[Periodic Check] ", level=1)
            time.sleep(interval_seconds)
    
    monitor_thread = threading.Thread(target=_check_thread, daemon=True)
    monitor_thread.start()
    print(f"Started periodic memory monitoring (interval: {interval_seconds}s)")
    return monitor_thread

# Context manager for timing and memory tracking
class TimingAndMemoryContext:
    def __init__(self, name="Operation", log_memory=True, memory_log_level=1):
        self.name = name
        self.log_memory = log_memory
        self.memory_log_level = memory_log_level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        # Only log memory at start for important operations (level 1) or if verbosity is high
        if self.log_memory and MONITORING_ENABLED:
            log_memory_usage(f"[{self.name} START] ", level=self.memory_log_level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        # Log memory at end of operations if monitoring is enabled
        if self.log_memory and MONITORING_ENABLED:
            log_memory_usage(f"[{self.name} END] ", level=self.memory_log_level)
        
        # Always log timing information if monitoring is enabled, otherwise skip it
        if MONITORING_ENABLED:
            if MEMORY_LOG_VERBOSITY >= 1 or self.memory_log_level <= 1:
                logging.info(f"[TIMING] {self.name} completed in {duration:.2f} seconds")
            print(f"[TIMING] {self.name} completed in {duration:.2f} seconds")


def get_safe_batch_size(input_shape, attr_shape, memory_fraction=0.25):
    """Calculate a safe batch size given input shapes and desired memory fraction"""
    if not torch.cuda.is_available():
        return 1000  # Default for CPU
        
    # Get GPU specs
    total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    safe_memory_mb = total_memory_mb * memory_fraction
    
    # Calculate memory per sample
    ts_size = np.prod(input_shape[1:])  # Time steps × features
    total_elements = ts_size + attr_shape[1]
    bytes_per_sample = total_elements * 4 * 3  # float32 × input/output/gradients
    mb_per_sample = bytes_per_sample / (1024**2)
    
    # Calculate batch size with safety margin
    batch_size = int(safe_memory_mb / mb_per_sample)
    return max(10, min(10000, batch_size))  # Reasonable bounds

def force_cuda_memory_cleanup():
    """Aggressive GPU memory cleanup"""
    if torch.cuda.is_available():
        # Release all memory
        torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Force CUDA synchronization
        torch.cuda.synchronize()
        
        # Log current memory state if monitoring is enabled and verbosity is high enough
        if MONITORING_ENABLED and MEMORY_LOG_VERBOSITY >= 1:
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"After cleanup: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")

def diagnose_model_device(model):
    """Diagnose if a PyTorch model is correctly on GPU."""
    if not hasattr(model, 'parameters'):
        print("Not a PyTorch model with parameters")
        return False
    
    device_counts = {}
    for name, param in model.named_parameters():
        device = param.device
        device_counts[str(device)] = device_counts.get(str(device), 0) + 1
    
    print(f"Model parameter device distribution:")
    for device, count in device_counts.items():
        print(f"  - {device}: {count} parameters")
    
    # Check if all parameters are on the same device
    is_consistent = len(device_counts) == 1
    print(f"Model has consistent device: {is_consistent}")
    
    # Check if model is on GPU
    is_on_gpu = 'cuda' in next(iter(device_counts.keys()))
    print(f"Model is on GPU: {is_on_gpu}")
    
    return is_on_gpu and is_consistent