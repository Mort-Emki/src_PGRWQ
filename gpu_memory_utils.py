import torch
import time
import threading
from typing import Optional, Callable
import os
import psutil
import numpy as np

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

def log_memory_usage(prefix=""):
    """Log both GPU and system memory usage."""
    sys_memory = get_system_memory_info()
    proc_memory_mb = get_process_memory_info()
    
    system_mem_str = f"System Memory: {sys_memory['used_gb']:.1f}GB / {sys_memory['total_gb']:.1f}GB ({sys_memory['percent']}%)"
    process_mem_str = f"Process Memory: {proc_memory_mb:.1f}MB"
    
    if torch.cuda.is_available():
        gpu_mem_str = get_gpu_memory_summary()
        print(f"{prefix}[MEMORY] {gpu_mem_str} | {process_mem_str} | {system_mem_str}")
    else:
        print(f"{prefix}[MEMORY] {process_mem_str} | {system_mem_str}")

class MemoryTracker:
    """Track memory usage over time with timestamps."""
    
    def __init__(self, interval_seconds=5.0, track_gpu=True, track_system=True):
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

def periodic_memory_check(interval_seconds=60):
    """Start a background thread to log memory status periodically."""
    def _check_thread():
        while True:
            log_memory_usage("[Periodic Check] ")
            time.sleep(interval_seconds)
    
    monitor_thread = threading.Thread(target=_check_thread, daemon=True)
    monitor_thread.start()
    print(f"Started periodic memory monitoring (interval: {interval_seconds}s)")
    return monitor_thread

# Context manager for timing and memory tracking
class TimingAndMemoryContext:
    def __init__(self, name="Operation", log_memory=True):
        self.name = name
        self.log_memory = log_memory
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.log_memory:
            log_memory_usage(f"[{self.name} START] ")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if self.log_memory:
            log_memory_usage(f"[{self.name} END] ")
        print(f"[TIMING] {self.name} completed in {duration:.2f} seconds")