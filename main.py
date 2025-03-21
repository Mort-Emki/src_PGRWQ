import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from data_processing import load_daily_data, load_river_attributes
from model_training.train import iterative_training_procedure
import os
import time
import sys
import logging
import datetime
from logging_utils import setup_logging, restore_stdout_stderr, ensure_dir_exists
from tqdm_logging import tqdm
import threading
import time
import datetime

# Import memory monitoring utilities or define fallback functions
try:
    from model_training.gpu_memory_utils import (
        log_memory_usage, 
        TimingAndMemoryContext, 
        MemoryTracker, 
        periodic_memory_check,
        get_gpu_memory_info
    )
except ImportError:
    # Create a minimal implementation of memory monitoring
    def log_memory_usage(prefix=""):
        """Log GPU memory usage with a prefix."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            print(f"{prefix}GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
    
    class TimingAndMemoryContext:
        """Context manager for timing and GPU memory tracking."""
        def __init__(self, name="Operation", log_memory=True):
            self.name = name
            self.log_memory = log_memory
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            if self.log_memory and torch.cuda.is_available():
                log_memory_usage(f"[{self.name} START] ")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            if self.log_memory and torch.cuda.is_available():
                log_memory_usage(f"[{self.name} END] ")
            print(f"[TIMING] {self.name} completed in {duration:.2f} seconds")
    
    def periodic_memory_check(interval_seconds=60):
        """Start periodic memory check in background thread."""
        import threading
        
        def _check_memory():
            while True:
                if torch.cuda.is_available():
                    log_memory_usage("[Periodic] ")
                time.sleep(interval_seconds)
        
        monitor_thread = threading.Thread(target=_check_memory, daemon=True)
        monitor_thread.start()
        print(f"Started periodic memory monitoring (interval: {interval_seconds}s)")
        return monitor_thread
    
    class MemoryTracker:
        """Simple memory tracker class."""
        def __init__(self, interval_seconds=10):
            self.interval = interval_seconds
        
        def start(self):
            log_memory_usage("[Memory Tracker Started] ")
        
        def stop(self):
            log_memory_usage("[Memory Tracker Stopped] ")
        
        def report(self):
            log_memory_usage("[Memory Report] ")
            return {}
    
    def get_gpu_memory_info():
        """Get GPU memory info as a dictionary."""
        if not torch.cuda.is_available():
            return {"available": False}
        
        # Get memory usage in bytes
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        
        # Convert to MB for readability
        allocated_mb = allocated / (1024 * 1024)
        reserved_mb = reserved / (1024 * 1024)
        max_allocated_mb = max_allocated / (1024 * 1024)
        
        # Get total memory of the GPU for percentage calculation
        device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
        total_memory = device_props.total_memory
        
        # Calculate percentage of usage
        usage_percent = (allocated / total_memory) * 100
        
        return {
            "allocated_mb": allocated_mb,
            "reserved_mb": reserved_mb,
            "max_allocated_mb": max_allocated_mb,
            "total_memory_mb": total_memory / (1024 * 1024),
            "usage_percent": usage_percent
        }
    
def create_memory_monitor_file(interval_seconds=30, log_dir="logs"):
    """Create a file to log memory usage at regular intervals."""
    import threading
    import time
    import datetime
    import os
    import logging
    import torch
    
    # Use absolute path for the log directory
    original_dir = os.getcwd()
    log_dir = os.path.abspath(log_dir)
    
    # Create directory if it doesn't exist
    try:
        os.makedirs(log_dir, exist_ok=True)  # exist_ok=True handles the case where the directory already exists
        logging.info(f"Created/verified directory for GPU memory logs: {log_dir}")
    except Exception as e:
        logging.error(f"Error creating directory {log_dir}: {str(e)}")
        # Use absolute path to current directory as fallback
        log_dir = original_dir
        logging.info(f"Using current directory for logs instead: {log_dir}")
    
    # Create log file with absolute path
    log_file = os.path.join(log_dir, "gpu_memory_log.csv")
    
    # Create or clear the file with headers
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,allocated_mb,reserved_mb,max_allocated_mb,percent_used\n")
        logging.info(f"Created GPU memory log file: {log_file}")
    except Exception as e:
        logging.error(f"Error creating GPU memory log file: {str(e)}")
        return None  # Return None if we can't create the file
    
    def _monitor_file():
        while True:
            try:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if torch.cuda.is_available():
                    info = get_gpu_memory_info()
                    try:
                        # Open with absolute path
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"{timestamp},{info['allocated_mb']:.2f},{info['reserved_mb']:.2f},"
                                   f"{info['max_allocated_mb']:.2f},{info['usage_percent']:.2f}\n")
                    except Exception as e:
                        logging.error(f"Error writing to GPU memory log ({log_file}): {str(e)}")
            except Exception as e:
                logging.error(f"Error in GPU memory monitoring: {str(e)}")
            
            # Sleep even if there was an error
            time.sleep(interval_seconds)
    
    file_monitor = threading.Thread(target=_monitor_file, daemon=True)
    file_monitor.start()
    logging.info(f"Started GPU memory logging to {log_file} (interval: {interval_seconds}s)")
    return file_monitor

def main():
    # Parse arguments first to get log_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--model_type", type=str, default="lstm", help="'rf' 或 'lstm'")
    parser.add_argument("--input_features", type=str, default="Feature1,Feature2,Feature3,Feature4,Feature5",
                        help="以逗号分隔的输入特征名称列表")
    parser.add_argument("--attr_features", type=str, default="Attr1,Attr2,Attr3",
                        help="以逗号分隔的属性特征名称列表")
    parser.add_argument("--memory_check_interval", type=int, default=30,
                        help="GPU 内存使用情况检查间隔（秒）")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="训练批次大小")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="日志保存目录")
    args = parser.parse_args()
    
    # Ensure log directory exists before any operation
    log_dir = ensure_dir_exists(args.log_dir)
    
    # Initialize logging system
    logger = setup_logging(log_dir=log_dir)
    
    # Log system information
    logging.info(f"PG-RWQ Training Pipeline Starting")
    logging.info(f"System time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"PyTorch version: {torch.__version__}")
    
    # Start GPU memory monitoring
    if torch.cuda.is_available():
        # Start periodic memory check in console
        periodic_monitor = periodic_memory_check(interval_seconds=args.memory_check_interval)
        # Start file-based memory logging
        file_monitor = create_memory_monitor_file(interval_seconds=args.memory_check_interval, 
                                                log_dir=log_dir)
        
        # Initial memory status
        log_memory_usage("[Initial GPU Status] ")


    # 设置数据根目录
    with TimingAndMemoryContext("Setting Working Directory"):
        os.chdir(r"D:\\PGRWQ\\data")

    # 将逗号分隔的字符串转换为列表
    input_features = [feat.strip() for feat in args.input_features.split(",") if feat.strip()]
    attr_features = [feat.strip() for feat in args.attr_features.split(",") if feat.strip()]

    # 手动指定输入特征列表和属性特征列表
    with TimingAndMemoryContext("Feature Definition"):
        input_features = ['surface_net_solar_radiation_mean', 'surface_pressure_mean', 'temperature_2m_mean', 
                         'u_component_of_wind_10m_mean', 'v_component_of_wind_10m_mean',
                         'volumetric_soil_water_layer_1_mean','volumetric_soil_water_layer_2_mean',
                         'temperature_2m_min','temperature_2m_max','total_precipitation_sum',
                         'potential_evaporation_sum','Qout']

        attr_features = [
            # 气候因子
            'pre_mm_syr',         # 年降水量，反映区域总体降水输入
            'pet_mean',           # 日均潜在蒸散，代表蒸发能力
            'aridity',            # 干旱指数，体现水分供需平衡
            'seasonality',        # 湿润指数的季节性，反映降水分布的不均性
            'high_prec_freq',     # 极端降水事件的频率，对面源污染具有脉冲效应

            # 土地利用
            'crp_pc_sse',         # 耕地比例，代表农业活动及化肥使用
            'for_pc_sse',         # 森林覆盖率，有助于截留和吸收营养盐
            'urb_pc_sse',         # 城市用地比例，反映城市化与生活污水排放
            'wet_pc_s01',         # 湿地比例，湿地具有调节营养盐的功能

            # 人类活动
            'nli_ix_sav',         # 夜间灯光指标，遥感反映人类活动密度
            'pop_ct_usu',         # 汇流口人口总数，直接关联生活污染

            # 水文因子
            'dis_m3_pyr',         # 年河流流量，决定稀释与传输过程
            'run_mm_syr',         # 地表径流深，指示面源污染传递能力

            # 土壤因子
            'cly_pc_sav',         # 土壤黏土比例，影响养分吸附与保留
            'soc_th_sav',         # 土壤有机碳含量，反映土壤微生物作用及营养盐固定

            # 地形因子
            'ele_mt_sav',         # 平均高程，表征区域地势特征
            'slp_dg_sav',         # 坡度，影响侵蚀和径流强度
            'sgr_dk_sav',         # 河道坡降，直接关联水流动力及输运效率

            # 补充因子
            'moisture_index',     # 湿润指数，综合反映区域水分状况
            'ero_kh_sav'          # 土壤侵蚀速率，影响沉积物及磷的输送
        ]

    # 根据指定列表自动调整 input_dim 和 attr_dim
    input_dim = len(input_features)
    attr_dim = len(attr_features)
    print(f"输入特征列表: {input_features} (维度: {input_dim})")
    print(f"属性特征列表: {attr_features} (维度: {attr_dim})")

    # 检查 GPU 是否可用
    with TimingAndMemoryContext("Device Setup"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备：{device}")
        
        if device == "cuda":
            # Log CUDA device information
            for i in range(torch.cuda.device_count()):
                device_properties = torch.cuda.get_device_properties(i)
                print(f"CUDA Device {i}: {device_properties.name}")
                print(f"  Total Memory: {device_properties.total_memory / (1024**3):.2f} GB")
                print(f"  CUDA Capability: {device_properties.major}.{device_properties.minor}")

    # 加载日尺度数据
    with TimingAndMemoryContext("Loading Daily Data"):
        daily_csv = "feature_daily_ts.csv"
        df = load_daily_data(daily_csv)
        print("日尺度数据基本信息：")
        print(f"  数据形状: {df.shape}")
        print(f"  列名: {df.columns.tolist()}")

    # 加载河段属性数据
    with TimingAndMemoryContext("Loading River Attributes"):
        attr_df = load_river_attributes("river_attributes_new.csv")
        print("\n河段属性数据基本信息：")
        print(f"  数据形状: {attr_df.shape}")
        print(f"  列名: {attr_df.columns.tolist()}")

    # 提取河段信息
    with TimingAndMemoryContext("Extracting River Network Info"):
        river_info = attr_df[['COMID', 'NextDownID']].copy()
        # 确保 NextDownID 为数字；若存在缺失值则填为 0
        river_info['NextDownID'] = pd.to_numeric(river_info['NextDownID'], errors='coerce').fillna(0).astype(int)
        
        # 加载COMID列表
        comid_wq_list = pd.read_csv("WQ_exist_comid.csv", header=None)[0].tolist()
        comid_era5_list = pd.read_csv("ERA5_exist_comid.csv", header=None)[0].tolist()

    # 调用迭代训练过程
    with TimingAndMemoryContext("Iterative Training Process"):
        final_model = iterative_training_procedure(
            df=df,
            attr_df=attr_df,
            input_features=input_features,
            attr_features=attr_features,
            river_info=river_info,
            target_cols=['TN','TP'],
            max_iterations=args.max_iterations,
            epsilon=args.epsilon,
            model_type=args.model_type,
            input_dim=input_dim,
            hidden_size=64,
            num_layers=1,
            attr_dim=attr_dim,
            fc_dim=32,
            device=device,
            comid_wq_list=comid_wq_list,
            comid_era5_list=comid_era5_list,
            input_cols=input_features
        )
    
    # Final memory report
    if torch.cuda.is_available():
        log_memory_usage("[Training Completed] ")
        
        # Report GPU memory statistics
        print("\n===== Final GPU Memory Statistics =====")
        print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
        print(f"Current memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"Current memory reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        print("GPU cache cleared")
        print(f"After clearing cache - Memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    
    print("迭代训练完成，最终模型已训练完毕。")

# Add a GPU memory logger context manager that can be used around key operations
class GPUMemoryLogger:
    """Context manager to log GPU memory usage before and after an operation."""
    def __init__(self, name):
        self.name = name
        self.start_memory = 0
        
    def __enter__(self):
        if torch.cuda.is_available():
            # Record starting memory
            self.start_memory = torch.cuda.memory_allocated()
            print(f"[{self.name}] Starting GPU memory: {self.start_memory / (1024**3):.4f} GB")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            # Calculate memory increase
            end_memory = torch.cuda.memory_allocated()
            diff = end_memory - self.start_memory
            print(f"[{self.name}] Ending GPU memory: {end_memory / (1024**3):.4f} GB")
            print(f"[{self.name}] Memory change: {diff / (1024**3):.4f} GB")
            
            # If there was a significant increase, additional details
            if diff > 1024**2:  # More than 1MB increase
                print(f"[{self.name}] Warning: Significant memory usage detected")

if __name__ == "__main__":
    try:
        # Initialize overall memory tracking
        overall_memory_tracker = MemoryTracker(interval_seconds=30)
        overall_memory_tracker.start()
        
        # Execute main function with memory monitoring
        with TimingAndMemoryContext("PGRWQ Training Pipeline"):
            main()
        
        # Get final memory report
        overall_memory_tracker.stop()
        stats = overall_memory_tracker.report()
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("Final GPU memory cleanup completed")
    
    except Exception as e:
        logging.exception(f"Error in main execution: {e}")
    
    finally:
        # Ensure logs are properly flushed and stdout/stderr are restored
        logging.info("Training process completed")
        logging.shutdown()
        restore_stdout_stderr()