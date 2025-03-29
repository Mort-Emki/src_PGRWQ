import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from .dataset import MyMultiBranchDataset
import time
import logging
from PGRWQI.model_training.gpu_memory_utils import log_memory_usage, TimingAndMemoryContext 
# Import memory monitoring utilities
# try:
#     from gpu_memory_utils import log_memory_usage, TimingAndMemoryContext
# except ImportError:
#     # Fallback implementation if the module is not available
#     def log_memory_usage(prefix=""):
#         if torch.cuda.is_available():
#             allocated = torch.cuda.memory_allocated() / (1024 * 1024)
#             reserved = torch.cuda.memory_reserved() / (1024 * 1024)
#             print(f"{prefix}GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
    
#     class TimingAndMemoryContext:
#         def __init__(self, name="Operation", log_memory=True):
#             self.name = name
#             self.log_memory = log_memory
#             self.start_time = None
        
#         def __enter__(self):
#             self.start_time = time.time()
#             if self.log_memory:
#                 log_memory_usage(f"[{self.name} START] ")
#             return self
        
#         def __exit__(self, exc_type, exc_val, exc_tb):
#             duration = time.time() - self.start_time
#             if self.log_memory:
#                 log_memory_usage(f"[{self.name} END] ")
#             print(f"[TIMING] {self.name} completed in {duration:.2f} seconds")

def calculate_nse(preds, targets):
    """Calculate the Nash–Sutcliffe efficiency (NSE)."""
    mean_targets = targets.mean()
    numerator = torch.sum((preds - targets) ** 2)
    denominator = torch.sum((targets - mean_targets) ** 2)
    nse = 1 - numerator / denominator
    return nse


def mean_absolute_percentage_error(y_pred, y_true):
    """
    MAPE: mean(|(y_pred - y_true) / y_true|) * 100
    """
    epsilon = 1e-6
    return torch.mean(torch.abs((y_pred - y_true) / (y_true + epsilon))) * 100


class MultiBranchModel(nn.Module):
    """
    多分支 LSTM 模型
    输入：
        input_dim: 时间序列特征维度
        hidden_size: LSTM 隐层维度
        num_layers: LSTM 层数
        attr_dim: 属性数据维度
        fc_dim: 属性数据全连接层输出维度
        output_dim: 模型输出维度（默认 1）
        use_attr: 是否使用属性数据
    输出：
        模型输出为一个标量
    """
    # In model_training/models.py, modify MultiBranchModel.__init__
    def __init__(self, input_dim, hidden_size, num_layers, attr_dim, fc_dim, output_dim=1, use_attr=True):
        super().__init__()
        self.use_attr = use_attr

        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )

        if self.use_attr:
            # Attribute branch: MLP
            self.attr_fc = nn.Sequential(
                nn.Linear(attr_dim, fc_dim),
                nn.ReLU(),
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU()
            )
            self.final_fc = nn.Linear(hidden_size + fc_dim, output_dim)
        else:
            self.final_fc = nn.Linear(hidden_size, output_dim)
        
        # Print model architecture and parameter count
        print(f"Model architecture initialized:")
        print(f" - LSTM: input_dim={input_dim}, hidden_size={hidden_size}, num_layers={num_layers}")
        if self.use_attr:
            print(f" - Attribute network: attr_dim={attr_dim}, fc_dim={fc_dim}")
        print(f" - Output dimension: {output_dim}")
        
        # Calculate and print parameter count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")

    def forward(self, x_ts, x_attr):
        lstm_out, _ = self.lstm(x_ts)
        ts_feat = lstm_out[:, -1, :]  # 取时间序列最后一时刻特征

        if self.use_attr:
            attr_feat = self.attr_fc(x_attr)
            combined = torch.cat([ts_feat, attr_feat], dim=1)
            out = self.final_fc(combined)
        else:
            out = self.final_fc(ts_feat)

        return out.squeeze(-1)

class CatchmentModel:
    """
    区域水质模型，支持 'rf'（随机森林）或 'lstm'
    输入：
        model_type: 'rf' 或 'lstm'
        input_dim, hidden_size, num_layers, attr_dim, fc_dim: 模型参数
        device: 训练设备
    输出：
        模型对象，提供 train_model、predict 和 predict_single 方法
    """
    def __init__(self, model_type='rf', input_dim=None, hidden_size=64, num_layers=1,
                attr_dim=None, fc_dim=32, device='cpu', memory_check_interval=5):
        self.model_type = model_type
        self.device = device
        self.memory_check_interval = memory_check_interval
        
        # Log initial memory state
        if self.device == 'cuda':
            log_memory_usage("[Model Init] ")
        
        if self.model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            self.base_model = RandomForestRegressor(n_estimators=50, random_state=42)
        elif self.model_type == 'lstm':
            assert input_dim is not None and attr_dim is not None, "LSTM 模式下必须指定 input_dim 和 attr_dim"
            # Create model
            self.base_model = MultiBranchModel(
                input_dim=input_dim, 
                hidden_size=hidden_size,
                num_layers=num_layers, 
                attr_dim=attr_dim, 
                fc_dim=fc_dim
            )
            
            # Move model to device and print confirmation
            self.base_model = self.base_model.to(device)
            print(f"Model moved to {device}")
            
            # Force a small tensor through the model to ensure it's on GPU
            if device == 'cuda':
                dummy_ts = torch.zeros((1, 10, input_dim), device=device)
                dummy_attr = torch.zeros((1, attr_dim), device=device)
                with torch.no_grad():
                    _ = self.base_model(dummy_ts, dummy_attr)
                print(f"Model tested on {device} with dummy input")
                
                # Print device for each parameter to confirm
                print("Parameter devices:")
                for name, param in self.base_model.named_parameters():
                    print(f" - {name}: {param.device}")
            
            # Log memory after model creation
            if self.device == 'cuda':
                log_memory_usage("[Model Created] ")
        else:
            raise ValueError("model_type 必须为 'rf' 或 'lstm'")

    def train_model(self, attr_dict, comid_arr_train, X_ts_train, Y_train, 
                    comid_arr_val, X_ts_val=None, Y_val=None, 
                    epochs=10, lr=1e-3, patience=3, batch_size=32):
        if self.model_type == 'rf':
            print('rf version is on the road')
            # RF implementation
        else:
            from torch.utils.data import DataLoader
            # Create training dataset and loader and val dataset and loader
            
            with TimingAndMemoryContext("Dataset Creation"):
                train_dataset = MyMultiBranchDataset(X_ts_train, Y_train, comid_arr_train, attr_dict)
                val_dataset = MyMultiBranchDataset(X_ts_val, Y_val, comid_arr_val, attr_dict)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            import torch.optim as optim
            import torch
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.base_model.parameters(), lr=lr)
            best_val_loss = float('inf')
            no_improve = 0

            # Track memory for each epoch
            for ep in range(epochs):
                # Log memory at start of epoch
                if self.device == 'cuda' and ep % self.memory_check_interval == 0:
                    log_memory_usage(f"[Epoch {ep+1}/{epochs} Start] ")
                
                # Training phase
                with TimingAndMemoryContext(f"Epoch {ep+1} Training", log_memory=(ep % self.memory_check_interval == 0)):
                    self.base_model.train()
                    total_loss = 0.0
                    
                    for batch_idx, (x_ts_batch, x_attr_batch, y_batch) in enumerate(train_loader):
                        # Log memory for first batch and periodically
                        if self.device == 'cuda' and ep % self.memory_check_interval == 0 and batch_idx == 0:
                            log_memory_usage(f"[Epoch {ep+1} First Batch] ")
                        
                        x_ts_batch = x_ts_batch.to(self.device, dtype=torch.float32)
                        x_attr_batch = x_attr_batch.to(self.device, dtype=torch.float32)
                        y_batch = y_batch.to(self.device, dtype=torch.float32)
                        
                        optimizer.zero_grad()
                        preds = self.base_model(x_ts_batch, x_attr_batch)
                        loss = criterion(preds, y_batch)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * x_ts_batch.size(0)
                        
                        # Clear GPU cache periodically 
                        if self.device == 'cuda' and batch_idx % 50 == 0:
                            torch.cuda.empty_cache()
                    
                    avg_train_loss = total_loss / len(train_loader.dataset)

                # Validation phase
                if val_loader is not None:
                    with TimingAndMemoryContext(f"Epoch {ep+1} Validation", log_memory=(ep % self.memory_check_interval == 0)):
                        self.base_model.eval()
                        total_val_loss = 0.0
                        total_val_mape = 0.0
                        total_val_nse = 0.0

                        with torch.no_grad():
                            for x_ts_val, x_attr_val, y_val in val_loader:
                                x_ts_val = x_ts_val.to(self.device, dtype=torch.float32)
                                x_attr_val = x_attr_val.to(self.device, dtype=torch.float32)
                                y_val = y_val.to(self.device, dtype=torch.float32)
                                preds_val = self.base_model(x_ts_val, x_attr_val)
                                loss_val = criterion(preds_val, y_val)
                                total_val_loss += loss_val.item() * x_ts_val.size(0)
                                total_val_mape += mean_absolute_percentage_error(preds_val, y_val) * x_ts_val.size(0)
                                total_val_nse += calculate_nse(preds_val, y_val) * x_ts_val.size(0)

                        avg_val_loss = total_val_loss / len(val_dataset)
                        avg_val_mape = total_val_mape / len(val_dataset)
                        avg_val_nse = total_val_nse / len(val_dataset)

                        print(f"Epoch [{ep+1}/{epochs}]")
                        print(" Validation  | "
                        f"Train Loss={avg_train_loss:.4f}, "
                        f"Val Loss={avg_val_loss:.4f}, "
                        f"Val MAPE={avg_val_mape:.4f}, "
                        f"Val NSE={avg_val_nse:.4f}")

                        # Early stopping check
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            no_improve = 0
                        else:
                            no_improve += 1
                        if no_improve >= patience:
                            print("Early stopping triggered!")
                            
                            # Final memory check before stopping
                            if self.device == 'cuda':
                                log_memory_usage("[Early Stopping] ")
                            break
                else:
                    print(f"[Epoch {ep+1}/{epochs}] Train MSE: {avg_train_loss:.4f}")
                
                # Clean up at the end of epoch
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

            # Final memory usage after training
            if self.device == 'cuda':
                log_memory_usage("[Training Complete] ")
    def predict(self, X_ts, X_attr):
        """Batch prediction with robust GPU memory handling"""
        with TimingAndMemoryContext("Batch Prediction"):
            if self.model_type == 'rf':
                # RF implementation remains the same
                N, T, D = X_ts.shape
                X_ts_flat = X_ts.reshape(N, T * D)
                X_combined = np.hstack([X_ts_flat, X_attr])
                return self.base_model.predict(X_combined)
            else:
                import torch
                self.base_model.eval()
                
                # Ensure model is on the right device
                if self.device == 'cuda':
                    self.base_model = self.base_model.to(self.device)
                
                total_samples = X_ts.shape[0]
                
                # Start with a conservative batch size
                if torch.cuda.is_available():
                    # Get GPU specs
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # in MB
                    # Start with 25% of total memory for safety
                    safe_memory_usage = total_memory * 0.25  # MB
                    
                    # Estimate memory per sample (in MB)
                    bytes_per_float = 4  # float32 is 4 bytes
                    sample_size = X_ts.shape[1] * X_ts.shape[2] + X_attr.shape[1]
                    bytes_per_sample = sample_size * bytes_per_float * 3  # Input, output, gradients
                    mb_per_sample = bytes_per_sample / (1024**2)
                    
                    # Calculate safe batch size
                    initial_batch_size = int(safe_memory_usage / mb_per_sample)
                    # batch_size = max(100, min(1000, initial_batch_size))  # Reasonable bounds
                    batch_size = max(1000,initial_batch_size)
                    print(f"Starting with batch size: {batch_size} (estimated {batch_size * mb_per_sample:.2f}MB)")
                else:
                    batch_size = 1000
                
                if self.device == 'cuda':
                    log_memory_usage(f"[Prediction Start] Processing {total_samples} samples")
                
                all_preds = []
                current_batch_size = batch_size
                
                i = 0
                while i < total_samples:
                    try:
                        # Try with current batch size
                        end_idx = min(i + current_batch_size, total_samples)
                        
                        # Create tensors on the right device
                        X_ts_torch = torch.tensor(X_ts[i:end_idx], dtype=torch.float32, device=self.device)
                        X_attr_torch = torch.tensor(X_attr[i:end_idx], dtype=torch.float32, device=self.device)
                        
                        # Get predictions
                        with torch.no_grad():
                            batch_preds = self.base_model(X_ts_torch, X_attr_torch)
                        
                        # Store predictions
                        all_preds.append(batch_preds.cpu().numpy())
                        
                        # Free memory
                        del X_ts_torch
                        del X_attr_torch
                        torch.cuda.empty_cache()
                        
                        # Move to next batch
                        i = end_idx
                        
                        # Log progress periodically
                        if i % (10 * current_batch_size) == 0 or i == total_samples:
                            print(f"Processed {i}/{total_samples} samples ({i/total_samples*100:.1f}%)")
                        
                    except RuntimeError as e:
                        # Check if this is an out-of-memory error
                        if "CUDA out of memory" in str(e):
                            # Reduce batch size and try again
                            torch.cuda.empty_cache()
                            old_batch_size = current_batch_size
                            current_batch_size = max(10, current_batch_size // 2)
                            print(f"⚠️ Out of memory with batch size {old_batch_size}. Reducing to {current_batch_size}")
                            
                            # If batch size is already very small, we might have other issues
                            if current_batch_size < 100:
                                print("⚠️ Warning: Very small batch size needed. Consider using CPU if this continues.")
                        else:
                            # Not a memory error, re-raise
                            raise
                
                if len(all_preds) == 0:
                    raise RuntimeError("Failed to process any batches successfully")
                    
                return np.concatenate(all_preds)

    def predict_single(self, X_ts_single, X_attr_single):
        """
        对单个样本预测
        输入：
            X_ts_single: (T, input_dim) 时间序列数据
            X_attr_single: (attr_dim,) 属性数据
        输出：
            返回单个预测值
        """
        return self.predict(X_ts_single[None, :], X_attr_single[None, :])[0]

    def save_model(self, path):
        """
        保存模型
        输入：
            path: 模型保存路径
        """
        if self.model_type == 'rf':
            import joblib
            joblib.dump(self.base_model, path)
        else:
            torch.save(self.base_model.state_dict(), path)
            
            # Log memory after saving
            if self.device == 'cuda':
                log_memory_usage("[Model Saved] ")

    def load_model(self, path):
        """
        加载模型
        输入：
            path: 模型保存路径
        """
        if self.model_type == 'rf':
            import joblib
            self.base_model = joblib.load(path)
        else:
            with TimingAndMemoryContext("Model Loading"):
                self.base_model.load_state_dict(torch.load(path))
                
                # Log memory after loading
                if self.device == 'cuda':
                    log_memory_usage("[Model Loaded] ")