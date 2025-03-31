"""
branch_lstm_model.py - 多分支LSTM模型实现

该模块包含用于水质预测的多分支LSTM模型实现，独立于其他模型类型。
所有模型共享标准化的接口，便于在主程序中灵活切换不同模型类型。
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from torch.utils.data import DataLoader, Dataset
from PGRWQI.model_training.gpu_memory_utils import log_memory_usage, TimingAndMemoryContext

# =============================================================================
# 数据集类
# =============================================================================

class MultiBranchDataset(Dataset):
    """
    多分支模型的数据集类
    
    用于处理时间序列特征+属性特征的混合输入数据
    """
    def __init__(self, X_ts, Y, comid_arr, attr_dict):
        """
        初始化数据集
        
        参数:
            X_ts: (N, T, input_dim) 时间序列数据
            Y: (N,) 目标标签
            comid_arr: (N,) 每个样本对应的河段ID
            attr_dict: { str(COMID): np.array([...]) } 河段的静态属性向量
        """
        self.X_ts = X_ts
        self.Y = Y
        self.comids = comid_arr
        self.attr_dict = attr_dict

    def __len__(self):
        """返回数据集大小"""
        return len(self.X_ts)

    def __getitem__(self, idx):
        """获取单个数据样本"""
        x_ts = self.X_ts[idx]
        y_val = self.Y[idx]
        comid_str = str(self.comids[idx])
        
        # 获取属性向量，若不存在则使用零向量
        if comid_str in self.attr_dict:
            x_attr = self.attr_dict[comid_str]
        else:
            x_attr = np.zeros_like(next(iter(self.attr_dict.values())))
            
        return x_ts, x_attr, y_val

# =============================================================================
# 评估指标
# =============================================================================

def calculate_nse(preds, targets):
    """
    计算纳什效率系数 (Nash–Sutcliffe Efficiency)
    
    参数:
        preds: 预测值张量
        targets: 目标值张量
        
    返回:
        NSE系数
    """
    mean_targets = targets.mean()
    numerator = torch.sum((preds - targets) ** 2)
    denominator = torch.sum((targets - mean_targets) ** 2)
    nse = 1 - numerator / denominator
    return nse

def mean_absolute_percentage_error(y_pred, y_true):
    """
    计算平均绝对百分比误差 (MAPE)
    
    参数:
        y_pred: 预测值张量
        y_true: 真实值张量
        
    返回:
        MAPE值
    """
    epsilon = 1e-6  # 防止除零错误
    return torch.mean(torch.abs((y_pred - y_true) / (y_true + epsilon))) * 100

# =============================================================================
# 网络模型定义
# =============================================================================

class MultiBranchLSTM(nn.Module):
    """
    多分支LSTM模型
    
    结合时间序列数据和静态属性数据，通过LSTM和MLP分支进行联合建模
    """
    def __init__(self, input_dim, hidden_size, num_layers, attr_dim, fc_dim, output_dim=1, use_attr=True):
        """
        初始化多分支LSTM模型
        
        参数:
            input_dim: 时间序列特征维度
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
            attr_dim: 属性数据维度
            fc_dim: 属性数据全连接层输出维度
            output_dim: 模型输出维度（默认1）
            use_attr: 是否使用属性数据
        """
        super().__init__()
        self.use_attr = use_attr

        # LSTM分支
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )

        if self.use_attr:
            # 属性分支：MLP
            self.attr_fc = nn.Sequential(
                nn.Linear(attr_dim, fc_dim),
                nn.ReLU(),
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU()
            )
            self.final_fc = nn.Linear(hidden_size + fc_dim, output_dim)
        else:
            self.final_fc = nn.Linear(hidden_size, output_dim)
        
        # 打印模型架构和参数数量
        print(f"模型架构初始化:")
        print(f" - LSTM: input_dim={input_dim}, hidden_size={hidden_size}, num_layers={num_layers}")
        if self.use_attr:
            print(f" - 属性网络: attr_dim={attr_dim}, fc_dim={fc_dim}")
        print(f" - 输出维度: {output_dim}")
        
        # 计算并打印参数数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"总参数量: {total_params:,}")

    def forward(self, x_ts, x_attr):
        """
        前向传播
        
        参数:
            x_ts: 时间序列输入, 形状为 (batch_size, seq_len, input_dim)
            x_attr: 属性输入, 形状为 (batch_size, attr_dim)
            
        返回:
            模型输出
        """
        lstm_out, _ = self.lstm(x_ts)
        ts_feat = lstm_out[:, -1, :]  # 取时间序列最后一时刻特征

        if self.use_attr:
            attr_feat = self.attr_fc(x_attr)
            combined = torch.cat([ts_feat, attr_feat], dim=1)
            out = self.final_fc(combined)
        else:
            out = self.final_fc(ts_feat)

        return out.squeeze(-1)

# =============================================================================
# 模型管理类
# =============================================================================

class BranchLSTMModel:
    """
    分支LSTM模型管理类
    
    提供模型创建、训练、预测和保存/加载功能
    """
    def __init__(self, input_dim, hidden_size=64, num_layers=1, attr_dim=20, 
                 fc_dim=32, output_dim=1, device='cuda', memory_check_interval=5):
        """
        初始化分支LSTM模型管理器
        
        参数:
            input_dim: 输入维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            attr_dim: 属性维度
            fc_dim: 全连接层维度
            output_dim: 输出维度
            device: 训练设备('cpu'或'cuda')
            memory_check_interval: 内存检查间隔(epochs)
        """
        self.device = device
        self.memory_check_interval = memory_check_interval
        
        # 记录初始内存状态
        if self.device == 'cuda':
            log_memory_usage("[模型初始化] ")
        
        # 创建模型
        self.model = MultiBranchLSTM(
            input_dim=input_dim, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            attr_dim=attr_dim, 
            fc_dim=fc_dim,
            output_dim=output_dim
        )
        
        # 将模型移动到指定设备
        self.model = self.model.to(self.device)
        print(f"模型已移动到{self.device}")
        
        # 在GPU上测试模型
        if self.device == 'cuda':
            self._test_model_on_device(input_dim, attr_dim)
        
        # 记录模型创建后的内存使用情况
        if self.device == 'cuda':
            log_memory_usage("[模型创建完成] ")
            
    def _test_model_on_device(self, input_dim, attr_dim):
        """
        测试模型是否正确加载到设备上
        
        参数:
            input_dim: 输入维度
            attr_dim: 属性维度
        """
        dummy_ts = torch.zeros((1, 10, input_dim), device=self.device)
        dummy_attr = torch.zeros((1, attr_dim), device=self.device)
        with torch.no_grad():
            _ = self.model(dummy_ts, dummy_attr)
        print(f"已在{self.device}上测试模型, 使用虚拟输入")
        
        # 打印每个参数的设备以确认
        print("参数设备:")
        for name, param in self.model.named_parameters():
            print(f" - {name}: {param.device}")

    def train(self, attr_dict, comid_arr_train, X_ts_train, Y_train, 
              comid_arr_val=None, X_ts_val=None, Y_val=None, 
              epochs=10, lr=1e-3, patience=3, batch_size=32):
        """
        训练模型
        
        参数:
            attr_dict: 属性字典
            comid_arr_train: 训练集COMID数组
            X_ts_train: 训练集时间序列数据
            Y_train: 训练集标签
            comid_arr_val: 验证集COMID数组
            X_ts_val: 验证集时间序列数据
            Y_val: 验证集标签
            epochs: 训练轮数
            lr: 学习率
            patience: 早停耐心值
            batch_size: 批次大小
            
        返回:
            训练历史记录
        """
        import torch.optim as optim
        
        # 创建数据集和数据加载器
        with TimingAndMemoryContext("创建数据集"):
            train_dataset = MultiBranchDataset(X_ts_train, Y_train, comid_arr_train, attr_dict)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            if X_ts_val is not None and Y_val is not None:
                val_dataset = MultiBranchDataset(X_ts_val, Y_val, comid_arr_val, attr_dict)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            else:
                val_loader = None
        
        # 初始化损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 初始化早停变量
        best_val_loss = float('inf')
        no_improve = 0
        history = {'train_loss': [], 'val_loss': [], 'val_mape': [], 'val_nse': []}

        # 按轮次进行训练
        for ep in range(epochs):
            # 记录轮次开始时的内存使用情况
            if self.device == 'cuda' and ep % self.memory_check_interval == 0:
                log_memory_usage(f"[轮次 {ep+1}/{epochs} 开始] ")
            
            # 训练阶段
            train_loss = self._train_epoch(train_loader, optimizer, criterion, ep)
            history['train_loss'].append(train_loss)
            
            # 验证阶段
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, criterion, ep)
                history['val_loss'].append(val_metrics['loss'])
                history['val_mape'].append(val_metrics['mape'])
                history['val_nse'].append(val_metrics['nse'])
                
                # 打印训练和验证指标
                self._print_epoch_metrics(ep, epochs, train_loss, val_metrics)
                
                # 早停检查
                current_val_loss = val_metrics['loss']
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve >= patience:
                    print("触发早停!")
                    
                    # 早停前的最终内存检查
                    if self.device == 'cuda':
                        log_memory_usage("[早停触发] ")
                    break
            else:
                print(f"[轮次 {ep+1}/{epochs}] 训练MSE: {train_loss:.4f}")
            
            # 清理轮次结束时的内存
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        # 训练完成后的最终内存使用情况
        if self.device == 'cuda':
            log_memory_usage("[训练完成] ")
            
        return history
    
    def _train_epoch(self, train_loader, optimizer, criterion, ep):
        """
        训练一个epoch
        
        参数:
            train_loader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数
            ep: 当前轮次
            
        返回:
            平均训练损失
        """
        with TimingAndMemoryContext(f"轮次 {ep+1} 训练", 
                                   log_memory=(ep % self.memory_check_interval == 0)):
            self.model.train()
            total_loss = 0.0
            
            for batch_idx, (x_ts_batch, x_attr_batch, y_batch) in enumerate(train_loader):
                # 记录首批次和定期的内存使用情况
                if self.device == 'cuda' and ep % self.memory_check_interval == 0 and batch_idx == 0:
                    log_memory_usage(f"[轮次 {ep+1} 首批次] ")
                
                # 将数据移动到设备
                x_ts_batch = x_ts_batch.to(self.device, dtype=torch.float32)
                x_attr_batch = x_attr_batch.to(self.device, dtype=torch.float32)
                y_batch = y_batch.to(self.device, dtype=torch.float32)
                
                # 前向传播, 计算损失和反向传播
                optimizer.zero_grad()
                preds = self.model(x_ts_batch, x_attr_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x_ts_batch.size(0)
                
                # 定期清理GPU缓存
                if self.device == 'cuda' and batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
            
            avg_train_loss = total_loss / len(train_loader.dataset)
            return avg_train_loss
    
    def _validate_epoch(self, val_loader, criterion, ep):
        """
        验证一个epoch
        
        参数:
            val_loader: 验证数据加载器
            criterion: 损失函数
            ep: 当前轮次
            
        返回:
            包含验证指标的字典
        """
        with TimingAndMemoryContext(f"轮次 {ep+1} 验证", 
                                   log_memory=(ep % self.memory_check_interval == 0)):
            self.model.eval()
            total_val_loss = 0.0
            total_val_mape = 0.0
            total_val_nse = 0.0

            with torch.no_grad():
                for x_ts_val, x_attr_val, y_val in val_loader:
                    # 将数据移动到设备
                    x_ts_val = x_ts_val.to(self.device, dtype=torch.float32)
                    x_attr_val = x_attr_val.to(self.device, dtype=torch.float32)
                    y_val = y_val.to(self.device, dtype=torch.float32)
                    
                    # 获取预测并计算指标
                    preds_val = self.model(x_ts_val, x_attr_val)
                    loss_val = criterion(preds_val, y_val)
                    total_val_loss += loss_val.item() * x_ts_val.size(0)
                    total_val_mape += mean_absolute_percentage_error(preds_val, y_val) * x_ts_val.size(0)
                    total_val_nse += calculate_nse(preds_val, y_val) * x_ts_val.size(0)

            # 计算平均指标
            dataset_size = len(val_loader.dataset)
            metrics = {
                'loss': total_val_loss / dataset_size,
                'mape': total_val_mape / dataset_size,
                'nse': total_val_nse / dataset_size
            }
            return metrics
    
    def _print_epoch_metrics(self, ep, epochs, train_loss, val_metrics):
        """
        打印轮次指标
        
        参数:
            ep: 当前轮次
            epochs: 总轮次
            train_loss: 训练损失
            val_metrics: 验证指标字典
        """
        print(f"轮次 [{ep+1}/{epochs}]")
        print(" 验证  | "
              f"训练损失={train_loss:.4f}, "
              f"验证损失={val_metrics['loss']:.4f}, "
              f"验证MAPE={val_metrics['mape']:.4f}, "
              f"验证NSE={val_metrics['nse']:.4f}")

    def predict(self, X_ts, X_attr):
        """
        批量预测
        
        参数:
            X_ts: 时间序列输入, 形状为(N, T, D)
            X_attr: 属性输入, 形状为(N, attr_dim)
            
        返回:
            预测结果
        """
        with TimingAndMemoryContext("批量预测"):
            self.model.eval()
            
            # 确保模型在正确的设备上
            if self.device == 'cuda':
                self.model = self.model.to(self.device)
            
            total_samples = X_ts.shape[0]
            
            # 计算合适的批处理大小
            batch_size = self._calculate_safe_batch_size(X_ts, X_attr)
            
            if self.device == 'cuda':
                log_memory_usage(f"[预测开始] 处理 {total_samples} 个样本")
            
            # 分批进行预测
            all_preds = []
            current_batch_size = batch_size
            
            i = 0
            while i < total_samples:
                try:
                    # 尝试使用当前批处理大小
                    end_idx = min(i + current_batch_size, total_samples)
                    
                    # 在正确的设备上创建张量
                    X_ts_torch = torch.tensor(X_ts[i:end_idx], dtype=torch.float32, device=self.device)
                    X_attr_torch = torch.tensor(X_attr[i:end_idx], dtype=torch.float32, device=self.device)
                    
                    # 获取预测
                    with torch.no_grad():
                        batch_preds = self.model(X_ts_torch, X_attr_torch)
                    
                    # 存储预测结果
                    all_preds.append(batch_preds.cpu().numpy())
                    
                    # 释放内存
                    del X_ts_torch
                    del X_attr_torch
                    torch.cuda.empty_cache()
                    
                    # 移动到下一批
                    i = end_idx
                    
                    # 定期记录进度
                    if i % (10 * current_batch_size) == 0 or i == total_samples:
                        print(f"已处理 {i}/{total_samples} 个样本 ({i/total_samples*100:.1f}%)")
                    
                except RuntimeError as e:
                    # 检查是否是内存不足错误
                    if "CUDA out of memory" in str(e):
                        # 减少批处理大小并重试
                        torch.cuda.empty_cache()
                        old_batch_size = current_batch_size
                        current_batch_size = max(10, current_batch_size // 2)
                        print(f"⚠️ 批处理大小 {old_batch_size} 内存不足。减小到 {current_batch_size}")
                        
                        # 如果批处理大小已经很小，可能存在其他问题
                        if current_batch_size < 100:
                            print("⚠️ 警告: 需要非常小的批处理大小。如果情况继续，请考虑使用CPU")
                    else:
                        # 不是内存错误，重新抛出
                        raise
            
            if len(all_preds) == 0:
                raise RuntimeError("未能成功处理任何批次")
                
            return np.concatenate(all_preds)
    
    def _calculate_safe_batch_size(self, X_ts, X_attr):
        """
        计算安全的批处理大小
        
        参数:
            X_ts: 时间序列输入
            X_attr: 属性输入
            
        返回:
            批处理大小
        """
        if torch.cuda.is_available():
            # 获取GPU规格
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # 单位: MB
            # 使用总内存的40%进行安全处理
            safe_memory_usage = total_memory * 0.40  # MB
            
            # 估计每个样本的内存(单位: MB)
            bytes_per_float = 4  # float32是4字节
            sample_size = X_ts.shape[1] * X_ts.shape[2] + X_attr.shape[1]
            bytes_per_sample = sample_size * bytes_per_float * 3  # 输入, 输出, 梯度
            mb_per_sample = bytes_per_sample / (1024**2)
            
            # 计算安全的批处理大小
            initial_batch_size = int(safe_memory_usage / mb_per_sample)
            batch_size = max(1000, initial_batch_size)
            print(f"起始批处理大小: {batch_size} (估计占用 {batch_size * mb_per_sample:.2f}MB)")
        else:
            batch_size = 1000
        
        return batch_size

    def predict_single(self, X_ts_single, X_attr_single):
        """
        对单个样本预测
        
        参数:
            X_ts_single: 时间序列数据, 形状为(T, input_dim)
            X_attr_single: 属性数据, 形状为(attr_dim,)
            
        返回:
            单个预测值
        """
        return self.predict(X_ts_single[None, :], X_attr_single[None, :])[0]

    def save(self, path):
        """
        保存模型
        
        参数:
            path: 模型保存路径
        """
        torch.save(self.model.state_dict(), path)
        
        # 保存后记录内存
        if self.device == 'cuda':
            log_memory_usage("[模型已保存] ")

    def load(self, path):
        """
        加载模型
        
        参数:
            path: 模型保存路径
        """
        with TimingAndMemoryContext("加载模型"):
            self.model.load_state_dict(torch.load(path))
            
            # 加载后记录内存
            if self.device == 'cuda':
                log_memory_usage("[模型已加载] ")

# =============================================================================
# 创建模型实例的工厂函数
# =============================================================================

def create_branchlstm_model(input_dim, hidden_size=64, num_layers=1, attr_dim=20, 
                            fc_dim=32, output_dim=1, device='cuda', memory_check_interval=5):
    """
    创建分支LSTM模型的工厂函数
    
    参数:
        input_dim: 输入维度
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        attr_dim: 属性维度
        fc_dim: 全连接层维度
        output_dim: 输出维度
        device: 训练设备('cpu'或'cuda')
        memory_check_interval: 内存检查间隔(epochs)
        
    返回:
        BranchLSTMModel实例
    """
    return BranchLSTMModel(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        attr_dim=attr_dim,
        fc_dim=fc_dim,
        output_dim=output_dim,
        device=device,
        memory_check_interval=memory_check_interval
    )