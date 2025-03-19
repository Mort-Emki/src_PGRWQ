import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from .dataset import MyMultiBranchDataset

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
    def __init__(self, 
                 input_dim, 
                 hidden_size, 
                 num_layers, 
                 attr_dim, 
                 fc_dim, 
                 output_dim=1, 
                 use_attr=True):
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
                 attr_dim=None, fc_dim=32, device='cpu'):
        self.model_type = model_type
        self.device = device
        if self.model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            self.base_model = RandomForestRegressor(n_estimators=50, random_state=42)
        elif self.model_type == 'lstm':
            assert input_dim is not None and attr_dim is not None, "LSTM 模式下必须指定 input_dim 和 attr_dim"
            self.base_model = MultiBranchModel(input_dim=input_dim, hidden_size=hidden_size,
                                                num_layers=num_layers, attr_dim=attr_dim, fc_dim=fc_dim).to(device)
        else:
            raise ValueError("model_type 必须为 'rf' 或 'lstm'")

    def train_model(self, attr_dict, comid_arr_train, X_ts_train,  Y_train, 
                    comid_arr_val, X_ts_val=None,  Y_val=None, 
                    epochs=10, lr=1e-3, patience=3, batch_size=32):
        if self.model_type == 'rf':
            print('rf version is on the road')
            # print('model_type 为 rf')
            # N, T, D = X_ts.shape
            # X_ts_flat = X_ts.reshape(N, T * D)
            # X_combined = np.hstack([X_ts_flat, X_attr])
            # self.base_model.fit(X_combined, Y)
            # print('Random Forest training complete.')
        else:
            from torch.utils.data import DataLoader
            # Create training dataset and loader and val dataset and loader

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

            for ep in range(epochs):      
                self.base_model.train()
                total_loss = 0.0
                for x_ts_batch, x_attr_batch, y_batch in train_loader:
                    x_ts_batch = x_ts_batch.to(self.device, dtype=torch.float32)
                    x_attr_batch = x_attr_batch.to(self.device, dtype=torch.float32)
                    y_batch = y_batch.to(self.device, dtype=torch.float32)
                    optimizer.zero_grad()
                    preds = self.base_model(x_ts_batch, x_attr_batch)
                    loss = criterion(preds, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * x_ts_batch.size(0)
                avg_train_loss = total_loss / len(train_loader.dataset)

                # Validation step (if validation data is provided)
                if val_loader is not None:
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

                    # # Early stopping based on validation loss
                    # if avg_val_loss < best_val_loss:
                    #     best_val_loss = avg_val_loss
                    #     no_improve = 0
                    # else:
                    #     no_improve += 1
                    # if no_improve >= patience:
                    #     print("Early stopping triggered!")
                    #     break
                else:
                    print(f"[Epoch {ep+1}/{epochs}] Train MSE: {avg_train_loss:.4f}")


    def predict(self, X_ts, X_attr):
        """
        批量预测
        输入：
            X_ts: (N, T, input_dim) 时间序列数据
            X_attr: (N, attr_dim) 属性数据
        输出：
            返回预测结果数组
        """
        if self.model_type == 'rf':
            N, T, D = X_ts.shape
            X_ts_flat = X_ts.reshape(N, T * D)
            import numpy as np
            X_combined = np.hstack([X_ts_flat, X_attr])
            return self.base_model.predict(X_combined)
        else:
            import torch
            self.base_model.eval()
            X_ts_torch = torch.tensor(X_ts, dtype=torch.float32, device=self.device)
            X_attr_torch = torch.tensor(X_attr, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                preds = self.base_model(X_ts_torch, X_attr_torch)
            return preds.cpu().numpy()

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
            self.base_model.load_state_dict(torch.load(path))