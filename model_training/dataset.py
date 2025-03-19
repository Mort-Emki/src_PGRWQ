import numpy as np
from torch.utils.data import Dataset

class MyMultiBranchDataset(Dataset):
    """
    用于区域水质模型训练的多分支数据集
    输入：
        X_ts: (N, T, input_dim) 时间序列数据
        Y: (N,) 目标标签（例如 TP）
        comid_arr: (N,) 每个样本对应的河段 ID
        attr_dict: { str(COMID) : np.array([...]) } 河段的静态属性向量
    输出：
        返回一个 Dataset 对象，可用于 PyTorch DataLoader
    """
    def __init__(self, X_ts, Y, comid_arr, attr_dict):
        self.X_ts = X_ts
        self.Y = Y
        self.comids = comid_arr
        self.attr_dict = attr_dict

    def __len__(self):
        return len(self.X_ts)

    def __getitem__(self, idx):
        x_ts = self.X_ts[idx]
        y_val = self.Y[idx]
        comid_str = str(self.comids[idx])
        if comid_str in self.attr_dict:
            x_attr = self.attr_dict[comid_str]
        else:
            x_attr = np.zeros_like(next(iter(self.attr_dict.values())))
        return x_ts, x_attr, y_val
