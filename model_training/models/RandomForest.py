"""
random_forest.py - 随机森林模型实现

该模块继承CatchmentModel父类，提供随机森林模型的完整实现。
保留了父类定义的接口，并实现了所有必要的抽象方法。
"""

import numpy as np
import logging
import os
from PGRWQI.model_training.models.models import CatchmentModel
from PGRWQI.model_training.gpu_memory_utils import TimingAndMemoryContext

class RandomForestModel(CatchmentModel):
    """
    随机森林模型实现 - 继承自CatchmentModel基类
    
    提供随机森林模型的完整功能
    """
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, 
                 memory_check_interval=5):
        """
        初始化随机森林模型
        
        参数:
            n_estimators: 决策树数量
            max_depth: 决策树最大深度
            random_state: 随机种子
            memory_check_interval: 内存检查间隔
        """
        # 调用父类构造函数
        super(RandomForestModel, self).__init__(
            model_type='rf',
            device='cpu',  # 随机森林总是在CPU上运行
            memory_check_interval=memory_check_interval
        )
        
        # 随机森林特定参数
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # 初始化模型
        self._init_model()
    
    def _init_model(self):
        """初始化随机森林模型"""
        from sklearn.ensemble import RandomForestRegressor
        
        self.base_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1  # 使用所有可用CPU
        )
        
        print(f"随机森林模型已初始化: {self.n_estimators}棵树")
    
    def train_model(self, attr_dict, comid_arr_train, X_ts_train, Y_train, 
                   comid_arr_val=None, X_ts_val=None, Y_val=None, 
                   epochs=None, lr=None, patience=None, batch_size=None):
        """
        训练随机森林模型
        
        实现父类的抽象方法，但随机森林忽略神经网络特有的参数
        
        参数:
            attr_dict: 属性字典
            comid_arr_train: 训练集河段ID数组
            X_ts_train: 训练集时间序列特征
            Y_train: 训练集目标值
            其他参数: 为了保持接口一致性而保留，但在随机森林中被忽略
        """
        with TimingAndMemoryContext("随机森林训练"):
            # 处理输入特征
            N, T, D = X_ts_train.shape
            X_ts_flat = X_ts_train.reshape(N, T * D)  # 展平时间序列
            
            # 建立属性矩阵
            attr_dim = len(next(iter(attr_dict.values())))
            X_attr = np.zeros((N, attr_dim), dtype=np.float32)
            
            for i, comid in enumerate(comid_arr_train):
                comid_str = str(comid)
                if comid_str in attr_dict:
                    X_attr[i] = attr_dict[comid_str]
            
            # 合并时间序列和属性特征
            X_combined = np.hstack([X_ts_flat, X_attr])
            
            # 训练随机森林
            print(f"开始训练随机森林，输入维度: {X_combined.shape}")
            self.base_model.fit(X_combined, Y_train)
            print(f"随机森林训练完成，树数量: {self.n_estimators}")
    
    def predict(self, X_ts, X_attr):
        """
        批量预测
        
        实现父类的抽象方法，使用随机森林进行预测
        
        参数:
            X_ts: 时间序列输入, 形状为(N, T, D)
            X_attr: 属性输入, 形状为(N, attr_dim)
            
        返回:
            预测结果, 形状为(N,)
        """
        with TimingAndMemoryContext("随机森林预测"):
            # 展平时间序列
            N, T, D = X_ts.shape
            X_ts_flat = X_ts.reshape(N, T * D)
            
            # 合并特征
            X_combined = np.hstack([X_ts_flat, X_attr])
            
            # 批量预测
            return self.base_model.predict(X_combined)
    
    def save_model(self, path):
        """
        保存模型
        
        实现父类的抽象方法，保存随机森林模型
        
        参数:
            path: 保存路径
        """
        import joblib
        
        # 确保目录存在
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # 保存模型
        joblib.dump(self.base_model, path)
        print(f"随机森林模型已保存到 {path}")
    
    def load_model(self, path):
        """
        加载模型
        
        实现父类的抽象方法，加载随机森林模型
        
        参数:
            path: 模型路径
        """
        import joblib
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        with TimingAndMemoryContext("加载随机森林模型"):
            self.base_model = joblib.load(path)
            print(f"随机森林模型已从 {path} 加载")
    
    def get_model_info(self):
        """
        获取模型信息
        
        扩展父类方法，添加随机森林特有信息
        
        返回:
            包含模型信息的字典
        """
        info = super().get_model_info()
        info.update({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state
        })
        
        # 添加特征重要性（如果模型已训练）
        if hasattr(self.base_model, 'feature_importances_'):
            info["feature_importances"] = self.base_model.feature_importances_.tolist()
        
        return info

# =============================================================================
# 创建模型实例的工厂函数
# =============================================================================

def create_random_forest_model(n_estimators=100, max_depth=None, random_state=42, 
                               memory_check_interval=5):
    """
    创建随机森林模型的工厂函数
    
    参数:
        n_estimators: 决策树数量
        max_depth: 决策树最大深度
        random_state: 随机种子
        memory_check_interval: 内存检查间隔
        
    返回:
        RandomForestModel实例
    """
    return RandomForestModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        memory_check_interval=memory_check_interval
    )