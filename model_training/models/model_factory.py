"""
model_factory.py - 模型工厂模块

提供统一的模型创建接口，根据类型动态加载不同的模型实现
"""

def create_model(model_type, **kwargs):
    """
    根据模型类型创建相应的模型实例
    
    参数:
        model_type: 模型类型 ('lstm', 'rf'等)
        **kwargs: 传递给模型构造函数的参数
        
    返回:
        模型实例
    """
    if model_type == 'lstm':
        from PGRWQI.model_training.models.BranchLstm import create_branch_lstm_model
        return create_branch_lstm_model(**kwargs)
    elif model_type == 'rf':
        from PGRWQI.model_training.models.RandomForest import create_random_forest_model
        return create_random_forest_model(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")