"""
model_factory.py - 模型工厂模块

提供统一的模型创建接口，根据类型动态加载不同的模型实现
"""
from typing import Dict, Any

def create_model(model_type: str, **kwargs):
    """
    Create a model instance based on model type
    
    Args:
        model_type: Model type ('lstm', 'rf', 'regression', etc.)
        **kwargs: Parameters to pass to model constructor
        
    Returns:
        Model instance
    """
    if model_type == 'lstm':
        from PGRWQI.model_training.models.BranchLstm import create_branch_lstm_model
        return create_branch_lstm_model(**kwargs)
    elif model_type == 'rf':
        from PGRWQI.model_training.models.RandomForest import create_random_forest_model
        return create_random_forest_model(**kwargs)
    elif model_type.startswith('regression'):
        from PGRWQI.model_training.models.RegressionModel import create_regression_model
        # Extract regression subtype if specified (e.g., 'regression_ridge')
        if '_' in model_type:
            reg_type = model_type.split('_')[1]
            return create_regression_model(reg_type=reg_type, **kwargs)
        else:
            # Default to linear regression
            return create_regression_model(**kwargs)
    elif model_type == 'informer':
        # Example of a new model type
        from PGRWQI.model_training.models.Informer import create_informer_model
        return create_informer_model(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")