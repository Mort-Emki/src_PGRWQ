"""
regression_model.py - Linear and regularized regression model implementations

This module provides regression model implementations for the PG-RWQ framework.
It includes Linear Regression, Ridge Regression, Lasso Regression, and ElasticNet options.
These models are simpler alternatives to the more complex LSTM or RF models,
particularly useful for interpretability and when data is limited.
"""

import numpy as np
import logging
import os
import pickle
from PGRWQI.model_training.models.models import CatchmentModel
from PGRWQI.model_training.gpu_memory_utils import TimingAndMemoryContext

class RegressionModel(CatchmentModel):
    """
    Regression model implementation - inherits from CatchmentModel base class
    
    Provides implementations of Linear, Ridge, Lasso, and ElasticNet regression models
    """
    def __init__(self, reg_type='linear', alpha=1.0, l1_ratio=0.5, 
                 input_dim=None, attr_dim=None, memory_check_interval=5):
        """
        Initialize regression model
        
        Parameters:
            reg_type: Type of regression ('linear', 'ridge', 'lasso', 'elasticnet')
            alpha: Regularization strength (for Ridge, Lasso and ElasticNet)
            l1_ratio: L1 ratio for ElasticNet (0 = Ridge, 1 = Lasso)
            input_dim: Input feature dimension (for time series)
            attr_dim: Attribute feature dimension
            memory_check_interval: Memory check interval
        """
        # Call parent constructor
        super(RegressionModel, self).__init__(
            model_type=f'regression_{reg_type}',
            device='cpu',  # Regression models run on CPU
            memory_check_interval=memory_check_interval
        )
        
        # Store regression parameters
        self.reg_type = reg_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.input_dim = input_dim
        self.attr_dim = attr_dim
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the appropriate regression model based on reg_type"""
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        
        print(f"Initializing {self.reg_type} regression model")
        
        if self.reg_type == 'linear':
            self.base_model = LinearRegression(fit_intercept=True)
        elif self.reg_type == 'ridge':
            self.base_model = Ridge(alpha=self.alpha)
        elif self.reg_type == 'lasso':
            self.base_model = Lasso(alpha=self.alpha)
        elif self.reg_type == 'elasticnet':
            self.base_model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
        else:
            raise ValueError(f"Unknown regression type: {self.reg_type}")
            
        # Feature names for better interpretability (will be set during training)
        self.feature_names = None
        
        # Store preprocessing statistics
        self.feature_means = None
        self.feature_stds = None
    
    def train_model(self, attr_dict, comid_arr_train, X_ts_train, Y_train, 
                   comid_arr_val=None, X_ts_val=None, Y_val=None, 
                   feature_names=None, standardize=True, **kwargs):
        """
        Train regression model
        
        Parameters:
            attr_dict: Dictionary mapping COMID to attribute features
            comid_arr_train: Array of COMIDs for training samples
            X_ts_train: Time series training features [N, T, D]
            Y_train: Training labels [N]
            comid_arr_val: Array of COMIDs for validation samples (optional)
            X_ts_val: Time series validation features (optional)
            Y_val: Validation labels (optional)
            feature_names: Names of features for interpretation (optional)
            standardize: Whether to standardize features (default: True)
            **kwargs: Additional params (ignored for regression models)
        """
        with TimingAndMemoryContext("Regression model training"):
            # Process input features
            N, T, D = X_ts_train.shape
            
            # Flatten time series features
            X_ts_flat = X_ts_train.reshape(N, T * D)
            
            # Build attribute matrix
            if attr_dict:
                attr_dim = len(next(iter(attr_dict.values())))
                X_attr = np.zeros((N, attr_dim), dtype=np.float32)
                
                for i, comid in enumerate(comid_arr_train):
                    comid_str = str(comid)
                    if comid_str in attr_dict:
                        X_attr[i] = attr_dict[comid_str]
                
                # Combine time series and attribute features
                X_combined = np.hstack([X_ts_flat, X_attr])
            else:
                X_combined = X_ts_flat
            
            # Store feature names if provided
            if feature_names:
                # Expand for time series features
                ts_names = []
                for t in range(T):
                    for d in range(D):
                        ts_names.append(f"t-{T-t}_{d}")
                
                # Add attribute names if available
                if attr_dict:
                    self.feature_names = ts_names + feature_names
                else:
                    self.feature_names = ts_names
            
            # Standardize features if requested
            if standardize:
                # Compute mean and std
                self.feature_means = np.mean(X_combined, axis=0)
                self.feature_stds = np.std(X_combined, axis=0)
                
                # Replace zero std with 1 to avoid division by zero
                self.feature_stds[self.feature_stds == 0] = 1.0
                
                # Standardize
                X_combined = (X_combined - self.feature_means) / self.feature_stds
            
            # Train the model
            print(f"Training {self.reg_type} regression model with {X_combined.shape[1]} features")
            self.base_model.fit(X_combined, Y_train)
            
            # Print coefficients for interpretability
            if hasattr(self.base_model, 'coef_'):
                print(f"\nModel coefficients (top 10 by magnitude):")
                coefs = self.base_model.coef_
                
                # Get indices of top coefficients by magnitude
                top_indices = np.argsort(np.abs(coefs))[-10:][::-1]
                
                # Print them with feature names if available
                if self.feature_names and len(self.feature_names) == len(coefs):
                    for idx in top_indices:
                        print(f"  {self.feature_names[idx]}: {coefs[idx]:.6f}")
                else:
                    for idx in top_indices:
                        print(f"  Feature {idx}: {coefs[idx]:.6f}")
                
                if hasattr(self.base_model, 'intercept_'):
                    print(f"  Intercept: {self.base_model.intercept_:.6f}")
            
            # Evaluate on training set
            train_pred = self.base_model.predict(X_combined)
            train_mse = np.mean((train_pred - Y_train) ** 2)
            train_mae = np.mean(np.abs(train_pred - Y_train))
            print(f"Training MSE: {train_mse:.6f}, MAE: {train_mae:.6f}")
            
            # Evaluate on validation set if provided
            if X_ts_val is not None and Y_val is not None:
                N_val, _, _ = X_ts_val.shape
                X_ts_val_flat = X_ts_val.reshape(N_val, T * D)
                
                if attr_dict:
                    X_attr_val = np.zeros((N_val, attr_dim), dtype=np.float32)
                    for i, comid in enumerate(comid_arr_val):
                        comid_str = str(comid)
                        if comid_str in attr_dict:
                            X_attr_val[i] = attr_dict[comid_str]
                    
                    X_val_combined = np.hstack([X_ts_val_flat, X_attr_val])
                else:
                    X_val_combined = X_ts_val_flat
                
                # Standardize validation data using training statistics
                if standardize:
                    X_val_combined = (X_val_combined - self.feature_means) / self.feature_stds
                
                # Evaluate on validation set
                val_pred = self.base_model.predict(X_val_combined)
                val_mse = np.mean((val_pred - Y_val) ** 2)
                val_mae = np.mean(np.abs(val_pred - Y_val))
                print(f"Validation MSE: {val_mse:.6f}, MAE: {val_mae:.6f}")
    
    def predict(self, X_ts, X_attr=None):
        """
        Batch prediction
        
        Parameters:
            X_ts: Time series features [N, T, D]
            X_attr: Attribute features [N, attr_dim] (optional)
            
        Returns:
            Predictions [N]
        """
        with TimingAndMemoryContext("Regression prediction"):
            N, T, D = X_ts.shape
            
            # Flatten time series
            X_ts_flat = X_ts.reshape(N, T * D)
            
            # Combine with attributes if provided
            if X_attr is not None:
                X_combined = np.hstack([X_ts_flat, X_attr])
            else:
                X_combined = X_ts_flat
            
            # Apply standardization if it was used during training
            if self.feature_means is not None and self.feature_stds is not None:
                # Check dimensions match
                if X_combined.shape[1] == self.feature_means.shape[0]:
                    X_combined = (X_combined - self.feature_means) / self.feature_stds
                else:
                    # Handle dimension mismatch (can happen if model was trained with different features)
                    # In this case, don't apply standardization and log a warning
                    logging.warning(f"Feature dimension mismatch during prediction: "
                                   f"got {X_combined.shape[1]}, expected {self.feature_means.shape[0]}. "
                                   f"Skipping standardization.")
            
            # Make predictions
            return self.base_model.predict(X_combined)
    
    def save_model(self, path):
        """
        Save model to file
        
        Parameters:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create a dictionary with model and metadata
        model_data = {
            'model': self.base_model,
            'type': self.reg_type,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'feature_names': self.feature_names,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds
        }
        
        # Save to file
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"{self.reg_type} regression model saved to {path}")
    
    def load_model(self, path):
        """
        Load model from file
        
        Parameters:
            path: Path to the saved model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with TimingAndMemoryContext("Loading regression model"):
            # Load model dictionary
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model
            self.base_model = model_data['model']
            self.reg_type = model_data['type']
            self.alpha = model_data['alpha']
            self.l1_ratio = model_data.get('l1_ratio', 0.5)  # Default if missing
            self.feature_names = model_data.get('feature_names', None)
            self.feature_means = model_data.get('feature_means', None)
            self.feature_stds = model_data.get('feature_stds', None)
            
            print(f"Loaded {self.reg_type} regression model from {path}")
    
    def get_model_info(self):
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        
        # Add regression-specific information
        info.update({
            'regression_type': self.reg_type,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
        })
        
        # Add coefficients if available
        if hasattr(self.base_model, 'coef_'):
            if self.feature_names and len(self.feature_names) == len(self.base_model.coef_):
                # Include feature names with coefficients
                coef_dict = {}
                for i, name in enumerate(self.feature_names):
                    coef_dict[name] = float(self.base_model.coef_[i])
                info['coefficients'] = coef_dict
            else:
                # Just include the coefficients
                info['coefficients'] = self.base_model.coef_.tolist()
            
            # Include intercept if available
            if hasattr(self.base_model, 'intercept_'):
                info['intercept'] = float(self.base_model.intercept_)
        
        return info


# =============================================================================
# Factory function to create regression model instances
# =============================================================================

def create_regression_model(reg_type='linear', alpha=1.0, l1_ratio=0.5, 
                           input_dim=None, attr_dim=None, memory_check_interval=5):
    """
    Create a regression model instance
    
    Parameters:
        reg_type: Type of regression ('linear', 'ridge', 'lasso', 'elasticnet')
        alpha: Regularization strength (for Ridge, Lasso and ElasticNet)
        l1_ratio: L1 ratio for ElasticNet (0 = Ridge, 1 = Lasso)
        input_dim: Input feature dimension (time series)
        attr_dim: Attribute feature dimension
        memory_check_interval: Memory check interval
        
    Returns:
        RegressionModel instance
    """
    return RegressionModel(
        reg_type=reg_type,
        alpha=alpha,
        l1_ratio=l1_ratio,
        input_dim=input_dim,
        attr_dim=attr_dim,
        memory_check_interval=memory_check_interval
    )