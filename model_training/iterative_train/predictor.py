"""
predictor.py - 统一预测器类

将批量预测和单一预测逻辑封装到一个类中，符合面向对象设计原则
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Union, Optional

from PGRWQI.model_training.gpu_memory_utils import TimingAndMemoryContext


class CatchmentPredictor:
    """
    流域预测器类
    
    封装了批量预测和单一预测的逻辑，管理数据处理器和模型管理器的交互
    """
    
    def __init__(self, 
                 data_handler, 
                 model_manager, 
                 all_target_cols: List[str], 
                 target_col: str):
        """
        初始化预测器
        
        参数:
            data_handler: 数据处理器实例
            model_manager: 模型管理器实例
            all_target_cols: 所有目标列列表
            target_col: 主目标列
        """
        self.data_handler = data_handler
        self.model_manager = model_manager
        self.all_target_cols = all_target_cols
        self.target_col = target_col
        
        logging.info(f"预测器初始化完成，目标列: {target_col}")
    
    def predict_batch(self, comid_batch: List) -> Dict:
        """
        批量预测多个河段
        
        参数:
            comid_batch: 河段COMID列表
            
        返回:
            预测结果字典 {comid: prediction_series}
        """
        with TimingAndMemoryContext(f"批量预测 {len(comid_batch)} 个河段"):
            # 准备批处理数据
            batch_data = self.data_handler.prepare_batch_prediction_data(
                comid_batch, self.all_target_cols, self.target_col
            )
            
            if batch_data is None:
                logging.warning(f"批次 {comid_batch[:5]}... 的数据准备失败")
                return {}
            
            # 处理批量预测结果
            results = self.model_manager.process_batch_prediction(batch_data)
            
            return results
    
    def predict_single(self, group: pd.DataFrame) -> pd.Series:
        """
        单一河段预测
        
        参数:
            group: 河段数据组（包含COMID和date列）
            
        返回:
            预测序列（pandas.Series）
        """
        comid = group.iloc[0]['COMID']
        
        # 使用批量预测处理单个河段
        results = self.predict_batch([comid])
        
        # 返回预测结果
        if comid in results:
            return results[comid]
        else:
            # 预测失败，返回零序列
            group_sorted = group.sort_values("date")
            logging.warning(f"河段 {comid} 预测失败，返回零序列")
            return pd.Series(0.0, index=group_sorted["date"])
    
    def predict_large_batch(self, 
                          comid_list: List, 
                          batch_size: int = 1000) -> Dict:
        """
        大批量预测，自动分批处理
        
        参数:
            comid_list: 河段COMID列表
            batch_size: 每批处理的河段数量
            
        返回:
            预测结果字典 {comid: prediction_series}
        """
        total_results = {}
        num_batches = (len(comid_list) + batch_size - 1) // batch_size
        
        logging.info(f"大批量预测开始，总共 {len(comid_list)} 个河段，分 {num_batches} 批处理")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(comid_list))
            batch_comids = comid_list[start_idx:end_idx]
            
            # 预测当前批次
            batch_results = self.predict_batch(batch_comids)
            total_results.update(batch_results)
            
            # 记录进度
            if (i + 1) % 5 == 0 or i == num_batches - 1:
                logging.info(f"已完成 {i + 1}/{num_batches} 批次，"
                           f"累计处理 {end_idx}/{len(comid_list)} 个河段")
        
        return total_results
    
    def update_target_col(self, new_target_col: str):
        """
        更新目标列
        
        参数:
            new_target_col: 新的目标列名
        """
        if new_target_col not in self.all_target_cols:
            raise ValueError(f"目标列 {new_target_col} 不在允许的列表中: {self.all_target_cols}")
        
        self.target_col = new_target_col
        logging.info(f"目标列已更新为: {new_target_col}")
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        返回:
            模型信息字典
        """
        info = {
            'target_col': self.target_col,
            'all_target_cols': self.all_target_cols,
            'data_handler_initialized': self.data_handler.initialized if hasattr(self.data_handler, 'initialized') else 'Unknown'
        }
        
        # 获取模型管理器信息
        if hasattr(self.model_manager, 'model') and self.model_manager.model:
            if hasattr(self.model_manager.model, 'get_model_info'):
                info['model_info'] = self.model_manager.model.get_model_info()
            else:
                info['model_type'] = self.model_manager.model_type
        
        return info


# 便捷函数，用于向后兼容
def create_optimized_batch_func(predictor: CatchmentPredictor):
    """
    创建优化的批处理函数（使用预测器类）
    
    参数:
        predictor: CatchmentPredictor实例
        
    返回:
        批处理函数
    """
    return predictor.predict_batch


def create_optimized_single_func(predictor: CatchmentPredictor):
    """
    创建优化的单一预测函数（使用预测器类）
    
    参数:
        predictor: CatchmentPredictor实例
        
    返回:
        单一预测函数
    """
    return predictor.predict_single