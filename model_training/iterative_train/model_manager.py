"""
model_manager.py - 模型管理模块

该模块负责模型的创建、加载、训练和预测，以及相关的辅助功能。
与数据处理模块配合，确保模型操作的一致性和模块化。
"""

import os
import torch
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime

# 导入项目中的函数
from PGRWQI.model_training.models.model_factory import create_model
from PGRWQI.model_training.models.models import CatchmentModel
from PGRWQI.model_training.gpu_memory_utils import (
    log_memory_usage, 
    TimingAndMemoryContext,
    force_cuda_memory_cleanup
)


class ModelManager:
    """
    模型管理器类，负责模型的创建、加载、训练和预测
    
    提供统一的接口管理模型生命周期，确保模型操作的一致性和可追踪性。
    """
    
    def __init__(self, model_type: str, device: str, model_dir: str):
        """
        初始化模型管理器
        
        参数:
            model_type: 模型类型，例如'lstm', 'rf'
            device: 计算设备，'cpu'或'cuda'
            model_dir: 模型保存目录
        """
        self.model_type = model_type
        self.device = device
        self.model_dir = model_dir
        self.model = None
        
        # 确保模型保存目录存在
        os.makedirs(model_dir, exist_ok=True)
        logging.info(f"模型管理器初始化: 类型={model_type}, 设备={device}, 存储目录={model_dir}")
        
    def create_or_load_model(self, 
                            build_params: Dict[str, Any],
                            train_params: Dict[str, Any],
                            model_path: str,
                            attr_dict: Optional[Dict[str, np.ndarray]] = None,
                            train_data: Optional[Tuple] = None) -> CatchmentModel:
        """
        创建新模型或加载已有模型
        
        参数:
            build_params: 模型构建参数
            train_params: 模型训练参数
            model_path: 模型路径
            attr_dict: 属性字典（仅训练时需要）
            train_data: 训练数据元组（仅训练时需要）
            
        返回:
            创建或加载的模型
        """
        with TimingAndMemoryContext(f"{'加载' if os.path.exists(model_path) else '创建'}模型"):
            # 创建模型实例
            model = create_model(
                model_type=self.model_type,
                **build_params
            )
            
            # 检查是否存在预训练模型
            if os.path.exists(model_path):
                model.load_model(model_path)
                logging.info(f"成功加载模型: {model_path}")
            elif train_data is not None:
                # 如果没有预训练模型且提供了训练数据，则训练新模型
                X_ts_train, comid_arr_train, Y_train, X_ts_val, comid_arr_val, Y_val = train_data
                
                with TimingAndMemoryContext("模型训练"):
                    model.train_model(
                        attr_dict, comid_arr_train, X_ts_train, Y_train, 
                        comid_arr_val, X_ts_val, Y_val, 
                        **train_params
                    )
                
                # 确保目录存在
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                # 保存模型
                model.save_model(model_path)
                logging.info(f"模型训练成功！已保存到 {model_path}")
            else:
                logging.warning("未提供训练数据，且找不到预训练模型，返回未训练的模型实例")
                
            self.model = model
            return model
    
    def batch_predict(self, X_ts: np.ndarray, X_attr: np.ndarray) -> np.ndarray:
        """
        批量预测
        
        参数:
            X_ts: 时间序列特征 [N, T, D]
            X_attr: 属性特征 [N, attr_dim]
            
        返回:
            预测结果 [N]
        """
        if self.model is None:
            raise ValueError("模型尚未加载或创建")
            
        with TimingAndMemoryContext("批量预测"):
            # 确保模型在正确的设备上
            if hasattr(self.model, 'device') and self.model.device != self.device:
                logging.warning(f"模型设备({self.model.device})与指定设备({self.device})不一致")
            
            try:
                # 使用模型进行预测
                predictions = self.model.predict(X_ts, X_attr)
                
                # 清理资源
                if self.device == 'cuda' and torch.cuda.is_available():
                    force_cuda_memory_cleanup()
                    
                return predictions
            except Exception as e:
                logging.error(f"预测过程中出错: {str(e)}")
                # 在CUDA错误时尝试降级为较小的批次
                if "CUDA" in str(e) and self.device == 'cuda':
                    return self._fallback_batch_predict(X_ts, X_attr)
                else:
                    raise
    
    def _fallback_batch_predict(self, X_ts: np.ndarray, X_attr: np.ndarray, 
                               batch_size: int = 100) -> np.ndarray:
        """
        回退策略：小批量预测
        
        参数:
            X_ts: 时间序列特征
            X_attr: 属性特征
            batch_size: 批处理大小
            
        返回:
            预测结果
        """
        logging.info(f"使用回退策略进行预测，批大小={batch_size}")
        total_samples = X_ts.shape[0]
        predictions = np.zeros(total_samples)
        
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_X_ts = X_ts[i:end_idx]
            batch_X_attr = X_attr[i:end_idx]
            
            try:
                batch_preds = self.model.predict(batch_X_ts, batch_X_attr)
                predictions[i:end_idx] = batch_preds
            except Exception as e:
                logging.error(f"处理批次 {i}:{end_idx} 失败: {str(e)}")
                # 对失败的批次填充0
                predictions[i:end_idx] = 0.0
                
            # 每批次后清理资源
            if self.device == 'cuda' and torch.cuda.is_available():
                force_cuda_memory_cleanup()
        
        return predictions
            
    def process_batch_prediction(self, 
                               batch_data: Dict,
                               results_dict: Dict[int, pd.Series] = None) -> Dict[int, pd.Series]:
        """
        处理批量预测结果并转换为适当的格式
        
        参数:
            batch_data: 从DataHandler.prepare_batch_prediction_data获得的数据
            results_dict: 存储结果的字典（可选）
            
        返回:
            包含预测结果的字典，键为COMID，值为预测序列
        """
        if results_dict is None:
            results_dict = {}
            
        if batch_data is None:
            return results_dict
        
        X_ts = batch_data['X_ts_scaled']
        X_attr = batch_data['X_attr_batch']
        valid_comids = batch_data['valid_comids']
        comid_indices = batch_data['comid_indices']
        groups = batch_data['groups']
        
        # 执行批量预测
        try:
            all_preds = self.batch_predict(X_ts, X_attr)
            
            # 将预测结果映射回河段
            for comid in valid_comids:
                start_idx, end_idx, dates, all_dates = comid_indices[comid]
                preds = all_preds[start_idx:end_idx]
                
                # 创建预测序列
                pred_series = pd.Series(preds, index=pd.to_datetime(dates))
                full_series = pd.Series(0.0, index=all_dates)
                full_series.update(pred_series)
                
                results_dict[comid] = full_series
        except Exception as e:
            logging.error(f"批量预测处理失败: {str(e)}")
            # 降级：为每个COMID创建零序列
            for comid in valid_comids:
                _, _, _, all_dates = comid_indices[comid]
                results_dict[comid] = pd.Series(0.0, index=all_dates)
        
        # 确保所有请求的COMID都有结果
        for comid in groups.keys():
            if comid not in results_dict:
                results_dict[comid] = pd.Series(0.0, index=groups[comid]["date"])
        
        return results_dict
    
    def verify_model(self, 
                    model: CatchmentModel,
                    test_data: Tuple,
                    attr_dict: Dict[str, np.ndarray],
                    comids_to_verify: List[int],
                    target_col: str = 'TN',
                    output_dir: str = 'model_verification',
                    date_range: Tuple[str, str] = None):
        """
        创建模型验证图，对比真实值和预测值
        
        参数:
            model: 要验证的模型
            test_data: 测试数据元组 (X_ts, Y, COMIDs, Dates)
            attr_dict: 属性字典
            comids_to_verify: 要验证的COMID列表
            target_col: 目标参数名称
            output_dir: 输出目录
            date_range: 日期范围元组 (开始日期, 结束日期)
        """
        # 解包测试数据
        X_ts_test, Y_test, COMIDs_test, Dates_test = test_data
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 解析日期范围
        start_date = None
        end_date = None
        if date_range:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            logging.info(f"筛选日期范围: {start_date} 到 {end_date}")
        
        # 按COMID分组数据
        comid_data = {}
        for i, comid in enumerate(COMIDs_test):
            if comid not in comids_to_verify:
                continue
                
            # 转换日期为datetime类型
            date = pd.to_datetime(Dates_test[i])
            
            # 按日期范围筛选
            if date_range:
                if date < start_date or date > end_date:
                    continue
            
            if comid not in comid_data:
                comid_data[comid] = {'dates': [], 'actual': [], 'idx': []}
            
            comid_data[comid]['dates'].append(date)
            comid_data[comid]['actual'].append(Y_test[i])
            comid_data[comid]['idx'].append(i)
        
        # 处理指定的COMID
        for comid in comids_to_verify:
            if comid not in comid_data or not comid_data[comid]['dates']:
                logging.warning(f"COMID {comid} 在测试数据中未找到或日期范围内无数据")
                continue
                
            # 获取此COMID的索引
            indices = comid_data[comid]['idx']
            
            # 获取特征
            X_ts_comid = X_ts_test[indices]
            
            # 创建属性输入
            attr_vec = attr_dict.get(str(comid), np.zeros_like(next(iter(attr_dict.values()))))
            X_attr_comid = np.tile(attr_vec, (len(indices), 1))
            
            # 获取预测结果
            predictions = model.predict(X_ts_comid, X_attr_comid)
            
            # 获取真实值和日期
            actuals = comid_data[comid]['actual']
            dates = comid_data[comid]['dates']
            
            # 创建数据框以便绘图
            df_plot = pd.DataFrame({
                'date': dates,
                'actual': actuals,
                'predicted': predictions
            })
            
            # 按日期排序
            df_plot = df_plot.sort_values('date')
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            
            # 绘制真实值
            plt.plot(df_plot['date'], df_plot['actual'], 'o-', color='blue', label='真实值')
            
            # 绘制预测值
            plt.plot(df_plot['date'], df_plot['predicted'], 'x-', color='orange', label='预测值')
            
            # 设置标签和标题
            plt.xlabel('日期')
            plt.ylabel(f'{target_col} 值')
            
            # 如果指定了日期范围，在标题中包含
            date_info = f" ({start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')})" if date_range else ""
            plt.title(f'站点 {comid} - 模型验证{date_info}')
            
            # 添加图例
            plt.legend()
            
            # 添加网格
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 格式化x轴以便更好地显示日期
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            
            # 保存图表
            filename = f'verification_{comid}'
            if date_range:
                # 在文件名中添加日期范围
                filename += f"_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
            
            plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)
            plt.close()
            
            logging.info(f"为COMID {comid}创建了验证图")
        
        # 创建所有数据点的散点图
        if comid_data:
            self._create_verification_scatter_plot(
                comid_data, model, X_ts_test, attr_dict, target_col, 
                output_dir, start_date, end_date
            )
        
        logging.info(f"所有验证图保存到 {output_dir}")
    
    def _create_verification_scatter_plot(self, 
                                        comid_data, 
                                        model, 
                                        X_ts_test, 
                                        attr_dict, 
                                        target_col, 
                                        output_dir, 
                                        start_date=None, 
                                        end_date=None):
        """创建验证散点图"""
        plt.figure(figsize=(8, 8))
        
        # 收集所有预测和真实值
        all_preds = []
        all_actuals = []
        
        for comid in comid_data:
            indices = comid_data[comid]['idx']
            
            # 获取特征
            X_ts_comid = X_ts_test[indices]
            
            # 创建属性输入
            attr_vec = attr_dict.get(str(comid), np.zeros_like(next(iter(attr_dict.values()))))
            X_attr_comid = np.tile(attr_vec, (len(indices), 1))
            
            # 获取预测
            preds = model.predict(X_ts_comid, X_attr_comid)
            actuals = comid_data[comid]['actual']
            
            all_preds.extend(preds)
            all_actuals.extend(actuals)
        
        # 创建散点图
        plt.scatter(all_actuals, all_preds, alpha=0.5)
        
        # 添加1:1线
        max_val = max(max(all_actuals), max(all_preds))
        min_val = min(min(all_actuals), min(all_preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 线')
        
        # 计算指标
        mse = np.mean((np.array(all_actuals) - np.array(all_preds))**2)
        rmse = np.sqrt(mse)
        r2 = np.corrcoef(all_actuals, all_preds)[0, 1]**2
        
        # 添加指标到图
        plt.text(0.05, 0.95, f'RMSE: {rmse:.3f}\nR²: {r2:.3f}', 
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # 设置标签和标题
        plt.xlabel(f'真实 {target_col}')
        plt.ylabel(f'预测 {target_col}')
        
        # 如果指定了日期范围，在标题中包含
        date_info = f" ({start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')})" if start_date and end_date else ""
        plt.title(f'模型验证 - 所有站点{date_info}')
        
        # 等比例坐标轴
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图
        filename = 'verification_all_stations'
        if start_date and end_date:
            filename += f"_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
        
        plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300)
        plt.close()