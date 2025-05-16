"""
data_handler.py - 数据处理与标准化模块

该模块整合了数据加载、滑动窗口创建和标准化操作，解决了X_ts和attr_dict
标准化不一致的问题。通过提供统一的数据获取接口，确保每次得到的数据都经过
一致的标准化处理。
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

# 导入项目中的函数
from PGRWQI.data_processing import (
    build_sliding_windows_for_subset, 
    standardize_time_series_all, 
    standardize_attributes
)
from PGRWQI.model_training.gpu_memory_utils import TimingAndMemoryContext

class DataHandler:
    """
    数据处理器类，负责数据加载、处理和标准化
    
    该类解决了X_ts和attr_dict标准化不一致的问题，通过内部维护原始数据和标准化器，
    确保每次生成的数据都经过一致的标准化处理。
    """
    
    def __init__(self):
        """初始化数据处理器"""
        # 原始数据
        self.df = None
        self.attr_df = None
        
        # 标准化器
        self.ts_scaler = None
        self.attr_scaler = None
        
        # 特征列表
        self.input_features = None
        self.attr_features = None
        
        # 缓存的属性字典(非标准化)
        self._raw_attr_dict = None
        
        # 是否已初始化
        self.initialized = False
        
    def initialize(self, 
                  df: pd.DataFrame, 
                  attr_df: pd.DataFrame,
                  input_features: List[str],
                  attr_features: List[str]):
        """
        初始化数据处理器
        
        参数:
            df: 包含时间序列数据的DataFrame
            attr_df: 包含属性数据的DataFrame
            input_features: 输入特征列表
            attr_features: 属性特征列表
        """
        self.df = df.copy()
        self.attr_df = attr_df.copy()
        self.input_features = input_features
        self.attr_features = attr_features
        
        # 构建原始属性字典
        self._raw_attr_dict = self._build_attribute_dictionary()
        
        # 构建并保存标准化器(但不应用)
        sample_data = self._get_sample_data()
        if sample_data is not None:
            X_sample, attr_dict_sample = sample_data
            # 确保样本数据足够大以创建健壮的标准化器
            _, self.ts_scaler = standardize_time_series_all(X_sample)
            _, self.attr_scaler = standardize_attributes(attr_dict_sample)
            logging.info("初始化了数据标准化器，将在所有预测中一致使用")
        else:
            logging.warning("无法初始化数据标准化器，这可能导致不一致的预测结果")
        
        self.initialized = True
        logging.info("数据处理器初始化完成")
    
    def _build_attribute_dictionary(self) -> Dict[str, np.ndarray]:
        """构建河段属性字典(非标准化版本)"""
        attr_dict = {}
        for row in self.attr_df.itertuples(index=False):
            comid = str(row.COMID)
            row_dict = row._asdict()
            attrs = []
            for attr in self.attr_features:
                if attr in row_dict:
                    attrs.append(row_dict[attr])
                else:
                    attrs.append(0.0)
            attr_dict[comid] = np.array(attrs, dtype=np.float32)
        return attr_dict
    
    def _get_sample_data(self, use_all_data=True) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        获取样本数据用于创建标准化器
        
        参数:
            use_all_data: 是否使用所有可用数据创建标准化器，默认为True
        
        返回:
            样本数据元组 (X_sample, attr_dict_sample)，如果无法创建则返回None
        """
        if self.df is None or self.attr_df is None:
            return None
        
        # 选择要使用的COMID列表
        if use_all_data:
            # 使用所有可用的COMID
            sample_comids = self.df['COMID'].unique()
            logging.info(f"使用全部 {len(sample_comids)} 个COMID数据创建标准化器")
        else:
            # 仅选择少量COMID用于样本生成
            sample_comids = self.df['COMID'].unique()[:min(10, len(self.df['COMID'].unique()))]
            logging.info(f"使用 {len(sample_comids)} 个样本COMID创建标准化器")
        
        # # 对于大数据集，可能需要限制每个COMID的数据量以防止内存不足
        # if use_all_data and len(sample_comids) > 100:
        #     # 如果COMID太多，可能需要随机抽样
        #     import random
        #     random.seed(42)  # 确保结果可重现
        #     sample_comids = random.sample(list(sample_comids), 100)
        #     logging.info(f"COMID过多，随机抽样 100 个用于创建标准化器")
        
        # 构建样本滑动窗口
        X_sample, _, _, _ = build_sliding_windows_for_subset(
            self.df, 
            sample_comids, 
            input_cols=self.input_features, 
            target_col="TN",               
            all_target_cols=["TN", "TP"],  
            time_window=10,
            skip_missing_targets=True      
        )
        
        if X_sample is None:
            logging.warning("无法创建滑动窗口样本数据，标准化器创建失败")
            return None
        
        # 获取样本属性字典
        attr_dict_sample = {k: v for k, v in self._raw_attr_dict.items() 
                            if k in [str(c) for c in sample_comids]}
        
        logging.info(f"成功创建标准化样本数据: X_sample.shape={X_sample.shape}, attr_dict_sample大小={len(attr_dict_sample)}")
        return X_sample, attr_dict_sample
    
    def get_standardized_data(self, 
                             comid_list: List, 
                             all_target_cols: List[str],
                             target_col: str,
                             time_window: int = 10,
                             skip_missing_targets: bool = True) -> Tuple:
        """
        获取标准化的训练数据
        
        参数:
            comid_list: 要处理的COMID列表
            all_target_cols: 所有目标列列表
            target_col: 主目标列
            time_window: 时间窗口大小
            skip_missing_targets: 是否跳过缺失目标值的样本
            
        返回:
            (X_ts_scaled, attr_dict_scaled, Y, COMIDs, Dates): 标准化后的数据
        """
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")
            
        with TimingAndMemoryContext("构建并标准化数据"):
            # 1. 构建滑动窗口
            X_ts, Y, COMIDs, Dates = build_sliding_windows_for_subset(
                self.df, 
                comid_list, 
                input_cols=self.input_features, 
                all_target_cols=all_target_cols,
                target_col=target_col,
                time_window=time_window,
                skip_missing_targets=skip_missing_targets
            )
            
            if X_ts is None:
                return None, None, None, None, None
            
            # 2. 标准化X_ts - 使用初始化时创建的ts_scaler
            N, T, input_dim = X_ts.shape
            X_ts_2d = X_ts.reshape(-1, input_dim)
            X_ts_scaled_2d = self.ts_scaler.transform(X_ts_2d)
            X_ts_scaled = X_ts_scaled_2d.reshape(N, T, input_dim)
            
            # 3. 为这些COMID提取属性字典并标准化
            comid_strs = [str(comid) for comid in COMIDs]
            attr_dict_subset = {k: self._raw_attr_dict[k] for k in comid_strs if k in self._raw_attr_dict}
            
            # 使用初始化时创建的attr_scaler
            attr_matrix = np.vstack([attr_dict_subset[k] for k in attr_dict_subset.keys()])
            attr_matrix_scaled = self.attr_scaler.transform(attr_matrix)
            attr_dict_scaled = {k: attr_matrix_scaled[i] for i, k in enumerate(attr_dict_subset.keys())}
            
            return X_ts_scaled, attr_dict_scaled, Y, COMIDs, Dates
    
    def get_standardized_attr_dict(self) -> Dict[str, np.ndarray]:
        """获取标准化的完整属性字典"""
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")
            
        # 使用初始化时创建的scaler，而不是重新拟合
        attr_matrix = np.vstack([self._raw_attr_dict[k] for k in self._raw_attr_dict.keys()])
        attr_matrix_scaled = self.attr_scaler.transform(attr_matrix)
        scaled_attr_dict = {k: attr_matrix_scaled[i] for i, k in enumerate(self._raw_attr_dict.keys())}
        
        return scaled_attr_dict
    
    def prepare_training_data_for_head_segments(self,
                                              comid_wq_list: List,
                                              comid_era5_list: List,
                                              all_target_cols: List[str],
                                              target_col: str,
                                              output_dir: str,
                                              model_version: str) -> Tuple:
        """
        为头部河段准备训练数据
        
        参数:
            comid_wq_list: 水质站点COMID列表
            comid_era5_list: ERA5数据覆盖的COMID列表 
            all_target_cols: 所有目标列列表
            target_col: 主目标列
            output_dir: 输出目录
            model_version: 模型版本号
            
        返回:
            (X_ts_scaled, attr_dict_scaled, Y, COMIDs, Dates): 标准化后的数据
        """
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")

        # 筛选头部河段
        with TimingAndMemoryContext("寻找头部站点"):
            # 简化头部站点选择标准
            attr_df_head_upstream = self.attr_df.copy()
            df_head_upstream = self.df[self.df['COMID'].isin(attr_df_head_upstream['COMID'])]
            
            comid_list_head = list(set(df_head_upstream['COMID'].unique().tolist()) 
                                 & set(comid_wq_list) 
                                 & set(comid_era5_list))
            
            # 保存头部河段COMID列表
            np.save(f"{output_dir}/comid_list_head_{model_version}.npy", comid_list_head)
            
            if len(comid_list_head) == 0:
                logging.warning("警告：找不到符合条件的头部河段，请检查输入。")
                return None, None, None, None, None
            
            logging.info(f"选择的头部河段数量：{len(comid_list_head)}")

        # 获取标准化的训练数据
        X_ts_scaled, attr_dict_scaled, Y, COMIDs, Dates = self.get_standardized_data(
            comid_list_head,
            all_target_cols,
            target_col,
            time_window=10
        )
        
        # 输出数据维度信息
        logging.info(f"X_ts_scaled.shape = {X_ts_scaled.shape}")
        logging.info(f"Y.shape = {Y.shape}")
        logging.info(f"COMIDs.shape = {COMIDs.shape}")
        logging.info(f"Dates.shape = {Dates.shape}")

        # 保存训练数据
        with TimingAndMemoryContext("保存训练数据"):
            np.savez(f"{output_dir}/upstreams_trainval_{model_version}.npz", 
                    X=X_ts_scaled, Y=Y, COMID=COMIDs, Date=Dates)
            logging.info("训练数据保存成功！")
            
        return X_ts_scaled, attr_dict_scaled, Y, COMIDs, Dates
    
    def prepare_batch_prediction_data(self, 
                                     comid_batch: List, 
                                     all_target_cols: List[str],
                                     target_col: str) -> Dict:
        """
        为批量预测准备数据
        
        参数:
            comid_batch: 要处理的COMID批次
            all_target_cols: 所有目标列列表
            target_col: 主目标列
            
        返回:
            包含预测所需数据的字典
        """
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")
            
        # 按COMID分组
        groups = {comid: group.sort_values("date").copy() 
                  for comid, group in self.df.groupby("COMID") 
                  if comid in comid_batch}
        
        # 收集所有X_ts数据和相关信息
        all_data = {
            'X_ts_list': [],
            'comids': [],
            'dates': [],
            'comid_indices': {}
        }
        
        current_idx = 0
        valid_comids = []
        
        # 为每个COMID准备滑动窗口数据
        for comid in comid_batch:
            if comid not in groups:
                continue
                
            group = groups[comid]
            dates = group["date"]
            
            # 构建滑动窗口
            X_ts_local, _, _, dates_local = build_sliding_windows_for_subset(
                df=group, 
                comid_list=[comid], 
                input_cols=self.input_features, 
                all_target_cols=all_target_cols, 
                target_col=target_col,
                time_window=10,
                skip_missing_targets=False
            )
            
            if X_ts_local is None or X_ts_local.shape[0] == 0:
                continue
            
            # 记录数据位置
            end_idx = current_idx + X_ts_local.shape[0]
            all_data['comid_indices'][comid] = (current_idx, end_idx, dates_local, dates)
            current_idx = end_idx
            valid_comids.append(comid)
            
            all_data['X_ts_list'].append(X_ts_local)
            all_data['comids'].extend([comid] * X_ts_local.shape[0])
            all_data['dates'].extend(dates_local)
        
        if not all_data['X_ts_list']:
            return None
        
        # 堆叠所有X_ts数据并标准化 - 使用初始化时创建的scaler
        X_ts_batch = np.vstack(all_data['X_ts_list'])
        N, T, input_dim = X_ts_batch.shape
        X_ts_2d = X_ts_batch.reshape(-1, input_dim)
        X_ts_scaled_2d = self.ts_scaler.transform(X_ts_2d)
        X_ts_scaled = X_ts_scaled_2d.reshape(N, T, input_dim)
        
        # 准备属性数据 - 使用初始化时创建的scaler
        attr_dict_scaled = self.get_standardized_attr_dict()
        
        attr_dim = next(iter(attr_dict_scaled.values())).shape[0]
        X_attr_batch = np.zeros((X_ts_scaled.shape[0], attr_dim), dtype=np.float32)
        
        for i, comid in enumerate(all_data['comids']):
            comid_str = str(comid)
            attr_vec = attr_dict_scaled.get(comid_str, np.zeros(attr_dim, dtype=np.float32))
            X_attr_batch[i] = attr_vec
        
        # 返回结果
        return {
            'X_ts_scaled': X_ts_scaled,
            'X_attr_batch': X_attr_batch,
            'valid_comids': valid_comids,
            'comid_indices': all_data['comid_indices'],
            'groups': groups
        }
    
    def prepare_next_iteration_data(self,
                                   df_flow: pd.DataFrame,
                                   target_col: str,
                                   col_y_n: str,
                                   col_y_up: str,
                                   time_window: int = 10) -> Tuple:
        """
        准备下一轮迭代的训练数据
        
        参数:
            df_flow: 汇流计算结果DataFrame
            target_col: 目标列
            col_y_n: y_n列名
            col_y_up: y_up列名
            time_window: 时间窗口大小
            
        返回:
            (X_ts_scaled, attr_dict_scaled, Y_label, COMIDs, Dates): 标准化后的数据
        """
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")
            
        # 标准化列名
        df_flow_copy = df_flow.copy()
        df_copy = self.df.copy()
        
        # 确保date列命名一致
        if 'date' in df_copy.columns and 'Date' in df_flow_copy.columns:
            df_flow_copy = df_flow_copy.rename(columns={'Date': 'date'})
        elif 'Date' in df_copy.columns and 'date' in df_flow_copy.columns:
            df_copy = df_copy.rename(columns={'Date': 'date'})
        
        # 检查必要的列
        required_cols = ['COMID', 'date', col_y_n, col_y_up]
        for col in required_cols:
            if col not in df_flow_copy.columns:
                logging.error(f"缺少必要的列: {col}")
                return None, None, None, None, None
        
        # 合并数据
        merged = pd.merge(
            df_copy, 
            df_flow_copy[required_cols], 
            on=['COMID', 'date'], 
            how='left'
        )
        
        # 计算E_label
        merged["E_label"] = merged[target_col] - merged[col_y_up]
        comid_list = merged["COMID"].unique().tolist()
        
        # 构建滑动窗口
        X_ts, _, COMIDs, Dates = build_sliding_windows_for_subset(
            merged,
            comid_list,
            input_cols=self.input_features,
            target_cols=["E_label"],
            time_window=time_window
        )
        
        if X_ts is None:
            return None, None, None, None, None
        

        # 标准化数据 - 使用初始化时创建的scaler
        N, T, input_dim = X_ts.shape
        X_ts_2d = X_ts.reshape(-1, input_dim)
        X_ts_scaled_2d = self.ts_scaler.transform(X_ts_2d)
        X_ts_scaled = X_ts_scaled_2d.reshape(N, T, input_dim)
        
        attr_dict_scaled = self.get_standardized_attr_dict()

        # 从merged中提取E_label值
        Y_label = []
        for cid, date_val in zip(COMIDs, Dates):
            subset = merged[(merged["COMID"] == cid) & (merged["date"] == date_val)]
            if not subset.empty:
                label_val = subset["E_label"].iloc[0]
            else:
                label_val = 0.0
            Y_label.append(label_val)
        
        Y_label = np.array(Y_label, dtype=np.float32)
        
        
        return X_ts_scaled, attr_dict_scaled, Y_label, COMIDs, Dates