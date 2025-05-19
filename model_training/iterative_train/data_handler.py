"""
data_handler.py - 简化优化版数据处理与标准化模块

简化的优化版本，专注于核心性能优化：
1. 预分组数据，避免重复分组
2. 预标准化属性
3. 可选的预计算滑动窗口（顺序处理）
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional

# 导入项目中的函数
from PGRWQI.data_processing import (
    build_sliding_windows_for_subset, 
    standardize_time_series_all, 
    standardize_attributes
)
from PGRWQI.model_training.gpu_memory_utils import TimingAndMemoryContext

class DataHandler:
    """
    简化优化版数据处理器类
    
    核心优化：
    1. 预分组数据，消除重复分组操作
    2. 预标准化属性，避免重复标准化
    3. 一致的标准化器使用
    4. 可选的预计算滑动窗口
    """
    
    def __init__(self, enable_precompute=False):
        """
        初始化数据处理器
        
        参数:
            enable_precompute: 是否启用预计算滑动窗口
        """
        # 基础属性
        self.df = None
        self.attr_df = None
        self.input_features = None
        self.attr_features = None
        self.initialized = False
        
        # 预计算选项
        self.enable_precompute = enable_precompute
        
        # 预分组数据（核心优化1）
        self._cached_groups = None
        
        # 预计算的滑动窗口（可选优化）
        self._cached_windows = {}
        
        # 标准化器（一致性保证）
        self.ts_scaler = None
        self.attr_scaler = None
        
        # 预标准化属性（核心优化2）
        self._raw_attr_dict = None
        self._standardized_attr_dict = None
        
    def initialize(self, 
                  df: pd.DataFrame, 
                  attr_df: pd.DataFrame,
                  input_features: List[str],
                  attr_features: List[str]):
        """
        初始化数据处理器，执行关键预处理
        
        参数:
            df: 包含时间序列数据的DataFrame
            attr_df: 包含属性数据的DataFrame
            input_features: 输入特征列表
            attr_features: 属性特征列表
        """
        with TimingAndMemoryContext("DataHandler简化初始化"):
            self.df = df.copy()
            self.attr_df = attr_df.copy()
            self.input_features = input_features
            self.attr_features = attr_features
            
            # 步骤1: 预分组数据（避免重复分组）
            self._precompute_groups()
            
            # 步骤2: 构建属性字典
            self._build_attribute_dictionary()
            
            # 步骤3: 初始化标准化器
            self._initialize_scalers()
            
            # 步骤4: 预标准化属性
            self._precompute_standardized_attributes()
            
            # 步骤5: 可选的预计算滑动窗口
            if self.enable_precompute:
                self._precompute_sliding_windows()
            
            self.initialized = True
            logging.info("DataHandler简化初始化完成")
            logging.info(f"预分组河段数: {len(self._cached_groups)}")
            if self.enable_precompute:
                logging.info(f"预计算窗口河段数: {len(self._cached_windows)}")
    
    def _precompute_groups(self):
        """预分组数据，避免后续重复分组"""
        with TimingAndMemoryContext("预分组数据"):
            self._cached_groups = {
                comid: group.sort_values("date").copy().reset_index(drop=True)
                for comid, group in self.df.groupby("COMID")
            }
            logging.info(f"预分组完成：{len(self._cached_groups)} 个河段")
    
    def _build_attribute_dictionary(self):
        """构建原始属性字典"""
        with TimingAndMemoryContext("构建属性字典"):
            self._raw_attr_dict = {}
            for row in self.attr_df.itertuples(index=False):
                comid = str(row.COMID)
                row_dict = row._asdict()
                attrs = []
                for attr in self.attr_features:
                    if attr in row_dict:
                        attrs.append(row_dict[attr])
                    else:
                        attrs.append(0.0)
                self._raw_attr_dict[comid] = np.array(attrs, dtype=np.float32)
            
            logging.info(f"属性字典构建完成：{len(self._raw_attr_dict)} 个河段")
    
    def _initialize_scalers(self):
        """初始化标准化器，确保一致性"""
        with TimingAndMemoryContext("初始化标准化器"):
            # 使用全部数据创建标准化器，确保一致性
            sample_data = self._get_sample_data(use_all_data=True)
            if sample_data is not None:
                X_sample, attr_dict_sample = sample_data
                _, self.ts_scaler = standardize_time_series_all(X_sample)
                _, self.attr_scaler = standardize_attributes(attr_dict_sample)
                logging.info("标准化器初始化完成")
            else:
                logging.warning("无法初始化标准化器")
    
    def _precompute_standardized_attributes(self):
        """预标准化所有属性，避免重复标准化"""
        if self.attr_scaler is None:
            logging.warning("属性标准化器未初始化，跳过预标准化")
            return
            
        with TimingAndMemoryContext("预标准化属性"):
            # 标准化所有属性
            attr_matrix = np.vstack([self._raw_attr_dict[k] for k in self._raw_attr_dict.keys()])
            attr_matrix_scaled = self.attr_scaler.transform(attr_matrix)
            self._standardized_attr_dict = {
                k: attr_matrix_scaled[i] 
                for i, k in enumerate(self._raw_attr_dict.keys())
            }
            logging.info("属性预标准化完成")
    
    def _precompute_sliding_windows(self, time_window=10):
        """预计算滑动窗口（顺序处理，内存换时间）"""
        with TimingAndMemoryContext("预计算滑动窗口"):
            for comid, group in self._cached_groups.items():
                input_data = group[self.input_features].values
                windows = []
                dates = []
                
                for i in range(len(input_data) - time_window + 1):
                    window = input_data[i:i + time_window]
                    if not np.isnan(window).any():
                        windows.append(window)
                        dates.append(group['date'].iloc[i + time_window - 1])
                
                if windows:
                    X = np.array(windows, dtype=np.float32)
                    # 立即标准化
                    N, T, D = X.shape
                    X_2d = X.reshape(-1, D)
                    X_scaled_2d = self.ts_scaler.transform(X_2d)
                    X_scaled = X_scaled_2d.reshape(N, T, D)
                    
                    self._cached_windows[comid] = {
                        'X': X,
                        'X_scaled': X_scaled,
                        'dates': dates
                    }
            
            logging.info(f"预计算滑动窗口完成：{len(self._cached_windows)} 个河段")
    
    def _get_sample_data(self, use_all_data=True) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """获取样本数据用于创建标准化器"""
        if self._cached_groups is None or not self._raw_attr_dict:
            return None
        
        # 选择COMID
        if use_all_data:
            sample_comids = list(self._cached_groups.keys())
            logging.info(f"使用全部 {len(sample_comids)} 个COMID数据创建标准化器")
        else:
            sample_comids = list(self._cached_groups.keys())[:min(10, len(self._cached_groups))]
            logging.info(f"使用 {len(sample_comids)} 个样本COMID创建标准化器")
        
        # 使用预分组的数据构建样本
        sample_df = pd.concat([
            self._cached_groups[comid].assign(COMID=comid) 
            for comid in sample_comids
        ], ignore_index=True)
        
        # 构建样本滑动窗口
        X_sample, _, _, _ = build_sliding_windows_for_subset(
            sample_df, 
            sample_comids, 
            input_cols=self.input_features, 
            target_col="TN",               
            all_target_cols=["TN", "TP"],  
            time_window=10,
            skip_missing_targets=True      
        )
        
        if X_sample is None:
            logging.warning("无法创建滑动窗口样本数据")
            return None
        
        # 获取样本属性字典
        attr_dict_sample = {
            k: v for k, v in self._raw_attr_dict.items() 
            if k in [str(c) for c in sample_comids]
        }
        
        logging.info(f"样本数据创建完成: X_sample.shape={X_sample.shape}")
        return X_sample, attr_dict_sample
    
    def get_groups(self, comid_list=None):
        """获取预分组数据"""
        if comid_list is None:
            return self._cached_groups
        else:
            return {
                comid: self._cached_groups[comid] 
                for comid in comid_list 
                if comid in self._cached_groups
            }
    
    def get_standardized_data(self, 
                             comid_list: List, 
                             all_target_cols: List[str],
                             target_col: str,
                             time_window: int = 10,
                             skip_missing_targets: bool = True) -> Tuple:
        """
        获取标准化的训练数据
        
        自动选择最优路径：预计算缓存 vs 实时计算
        """
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")
        
        # 检查是否可以使用预计算的窗口
        if (self.enable_precompute and 
            time_window == 10 and 
            all(comid in self._cached_windows for comid in comid_list)):
            return self._get_data_from_cache(comid_list, target_col, skip_missing_targets)
        else:
            return self._get_data_realtime(comid_list, all_target_cols, target_col, time_window, skip_missing_targets)
    
    def _get_data_from_cache(self, comid_list, target_col, skip_missing_targets):
        """从预计算缓存中获取数据（快速路径）"""
        with TimingAndMemoryContext("从缓存获取数据"):
            X_list = []
            Y_list = []
            comid_track = []
            date_track = []
            
            # 从预计算的窗口中提取数据
            for comid in comid_list:
                if comid not in self._cached_windows:
                    continue
                    
                windows_data = self._cached_windows[comid]
                X_scaled = windows_data['X_scaled']
                dates = windows_data['dates']
                
                # 获取对应的目标值
                group = self._cached_groups[comid]
                for i, date in enumerate(dates):
                    # 查找对应日期的目标值
                    target_row = group[group['date'] == date]
                    if not target_row.empty:
                        y_value = target_row[target_col].iloc[0]
                        if skip_missing_targets and np.isnan(y_value):
                            continue
                        X_list.append(X_scaled[i])
                        Y_list.append(y_value)
                        comid_track.append(comid)
                        date_track.append(date)
            
            if not X_list:
                return None, None, None, None, None
            
            X_ts_scaled = np.array(X_list)
            Y = np.array(Y_list)
            COMIDs = np.array(comid_track)
            Dates = np.array(date_track)
            
            # 获取标准化属性字典
            attr_dict_scaled = {
                k: v for k, v in self._standardized_attr_dict.items() 
                if k in [str(c) for c in comid_list]
            }
            
            return X_ts_scaled, attr_dict_scaled, Y, COMIDs, Dates
    
    def _get_data_realtime(self, comid_list, all_target_cols, target_col, time_window, skip_missing_targets):
        """实时计算数据（使用预分组数据优化）"""
        with TimingAndMemoryContext("实时构建数据"):
            # 使用预分组的数据，避免重新分组
            subset_df = pd.concat([
                self._cached_groups[comid].assign(COMID=comid) 
                for comid in comid_list 
                if comid in self._cached_groups
            ], ignore_index=True)
            
            # 构建滑动窗口
            X_ts, Y, COMIDs, Dates = build_sliding_windows_for_subset(
                subset_df, 
                comid_list, 
                input_cols=self.input_features, 
                all_target_cols=all_target_cols,
                target_col=target_col,
                time_window=time_window,
                skip_missing_targets=skip_missing_targets
            )
            
            if X_ts is None:
                return None, None, None, None, None
            
            # 标准化时间序列数据
            N, T, input_dim = X_ts.shape
            X_ts_2d = X_ts.reshape(-1, input_dim)
            X_ts_scaled_2d = self.ts_scaler.transform(X_ts_2d)
            X_ts_scaled = X_ts_scaled_2d.reshape(N, T, input_dim)
            
            # 获取标准化的属性字典
            comid_strs = [str(comid) for comid in COMIDs]
            attr_dict_scaled = {
                k: v for k, v in self._standardized_attr_dict.items() 
                if k in set(comid_strs)
            }
            
            return X_ts_scaled, attr_dict_scaled, Y, COMIDs, Dates
    
    def get_standardized_attr_dict(self) -> Dict[str, np.ndarray]:
        """获取标准化的完整属性字典"""
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")
        return self._standardized_attr_dict.copy()
    
    def prepare_training_data_for_head_segments(self,
                                              comid_wq_list: List,
                                              comid_era5_list: List,
                                              all_target_cols: List[str],
                                              target_col: str,
                                              output_dir: str,
                                              model_version: str) -> Tuple:
        """为头部河段准备训练数据（优化版）"""
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")

        # 筛选头部河段
        with TimingAndMemoryContext("寻找头部站点"):
            # 从预分组数据中筛选，避免重新分组
            available_comids = set(self._cached_groups.keys())
            comid_list_head = list(
                available_comids & set(comid_wq_list) & set(comid_era5_list)
            )
            
            # 保存头部河段COMID列表
            np.save(f"{output_dir}/comid_list_head_{model_version}.npy", comid_list_head)
            
            if len(comid_list_head) == 0:
                logging.warning("警告：找不到符合条件的头部河段")
                return None, None, None, None, None
            
            logging.info(f"选择的头部河段数量：{len(comid_list_head)}")

        # 获取标准化的训练数据
        result = self.get_standardized_data(
            comid_list_head,
            all_target_cols,
            target_col,
            time_window=10
        )
        
        X_ts_scaled, attr_dict_scaled, Y, COMIDs, Dates = result
        
        if X_ts_scaled is None:
            return None, None, None, None, None
        
        # 输出数据维度信息
        logging.info(f"头部河段训练数据: X_ts_scaled.shape = {X_ts_scaled.shape}")
        logging.info(f"Y.shape = {Y.shape}, COMIDs.shape = {COMIDs.shape}")

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
        优化版的批量预测数据准备
        
        使用预分组数据和可选的预计算窗口显著提升性能
        """
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")
        
        # 如果启用了预计算且有缓存数据，使用快速路径
        if (self.enable_precompute and 
            all(comid in self._cached_windows for comid in comid_batch)):
            return self._prepare_batch_from_cache(comid_batch)
        else:
            return self._prepare_batch_realtime(comid_batch, all_target_cols, target_col)
    
    def _prepare_batch_from_cache(self, comid_batch):
        """从预计算缓存准备批量数据（快速路径）"""
        with TimingAndMemoryContext("从缓存准备批量数据"):
            X_ts_list = []
            comid_indices = {}
            valid_comids = []
            
            current_idx = 0
            for comid in comid_batch:
                if comid not in self._cached_windows:
                    continue
                
                windows_data = self._cached_windows[comid]
                X_scaled = windows_data['X_scaled']
                dates = windows_data['dates']
                
                end_idx = current_idx + len(X_scaled)
                comid_indices[comid] = (current_idx, end_idx, dates, dates)
                current_idx = end_idx
                valid_comids.append(comid)
                
                X_ts_list.append(X_scaled)
            
            if not X_ts_list:
                return None
            
            # 堆叠数据（已经标准化）
            X_ts_batch = np.vstack(X_ts_list)
            
            # 准备属性数据
            attr_dim = next(iter(self._standardized_attr_dict.values())).shape[0]
            X_attr_batch = np.zeros((X_ts_batch.shape[0], attr_dim), dtype=np.float32)
            
            # 为每个样本分配正确的属性向量
            sample_idx = 0
            for comid in valid_comids:
                start_idx, end_idx, _, _ = comid_indices[comid]
                batch_size = end_idx - start_idx
                
                attr_vec = self._standardized_attr_dict.get(
                    str(comid), 
                    np.zeros(attr_dim, dtype=np.float32)
                )
                X_attr_batch[sample_idx:sample_idx + batch_size] = attr_vec
                sample_idx += batch_size
            
            return {
                'X_ts_scaled': X_ts_batch,
                'X_attr_batch': X_attr_batch,
                'valid_comids': valid_comids,
                'comid_indices': comid_indices,
                'groups': {}  # 不再需要groups
            }
    
    def _prepare_batch_realtime(self, comid_batch, all_target_cols, target_col):
        """实时准备批量数据（使用预分组数据优化）"""
        with TimingAndMemoryContext("实时准备批量数据"):
            # 使用预分组的数据，避免重新分组
            filtered_groups = self.get_groups(comid_batch)
            
            # 收集所有数据
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
                if comid not in filtered_groups:
                    continue
                    
                group = filtered_groups[comid]
                
                # 从预分组数据创建临时DataFrame
                temp_df = group.copy()
                temp_df['COMID'] = comid  # 确保COMID列存在
                
                X_ts_local, _, _, dates_local = build_sliding_windows_for_subset(
                    df=temp_df, 
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
                all_data['comid_indices'][comid] = (
                    current_idx, end_idx, dates_local, group['date'].tolist()
                )
                current_idx = end_idx
                valid_comids.append(comid)
                
                all_data['X_ts_list'].append(X_ts_local)
                all_data['comids'].extend([comid] * X_ts_local.shape[0])
                all_data['dates'].extend(dates_local)
            
            if not all_data['X_ts_list']:
                return None
            
            # 堆叠并标准化X_ts数据
            X_ts_batch = np.vstack(all_data['X_ts_list'])
            N, T, input_dim = X_ts_batch.shape
            X_ts_2d = X_ts_batch.reshape(-1, input_dim)
            X_ts_scaled_2d = self.ts_scaler.transform(X_ts_2d)
            X_ts_scaled = X_ts_scaled_2d.reshape(N, T, input_dim)
            
            # 准备属性数据（使用预标准化的数据）
            attr_dim = next(iter(self._standardized_attr_dict.values())).shape[0]
            X_attr_batch = np.zeros((X_ts_scaled.shape[0], attr_dim), dtype=np.float32)
            
            for i, comid in enumerate(all_data['comids']):
                comid_str = str(comid)
                attr_vec = self._standardized_attr_dict.get(
                    comid_str, 
                    np.zeros(attr_dim, dtype=np.float32)
                )
                X_attr_batch[i] = attr_vec
            
            return {
                'X_ts_scaled': X_ts_scaled,
                'X_attr_batch': X_attr_batch,
                'valid_comids': valid_comids,
                'comid_indices': all_data['comid_indices'],
                'groups': filtered_groups
            }
    
    def prepare_next_iteration_data(self,
                                   df_flow: pd.DataFrame,
                                   target_col: str,
                                   col_y_n: str,
                                   col_y_up: str,
                                   time_window: int = 10) -> Tuple:
        """
        准备下一轮迭代的训练数据（使用预分组数据优化）
        """
        if not self.initialized:
            raise ValueError("数据处理器尚未初始化")
            
        with TimingAndMemoryContext("准备下一轮迭代数据"):
            # 标准化列名
            df_flow_copy = df_flow.copy()
            if 'Date' in df_flow_copy.columns:
                df_flow_copy = df_flow_copy.rename(columns={'Date': 'date'})
            
            # 检查必要的列
            required_cols = ['COMID', 'date', col_y_n, col_y_up]
            for col in required_cols:
                if col not in df_flow_copy.columns:
                    logging.error(f"缺少必要的列: {col}")
                    return None, None, None, None, None
            
            # 使用预分组数据进行合并，避免重新分组
            merged_groups = {}
            for comid, group in self._cached_groups.items():
                # 为每个组添加来自df_flow的信息
                flow_info = df_flow_copy[df_flow_copy['COMID'] == comid][required_cols]
                if not flow_info.empty:
                    merged_group = pd.merge(group, flow_info, on=['COMID', 'date'], how='left')
                    merged_group["E_label"] = merged_group[target_col] - merged_group[col_y_up]
                    merged_groups[comid] = merged_group
            
            # 构建数据
            comid_list = list(merged_groups.keys())
            if not comid_list:
                logging.error("没有找到可用的河段数据")
                return None, None, None, None, None
            
            # 合并所有组
            merged_df = pd.concat(merged_groups.values(), ignore_index=True)
            
            # 构建滑动窗口
            X_ts, _, COMIDs, Dates = build_sliding_windows_for_subset(
                merged_df,
                comid_list,
                input_cols=self.input_features,
                target_cols=["E_label"],
                time_window=time_window
            )
            
            if X_ts is None:
                return None, None, None, None, None
            
            # 标准化数据
            N, T, input_dim = X_ts.shape
            X_ts_2d = X_ts.reshape(-1, input_dim)
            X_ts_scaled_2d = self.ts_scaler.transform(X_ts_2d)
            X_ts_scaled = X_ts_scaled_2d.reshape(N, T, input_dim)
            
            # 获取标准化属性字典
            attr_dict_scaled = self._standardized_attr_dict.copy()

            # 提取E_label值
            Y_label = []
            for cid, date_val in zip(COMIDs, Dates):
                if cid in merged_groups:
                    subset = merged_groups[cid][
                        (merged_groups[cid]["COMID"] == cid) & 
                        (merged_groups[cid]["date"] == date_val)
                    ]
                    if not subset.empty:
                        label_val = subset["E_label"].iloc[0]
                    else:
                        label_val = 0.0
                else:
                    label_val = 0.0
                Y_label.append(label_val)
            
            Y_label = np.array(Y_label, dtype=np.float32)
            
            return X_ts_scaled, attr_dict_scaled, Y_label, COMIDs, Dates