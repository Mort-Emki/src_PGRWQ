"""
evaluation.py - 评估和收敛检查模块

该模块提供评估模型性能、检查训练收敛性以及验证数据质量的功能。
负责监控训练进度、检测异常值并确保数据一致性。
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Set


class ConvergenceChecker:
    """收敛性检查器，负责监控迭代训练的收敛情况"""
    
    def __init__(self, epsilon: float = 0.01, stability_threshold: float = 0.01, 
                history_window: int = 3):
        """
        初始化收敛性检查器
        
        参数:
            epsilon: 收敛阈值，当误差小于此值时认为收敛
            stability_threshold: 稳定性阈值，当误差变化率小于此值时认为趋势稳定
            history_window: 检查误差趋势稳定性时考虑的历史窗口大小
        """
        self.epsilon = epsilon
        self.stability_threshold = stability_threshold
        self.history_window = history_window
        self.error_history = []
        
    def check_convergence(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         iteration: int) -> Tuple[bool, Dict[str, float]]:
        """
        检查当前迭代是否达到收敛条件
        
        参数:
            y_true: 真实值数组
            y_pred: 预测值数组
            iteration: 当前迭代次数
            
        返回:
            (converged, stats): 是否收敛和统计信息
        """
        # 检查y_true是否存在有效值
        valid_mask = ~np.isnan(y_true)
        valid_count = np.sum(valid_mask)
        
        if valid_count == 0:
            logging.warning("警告：没有有效的观测数据，无法评估收敛性")
            return False, None
        
        # 只使用有效数据计算残差
        valid_y_true = y_true[valid_mask]
        valid_y_pred = y_pred[valid_mask]
        residual = valid_y_true - valid_y_pred
        
        # 计算误差统计量（安全方式）
        mae = np.mean(np.abs(residual))
        
        # 安全计算MSE
        try:
            residual_64 = residual.astype(np.float64)
            mse = np.mean(np.square(residual_64))
        except:
            # 溢出处理：逐元素计算
            mse = np.mean([float(r)**2 for r in residual])
        
        # 计算RMSE和最大残差
        rmse = np.sqrt(mse)
        max_resid = np.max(np.abs(residual))
        
        # 汇总统计信息
        stats = {
            'iteration': iteration,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'max_resid': max_resid,
            'valid_data_points': valid_count
        }
        
        # 记录误差历史
        self.error_history.append(stats)
        
        # 输出误差信息
        logging.info(f"迭代 {iteration+1} 误差统计 (基于 {valid_count} 个有效观测点):")
        logging.info(f"  平均绝对误差 (MAE): {mae:.4f}")
        logging.info(f"  均方误差 (MSE): {mse:.4f}")
        logging.info(f"  均方根误差 (RMSE): {rmse:.4f}")
        logging.info(f"  最大绝对残差: {max_resid:.4f}")
        
        # 检查收敛条件
        if mae < self.epsilon:
            logging.info(f"收敛! 平均绝对误差 ({mae:.4f}) 小于阈值 ({self.epsilon})")
            return True, stats
        
        # 检查误差趋势是否稳定
        if self.check_error_trend_stability():
            return True, stats
            
        return False, stats
    
    def check_error_trend_stability(self) -> bool:
        """
        检查误差趋势是否稳定
        
        返回:
            是否因误差趋势稳定而收敛
        """
        if len(self.error_history) < self.history_window:
            return False
            
        # 获取最近几轮的误差
        recent_errors = [entry['mae'] for entry in self.error_history[-self.history_window:]]
        
        # 计算误差变化率
        error_changes = []
        for i in range(1, len(recent_errors)):
            prev_error = recent_errors[i-1]
            if prev_error > 0:
                change = (prev_error - recent_errors[i]) / prev_error
                error_changes.append(change)
        
        # 检查变化率是否都小于阈值
        if error_changes and all(abs(change) < self.stability_threshold for change in error_changes):
            logging.info(f"收敛! 误差变化趋于稳定，最近几轮MAE: {recent_errors}")
            return True
        
        return False
    
    def get_error_history(self) -> List[Dict[str, float]]:
        """获取完整的误差历史记录"""
        return self.error_history


class DataValidator:
    """数据验证器，负责检查数据质量和一致性"""
    
    def __init__(self, max_abnormal_value: float = 1e6, 
                max_allowed_percent: float = 1.0):
        """
        初始化数据验证器
        
        参数:
            max_abnormal_value: 允许的最大异常值绝对值
            max_allowed_percent: 允许的最大异常比例（百分比）
        """
        self.max_abnormal_value = max_abnormal_value
        self.max_allowed_percent = max_allowed_percent
        
    def check_dataframe_abnormalities(self, 
                                     df: pd.DataFrame, 
                                     iteration: int, 
                                     target_cols: List[str]) -> Tuple[bool, Dict]:
        """
        检查DataFrame中是否存在异常值
        
        参数:
            df: 要检查的DataFrame
            iteration: 当前迭代次数
            target_cols: 目标列列表
            
        返回:
            (is_valid, report): 数据是否有效和异常报告
        """
        logging.info(f"检查迭代 {iteration} 的结果是否有异常值...")
        
        # 初始化报告
        report = {
            "迭代": iteration,
            "总行数": len(df),
            "检查列": [],
            "异常计数": {},
            "NaN计数": {},
            "无穷值计数": {},
            "极端值计数": {},
            "最大值": {},
            "最小值": {},
            "是否有效": True
        }
        
        # 定义要检查的列
        cols_to_check = []
        for param in target_cols:
            # 添加E, y_up, y_n列
            cols_to_check.extend([
                f'E_{iteration}_{param}',
                f'y_up_{iteration}_{param}',
                f'y_n_{iteration}_{param}'
            ])
            
            # 添加调试列
            debug_cols = [col for col in df.columns 
                         if col.startswith(f'debug_') and col.endswith(f'_{param}')]
            cols_to_check.extend(debug_cols)
        
        # 检查Qout列（常见问题来源）
        if 'Qout' in df.columns:
            cols_to_check.append('Qout')
        
        report["检查列"] = cols_to_check
        
        # 检查每列
        for col in cols_to_check:
            if col not in df.columns:
                logging.warning(f"列 {col} 在数据中不存在，跳过检查")
                continue
                
            # 获取列值
            values = df[col]
            
            # 检查NaN值
            nan_count = values.isna().sum()
            report["NaN计数"][col] = nan_count
            
            # 检查无穷值
            inf_mask = ~values.isna() & (values.abs() == float('inf'))
            inf_count = inf_mask.sum()
            report["无穷值计数"][col] = inf_count
            
            # 检查极端值（排除NaN和无穷）
            valid_values = values.dropna()
            valid_values = valid_values[valid_values.abs() != float('inf')]
            
            extreme_count = (valid_values.abs() > self.max_abnormal_value).sum()
            report["极端值计数"][col] = extreme_count
            
            # 计算异常比例
            total_abnormal = nan_count + inf_count + extreme_count
            abnormal_percent = (total_abnormal / len(df) * 100) if len(df) > 0 else 0
            report["异常计数"][col] = total_abnormal
            
            # 获取最大和最小值
            if not valid_values.empty:
                report["最大值"][col] = valid_values.max()
                report["最小值"][col] = valid_values.min()
            
            # 记录异常
            if abnormal_percent > 0.01:  # 大于0.01%就记录
                logging.warning(f"列 {col} 包含 {total_abnormal} 个异常值 ({abnormal_percent:.2f}%): "
                              f"{nan_count} 个NaN, {inf_count} 个无穷值, {extreme_count} 个极端值")
                
                # 检查异常比例是否超过允许的阈值
                if abnormal_percent > self.max_allowed_percent:
                    report["是否有效"] = False
                    logging.error(f"列 {col} 异常值过多! {abnormal_percent:.2f}% 超出阈值 {self.max_allowed_percent}%")
        
        # 记录最终结果
        if report["是否有效"]:
            logging.info(f"迭代 {iteration} 数据检查通过，异常值在可接受范围内")
        else:
            logging.error(f"迭代 {iteration} 数据检查失败，包含过多异常值")
        
        return report["是否有效"], report
    
    def validate_data_coherence(self, 
                              df: pd.DataFrame, 
                              df_flow: pd.DataFrame, 
                              input_cols: List[str], 
                              target_cols: List[str],
                              iteration: int) -> bool:
        """
        验证原始数据和流结果之间的一致性
        
        参数:
            df: 原始数据DataFrame
            df_flow: 流路由结果DataFrame
            input_cols: 输入特征列
            target_cols: 目标列
            iteration: 当前迭代
            
        返回:
            是否一致
        """
        logging.info("数据一致性验证开始")
        is_coherent = True
        
        # 检查公共COMID
        df_comids = set(df['COMID'].unique())
        flow_comids = set(df_flow['COMID'].unique())
        
        # 检查流结果中不在原始数据中的COMID
        missing_comids = flow_comids - df_comids
        if missing_comids:
            logging.warning(f"警告: 流结果中有 {len(missing_comids)} 个COMID不在原始数据中")
            is_coherent = False
        
        # 检查日期格式
        date_col_df = self._find_date_column(df)
        date_col_flow = self._find_date_column(df_flow)
        
        if date_col_df != date_col_flow:
            logging.warning(f"警告: 日期列名不匹配 - '{date_col_df}' vs '{date_col_flow}'")
            is_coherent = False
        
        # 检查列名问题
        for param in target_cols:
            expected_cols = [
                f'E_{iteration}_{param}', 
                f'y_up_{iteration}_{param}', 
                f'y_n_{iteration}_{param}'
            ]
            missing_cols = [col for col in expected_cols if col not in df_flow.columns]
            if missing_cols:
                logging.warning(f"警告: 流结果中缺少预期列: {missing_cols}")
                is_coherent = False
        
        # 检查输入列不一致
        if input_cols:
            missing_inputs = [col for col in input_cols if col not in df.columns]
            if missing_inputs:
                logging.warning(f"警告: 原始数据中缺少输入列: {missing_inputs}")
                is_coherent = False
        
        # 输出总体评估
        logging.info(f"数据一致性检查: {'通过' if is_coherent else '失败'}")
        return is_coherent
    
    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """在DataFrame中查找日期列"""
        for col in ['date', 'Date']:
            if col in df.columns:
                return col
        return None
    
    def fix_dataframe_abnormalities(self, 
                                  df: pd.DataFrame, 
                                  iteration: int,
                                  target_cols: List[str],
                                  reasonable_max: float = 100.0) -> pd.DataFrame:
        """
        修复DataFrame中的异常值
        
        参数:
            df: 要修复的DataFrame
            iteration: 当前迭代
            target_cols: 目标列
            reasonable_max: 合理的最大值
            
        返回:
            修复后的DataFrame
        """
        df_fixed = df.copy()
        
        for param in target_cols:
            # 修复E值
            e_col = f'E_{iteration}_{param}'
            if e_col in df_fixed.columns:
                df_fixed[e_col] = df_fixed[e_col].clip(-reasonable_max, reasonable_max)
                logging.info(f"已将 {e_col} 列限制在 ±{reasonable_max} 范围内")
                
            # 修复y_up值
            y_up_col = f'y_up_{iteration}_{param}'
            if y_up_col in df_fixed.columns:
                df_fixed[y_up_col] = df_fixed[y_up_col].clip(-reasonable_max, reasonable_max)
                logging.info(f"已将 {y_up_col} 列限制在 ±{reasonable_max} 范围内")
                
            # 重新计算y_n值
            y_n_col = f'y_n_{iteration}_{param}'
            if y_n_col in df_fixed.columns and e_col in df_fixed.columns and y_up_col in df_fixed.columns:
                df_fixed[y_n_col] = df_fixed[e_col] + df_fixed[y_up_col]
                logging.info(f"已重新计算 {y_n_col} 列")
        
        return df_fixed