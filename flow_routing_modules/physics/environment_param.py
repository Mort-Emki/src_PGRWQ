"""
environment.py - 环境调整因子模块

本模块提供了一系列函数，用于计算环境因素（如温度、浓度）对水质参数转化过程的影响。
这些调整因子对于准确模拟河道中的生物地球化学过程至关重要。

主要功能:
1. 计算温度对吸收速率的影响
2. 计算氮浓度对反硝化过程的影响
3. 计算河段间的物质保留系数

参考文献:
- 温度调整公式基于Arrhenius方程
- 氮浓度调整公式基于实验数据拟合
"""

import numpy as np
import pandas as pd
import logging


def compute_temperature_factor(temperature: pd.Series, parameter: str = "TN") -> pd.Series:
    """
    计算温度调整因子
    
    使用公式 f(t) = α^(t-20) 计算温度对吸收速率的影响，其中：
    - t是水温(°C)
    - α是温度系数，TN为1.0717，TP为1.06
    - 20°C是参考温度
    
    参数:
        temperature: 温度序列(°C)
        parameter: 水质参数，"TN"或"TP"
        
    返回:
        温度调整因子序列
    """
    if parameter == "TN":
        alpha = 1.0717  # TN的α值
    else:  # TP
        alpha = 1.06    # TP的α值
    
    # 计算 α^(t-20)
    return np.power(alpha, temperature - 20) 


def compute_nitrogen_concentration_factor(N_concentration: pd.Series) -> pd.Series:
    """
    计算氮浓度调整因子
    
    基于以下实验观测值构建的分段函数:
    - 当CN = 0.0001 mg/L时，f(CN) = 7.2
    - 当CN = 1 mg/L时，f(CN) = 1
    - 当CN = 100 mg/L时，f(CN) = 0.37
    - 浓度更高时保持不变
    
    在各指定浓度点之间使用对数线性插值。
    这一调整反映了高氮负载条件下由于电子供体限制导致的反硝化效率下降。
    
    参数:
        N_concentration: 氮浓度序列(mg/L)
    
    返回:
        浓度调整因子序列
    """
    # 创建副本以避免修改原始数据
    result = pd.Series(index=N_concentration.index, dtype=float)
    
    # 情况1: CN ≤ 0.0001 mg/L
    mask_lowest = N_concentration <= 0.0001
    result[mask_lowest] = 7.2
    
    # 情况2: 0.0001 < CN ≤ 1 mg/L
    mask_low = (N_concentration > 0.0001) & (N_concentration <= 1)
    # (0.0001, 7.2) 和 (1, 1) 之间的对数线性插值
    log_ratio_low = (np.log10(N_concentration[mask_low]) - np.log10(0.0001)) / (np.log10(1) - np.log10(0.0001))
    result[mask_low] = 7.2 - log_ratio_low * (7.2 - 1)
    
    # 情况3: 1 < CN ≤ 100 mg/L
    mask_mid = (N_concentration > 1) & (N_concentration <= 100)
    # (1, 1) 和 (100, 0.37) 之间的对数线性插值
    log_ratio_mid = (np.log10(N_concentration[mask_mid]) - np.log10(1)) / (np.log10(100) - np.log10(1))
    result[mask_mid] = 1 - log_ratio_mid * (1 - 0.37)
    
    # 情况4: CN > 100 mg/L
    mask_high = N_concentration > 100
    result[mask_high] = 0.37
    
    return result


def compute_retainment_factor(v_f: float, Q_up: pd.Series, Q_down: pd.Series, 
                            W_up: pd.Series, W_down: pd.Series,
                            length_up: float, length_down: float,
                            temperature: pd.Series = None,
                            N_concentration: pd.Series = None,
                            parameter: str = "TN") -> pd.Series:
    """
    计算上下游河段间的物质保留系数
    
    保留系数公式: 
    R(Ωj, Ωi) = exp(-v_f·S(Ωj)/(2·Q(Ωj))) · exp(-v_f·S(Ωi)/(2·Q(Ωi)))
    
    其中:
    - v_f 是吸收速率(m/yr)，会根据温度和浓度进行调整
    - S 是河段面积(m²)
    - Q 是流量(m³/s)
    
    参数:
        v_f: 基础吸收速率参数(m/yr)
        Q_up: 上游流量序列(m³/s)
        Q_down: 下游流量序列(m³/s)
        W_up: 上游河道宽度序列(m)
        W_down: 下游河道宽度序列(m)
        length_up: 上游河段长度(km)
        length_down: 下游河段长度(km)
        temperature: 温度序列(°C)，如果提供则计算温度调整
        N_concentration: 氮浓度序列(mg/L)，如果提供且参数为TN，则计算浓度调整
        parameter: 水质参数，"TN"或"TP"

    返回:
        保留系数序列(0-1之间)，表示从上游传输到下游的物质比例
    """
    # 1. 单位转换和安全处理
    min_flow = 0.001  # 最小流量阈值，1 L/s
    Q_up_adj = Q_up.replace(0, min_flow).clip(lower=min_flow)
    Q_down_adj = Q_down.replace(0, min_flow).clip(lower=min_flow)
    
    # 长度从km转成m
    length_up_m = length_up * 1000.0
    length_down_m = length_down * 1000.0
    
    # v_f从m/yr转成m/s
    seconds_per_year = 365.25 * 24 * 60 * 60
    v_f_m_per_second = v_f / seconds_per_year
    
    # 2. 应用环境调整因子
    # 温度调整
    if temperature is not None:
        temp_factor = compute_temperature_factor(temperature, parameter)
        v_f_adjusted = v_f_m_per_second * temp_factor
    else:
        v_f_adjusted = v_f_m_per_second
    
    # 浓度调整（仅适用于TN）
    if parameter == "TN" and N_concentration is not None:
        conc_factor = compute_nitrogen_concentration_factor(N_concentration)
        v_f_adjusted = v_f_adjusted * conc_factor
    
    # 3. 计算河段面积
    S_up = W_up * length_up_m
    S_down = W_down * length_down_m
    
    # 4. 计算指数项，限制范围避免数值溢出
    exp_up = (-v_f_adjusted * S_up / (2 * Q_up_adj)).clip(-50, 50)
    exp_down = (-v_f_adjusted * S_down / (2 * Q_down_adj)).clip(-50, 50)
    
    # 5. 计算保留系数
    R_up = np.exp(exp_up)
    R_down = np.exp(exp_down)
    R = R_up * R_down
    
    # 6. 填充可能的NaN值
    return R.fillna(0.0)


def compute_hydraulic_residence_time(length_m, width_m, depth_m, flow_m3s):
    """
    计算水力停留时间
    
    公式: τ = V/Q，其中V是体积，Q是流量
    
    参数:
        length_m: 河段长度(m)
        width_m: 河道宽度(m)
        depth_m: 河道深度(m)
        flow_m3s: 流量(m³/s)
        
    返回:
        水力停留时间(s)
    """
    # 计算河段体积(m³)
    volume = length_m * width_m * depth_m
    
    # 避免除零错误
    flow_safe = np.maximum(flow_m3s, 0.001)
    
    # 计算停留时间(s)
    residence_time = volume / flow_safe
    
    return residence_time


def compute_phosphorus_sorption(suspended_solids, phosphorus_conc, temperature=20):
    """
    计算磷的吸附过程
    
    磷在河道中的吸附过程，考虑悬浮固体和温度的影响
    
    参数:
        suspended_solids: 悬浮固体浓度(mg/L)
        phosphorus_conc: 磷浓度(mg/L)
        temperature: 水温(°C)，默认20°C
        
    返回:
        吸附速率(1/day)
    """
    # 基础吸附系数
    base_sorption_rate = 0.3
    
    # 温度调整
    temp_factor = compute_temperature_factor(pd.Series([temperature]), "TP").iloc[0]
    
    # 悬浮固体浓度影响（简化模型）
    ss_factor = np.minimum(1.0, suspended_solids / 100.0)
    
    # 浓度影响（Langmuir等温线简化）
    conc_factor = 1.0 / (1.0 + phosphorus_conc)
    
    # 计算总吸附速率
    sorption_rate = base_sorption_rate * temp_factor * ss_factor * conc_factor
    
    return sorption_rate