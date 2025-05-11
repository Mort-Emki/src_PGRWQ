"""
geometry.py - 河道几何特性模块

本模块提供了处理河道几何特性的功能，包括获取河段长度和计算河道宽度。
这些几何特性是汇流计算中的关键输入参数，影响河段间的物质运移和转化。

主要功能:
1. 获取河段长度
2. 基于流量计算河道宽度

参考文献:
- 河道宽度计算公式 W = aQ^b 基于Raymond et al. (2012) 实际上来源于Leopold和Maddock在1953年出版的经典著作
《The Hydraulic Geometry of Stream Channels and Some Physiographic Implications》
"""

import numpy as np
import pandas as pd
import logging


def get_river_length(comid, attr_dict, river_info=None):
    """
    获取指定COMID的河段长度
    
    本函数尝试从多个数据源获取河段长度信息，优先顺序为：
    1. 属性字典(attr_dict)中的'lengthkm'
    2. 河网信息(river_info)中的'lengthkm'
    3. 如果都找不到，则使用默认值1.0 km
    
    参数:
        comid: 河段的COMID(字符串或整数)
        attr_dict: COMID到属性数组的映射字典
        river_info: 包含河网信息的DataFrame(可选)
        
    返回:
        float: 河段长度(km)
    """
    comid_str = str(comid)
    
    # 首先尝试从attr_dict获取长度
    if comid_str in attr_dict:
        # 获取该COMID的所有属性值
        attrs = attr_dict[comid_str]
        
        # 检查'lengthkm'是否在属性字典中
        if isinstance(attrs, dict) and 'lengthkm' in attrs:
            return float(attrs['lengthkm'])
    
    # 如果提供了river_info，尝试从中获取长度
    if river_info is not None and 'COMID' in river_info.columns and 'lengthkm' in river_info.columns:
        length_data = river_info[river_info['COMID'] == comid]
        if not length_data.empty:
            return float(length_data['lengthkm'].iloc[0])
    
    # 如果找不到COMID，使用默认长度
    logging.warning(f"未找到COMID {comid}的长度，使用默认值")
    return 1.0  # 默认长度1km
    

def calculate_river_width(Q: pd.Series) -> pd.Series:
    """
    根据流量计算河道宽度
    
    使用公式 W = aQ^b 计算河道宽度，其中：
    - W是河道宽度(m)
    - Q是流量(m³/s)
    - a和b是系数，这里使用的值为 lnW = 2.10 + 0.45lnQ
    
    参数:
        Q: 流量序列(m³/s)
        
    返回:
        河道宽度序列(m)
    """
    # 避免log(0)错误
    Q_adj = Q.replace(0, np.nan)
    
    # 使用公式 lnW = 2.10 + 0.45lnQ
    lnW = 2.10 + 0.45 * np.log(Q_adj)
    W = np.exp(lnW)
    
    # 填充可能的NaN值（对应流量为0的情况）
    return W.fillna(0.0)


def calculate_river_surface_area(length_km, width_m):
    """
    计算河段表面积
    
    参数:
        length_km: 河段长度(km)
        width_m: 河道宽度(m)
        
    返回:
        河段表面积(m²)
    """
    # 将长度从km转换为m
    length_m = length_km * 1000.0
    
    # 计算面积(m²)
    return length_m * width_m


def estimate_river_depth(Q: pd.Series, a: float = 0.27, b: float = 0.39) -> pd.Series:
    """
    根据流量估计河道深度
    
    使用公式 D = aQ^b 估计河道深度，其中：
    - D是河道深度(m)
    - Q是流量(m³/s)
    - a和b是系数，默认值基于常用的水力几何关系
    
    参数:
        Q: 流量序列(m³/s)
        a: 深度公式系数a，默认0.27
        b: 深度公式系数b，默认0.39
        
    返回:
        河道深度序列(m)
    """
    # 避免负值或零值导致的问题
    Q_adj = Q.replace(0, np.nan).abs()
    
    # 计算深度
    depth = a * np.power(Q_adj, b)
    
    # 填充可能的NaN值
    return depth.fillna(0.1)  # 使用最小深度0.1m作为缺省值