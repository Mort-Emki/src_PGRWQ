"""
calculate_retention_coefficients.py - 计算并保存保留系数R

该程序计算每个河段到其下游河段的保留系数，并保存为时间序列格式。
保留系数表示物质从上游河段传输到下游河段时的保留比例。
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import os

# 导入已有的计算函数
from PGRWQI.flow_routing_modules.core.geometry import get_river_length, calculate_river_width
from PGRWQI.flow_routing_modules.physics.environment_param import compute_retainment_factor
from PGRWQI.data_processing import load_daily_data, load_river_attributes


def calculate_retention_coefficients(
    daily_data_path: str,
    attr_data_path: str,
    output_path: str = "retention_coefficients.csv",
    parameters: List[str] = ["TN", "TP"],
    v_f_TN: float = 35.0,
    v_f_TP: float = 44.5
):
    """
    计算并保存保留系数R
    
    参数:
        daily_data_path: 日尺度数据文件路径
        attr_data_path: 河段属性数据文件路径  
        output_path: 输出文件路径
        parameters: 要计算的水质参数列表
        v_f_TN: TN的吸收速率参数 (m/yr)
        v_f_TP: TP的吸收速率参数 (m/yr)
    """
    
    # 1. 加载数据
    print("加载数据...")
    df = load_daily_data(daily_data_path)
    attr_df = load_river_attributes(attr_data_path)
    
    # 确保日期列为datetime格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. 构建河网拓扑和属性字典
    print("构建河网拓扑...")
    
    # 创建NextDownID映射
    topo_dict = attr_df.set_index('COMID')['NextDownID'].to_dict()
    
    # 创建属性字典（用于获取河段长度）
    attr_dict = {}
    for _, row in attr_df.iterrows():
        attr_dict[str(row['COMID'])] = {
            'lengthkm': row.get('lengthkm', 1.0)
        }
    
    # 3. 计算保留系数
    print("计算保留系数...")
    
    # 存储结果
    results = []
    
    # 按河段分组处理
    for comid, group in df.groupby('COMID'):
        # 获取下游河段ID
        next_down_id = topo_dict.get(comid, 0)
        
        # 跳过没有下游的河段（终端河段）
        if next_down_id == 0:
            continue
            
        # 检查下游河段是否有数据
        down_data = df[df['COMID'] == next_down_id]
        if down_data.empty:
            continue
            
        # 按日期排序
        group = group.sort_values('date')
        down_data = down_data.sort_values('date')
        
        # 找到共同的日期
        common_dates = set(group['date']) & set(down_data['date'])
        if not common_dates:
            continue
            
        common_dates = sorted(common_dates)
        
        # 获取共同日期的数据
        up_subset = group[group['date'].isin(common_dates)].set_index('date')
        down_subset = down_data[down_data['date'].isin(common_dates)].set_index('date')
        
        # 计算河道宽度
        up_subset['width'] = calculate_river_width(up_subset['Qout'])
        down_subset['width'] = calculate_river_width(down_subset['Qout'])
        
        # 获取河段长度
        length_up = get_river_length(comid, attr_dict)
        length_down = get_river_length(next_down_id, attr_dict)
        
        # 获取温度数据（如果有）
        temperature = up_subset.get('temperature_2m_mean', None)
        
        # 为每个参数计算保留系数
        for param in parameters:
            # 选择相应的吸收速率
            v_f = v_f_TN if param == "TN" else v_f_TP
            
            # 获取氮浓度数据（仅对TN有效）
            N_concentration = None
            if param == "TN" and param in up_subset.columns:
                N_concentration = up_subset[param]
            
            # 计算保留系数
            R_series = compute_retainment_factor(
                v_f=v_f,
                Q_up=up_subset['Qout'],
                Q_down=down_subset['Qout'],
                W_up=up_subset['width'],
                W_down=down_subset['width'],
                length_up=length_up,
                length_down=length_down,
                temperature=temperature,
                N_concentration=N_concentration,
                parameter=param
            )
            
            # 保存结果
            for date in common_dates:
                if date in R_series.index:
                    results.append({
                        'COMID': comid,
                        'NextDownID': next_down_id,
                        'date': date,
                        f'R_{param}': R_series.loc[date],
                        'length_up_km': length_up,
                        'length_down_km': length_down
                    })
    
    # 4. 保存结果
    print(f"保存结果到 {output_path}...")
    
    if not results:
        print("警告: 没有计算出任何保留系数")
        return
    
    # 转换为DataFrame
    result_df = pd.DataFrame(results)
    
    # 重新组织数据格式，使其类似于feature_ts_daily.csv
    # 按COMID和date分组，将不同参数的R值放在同一行
    pivot_data = []
    
    for (comid, next_down, date), group in result_df.groupby(['COMID', 'NextDownID', 'date']):
        row = {
            'COMID': comid,
            'NextDownID': next_down, 
            'date': date,
            'length_up_km': group['length_up_km'].iloc[0],
            'length_down_km': group['length_down_km'].iloc[0]
        }
        
        # 添加各参数的R值
        for param in parameters:
            r_col = f'R_{param}'
            if r_col in group.columns:
                param_data = group[group.columns.str.contains(r_col)]
                if not param_data.empty:
                    row[r_col] = param_data[r_col].iloc[0]
        
        pivot_data.append(row)
    
    # 创建最终的DataFrame
    final_df = pd.DataFrame(pivot_data)
    
    # 按COMID和date排序
    final_df = final_df.sort_values(['COMID', 'date'])
    
    # 保存到CSV文件
    final_df.to_csv(output_path, index=False)
    
    # 打印统计信息
    print(f"完成! 计算了 {len(final_df)} 条保留系数记录")
    print(f"涉及 {final_df['COMID'].nunique()} 个河段")
    print(f"时间范围: {final_df['date'].min()} 到 {final_df['date'].max()}")
    
    # 打印保留系数的统计信息
    for param in parameters:
        r_col = f'R_{param}'
        if r_col in final_df.columns:
            r_values = final_df[r_col].dropna()
            if len(r_values) > 0:
                print(f"{param} 保留系数统计:")
                print(f"  平均值: {r_values.mean():.4f}")
                print(f"  标准差: {r_values.std():.4f}")
                print(f"  范围: {r_values.min():.4f} - {r_values.max():.4f}")


def analyze_retention_patterns(retention_file: str):
    """
    分析保留系数的时空变化模式
    
    参数:
        retention_file: 保留系数文件路径
    """
    print("分析保留系数模式...")
    
    df = pd.read_csv(retention_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # 按月份分析
    df['month'] = df['date'].dt.month
    monthly_stats = df.groupby('month')[['R_TN', 'R_TP']].agg(['mean', 'std']).round(4)
    print("\n月度保留系数统计:")
    print(monthly_stats)
    
    # 按河段长度分析
    df['length_category'] = pd.cut(df['length_up_km'], 
                                  bins=[0, 1, 5, 10, float('inf')], 
                                  labels=['<1km', '1-5km', '5-10km', '>10km'])
    length_stats = df.groupby('length_category')[['R_TN', 'R_TP']].agg(['mean', 'std']).round(4)
    print("\n按河段长度分类的保留系数统计:")
    print(length_stats)


if __name__ == "__main__":
    # 配置参数
    daily_data_path = "D:\\PGRWQ\\data\\feature_daily_ts.csv"  # 修改为您的数据路径
    attr_data_path = "D:\\PGRWQ\\data\\river_attributes_new.csv"   # 修改为您的属性数据路径
    output_path = "retention_coefficients_timeseries.csv"
    
    # 计算保留系数
    calculate_retention_coefficients(
        daily_data_path=daily_data_path,
        attr_data_path=attr_data_path,
        output_path=output_path,
        parameters=["TN", "TP"],
        v_f_TN=35.0,
        v_f_TP=44.5
    )
    
    # 分析结果
    if os.path.exists(output_path):
        analyze_retention_patterns(output_path)
    
    print("程序执行完成!")