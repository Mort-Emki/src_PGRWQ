#!/usr/bin/env python
"""
分析E值文件，检查负值在训练河段和非训练河段上的分布情况
以及检测无效值（NaN、Inf）。

"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 设置matplotlib支持中文
def set_chinese_font():
    """设置matplotlib以支持中文显示"""
    # 尝试不同的中文字体
    font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STSong', 'Arial Unicode MS']
    
    for font in font_list:
        plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
    
    # 正确显示负号
    plt.rcParams['axes.unicode_minus'] = False
    
    # 增大字体大小以提高可读性
    plt.rcParams['font.size'] = 12

# 应用中文字体设置
set_chinese_font()

def validate_inputs(e_file_path, head_comids_path):
    """验证输入文件的有效性"""
    errors = []
    
    # 检查文件是否存在
    if not os.path.exists(e_file_path):
        errors.append(f"E值文件不存在: {e_file_path}")
    
    if not os.path.exists(head_comids_path):
        errors.append(f"头部河段文件不存在: {head_comids_path}")
    
    # 检查文件格式
    if e_file_path.endswith('.csv'):
        try:
            df = pd.read_csv(e_file_path)
            required_columns = ['COMID', 'Date', 'E_value']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                errors.append(f"E值文件缺少必要的列: {missing_columns}")
        except Exception as e:
            errors.append(f"读取E值文件失败: {str(e)}")
    else:
        errors.append(f"E值文件必须是CSV格式: {e_file_path}")
    
    if head_comids_path.endswith('.npy'):
        try:
            np.load(head_comids_path)
        except Exception as e:
            errors.append(f"读取头部河段文件失败: {str(e)}")
    else:
        errors.append(f"头部河段文件必须是.npy格式: {head_comids_path}")
    
    return errors

def analyze_e_values(e_file_path, head_comids_path):
    """
    分析E值文件，检查负值在训练河段和非训练河段上的分布情况。
    
    Args:
        e_file_path: E值CSV文件路径
        head_comids_path: 头部河段COMID的numpy文件路径
    
    Returns:
        分析结果的字典
    """
    print(f"正在分析E值文件: {e_file_path}")
    print(f"头部河段文件: {head_comids_path}")
    
    # 加载数据
    df_e = pd.read_csv(e_file_path)
    head_comids = np.load(head_comids_path)
    
    print(f"E值文件包含 {len(df_e)} 条记录, {df_e['COMID'].nunique()} 个不同的COMID")
    print(f"头部河段文件包含 {len(head_comids)} 个COMID")
    
    # 转换头部COMID为集合，便于快速查找
    head_comids_set = set(map(str, head_comids))  # 确保所有COMID都是字符串类型
    
    # 将E值文件中的COMID也转为字符串以确保一致性
    df_e['COMID'] = df_e['COMID'].astype(str)
    
    # 区分训练组和非训练组
    df_e['is_train_comid'] = df_e['COMID'].isin(head_comids_set)
    
    # 分组统计
    train_group = df_e[df_e['is_train_comid']]
    non_train_group = df_e[~df_e['is_train_comid']]
    
    # 初始化结果字典
    results = {
        'total_records': len(df_e),
        'total_comids': df_e['COMID'].nunique(),
        'train_comids': {
            'count': train_group['COMID'].nunique(),
            'records': len(train_group),
            'neg_records': (train_group['E_value'] < 0).sum(),
            'nan_records': train_group['E_value'].isna().sum(),
            'inf_records': (~train_group['E_value'].isna() & (train_group['E_value'].abs() == float('inf'))).sum(),
        },
        'non_train_comids': {
            'count': non_train_group['COMID'].nunique(),
            'records': len(non_train_group),
            'neg_records': (non_train_group['E_value'] < 0).sum(),
            'nan_records': non_train_group['E_value'].isna().sum(),
            'inf_records': (~non_train_group['E_value'].isna() & (non_train_group['E_value'].abs() == float('inf'))).sum(),
        }
    }
    
    # 计算百分比
    if results['train_comids']['records'] > 0:
        results['train_comids']['neg_pct'] = results['train_comids']['neg_records'] / results['train_comids']['records'] * 100
        results['train_comids']['nan_pct'] = results['train_comids']['nan_records'] / results['train_comids']['records'] * 100
        results['train_comids']['inf_pct'] = results['train_comids']['inf_records'] / results['train_comids']['records'] * 100
    else:
        results['train_comids']['neg_pct'] = 0
        results['train_comids']['nan_pct'] = 0
        results['train_comids']['inf_pct'] = 0
    
    if results['non_train_comids']['records'] > 0:
        results['non_train_comids']['neg_pct'] = results['non_train_comids']['neg_records'] / results['non_train_comids']['records'] * 100
        results['non_train_comids']['nan_pct'] = results['non_train_comids']['nan_records'] / results['non_train_comids']['records'] * 100
        results['non_train_comids']['inf_pct'] = results['non_train_comids']['inf_records'] / results['non_train_comids']['records'] * 100
    else:
        results['non_train_comids']['neg_pct'] = 0
        results['non_train_comids']['nan_pct'] = 0
        results['non_train_comids']['inf_pct'] = 0
    
    # 计算统计值
    if not train_group.empty:
        train_e_values = train_group['E_value'].dropna()
        train_valid_e = train_e_values[~np.isinf(train_e_values)]
        if not train_valid_e.empty:
            results['train_comids']['min'] = train_valid_e.min()
            results['train_comids']['max'] = train_valid_e.max()
            results['train_comids']['mean'] = train_valid_e.mean()
            results['train_comids']['median'] = train_valid_e.median()
            results['train_comids']['std'] = train_valid_e.std()
    
    if not non_train_group.empty:
        non_train_e_values = non_train_group['E_value'].dropna()
        non_train_valid_e = non_train_e_values[~np.isinf(non_train_e_values)]
        if not non_train_valid_e.empty:
            results['non_train_comids']['min'] = non_train_valid_e.min()
            results['non_train_comids']['max'] = non_train_valid_e.max()
            results['non_train_comids']['mean'] = non_train_valid_e.mean()
            results['non_train_comids']['median'] = non_train_valid_e.median()
            results['non_train_comids']['std'] = non_train_valid_e.std()
    
    # 计算各个河段的负值比例
    comid_neg_pct = {}
    for comid, group in df_e.groupby('COMID'):
        neg_count = (group['E_value'] < 0).sum()
        neg_pct = neg_count / len(group) * 100
        comid_neg_pct[comid] = {
            'total': len(group),
            'neg_count': neg_count,
            'neg_pct': neg_pct,
            'is_train': comid in head_comids_set
        }
    
    # 找出负值比例最高的河段（前10个）
    top_neg_comids = sorted(comid_neg_pct.items(), key=lambda x: x[1]['neg_pct'], reverse=True)[:10]
    results['top_neg_comids'] = top_neg_comids
    
    # 找出每个分组中负值比例最高的10个河段
    train_comids_neg_pct = {comid: info for comid, info in comid_neg_pct.items() if info['is_train']}
    non_train_comids_neg_pct = {comid: info for comid, info in comid_neg_pct.items() if not info['is_train']}
    
    top_train_neg_comids = sorted(train_comids_neg_pct.items(), key=lambda x: x[1]['neg_pct'], reverse=True)[:10]
    top_non_train_neg_comids = sorted(non_train_comids_neg_pct.items(), key=lambda x: x[1]['neg_pct'], reverse=True)[:10]
    
    results['top_train_neg_comids'] = top_train_neg_comids
    results['top_non_train_neg_comids'] = top_non_train_neg_comids
    
    # 统计极端负E值的分布
    extreme_thresholds = [-100, -50, -20, -10, -5, -1]
    train_extreme_counts = {}
    non_train_extreme_counts = {}
    
    for threshold in extreme_thresholds:
        train_extreme_counts[threshold] = (train_group['E_value'] < threshold).sum()
        non_train_extreme_counts[threshold] = (non_train_group['E_value'] < threshold).sum()
    
    results['train_extreme_counts'] = train_extreme_counts
    results['non_train_extreme_counts'] = non_train_extreme_counts
    
    return results

def print_results(results):
    """打印分析结果"""
    if results is None:
        print("无分析结果")
        return
    
    print("\n===== E值分析结果 =====")
    print(f"总记录数: {results['total_records']}")
    print(f"总河段数: {results['total_comids']}")
    
    print("\n训练河段 (头部河段):")
    print(f"  河段数: {results['train_comids']['count']}")
    print(f"  记录数: {results['train_comids']['records']}")
    print(f"  负值记录: {results['train_comids']['neg_records']} ({results['train_comids']['neg_pct']:.2f}%)")
    print(f"  NaN记录: {results['train_comids']['nan_records']} ({results['train_comids']['nan_pct']:.2f}%)")
    print(f"  Inf记录: {results['train_comids']['inf_records']} ({results['train_comids']['inf_pct']:.2f}%)")
    
    if 'min' in results['train_comids']:
        print(f"  值域范围: [{results['train_comids']['min']:.4f}, {results['train_comids']['max']:.4f}]")
        print(f"  均值: {results['train_comids']['mean']:.4f}, 中位数: {results['train_comids']['median']:.4f}")
        print(f"  标准差: {results['train_comids']['std']:.4f}")
    
    print("\n非训练河段:")
    print(f"  河段数: {results['non_train_comids']['count']}")
    print(f"  记录数: {results['non_train_comids']['records']}")
    print(f"  负值记录: {results['non_train_comids']['neg_records']} ({results['non_train_comids']['neg_pct']:.2f}%)")
    print(f"  NaN记录: {results['non_train_comids']['nan_records']} ({results['non_train_comids']['nan_pct']:.2f}%)")
    print(f"  Inf记录: {results['non_train_comids']['inf_records']} ({results['non_train_comids']['inf_pct']:.2f}%)")
    
    if 'min' in results['non_train_comids']:
        print(f"  值域范围: [{results['non_train_comids']['min']:.4f}, {results['non_train_comids']['max']:.4f}]")
        print(f"  均值: {results['non_train_comids']['mean']:.4f}, 中位数: {results['non_train_comids']['median']:.4f}")
        print(f"  标准差: {results['non_train_comids']['std']:.4f}")
    
    print("\n极端负E值分布:")
    thresholds = sorted(results['train_extreme_counts'].keys())
    print("  |  阈值  | 训练河段 | 非训练河段 |")
    print("  |--------|----------|------------|")
    for threshold in thresholds:
        train_count = results['train_extreme_counts'][threshold]
        non_train_count = results['non_train_extreme_counts'][threshold]
        print(f"  | < {threshold:4} | {train_count:8} | {non_train_count:10} |")
    
    print("\n负值比例最高的训练河段 (前10个):")
    for i, (comid, info) in enumerate(results['top_train_neg_comids']):
        print(f"  {i+1}. COMID {comid}: {info['neg_count']}/{info['total']} ({info['neg_pct']:.2f}%)")
    
    print("\n负值比例最高的非训练河段 (前10个):")
    for i, (comid, info) in enumerate(results['top_non_train_neg_comids']):
        print(f"  {i+1}. COMID {comid}: {info['neg_count']}/{info['total']} ({info['neg_pct']:.2f}%)")
    
    # 结论分析
    train_neg_pct = results['train_comids']['neg_pct']
    non_train_neg_pct = results['non_train_comids']['neg_pct']
    
    print("\n===== 结论分析 =====")
    if train_neg_pct > 20 and non_train_neg_pct > 20:
        print("结论: 训练河段和非训练河段均存在大量负E值，可能是模型设计问题或数据集问题。")
    elif train_neg_pct > 20:
        print("结论: 训练河段存在较多负E值，但非训练河段较少，说明模型训练时就学习了负值模式。")
    elif non_train_neg_pct > 20:
        print("结论: 非训练河段存在较多负E值，但训练河段较少，说明模型在泛化时表现不稳定。")
    else:
        print("结论: 训练河段和非训练河段的负E值比例都较低，负值问题不严重。")
    
    # 进一步分析
    if abs(train_neg_pct - non_train_neg_pct) > 10:
        print(f"训练河段与非训练河段的负值比例差异较大 ({abs(train_neg_pct - non_train_neg_pct):.2f}%)，")
        if train_neg_pct > non_train_neg_pct:
            print("训练数据可能存在特殊性，导致模型在训练河段上预测出更多负值。")
        else:
            print("模型在非训练河段上泛化性较差，导致预测出更多负值。")
    
    # 检测无效值
    has_invalid = (results['train_comids']['nan_records'] > 0 or results['train_comids']['inf_records'] > 0 or
                 results['non_train_comids']['nan_records'] > 0 or results['non_train_comids']['inf_records'] > 0)
    
    if has_invalid:
        print("\n⚠️ 检测到无效值 (NaN 或 Inf)，建议检查数据处理流程。")

# def create_visualizations(e_file_path, head_comids_path, output_dir='.'):
#     """创建可视化图表"""
#     # 加载数据
#     df_e = pd.read_csv(e_file_path)
#     head_comids = np.load(head_comids_path)
    
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 转换头部COMID为集合，便于快速查找
#     head_comids_set = set(map(str, head_comids))
    
#     # 将E值文件中的COMID也转为字符串以确保一致性
#     df_e['COMID'] = df_e['COMID'].astype(str)
    
#     # 区分训练组和非训练组
#     df_e['is_train_comid'] = df_e['COMID'].isin(head_comids_set)
    
#     # 分组
#     train_group = df_e[df_e['is_train_comid']]
#     non_train_group = df_e[~df_e['is_train_comid']]
    
#     # 绘制E值分布直方图
#     plt.figure(figsize=(12, 8))
    
#     plt.subplot(2, 1, 1)
#     plt.hist(train_group['E_value'].dropna(), bins=50, alpha=0.7, label='训练河段')
#     plt.axvline(x=0, color='r', linestyle='--')
#     plt.title('训练河段E值分布')
#     plt.xlabel('E值')
#     plt.ylabel('频率')
#     plt.legend()
    
#     plt.subplot(2, 1, 2)
#     plt.hist(non_train_group['E_value'].dropna(), bins=50, alpha=0.7, label='非训练河段')
#     plt.axvline(x=0, color='r', linestyle='--')
#     plt.title('非训练河段E值分布')
#     plt.xlabel('E值')
#     plt.ylabel('频率')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'e_value_distribution.png'))
#     print(f"已保存E值分布图到 {os.path.join(output_dir, 'e_value_distribution.png')}")
    
#     # 绘制负值比例对比图
#     plt.figure(figsize=(10, 6))
#     neg_pct_train = (train_group['E_value'] < 0).mean() * 100
#     neg_pct_non_train = (non_train_group['E_value'] < 0).mean() * 100
    
#     plt.bar(['训练河段', '非训练河段'], [neg_pct_train, neg_pct_non_train])
#     plt.title('负E值比例对比')
#     plt.ylabel('负值比例 (%)')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
    
#     for i, v in enumerate([neg_pct_train, neg_pct_non_train]):
#         plt.text(i, v + 1, f'{v:.2f}%', ha='center')
    
#     plt.savefig(os.path.join(output_dir, 'negative_e_value_comparison.png'))
#     print(f"已保存负值比例对比图到 {os.path.join(output_dir, 'negative_e_value_comparison.png')}")
    
#     # 分析每个河段的负值比例
#     comid_neg_pct = {}
#     for comid, group in df_e.groupby('COMID'):
#         neg_count = (group['E_value'] < 0).sum()
#         neg_pct = neg_count / len(group) * 100
#         comid_neg_pct[comid] = {
#             'neg_pct': neg_pct,
#             'is_train': comid in head_comids_set
#         }
    
#     # 绘制散点图：河段的负值比例vs是否为训练河段
#     train_neg_pcts = [info['neg_pct'] for comid, info in comid_neg_pct.items() if info['is_train']]
#     non_train_neg_pcts = [info['neg_pct'] for comid, info in comid_neg_pct.items() if not info['is_train']]
    
#     plt.figure(figsize=(12, 6))
#     plt.scatter([0] * len(train_neg_pcts), train_neg_pcts, 
#                 alpha=0.5, label=f'训练河段 (n={len(train_neg_pcts)})')
#     plt.scatter([1] * len(non_train_neg_pcts), non_train_neg_pcts, 
#                 alpha=0.5, label=f'非训练河段 (n={len(non_train_neg_pcts)})')
    
#     plt.boxplot([train_neg_pcts, non_train_neg_pcts], positions=[0, 1], widths=0.5)
    
#     plt.xticks([0, 1], ['训练河段', '非训练河段'])
#     plt.ylabel('负E值比例 (%)')
#     plt.title('各河段负E值比例分布')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     plt.savefig(os.path.join(output_dir, 'comid_negative_e_distribution.png'))
#     print(f"已保存河段负值分布图到 {os.path.join(output_dir, 'comid_negative_e_distribution.png')}")
    
#     # 根据是否为训练河段，绘制E值的箱线图
#     plt.figure(figsize=(10, 6))
    
#     # 准备数据
#     train_e_values = train_group['E_value'].dropna()
#     non_train_e_values = non_train_group['E_value'].dropna()
    
#     # 确保值在合理范围内
#     train_e_values = train_e_values[train_e_values.between(-100, 100)]
#     non_train_e_values = non_train_e_values[non_train_e_values.between(-100, 100)]
    
#     plt.boxplot([train_e_values, non_train_e_values], 
#                 labels=['训练河段', '非训练河段'],
#                 showfliers=True)
    
#     plt.axhline(y=0, color='r', linestyle='--')
#     plt.title('E值箱线图对比')
#     plt.ylabel('E值')
#     plt.grid(True, alpha=0.3)
    
#     plt.savefig(os.path.join(output_dir, 'e_value_boxplot.png'))
#     print(f"已保存E值箱线图到 {os.path.join(output_dir, 'e_value_boxplot.png')}")
    
#     # 生成报告文件
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     report_path = os.path.join(output_dir, f'e_value_analysis_report_{timestamp}.txt')
    
#     # 重定向标准输出到文件
#     original_stdout = sys.stdout
#     with open(report_path, 'w') as f:
#         sys.stdout = f
        
#         print("===== E值分析报告 =====")
#         print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         print(f"E值文件: {e_file_path}")
#         print(f"头部河段文件: {head_comids_path}")
#         print("\n")
        
#         # 重新分析并打印结果
#         results = analyze_e_values(e_file_path, head_comids_path)
#         print_results(results)
        
#     # 恢复标准输出
#     sys.stdout = original_stdout
#     print(f"已生成详细分析报告: {report_path}")

# 返回所有图表结果
def get_all_figures():
    """返回当前的所有图表，用于Jupyter中显示"""
    return plt.get_fignums()

# 提供一个可选的英文版本，以防中文字体设置仍无法正常工作
def create_visualizations(e_file_path, head_comids_path, output_dir='.'):
    """创建英文版可视化图表 (如果中文显示有问题可使用这个函数)"""
    # 加载数据
    df_e = pd.read_csv(e_file_path)
    head_comids = np.load(head_comids_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换头部COMID为集合，便于快速查找
    head_comids_set = set(map(str, head_comids))
    
    # 将E值文件中的COMID也转为字符串以确保一致性
    df_e['COMID'] = df_e['COMID'].astype(str)
    
    # 区分训练组和非训练组
    df_e['is_train_comid'] = df_e['COMID'].isin(head_comids_set)
    
    # 分组
    train_group = df_e[df_e['is_train_comid']]
    non_train_group = df_e[~df_e['is_train_comid']]
    
    # 绘制E值分布直方图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.hist(train_group['E_value'].dropna(), bins=50, alpha=0.7, label='Training Segments')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('E Value Distribution - Training Segments')
    plt.xlabel('E Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.hist(non_train_group['E_value'].dropna(), bins=50, alpha=0.7, label='Non-training Segments')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('E Value Distribution - Non-training Segments')
    plt.xlabel('E Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'e_value_distribution_en.png'))
    print(f"Saved E value distribution chart to {os.path.join(output_dir, 'e_value_distribution_en.png')}")
    
    # 绘制负值比例对比图
    plt.figure(figsize=(10, 6))
    neg_pct_train = (train_group['E_value'] < 0).mean() * 100
    neg_pct_non_train = (non_train_group['E_value'] < 0).mean() * 100
    
    plt.bar(['Training Segments', 'Non-training Segments'], [neg_pct_train, neg_pct_non_train])
    plt.title('Negative E Value Percentage Comparison')
    plt.ylabel('Negative Value Percentage (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate([neg_pct_train, neg_pct_non_train]):
        plt.text(i, v + 1, f'{v:.2f}%', ha='center')
    
    plt.savefig(os.path.join(output_dir, 'negative_e_value_comparison_en.png'))
    print(f"Saved negative value percentage chart to {os.path.join(output_dir, 'negative_e_value_comparison_en.png')}")
    
    # 分析每个河段的负值比例
    comid_neg_pct = {}
    for comid, group in df_e.groupby('COMID'):
        neg_count = (group['E_value'] < 0).sum()
        neg_pct = neg_count / len(group) * 100
        comid_neg_pct[comid] = {
            'neg_pct': neg_pct,
            'is_train': comid in head_comids_set
        }
    
    # 绘制散点图：河段的负值比例vs是否为训练河段
    train_neg_pcts = [info['neg_pct'] for comid, info in comid_neg_pct.items() if info['is_train']]
    non_train_neg_pcts = [info['neg_pct'] for comid, info in comid_neg_pct.items() if not info['is_train']]
    
    plt.figure(figsize=(12, 6))
    plt.scatter([0] * len(train_neg_pcts), train_neg_pcts, 
                alpha=0.5, label=f'Training Segments (n={len(train_neg_pcts)})')
    plt.scatter([1] * len(non_train_neg_pcts), non_train_neg_pcts, 
                alpha=0.5, label=f'Non-training Segments (n={len(non_train_neg_pcts)})')
    
    plt.boxplot([train_neg_pcts, non_train_neg_pcts], positions=[0, 1], widths=0.5)
    
    plt.xticks([0, 1], ['Training Segments', 'Non-training Segments'])
    plt.ylabel('Negative E Value Percentage (%)')
    plt.title('Distribution of Negative E Value Percentage by Segment')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'comid_negative_e_distribution_en.png'))
    print(f"Saved segment negative value distribution chart to {os.path.join(output_dir, 'comid_negative_e_distribution_en.png')}")
    
    # 根据是否为训练河段，绘制E值的箱线图
    plt.figure(figsize=(10, 6))
    
    # 准备数据
    train_e_values = train_group['E_value'].dropna()
    non_train_e_values = non_train_group['E_value'].dropna()
    
    # 确保值在合理范围内
    train_e_values = train_e_values[train_e_values.between(-100, 100)]
    non_train_e_values = non_train_e_values[non_train_e_values.between(-100, 100)]
    
    plt.boxplot([train_e_values, non_train_e_values], 
                labels=['Training Segments', 'Non-training Segments'],
                showfliers=True)
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('E Value Boxplot Comparison')
    plt.ylabel('E Value')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'e_value_boxplot_en.png'))
    print(f"Saved E value boxplot to {os.path.join(output_dir, 'e_value_boxplot_en.png')}")

def analyze_comid_e_values(e_file_path, comid, output_dir='.', start_date=None, end_date=None):
    """
    分析指定COMID的E值，计算负值比例并绘制时间序列图
    
    Args:
        e_file_path: E值CSV文件路径
        comid: 要分析的COMID
        output_dir: 输出目录路径
        start_date: 时间序列起始日期，格式为'YYYY-MM-DD'或datetime对象，None表示从最早日期开始
        end_date: 时间序列结束日期，格式为'YYYY-MM-DD'或datetime对象，None表示到最晚日期结束
    
    Returns:
        分析结果的字典
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from datetime import datetime
    
    print(f"正在分析COMID {comid}的E值...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    df_e = pd.read_csv(e_file_path)
    
    # 将COMID转为字符串以确保一致性
    df_e['COMID'] = df_e['COMID'].astype(str)
    comid = str(comid)
    
    # 筛选指定COMID的数据
    comid_data = df_e[df_e['COMID'] == comid]
    
    if comid_data.empty:
        print(f"错误: 在E值文件中未找到COMID {comid}的数据")
        return None
    
    # 计算负值比例
    total_records = len(comid_data)
    neg_records = (comid_data['E_value'] < 0).sum()
    neg_percentage = (neg_records / total_records) * 100 if total_records > 0 else 0
    
    # 统计无效值
    nan_records = comid_data['E_value'].isna().sum()
    inf_records = (~comid_data['E_value'].isna() & (comid_data['E_value'].abs() == float('inf'))).sum()
    
    # 计算基本统计量
    valid_values = comid_data['E_value'].dropna()
    valid_values = valid_values[~np.isinf(valid_values)]
    
    stats = {}
    if not valid_values.empty:
        stats = {
            'min': valid_values.min(),
            'max': valid_values.max(),
            'mean': valid_values.mean(),
            'median': valid_values.median(),
            'std': valid_values.std()
        }
    
    # 准备结果字典
    results = {
        'comid': comid,
        'total_records': total_records,
        'neg_records': neg_records,
        'neg_percentage': neg_percentage,
        'nan_records': nan_records,
        'inf_records': inf_records,
        'stats': stats
    }
    
    # 打印结果
    print(f"\n===== COMID {comid} E值分析结果 =====")
    print(f"总记录数: {total_records}")
    print(f"负值记录: {neg_records} ({neg_percentage:.2f}%)")
    print(f"NaN记录: {nan_records}")
    print(f"Inf记录: {inf_records}")
    
    if stats:
        print(f"值域范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"均值: {stats['mean']:.4f}, 中位数: {stats['median']:.4f}")
        print(f"标准差: {stats['std']:.4f}")
    
    # 绘制E值时间序列图
    if not comid_data.empty and 'Date' in comid_data.columns:
        # 确保日期列为日期类型
        if comid_data['Date'].dtype == 'object':
            comid_data['Date'] = pd.to_datetime(comid_data['Date'])
        
        # 按日期排序
        comid_data = comid_data.sort_values('Date')
        
        # 时间范围过滤
        time_filtered = comid_data.copy()
        date_filter_applied = False
        filter_description = ""
        
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            time_filtered = time_filtered[time_filtered['Date'] >= start_date]
            date_filter_applied = True
            filter_description += f"起始: {start_date.strftime('%Y-%m-%d')} "
        
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            time_filtered = time_filtered[time_filtered['Date'] <= end_date]
            date_filter_applied = True
            filter_description += f"结束: {end_date.strftime('%Y-%m-%d')}"
        
        # 如果过滤后没有数据，提示用户并使用所有数据
        if date_filter_applied and time_filtered.empty:
            print(f"警告: 在指定的时间范围内没有数据 ({filter_description})，将显示所有数据")
            time_filtered = comid_data.copy()
            date_filter_applied = False
        
        # 为图形标题准备时间范围描述
        title_time_range = ""
        if date_filter_applied:
            title_time_range = f" ({filter_description})"
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_filtered['Date'], time_filtered['E_value'], marker='o', linestyle='-')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.title(f'COMID {comid} E值时间序列{title_time_range}')
        plt.xlabel('日期')
        plt.ylabel('E值')
        plt.grid(True, alpha=0.3)
        
        # 设置日期格式
        plt.gcf().autofmt_xdate()
        
        # 添加负值标记
        neg_data = time_filtered[time_filtered['E_value'] < 0]
        if not neg_data.empty:
            neg_pct_filtered = (len(neg_data) / len(time_filtered)) * 100
            plt.scatter(neg_data['Date'], neg_data['E_value'], color='red', s=50, 
                       label=f'负值 ({neg_pct_filtered:.1f}%)', zorder=5)
            plt.legend()
        
        # 添加文本标注显示负值比例（全局和当前筛选）
        annotation_text = f'全部数据负值比例: {neg_percentage:.2f}%'
        if date_filter_applied:
            neg_pct_filtered = (len(neg_data) / len(time_filtered)) * 100 if len(time_filtered) > 0 else 0
            annotation_text += f'\n选定范围负值比例: {neg_pct_filtered:.2f}%'
            
        plt.annotate(annotation_text, 
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    va='top')
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_range_str = ""
        if date_filter_applied:
            start_str = start_date.strftime("%Y%m%d") if start_date else "start"
            end_str = end_date.strftime("%Y%m%d") if end_date else "end"
            date_range_str = f"_{start_str}_to_{end_str}"
            
        file_path = os.path.join(output_dir, f'comid_{comid}_e_values{date_range_str}_{timestamp}.png')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        
        print(f"已保存E值时间序列图: {file_path}")
    
    return results

# 示例使用方法
# 分析所有数据
# analyze_comid_e_values("E_0_TN.csv", "12345678", "output_results")
# 
# 分析特定时间范围的数据
# analyze_comid_e_values("E_0_TN.csv", "12345678", "output_results", 
#                       start_date="2020-01-01", end_date="2020-12-31")
# Jupyter Notebook使用示例:
"""
# 1. 导入模块
%matplotlib inline
import analyze_e_values_fixed as analyzer

# 2. 验证输入文件
errors = analyzer.validate_inputs("E_0_TN.csv", "comid_list_head.npy")
if errors:
    for error in errors:
        print(f"错误: {error}")
else:
    # 3. 分析E值
    results = analyzer.analyze_e_values("E_0_TN.csv", "comid_list_head.npy")
    
    # 4. 打印结果
    analyzer.print_results(results)
    
    # 5. 创建可视化图表（使用中文版）
    analyzer.create_visualizations("E_0_TN.csv", "comid_list_head.npy", "output_results")
    
    # 如果中文显示有问题，可以使用英文版
    # analyzer.create_visualizations_en("E_0_TN.csv", "comid_list_head.npy", "output_results")
"""