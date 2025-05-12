"""
debug_utils.py - 调试信息收集和保存工具

本模块提供了收集、管理和保存汇流计算中间结果的功能，
以便于后续分析和定位问题。

主要功能:
1. 收集各种中间计算结果
2. 组织和管理调试信息
3. 提供多种格式的导出功能
4. 支持查询和分析调试信息
"""

import os
import json
import pandas as pd
import logging
from datetime import datetime


class DebugInfoCollector:
    """
    调试信息收集器类
    
    该类负责收集、组织和保存汇流计算过程中的中间结果和调试信息。
    它提供了各种方法来添加不同类型的信息，并支持将这些信息保存为
    JSON、CSV等多种格式。
    """
    
    def __init__(self, iteration, target_col, output_dir="debug_output"):
        """
        初始化调试信息收集器
        
        参数:
            iteration: 迭代次数
            target_col: 目标列（水质参数）
            output_dir: 调试信息输出目录
        """
        self.iteration = iteration
        self.target_col = target_col
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化各类调试信息容器
        self.parameters = {}
        self.missing_comids = set()
        self.segment_info = {}
        self.routing_steps = []
        self.node_calculations = {}
        self.warnings = []
        self.errors = []
        self.summary = {}
    
    def add_parameters(self, params):
        """添加计算参数信息"""
        self.parameters.update(params)
    
    def add_missing_comids(self, missing_comids):
        """添加缺失数据的COMID集合"""
        self.missing_comids.update(missing_comids)
    
    def add_segment_info(self, comid, segment_type, data):
        """
        添加河段信息
        
        参数:
            comid: 河段ID
            segment_type: 河段类型（如'headwater'）
            data: 河段数据
        """
        self.segment_info[str(comid)] = {
            'type': segment_type,
            'data': data
        }
    
    def add_routing_step(self, upstream_comid, downstream_comid, step_data):
        """
        添加汇流计算步骤信息
        
        参数:
            upstream_comid: 上游河段ID
            downstream_comid: 下游河段ID
            step_data: 计算步骤数据
        """
        self.routing_steps.append({
            'upstream_comid': str(upstream_comid),
            'downstream_comid': str(downstream_comid),
            'data': step_data
        })
    
    def add_node_calculation(self, comid, node_data):
        """
        添加节点（河段）计算信息
        
        参数:
            comid: 河段ID
            node_data: 节点计算数据
        """
        self.node_calculations[str(comid)] = node_data
    
    def add_warning(self, warning_type, message, details=None):
        """
        添加警告信息
        
        参数:
            warning_type: 警告类型
            message: 警告消息
            details: 详细信息（可选）
        """
        self.warnings.append({
            'type': warning_type,
            'message': message,
            'details': details,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def add_error(self, error_type, message, details=None):
        """
        添加错误信息
        
        参数:
            error_type: 错误类型
            message: 错误消息
            details: 详细信息（可选）
        """
        self.errors.append({
            'type': error_type,
            'message': message,
            'details': details,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def add_summary(self, summary_data):
        """添加汇总信息"""
        self.summary.update(summary_data)
    
    def save(self):
        """保存所有调试信息"""
        # 创建基础文件名
        base_filename = f"debug_iteration_{self.iteration}_{self.target_col}_{self.timestamp}"
        
        # 保存参数和摘要信息
        self._save_json({
            'parameters': self.parameters,
            'summary': self.summary,
            'missing_comids': list(self.missing_comids),
            'warnings_count': len(self.warnings),
            'errors_count': len(self.errors)
        }, f"{base_filename}_summary.json")
        
        # 保存警告和错误信息
        if self.warnings or self.errors:
            self._save_json({
                'warnings': self.warnings,
                'errors': self.errors
            }, f"{base_filename}_issues.json")
        
        # 保存河段信息
        if self.segment_info:
            # 保存所有河段的摘要信息
            segments_summary = {
                comid: {
                    'type': info['type'],
                    'first_date': info['data']['dates'][0] if 'dates' in info['data'] and info['data']['dates'] else None,
                    'last_date': info['data']['dates'][-1] if 'dates' in info['data'] and info['data']['dates'] else None,
                    'data_points': len(info['data']['dates']) if 'dates' in info['data'] else 0
                }
                for comid, info in self.segment_info.items()
            }
            self._save_json(segments_summary, f"{base_filename}_segments_summary.json")
            
            # 为头部河段保存详细信息
            headwater_segments = {
                comid: info['data']
                for comid, info in self.segment_info.items()
                if info['type'] == 'headwater'
            }
            if headwater_segments:
                self._save_json(headwater_segments, f"{base_filename}_headwater_segments.json")
        
        # 保存汇流步骤信息
        if self.routing_steps:
            # 保存一个摘要
            routing_summary = {
                f"{step['upstream_comid']}->{step['downstream_comid']}": {
                    'dates_count': len(step['data']['dates']) if 'dates' in step['data'] else 0,
                    'first_date': step['data']['dates'][0] if 'dates' in step['data'] and step['data']['dates'] else None,
                    'last_date': step['data']['dates'][-1] if 'dates' in step['data'] and step['data']['dates'] else None
                }
                for step in self.routing_steps
            }
            self._save_json(routing_summary, f"{base_filename}_routing_summary.json")
            
            # 保存前100个步骤的详细信息
            if len(self.routing_steps) > 100:
                self._save_json(self.routing_steps[:100], f"{base_filename}_routing_steps_sample.json")
            else:
                self._save_json(self.routing_steps, f"{base_filename}_routing_steps.json")
            
            # 尝试将步骤数据转换为DataFrame以便保存为CSV
            try:
                # 创建一个扁平化版本的步骤数据
                flat_steps = []
                for step in self.routing_steps:
                    # 跳过没有日期数据的步骤
                    if 'dates' not in step['data'] or not step['data']['dates']:
                        continue
                    
                    upstream = step['upstream_comid']
                    downstream = step['downstream_comid']
                    
                    for i, date in enumerate(step['data']['dates']):
                        flat_step = {
                            'date': date,
                            'upstream_comid': upstream,
                            'downstream_comid': downstream
                        }
                        
                        # 添加数据点
                        for key, values in step['data'].items():
                            if key != 'dates' and values is not None and i < len(values):
                                flat_step[key] = values[i]
                        
                        flat_steps.append(flat_step)
                
                if flat_steps:
                    # 转换为DataFrame并保存为CSV
                    steps_df = pd.DataFrame(flat_steps)
                    steps_df.to_csv(os.path.join(self.output_dir, f"{base_filename}_routing_steps.csv"), index=False)
            except Exception as e:
                logging.error(f"保存汇流步骤为CSV时出错: {str(e)}")
        
        # 保存节点计算信息
        if self.node_calculations:
            # 保存节点计算摘要
            node_summary = {
                comid: {
                    'dates_count': len(data['dates']) if 'dates' in data and data['dates'] else 0,
                    'first_date': data['dates'][0] if 'dates' in data and data['dates'] else None,
                    'last_date': data['dates'][-1] if 'dates' in data and data['dates'] else None
                }
                for comid, data in self.node_calculations.items()
            }
            self._save_json(node_summary, f"{base_filename}_nodes_summary.json")
            
            # 保存样本节点数据
            sample_nodes = dict(list(self.node_calculations.items())[:20])
            self._save_json(sample_nodes, f"{base_filename}_nodes_sample.json")
    
    def _save_json(self, data, filename):
        """保存数据为JSON文件"""
        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logging.error(f"保存JSON文件 {filepath} 失败: {str(e)}")
    
    def get_query_interface(self):
        """
        获取查询接口，用于后续分析
        
        返回:
            DebugQueryInterface实例
        """
        return DebugQueryInterface(self)


class DebugQueryInterface:
    """
    调试信息查询接口类
    
    该类提供对收集的调试信息的查询和分析功能，
    支持按河段、日期、计算步骤等条件进行查询。
    """
    
    def __init__(self, debug_collector):
        """
        初始化查询接口
        
        参数:
            debug_collector: DebugInfoCollector实例
        """
        self.collector = debug_collector
    
    def get_segment_info(self, comid):
        """
        获取指定河段的信息
        
        参数:
            comid: 河段ID
        
        返回:
            河段信息字典
        """
        return self.collector.segment_info.get(str(comid))
    
    def get_routing_steps(self, upstream_comid=None, downstream_comid=None):
        """
        获取汇流计算步骤
        
        参数:
            upstream_comid: 上游河段ID（可选）
            downstream_comid: 下游河段ID（可选）
        
        返回:
            匹配条件的汇流步骤列表
        """
        results = []
        for step in self.collector.routing_steps:
            if upstream_comid and str(step['upstream_comid']) != str(upstream_comid):
                continue
            if downstream_comid and str(step['downstream_comid']) != str(downstream_comid):
                continue
            results.append(step)
        return results
    
    def get_node_calculation(self, comid):
        """
        获取节点计算信息
        
        参数:
            comid: 河段ID
        
        返回:
            节点计算数据
        """
        return self.collector.node_calculations.get(str(comid))
    
    def get_issues(self, issue_type=None):
        """
        获取问题信息（警告和错误）
        
        参数:
            issue_type: 问题类型（可选）
        
        返回:
            匹配条件的问题列表
        """
        issues = self.collector.warnings + self.collector.errors
        if issue_type:
            return [issue for issue in issues if issue['type'] == issue_type]
        return issues
    
    def search_comid_interactions(self, comid):
        """
        搜索与指定河段相关的所有交互
        
        参数:
            comid: 河段ID
        
        返回:
            与指定河段相关的所有信息
        """
        comid_str = str(comid)
        results = {
            'segment_info': self.get_segment_info(comid),
            'as_upstream': self.get_routing_steps(upstream_comid=comid),
            'as_downstream': self.get_routing_steps(downstream_comid=comid),
            'node_calculation': self.get_node_calculation(comid)
        }
        return results
    
    def to_dataframe(self, data_type, **filters):
        """
        将指定类型的调试数据转换为DataFrame
        
        参数:
            data_type: 数据类型（'routing_steps', 'node_calculations', 'segment_info'等）
            **filters: 过滤条件
        
        返回:
            pandas.DataFrame
        """
        if data_type == 'routing_steps':
            # 提取符合过滤条件的汇流步骤
            steps = self.get_routing_steps(**filters)
            
            # 扁平化处理
            flat_steps = []
            for step in steps:
                if 'dates' not in step['data'] or not step['data']['dates']:
                    continue
                    
                for i, date in enumerate(step['data']['dates']):
                    flat_step = {
                        'date': date,
                        'upstream_comid': step['upstream_comid'],
                        'downstream_comid': step['downstream_comid']
                    }
                    
                    for key, values in step['data'].items():
                        if key != 'dates' and values is not None and i < len(values):
                            flat_step[key] = values[i]
                    
                    flat_steps.append(flat_step)
            
            return pd.DataFrame(flat_steps)
        
        elif data_type == 'node_calculations':
            # 提取符合过滤条件的节点计算数据
            if 'comid' in filters:
                node_data = self.get_node_calculation(filters['comid'])
                if not node_data or 'dates' not in node_data:
                    return pd.DataFrame()
                
                # 扁平化处理
                flat_nodes = []
                for i, date in enumerate(node_data['dates']):
                    flat_node = {'date': date, 'comid': filters['comid']}
                    
                    for key, values in node_data.items():
                        if key != 'dates' and values is not None and i < len(values):
                            flat_node[key] = values[i]
                    
                    flat_nodes.append(flat_node)
                
                return pd.DataFrame(flat_nodes)
            else:
                return pd.DataFrame()
        
        elif data_type == 'issues':
            return pd.DataFrame(self.get_issues())
        
        else:
            return pd.DataFrame()