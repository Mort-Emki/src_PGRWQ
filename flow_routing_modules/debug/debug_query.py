"""
debug_query.py - 汇流计算调试信息查询模块

该模块提供了加载和查询汇流计算调试信息的功能，
便于分析和定位汇流计算中的问题。
"""

import json
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

class FlowDebugQuery:
    """汇流计算调试信息查询器"""
    
    def __init__(self, debug_file_path: str):
        """
        初始化查询器
        
        参数:
            debug_file_path: 调试信息JSON文件路径
        """
        self.debug_file_path = debug_file_path
        self.debug_data = None
        self.metadata = None
        self.load_debug_info()
    
    def load_debug_info(self):
        """加载调试信息"""
        try:
            with open(self.debug_file_path, 'r', encoding='utf-8') as f:
                self.debug_data = json.load(f)
            
            self.metadata = self.debug_data.get('metadata', {})
            print(f"成功加载调试信息:")
            print(f"  - 目标列: {self.metadata.get('target_col', 'Unknown')}")
            print(f"  - 时间戳: {self.metadata.get('timestamp', 'Unknown')}")
            print(f"  - 节点计算记录: {self.metadata.get('total_node_calculations', 0)}")
            print(f"  - 流向计算记录: {self.metadata.get('total_flow_calculations', 0)}")
            print(f"  - 警告数: {self.metadata.get('total_warnings', 0)}")
            print(f"  - 错误数: {self.metadata.get('total_errors', 0)}")
            
        except Exception as e:
            print(f"加载调试信息失败: {str(e)}")
            raise
    
    def query_node_calculation(self, comid: int, date: str = None) -> List[Dict]:
        """
        查询节点计算信息
        
        参数:
            comid: 河段ID
            date: 日期字符串 (可选)
        
        返回:
            匹配的节点计算记录列表
        """
        results = []
        node_calcs = self.debug_data.get('node_calculations', {})
        
        for key, data in node_calcs.items():
            # 解析键: comid_date
            if '_' in key:
                key_comid = int(key.split('_')[0])
                key_date = '_'.join(key.split('_')[1:])
                
                if key_comid == comid and (date is None or key_date == date):
                    results.append(data)
        
        return results
    
    def query_flow_calculation(self, upstream_comid: int = None, 
                              downstream_comid: int = None, 
                              date: str = None) -> List[Dict]:
        """
        查询流向计算信息
        
        参数:
            upstream_comid: 上游河段ID (可选)
            downstream_comid: 下游河段ID (可选)
            date: 日期字符串 (可选)
        
        返回:
            匹配的流向计算记录列表
        """
        results = []
        flow_calcs = self.debug_data.get('flow_calculations', {})
        
        for key, data in flow_calcs.items():
            # 解析键: upstream_downstream_date
            parts = key.split('_')
            if len(parts) >= 3:
                key_upstream = int(parts[0])
                key_downstream = int(parts[1])
                key_date = '_'.join(parts[2:])
                
                match = True
                if upstream_comid is not None and key_upstream != upstream_comid:
                    match = False
                if downstream_comid is not None and key_downstream != downstream_comid:
                    match = False
                if date is not None and key_date != date:
                    match = False
                
                if match:
                    results.append(data)
        
        return results
    
    def find_large_values(self, threshold: float = 100) -> Dict[str, List]:
        """
        查找异常大的计算结果
        
        参数:
            threshold: 阈值
        
        返回:
            包含异常值的字典
        """
        large_values = {
            'large_y_n': [],
            'large_y_up': [],
            'large_contribution': [],
            'large_E': []
        }
        
        # 检查节点计算中的大值
        node_calcs = self.debug_data.get('node_calculations', {})
        for key, data in node_calcs.items():
            comid = int(key.split('_')[0])
            date = '_'.join(key.split('_')[1:])
            
            if abs(data['y_n_value']) > threshold:
                large_values['large_y_n'].append({
                    'comid': comid,
                    'date': date,
                    'value': data['y_n_value'],
                    'E_value': data['E_value'],
                    'y_up_value': data['y_up_value'],
                    'Qout': data['Qout']
                })
            
            if abs(data['y_up_value']) > threshold:
                large_values['large_y_up'].append({
                    'comid': comid,
                    'date': date,
                    'value': data['y_up_value'],
                    'E_value': data['E_value'],
                    'y_n_value': data['y_n_value'],
                    'Qout': data['Qout']
                })
            
            if abs(data['E_value']) > threshold:
                large_values['large_E'].append({
                    'comid': comid,
                    'date': date,
                    'value': data['E_value'],
                    'y_up_value': data['y_up_value'],
                    'y_n_value': data['y_n_value'],
                    'Qout': data['Qout']
                })
        
        # 检查流向计算中的大值
        flow_calcs = self.debug_data.get('flow_calculations', {})
        for key, data in flow_calcs.items():
            parts = key.split('_')
            upstream = int(parts[0])
            downstream = int(parts[1])
            date = '_'.join(parts[2:])
            
            # 考虑流量规模的贡献阈值
            contribution_threshold = threshold * max(1, data['Q_upstream'] / 10)
            if abs(data['contribution']) > contribution_threshold:
                large_values['large_contribution'].append({
                    'upstream_comid': upstream,
                    'downstream_comid': downstream,
                    'date': date,
                    'contribution': data['contribution'],
                    'y_n_upstream': data['y_n_upstream'],
                    'retention_coefficient': data['retention_coefficient'],
                    'Q_upstream': data['Q_upstream']
                })
        
        return large_values
    
    def analyze_comid_flow_balance(self, comid: int, date: str = None) -> Dict:
        """
        分析特定河段的流量平衡
        
        参数:
            comid: 河段ID
            date: 日期字符串 (可选)
        
        返回:
            流量平衡分析结果
        """
        # 获取该河段的节点计算
        node_data = self.query_node_calculation(comid, date)
        
        # 获取流入该河段的计算
        inflow_data = self.query_flow_calculation(downstream_comid=comid, date=date)
        
        # 获取从该河段流出的计算
        outflow_data = self.query_flow_calculation(upstream_comid=comid, date=date)
        
        analysis = {
            'comid': comid,
            'date': date,
            'node_calculations': len(node_data),
            'inflow_contributions': len(inflow_data),
            'outflow_contributions': len(outflow_data),
            'total_inflow_contribution': sum(d['contribution'] for d in inflow_data),
            'node_data': node_data[0] if node_data else None,
            'inflow_details': inflow_data,
            'outflow_details': outflow_data
        }
        
        return analysis
    
    def get_warnings_and_errors(self) -> Dict:
        """获取所有警告和错误"""
        return {
            'warnings': self.debug_data.get('warnings', []),
            'errors': self.debug_data.get('errors', [])
        }
    
    def export_large_values_csv(self, threshold: float = 100, output_file: str = None):
        """
        导出异常值到CSV文件
        
        参数:
            threshold: 阈值
            output_file: 输出文件名 (可选)
        """
        large_values = self.find_large_values(threshold)
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"large_values_analysis_{threshold}_{timestamp}.csv"
        
        # 合并所有异常值
        all_anomalies = []
        
        for category, items in large_values.items():
            for item in items:
                item['category'] = category
                all_anomalies.append(item)
        
        if all_anomalies:
            df = pd.DataFrame(all_anomalies)
            df.to_csv(output_file, index=False)
            print(f"异常值分析结果已保存到: {output_file}")
        else:
            print(f"未发现超过阈值 {threshold} 的异常值")
    
    def print_summary(self):
        """打印调试信息摘要"""
        print("=== 调试信息摘要 ===")
        print(f"目标参数: {self.metadata.get('target_col', 'Unknown')}")
        print(f"计算时间: {self.metadata.get('timestamp', 'Unknown')}")
        print(f"节点计算记录: {self.metadata.get('total_node_calculations', 0)}")
        print(f"流向计算记录: {self.metadata.get('total_flow_calculations', 0)}")
        print(f"警告数: {self.metadata.get('total_warnings', 0)}")
        print(f"错误数: {self.metadata.get('total_errors', 0)}")
        
        # 显示一些统计信息
        large_values = self.find_large_values(threshold=50)
        print(f"\n=== 异常值统计 (阈值=50) ===")
        for category, items in large_values.items():
            if items:
                print(f"{category}: {len(items)} 个异常值")
                # 显示最大的几个
                if category in ['large_y_n', 'large_y_up', 'large_E']:
                    sorted_items = sorted(items, key=lambda x: abs(x['value']), reverse=True)[:3]
                    for item in sorted_items:
                        print(f"  - COMID {item['comid']}, 日期 {item['date']}: {item['value']:.2f}")
                elif category == 'large_contribution':
                    sorted_items = sorted(items, key=lambda x: abs(x['contribution']), reverse=True)[:3]
                    for item in sorted_items:
                        print(f"  - {item['upstream_comid']}->{item['downstream_comid']}, 日期 {item['date']}: {item['contribution']:.2f}")
    
    def interactive_query(self):
        """交互式查询界面"""
        print("\n=== 交互式查询 ===")
        print("输入查询命令，输入 'help' 查看帮助，输入 'quit' 退出")
        
        while True:
            try:
                cmd = input("\n> ").strip().lower()
                
                if cmd == 'quit':
                    break
                elif cmd == 'help':
                    self._print_help()
                elif cmd == 'summary':
                    self.print_summary()
                elif cmd.startswith('node'):
                    self._handle_node_query(cmd)
                elif cmd.startswith('flow'):
                    self._handle_flow_query(cmd)
                elif cmd.startswith('large'):
                    self._handle_large_query(cmd)
                elif cmd.startswith('balance'):
                    self._handle_balance_query(cmd)
                else:
                    print("未知命令，输入 'help' 查看帮助")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"查询出错: {str(e)}")
    
    def _print_help(self):
        """打印帮助信息"""
        print("""
可用命令:
  summary                     - 显示调试信息摘要
  node <comid> [date]        - 查询节点计算信息
  flow <up_comid> <down_comid> [date] - 查询流向计算信息
  large [threshold]          - 查找异常大的值
  balance <comid> [date]     - 分析流量平衡
  help                       - 显示此帮助
  quit                       - 退出

示例:
  node 12345
  node 12345 2023-01-01
  flow 12345 67890
  large 100
  balance 12345
        """)
    
    def _handle_node_query(self, cmd):
        """处理节点查询命令"""
        parts = cmd.split()
        if len(parts) < 2:
            print("用法: node <comid> [date]")
            return
        
        comid = int(parts[1])
        date = parts[2] if len(parts) > 2 else None
        
        results = self.query_node_calculation(comid, date)
        if results:
            for i, result in enumerate(results):
                print(f"节点计算 {i+1}:")
                for key, value in result.items():
                    print(f"  {key}: {value}")
        else:
            print("未找到匹配的节点计算记录")
    
    def _handle_flow_query(self, cmd):
        """处理流向查询命令"""
        parts = cmd.split()
        if len(parts) < 3:
            print("用法: flow <upstream_comid> <downstream_comid> [date]")
            return
        
        upstream = int(parts[1])
        downstream = int(parts[2])
        date = parts[3] if len(parts) > 3 else None
        
        results = self.query_flow_calculation(upstream, downstream, date)
        if results:
            for i, result in enumerate(results):
                print(f"流向计算 {i+1}:")
                for key, value in result.items():
                    print(f"  {key}: {value}")
        else:
            print("未找到匹配的流向计算记录")
    
    def _handle_large_query(self, cmd):
        """处理异常值查询命令"""
        parts = cmd.split()
        threshold = float(parts[1]) if len(parts) > 1 else 100
        
        large_values = self.find_large_values(threshold)
        for category, items in large_values.items():
            if items:
                print(f"\n{category} ({len(items)} 个):")
                for item in items[:5]:  # 只显示前5个
                    print(f"  {item}")
    
    def _handle_balance_query(self, cmd):
        """处理流量平衡查询命令"""
        parts = cmd.split()
        if len(parts) < 2:
            print("用法: balance <comid> [date]")
            return
        
        comid = int(parts[1])
        date = parts[2] if len(parts) > 2 else None
        
        balance = self.analyze_comid_flow_balance(comid, date)
        print(f"COMID {comid} 流量平衡分析:")
        for key, value in balance.items():
            if key not in ['inflow_details', 'outflow_details', 'node_data']:
                print(f"  {key}: {value}")


def quick_analysis(debug_file_path: str, threshold: float = 100):
    """
    快速分析调试信息
    
    参数:
        debug_file_path: 调试信息文件路径
        threshold: 异常值阈值
    """
    query = FlowDebugQuery(debug_file_path)
    query.print_summary()
    
    print(f"\n=== 快速异常值分析 (阈值={threshold}) ===")
    large_values = query.find_large_values(threshold)
    
    for category, items in large_values.items():
        if items:
            print(f"\n{category}: {len(items)} 个异常值")
            # 显示最严重的几个
            if category in ['large_y_n', 'large_y_up', 'large_E']:
                sorted_items = sorted(items, key=lambda x: abs(x['value']), reverse=True)[:5]
                for item in sorted_items:
                    print(f"  COMID {item['comid']}, {item['date']}: {item['value']:.2f}")
            elif category == 'large_contribution':
                sorted_items = sorted(items, key=lambda x: abs(x['contribution']), reverse=True)[:5]
                for item in sorted_items:
                    print(f"  {item['upstream_comid']}->{item['downstream_comid']}, {item['date']}: {item['contribution']:.2f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python debug_query.py <debug_file_path> [threshold]")
        print("示例: python debug_query.py debug_info_TN_20231201_143022.json 100")
        sys.exit(1)
    
    debug_file = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    if not os.path.exists(debug_file):
        print(f"文件不存在: {debug_file}")
        sys.exit(1)
    
    # 执行快速分析
    quick_analysis(debug_file, threshold)
    
    # 启动交互式查询
    query = FlowDebugQuery(debug_file)
    query.interactive_query()