"""
topology.py - 河网拓扑结构模块

本模块提供了处理河网拓扑结构的功能，包括构建河网拓扑、
计算河段入度、处理缺失数据河段等。这些功能是执行汇流计算的基础。

主要功能:
1. 构建河网拓扑结构
2. 处理缺失数据河段的绕过路径
3. 计算河段入度
4. 检测河网中的环路和异常
"""

import pandas as pd
import logging
from typing import Dict, Set, List, Tuple, Any


def build_river_network_topology(river_info, missing_data_comids=None):
    """
    构建河网拓扑结构，处理缺失数据河段的绕过路径
    
    本函数基于河网信息构建下游河段映射，并可选择性地处理缺失数据的河段。
    当有缺失数据的河段时，会自动寻找"绕过路径"，直接连接到下一个有数据的下游河段。
    
    参数:
        river_info: 包含河网信息的DataFrame，必须包含'COMID'和'NextDownID'列
        missing_data_comids: 缺失数据的河段ID集合(可选)
    
    返回:
        dict: 更新后的下游河段映射字典，键为COMID，值为下游COMID
    """
    if missing_data_comids is None:
        missing_data_comids = set()
        
    # 从河网信息中构建下游河段映射
    next_down_ids = river_info.set_index('COMID')['NextDownID'].to_dict()
    
    # 如果有缺失数据的河段，处理绕过路径
    if missing_data_comids:
        # 递归查找下一个可用的下游河段（跳过缺失的河段）
        def find_next_available_downstream(comid, visited=None):
            """递归查找下一个非缺失的下游河段"""
            if visited is None:
                visited = set()
            
            if comid in visited:  # 检测循环引用
                return 0
            visited.add(comid)
            
            next_down = next_down_ids.get(comid, 0)
            if next_down == 0:  # 已到末端
                return 0
            
            if str(next_down) not in missing_data_comids:  # 下游河段有数据
                return next_down
            
            # 下游河段无数据，继续查找其下游
            return find_next_available_downstream(next_down, visited)
        
        # 处理所有河段，查找下一个可用下游
        bypassed_count = 0
        modified_next_down = {}
        
        for comid in list(next_down_ids.keys()):
            if str(comid) in missing_data_comids:
                continue  # 跳过处理缺失数据的河段
            
            next_down = next_down_ids.get(comid, 0)
            if next_down != 0 and str(next_down) in missing_data_comids:
                # 查找下一个可用的非缺失数据的下游
                next_available = find_next_available_downstream(next_down)
                modified_next_down[comid] = next_available
                bypassed_count += 1
        
        # 更新下游映射
        next_down_ids.update(modified_next_down)
        logging.info(f"在河网拓扑中绕过了 {bypassed_count} 个缺失数据的河段")
    
    return next_down_ids


def calculate_indegrees(comids, next_down_ids):
    """
    计算河网中每个节点的入度（上游河段数量）
    
    入度定义为流入该河段的上游河段数量。头部河段（无上游）的入度为0。
    
    参数:
        comids: 河段ID列表或集合
        next_down_ids: 下游河段映射字典，键为COMID，值为下游COMID
    
    返回:
        dict: 每个河段的入度字典，键为COMID，值为入度
    """
    indegree = {comid: 0 for comid in comids}
    
    for comid in comids:
        next_down = next_down_ids.get(comid, 0)
        if next_down != 0 and next_down in indegree:
            indegree[next_down] = indegree.get(next_down, 0) + 1
            
    return indegree


def detect_cycles_in_network(next_down_ids):
    """
    检测河网中的循环引用
    
    参数:
        next_down_ids: 下游河段映射字典
    
    返回:
        list: 检测到的循环列表，每个循环是一个COMID序列
    """
    cycles = []
    visited = set()
    
    def detect_cycle(comid, path):
        """递归检测循环"""
        if comid in path:
            # 找到一个循环
            cycle_start = path.index(comid)
            return path[cycle_start:] + [comid]
        
        if comid in visited or comid == 0:
            return None
        
        visited.add(comid)
        path.append(comid)
        
        next_down = next_down_ids.get(comid, 0)
        if next_down == 0:
            # 已到末端
            path.pop()
            return None
        
        cycle = detect_cycle(next_down, path)
        path.pop()
        
        return cycle
    
    # 检查每个河段
    for comid in next_down_ids:
        if comid not in visited:
            cycle = detect_cycle(comid, [])
            if cycle:
                cycles.append(cycle)
    
    return cycles


def find_headwater_segments(indegree):
    """
    找出所有头部河段（入度为0）
    
    参数:
        indegree: 入度字典，键为COMID，值为入度
    
    返回:
        list: 头部河段COMID列表
    """
    return [comid for comid, degree in indegree.items() if degree == 0]


def find_terminal_segments(next_down_ids):
    """
    找出所有终端河段（NextDownID为0）
    
    参数:
        next_down_ids: 下游河段映射字典
    
    返回:
        list: 终端河段COMID列表
    """
    return [comid for comid, next_down in next_down_ids.items() if next_down == 0]


def calculate_stream_order(next_down_ids, indegree):
    """
    计算河流序数（Strahler序）
    
    参数:
        next_down_ids: 下游河段映射字典
        indegree: 入度字典
    
    返回:
        dict: 河段序数字典，键为COMID，值为序数
    """
    # 找出所有头部河段
    headwaters = find_headwater_segments(indegree)
    
    # 初始化序数
    stream_order = {comid: 0 for comid in next_down_ids}
    for hw in headwaters:
        stream_order[hw] = 1
    
    # 创建上游映射（反向映射）
    upstream_map = {}
    for comid, next_down in next_down_ids.items():
        if next_down == 0:
            continue
        if next_down not in upstream_map:
            upstream_map[next_down] = []
        upstream_map[next_down].append(comid)
    
    # 处理的队列（先处理头部河段）
    queue = headwaters.copy()
    processed = set(headwaters)
    
    while queue:
        comid = queue.pop(0)
        next_down = next_down_ids.get(comid, 0)
        
        if next_down == 0 or next_down not in upstream_map:
            continue
        
        # 获取所有上游
        upstreams = upstream_map[next_down]
        
        # 只有当所有上游都已处理时，才处理下游
        if all(up in processed for up in upstreams):
            # 计算下游的序数
            up_orders = [stream_order[up] for up in upstreams]
            max_order = max(up_orders)
            max_count = up_orders.count(max_order)
            
            if max_count > 1:
                # 如果有多个相同的最大序数，下游序数加1
                stream_order[next_down] = max_order + 1
            else:
                # 否则保持最大序数
                stream_order[next_down] = max_order
            
            processed.add(next_down)
            queue.append(next_down)
    
    return stream_order


def calculate_drainage_area(next_down_ids, area_dict):
    """
    计算每个河段的汇水面积
    
    参数:
        next_down_ids: 下游河段映射字典
        area_dict: 每个河段的本地汇水面积字典
    
    返回:
        dict: 累积汇水面积字典
    """
    # 创建上游映射
    upstream_map = {}
    for comid, next_down in next_down_ids.items():
        if next_down == 0:
            continue
        if next_down not in upstream_map:
            upstream_map[next_down] = []
        upstream_map[next_down].append(comid)
    
    # 累积汇水面积字典
    cumulative_area = {}
    
    # 递归计算累积汇水面积
    def calculate_area(comid):
        # 如果已经计算过，直接返回
        if comid in cumulative_area:
            return cumulative_area[comid]
        
        # 本地汇水面积
        local_area = area_dict.get(comid, 0)
        
        # 如果没有上游，只返回本地面积
        if comid not in upstream_map:
            cumulative_area[comid] = local_area
            return local_area
        
        # 计算所有上游的累积面积
        upstream_area = sum(calculate_area(up) for up in upstream_map[comid])
        
        # 本地面积加上游面积
        total_area = local_area + upstream_area
        cumulative_area[comid] = total_area
        
        return total_area
    
    # 计算每个河段的累积面积
    for comid in next_down_ids:
        if comid not in cumulative_area:
            calculate_area(comid)
    
    return cumulative_area