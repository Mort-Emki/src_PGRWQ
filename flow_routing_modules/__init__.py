# flow_routing_modules/__init__.py
"""
flow_routing_modules - 河网汇流计算模块包

重组后的模块结构：
- core: 核心计算功能
- physics: 物理过程计算
- debug: 调试和分析工具
"""

# 导入核心功能，提供简洁的接口
from .core.flow_routing import flow_routing_calculation
from .core.topology import (
    build_river_network_topology,
    calculate_indegrees,
    find_headwater_segments,
    find_terminal_segments
)
from .core.geometry import (
    get_river_length,
    calculate_river_width,
    calculate_river_surface_area,
    estimate_river_depth
)

# 导入物理过程
from .physics.environment_param import (
    compute_temperature_factor,
    compute_nitrogen_concentration_factor,
    compute_retainment_factor
)
from .physics.e_values import (
    load_e_values,
    apply_e_values,
    calculate_e_values
)

# 导入调试工具
from .debug.debug_query import FlowDebugQuery, quick_analysis

__version__ = "2.0.0"
__all__ = [
    # 核心功能
    'flow_routing_calculation',
    'build_river_network_topology',
    'calculate_indegrees',
    'find_headwater_segments',
    'find_terminal_segments',
    'get_river_length',
    'calculate_river_width',
    'calculate_river_surface_area',
    'estimate_river_depth',
    
    # 物理过程
    'compute_temperature_factor',
    'compute_nitrogen_concentration_factor',
    'compute_retainment_factor',
    'load_e_values',
    'apply_e_values',
    'calculate_e_values',
    
    # 调试工具
    'FlowDebugQuery',
    'quick_analysis'
]