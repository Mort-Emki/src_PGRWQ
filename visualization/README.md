# PG-RWQ Visualization Module

This module provides tools for visualizing PG-RWQ model results, particularly the local contribution (E) values for river segments on geographic maps.

## Features

- Extract local contribution (E) values from flow routing results
- Load and process river network shapefiles
- Create static map visualizations using Matplotlib and GeoPandas
- Create interactive map visualizations using Folium
- Highlight river segments with E values above specified thresholds
- Support for both TN and TP parameters

## Requirements

- Python 3.6+
- Core dependencies:
  - pandas
  - numpy
  - matplotlib
- Geographic visualization:
  - geopandas
  - contextily (optional, for basemaps)
  - folium (optional, for interactive maps)

## Installation

Install the required dependencies:

```bash
pip install pandas numpy matplotlib geopandas
pip install contextily folium  # Optional but recommended for full functionality
```

## Usage

### Command Line Interface

The main script `visualize_e_values.py` provides a command-line interface for visualization:

```bash
python visualization/visualize_e_values.py \
    --flow_results path/to/flow_routing_results.csv \
    --shapefile path/to/river_network.shp \
    --target TN \
    --output_dir visualizations/TN \
    --cmap viridis \
    --basemap
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--flow_results` | Path to the flow routing results CSV file (required) |
| `--shapefile` | Path to the river network shapefile (required) |
| `--iteration` | Iteration number to visualize (default: latest) |
| `--target` | Target parameter to visualize: TN or TP (default: TN) |
| `--output_dir` | Directory to save output files (default: visualizations) |
| `--cmap` | Colormap for static map (default: viridis) |
| `--basemap` | Add basemap to static map (flag) |
| `--highlight_threshold` | Highlight river segments with E values above this threshold |
| `--log_dir` | Directory to save log files (default: logs) |
| `--no_interactive` | Skip creating interactive map (flag) |

### Python API

You can also use the module's functions in your own Python code:

```python
from visualization.extractor import extract_e_values
from visualization.geo_utils import load_river_network_geo, join_e_values_with_geo
from visualization.static_map import create_e_value_map
from visualization.interactive_map import create_interactive_map

# 1. Extract E values from flow routing results
e_values = extract_e_values(df_flow, iteration=5)

# 2. Load river network shapefile
river_gdf = load_river_network_geo('path/to/river_network.shp')

# 3. Join E values with geographic data
merged_gdf = join_e_values_with_geo(e_values, river_gdf)

# 4. Create static map
fig = create_e_value_map(merged_gdf, 'E_5', title="TN Local Contribution (E)")

# 5. Create interactive map
interactive_map = create_interactive_map(merged_gdf, 'E_5', target_parameter="TN")
```

## Output

The visualization script creates two types of outputs:

1. **Static Maps (PNG files):**
   - Generated using Matplotlib and GeoPandas
   - Supports various colormaps
   - Optional basemaps via contextily

2. **Interactive Maps (HTML files):**
   - Generated using Folium
   - Interactive tooltips showing COMID and E values
   - Popups with additional information
   - Layer controls for toggling different data layers

## Integration with PG-RWQ

This module integrates with the existing PG-RWQ codebase, leveraging the same logging system and data structures. It can be used after running the flow routing calculation to visualize the results.

Example integration with main workflow:

```python
import flow_routing
from visualization.visualize_e_values import main as visualize_main
import sys

# Run flow routing calculation
df_flow = flow_routing.flow_routing_calculation(...)

# Save flow routing results
df_flow.to_csv('flow_results.csv', index=False)

# Run visualization
sys.argv = [
    'visualize_e_values.py',
    '--flow_results', 'flow_results.csv',
    '--shapefile', 'river_network.shp',
    '--target', 'TN',
    '--output_dir', 'visualizations',
    '--basemap'
]
visualize_main()
```

## Further Development

Potential enhancements for future development:

- Animation of E values across iterations
- Comparative visualization of TN vs TP
- Watershed-level aggregation and visualization
- Integration of upstream-downstream connectivity visualization
- 3D visualization options for E values