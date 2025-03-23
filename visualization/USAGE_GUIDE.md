# PG-RWQ Visualization Module Usage Guide

This guide explains how to use the PG-RWQ visualization module to create maps of local contribution (E) values for river segments.

## Table of Contents

- [PG-RWQ Visualization Module Usage Guide](#pg-rwq-visualization-module-usage-guide)
  - [Table of Contents](#table-of-contents)
  - [Basic Usage](#basic-usage)
  - [Command Line Arguments](#command-line-arguments)
  - [Visualization Types](#visualization-types)
    - [Static Maps](#static-maps)
    - [Interactive Maps](#interactive-maps)
  - [Obtaining River Network Shapefiles](#obtaining-river-network-shapefiles)
    - [NHDPlus Data (for U.S. watersheds)](#nhdplus-data-for-us-watersheds)
    - [Custom Shapefiles](#custom-shapefiles)
  - [Integration with Main Workflow](#integration-with-main-workflow)
  - [Python API](#python-api)
  - [Customization](#customization)
    - [Colormaps](#colormaps)
    - [Highlighting High E Values](#highlighting-high-e-values)
  - [Advanced Examples](#advanced-examples)
    - [Comparing Multiple Iterations](#comparing-multiple-iterations)
    - [Creating Maps for Both TN and TP](#creating-maps-for-both-tn-and-tp)

## Basic Usage

The visualization module can be used as a standalone script:

```bash
python visualization/visualize_e_values.py \
    --flow_results path/to/flow_routing_results.csv \
    --shapefile path/to/river_network.shp
```

This will:
1. Load the flow routing results from the CSV file
2. Extract the E values from the latest iteration
3. Load the river network shapefile
4. Create both static and interactive maps
5. Save the maps to the `visualizations` directory

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--flow_results` | Path to the flow routing results CSV file | Required |
| `--shapefile` | Path to the river network shapefile | Required |
| `--iteration` | Iteration number to visualize | Latest iteration |
| `--target` | Target parameter to visualize (TN or TP) | TN |
| `--output_dir` | Directory to save output files | visualizations |
| `--cmap` | Colormap for static map | viridis |
| `--basemap` | Add basemap to static map | False (flag) |
| `--highlight_threshold` | Highlight river segments with E values above this threshold | None |
| `--log_dir` | Directory to save log files | logs |
| `--no_interactive` | Skip creating interactive map | False (flag) |

## Visualization Types

### Static Maps

Static maps are saved as PNG files and created using Matplotlib and GeoPandas. They provide a fixed view of the river network with E values color-coded according to the specified colormap.

Example:
```bash
python visualization/visualize_e_values.py \
    --flow_results results.csv \
    --shapefile river_network.shp \
    --cmap Blues \
    --basemap
```

### Interactive Maps

Interactive maps are saved as HTML files and created using Folium. They provide an interactive web-based map that allows zooming, panning, and hovering over river segments to see details.

Example:
```bash
python visualization/visualize_e_values.py \
    --flow_results results.csv \
    --shapefile river_network.shp \
    --highlight_threshold 0.5
```

To disable interactive maps (if Folium is not installed):
```bash
python visualization/visualize_e_values.py \
    --flow_results results.csv \
    --shapefile river_network.shp \
    --no_interactive
```

## Obtaining River Network Shapefiles

The visualization module requires a shapefile containing the river network with COMID identifiers that match those in your flow routing results. Here are some ways to obtain such shapefiles:

### NHDPlus Data (for U.S. watersheds)

If your COMIDs correspond to NHDPlus, you can download the data from:
- [USGS NHDPlus](https://www.usgs.gov/national-hydrography/access-national-hydrography-products)
- [EPA NHDPlus](https://www.epa.gov/waterdata/nhdplus-national-hydrography-dataset-plus)

### Custom Shapefiles

If you're using custom COMIDs, ensure your shapefile contains:
1. A 'COMID' column (or similar identifier that can be mapped to your COMIDs)
2. Geometry data for the river segments

## Integration with Main Workflow

The visualization module can be integrated with the main PG-RWQ workflow by adding the following arguments to your main.py command:

```bash
python main.py \
    --river_shapefile path/to/river_network.shp \
    --target TN \
    --basemap \
    --highlight_threshold 0.5 \
    --cmap viridis
```

This will automatically run the visualization after the flow routing calculation is complete.

## Python API

You can also use the visualization functions directly in your Python code:

```python
# 1. Import the necessary functions
from visualization.extractor import extract_e_values
from visualization.geo_utils import load_river_network_geo, join_e_values_with_geo
from visualization.static_map import create_e_value_map
from visualization.interactive_map import create_interactive_map

# 2. Load your data
df_flow = pd.read_csv('flow_results.csv')

# 3. Extract E values
e_values = extract_e_values(df_flow, iteration=5, target_col='TN')

# 4. Load river network shapefile
river_gdf = load_river_network_geo('river_network.shp')

# 5. Join E values with geographic data
merged_gdf = join_e_values_with_geo(e_values, river_gdf)

# 6. Create static map
fig = create_e_value_map(
    merged_gdf, 
    'E_5', 
    title="TN Local Contribution (E) - Iteration 5",
    cmap='viridis',
    add_basemap=True,
    legend_label="TN E Value",
    save_path="tn_e_values_map.png"
)

# 7. Create interactive map
interactive_map = create_interactive_map(
    merged_gdf, 
    'E_5', 
    target_parameter="TN",
    title="TN Local Contribution (E) - Iteration 5",
    save_path="interactive_tn_e_values.html"
)
```

## Customization

### Colormaps

You can use any Matplotlib colormap for the static maps:
- Sequential: 'viridis', 'plasma', 'inferno', 'magma', 'Blues', 'Greens', 'Reds', 'YlOrRd', etc.
- Diverging: 'RdBu', 'RdYlBu', 'BrBG', 'PiYG', etc.
- Qualitative: 'Set1', 'Set2', 'Set3', 'tab10', etc.

Example:
```bash
python visualization/visualize_e_values.py \
    --flow_results results.csv \
    --shapefile river_network.shp \
    --cmap RdYlBu_r
```

### Highlighting High E Values

To highlight river segments with high E values:
```bash
python visualization/visualize_e_values.py \
    --flow_results results.csv \
    --shapefile river_network.shp \
    --highlight_threshold 0.5
```

This will:
- In static maps: Mark river segments with E > 0.5 in red
- In interactive maps: Create a separate layer for high E value segments

## Advanced Examples

### Comparing Multiple Iterations

To create visualizations for multiple iterations:

```bash
for i in {1..5}; do
    python visualization/visualize_e_values.py \
        --flow_results results.csv \
        --shapefile river_network.shp \
        --iteration $i \
        --output_dir visualizations/iteration_$i
done
```

### Creating Maps for Both TN and TP

```bash
# Create TN visualization
python visualization/visualize_e_values.py \
    --flow_results results.csv \
    --shapefile river_network.shp \
    --target TN \
    --output_dir visualizations/TN \
    --cmap Blues

# Create TP visualization
python visualization/visualize_e_values.py \
    --flow_results results.csv \
    --shapefile river_network.shp \
    --target TP \
    --output_dir visualizations/TP \
    --cmap Reds
```