"""
Interactive Map Visualization

This module provides functions for creating interactive maps of 
E values for river segments using Folium.
"""

import logging
import os
import numpy as np

# Check if folium is available
try:
    import folium
    from folium.features import GeoJsonTooltip, GeoJsonPopup
    import branca.colormap as cm
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    logging.warning("Folium not installed. Interactive maps will not be available.")

def create_interactive_map(merged_gdf, e_column, target_parameter="TN",
                          title="River E Values", zoom_start=10, 
                          save_path=None, highlight_threshold=None):
    """
    Create an interactive map of E values using Folium
    
    Parameters:
    -----------
    merged_gdf : geopandas.GeoDataFrame
        GeoDataFrame with river segments and E values
    e_column : str
        Column name containing E values
    target_parameter : str
        Name of the target parameter (e.g., "TN" or "TP")
    title : str
        Map title
    zoom_start : int
        Initial zoom level
    save_path : str or None
        Path to save the HTML file, if None, the file is not saved
    highlight_threshold : float or None
        If provided, highlight river segments with E values above this threshold
    
    Returns:
    --------
    folium.Map or None
        Folium map object if folium is available, None otherwise
    """
    if not HAS_FOLIUM:
        logging.error("Folium is required for interactive maps. Please install folium.")
        return None
    
    logging.info(f"Creating interactive map for {e_column}")
    
    # Make a copy to avoid modifying the original
    plot_gdf = merged_gdf.copy()
    
    # Convert to WGS84 for Folium
    if plot_gdf.crs and plot_gdf.crs.to_string() != 'EPSG:4326':
        plot_gdf = plot_gdf.to_crs(epsg=4326)
        logging.info(f"Converted GeoDataFrame from {merged_gdf.crs} to EPSG:4326 for Folium")
    
    # Get the center of the map
    center = [
        plot_gdf.geometry.centroid.y.mean(),
        plot_gdf.geometry.centroid.x.mean()
    ]
    
    # Create a map
    m = folium.Map(location=center, zoom_start=zoom_start, 
                   tiles="OpenStreetMap")
    
    # Add a title
    title_html = f'''
        <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Create a colormap
    # Get min and max values for colormap, excluding NaN
    vmin = plot_gdf[e_column].dropna().min()
    vmax = plot_gdf[e_column].dropna().max()
    
    # Create a colormap
    colormap = cm.LinearColormap(
        colors=['blue', 'cyan', 'yellow', 'red'],
        vmin=vmin, vmax=vmax,
        caption=f'Local Contribution (E) - {target_parameter}'
    )
    
    # Add the colormap to the map
    m.add_child(colormap)
    
    # Create hover tooltip
    tooltip = GeoJsonTooltip(
        fields=['COMID', e_column],
        aliases=['River Segment ID:', f'{target_parameter} E Value:'],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 2px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=800,
    )
    
    # Create popup for click
    popup = GeoJsonPopup(
        fields=['COMID', e_column, 'order_'] if 'order_' in plot_gdf.columns 
               else ['COMID', e_column],
        aliases=['River Segment ID:', f'{target_parameter} E Value:', 'Stream Order:'] 
               if 'order_' in plot_gdf.columns else ['River Segment ID:', f'{target_parameter} E Value:'],
        localize=True,
        labels=True,
        style='font-family: courier new; font-size: 12px;'
    )
    
    # Plot segments with no E values in gray
    if plot_gdf[e_column].isna().any():
        no_e_values = plot_gdf[plot_gdf[e_column].isna()]
        folium.GeoJson(
            no_e_values,
            name="River Segments (No E Value)",
            style_function=lambda x: {
                'color': 'gray',
                'weight': 1,
                'opacity': 0.5
            },
            tooltip=folium.features.GeoJsonTooltip(
                fields=['COMID'],
                aliases=['River Segment ID:'],
                localize=True,
                sticky=False,
                labels=True
            )
        ).add_to(m)
        logging.info(f"Added {len(no_e_values)} river segments with no E values to the map")
    
    # Plot segments with E values with color scale
    e_values_gdf = plot_gdf.dropna(subset=[e_column])
    
    # If a highlight threshold is provided, split the data
    if highlight_threshold is not None:
        # Normal E values below threshold
        normal_e_values = e_values_gdf[e_values_gdf[e_column] <= highlight_threshold]
        
        # Add to map if not empty
        if not normal_e_values.empty:
            folium.GeoJson(
                normal_e_values,
                name="River Segments (Normal E Value)",
                style_function=lambda feature: {
                    'color': colormap(feature['properties'][e_column]),
                    'weight': 2,
                    'opacity': 0.7
                },
                tooltip=tooltip,
                popup=popup
            ).add_to(m)
            logging.info(f"Added {len(normal_e_values)} river segments with normal E values to the map")
        
        # High E values above threshold
        high_e_values = e_values_gdf[e_values_gdf[e_column] > highlight_threshold]
        
        # Add to map if not empty
        if not high_e_values.empty:
            folium.GeoJson(
                high_e_values,
                name=f"High E Values (> {highlight_threshold})",
                style_function=lambda feature: {
                    'color': 'red',
                    'weight': 3.5,
                    'opacity': 0.9
                },
                tooltip=tooltip,
                popup=popup
            ).add_to(m)
            logging.info(f"Added {len(high_e_values)} river segments with high E values to the map")
    else:
        # No threshold, add all E values with color scale
        folium.GeoJson(
            e_values_gdf,
            name="River Segments (with E Value)",
            style_function=lambda feature: {
                'color': colormap(feature['properties'][e_column]),
                'weight': 2.5,
                'opacity': 0.8
            },
            tooltip=tooltip,
            popup=popup
        ).add_to(m)
        logging.info(f"Added {len(e_values_gdf)} river segments with E values to the map")
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save to HTML file if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        m.save(save_path)
        logging.info(f"Interactive map saved to {save_path}")
    
    return m