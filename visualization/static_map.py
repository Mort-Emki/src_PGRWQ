"""
Static Map Visualization

This module provides functions for creating static map visualizations
of E values for river segments using Matplotlib and GeoPandas.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import logging
import os

# Try to import contextily for basemap support, but don't fail if not available
try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    logging.warning("Contextily not installed. Basemaps will not be available.")

def create_e_value_map(merged_gdf, e_column, title="River Segment E Values", 
                       cmap='Blues', figsize=(12, 12), add_basemap=True,
                       legend_label="E Value", save_path=None,
                       highlight_threshold=None):
    """
    Create a map visualization of E values for river segments
    
    Parameters:
    -----------
    merged_gdf : geopandas.GeoDataFrame
        GeoDataFrame with river segments and E values
    e_column : str
        Column name containing E values
    title : str
        Map title
    cmap : str or matplotlib.colors.Colormap
        Colormap to use for the visualization
    figsize : tuple
        Figure size (width, height)
    add_basemap : bool
        Whether to add a basemap
    legend_label : str
        Label for the legend
    save_path : str or None
        Path to save the figure, if None, the figure is not saved
    highlight_threshold : float or None
        If provided, highlight river segments with E values above this threshold
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    logging.info(f"Creating static map visualization for {e_column}")
    
    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set background color
    ax.set_facecolor('#F2F2F2')
    
    # Create a copy to avoid modifying the original
    plot_gdf = merged_gdf.copy()
    
    # Get the min and max values for color scaling (excluding NaN)
    vmin = plot_gdf[e_column].min()
    vmax = plot_gdf[e_column].max()
    
    logging.info(f"E value range: {vmin:.4f} to {vmax:.4f}")
    
    # Plot the river segments without E values in light grey
    if plot_gdf[e_column].isna().any():
        plot_gdf[plot_gdf[e_column].isna()].plot(
            ax=ax, 
            color='lightgrey', 
            linewidth=0.5, 
            zorder=1,
            label='No E Value'
        )
        logging.info("Plotted river segments without E values in light grey")
    
    # Plot river segments with E values
    rivers = plot_gdf.dropna(subset=[e_column]).plot(
        column=e_column, 
        ax=ax, 
        linewidth=1.5, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax,
        zorder=2,
        legend=True,
        legend_kwds={
            'label': legend_label,
            'orientation': 'horizontal',
            'shrink': 0.8,
            'pad': 0.05
        }
    )
    logging.info("Plotted river segments with E values using colormap")
    
    # Highlight river segments above threshold if provided
    if highlight_threshold is not None:
        high_e_segments = plot_gdf[(~plot_gdf[e_column].isna()) & 
                                  (plot_gdf[e_column] > highlight_threshold)]
        if not high_e_segments.empty:
            high_e_segments.plot(
                ax=ax,
                color='red',
                linewidth=2.5,
                zorder=3,
                label=f'E > {highlight_threshold}'
            )
            # Add a legend for highlighted segments
            plt.legend()
            logging.info(f"Highlighted {len(high_e_segments)} river segments with E > {highlight_threshold}")
    
    # Add basemap if requested and contextily is available
    if add_basemap and HAS_CONTEXTILY:
        try:
            # Try to reproject to Web Mercator (EPSG:3857) if not already
            if plot_gdf.crs and plot_gdf.crs.to_string() != 'EPSG:3857':
                plot_gdf = plot_gdf.to_crs(epsg=3857)
                ax.set_aspect('equal')
                fig, ax = plt.subplots(figsize=figsize)
                
                # Re-plot everything in the correct projection
                if plot_gdf[e_column].isna().any():
                    plot_gdf[plot_gdf[e_column].isna()].plot(
                        ax=ax, 
                        color='lightgrey', 
                        linewidth=0.5, 
                        zorder=1
                    )
                
                rivers = plot_gdf.dropna(subset=[e_column]).plot(
                    column=e_column, 
                    ax=ax, 
                    linewidth=1.5, 
                    cmap=cmap, 
                    vmin=vmin, 
                    vmax=vmax,
                    zorder=2,
                    legend=True,
                    legend_kwds={
                        'label': legend_label,
                        'orientation': 'horizontal',
                        'shrink': 0.8,
                        'pad': 0.05
                    }
                )
                
                # Replot highlighted segments if needed
                if highlight_threshold is not None:
                    high_e_segments = plot_gdf[(~plot_gdf[e_column].isna()) & 
                                             (plot_gdf[e_column] > highlight_threshold)]
                    if not high_e_segments.empty:
                        high_e_segments.plot(
                            ax=ax,
                            color='red',
                            linewidth=2.5,
                            zorder=3,
                            label=f'E > {highlight_threshold}'
                        )
                        plt.legend()
            
            # Add the basemap
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
            logging.info("Added OpenStreetMap basemap")
        except Exception as e:
            logging.error(f"Could not add basemap: {str(e)}")
            logging.info("Make sure your GeoDataFrame has a valid CRS and contextily is installed.")
    elif add_basemap and not HAS_CONTEXTILY:
        logging.warning("Basemap requested but contextily not available. Install contextily to use basemaps.")
    
    # Add title and labels
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Figure saved to {save_path}")
    
    return fig