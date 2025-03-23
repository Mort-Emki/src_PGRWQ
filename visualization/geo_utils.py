"""
Geographic Data Utilities

This module provides functions for handling geographic data for 
PG-RWQ visualization, including loading river network shapefiles
and joining them with model outputs.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import os

def load_river_network_geo(shapefile_path):
    """
    Load the geographic data for the river network
    
    Parameters:
    -----------
    shapefile_path : str
        Path to the shapefile containing river network data
    
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with river segments and their geometries
    """
    logging.info(f"Loading river network shapefile from {shapefile_path}")
    
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    logging.info(f"Loaded shapefile with {len(gdf)} river segments")
    
    # Log the CRS information
    if gdf.crs:
        logging.info(f"Shapefile CRS: {gdf.crs}")
    else:
        logging.warning("Shapefile has no CRS information")
    
    # Log the columns in the shapefile
    logging.info(f"Shapefile columns: {list(gdf.columns)}")
    
    # Ensure COMID is present and handle different naming conventions
    if 'COMID' in gdf.columns:
        gdf['COMID'] = gdf['COMID'].astype(str)
    # For NHDPlus data, the column might be named COMID or some variation
    elif 'ComID' in gdf.columns:
        gdf = gdf.rename(columns={'ComID': 'COMID'})
        gdf['COMID'] = gdf['COMID'].astype(str)
    # If using NHDPlus, the COMID column might be 'REACHCODE'
    elif 'REACHCODE' in gdf.columns:
        gdf = gdf.rename(columns={'REACHCODE': 'COMID'})
        gdf['COMID'] = gdf['COMID'].astype(str)
    else:
        logging.warning("COMID column not found in the shapefile. Please ensure you have the correct identifier column.")
    
    return gdf

def join_e_values_with_geo(e_values, river_gdf):
    """
    Join E values with the geographic data
    
    Parameters:
    -----------
    e_values : pandas.DataFrame
        DataFrame with COMID and E values from extract_e_values()
    river_gdf : geopandas.GeoDataFrame
        GeoDataFrame with river segments from load_river_network_geo()
    
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with river segments and their E values
    """
    logging.info("Joining E values with geographic data")
    
    # Make sure COMID is string in both dataframes for proper joining
    e_values['COMID'] = e_values['COMID'].astype(str)
    river_gdf['COMID'] = river_gdf['COMID'].astype(str)
    
    # Merge E values with geographic data
    merged_gdf = river_gdf.merge(e_values, on='COMID', how='left')
    
    # Check how many river segments have E values
    e_column = e_values.columns[1]  # The column name for E values
    n_with_e = merged_gdf.dropna(subset=[e_column]).shape[0]
    n_total = merged_gdf.shape[0]
    n_missing = n_total - n_with_e
    
    logging.info(f"Successfully joined {n_with_e} river segments with E values")
    logging.info(f"Total river segments: {n_total}")
    logging.info(f"River segments without E values: {n_missing} ({n_missing/n_total*100:.1f}%)")
    
    return merged_gdf