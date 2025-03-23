"""
E Value Extraction Utilities

This module provides functions for extracting E values from PG-RWQ
flow routing results for visualization purposes.
"""

import pandas as pd
import numpy as np
import logging

def extract_e_values(df_flow, iteration=None, target_col="TN"):
    """
    Extract E values from the flow routing results for visualization
    
    Parameters:
    -----------
    df_flow : pandas.DataFrame
        The output from flow_routing_calculation function
    iteration : int or None
        The iteration number to extract. If None, extract the last iteration.
    target_col : str
        Target parameter column name ("TN" or "TP")
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with COMID and E values
    """
    logging.info("Extracting E values for visualization")
    
    # If iteration is not specified, find the last iteration
    if iteration is None:
        e_cols = [col for col in df_flow.columns if col.startswith('E_')]
        if not e_cols:
            raise ValueError("No E values found in the DataFrame")
        
        e_cols.sort(key=lambda x: int(x.split('_')[1]) if len(x.split('_')) > 1 and x.split('_')[1].isdigit() else 0)
        latest_e_col = e_cols[-1]
        iteration = int(latest_e_col.split('_')[1]) if len(latest_e_col.split('_')) > 1 else 0
        logging.info(f"Using latest iteration: {iteration}")
    else:
        latest_e_col = f'E_{iteration}'
        
    # Check if the column exists
    if latest_e_col not in df_flow.columns:
        raise ValueError(f"Column {latest_e_col} not found in DataFrame")
    
    # Extract COMID and E values
    # Aggregate by COMID and take the mean E value (since there may be multiple dates)
    e_values = df_flow.groupby('COMID')[latest_e_col].mean().reset_index()
    
    logging.info(f"Extracted E values for {len(e_values)} river segments")
    
    # Add some statistics for logging
    e_stats = {
        'min': e_values[latest_e_col].min(),
        'max': e_values[latest_e_col].max(),
        'mean': e_values[latest_e_col].mean(),
        'median': e_values[latest_e_col].median(),
        'std': e_values[latest_e_col].std()
    }
    
    logging.info(f"E value statistics: {e_stats}")
    
    return e_values