#!/usr/bin/env python
"""
River Segment E Value Visualization

This script visualizes the local contribution (E) values for river segments from PG-RWQ model results.
It creates both static maps (PNG) using Matplotlib/GeoPandas and interactive maps (HTML) using Folium.

Usage:
    python visualize_e_values.py --flow_results path/to/flow_results.csv --shapefile path/to/river_network.shp
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import from parent modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utilities from PG-RWQ
from logging_utils import setup_logging, restore_stdout_stderr, ensure_dir_exists

# Import visualization modules
from visualization.extractor import extract_e_values
from visualization.geo_utils import load_river_network_geo, join_e_values_with_geo
from visualization.static_map import create_e_value_map
from visualization.interactive_map import create_interactive_map

def main():
    """Main function for the visualization script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize E values on a map')
    parser.add_argument('--flow_results', required=True, 
                        help='Path to the flow routing results CSV file')
    parser.add_argument('--shapefile', required=True, 
                        help='Path to the river network shapefile')
    parser.add_argument('--iteration', type=int, 
                        help='Iteration number to visualize (default: latest)')
    parser.add_argument('--target', default='TN', choices=['TN', 'TP'],
                        help='Target parameter to visualize (TN or TP)')
    parser.add_argument('--output_dir', default='visualizations',
                        help='Directory to save output files')
    parser.add_argument('--cmap', default='viridis',
                        help='Colormap for static map (e.g., viridis, Blues, YlOrRd)')
    parser.add_argument('--basemap', action='store_true',
                        help='Add basemap to static map')
    parser.add_argument('--highlight_threshold', type=float,
                        help='Highlight river segments with E values above this threshold')
    parser.add_argument('--log_dir', default='logs',
                        help='Directory to save log files')
    parser.add_argument('--no_interactive', action='store_true',
                        help='Skip creating interactive map (useful if folium is not installed)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = ensure_dir_exists(args.log_dir)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(log_dir=log_dir, log_filename=f"visualization_{timestamp}.log")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 1. Load flow routing results
        logging.info(f"Loading flow routing results from {args.flow_results}")
        try:
            df_flow = pd.read_csv(args.flow_results)
            logging.info(f"Flow results loaded successfully: {df_flow.shape[0]} rows, {df_flow.shape[1]} columns")
        except Exception as e:
            logging.error(f"Error loading flow results: {str(e)}")
            return 1
        
        # 2. Extract E values
        try:
            if args.iteration is None:
                logging.info("No iteration specified, finding the latest one")
                e_values = extract_e_values(df_flow, target_col=args.target)
                iteration = e_values.columns[1].split('_')[1]  # Get iteration from column name
                logging.info(f"Using latest iteration: {iteration}")
            else:
                logging.info(f"Extracting E values for iteration {args.iteration}")
                e_values = extract_e_values(df_flow, iteration=args.iteration, target_col=args.target)
                iteration = args.iteration
            
            e_column = e_values.columns[1]
            logging.info(f"Using E values from column: {e_column}")
        except Exception as e:
            logging.error(f"Error extracting E values: {str(e)}")
            return 1
        
        # 3. Load geographic data
        try:
            logging.info(f"Loading river network shapefile from {args.shapefile}")
            river_gdf = load_river_network_geo(args.shapefile)
        except Exception as e:
            logging.error(f"Error loading river network shapefile: {str(e)}")
            return 1
        
        # 4. Join E values with geographic data
        try:
            logging.info("Joining E values with geographic data")
            merged_gdf = join_e_values_with_geo(e_values, river_gdf)
        except Exception as e:
            logging.error(f"Error joining E values with geographic data: {str(e)}")
            return 1
        
        # 5. Create static map
        try:
            logging.info("Creating static map")
            static_map_path = os.path.join(args.output_dir, 
                                           f"river_{args.target}_E_values_iteration_{iteration}.png")
            
            fig = create_e_value_map(
                merged_gdf, 
                e_column, 
                title=f"Local Contribution (E) - {args.target} - Iteration {iteration}",
                cmap=args.cmap,
                add_basemap=args.basemap,
                legend_label=f"{args.target} E Value",
                save_path=static_map_path,
                highlight_threshold=args.highlight_threshold
            )
        except Exception as e:
            logging.error(f"Error creating static map: {str(e)}")
            # Continue to interactive map even if static map fails
        
        # 6. Create interactive map if requested
        if not args.no_interactive:
            try:
                logging.info("Creating interactive map")
                interactive_map_path = os.path.join(args.output_dir, 
                                                   f"interactive_river_{args.target}_E_values_iteration_{iteration}.html")
                
                interactive_map = create_interactive_map(
                    merged_gdf, 
                    e_column, 
                    target_parameter=args.target,
                    title=f"Local Contribution (E) - {args.target} - Iteration {iteration}",
                    save_path=interactive_map_path,
                    highlight_threshold=args.highlight_threshold
                )
            except Exception as e:
                logging.error(f"Error creating interactive map: {str(e)}")
        else:
            logging.info("Skipping interactive map creation as requested")
        
        logging.info(f"Visualization complete. Output files saved to {args.output_dir}/")
        if 'static_map_path' in locals():
            logging.info(f"Static map: {os.path.basename(static_map_path)}")
        if 'interactive_map_path' in locals() and not args.no_interactive:
            logging.info(f"Interactive map: {os.path.basename(interactive_map_path)}")
        
        # Show the static map
        if 'fig' in locals():
            plt.show()
        
    except Exception as e:
        logging.exception(f"Unexpected error: {str(e)}")
        return 1
    finally:
        # Cleanup and restore stdout/stderr
        logging.info("Visualization script completed")
        logging.shutdown()
        restore_stdout_stderr()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())