import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import os
from datetime import datetime

def plot_memory_usage(log_file="gpu_memory_log.csv", save_path=None, show_plot=True):
    """
    Plot GPU memory usage from the log file.
    
    Args:
        log_file: Path to the CSV log file
        save_path: Path to save the output figure (if None, figure is not saved)
        show_plot: Whether to display the plot
    """
    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found")
        return
    
    # Read the log file
    df = pd.read_csv(log_file)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate elapsed time in minutes from start
    start_time = df['timestamp'].min()
    df['elapsed_minutes'] = (df['timestamp'] - start_time).dt.total_seconds() / 60
    
    # Set a nicer style
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 8))
    
    # Plot memory usage
    plt.subplot(2, 1, 1)
    plt.plot(df['elapsed_minutes'], df['allocated_mb'], 'b-', label='Allocated Memory (MB)')
    plt.plot(df['elapsed_minutes'], df['reserved_mb'], 'r--', label='Reserved Memory (MB)')
    plt.plot(df['elapsed_minutes'], df['max_allocated_mb'], 'g-.', label='Max Allocated Memory (MB)')
    plt.ylabel('Memory (MB)')
    plt.title('GPU Memory Usage Over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot percentage used
    plt.subplot(2, 1, 2)
    plt.plot(df['elapsed_minutes'], df['percent_used'], 'b-')
    plt.axhline(y=90, color='r', linestyle='--', label='90% Usage Warning')
    plt.xlabel('Elapsed Time (minutes)')
    plt.ylabel('GPU Memory Usage (%)')
    plt.title('GPU Memory Usage Percentage')
    plt.grid(True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Add annotations for key statistics
    max_usage = df['percent_used'].max()
    avg_usage = df['percent_used'].mean()
    final_usage = df['percent_used'].iloc[-1]
    
    plt.figtext(0.02, 0.02, 
                f"Max Usage: {max_usage:.2f}% | Avg Usage: {avg_usage:.2f}% | Final Usage: {final_usage:.2f}%", 
                fontsize=10)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(f"{save_path}/memory_usage_{timestamp}.png", dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}/memory_usage_{timestamp}.png")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def analyze_memory_spikes(log_file="gpu_memory_log.csv", threshold_percent=10):
    """
    Analyze memory spikes from the log file.
    
    Args:
        log_file: Path to the CSV log file
        threshold_percent: Threshold for identifying memory spikes (percent change)
    """
    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found")
        return
    
    # Read the log file
    df = pd.read_csv(log_file)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate memory changes between consecutive records
    df['memory_change_mb'] = df['allocated_mb'].diff()
    df['memory_change_percent'] = df['memory_change_mb'] / df['allocated_mb'].shift(1) * 100
    
    # Identify significant spikes
    spikes = df[abs(df['memory_change_percent']) > threshold_percent].copy()
    
    # Print summary of spikes
    if len(spikes) > 0:
        print(f"Found {len(spikes)} memory spikes (changes > {threshold_percent}%):")
        for i, row in spikes.iterrows():
            direction = "Increase" if row['memory_change_mb'] > 0 else "Decrease"
            print(f"{row['timestamp']}: {direction} of {abs(row['memory_change_mb']):.2f} MB ({abs(row['memory_change_percent']):.2f}%)")
        
        # Plot the spikes
        plt.figure(figsize=(14, 6))
        plt.plot(df['timestamp'], df['allocated_mb'], 'b-', label='Allocated Memory (MB)')
        plt.scatter(spikes['timestamp'], spikes['allocated_mb'], c='r', s=50, label='Memory Spikes')
        plt.xlabel('Time')
        plt.ylabel('Memory (MB)')
        plt.title('GPU Memory Usage with Spikes Highlighted')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No significant memory spikes found (threshold: {threshold_percent}%)")

def main():
    parser = argparse.ArgumentParser(description='Visualize GPU memory usage logs')
    parser.add_argument('--log_file', type=str, default='gpu_memory_log.csv',
                        help='Path to the memory log CSV file')
    parser.add_argument('--save_path', type=str, default='memory_plots',
                        help='Directory to save the output plots')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display the plot (only save)')
    parser.add_argument('--analyze_spikes', action='store_true',
                        help='Analyze memory usage spikes')
    parser.add_argument('--spike_threshold', type=float, default=10.0,
                        help='Threshold percentage for spike detection')
    
    args = parser.parse_args()
    
    # Plot memory usage
    plot_memory_usage(args.log_file, args.save_path, not args.no_show)
    
    # Analyze spikes if requested
    if args.analyze_spikes:
        analyze_memory_spikes(args.log_file, args.spike_threshold)

if __name__ == "__main__":
    main()