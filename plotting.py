#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os
try:
    from .coverage_helpers import (
        compute_coverage_from_csv_gz,
        compute_coverage_from_arr,
        THRESHOLD,
    )
except Exception as e:
    print(f"Error importing coverage helpers: {e}")
    raise
try:
    from .rate_helpers import compute_rate_from_arr, compute_rate_from_csv
except Exception as e:
    print(f"Error importing rate helpers: {e}")
    raise

def plot_rss_3d(csv_file, scale_factor=1.0, min_rss=None, max_rss=None, output_file=None):
    """
    Create a 3D plot of RSS values from a CSV file.
    Args:
        csv_file (str): Path to the CSV file containing RSS values
        scale_factor (float): Factor to scale the Z-axis (RSS values)
        min_rss (float): Minimum RSS value for normalization (optional)
        max_rss (float): Maximum RSS value for normalization (optional)
        output_file (str): Path to save the plot image (optional)
    """
    rss = pd.read_csv(csv_file, header=None).values
    dir_path = os.path.dirname(csv_file)
    all_files = [f for f in os.listdir(dir_path) if f.startswith('rss_munich_') and (f.endswith('.csv') or f.endswith('.csv.gz'))]
    x_coords, y_coords = [], []
    for f in all_files:
        try:
            coords = f.replace('rss_munich_', '').replace('.csv.gz', '').replace('.csv', '')
            x, y, _ = map(float, coords.split(','))
            x_coords.append(x)
            y_coords.append(y)
        except:
            continue
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    y, x = np.mgrid[0:rss.shape[0], 0:rss.shape[1]]
    x_scale = (x_max - x_min) / rss.shape[1]
    y_scale = (y_max - y_min) / rss.shape[0]
    x = x_min + x * x_scale
    y = y_min + y * y_scale
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    margin = 0.05
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    if min_rss is None:
        min_rss = np.min(rss)
    if max_rss is None:
        max_rss = np.max(rss)
    rss_positive = rss - np.min(rss) + 1
    rss_normalized = np.log10(rss_positive) * scale_factor
    surf = ax.plot_surface(x, y, rss_normalized, cmap='viridis',
                          linewidth=0.5, antialiased=True, alpha=0.9,
                          edgecolor='k', shade=True)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5,
                       label=f'RSS (dB, normalized, scale={scale_factor})',
                       format='%.1f')
    filename = os.path.basename(csv_file)
    try:
        if filename.startswith('rss_munich_'):
            coords = filename.replace('rss_munich_', '').replace('.csv.gz', '').replace('.csv', '')
            tx_x, tx_y, tx_z = map(float, coords.split(','))
            z_offset = (np.max(rss_normalized) - np.min(rss_normalized)) * 0.05
            ax.scatter(tx_x, tx_y, np.min(rss_normalized) - z_offset, 
                      color='red', marker='^', s=200, label='Transmitter Position')
    except Exception as e:
        print(f"Warning: Could not plot transmitter position: {e}")
    ax.set_xlabel(f'X coordinate (meters) [{x_min:.1f}, {x_max:.1f}]')
    ax.set_ylabel(f'Y coordinate (meters) [{y_min:.1f}, {y_max:.1f}]')
    ax.set_zlabel('RSS (dB, normalized)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.view_init(elev=25, azim=45)
    plt.style.use('default')
    ax.set_title('Radio Signal Strength Map\n' + 
                 f'Transmitter at ({tx_x:.1f}, {tx_y:.1f}, {tx_z:.1f})',
                 pad=20)
    ax.legend()
    plt.tight_layout()
    if output_file:
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(output_file, 
                       dpi=300, 
                       bbox_inches='tight',
                       pad_inches=0.2,
                       facecolor='white',
                       edgecolor='none')
            print(f"Plot saved to {output_file}")
        except Exception as e:
            print(f"Error saving plot to {output_file}: {e}")
    plt.show()

def plot_coverage_3d(data_or_csv, output_file=None):
    """
    Plot a 3D coverage map from either a directory of RSS files or a coverage summary CSV file.
    Args:
        data_or_csv (str): Directory containing RSS files (for raw computation) or path to coverage summary CSV file.
        output_file (str): Path to save the plot image (optional)
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import griddata

    if os.path.isdir(data_or_csv):
        all_files = [f for f in os.listdir(data_or_csv) if f.startswith('rss_munich_') and f.endswith('.csv.gz')]
        tx_positions = []
        coverage_values = []
        total_files = len(all_files)
        print(f"Found {total_files} RSS files to process")
        for idx, filename in enumerate(all_files, 1):
            print(f"\rProcessing file {idx}/{total_files} ({filename})", end="", flush=True)
            try:
                coords = filename.replace('rss_munich_', '').replace('.csv.gz', '')
                tx_x, tx_y, tx_z = map(float, coords.split(','))
                file_path = os.path.join(data_or_csv, filename)
                rss_array = pd.read_csv(file_path, header=None).values
                from coverage_helpers import compute_coverage_from_arr
                coverage = compute_coverage_from_arr(rss_array)
                tx_positions.append([tx_x, tx_y])
                coverage_values.append(coverage)
            except Exception as e:
                print(f"\nError processing {filename}: {e}")
                continue
        print("\nAll files processed successfully!")
        print(f"Generated coverage data for {len(tx_positions)} transmitter positions")
        tx_positions = np.array(tx_positions)
        coverage_values = np.array(coverage_values)
    else:
        df = pd.read_csv(data_or_csv)
        tx_positions = df[['x', 'y']].values
        # Auto-detect coverage column
        if 'coverage' in df.columns:
            coverage_values = df['coverage'].values
        else:
            cov_cols = [c for c in df.columns if c.startswith('coverage_thr')]
            if not cov_cols:
                raise ValueError("No coverage column found in summary file. Columns: " + str(df.columns))
            coverage_values = df[cov_cols[0]].values

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Filter out NaN values before interpolation
    tx_positions = np.array(tx_positions)
    coverage_values = np.array(coverage_values)
    mask = (~np.isnan(tx_positions).any(axis=1)) & (~np.isnan(coverage_values))
    tx_positions_valid = tx_positions[mask]
    coverage_values_valid = coverage_values[mask]
    if len(tx_positions_valid) == 0:
        raise ValueError("No valid transmitter positions for coverage plot.")
    x_min, x_max = np.min(tx_positions_valid[:, 0]), np.max(tx_positions_valid[:, 0])
    y_min, y_max = np.min(tx_positions_valid[:, 1]), np.max(tx_positions_valid[:, 1])
    margin = 0.05
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    grid_x, grid_y = np.mgrid[
        x_min - x_margin: x_max + x_margin: 100j,
        y_min - y_margin: y_max + y_margin: 100j
    ]
    grid_z = griddata(
        points=tx_positions_valid,
        values=coverage_values_valid,
        xi=(grid_x, grid_y),
        method='cubic',
        fill_value=np.min(coverage_values_valid)
    )
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', linewidth=0.5, antialiased=True, alpha=1.0, edgecolor='k')
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Coverage', format='%.2e')
    ax.set_xlabel('X coordinate (meters)')
    ax.set_ylabel('Y coordinate (meters)')
    ax.set_zlabel('Coverage')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.view_init(elev=30, azim=45)
    plt.title('Coverage Map', pad=20)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white', edgecolor='none')
        print(f"Plot saved to {output_file}")
    plt.show()

def plot_rate_3d(data_or_csv, output_file=None):
    """
    Plot a 3D rate map from either a directory of RSS files or a rate summary CSV file.
    Args:
        data_or_csv (str): Directory containing RSS files (for raw computation) or path to rate summary CSV file.
        output_file (str): Path to save the plot image (optional)
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import griddata

    if os.path.isdir(data_or_csv):
        from rate_helpers import compute_rate_from_csv
        all_files = [f for f in os.listdir(data_or_csv) if f.startswith('rss_munich_') and (f.endswith('.csv') or f.endswith('.csv.gz'))]
        tx_positions = []
        rate_values = []
        total_files = len(all_files)
        print(f"Found {total_files} RSS files to process")
        for idx, filename in enumerate(all_files, 1):
            print(f"\rProcessing file {idx}/{total_files} ({filename})", end="", flush=True)
            try:
                coords = filename.replace('rss_munich_', '').replace('.csv.gz', '').replace('.csv', '')
                tx_x, tx_y, tx_z = map(float, coords.split(','))
                file_path = os.path.join(data_or_csv, filename)
                rate = compute_rate_from_csv(file_path)
                tx_positions.append([tx_x, tx_y])
                rate_values.append(rate)
            except Exception as e:
                print(f"\nError processing {filename}: {e}")
                continue
        print("\nAll files processed successfully!")
        print(f"Generated rate data for {len(tx_positions)} transmitter positions")
        tx_positions = np.array(tx_positions)
        rate_values = np.array(rate_values)
    else:
        df = pd.read_csv(data_or_csv)
        tx_positions = df[['x', 'y']].values
        rate_values = df['avg_rate'].values if 'avg_rate' in df.columns else df['rate'].values

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x_min, x_max = np.min(tx_positions[:, 0]), np.max(tx_positions[:, 0])
    y_min, y_max = np.min(tx_positions[:, 1]), np.max(tx_positions[:, 1])
    margin = 0.05
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    grid_x, grid_y = np.mgrid[
        x_min - x_margin: x_max + x_margin: 100j,
        y_min - y_margin: y_max + y_margin: 100j
    ]
    grid_z = griddata(
        points=tx_positions,
        values=rate_values,
        xi=(grid_x, grid_y),
        method='cubic',
        fill_value=np.min(rate_values)
    )
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='plasma', linewidth=0.5, antialiased=True, alpha=1.0, edgecolor='k')
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Rate (bits/s/Hz)', format='%.2e')
    ax.set_xlabel('X coordinate (meters)')
    ax.set_ylabel('Y coordinate (meters)')
    ax.set_zlabel('Rate (bits/s/Hz)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.view_init(elev=30, azim=45)
    plt.title('Rate Map', pad=20)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white', edgecolor='none')
        print(f"Plot saved to {output_file}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot 3D RSS map, coverage map, or rate map from CSV files.')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    rss_parser = subparsers.add_parser('rss', help='Plot 3D RSS map from single CSV file')
    rss_parser.add_argument('csv_file', help='Path to the CSV file containing RSS values')
    rss_parser.add_argument('--scale', type=float, default=1.0,
                          help='Scale factor for RSS values (default: 1.0)')
    rss_parser.add_argument('--min_rss', type=float, default=None,
                          help='Minimum RSS value for normalization')
    rss_parser.add_argument('--max_rss', type=float, default=None,
                          help='Maximum RSS value for normalization')
    rss_parser.add_argument('--output', type=str, default=None,
                          help='Path to save the plot image (e.g., plot.png, plot.pdf)')
    coverage_parser = subparsers.add_parser('coverage', help='Plot 3D coverage map from RSS directory or coverage summary CSV')
    coverage_parser.add_argument('data_or_csv', help='Path to directory with RSS files or coverage summary CSV file')
    coverage_parser.add_argument('--output', type=str, default=None, help='Path to save the plot image (e.g., coverage_plot.png)')
    rate_parser = subparsers.add_parser('rate', help='Plot 3D rate map from RSS directory or rate summary CSV')
    rate_parser.add_argument('data_or_csv', help='Path to directory with RSS files or rate summary CSV file')
    rate_parser.add_argument('--output', type=str, default=None, help='Path to save the plot image (e.g., rate_plot.png)')
    args = parser.parse_args()
    if args.command == 'rss':
        plot_rss_3d(args.csv_file, args.scale, args.min_rss, args.max_rss, args.output)
    elif args.command == 'coverage':
        plot_coverage_3d(args.data_or_csv, args.output)
    elif args.command == 'rate':
        plot_rate_3d(args.data_or_csv, args.output)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
