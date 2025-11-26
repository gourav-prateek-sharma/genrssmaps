#!/usr/bin/env python3
"""
Plotting utilities for RSS strength maps and coverage data.

Provides functions to:
1. Plot RSS strength from RSS data files (CSV, HDF5, NPZ, Parquet, Zarr)
2. Plot coverage maps from summary CSV files with configurable thresholds
3. CLI interface for both plotting modes
"""

import argparse
import os
import sys
from typing import Optional, List, Union
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False
    plt = None
    colors = None


def _load_rss_array(file_path: str) -> np.ndarray:
    """Load RSS array from various file formats (CSV, CSV.GZ, HDF5, NPZ, Parquet, Zarr)."""
    lower = file_path.lower()
    
    if lower.endswith('.csv.gz') or lower.endswith('.csv'):
        # CSV or CSV.GZ
        df = pd.read_csv(file_path, header=None)
        return df.values
    
    elif lower.endswith('.npz'):
        # NPZ format
        npz = np.load(file_path, allow_pickle=False)
        keys = list(npz.keys())
        if 'rss' in keys:
            return npz['rss']
        elif keys:
            return npz[keys[0]]
        else:
            raise ValueError(f"No arrays found in {file_path}")
    
    elif lower.endswith('.h5') or lower.endswith('.hdf5'):
        # HDF5 format
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 files. Install via: pip install h5py")
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            if 'rss' in keys:
                return f['rss'][()]
            elif keys:
                return f[keys[0]][()]
            else:
                raise ValueError(f"No datasets found in {file_path}")
    
    elif lower.endswith('.parquet'):
        # Parquet format
        df = pd.read_parquet(file_path)
        return df.values
    
    elif lower.endswith('.zarr'):
        # Zarr format
        try:
            import zarr
        except ImportError:
            raise ImportError("zarr is required for Zarr files. Install via: pip install zarr")
        z = zarr.open(file_path, mode='r')
        if hasattr(z, 'keys') and 'rss' in z.keys():
            return np.array(z['rss'])
        elif hasattr(z, 'keys'):
            keys = list(z.keys())
            if keys:
                return np.array(z[keys[0]])
        return np.array(z)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def plot_rss_strength(file_path: str, output_path: Optional[str] = None, 
                     title: Optional[str] = None, cmap: str = 'viridis',
                     vmin: Optional[float] = None, vmax: Optional[float] = None,
                     show: bool = False):
    """
    Plot RSS strength from an RSS data file.
    
    Args:
        file_path (str): Path to RSS file (CSV, HDF5, NPZ, Parquet, Zarr)
        output_path (str, optional): Path to save the plot image. If None, only shows if show=True
        title (str, optional): Plot title. Defaults to filename
        cmap (str): Matplotlib colormap name (default: 'viridis')
        vmin (float, optional): Minimum value for colorbar
        vmax (float, optional): Maximum value for colorbar
        show (bool): Whether to display the plot (default: False)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install via: pip install matplotlib")
    
    # Load RSS array
    try:
        rss_array = _load_rss_array(file_path)
    except Exception as e:
        raise RuntimeError(f"Error loading RSS file '{file_path}': {e}") from e
    
    # Convert to dBm if needed
    with np.errstate(divide='ignore'):
        rss_dbm = 10 * np.log10(rss_array)

    # 3D surface plot: x/y are receiver coordinates, z is signal strength
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate receiver grid coordinates
    ny, nx = rss_dbm.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    surf = ax.plot_surface(X, Y, rss_dbm, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='RSS Strength (dBm)')

    ax.set_xlabel('Receiver X Index')
    ax.set_ylabel('Receiver Y Index')
    ax.set_zlabel('RSS Strength (dBm)')
    if title is None:
        title = f"RSS Strength Map - {os.path.basename(file_path)}"
    ax.set_title(title)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    if show:
        plt.show()
    return fig


def plot_coverage_map(summary_csv: str, thresholds: Optional[List[float]] = None,
                     output_dir: Optional[str] = None, cmap: str = 'RdYlGn',
                     show: bool = False):
    """
    Plot coverage maps from a summary CSV file with one subplot per threshold.
    
    The summary CSV should have columns: x, y, z, coverage_thr<threshold1>, coverage_thr<threshold2>, ...
    
    Args:
        summary_csv (str): Path to summary CSV file
        thresholds (list, optional): List of thresholds to plot. If None, plots all available
        output_dir (str, optional): Directory to save plot images. If None, only shows if show=True
        cmap (str): Matplotlib colormap name (default: 'RdYlGn')
        show (bool): Whether to display the plots (default: False)
    
    Returns:
        dict: Mapping of threshold -> figure object
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install via: pip install matplotlib")
    
    # Load summary CSV
    try:
        df = pd.read_csv(summary_csv)
    except Exception as e:
        raise RuntimeError(f"Error loading summary CSV '{summary_csv}': {e}") from e
    
    # Find available coverage columns (support both coverage_thr-110p0 and coverage_thr110p0)
    coverage_cols = [col for col in df.columns if col.startswith('coverage_thr')]
    if not coverage_cols:
        raise ValueError(f"No coverage columns found in {summary_csv}. Expected 'coverage_thr<threshold>'")

    # Extract thresholds from column names (handle both with and without dash)
    available_thresholds = []
    col_map = {}  # Map normalized threshold string to actual column name
    for col in coverage_cols:
        thr_str = col.replace('coverage_thr', '')
        thr_str_norm = thr_str.replace('-', '').replace('p', '.').replace('_', '')
        try:
            thr = float(thr_str_norm)
            available_thresholds.append(thr)
            # Map both with and without dash for lookup
            col_map[f"coverage_thr{str(thr).replace('.', 'p')}"] = col
            col_map[f"coverage_thr-{str(thr).replace('.', 'p')}"] = col
        except ValueError:
            pass

    # Filter thresholds if specified
    if thresholds is None:
        thresholds = sorted(available_thresholds)
    else:
        thresholds = sorted(thresholds)
        # Verify requested thresholds are available
        for thr in thresholds:
            col_name1 = f"coverage_thr{str(thr).replace('.', 'p')}"
            col_name2 = f"coverage_thr-{str(thr).replace('.', 'p')}"
            if col_name1 not in col_map and col_name2 not in col_map:
                raise ValueError(f"Threshold {thr} not found in summary file. Available: {available_thresholds}")

    figures = {}

    for thr in thresholds:
        # Try both formats for column name
        col_name1 = f"coverage_thr{str(thr).replace('.', 'p')}"
        col_name2 = f"coverage_thr-{str(thr).replace('.', 'p')}"
        col_name = col_map.get(col_name1) or col_map.get(col_name2)

        # Get x, y, z from dataframe
        x = df['x'].values
        y = df['y'].values
        coverage = df[col_name].values

        # 3D surface plot: x/y are transmitter coordinates, z is coverage
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Remove NaNs for plotting
        mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(coverage))
        x_plot = x[mask]
        y_plot = y[mask]
        z_plot = coverage[mask]

        # Try to create a grid if possible
        try:
            x_unique = np.unique(x_plot)
            y_unique = np.unique(y_plot)
            if len(x_unique) > 1 and len(y_unique) > 1:
                X, Y = np.meshgrid(x_unique, y_unique)
                Z = np.full_like(X, np.nan, dtype=float)
                for xi, yi, cov in zip(x_plot, y_plot, z_plot):
                    xi_idx = np.where(x_unique == xi)[0]
                    yi_idx = np.where(y_unique == yi)[0]
                    if len(xi_idx) > 0 and len(yi_idx) > 0:
                        Z[yi_idx[0], xi_idx[0]] = cov
                surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none')
            else:
                surf = ax.scatter(x_plot, y_plot, z_plot, c=z_plot, cmap=cmap)
        except Exception:
            surf = ax.scatter(x_plot, y_plot, z_plot, c=z_plot, cmap=cmap)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Coverage')
        ax.set_xlabel('Transmitter X Coordinate')
        ax.set_ylabel('Transmitter Y Coordinate')
        ax.set_zlabel('Coverage')
        ax.set_title(f"Coverage Map @ {thr} dBm")

        figures[thr] = fig

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_file = os.path.join(output_dir, f"coverage_map_thr{str(thr).replace('.', 'p')}.png")
            fig.savefig(out_file, dpi=150, bbox_inches='tight')
            print(f"Saved plot to: {out_file}")
        if show:
            plt.show()
    return figures


def main(argv: Optional[list] = None):
    """CLI interface for plotting."""
    parser = argparse.ArgumentParser(description="Plot RSS strength maps and coverage data")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Subcommand: plot RSS strength
    rss_parser = subparsers.add_parser('rss', help='Plot RSS strength from a file')
    rss_parser.add_argument('file', type=str, help='Path to RSS file (CSV, HDF5, NPZ, Parquet, Zarr)')
    rss_parser.add_argument('--output', type=str, default=None, help='Output image path')
    rss_parser.add_argument('--title', type=str, default=None, help='Plot title')
    rss_parser.add_argument('--cmap', type=str, default='viridis', help='Colormap name (default: viridis)')
    rss_parser.add_argument('--vmin', type=float, default=None, help='Min value for colorbar')
    rss_parser.add_argument('--vmax', type=float, default=None, help='Max value for colorbar')
    rss_parser.add_argument('--show', action='store_true', help='Display plot')
    
    # Subcommand: plot coverage
    cov_parser = subparsers.add_parser('coverage', help='Plot coverage maps from summary CSV')
    cov_parser.add_argument('summary_csv', type=str, help='Path to summary CSV file')
    cov_parser.add_argument('--thresholds', type=float, nargs='+', default=None,
                           help='Thresholds to plot (space-separated). If not specified, plots all')
    cov_parser.add_argument('--output-dir', type=str, default=None, help='Output directory for images')
    cov_parser.add_argument('--cmap', type=str, default='RdYlGn', help='Colormap name (default: RdYlGn)')
    cov_parser.add_argument('--show', action='store_true', help='Display plots')
    
    args = parser.parse_args(argv)
    
    if not _HAS_MATPLOTLIB:
        print("Error: matplotlib is required for plotting. Install via: pip install matplotlib")
        sys.exit(1)
    
    if args.command == 'rss':
        try:
            plot_rss_strength(args.file, output_path=args.output, title=args.title,
                            cmap=args.cmap, vmin=args.vmin, vmax=args.vmax, show=args.show)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == 'coverage':
        try:
            plot_coverage_map(args.summary_csv, thresholds=args.thresholds,
                            output_dir=args.output_dir, cmap=args.cmap, show=args.show)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
