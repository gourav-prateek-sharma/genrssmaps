#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
from sionna.rt import load_scene, scene as rt_scene
from .coverage_helpers import rss_map_full, MAX_DEPTH, CELL_SIZE, SAMPLES_PER_TX
from .scene_helpers import get_scene_bounds3d
import traceback
import gzip
import shutil
from typing import Tuple, Optional, Dict, Any
from .storage_utils import save_rss_data, get_format_info

def sanitize_filename_part(s: str) -> str:
    """Make a safe filename part (replace commas/spaces and colons)."""
    return s.replace(" ", "_").replace(",", "_").replace(":", "_").replace("/", "_")

def rss_write_csv(scene_obj, tx_position, max_depth=MAX_DEPTH, cell_size=CELL_SIZE,
                  samples_per_tx=SAMPLES_PER_TX, csv_file="rss_output.csv", compress=False) -> Tuple[np.ndarray, str]:
    """
    Generates the RSS tensor using rss_map_full and writes it to a CSV file.
    Returns (rss_array (2D numpy), final outfile path).
    """
    # Generate RSS (we handle file writing here)
    rss = rss_map_full(scene_obj, tx_position=tx_position, max_depth=max_depth,
                       cell_size=cell_size, samples_per_tx=samples_per_tx, csv_file=None)

    # Convert to numpy array if necessary
    if hasattr(rss, "numpy"):
        rss_array = rss.numpy()
    else:
        rss_array = np.array(rss)

    # Squeeze to 2-D if possible (we expect height x width)
    if rss_array.ndim != 2:
        rss_array = np.squeeze(rss_array)
        if rss_array.ndim != 2:
            raise ValueError(f"After squeezing, RSS array must have 2 dimensions. Current shape: {rss_array.shape}")

    # Prepare final path and compression
    out_path = csv_file
    if compress and not out_path.endswith(".gz"):
        out_path = out_path + ".gz"

    # Write via pandas to a temporary uncompressed CSV then gzip it (safer)
    tmp_csv = out_path if not compress else out_path + ".tmpcsv"

    df = pd.DataFrame(rss_array)
    df.to_csv(tmp_csv, index=False, header=False)

    if compress:
        with open(tmp_csv, "rb") as f_in, gzip.open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(tmp_csv)

    return rss_array, out_path


def rss_write_efficient(scene_obj, tx_position, max_depth=MAX_DEPTH, cell_size=CELL_SIZE,
                       samples_per_tx=SAMPLES_PER_TX, output_file="rss_output", 
                       format_type="npz", compression_level=6,
                       include_metadata=True) -> Tuple[np.ndarray, str]:
    """
    Generate RSS tensor and save in efficient storage format.
    
    Args:
        scene_obj: Sionna scene object
        tx_position: Transmitter position [x, y, z]
        max_depth: Ray tracing depth
        cell_size: Grid cell size
        samples_per_tx: Number of samples per transmitter
        output_file: Output filename (extension will be auto-added)
        format_type: Storage format ('npz', 'hdf5', 'parquet', 'zarr', 'csv')
        compression_level: Compression level 0-9 (higher = better compression)
        include_metadata: Whether to save metadata with the RSS data
    
    Returns:
        Tuple of (rss_array, final_output_path)
        
    Raises:
        ImportError: If required dependencies for format are missing
        ValueError: If format is not supported
    """
    
    # Generate RSS map
    rss = rss_map_full(scene_obj, tx_position=tx_position, max_depth=max_depth,
                       cell_size=cell_size, samples_per_tx=samples_per_tx, csv_file=None)
    
    # Convert to numpy array if necessary
    if hasattr(rss, "numpy"):
        rss_array = rss.numpy()
    else:
        rss_array = np.array(rss)
    
    # Squeeze to 2-D if possible
    if rss_array.ndim != 2:
        rss_array = np.squeeze(rss_array)
        if rss_array.ndim != 2:
            raise ValueError(f"After squeezing, RSS array must have 2 dimensions. Current shape: {rss_array.shape}")
    
    # Prepare metadata if requested
    metadata = None
    if include_metadata:
        metadata = {
            'tx_position': list(tx_position) if hasattr(tx_position, '__iter__') else [tx_position],
            'max_depth': max_depth,
            'cell_size': list(cell_size) if hasattr(cell_size, '__iter__') else [cell_size],
            'samples_per_tx': samples_per_tx,
            'data_shape': list(rss_array.shape),
            'format_version': '1.0'
        }
        
        # Try to get scene name if available
        try:
            if hasattr(scene_obj, 'name'):
                metadata['scene_name'] = scene_obj.name
            elif hasattr(scene_obj, '_name'):
                metadata['scene_name'] = scene_obj._name
        except:
            pass
    
    # Save using efficient storage
    final_path = save_rss_data(
        rss_array=rss_array,
        filepath=output_file,
        format_type=format_type,
        compression_level=compression_level,
        metadata=metadata
    )
    
    return rss_array, final_path


def load_tx_positions_from_file(path):
    """
    Robustly read tx positions from a file. Accepts comma- or whitespace-separated files.
    Uses pandas (attempt header=None and header=0) and falls back to numpy.
    Returns an Nx3 ndarray.
    """
    # First try pandas with header=0; many CSVs have headers that pandas can handle
    try:
        df = pd.read_csv(path, header=0)
        # Keep first three numeric columns (or first three columns if numeric selection fails)
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.shape[1] >= 3:
            arr = df_numeric.iloc[:, :3].values
        else:
            arr = df.iloc[:, :3].values
    except Exception:
        # fallback to numpy loadtxt with common delimiters
        try:
            arr = np.loadtxt(path, delimiter=",")
        except Exception:
            arr = np.loadtxt(path)
    arr = np.atleast_2d(arr[:, :3])
    return arr

def make_expected_filename(scene: str, pos: np.ndarray, cell_size: Tuple[float, float], 
                          format_type: str = "csv", compress: bool = False) -> str:
    """
    Build the expected filename for a tx position without index and including cell size.
    Format:
      rss_{scene}_cell{cellx}x{celly}_x{X}_y{Y}_z{Z}.{extension}
    coords formatted with 4 decimal places.
    """
    cellx, celly = (cell_size[0], cell_size[1]) if hasattr(cell_size, "__len__") else (cell_size, cell_size)
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    coord_part = f"x{x:.4f}_y{y:.4f}_z{z:.4f}"
    coord_part = sanitize_filename_part(coord_part)
    
    # Determine extension based on format
    if format_type.lower() == "csv":
        ext = "csv.gz" if compress else "csv"
    elif format_type.lower() in ["hdf5", "h5"]:
        ext = "h5"
    elif format_type.lower() == "npz":
        ext = "npz"
    elif format_type.lower() == "parquet":
        ext = "parquet"
    elif format_type.lower() == "zarr":
        ext = "zarr"
    else:
        ext = format_type.lower()
    
    fname = f"rss_{scene}_cell{cellx}x{celly}_{coord_part}.{ext}"
    return fname

def gather_existing_files(out_dir: str, scene: str, cell_size: Tuple[float, float], 
                         format_type: str = "csv", compress: bool = False) -> set:
    """
    Returns a set of filenames already present in out_dir that match the pattern for the scene and cell size.
    This avoids recomputing rss for tx positions that already have files.
    """
    # Build prefix to filter
    cellx, celly = (cell_size[0], cell_size[1]) if hasattr(cell_size, "__len__") else (cell_size, cell_size)
    prefix = f"rss_{scene}_cell{cellx}x{celly}_"
    existing = set()
    if not os.path.isdir(out_dir):
        return existing
    
    # Define valid extensions based on format
    valid_extensions = []
    if format_type.lower() == "csv":
        valid_extensions = [".csv", ".csv.gz"]
    elif format_type.lower() in ["hdf5", "h5"]:
        valid_extensions = [".h5", ".hdf5"]
    elif format_type.lower() == "npz":
        valid_extensions = [".npz"]
    elif format_type.lower() == "parquet":
        valid_extensions = [".parquet"]
    elif format_type.lower() == "zarr":
        valid_extensions = [".zarr"]
    else:
        valid_extensions = [f".{format_type.lower()}"]
    
    for fn in os.listdir(out_dir):
        if fn.startswith(prefix):
            for ext in valid_extensions:
                if fn.endswith(ext):
                    existing.add(fn)
                    break
            # Also check for zarr directories
            if format_type.lower() == "zarr":
                zarr_path = os.path.join(out_dir, fn)
                if os.path.isdir(zarr_path) and fn.endswith('.zarr'):
                    existing.add(fn)
    
    return existing

def main():
    parser = argparse.ArgumentParser(description="Generate RSS CSV files for transmitter positions.")
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene name string, e.g., 'munich' (assumes attribute in sionna.rt.scene)")
    parser.add_argument("--tx_positions_file", type=str, default="",
                        help="Path to CSV file containing tx positions (one per line as x,y,z). Ignored if --N > 0.")
    parser.add_argument("--N", type=int, default=0,
                        help="Number of random transmitter positions to generate. If >0, overrides tx_positions_file.")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Output directory where the RSS CSV files will be stored")
    parser.add_argument("--format", type=str, default="csv", 
                        choices=['csv', 'npz', 'hdf5', 'h5', 'parquet', 'zarr'],
                        help="Output format (default: csv). Options: csv, npz, hdf5, parquet, zarr")
    parser.add_argument("--compression_level", type=int, default=6, choices=range(0, 10),
                        help="Compression level 0-9 (default: 6, higher = better compression)")
    parser.add_argument("--compress", action="store_true",
                        help="Compress output files using gzip (only for CSV format)")
    parser.add_argument("--no_metadata", action="store_true",
                        help="Don't include metadata in output files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run storage format benchmark and exit")
    args = parser.parse_args()

    # Handle benchmark mode
    if args.benchmark:
        from .storage_utils import benchmark_formats, get_format_info
        
        print("Running storage format benchmark...")
        print("Creating sample RSS data...")
        
        # Create sample data for benchmarking
        sample_rss = np.random.uniform(-120, -30, size=(200, 300)).astype(np.float32)
        sample_metadata = {
            'tx_position': [10.0, 5.0, 3.0],
            'scene_name': 'benchmark_test',
            'max_depth': 30,
            'cell_size': [2.0, 2.0, 2.0]
        }
        
        results = benchmark_formats(sample_rss, sample_metadata)
        
        print("\n" + "="*80)
        print("STORAGE FORMAT BENCHMARK RESULTS")
        print("="*80)
        print(f"{'Format':<12} {'Size (KB)':<12} {'Compression':<12} {'Save (ms)':<12} {'Load (ms)':<12} {'Status':<15}")
        print("-"*80)
        
        for fmt, result in results.items():
            if 'error' in result:
                print(f"{fmt:<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'Error':<15}")
                print(f"   Error: {result['error']}")
            else:
                size_kb = result['file_size'] / 1024
                compression = f"{result['compression_ratio']:.1f}x"
                save_ms = f"{result['save_time']*1000:.1f}"
                load_ms = f"{result['load_time']*1000:.1f}"
                status = "✓ OK" if result['data_integrity'] else "✗ FAIL"
                
                print(f"{fmt:<12} {size_kb:<12.1f} {compression:<12} {save_ms:<12} {load_ms:<12} {status:<15}")
        
        print("\nFormat Information:")
        formats_info = get_format_info()
        for fmt, info in formats_info.items():
            if info['available']:
                print(f"  {fmt}: {info['recommended_for']}")
            else:
                print(f"  {fmt}: Not available (install required dependencies)")
        
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # Load the scene attribute
    try:
        scene_attr = getattr(rt_scene, args.scene)
    except AttributeError:
        raise ValueError(f"Scene '{args.scene}' not found in sionna.rt.scene.")

    scene_obj = load_scene(scene_attr, merge_shapes=True)

    # Determine tx_positions
    if args.N > 0:
        x_min, x_max, y_min, y_max, z_min, z_max = get_scene_bounds3d(scene_obj)
        low = np.array([x_min, y_min, z_min])
        high = np.array([x_max, y_max, z_max])
        tx_positions = np.random.uniform(low, high, size=(args.N, 3))
        print(f"Generated {args.N} random tx positions using scene bounds:")
        print(f"  x: [{x_min}, {x_max}], y: [{y_min}, {y_max}], z: [{z_min}, {z_max}]")
    else:
        if not args.tx_positions_file:
            raise ValueError("Either provide --tx_positions_file or set --N > 0 to generate random positions.")
        tx_positions = load_tx_positions_from_file(args.tx_positions_file)
        print(f"Loaded {tx_positions.shape[0]} tx positions from {args.tx_positions_file}")

    # Cell size used for RSS generation and included in filenames
    cell_size_for_name = CELL_SIZE if hasattr(CELL_SIZE, "__len__") else (CELL_SIZE, CELL_SIZE)

    # Scan output directory for existing files matching scene and cell size
    existing_files = gather_existing_files(args.out_dir, args.scene, cell_size_for_name, args.format, args.compress)
    print(f"Found {len(existing_files)} existing RSS files in '{args.out_dir}' for scene '{args.scene}' and cell {cell_size_for_name}.")

    # Build list of positions to generate (skip ones whose expected filename already exists unless overwrite)
    tasks = []
    for pos in tx_positions:
        expected_name = make_expected_filename(args.scene, pos, cell_size_for_name, args.format, args.compress)
        expected_path = os.path.join(args.out_dir, expected_name)
        if os.path.exists(expected_path) and not args.overwrite:
            # already present, skip
            continue
        # For zarr format, also check if directory exists
        if args.format.lower() == "zarr" and os.path.isdir(expected_path) and not args.overwrite:
            continue
        tasks.append((pos, expected_path))

    to_generate = len(tasks)
    total_positions = tx_positions.shape[0]
    print(f"{to_generate}/{total_positions} positions will be (re)generated (overwrite={args.overwrite}).")

    successes = 0
    failures = 0

    for idx, (pos, expected_path) in enumerate(tasks, start=1):
        try:
            # Use efficient storage format for non-CSV formats
            if args.format.lower() != "csv":
                # Remove extension from expected path for the efficient function
                base_path = os.path.splitext(expected_path)[0]
                if expected_path.endswith('.zarr'):
                    base_path = expected_path[:-5]  # Remove .zarr
                
                rss_array, out_file = rss_write_efficient(
                    scene_obj, tx_position=pos,
                    max_depth=MAX_DEPTH, cell_size=CELL_SIZE,
                    samples_per_tx=SAMPLES_PER_TX, 
                    output_file=base_path,
                    format_type=args.format,
                    compression_level=args.compression_level,
                    include_metadata=not args.no_metadata
                )
            else:
                # Use legacy CSV function
                rss_array, out_file = rss_write_csv(
                    scene_obj, tx_position=pos,
                    max_depth=MAX_DEPTH, cell_size=CELL_SIZE,
                    samples_per_tx=SAMPLES_PER_TX, 
                    csv_file=expected_path,
                    compress=args.compress
                )
            
            print(f"[{idx}/{to_generate}] RSS for tx {pos.tolist()} saved to {out_file} (shape {rss_array.shape})")
            successes += 1
        except Exception as e:
            failures += 1
            print(f"[{idx}/{to_generate}] ERROR generating RSS for tx {pos.tolist()}: {e}")
            traceback.print_exc()

    # Final summary
    print("\n==== SUMMARY ====")
    print(f"Total TX positions in master list: {total_positions}")
    print(f"Positions skipped because files already exist: {total_positions - to_generate}")
    print(f"Positions generated this run: {to_generate}")
    print(f"Successful writes: {successes}")
    print(f"Failures: {failures}")
    print(f"Output directory: {os.path.abspath(args.out_dir)}")
    print("=================\n")

if __name__ == '__main__':
    main()
