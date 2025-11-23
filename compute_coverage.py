#!/usr/bin/env python3
"""
Generic CLI for computing coverage from RSS files in multiple formats.

Supports: .csv, .csv.gz, .npz, .h5/.hdf5, .parquet, .zarr

Usage examples:
  # Single file (auto-detect format by extension)
  python -m genrssmaps.compute_coverage --file /path/to/rss_munich_1.0,2.0,20.0.csv.gz --scene munich

  # Directory -> output CSV summarizing filename,x,y,z,coverage
  python -m genrssmaps.compute_coverage --dir /path/to/dir --output coverage_summary.csv --scene munich

This module delegates to functions in `coverage_helpers` for the actual coverage
calculation when possible (e.g., `compute_coverage_from_csv_gz` or
`compute_coverage_from_arr`).
"""

from __future__ import annotations

import argparse
import os
import sys
import re
from typing import Optional, Tuple, Any

import numpy as np

try:
    from .coverage_helpers import (
        compute_coverage_from_csv_gz,
        compute_coverage_from_arr,
        THRESHOLD,
    )
except Exception as e:
    print(f"Error importing coverage helpers: {e}")
    raise


def load_rx_indices(path: str):
    import numpy as _np

    try:
        arr = _np.loadtxt(path, delimiter=',')
    except Exception:
        arr = _np.loadtxt(path)
    arr = _np.atleast_2d(arr[:, :2]).astype(int)
    return arr


def _parse_tx_coords_from_filename(fname: str, scene_name: str) -> Optional[Tuple[float, float, float]]:
    """Try to parse tx coords from filenames like
    `rss_<scene>_<x>,<y>,<z>.<ext>` where ext may be `csv.gz`, `h5`, `npz`, etc.
    Returns (x,y,z) or None.
    """
    # Accept extensions like .csv.gz, .csv, .h5, .npz, .parquet, .zarr
    pattern = rf"rss_{re.escape(scene_name)}_([\-\d.]+),([\-\d.]+),([\-\d.]+)\.(?:csv\.gz|csv|h5|hdf5|npz|parquet|zarr)$"
    m = re.match(pattern, os.path.basename(fname))
    if not m:
        return None
    return tuple(map(float, m.groups()))


def _load_array_from_npz(path: str) -> np.ndarray:
    npz = np.load(path, allow_pickle=False)
    # If single array saved unnamed, numpy returns an NpzFile mapping.
    # Prefer array under key 'rss' or take first array available.
    keys = list(npz.keys())
    if not keys:
        raise ValueError(f"No arrays found in npz file: {path}")
    key = None
    if 'rss' in keys:
        key = 'rss'
    else:
        key = keys[0]
    arr = npz[key]
    return arr


def _load_array_from_h5(path: str) -> np.ndarray:
    try:
        import h5py
    except Exception as e:
        raise ImportError("h5py is required to read .h5 files. Install via 'pip install h5py'") from e
    with h5py.File(path, 'r') as f:
        # prefer dataset named 'rss' or take first dataset
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"No datasets found in h5 file: {path}")
        key = 'rss' if 'rss' in keys else keys[0]
        data = f[key][()]
    return np.array(data)


def _load_array_from_parquet(path: str) -> np.ndarray:
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError("pandas (with pyarrow) is required to read parquet files. Install via 'pip install pandas pyarrow'") from e
    df = pd.read_parquet(path)
    return df.values


def _load_array_from_zarr(path: str) -> np.ndarray:
    try:
        import zarr
    except Exception as e:
        raise ImportError("zarr is required to read .zarr files. Install via 'pip install zarr'") from e
    z = zarr.open(path, mode='r')
    # zarr groups may contain multiple arrays; prefer 'rss' else first array
    if hasattr(z, 'keys'):
        keys = list(z.keys())
        if 'rss' in keys:
            return np.array(z['rss'])
        else:
            return np.array(z[keys[0]])
    # if z is an array
    return np.array(z)


def _load_array_from_file(path: str, forced_format: Optional[str] = None) -> np.ndarray:
    ext = forced_format or os.path.splitext(path)[1].lower()
    # Handle .gz special-case
    if path.endswith('.csv.gz') or ext in ('.csv', '.gz') or path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(path, header=None)
        return df.values
    if ext == '.npz' or path.endswith('.npz'):
        return _load_array_from_npz(path)
    if ext in ('.h5', '.hdf5') or path.endswith('.h5') or path.endswith('.hdf5'):
        return _load_array_from_h5(path)
    if ext == '.parquet' or path.endswith('.parquet'):
        return _load_array_from_parquet(path)
    if ext == '.zarr' or path.endswith('.zarr'):
        return _load_array_from_zarr(path)
    # Fallback: try pandas read_csv
    try:
        import pandas as pd
        df = pd.read_csv(path, header=None)
        return df.values
    except Exception as e:
        raise ValueError(f"Unsupported or unreadable file format for path '{path}': {e}") from e


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description="Compute coverage from RSS files (multiple formats)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single RSS file (.csv/.csv.gz/.npz/.h5/.parquet/.zarr)")
    group.add_argument("--dir", type=str, help="Directory containing RSS files to process (filename should encode tx coords if you want x,y,z output)")
    parser.add_argument("--output", type=str, default="coverage_summary.csv",
                        help="Output CSV path when running on a directory. Default: coverage_summary.csv")
    parser.add_argument("--format", type=str, default=None,
                        help="Force input format (csv,npz,h5,parquet,zarr). By default auto-detect from file extension")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Coverage threshold in dBm (default: package THRESHOLD)")
    parser.add_argument("--scene", type=str, default="munich",
                        help="Scene name used to parse filenames (default: munich)")
    parser.add_argument("--rx_indices_file", type=str, default=None,
                        help="Optional CSV file with receiver grid indices (r,c) to restrict coverage computation")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args(argv)

    threshold = args.threshold if args.threshold is not None else THRESHOLD

    rx_indices = None
    if args.rx_indices_file:
        try:
            rx_indices = load_rx_indices(args.rx_indices_file)
        except Exception as e:
            print(f"Failed to read rx indices file '{args.rx_indices_file}': {e}")
            sys.exit(2)

    if args.file:
        file_path = args.file
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            sys.exit(2)

        # If CSV.GZ and filename encodes coords, reuse specialized function to extract tx coords
        if file_path.endswith('.csv.gz'):
            try:
                tx_x, tx_y, tx_z, coverage = compute_coverage_from_csv_gz(
                    file_path, rx_grid_indices=rx_indices, threshold_dbm=threshold, scene_name=args.scene
                )
                print(f"{os.path.basename(file_path)} -> tx=({tx_x:.6f},{tx_y:.6f},{tx_z:.6f}), coverage={coverage:.6f}")
                return
            except Exception:
                # Fall back to generic loader if specialized fails
                pass

        # Generic loading path
        try:
            arr = _load_array_from_file(file_path, forced_format=(args.format and args.format.lower()))
        except Exception as e:
            print(f"Error loading file '{file_path}': {e}")
            raise

        # compute coverage
        coverage = compute_coverage_from_arr(arr, threshold_dbm=threshold)
        coords = _parse_tx_coords_from_filename(file_path, args.scene)
        if coords is not None:
            tx_x, tx_y, tx_z = coords
            print(f"{os.path.basename(file_path)} -> tx=({tx_x:.6f},{tx_y:.6f},{tx_z:.6f}), coverage={coverage:.6f}")
        else:
            print(f"{os.path.basename(file_path)} -> coverage={coverage:.6f}")

    else:
        # directory mode: gather supported files and iterate
        dir_path = args.dir
        if not os.path.isdir(dir_path):
            print(f"Directory not found: {dir_path}")
            sys.exit(2)

        supported_exts = ['.csv', '.csv.gz', '.npz', '.h5', '.hdf5', '.parquet', '.zarr']
        results = []
        # list files and filter
        for fname in sorted(os.listdir(dir_path)):
            fpath = os.path.join(dir_path, fname)
            if not os.path.isfile(fpath):
                continue
            lower = fname.lower()
            if any(lower.endswith(ext) for ext in supported_exts):
                try:
                    if lower.endswith('.csv.gz'):
                        # reuse specialized function when possible to get coords
                        try:
                            tx_x, tx_y, tx_z, coverage = compute_coverage_from_csv_gz(fpath, rx_grid_indices=rx_indices, threshold_dbm=threshold, scene_name=args.scene)
                            results.append((fname, tx_x, tx_y, tx_z, coverage))
                            if args.verbose:
                                print(f"Processed {fname} -> coverage={coverage:.6f}")
                            continue
                        except Exception:
                            # fallback to generic loader
                            pass

                    arr = _load_array_from_file(fpath)
                    coverage = compute_coverage_from_arr(arr, threshold_dbm=threshold)
                    coords = _parse_tx_coords_from_filename(fname, args.scene)
                    if coords is not None:
                        tx_x, tx_y, tx_z = coords
                    else:
                        tx_x = tx_y = tx_z = float('nan')
                    results.append((fname, tx_x, tx_y, tx_z, coverage))
                    if args.verbose:
                        print(f"Processed {fname} -> coverage={coverage:.6f}")
                except Exception as e:
                    print(f"Skipping {fname}: {e}")

        # write output CSV
        import csv
        out_csv = args.output
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'x', 'y', 'z', 'coverage'])
            for row in results:
                writer.writerow(row)
        print(f"Wrote coverage data for {len(results)} files to {out_csv}")


if __name__ == '__main__':
    main()
