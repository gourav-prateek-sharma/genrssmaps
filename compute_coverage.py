#!/usr/bin/env python3
"""
Generic CLI for computing coverage from RSS files in multiple formats.

Supports: .csv, .csv.gz, .npz, .h5/.hdf5, .parquet, .zarr

Usage examples:
  # Single file (auto-detect format by extension)
  python -m genrssmaps.compute_coverage --file /path/to/rss_munich_1.0,2.0,20.0.csv.gz --scene munich

  # 
  # ory -> output CSV summarizing filename,x,y,z,coverage
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
    parser.add_argument("--threshold", type=float, nargs='+', default=None,
                        help="Coverage threshold(s) in dBm (space-separated if multiple). Default: package THRESHOLD")
    parser.add_argument("--scene", type=str, default="munich",
                        help="Scene name used to parse filenames (default: munich)")
    parser.add_argument("--rx_indices_file", type=str, default=None,
                        help="Optional CSV file with receiver grid indices (r,c) to restrict coverage computation")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args(argv)

    # Handle multiple thresholds
    if args.threshold is not None:
        thresholds = list(args.threshold)
    else:
        thresholds = [THRESHOLD]

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

        # Load the array once
        if file_path.endswith('.csv.gz'):
            try:
                tx_x, tx_y, tx_z, _ = compute_coverage_from_csv_gz(
                    file_path, rx_grid_indices=rx_indices, threshold_dbm=thresholds[0], scene_name=args.scene
                )
                coords_found = True
            except Exception:
                coords_found = False
                arr = _load_array_from_file(file_path, forced_format=(args.format and args.format.lower()))
                coords = _parse_tx_coords_from_filename(file_path, args.scene)
                if coords is not None:
                    tx_x, tx_y, tx_z = coords
                    coords_found = True
        else:
            coords_found = False
            arr = _load_array_from_file(file_path, forced_format=(args.format and args.format.lower()))
            coords = _parse_tx_coords_from_filename(file_path, args.scene)
            if coords is not None:
                tx_x, tx_y, tx_z = coords
                coords_found = True

        # Compute coverage for each threshold
        if len(thresholds) == 1:
            # Single threshold: print simple format
            if coords_found and 'arr' not in locals():
                arr = _load_array_from_file(file_path, forced_format=(args.format and args.format.lower()))
            coverage = compute_coverage_from_arr(arr, threshold_dbm=thresholds[0])
            if coords_found:
                print(f"{os.path.basename(file_path)} -> tx=({tx_x:.6f},{tx_y:.6f},{tx_z:.6f}), coverage={coverage:.6f}")
            else:
                print(f"{os.path.basename(file_path)} -> coverage={coverage:.6f}")
        else:
            # Multiple thresholds: print all
            if 'arr' not in locals():
                arr = _load_array_from_file(file_path, forced_format=(args.format and args.format.lower()))
            if coords_found:
                output = f"{os.path.basename(file_path)} -> tx=({tx_x:.6f},{tx_y:.6f},{tx_z:.6f})"
            else:
                output = f"{os.path.basename(file_path)}"
            for thr in thresholds:
                coverage = compute_coverage_from_arr(arr, threshold_dbm=thr)
                output += f", coverage@{thr}dBm={coverage:.6f}"
            print(output)

    else:
        # directory mode: gather supported files and iterate
        dir_path = args.dir
        if not os.path.isdir(dir_path):
            print(f"Directory not found: {dir_path}")
            sys.exit(2)

        supported_exts = ['.csv', '.csv.gz', '.npz', '.h5', '.hdf5', '.parquet', '.zarr']

        # Compute output CSV path including threshold tokens
        out_csv = args.output
        try:
            # For multiple thresholds, include all in filename; for single, include just that one
            if len(thresholds) == 1:
                thr_token = str(thresholds[0]).replace('.', 'p')
                thr_tokens_str = thr_token
            else:
                thr_tokens_str = "_".join([str(t).replace('.', 'p') for t in thresholds])
        except Exception:
            thr_tokens_str = None

        # If user used the default output name, prefer coverage_summary_{threshold(s)}.csv
        default_output_name = 'coverage_summary.csv'
        if args.output == default_output_name and thr_tokens_str is not None:
            out_dir = os.path.dirname(out_csv)
            out_csv_final = os.path.join(out_dir, f"coverage_summary_{thr_tokens_str}.csv") if out_dir else f"coverage_summary_{thr_tokens_str}.csv"
        else:
            if thr_tokens_str:
                out_dir = os.path.dirname(out_csv)
                out_base = os.path.basename(out_csv)
                name, ext = os.path.splitext(out_base)
                if thr_tokens_str not in name:
                    name = f"{name}_{thr_tokens_str}"
                out_csv_final = os.path.join(out_dir, name + ext) if out_dir else name + ext
            else:
                out_csv_final = out_csv

        # If the summary already exists, read already-processed filenames and resume
        import csv
        existing_filenames = set()
        if os.path.exists(out_csv_final):
            try:
                with open(out_csv_final, 'r', newline='') as ef:
                    reader = csv.reader(ef)
                    header = next(reader, None)
                    fname_idx = 0
                    if header and 'filename' in header:
                        fname_idx = header.index('filename')
                    for row in reader:
                        if row:
                            existing_filenames.add(row[fname_idx])
                if args.verbose:
                    print(f"Resuming: found {len(existing_filenames)} entries in {out_csv_final}")
            except Exception as e:
                print(f"Warning: could not read existing output file '{out_csv_final}': {e}")

        # Open output file for append (or create) and write header if needed
        mode = 'a' if os.path.exists(out_csv_final) else 'w'
        with open(out_csv_final, mode, newline='') as out_f:
            writer = csv.writer(out_f)
            if mode == 'w':
                # Build header based on number of thresholds
                header = ['filename', 'x', 'y', 'z']
                for thr in thresholds:
                    header.append(f"coverage_thr{thr}")
                writer.writerow(header)

            processed = 0
            # iterate files in deterministic order and append results as they complete
            for fname in sorted(os.listdir(dir_path)):
                if fname in existing_filenames:
                    # skip already-processed
                    continue
                fpath = os.path.join(dir_path, fname)
                if not os.path.isfile(fpath):
                    continue
                lower = fname.lower()
                if not any(lower.endswith(ext) for ext in supported_exts):
                    continue
                try:
                    # Load array once
                    arr = None
                    tx_x = tx_y = tx_z = ""
                    
                    if lower.endswith('.csv.gz'):
                        try:
                            tx_x, tx_y, tx_z, _ = compute_coverage_from_csv_gz(fpath, rx_grid_indices=rx_indices, threshold_dbm=thresholds[0], scene_name=args.scene)
                        except Exception:
                            coords = _parse_tx_coords_from_filename(fname, args.scene)
                            if coords is not None:
                                tx_x, tx_y, tx_z = coords
                    
                    if arr is None:
                        arr = _load_array_from_file(fpath)
                        if tx_x == "":  # coords not yet set
                            coords = _parse_tx_coords_from_filename(fname, args.scene)
                            if coords is not None:
                                tx_x, tx_y, tx_z = coords
                    
                    # Compute coverage for each threshold
                    row = [fname, tx_x, tx_y, tx_z]
                    for thr in thresholds:
                        coverage = compute_coverage_from_arr(arr, threshold_dbm=thr)
                        row.append(coverage)
                    
                    writer.writerow(row)
                    out_f.flush()
                    try:
                        os.fsync(out_f.fileno())
                    except Exception:
                        pass
                    existing_filenames.add(fname)
                    processed += 1
                    if args.verbose:
                        print(f"Processed {fname}")
                except Exception as e:
                    print(f"Skipping {fname}: {e}")

        print(f"Wrote/updated coverage data; total entries now: {len(existing_filenames)} -> {out_csv_final}")


if __name__ == '__main__':
    main()
