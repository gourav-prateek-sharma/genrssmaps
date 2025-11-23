#!/usr/bin/env python3
"""
CLI entrypoint for computing coverage from rss CSV.GZ files.

Usage examples:
  # Single file
  python -m genrssmaps.compute_coverage_from_csv_gz --file /path/to/rss_munich_x,y,z.csv.gz --scene munich

  # Directory -> summary CSV
  python -m genrssmaps.compute_coverage_from_csv_gz --dir /path/to/dir --output coverage_summary.csv --scene munich

Options:
  --threshold: coverage threshold in dBm (default uses package THRESHOLD)
  --rx_indices_file: optional CSV file containing rows of receiver grid indices (r,c) to restrict coverage computation
"""

import argparse
import os
import sys
from typing import Optional

try:
    from .coverage_helpers import (
        compute_coverage_from_csv_gz,
        compute_coverage_for_directory_to_csv,
        THRESHOLD,
    )
except Exception as e:
    print(f"Error importing coverage helpers: {e}")
    raise


def load_rx_indices(path: str):
    """Load rx grid indices from a CSV or whitespace file into an (N,2) int list."""
    import numpy as np

    try:
        arr = np.loadtxt(path, delimiter=',')
    except Exception:
        arr = np.loadtxt(path)
    arr = np.atleast_2d(arr[:, :2]).astype(int)
    return arr


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description="Compute coverage from rss CSV.GZ files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single .csv.gz RSS file (name must encode tx coordinates)")
    group.add_argument("--dir", type=str, help="Directory containing rss_<scene>_*.csv.gz files to process")
    parser.add_argument("--output", type=str, default="coverage_summary.csv",
                        help="Output CSV path when running on a directory. Default: coverage_summary.csv")
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
        try:
            tx_x, tx_y, tx_z, coverage = compute_coverage_from_csv_gz(
                file_path, rx_grid_indices=rx_indices, threshold_dbm=threshold, scene_name=args.scene
            )
            print(f"{os.path.basename(file_path)} -> tx=({tx_x:.6f},{tx_y:.6f},{tx_z:.6f}), coverage={coverage:.6f}")
        except Exception as e:
            print(f"Error computing coverage for '{file_path}': {e}")
            raise

    else:
        # directory mode
        dir_path = args.dir
        if not os.path.isdir(dir_path):
            print(f"Directory not found: {dir_path}")
            sys.exit(2)
        out_csv = args.output
        try:
            compute_coverage_for_directory_to_csv(dir_path, out_csv, threshold_dbm=threshold, scene_name=args.scene, rx_grid_indices=rx_indices)
            if args.verbose:
                print(f"Wrote coverage summary to: {out_csv}")
        except Exception as e:
            print(f"Error processing directory '{dir_path}': {e}")
            raise


if __name__ == '__main__':
    main()
