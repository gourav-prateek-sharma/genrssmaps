#!/usr/bin/env python3
"""
CLI wrapper to compute coverage from HDF5 (.h5/.hdf5) RSS files using
the `compute_coverage_from_h5` helper in `coverage_helpers`.

Usage:
  python -m genrssmaps.compute_coverage_from_h5 /path/to/rss_file.h5

This module is intentionally small: it loads optional receiver index files
and forwards options to the helper function.
"""
from __future__ import annotations

import argparse
import sys
import numpy as np

from .coverage_helpers import compute_coverage_from_h5, THRESHOLD


def load_rx_indices(path: str):
    try:
        arr = np.loadtxt(path, delimiter=',')
    except Exception:
        arr = np.loadtxt(path)
    arr = np.atleast_2d(arr[:, :2]).astype(int)
    return arr


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute coverage from a .h5 RSS file")
    parser.add_argument("file", type=str, help="Path to .h5/.hdf5 file containing RSS data")
    parser.add_argument("--rx_indices_file", type=str, default=None,
                        help="Optional CSV file with receiver grid indices (r,c) to restrict coverage computation")
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"Coverage threshold in dBm (default: {THRESHOLD})")
    parser.add_argument("--scene", type=str, default="munich",
                        help="Scene name used to parse filenames if needed (default: munich)")

    args = parser.parse_args(argv)

    rx_indices = None
    if args.rx_indices_file:
        try:
            rx_indices = load_rx_indices(args.rx_indices_file)
        except Exception as e:
            print(f"Failed to read rx indices file '{args.rx_indices_file}': {e}")
            return 2

    threshold = args.threshold if args.threshold is not None else THRESHOLD

    try:
        tx_x, tx_y, tx_z, coverage = compute_coverage_from_h5(
            args.file, rx_grid_indices=rx_indices, threshold_dbm=threshold, scene_name=args.scene
        )
    except Exception as e:
        print(f"Error computing coverage for '{args.file}': {e}")
        return 1

    print(f"{args.file} -> tx=({tx_x:.6f},{tx_y:.6f},{tx_z:.6f}), coverage={coverage:.6f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
