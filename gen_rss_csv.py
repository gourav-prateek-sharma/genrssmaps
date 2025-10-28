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
from typing import Tuple

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

def make_expected_filename(scene: str, pos: np.ndarray, cell_size: Tuple[float, float], compress: bool) -> str:
    """
    Build the expected filename for a tx position without index and including cell size.
    Format:
      rss_{scene}_cell{cellx}x{celly}_x{X}_y{Y}_z{Z}.csv[.gz]
    coords formatted with 4 decimal places.
    """
    cellx, celly = (cell_size[0], cell_size[1]) if hasattr(cell_size, "__len__") else (cell_size, cell_size)
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    coord_part = f"x{x:.4f}_y{y:.4f}_z{z:.4f}"
    coord_part = sanitize_filename_part(coord_part)
    fname = f"rss_{scene}_cell{cellx}x{celly}_{coord_part}.csv"
    if compress:
        fname = fname + ".gz"
    return fname

def gather_existing_files(out_dir: str, scene: str, cell_size: Tuple[float, float], compress: bool) -> set:
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
    for fn in os.listdir(out_dir):
        if fn.startswith(prefix) and (fn.endswith(".csv") or fn.endswith(".csv.gz")):
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
    parser.add_argument("--compress", action="store_true",
                        help="Compress output files using gzip")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

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
    existing_files = gather_existing_files(args.out_dir, args.scene, cell_size_for_name, args.compress)
    print(f"Found {len(existing_files)} existing RSS files in '{args.out_dir}' for scene '{args.scene}' and cell {cell_size_for_name}.")

    # Build list of positions to generate (skip ones whose expected filename already exists unless overwrite)
    tasks = []
    for pos in tx_positions:
        expected_name = make_expected_filename(args.scene, pos, cell_size_for_name, args.compress)
        expected_path = os.path.join(args.out_dir, expected_name)
        if os.path.exists(expected_path) and not args.overwrite:
            # already present, skip
            continue
        # else add to tasks (we store both pos and expected path)
        # if not compress but .gz file exists, consider it existing too
        if not args.compress:
            gz_path = expected_path + ".gz"
            if os.path.exists(gz_path) and not args.overwrite:
                continue
        tasks.append((pos, expected_path))

    to_generate = len(tasks)
    total_positions = tx_positions.shape[0]
    print(f"{to_generate}/{total_positions} positions will be (re)generated (overwrite={args.overwrite}).")

    successes = 0
    failures = 0

    for idx, (pos, expected_path) in enumerate(tasks, start=1):
        csv_path = expected_path
        # If compress True but expected_path doesn't end with .gz, ensure rss_write_csv will add .gz.
        # But we provide expected_path as the "csv_file" argument (rss_write_csv will append .gz internally).
        try:
            rss_array, out_file = rss_write_csv(scene_obj, tx_position=pos,
                                                max_depth=MAX_DEPTH, cell_size=CELL_SIZE,
                                                samples_per_tx=SAMPLES_PER_TX, csv_file=csv_path,
                                                compress=args.compress)
            print(f"[{idx}/{to_generate}] RSS for tx {pos.tolist()} saved to {out_file}  (shape {rss_array.shape})")
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
