#!/usr/bin/env python3
"""
generate_subgrids.py

Generate all coarse subgrids (downsampled versions) from a fine-resolution XY plane
by enumerating every possible phase (offset) of the coarse sampling grid.

Outputs:
 - master CSV with rows (x,y,z,ox,oy) for every sampled point across all phases
 - optionally one CSV per phase in out_dir/subgrids/
 - JSON manifest with metadata for each phase

Usage example:
 python generate_subgrids.py --scene munich --cell_size 1.0 --step_size 4.0 \
    --z_height 20 --out_dir rm_data --per_phase_csv --master_csv
"""

import os
import argparse
import json
import numpy as np
from scene_helpers import get_scene_bounds3d, get_sionna_scene

EPS = 1e-9  # tolerance for floating comparisons (meters)


# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------
def compute_fine_coords(x_min, x_max, y_min, y_max, cell_size_x, cell_size_y):
    """Compute fine-grid centers along X and Y."""
    nx = int(np.floor((x_max - x_min) / cell_size_x))
    ny = int(np.floor((y_max - y_min) / cell_size_y))
    if nx <= 0 or ny <= 0:
        return np.array([]), np.array([])
    x_coords = x_min + (np.arange(nx) + 0.5) * cell_size_x
    y_coords = y_min + (np.arange(ny) + 0.5) * cell_size_y
    return x_coords, y_coords


def compute_downsample_factors(step_size_x, step_size_y, cell_size_x, cell_size_y, force_round=False):
    """Compute integer downsample factors kx, ky from step_size / cell_size."""
    rx = step_size_x / cell_size_x
    ry = step_size_y / cell_size_y
    kx = int(np.round(rx))
    ky = int(np.round(ry))

    def check_and_report(r, k, axis):
        diff = abs(r - k)
        if diff > EPS:
            if not force_round:
                raise ValueError(
                    f"On axis {axis}: step_size/cell_size = {r:.6f} not integer. "
                    f"Either choose step_size that is integer multiple of cell_size or enable --force_round."
                )
            else:
                print(f"Warning: On axis {axis} step_size/cell_size = {r:.6f} not integer; rounding to {k}.")

    check_and_report(rx, kx, "X")
    check_and_report(ry, ky, "Y")

    eff_step_x = kx * cell_size_x
    eff_step_y = ky * cell_size_y
    if kx < 1 or ky < 1:
        raise ValueError("Computed downsample factor < 1. Check step_size and cell_size values.")

    return kx, ky, eff_step_x, eff_step_y


def enumerate_phases(nx, ny, kx, ky):
    """Return list of phases, each with ix (x indices) and jy (y indices)."""
    phases = []
    for ox in range(kx):
        ix = np.arange(ox, nx, kx)
        for oy in range(ky):
            jy = np.arange(oy, ny, ky)
            phases.append({"ox": int(ox), "oy": int(oy), "ix": ix, "jy": jy})
    return phases


def materialize_positions_for_phase(x_coords, y_coords, z_height, phase):
    """Return Nx3 array of (x,y,z) for given phase."""
    ix = phase["ix"]
    jy = phase["jy"]
    if ix.size == 0 or jy.size == 0:
        return np.zeros((0, 3))
    Xidx, Yidx = np.meshgrid(ix, jy, indexing="xy")
    xs = x_coords[Xidx.ravel()]
    ys = y_coords[Yidx.ravel()]
    zs = np.full(xs.shape, z_height)
    return np.column_stack((xs, ys, zs))


def save_master_csv(path, records):
    """Save master CSV (x,y,z,ox,oy)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = "x,y,z,ox,oy"
    np.savetxt(path, records, delimiter=",", header=header, comments="", fmt="%.6f,%.6f,%.6f,%d,%d")
    print(f"Saved master CSV: {path} ({len(records)} rows)")


def save_phase_csv(out_dir, scene, phase, positions):
    """Save per-phase CSV (x,y,z)."""
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{scene}_tile_ox{phase['ox']}_oy{phase['oy']}.csv")
    header = "x,y,z"
    if positions.shape[0] == 0:
        with open(filename, "w") as f:
            f.write(header + "\n")
    else:
        np.savetxt(filename, positions, delimiter=",", header=header, comments="", fmt="%.6f,%.6f,%.6f")
    return filename


# -----------------------------------------------------------
# Main logic
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate all subgrid phases from a fine XY grid.")
    parser.add_argument("--scene", type=str, required=True, help="Scene name in sionna.rt.scene")
    parser.add_argument("--cell_size", type=float, default=1.0, help="Fine cell size (meters)")
    parser.add_argument("--step_size", type=float, required=True, help="Coarse step size (meters)")
    parser.add_argument("--z_height", type=float, default=20.0, help="Z coordinate (constant for all points)")
    parser.add_argument("--force_round", action="store_true", help="Round step_size/cell_size to integer factor")
    parser.add_argument("--out_dir", type=str, default="rm_data", help="Output directory")
    parser.add_argument("--per_phase_csv", action="store_true", help="Write one CSV per phase")
    parser.add_argument("--master_csv", action="store_true", help="Write one master CSV with all points")
    args = parser.parse_args()

    # Load scene and bounds
    scene_obj = get_sionna_scene(getattr(__import__('sionna.rt.scene', fromlist=[args.scene]), args.scene))
    x_min, x_max, y_min, y_max, z_min, z_max = get_scene_bounds3d(scene_obj)

    print(f"Scene '{args.scene}' bounds:")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}], Y: [{y_min:.2f}, {y_max:.2f}]")

    # Fine grid
    x_coords, y_coords = compute_fine_coords(x_min, x_max, y_min, y_max, args.cell_size, args.cell_size)
    nx, ny = len(x_coords), len(y_coords)
    print(f"Fine grid resolution: {args.cell_size} m  →  nx={nx}, ny={ny}, total fine cells={nx*ny}")

    # Downsample
    kx, ky, eff_step_x, eff_step_y = compute_downsample_factors(
        args.step_size, args.step_size, args.cell_size, args.cell_size, force_round=args.force_round
    )
    print(f"Downsample factors: kx={kx}, ky={ky}  →  effective step=({eff_step_x},{eff_step_y})")
    print(f"Total number of phases: {kx * ky}")

    # Enumerate all phases
    phases = enumerate_phases(nx, ny, kx, ky)

    # Prepare outputs
    total_points = 0
    master_records = []
    subgrids_dir = os.path.join(args.out_dir, "subgrids")

    for idx, phase in enumerate(phases):
        ox, oy = phase["ox"], phase["oy"]
        positions = materialize_positions_for_phase(x_coords, y_coords, args.z_height, phase)
        npoints = positions.shape[0]
        total_points += npoints
        print(f"Phase {idx+1:2d}/{len(phases):2d}: ox={ox}, oy={oy}  →  {npoints} points")

        if args.per_phase_csv:
            save_phase_csv(subgrids_dir, args.scene, phase, positions)

        if args.master_csv and npoints > 0:
            labels = np.column_stack(
                (positions, np.full((npoints, 1), ox, dtype=int), np.full((npoints, 1), oy, dtype=int))
            )
            master_records.append(labels)

    # Save master CSV
    if args.master_csv and master_records:
        all_master = np.vstack(master_records)
        master_path = os.path.join(
            args.out_dir, f"master_{args.scene}_cell{args.cell_size}_step{args.step_size}.csv"
        )
        save_master_csv(master_path, all_master)

    # Final summary
    print("\n===== SUMMARY =====")
    print(f"Scene: {args.scene}")
    print(f"Fine cell size: {args.cell_size} m")
    print(f"Coarse step size: {args.step_size} m")
    print(f"Downsample factors: kx={kx}, ky={ky}")
    print(f"Total phases: {len(phases)}")
    print(f"Total (x,y,z) points generated across all phases: {total_points}")
    print(f"Output directory: {args.out_dir}")
    print("====================\n")


if __name__ == "__main__":
    main()
