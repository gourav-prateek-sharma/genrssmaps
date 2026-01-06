import numpy as np
import pandas as pd
# TensorFlow is an optional heavy dependency for some workflows; allow import to proceed
try:
    import tensorflow as tf
    _HAS_TF = True
except Exception:
    tf = None
    _HAS_TF = False

# Import only the non-Sionna dependent functions from scene_helpers
from .scene_helpers import grid_indices_to_center_coordinate, coordinate_to_grid_indices

# Import Sionna-dependent functions lazily when needed
# def _import_sionna_scene_helpers():
#     """Lazy import of Sionna-dependent scene helper functions"""
#     from scene_helpers import remove_all_transmitters, get_scene_bounds3d, get_sionna_scene, get_scene_name
#     return remove_all_transmitters, get_scene_bounds3d, get_sionna_scene, get_scene_name
from .scene_helpers import remove_all_transmitters, get_scene_bounds3d, get_sionna_scene, get_scene_name


import re
import os

MAX_DEPTH=30
CELL_SIZE=(2,2,2)  # Use a tuple of length 2 for Sionna RT compatibility
SAMPLES_PER_TX = 10**8
THRESHOLD = -100  # dBm threshold for coverage calculation

# Import relevant components from Sionna RT if available. If not available, set a flag
try:
    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                          PathSolver, RadioMapSolver, subcarrier_frequencies
    _HAS_SIONNA = True
except Exception:
    # Sionna is not available in this environment; functions that require it will raise at runtime
    load_scene = None
    PlanarArray = None
    Transmitter = None
    Receiver = None
    Camera = None
    PathSolver = None
    RadioMapSolver = None
    subcarrier_frequencies = None
    _HAS_SIONNA = False

def rss_map_full(scene, tx_position=[8.5,21,27], max_depth=MAX_DEPTH, cell_size=CELL_SIZE, samples_per_tx=SAMPLES_PER_TX, csv_file="full_rss_map.csv"):
    if not _HAS_SIONNA:
        raise ImportError("sionna.rt is required to run rss_map_full but is not installed in this environment.")
    remove_all_transmitters(scene)
    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="V")

    # Create transmitter
    tx = Transmitter(name="tx",
                    position= tx_position,
                    display_radius=2)

    # Add transmitter instance to scene
    scene.add(tx)
    rm_solver = RadioMapSolver()
    # Ensure cell_size is a tuple of length 2 for Sionna RT
    cell_size_2d = tuple(cell_size[:2])
    rm = rm_solver(scene=scene,
                   max_depth=max_depth,
                   cell_size=cell_size_2d,
                   samples_per_tx=samples_per_tx)
    return rm.rss.numpy() if hasattr(rm.rss, 'numpy') else rm.rss

def generate_all_rx_positions(scene, cell_size=CELL_SIZE[0:2], height=20.0):
    """
    Generate all receiver positions covering the scene at a given height and cell size.
    Args:
        scene: Sionna scene object
        cell_size: tuple (dx, dy) for grid spacing
        height: z-coordinate for all receiver positions
    Returns:
        np.ndarray of shape (N, 3): receiver positions (x, y, z)
    """
    x_min, x_max, y_min, y_max, z_min, z_max = get_scene_bounds3d(scene)
    x_centers = np.arange(x_min, x_max, cell_size[0])
    y_centers = np.arange(y_min, y_max, cell_size[1])
    X, Y = np.meshgrid(x_centers, y_centers)
    rx_positions = np.column_stack((X.flatten(), Y.flatten(), np.full(X.size, height)))
    return rx_positions

def generate_random_rx_positions(scene, N, height=20.0, seed=None):
    """
    Generate N random receiver positions within the scene bounds at a given height.
    Args:
        scene: Sionna scene object
        N: Number of random receiver positions
        height: z-coordinate for all receiver positions
        seed: Optional random seed for reproducibility
    Returns:
        np.ndarray of shape (N, 3): receiver positions (x, y, z)
    """
    x_min, x_max, y_min, y_max, z_min, z_max = get_scene_bounds3d(scene)
    rng = np.random.default_rng(seed)
    x_rand = rng.uniform(x_min, x_max, N)
    y_rand = rng.uniform(y_min, y_max, N)
    z_rand = np.full(N, height)
    rx_positions = np.column_stack((x_rand, y_rand, z_rand))
    return rx_positions

def compute_coverage(scene_name, tx_grid_idx=None, rx_grid_indices=None, threshold_dbm=-100):
    """
    Compute coverage for a given scene and transmitter grid index, optionally restricted to specific receiver grid indices.
    Args:
        scene: Sionna scene object
        tx_grid_idx: Transmitter grid index (row, col)
        rx_grid_indices: Optional array-like of receiver grid indices (N x 2)
    Returns:
        float: Coverage value (ratio of area above threshold)
    """
    # Load the Sionna scene corresponding to the scene name
    scene = get_sionna_scene(scene_name)

    # If tx_grid_idx is provided, convert to world coordinate for Sionna
    if tx_grid_idx is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = get_scene_bounds3d(scene)
        cell_size = CELL_SIZE
        tx_position = grid_indices_to_center_coordinate(tx_grid_idx, x_min, y_min, z_min, cell_size)
    else:
        tx_position = [8.5, 21, 27]  # Default
    rss = rss_map_full(scene, tx_position=tx_position, csv_file=None)
    rss_array = rss if not hasattr(rss, 'numpy') else rss.numpy()
    if rx_grid_indices is None:
        coverage = compute_coverage_from_arr(rss_array, threshold_dbm=threshold_dbm)
        return coverage
    else:
        H, W = rss_array.shape
        indices = np.array(rx_grid_indices, dtype=int)
        valid_mask = (indices[:,0] >= 0) & (indices[:,0] < H) & (indices[:,1] >= 0) & (indices[:,1] < W)
        indices = indices[valid_mask]
        rss_selected = rss_array[indices[:,0], indices[:,1]]
        with np.errstate(divide='ignore'):
            rss_dbm = 10 * np.log10(rss_selected)
        above_threshold_samples = np.sum(rss_dbm > threshold_dbm)
        total_samples = len(rss_selected)
        coverage = above_threshold_samples / total_samples if total_samples > 0 else 0.0
        return coverage

def compute_coverage_from_arr(rss_array, threshold_dbm=-100):
    """
    Compute coverage from RSS array.
    Coverage is defined as the ratio of the area where RSS is above the threshold.
    
    Args:
        rss_array (numpy.ndarray): 2D array of RSS values in absolute (linear) scale
        threshold_dbm (float): RSS threshold for coverage calculation in dBm (default: -100 dBm)
    
    Returns:
        float: Coverage value (ratio of area above threshold)
    """
    # Convert absolute RSS values to dBm
    # dBm = 10 * log10(absolute_value)
    with np.errstate(divide='ignore'):  # Ignore warnings about log of zero
        rss_dbm = 10 * np.log10(rss_array)
    
    # Count total and above-threshold samples
    total_samples = rss_array.size
    above_threshold_samples = np.sum(rss_dbm > threshold_dbm)
    
    # Compute coverage as ratio
    coverage = above_threshold_samples / total_samples
    
    return coverage

def compute_coverage_from_csv(csv_file, rx_grid_indices=None, threshold_dbm=THRESHOLD):
    """
    Computes the coverage from a CSV file containing RSS data, optionally restricted to rx_grid_indices.
    Args:
        csv_file (str): Path to the CSV file.
        rx_grid_indices (array-like): Optional list of receiver grid indices (N x 2)
        threshold_dbm (float): RSS threshold for coverage calculation in dBm
    Returns:
        float: Coverage value.
    """
    df = pd.read_csv(csv_file, header=None)
    rss_array = df.values
    if rx_grid_indices is None:
        return compute_coverage_from_arr(rss_array, threshold_dbm=threshold_dbm)
    else:
        H, W = rss_array.shape
        indices = np.array(rx_grid_indices, dtype=int)
        valid_mask = (indices[:,0] >= 0) & (indices[:,0] < H) & (indices[:,1] >= 0) & (indices[:,1] < W)
        indices = indices[valid_mask]
        rss_selected = rss_array[indices[:,0], indices[:,1]]
        return compute_coverage_from_arr(rss_selected, threshold_dbm=threshold_dbm)

def compute_coverage_from_csv_gz(file_path, rx_grid_indices=None, threshold_dbm=THRESHOLD, scene_name="munich"):
    """
    Given a .csv.gz RSS file whose name encodes the transmitter coordinates, compute the coverage, optionally restricted to rx_grid_indices.
    Args:
        file_path (str): Path to the .csv.gz file (e.g., rss_munich_625.42,-457.60,20.00.csv.gz)
        rx_grid_indices (array-like): Optional list of receiver grid indices (N x 2)
        threshold_dbm (float): Coverage threshold in dBm
        scene_name (str): Scene name (default: "munich")
    Returns:
        tuple: (tx_x, tx_y, tx_z, coverage)
    """
    filename = os.path.basename(file_path)
    m = re.match(r"rss_" + re.escape(scene_name) + r"_([\-\d.]+),([\-\d.]+),([\-\d.]+)\.csv\.gz", filename)
    if not m:
        raise ValueError(f"Filename {filename} does not match expected pattern for scene '{scene_name}'.")
    tx_x, tx_y, tx_z = map(float, m.groups())
    rss_array = pd.read_csv(file_path, header=None).values
    if rx_grid_indices is None:
        coverage = compute_coverage_from_arr(rss_array, threshold_dbm=threshold_dbm)
    else:
        H, W = rss_array.shape
        indices = np.array(rx_grid_indices, dtype=int)
        valid_mask = (indices[:,0] >= 0) & (indices[:,0] < H) & (indices[:,1] >= 0) & (indices[:,1] < W)
        indices = indices[valid_mask]
        rss_selected = rss_array[indices[:,0], indices[:,1]]
        coverage = compute_coverage_from_arr(rss_selected, threshold_dbm=threshold_dbm)
    return tx_x, tx_y, tx_z, coverage


def compute_coverage_from_h5(file_path, rx_grid_indices=None, threshold_dbm=THRESHOLD, scene_name="munich"):
    """
    Given an HDF5 (.h5/.hdf5) RSS file whose name may encode the transmitter coordinates,
    compute the coverage, optionally restricted to rx_grid_indices.
    Returns: (tx_x, tx_y, tx_z, coverage)
    """
    filename = os.path.basename(file_path)

    # Try parse patterns: tokenized _x{X}_y{Y}_z{Z} or legacy rss_{scene}_{x},{y},{z}.h5
    m = re.search(r"_x(?P<x>[-+]?\d*\.?\d+)_y(?P<y>[-+]?\d*\.?\d+)_z(?P<z>[-+]?\d*\.?\d+)", filename)
    if m:
        tx_x, tx_y, tx_z = float(m.group('x')), float(m.group('y')), float(m.group('z'))
    else:
        m2 = re.match(r"rss_" + re.escape(scene_name) + r"_([\-\d.]+),([\-\d.]+),([\-\d.]+)\.(?:h5|hdf5)$", filename)
        if m2:
            tx_x, tx_y, tx_z = map(float, m2.groups())
        else:
            # fallback: last three numeric tokens
            nums = re.findall(r"[-+]?\d*\.?\d+", filename)
            if len(nums) >= 3:
                tx_x, tx_y, tx_z = map(float, nums[-3:])
            else:
                raise ValueError(f"Filename {filename} does not match expected pattern for scene '{scene_name}'.")

    try:
        import h5py
    except Exception as e:
        raise ImportError("h5py is required to read .h5 files. Install via 'pip install h5py'") from e

    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"No datasets found in h5 file: {file_path}")
        # prefer common dataset names
        if 'rss' in keys:
            key = 'rss'
        elif 'rss_data' in keys:
            key = 'rss_data'
        else:
            key = keys[0]
        data = f[key][()]

    rss_array = np.array(data)
    if rss_array.ndim != 2:
        rss_array = np.squeeze(rss_array)
        if rss_array.ndim != 2:
            raise ValueError(f"After squeezing, RSS array must have 2 dimensions. Current shape: {rss_array.shape}")

    if rx_grid_indices is None:
        coverage = compute_coverage_from_arr(rss_array, threshold_dbm=threshold_dbm)
    else:
        H, W = rss_array.shape
        indices = np.array(rx_grid_indices, dtype=int)
        valid_mask = (indices[:,0] >= 0) & (indices[:,0] < H) & (indices[:,1] >= 0) & (indices[:,1] < W)
        indices = indices[valid_mask]
        rss_selected = rss_array[indices[:,0], indices[:,1]]
        coverage = compute_coverage_from_arr(rss_selected, threshold_dbm=threshold_dbm)

    return tx_x, tx_y, tx_z, coverage


def compute_coverage_for_directory_to_csv(dir_path, output_csv, threshold_dbm=THRESHOLD, scene_name="munich", rx_grid_indices=None):
    """
    For all rss_<scene_name>_*.csv.gz files in the directory, compute coverage and write to a CSV file.
    If threshold_dbm is a list, compute coverage for each threshold separately.
    
    Args:
        dir_path (str): Directory containing rss_<scene_name>_*.csv.gz files
        output_csv (str): Path to output CSV file
        threshold_dbm (float or list): Coverage threshold(s) in dBm (default: global THRESHOLD)
                                       If list, coverage is computed for each threshold
        scene_name (str): Scene name (default: "munich")
        rx_grid_indices (array-like): Optional list of receiver grid indices (N x 2)
    """
    import glob
    
    # Normalize threshold_dbm to a list
    if isinstance(threshold_dbm, (list, tuple)):
        thresholds = list(threshold_dbm)
    else:
        thresholds = [threshold_dbm]
    
    pattern = os.path.join(dir_path, f'rss_{scene_name}_*.csv.gz')
    files = glob.glob(pattern)
    total = len(files)
    print(f"Found {total} files to process with {len(thresholds)} threshold(s).")
    
    results = []
    for idx, file_path in enumerate(files, 1):
        try:
            tx_x, tx_y, tx_z, _ = compute_coverage_from_csv_gz(file_path, rx_grid_indices=rx_grid_indices, threshold_dbm=thresholds[0], scene_name=scene_name)
            row = [tx_x, tx_y, tx_z]
            
            # Compute coverage for each threshold
            for thr in thresholds:
                _, _, _, coverage = compute_coverage_from_csv_gz(file_path, rx_grid_indices=rx_grid_indices, threshold_dbm=thr, scene_name=scene_name)
                row.append(coverage)
            
            results.append(row)
            print(f"\rProcessed {idx}/{total}: {os.path.basename(file_path)}", end="", flush=True)
        except Exception as e:
            print(f"\nSkipping {file_path}: {e}")
    
    print("\nAll files processed.")
    
    # Build column names
    columns = ["x", "y", "z"]
    for thr in thresholds:
        columns.append(f"coverage_thr{thr}")
    
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Wrote coverage data for {len(results)} files to {output_csv}")

def find_closest_rss_file(coord, rm_data_dir, scene_name="munich"): 
    """
    Given a coordinate (x, y, z) and a directory containing rss_<scene_name>_<x>,<y>,<z>.csv.gz files,
    return the path to the file whose name has the closest coordinate.
    Args:
        coord (tuple/list): (x, y, z) coordinate
        rm_data_dir (str): Path to directory containing rss files
        scene_name (str): Scene name (default: "munich")
    Returns:
        str: Path to the closest file
    """
    pattern = re.compile(rf"rss_{scene_name}_([\-\d.]+),([\-\d.]+),([\-\d.]+)\.csv\.gz$")
    min_dist = float('inf')
    closest_file = None
    for fname in os.listdir(rm_data_dir):
        m = pattern.match(fname)
        if m:
            file_coord = np.array(list(map(float, m.groups())))
            dist = np.linalg.norm(np.array(coord) - file_coord)
            if dist < min_dist:
                min_dist = dist
                closest_file = os.path.join(rm_data_dir, fname)
    return closest_file

def compute_coverage_for_closest_coordinate(coord, rm_data_dir, threshold_dbm=THRESHOLD, scene_name="munich", rx_grid_indices=None):
    """
    Given a coordinate and a directory of rss_<scene_name>_<x>,<y>,<z>.csv.gz files,
    find the file with the closest coordinate and compute its coverage.
    Args:
        coord (tuple/list): (x, y, z) coordinate
        rm_data_dir (str): Path to directory containing rss files
        threshold_dbm (float): Coverage threshold in dBm (default: global THRESHOLD)
        scene_name (str): Scene name (default: "munich")
        rx_grid_indices (array-like): Optional list of receiver grid indices (N x 2)
    Returns:
        tuple: (closest_coord, coverage, file_path)
    """
    file_path = find_closest_rss_file(coord, rm_data_dir, scene_name=scene_name)
    if file_path is None:
        raise FileNotFoundError("No matching RSS file found in directory.")
    fname = os.path.basename(file_path)
    m = re.match(rf"rss_{scene_name}_([\-\d.]+),([\-\d.]+),([\-\d.]+)\.csv\.gz$", fname)
    if not m:
        raise ValueError(f"Filename {fname} does not match expected pattern for scene '{scene_name}'.")
    closest_coord = tuple(map(float, m.groups()))
    coverage_tuple = compute_coverage_from_csv_gz(file_path, rx_grid_indices=rx_grid_indices, threshold_dbm=threshold_dbm, scene_name=scene_name)
    return coverage_tuple[-1] #closest_coord, coverage, file_path