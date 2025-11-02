"""
genrssmaps - A package for generating RSS (Received Signal Strength) maps using Sionna RT.

This package provides tools for:
- Generating RSS coverage maps
- Processing transmitter positions  
- Scene analysis and manipulation
- Coverage analysis helpers
"""

__version__ = "0.1.0"
__author__ = "Gourav Prateek Sharma"

# Handle NumPy compatibility issues
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Import main functionality with error handling
try:
    # Try to handle NumPy compatibility issues
    import numpy as np
    # Force reload if there are version issues
    if hasattr(np, '__version__'):
        pass  # NumPy loaded successfully
except (ImportError, ValueError) as e:
    if "numpy.dtype size changed" in str(e):
        print("Warning: NumPy compatibility issue detected. Please reinstall numpy:")
        print("  pip install --force-reinstall numpy")
    raise e

try:
    from .coverage_helpers import (
        rss_map_full,
        MAX_DEPTH,
        CELL_SIZE, 
        SAMPLES_PER_TX,
        THRESHOLD
    )
    _coverage_available = True
except ImportError as e:
    print(f"Warning: Could not import coverage_helpers: {e}")
    _coverage_available = False
except ValueError as e:
    if "numpy.dtype size changed" in str(e):
        print("Warning: NumPy compatibility issue in coverage_helpers.")
        print("Please try: pip install --force-reinstall numpy tensorflow")
    print(f"Warning: Could not import coverage_helpers: {e}")
    _coverage_available = False

try:
    from .scene_helpers import (
        get_scene_bounds3d,
        get_sionna_scene,
        grid_indices_to_center_coordinate,
        coordinate_to_grid_indices
    )
    _scene_available = True
except ImportError as e:
    print(f"Warning: Could not import scene_helpers: {e}")
    _scene_available = False
except ValueError as e:
    if "numpy.dtype size changed" in str(e):
        print("Warning: NumPy compatibility issue in scene_helpers.")
    print(f"Warning: Could not import scene_helpers: {e}")
    _scene_available = False

try:
    from .gen_rss_csv import (
        rss_write_csv,
        rss_write_efficient,
        sanitize_filename_part
    )
    _gen_rss_available = True
except ImportError as e:
    print(f"Warning: Could not import gen_rss_csv: {e}")
    _gen_rss_available = False
except ValueError as e:
    if "numpy.dtype size changed" in str(e):
        print("Warning: NumPy compatibility issue in gen_rss_csv.")
    print(f"Warning: Could not import gen_rss_csv: {e}")
    _gen_rss_available = False

try:
    from .storage_utils import (
        save_rss_data,
        load_rss_data,
        get_format_info,
        benchmark_formats
    )
    _storage_available = True
except ImportError as e:
    print(f"Warning: Could not import storage_utils: {e}")
    _storage_available = False
except ValueError as e:
    if "numpy.dtype size changed" in str(e):
        print("Warning: NumPy compatibility issue in storage_utils.")
    print(f"Warning: Could not import storage_utils: {e}")
    _storage_available = False

# Build __all__ based on what's available
__all__ = []
if _coverage_available:
    __all__.extend(['rss_map_full', 'MAX_DEPTH', 'CELL_SIZE', 'SAMPLES_PER_TX', 'THRESHOLD'])
if _scene_available:
    __all__.extend(['get_scene_bounds3d', 'get_sionna_scene', 'grid_indices_to_center_coordinate', 'coordinate_to_grid_indices'])
if _gen_rss_available:
    __all__.extend(['rss_write_csv', 'rss_write_efficient', 'sanitize_filename_part'])
if _storage_available:
    __all__.extend(['save_rss_data', 'load_rss_data', 'get_format_info', 'benchmark_formats'])

# Show import status and help message if there were issues
_total_modules = 4
_loaded_modules = sum([_coverage_available, _scene_available, _gen_rss_available, _storage_available])

if _loaded_modules < _total_modules:
    print(f"\ngenrssmaps: {_loaded_modules}/{_total_modules} modules loaded successfully.")
    print("If you're seeing NumPy compatibility errors, try:")
    print("  pip install --force-reinstall numpy tensorflow")
    print("For more help, see TROUBLESHOOTING.md in the repository.")
    print()