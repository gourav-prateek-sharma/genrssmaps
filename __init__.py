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

# Import main functionality with error handling
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

try:
    from .scene_helpers import (
        get_scene_bounds3d,
        grid_indices_to_center_coordinate,
        coordinate_to_grid_indices
    )
    _scene_available = True
except ImportError as e:
    print(f"Warning: Could not import scene_helpers: {e}")
    _scene_available = False

try:
    from .gen_rss_csv import (
        rss_write_csv,
        sanitize_filename_part
    )
    _gen_rss_available = True
except ImportError as e:
    print(f"Warning: Could not import gen_rss_csv: {e}")
    _gen_rss_available = False

# Build __all__ based on what's available
__all__ = []
if _coverage_available:
    __all__.extend(['rss_map_full', 'MAX_DEPTH', 'CELL_SIZE', 'SAMPLES_PER_TX', 'THRESHOLD'])
if _scene_available:
    __all__.extend(['get_scene_bounds3d', 'grid_indices_to_center_coordinate', 'coordinate_to_grid_indices'])
if _gen_rss_available:
    __all__.extend(['rss_write_csv', 'sanitize_filename_part'])