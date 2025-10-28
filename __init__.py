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

# Import main functionality
from .coverage_helpers import (
    rss_map_full,
    MAX_DEPTH,
    CELL_SIZE, 
    SAMPLES_PER_TX,
    THRESHOLD
)

from .scene_helpers import (
    get_scene_bounds3d,
    grid_indices_to_center_coordinate,
    coordinate_to_grid_indices
)

from .gen_rss_csv import (
    rss_write_csv,
    sanitize_filename_part
)

__all__ = [
    'rss_map_full',
    'rss_write_csv', 
    'sanitize_filename_part',
    'get_scene_bounds3d',
    'grid_indices_to_center_coordinate',
    'coordinate_to_grid_indices',
    'MAX_DEPTH',
    'CELL_SIZE',
    'SAMPLES_PER_TX', 
    'THRESHOLD'
]