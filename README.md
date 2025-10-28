# genrssmaps

A Python package for generating RSS (Received Signal Strength) maps using Sionna RT ray-tracing simulations.

## Features

- Generate RSS coverage maps for wireless communication scenarios
- Process transmitter position data
- Scene analysis and manipulation utilities
- Coverage analysis helpers with configurable parameters
- Support for multiple file formats (CSV with optional compression)

## Installation

### From GitHub (Recommended)

You can install this package directly from GitHub using pip:

```bash
pip install git+https://github.com/gourav-prateek-sharma/genrssmaps.git
```

### Development Installation

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/gourav-prateek-sharma/genrssmaps.git
cd genrssmaps
pip install -e .
```

## Prerequisites

This package requires [Sionna](https://nvlabs.github.io/sionna/) to be installed separately, as it's not available on PyPI. Please follow the official Sionna installation instructions:

```bash
# Install Sionna (GPU version recommended)
pip install tensorflow[and-cuda]  # For GPU support
pip install sionna
```

## Usage

### As a Python Package

```python
import genrssmaps

# Generate RSS map
from genrssmaps import rss_map_full, rss_write_csv
from sionna.rt import load_scene

# Load your scene
scene = load_scene("path/to/your/scene.xml")

# Define transmitter position
tx_position = [10.0, 5.0, 3.0]  # [x, y, z] coordinates

# Generate and save RSS map
rss_array, output_path = rss_write_csv(
    scene, 
    tx_position, 
    csv_file="coverage_map.csv",
    compress=True
)
```

### Command Line Tools

After installation, you'll have access to command-line tools:

```bash
# Generate RSS CSV from command line
gen-rss-csv --scene path/to/scene.xml --tx-position 10,5,3 --output coverage.csv

# Generate transmitter positions  
gen-tx-pos --help
```

## Configuration

The package uses several default parameters that can be customized:

- `MAX_DEPTH`: Maximum ray-tracing depth (default: 30)
- `CELL_SIZE`: Grid cell size for coverage maps (default: (2,2,2))
- `SAMPLES_PER_TX`: Number of samples per transmitter (default: 10^8)
- `THRESHOLD`: RSS threshold in dBm (default: -100)

## API Reference

### Main Functions

- `rss_map_full()`: Generate RSS coverage map using ray-tracing
- `rss_write_csv()`: Generate RSS map and save to CSV file
- `get_scene_bounds3d()`: Get 3D bounds of a scene
- `sanitize_filename_part()`: Clean filename strings

### Modules

- `genrssmaps.coverage_helpers`: RSS computation and coverage analysis
- `genrssmaps.scene_helpers`: Scene manipulation utilities
- `genrssmaps.gen_rss_csv`: CSV generation and file I/O
- `genrssmaps.gen_tx_pos`: Transmitter position utilities

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- TensorFlow >= 2.8.0
- Sionna (install separately)

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Issues

Please report issues at: https://github.com/gourav-prateek-sharma/genrssmaps/issues