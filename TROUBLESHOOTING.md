# Troubleshooting Guide

## NumPy Compatibility Issues

If you encounter errors like `ValueError: numpy.dtype size changed, may indicate binary incompatibility`, this is typically due to version mismatches between NumPy and compiled extensions (especially TensorFlow/Sionna).

### Quick Fixes:

1. **Reinstall NumPy:**
   ```bash
   pip install --force-reinstall numpy
   ```

2. **Reinstall TensorFlow and dependencies:**
   ```bash
   pip install --force-reinstall tensorflow numpy
   ```

3. **Fresh environment (recommended):**
   ```bash
   # Create new conda environment
   conda create -n genrss python=3.10
   conda activate genrss
   
   # Install dependencies in correct order
   pip install numpy>=1.21.0
   pip install tensorflow>=2.8.0
   pip install sionna
   pip install git+https://github.com/gourav-prateek-sharma/genrssmaps.git[all]
   ```

4. **Colab/Jupyter specific:**
   ```python
   # In a notebook cell, restart runtime after running:
   !pip install --force-reinstall numpy tensorflow
   ```

### Environment Requirements:

- Python >= 3.8
- NumPy >= 1.20.0
- TensorFlow >= 2.8.0
- Compatible versions are crucial for Sionna

### Alternative Import Strategy:

If you continue having issues, try importing components individually:

```python
# Instead of importing everything at once
try:
    from genrssmaps.storage_utils import save_rss_data, load_rss_data
    from genrssmaps.scene_helpers import get_sionna_scene, get_scene_bounds3d
    print("Successfully imported core functions")
except ImportError as e:
    print(f"Import error: {e}")
```

### Verification:

Test your environment with:

```python
import numpy as np
import tensorflow as tf
print(f"NumPy version: {np.__version__}")
print(f"TensorFlow version: {tf.__version__}")

# Test basic functionality
x = np.array([1, 2, 3])
print(f"NumPy test: {x.sum()}")
```