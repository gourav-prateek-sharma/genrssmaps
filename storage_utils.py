#!/usr/bin/env python3
"""
Storage utilities for RSS coverage maps with multiple efficient formats.
Provides alternatives to CSV for better compression and faster I/O.
"""

import numpy as np
import pandas as pd
import os
import gzip
import shutil
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path

# Optional imports with fallbacks
try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    h5py = None
    HAS_HDF5 = False

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PARQUET = True
except ImportError:
    pq = pa = None
    HAS_PARQUET = False

try:
    import zarr
    import numcodecs
    HAS_ZARR = True
except ImportError:
    zarr = numcodecs = None
    HAS_ZARR = False


def save_rss_data(
    rss_array: np.ndarray,
    filepath: str,
    format_type: str = "npz",
    compression_level: int = 6,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Save RSS array in various efficient formats.
    
    Args:
        rss_array: 2D numpy array with RSS data
        filepath: Output file path (extension will be adjusted based on format)
        format_type: Storage format - 'npz', 'hdf5', 'parquet', 'zarr', 'csv'
        compression_level: Compression level (0-9, higher = better compression)
        metadata: Dictionary with metadata (tx_position, scene info, etc.)
        **kwargs: Additional format-specific arguments
    
    Returns:
        Final filepath with correct extension
    """
    
    # Ensure we have a 2D array
    if rss_array.ndim != 2:
        rss_array = np.squeeze(rss_array)
        if rss_array.ndim != 2:
            raise ValueError(f"RSS array must be 2D. Got shape: {rss_array.shape}")
    
    # Get base path without extension
    base_path = Path(filepath).with_suffix('')
    
    if format_type.lower() == "npz":
        return _save_npz(rss_array, base_path, compression_level, metadata)
    elif format_type.lower() == "hdf5" or format_type.lower() == "h5":
        return _save_hdf5(rss_array, base_path, compression_level, metadata)
    elif format_type.lower() == "parquet":
        return _save_parquet(rss_array, base_path, compression_level, metadata)
    elif format_type.lower() == "zarr":
        return _save_zarr(rss_array, base_path, compression_level, metadata, **kwargs)
    elif format_type.lower() == "csv":
        return _save_csv(rss_array, base_path, compression_level > 0)
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def load_rss_data(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load RSS array from various formats.
    
    Args:
        filepath: Path to the RSS data file
    
    Returns:
        Tuple of (rss_array, metadata_dict)
    """
    
    path = Path(filepath)
    
    if path.suffix.lower() == ".npz":
        return _load_npz(filepath)
    elif path.suffix.lower() in [".h5", ".hdf5"]:
        return _load_hdf5(filepath)
    elif path.suffix.lower() == ".parquet":
        return _load_parquet(filepath)
    elif path.name.endswith(".zarr") or path.is_dir():
        return _load_zarr(filepath)
    elif path.suffix.lower() in [".csv", ".gz"]:
        return _load_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def _save_npz(rss_array: np.ndarray, base_path: Path, compression_level: int, metadata: Optional[Dict]) -> str:
    """Save as compressed NumPy .npz format"""
    output_path = str(base_path.with_suffix('.npz'))
    
    # Prepare data dictionary
    save_dict = {'rss_data': rss_array}
    
    # Add metadata as separate arrays
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                save_dict[f'meta_{key}'] = np.array(value)
            elif isinstance(value, (int, float, str)):
                save_dict[f'meta_{key}'] = np.array([value])
            else:
                save_dict[f'meta_{key}'] = np.array([str(value)])
    
    # Save with compression
    np.savez_compressed(output_path, **save_dict)
    return output_path


def _load_npz(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load from .npz format"""
    data = np.load(filepath)
    rss_array = data['rss_data']
    
    # Extract metadata
    metadata = {}
    for key in data.keys():
        if key.startswith('meta_'):
            meta_key = key[5:]  # Remove 'meta_' prefix
            value = data[key]
            if value.size == 1:
                metadata[meta_key] = value.item()
            else:
                metadata[meta_key] = value.tolist()
    
    return rss_array, metadata


def _save_hdf5(rss_array: np.ndarray, base_path: Path, compression_level: int, metadata: Optional[Dict]) -> str:
    """Save as HDF5 format"""
    if not HAS_HDF5:
        raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")
    
    output_path = str(base_path.with_suffix('.h5'))
    
    with h5py.File(output_path, 'w') as f:
        # Save main data with compression
        compression = 'gzip' if compression_level > 0 else None
        f.create_dataset('rss_data', data=rss_array, 
                        compression=compression, 
                        compression_opts=compression_level,
                        shuffle=True,  # Improve compression
                        fletcher32=True)  # Add checksum
        
        # Save metadata as attributes
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value
                
        # Add format info
        f.attrs['format_version'] = '1.0'
        f.attrs['data_shape'] = rss_array.shape
    
    return output_path


def _load_hdf5(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load from HDF5 format"""
    if not HAS_HDF5:
        raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")
    
    with h5py.File(filepath, 'r') as f:
        rss_array = f['rss_data'][:]
        
        # Extract metadata from attributes
        metadata = {}
        for key, value in f.attrs.items():
            if key not in ['format_version', 'data_shape']:
                metadata[key] = value
    
    return rss_array, metadata


def _save_parquet(rss_array: np.ndarray, base_path: Path, compression_level: int, metadata: Optional[Dict]) -> str:
    """Save as Parquet format"""
    if not HAS_PARQUET:
        raise ImportError("pyarrow is required for Parquet format. Install with: pip install pyarrow")
    
    output_path = str(base_path.with_suffix('.parquet'))
    
    # Convert 2D array to DataFrame with position indices
    height, width = rss_array.shape
    
    # Create a long-format DataFrame (more efficient for sparse data)
    rows, cols = np.where(~np.isnan(rss_array))  # Only store non-NaN values
    values = rss_array[rows, cols]
    
    df = pd.DataFrame({
        'row': rows.astype(np.uint16),  # Use smaller dtypes
        'col': cols.astype(np.uint16),
        'rss': values.astype(np.float32)  # Use float32 instead of float64
    })
    
    # Add metadata to parquet metadata
    parquet_metadata = {'height': height, 'width': width}
    if metadata:
        parquet_metadata.update(metadata)
    
    # Convert metadata to string format for parquet
    meta_df = pd.DataFrame([parquet_metadata])
    
    # Save with high compression
    compression_map = {0: None, 1: 'snappy', 2: 'snappy', 3: 'gzip', 
                      4: 'gzip', 5: 'gzip', 6: 'brotli', 7: 'brotli', 8: 'brotli', 9: 'brotli'}
    compression = compression_map.get(compression_level, 'brotli')
    
    # Create a table with metadata
    table = pa.Table.from_pandas(df)
    
    # Add custom metadata
    existing_meta = table.schema.metadata or {}
    existing_meta.update({k.encode(): str(v).encode() for k, v in parquet_metadata.items()})
    table = table.replace_schema_metadata(existing_meta)
    
    pq.write_table(table, output_path, compression=compression)
    return output_path


def _load_parquet(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load from Parquet format"""
    if not HAS_PARQUET:
        raise ImportError("pyarrow is required for Parquet format. Install with: pip install pyarrow")
    
    table = pq.read_table(filepath)
    df = table.to_pandas()
    
    # Extract metadata
    metadata = {}
    if table.schema.metadata:
        for key, value in table.schema.metadata.items():
            try:
                # Try to convert back to appropriate type
                key_str = key.decode()
                value_str = value.decode()
                
                if key_str in ['height', 'width']:
                    metadata[key_str] = int(value_str)
                else:
                    # Try to parse as number, otherwise keep as string
                    try:
                        metadata[key_str] = float(value_str)
                        if metadata[key_str].is_integer():
                            metadata[key_str] = int(metadata[key_str])
                    except ValueError:
                        metadata[key_str] = value_str
            except:
                continue
    
    # Reconstruct 2D array
    height = metadata.get('height', df['row'].max() + 1)
    width = metadata.get('width', df['col'].max() + 1)
    
    rss_array = np.full((height, width), np.nan, dtype=np.float32)
    rss_array[df['row'], df['col']] = df['rss']
    
    return rss_array, metadata


def _save_zarr(rss_array: np.ndarray, base_path: Path, compression_level: int, metadata: Optional[Dict], **kwargs) -> str:
    """Save as Zarr format"""
    if not HAS_ZARR:
        raise ImportError("zarr is required for Zarr format. Install with: pip install zarr")
    
    output_path = str(base_path.with_suffix('.zarr'))
    
    # Configure compression
    if compression_level > 0:
        compressor = numcodecs.Blosc(cname='zstd', clevel=compression_level, shuffle=numcodecs.Blosc.SHUFFLE)
    else:
        compressor = None
    
    # Create zarr array
    chunks = kwargs.get('chunks', (min(512, rss_array.shape[0]), min(512, rss_array.shape[1])))
    
    z = zarr.open(output_path, mode='w', 
                  shape=rss_array.shape, 
                  dtype=rss_array.dtype,
                  chunks=chunks,
                  compressor=compressor)
    
    z[:] = rss_array
    
    # Add metadata
    if metadata:
        z.attrs.update(metadata)
    
    return output_path


def _load_zarr(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load from Zarr format"""
    if not HAS_ZARR:
        raise ImportError("zarr is required for Zarr format. Install with: pip install zarr")
    
    z = zarr.open(filepath, mode='r')
    rss_array = z[:]
    metadata = dict(z.attrs)
    
    return rss_array, metadata


def _save_csv(rss_array: np.ndarray, base_path: Path, compress: bool) -> str:
    """Save as CSV format (legacy support)"""
    if compress:
        output_path = str(base_path.with_suffix('.csv.gz'))
        df = pd.DataFrame(rss_array)
        df.to_csv(output_path, index=False, header=False, compression='gzip')
    else:
        output_path = str(base_path.with_suffix('.csv'))
        df = pd.DataFrame(rss_array)
        df.to_csv(output_path, index=False, header=False)
    
    return output_path


def _load_csv(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load from CSV format"""
    if filepath.endswith('.gz'):
        df = pd.read_csv(filepath, header=None, compression='gzip')
    else:
        df = pd.read_csv(filepath, header=None)
    
    return df.values, {}


def get_format_info():
    """Get information about available formats and their characteristics"""
    formats = {
        'npz': {
            'description': 'NumPy compressed format',
            'compression': 'Excellent',
            'speed': 'Very Fast',
            'metadata': 'Limited',
            'available': True,
            'recommended_for': 'General use, fastest loading'
        },
        'hdf5': {
            'description': 'Hierarchical Data Format 5',
            'compression': 'Excellent',
            'speed': 'Fast',
            'metadata': 'Excellent',
            'available': HAS_HDF5,
            'recommended_for': 'Large datasets, rich metadata'
        },
        'parquet': {
            'description': 'Apache Parquet columnar format',
            'compression': 'Excellent',
            'speed': 'Fast',
            'metadata': 'Good',
            'available': HAS_PARQUET,
            'recommended_for': 'Sparse data, analytics workflows'
        },
        'zarr': {
            'description': 'Chunked compressed arrays',
            'compression': 'Excellent',
            'speed': 'Fast',
            'metadata': 'Good',
            'available': HAS_ZARR,
            'recommended_for': 'Very large arrays, cloud storage'
        },
        'csv': {
            'description': 'Comma-separated values',
            'compression': 'Poor',
            'speed': 'Slow',
            'metadata': 'None',
            'available': True,
            'recommended_for': 'Human readable, compatibility'
        }
    }
    
    return formats


def benchmark_formats(rss_array: np.ndarray, test_metadata: Optional[Dict] = None, temp_dir: str = "/tmp") -> Dict:
    """
    Benchmark different storage formats for size and speed.
    
    Args:
        rss_array: Sample RSS array to test
        test_metadata: Sample metadata to include
        temp_dir: Directory for temporary test files
    
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    if test_metadata is None:
        test_metadata = {
            'tx_position': [10.0, 5.0, 3.0],
            'scene_name': 'test_scene',
            'max_depth': 30,
            'cell_size': [2.0, 2.0, 2.0]
        }
    
    formats = get_format_info()
    results = {}
    
    for fmt_name, fmt_info in formats.items():
        if not fmt_info['available']:
            continue
            
        try:
            test_path = os.path.join(temp_dir, f"test_rss.{fmt_name}")
            
            # Time saving
            start_time = time.time()
            saved_path = save_rss_data(rss_array, test_path, fmt_name, 
                                     compression_level=6, metadata=test_metadata)
            save_time = time.time() - start_time
            
            # Get file size
            file_size = os.path.getsize(saved_path)
            if os.path.isdir(saved_path):  # For zarr
                file_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                              for dirpath, dirnames, filenames in os.walk(saved_path)
                              for filename in filenames)
            
            # Time loading
            start_time = time.time()
            loaded_array, loaded_metadata = load_rss_data(saved_path)
            load_time = time.time() - start_time
            
            # Verify data integrity
            data_match = np.allclose(rss_array, loaded_array, equal_nan=True)
            
            results[fmt_name] = {
                'save_time': save_time,
                'load_time': load_time,
                'file_size': file_size,
                'compression_ratio': rss_array.nbytes / file_size,
                'data_integrity': data_match,
                'metadata_preserved': len(loaded_metadata) > 0 if test_metadata else True
            }
            
            # Cleanup
            if os.path.isdir(saved_path):
                shutil.rmtree(saved_path)
            else:
                os.remove(saved_path)
                
        except Exception as e:
            results[fmt_name] = {'error': str(e)}
    
    return results