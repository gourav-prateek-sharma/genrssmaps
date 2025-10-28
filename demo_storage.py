#!/usr/bin/env python3
"""
Example script demonstrating efficient RSS storage formats
"""

import numpy as np
from genrssmaps import save_rss_data, load_rss_data, get_format_info, benchmark_formats

def create_sample_rss_data():
    """Create sample RSS data for demonstration"""
    # Create a realistic RSS map with some structure
    height, width = 200, 300
    
    # Create base RSS values
    rss_data = np.random.uniform(-120, -30, size=(height, width)).astype(np.float32)
    
    # Add some structure (buildings cause shadows/higher path loss)
    center_y, center_x = height // 2, width // 2
    
    # Create a few "building" areas with higher path loss
    for i in range(5):
        y = np.random.randint(20, height - 20)
        x = np.random.randint(20, width - 20)
        h = np.random.randint(15, 40)
        w = np.random.randint(15, 40)
        rss_data[y:y+h, x:x+w] -= np.random.uniform(20, 40)
    
    # Add some NaN values for areas with no coverage
    mask = np.random.random((height, width)) < 0.05  # 5% no coverage
    rss_data[mask] = np.nan
    
    return rss_data

def demo_formats():
    """Demonstrate different storage formats"""
    print("RSS Storage Format Demonstration")
    print("=" * 50)
    
    # Create sample data
    rss_data = create_sample_rss_data()
    print(f"Sample RSS data shape: {rss_data.shape}")
    print(f"Data range: {np.nanmin(rss_data):.1f} to {np.nanmax(rss_data):.1f} dBm")
    print(f"Memory usage: {rss_data.nbytes / 1024:.1f} KB")
    
    # Sample metadata
    metadata = {
        'tx_position': [10.0, 5.0, 3.0],
        'scene_name': 'demo_scene',
        'max_depth': 30,
        'cell_size': [2.0, 2.0, 2.0],
        'samples_per_tx': 1000000
    }
    
    print(f"\nMetadata: {metadata}")
    print("\n" + "=" * 50)
    
    # Show available formats
    formats = get_format_info()
    print("Available Storage Formats:")
    for fmt, info in formats.items():
        status = "✓" if info['available'] else "✗"
        print(f"  {status} {fmt}: {info['description']}")
        if info['available']:
            print(f"    - Compression: {info['compression']}")
            print(f"    - Speed: {info['speed']}")
            print(f"    - Recommended for: {info['recommended_for']}")
    
    print("\n" + "=" * 50)
    print("Running benchmark...")
    
    # Run benchmark
    results = benchmark_formats(rss_data, metadata)
    
    print("\nBenchmark Results:")
    print(f"{'Format':<12} {'Size (KB)':<12} {'Compression':<12} {'Save (ms)':<12} {'Load (ms)':<12}")
    print("-" * 70)
    
    csv_size = None
    for fmt, result in results.items():
        if 'error' not in result:
            size_kb = result['file_size'] / 1024
            compression = f"{result['compression_ratio']:.1f}x"
            save_ms = f"{result['save_time']*1000:.1f}"
            load_ms = f"{result['load_time']*1000:.1f}"
            
            print(f"{fmt:<12} {size_kb:<12.1f} {compression:<12} {save_ms:<12} {load_ms:<12}")
            
            if fmt == 'csv':
                csv_size = size_kb
    
    # Show space savings compared to CSV
    if csv_size:
        print(f"\nSpace savings compared to CSV:")
        for fmt, result in results.items():
            if 'error' not in result and fmt != 'csv':
                size_kb = result['file_size'] / 1024
                saving = ((csv_size - size_kb) / csv_size) * 100
                print(f"  {fmt}: {saving:.1f}% smaller")
    
    print("\n" + "=" * 50)
    print("Format Recommendations:")
    print("- NPZ: Best for general use, fastest loading")
    print("- HDF5: Best for datasets with rich metadata") 
    print("- Parquet: Best for sparse data and analytics")
    print("- Zarr: Best for very large arrays and cloud storage")
    print("- CSV: Use only for human-readable output")

def demo_save_load():
    """Demonstrate saving and loading data"""
    print("\n" + "=" * 50)
    print("Save/Load Demonstration")
    
    # Create sample data
    rss_data = create_sample_rss_data()
    metadata = {
        'tx_position': [15.0, 10.0, 5.0],
        'scene_name': 'test_scene'
    }
    
    # Test NPZ format (most compatible)
    print("Testing NPZ format...")
    filepath = save_rss_data(rss_data, "test_rss", "npz", metadata=metadata)
    print(f"Saved to: {filepath}")
    
    loaded_data, loaded_metadata = load_rss_data(filepath)
    print(f"Loaded data shape: {loaded_data.shape}")
    print(f"Loaded metadata: {loaded_metadata}")
    
    # Verify data integrity
    data_match = np.allclose(rss_data, loaded_data, equal_nan=True)
    print(f"Data integrity check: {'✓ PASS' if data_match else '✗ FAIL'}")
    
    # Clean up
    import os
    os.remove(filepath)
    print("Cleanup completed")

if __name__ == "__main__":
    demo_formats()
    demo_save_load()