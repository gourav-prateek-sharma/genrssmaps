#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the README file if it exists
def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    return "A package for generating RSS (Received Signal Strength) maps using Sionna RT."

setup(
    name="genrssmaps",
    version="0.1.0",
    author="Gourav Prateek Sharma",
    author_email="gourav.4871example.com",  # Replace with your email
    description="A package for generating RSS (Received Signal Strength) maps using Sionna RT",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/gourav-prateek-sharma/genrssmaps",  # Replace with your GitHub URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "tensorflow>=2.8.0",  # For Sionna compatibility
        # Note: sionna is not on PyPI, users need to install it separately
    ],
    extras_require={
        "all": [
            "h5py>=3.0.0",
            "pyarrow>=5.0.0", 
            "zarr>=2.10.0",
        ],
        "hdf5": ["h5py>=3.0.0"],
        "parquet": ["pyarrow>=5.0.0"],
        "zarr": ["zarr>=2.10.0"],
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "gen-rss-csv=genrssmaps.gen_rss_csv:main",
            "gen-tx-pos=genrssmaps.gen_tx_pos:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)