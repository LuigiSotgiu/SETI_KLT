# SETI KLT Project

This repository contains Python scripts and utilities for applying various Karhunen-Loève Transform (KLT) methodologies to radio astronomy data. It was primarily developed to process and analyze GUPPI raw files, commonly used in SETI research.

## Repository Structure

*   **`KLT.py`**: The main module of the repository. It contains the `KLT` class, which provides an object-oriented interface to execute different variants of the KLT:
    *   **MR-KLT** (Multiple-Realization KLT)
    *   **C-KLT** (Covariance KLT)
    *   **T-KLT** (Toeplitz KLT)
    It also includes helper methods for reading GUPPI files, extracting relevant channels (e.g., searching for the Voyager 1 signal), and plotting the eigenspectrum and Power Spectral Density (PSD).

*   **`KLT_test.ipynb`**: The main Jupyter notebook showing a practical usage example. It demonstrates how to use the `KLT` class to:
    1.  Find the most interesting frequency channel in a `.raw` file.
    2.  Apply the T-KLT and C-KLT variants.
    3.  Visualize the results (eigenspectrum and PSD).

*   **`data/`**: This folder contains the raw data files (`.raw` format) used for testing.

## Dependencies

The project relies on standard scientific Python libraries and specific astronomy tools. The main dependencies are listed in the `pyproject.toml` file and include:
*   `numpy` & `scipy`
*   `matplotlib`
*   `blimpy`
*   `tqdm`
*   `astropy`