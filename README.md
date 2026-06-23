# SETI KLT Project: Signal Denoising with Karhunen-Loève Transform

This repository contains a Python framework developed during an internship to apply various Karhunen-Loève Transform (KLT) methodologies to radio astronomy data. The primary goal of this project is to process and analyze time-voltage `GUPPI RAW` files (commonly used in SETI research) to test the extraction and reconstruction of weak continuous-wave (CW) signals submerged in thermal noise.

## Background: Why KLT?

In traditional radio astronomy, signals are often detected by accumulating power in frequency bins over time (using FFTs). However, if a signal is extremely weak, the variance of the thermal noise can mask it completely.

This project explores the **Karhunen-Loève Transform (KLT)** as an alternative algorithmic denoising step. Instead of looking directly at frequencies, the KLT analyzes the data in the *covariance domain*. By calculating the covariance matrix of the incoming voltage time-series and performing an Eigendecomposition, the KLT separates the data into:
1. **The Signal Subspace:** Captured by the dominant eigenvalues and eigenvectors (correlated signals).
2. **The Noise Subspace:** The remaining minor eigenvalues (uncorrelated thermal noise).

### Implemented KLT Variants
The core `KLT` class currently tests three structural approaches:
* **C-KLT (Covariance KLT):** Segments a 1D baseband stream into 2D windows to build the spatial covariance matrix.
* **T-KLT (Toeplitz KLT):** Builds a deterministic Toeplitz matrix from the 1D autocorrelation function of the signal.
* **MR-KLT (Multiple-Realization KLT):** Designed for multi-antenna or multi-polarization arrays where data is inherently 2D.

---

### Deep Dive: Covariance KLT (C-KLT) Framework

Following the formulation analyzed in our reference paper (*Trudu et al., 2020*), the C-KLT models a 1D discrete time-series voltage signal $x[n]$ by projecting it onto an adaptive, data-driven orthogonal basis rather than using static trigonometric kernels like the Fourier Transform.

The algorithmic pipeline implemented in this package follows four mathematical steps:

1. **Trajectory Matrix Formulation** The 1D complex baseband voltage vector of total length $N$ is segmented into $K$ blocks of length $W$ (where $W$ represents the `WINDOW_SIZE`). These blocks are arranged as rows to build the trajectory data matrix $X \in \mathbb{C}^{K \times W}$:
   $$X = \begin{bmatrix} 
   x[0] & x[1] & \cdots & x[W-1] \\ 
   x[W] & x[W+1] & \cdots & x[2W-1] \\ 
   \vdots & \vdots & \ddots & \vdots \\
   x[(K-1)W] & x[(K-1)W+1] & \cdots & x[KW-1] 
   \end{bmatrix}$$

2. **Covariance Matrix Estimation** The empirical spatial covariance matrix $R_{xx} \in \mathbb{C}^{W \times W}$ is estimated from the rows of $X$. In the standard C-KLT implementation, data is centered by subtracting the column-wise mean vector $\mu_X$ to isolate purely stochastic and alternating components:
   $$R_{xx} = \frac{1}{K} (X - \mu_X)^H (X - \mu_X)$$

3. **Eigendecomposition & Subspace Separation** We solve the characteristic equation $R_{xx}V = V\Lambda$, where $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_W)$ contains the real-valued eigenvalues sorted in descending order ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_W$). 
   * **Signal Subspace:** The first $M$ dominant eigenvalues ($\lambda_1 \dots \lambda_M$) capture the deterministic, highly-correlated energy of the cosmic or artificial carrier wave.
   * **Noise Subspace:** The remaining $W - M$ eigenvalues form a flat plateau representing the uncorrelated variance ($\sigma^2$) of the thermal noise floor.

4. **Low-Rank Truncated Reconstruction** By selecting only the sub-matrix of primary eigenvectors $V_M \in \mathbb{C}^{W \times M}$, the centered data matrix is projected onto the signal subspace and transformed back into the time domain. Re-adding the mean vector yields the denoised time-series matrix $\hat{X}$, which is then flattened back into a clean 1D baseband signal.

*Note: The implementation of **T-KLT** (Toeplitz Autocorrelation) and **MR-KLT** (Multiple-Realization for multi-feed receiver arrays) are currently flagged as Work in Progress (WIP) and will be documented here upon baseline validation.*

---

## Repository Structure

The project is structured as an installable Python package (`seti_klt`) accompanied by practical test notebooks. 

```text
.
├── pyproject.toml              # Package definition and dependencies manager (via uv)
├── setigen_npol_fix.patch      # Essential patch for the Setigen library (NPOL header fix)
│
├── src/seti_klt/               # CORE PACKAGE
│   ├── KLT.py                  # The main KLT class and algorithmic core
│   └── utils/                  # I/O utilities, dynamic paths, and structured loggers
│
├── notebooks/                  # TESTS & ANALYSIS
│   ├── injected_signals/       # Synthetic signal tests 
│   └── voyager_1/              # Real-world telemetry extraction tests from GBT observations
│
├── data/                       # RAW DATA (Ignored in Git, local heavy files)
└── outputs/                    # RESULTS (Locally generated figures, logs, and raw cuts)
```

> **Note:** For detailed explanations on specific tests (like the "Synchronous Phase" phenomenon or Voyager 1 decoding), please refer to the specific `README.md` files located inside the `notebooks/` subdirectories.

---

## Installation & Setup

This project uses `uv` (or standard `pip` + `venv`) for dependency management. To ensure paths resolve correctly and local imports work inside the notebooks, the project must be installed in **editable mode**.

**1. Clone the repository:**
```bash
git clone [https://github.com/LuigiSotgiu/seti_klt.git](https://github.com/LuigiSotgiu/seti_klt.git)
cd seti_klt
```

**2. Apply the Setigen patch:**
*Note: A bug in the `setigen` library handles the `NPOL` header incorrectly when generating GUPPI outputs. Apply our patch to your local setigen folder before generating synthetic files.*
```bash
# Assuming you cloned setigen next to this repo or downloaded it locally
cd path/to/local/setigen
git apply path/to/seti_klt/setigen_npol_fix.patch
```

**3. Install the environment and the local package:**
```bash
# Install the patched setigen locally
uv pip install -e path/to/local/setigen

# Install this package (seti_klt) in editable mode
uv pip install -e .
```

---

## Quick Start Example

Once installed, you can use the package in your scripts or Jupyter Notebooks seamlessly:

```python
from seti_klt.KLT import KLT
from seti_klt.utils.io import get_data_path

# 1. Initialize the KLT pipeline
klt = KLT()

# 2. Load Raw Voltage Data
data_path = get_data_path() / "blc3_guppi_57386_VOYAGER1_0004.0000.raw"
klt.load_data_from_guppi(
    file_path=str(data_path), 
    channel=0, 
    num_samples=2**20, 
    polarization=0
)

# 3. Apply Covariance KLT
klt.apply_cklt(window_size=1024, n_eigenvectors=3)

# 4. Extract and visualize results
klt.plot_eigenspectrum(n_components=10)
klt.plot_waterfall_comparison(fft_length=1024)
```