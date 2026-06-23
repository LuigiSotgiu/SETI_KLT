# Real-World Data Analysis: The Voyager 1 Case Study

This directory contains the testing pipeline for applying the Karhunen-Loève Transform (KLT) to real radio astronomy data. Specifically, we analyze `GUPPI RAW` baseband data capturing the transmission from the Voyager 1 spacecraft, recorded by the Green Bank Telescope (GBT).

While synthetic signals provide a clean testing environment, real-world data introduces complex hardware and environmental challenges that test the robustness of the KLT algorithm.

---

## The Challenges of Real-World Data

Working with actual observatory data requires accounting for physical and digital anomalies that do not exist in ideal simulations:

### 1. Hardware-Induced Inversions
Radio telescope receiver chains use analog and digital downconverters. Depending on the mixing stages (e.g., in the ROACH2 FPGA boards at GBT), the frequency axis within a specific coarse channel might be inverted. To properly phase-align complex signals (I/Q data) and correct the polarity, it is sometimes necessary to apply a complex conjugation (`Q = -Q`). Our KLT projection logic natively respects complex matrix algebra rules to prevent mirroring artifacts during signal reconstruction.

### 2. Complex Noise Floors
Unlike the perfectly flat Gaussian noise generated in simulations, real observations contain dynamic variations in the noise floor, bandpass roll-off from the Polyphase Filterbank (PFB), and potential Radio Frequency Interference (RFI). 

---

## Multi-Component KLT & Telemetry

In our synthetic tests, we searched for a pure, unmodulated Continuous-Wave (CW) signal. A pure CW signal is mathematically highly correlated and can typically be captured entirely by retaining a single dominant eigenvector (`N_EIGENVECS = 1`).

Voyager 1, however, transmits a central carrier wave flanked by **telemetry sidebands** (phase-modulated data). 
If we only keep the first eigenvector, the KLT will isolate the carrier but might discard the data sidebands into the noise subspace. To reconstruct the complete signal (carrier + telemetry), we extend our analysis to a **Multi-Component KLT**, retaining the top $N$ eigenvectors (e.g., `N_EIGENVECS = 3` or more) that represent the broader signal subspace.

---

## Notebook Overview

### `CKLT_Voyager1.ipynb`
This notebook walks through the real-data extraction process:
1. **Data Loading:** We read the heavy `GUPPI RAW` file directly from the global `data/` directory, minimizing data duplication.
2. **Sanity Check:** We run a channel power scan to confirm the coarse channel containing the signal before running computationally expensive algorithms.
3. **Multi-Component CKLT:** We apply the Covariance KLT keeping the top 3 eigenvectors to preserve the spacecraft's modulation.
4. **Visualization:** We plot the eigenspectrum to observe the rank of the signal subspace, and we generate the waterfall and PSD plots to visually assess the denoising performance against the real GBT noise floor.