# Synthetic Signal Analysis & The Synchronous Phase Phenomenon

This directory contains notebooks dedicated to testing the Karhunen-Loève Transform (KLT) in a controlled environment. By using the `setigen` library, we generate synthetic radio frequency (RF) data, inject artificial continuous-wave (CW) signals at specific Signal-to-Noise Ratios (SNR), and evaluate the KLT's ability to extract them.

Beyond standard testing, this folder serves as an exploration of a fundamental mathematical challenge encountered when applying spatial decomposition techniques to time-series data: **The Synchronous Phase Phenomenon**.

---

## 📖 Theoretical Background

To understand the tests in this directory, it is helpful to understand how 1D time-series data is processed by the Covariance KLT (C-KLT).

### 1. From 1D to 2D: The Windowing Process
A radio telescope records a single, long 1D array of voltage samples over time. However, the KLT (like Principal Component Analysis, PCA) requires a 2D matrix (Observations $\times$ Variables) to compute a covariance matrix.

To solve this, the 1D voltage stream is "chopped" into consecutive segments of length $W$ (the `WINDOW_SIZE`). These segments are stacked vertically to form a 2D matrix:
* **Rows ($K$):** Different time observations (Realizations).
* **Columns ($W$):** Time steps within a single window.

### 2. Centering the Data
To calculate the statistical **Covariance Matrix**, standard linear algebra requires the data to be "centered". This is done by calculating the mean of each column across all rows, and subtracting this mean array from every row in the matrix:
```text
Centered_Matrix = Matrix - Column_Means
```

---

## ⚠️ The Synchronous Phase Problem

During our internship testing, we encountered a strange anomaly: when a very strong synthetic signal was injected at specific frequencies, the KLT completely failed to detect it. The dominant eigenvalue collapsed, and the reconstructed signal was just noise. Why?

The root cause lies in how continuous waves interact with the windowing process.

Imagine a baseband sine wave signal. If the frequency of this signal $f_0$ perfectly aligns with the window size $W$, such that the signal completes an **exact integer number of cycles** within one window (e.g., $f_0 \cdot W = \text{integer}$), a destructive phenomenon occurs:

1. **Phase Reset:** Because the window length matches the period of the signal perfectly, the phase of the sine wave restarts at the exact same point at the beginning of *every single row*.
2. **Identical Rows:** For the signal component, Row 1 is identical to Row 2, which is identical to Row 3, and so on.
3. **The Trap of the Mean:** When the algorithm calculates the `Column_Means`, it averages these identical rows. The random thermal noise averages out (towards zero), but the signal survives perfectly intact inside the mean vector.
4. **Signal Erasure:** When the algorithm subtracts the `Column_Means` from the matrix to center it, **it subtracts the signal out of the matrix completely**.

The resulting centered matrix contains only random thermal noise. The KLT models the noise, extracts the noise, and fails to see the signal. We call this the **Synchronous Phase Phenomenon**.

### The Solution: Fractional Offsets
To prevent the KLT from erasing the signal during the centering phase, we must ensure that the phase of the signal *rotates* from one row to the next. 

We achieve this by injecting our synthetic signals using a **fractional offset** relative to the frequency bins (e.g., `BASEBAND_OFFSET_FRAC = 0.3`). Because $0.3 \times W$ does not yield an integer, the signal's phase shifts slightly with every new row. Consequently, the `Column_Means` destructively interfere and average to near-zero, leaving the pure signal safely inside the matrix for the KLT to discover.

---

## 📓 Notebook Overview

### `injected_signal.ipynb`
This is the main testing ground for synthetic data. In this notebook, you will see the theory put into practice:
1. **Generation:** We simulate a Polyphase Filterbank (PFB) output containing thermal noise and a weak CW signal.
2. **Injection Strategy:** We explicitly use fractional offsets (like `0.3`) to bypass the synchronous phase problem.
3. **Diagnostics:** We verify the signal's presence using a broadband power scan.
4. **KLT Application:** We run the C-KLT, extract the principal eigenvector, and plot the eigenspectrum, dynamic waterfall, and integrated Power Spectral Density (PSD) to measure the SNR improvement.