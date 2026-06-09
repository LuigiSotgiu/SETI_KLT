"""
klt.py
------
Karhunen-Loève Transform (KLT) utilities.

Supported variants
    MR-KLT  Multiple-Realization KLT   (2-D signal matrix)
    C-KLT   Covariance KLT             (1-D signal vector, windowed)
    T-KLT   Toeplitz KLT               (1-D signal vector)
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.linalg import toeplitz
from scipy.signal import find_peaks
from blimpy import GuppiRaw
from tqdm import tqdm
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class KLT:
    """Karhunen-Loève Transform with MR-KLT, C-KLT and T-KLT variants."""

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(self) -> None:
        self.signal: np.ndarray | None = None
        self.noise: np.ndarray | None = None
        self.eigenvalues: np.ndarray | None = None
        self.eigenvectors: np.ndarray | None = None
        self.covariance_matrix: np.ndarray | None = None
        self.reconstructed_signal: np.ndarray | None = None

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def load_signal_from_array(self, data: np.ndarray) -> None:
        """Load a signal from a NumPy array."""
        self.signal = data

    def load_noise_from_array(self, noise_data: np.ndarray) -> None:
        """Load a noise signal from a NumPy array."""
        self.noise = noise_data

    def find_best_channel_guppi(
        self,
        file_path: str,
        n_peaks_threshold: int = 3,
        sigma_threshold: float = 5.0,
    ) -> int:
        """Return the index of the most signal-rich channel in a GUPPI raw file.

        Primary criterion  - channel with at least *n_peaks_threshold* peaks
                             standing more than *sigma_threshold* above the
                             spectral mean.
        Fallback criterion - channel with the single highest spectral peak.

        Args:
            file_path:         Path to the GUPPI raw file.
            n_peaks_threshold: Minimum number of peaks to consider a channel.
            sigma_threshold:   Detection threshold in units of standard deviation.

        Returns:
            Index of the selected channel.
        """
        raw_file = GuppiRaw(file_path)
        header, data = raw_file.read_next_data_block()
        print(f"Raw data shape: {data.shape}  →  (channels, polarisations, samples)")

        best_channel = 0
        best_score = 0.0
        fallback_channel = 0
        fallback_peak = 0.0
        found_candidate = False

        for ch in range(data.shape[0]):
            spectrum = np.abs(np.fft.fft(data[ch, :, 0]))
            spectrum[0] = 0  # suppress DC

            # --- fallback: absolute maximum ---
            peak_value = float(np.max(spectrum))
            if peak_value > fallback_peak:
                fallback_peak = peak_value
                fallback_channel = ch

            # --- primary: multi-peak criterion ---
            threshold = np.mean(spectrum) + sigma_threshold * np.std(spectrum)
            peaks, _ = find_peaks(spectrum, height=threshold, distance=5)

            if len(peaks) >= n_peaks_threshold:
                score = float(np.sum(np.sort(spectrum[peaks])[-n_peaks_threshold:]))
                if score > best_score:
                    best_score = score
                    best_channel = ch
                    found_candidate = True

        if not found_candidate:
            print(
                f"No channel with {n_peaks_threshold} clear peaks found. "
                "Falling back to single-peak criterion."
            )
            best_channel = fallback_channel

        print(f"Best channel: {best_channel}")
        return best_channel

    def load_data_from_guppi(
        self,
        file_path: str,
        channel: int,
        num_samples: int,
    ) -> None:
        """Load, concatenate, and normalise samples from a GUPPI raw file.

        Args:
            file_path:   Path to the GUPPI raw file.
            channel:     Channel index to extract.
            num_samples: Total number of samples to load.
        """
        raw_file = GuppiRaw(file_path)
        blocks: list[np.ndarray] = []
        total = 0

        print("Starting sample extraction …")
        with tqdm(total=num_samples, desc="Progress", unit="samples", unit_scale=True) as pbar:
            while total < num_samples:
                try:
                    _, data = raw_file.read_next_data_block()
                    if data is None:
                        break
                    block = data[channel, :, 0].flatten()
                    blocks.append(block)
                    total += len(block)
                    pbar.update(len(block))
                except Exception:
                    break  # end of file

        vector = np.concatenate(blocks)[:num_samples]
        vector = (vector - vector.mean()) / vector.std()  # zero-mean, unit-variance

        self.signal = vector
        print(f"Loaded and normalised {len(self.signal):,} samples.")

    # -----------------------------------------------------------------------
    # KLT variants
    # -----------------------------------------------------------------------

    def apply_mrklt(self, n_eigenvectors: int = 1) -> None:
        """Multiple-Realization KLT (MR-KLT).

        The signal must be a 2-D array with shape (M realisations, N samples).

        Args:
            n_eigenvectors: Number of principal components kept for reconstruction.
        """
        self._require_signal()
        if self.signal.ndim != 2:
            raise ValueError("MR-KLT requires a 2-D signal matrix (M realisations × N samples).")

        data = self.signal if self.noise is None else self.signal + self.noise
        self.covariance_matrix = np.cov(data, rowvar=False)
        self._compute_eigensystem()

        mu = self.signal.mean(axis=0)
        V = self.eigenvectors[:, :n_eigenvectors]
        coefficients = (self.signal - mu) @ V
        self.reconstructed_signal = coefficients @ V.conj().T + mu

    def apply_cklt(self, window_size: int, n_eigenvectors: int = 1) -> None:
        """Covariance KLT (C-KLT).

        The signal must be a 1-D vector. It is segmented into non-overlapping
        windows of *window_size* samples to form a realisation matrix.

        Args:
            window_size:    Number of samples per window (W).
            n_eigenvectors: Number of principal components kept for reconstruction.
        """
        self._require_signal(ndim=1)

        W = window_size
        K = int(len(self.signal) // W)
        if K < 1:
            raise ValueError("Signal is too short for the requested window size.")

        data = self.signal if self.noise is None else self.signal + self.noise

        matrix = data[: K * W].reshape(K, W)
        self.covariance_matrix = np.cov(matrix, rowvar=False)
        self._compute_eigensystem()

        col_means = matrix.mean(axis=0)
        all_coefficients = (matrix - col_means) @ self.eigenvectors

        V = self.eigenvectors[:, :n_eigenvectors]
        kept_coefficients = all_coefficients[:, :n_eigenvectors]
        self.reconstructed_signal = (kept_coefficients @ V.T + col_means).flatten()

    def apply_tklt(self, n_eigenvectors: int = 1) -> None:
        """Toeplitz KLT (T-KLT).

        The signal must be a 1-D vector. The covariance matrix is approximated
        by a Toeplitz matrix built from the signal's autocorrelation.

        Args:
            n_eigenvectors: Number of principal components kept for reconstruction.
        """
        self._require_signal(ndim=1)

        data = self.signal if self.noise is None else self.signal + self.noise

        autocorr = self._autocorrelation(data)
        self.covariance_matrix = self._toeplitz_matrix(autocorr)
        self._compute_eigensystem()

        mu = data.mean()
        all_coefficients = (data - mu) @ self.eigenvectors

        V = self.eigenvectors[:, :n_eigenvectors]
        kept_coefficients = all_coefficients[:n_eigenvectors]
        self.reconstructed_signal = kept_coefficients @ V.T + mu

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------

    def plot_eigenspectrum(self, n_components: int = 1, save_path: str | None = None) -> tuple[plt.Figure, plt.Axes]:
        """Plot the sorted eigenspectrum, highlighting the kept components.

        Args:
            n_components: Number of eigenvalues highlighted as *kept*.
            save_path:    Optional file path to save the figure.

        Returns:
            figure: 
            axes: 
        """
        if self.eigenvalues is None:
            raise ValueError("No eigenvalues available. Apply a KLT method first.")

        fig, ax = plt.subplots()
        ax.scatter(
            np.arange(n_components, len(self.eigenvalues)),
            self.eigenvalues[n_components:],
            color="black", marker=".", label="Rejected eigenvalues",
        )
        ax.scatter(
            np.arange(n_components),
            self.eigenvalues[:n_components],
            color="red", marker=".", label=f"Kept eigenvalues ({n_components})",
        )
        ax.set(
            xlabel="Eigenvalue index (sorted descending)",
            ylabel="Eigenvalue magnitude",
            title="Eigenspectrum",
        )
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path)

        return fig, ax

    def plot_psd_comparison(self, fs: float = 1.0, save_path: str | None = None) -> tuple[plt.Figure, tuple[plt.Axes]]:
        """Plot the Power Spectral Density of the original vs. reconstructed signal.

        Args:
            fs:        Sampling frequency used for the PSD calculation.
            save_path: Optional file path to save the figure.

        Returns:
            figure: 
            axes: 
        """
        if self.signal is None or self.reconstructed_signal is None:
            raise ValueError("Both original and reconstructed signals must be available.")

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

        data = self.signal if self.noise is None else self.signal + self.noise

        axes[0].psd(data, NFFT=1024, Fs=fs, color="navy", label="Original")
        axes[1].psd(self.reconstructed_signal, NFFT=1024, Fs=fs, color="red", label="Reconstructed")

        for ax, title in zip(axes, ("Original signal PSD", "Reconstructed signal PSD")):
            ax.set_title(title)
            ax.set_ylabel("PSD [dB/Hz]")
            ax.legend()
            ax.grid(True)
        axes[1].set_xlabel("Frequency")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)

        return fig, axes

    # -----------------------------------------------------------------------
    # Getters
    # -----------------------------------------------------------------------

    def get_reconstructed_signal(self) -> np.ndarray | None:
        """Return the reconstructed signal."""
        return self.reconstructed_signal

    def get_eigenvalues(self) -> np.ndarray | None:
        """Return the sorted eigenvalues."""
        return self.eigenvalues

    def get_eigenvectors(self) -> np.ndarray | None:
        """Return the sorted eigenvectors."""
        return self.eigenvectors

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _require_signal(self, ndim: int | None = None) -> None:
        """Raise if no signal is loaded, or if its dimensionality is wrong."""
        if self.signal is None:
            raise ValueError("No signal loaded. Call load_signal_from_array() first.")
        if ndim is not None and self.signal.ndim != ndim:
            raise ValueError(f"Expected a {ndim}-D signal, got {self.signal.ndim}-D.")
        if self.noise is not None and self.signal.ndim != ndim:
            raise ValueError("Signal and noise must have the same dimensionality.")

    def _compute_eigensystem(self) -> None:
        """Compute and sort eigenvalues / eigenvectors from the covariance matrix."""
        eigenvalues, eigenvectors = linalg.eigh(self.covariance_matrix)
        order = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[order]
        self.eigenvectors = eigenvectors[:, order]

    def _autocorrelation(self, vector: np.ndarray) -> np.ndarray:
        """Return the non-negative-lag autocorrelation of *vector*."""
        full = np.correlate(vector, vector, mode="full")
        return full[full.size // 2:]

    def _toeplitz_matrix(self, autocorr: np.ndarray) -> np.ndarray:
        """Build a normalised Toeplitz matrix from an autocorrelation vector."""
        normalised = autocorr / autocorr[0]
        return toeplitz(normalised)