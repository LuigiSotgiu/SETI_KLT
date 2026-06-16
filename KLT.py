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
        print(f"Raw data shape: {data.shape}  →  (channels, samples, polarisations)")

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
    
    def find_channel_by_frequency(
        self, 
        file_path: str, 
        target_freq_mhz: float
    ) -> int:
        """Return the index of the channel containing the specified frequency.

        Reads the GUPPI file header to extract the center frequency (OBSFREQ),
        total bandwidth (OBSBW), and number of channels (OBSNCHAN) to calculate
        the boundaries of each channel and map the input frequency.

        Args:
            file_path:       Path to the GUPPI raw file.
            target_freq_mhz: Target frequency in MHz.

        Returns:
            Index of the selected channel. Raises ValueError if the frequency is out of bounds.
        """
        raw_file = GuppiRaw(file_path)
        header, _ = raw_file.read_next_data_block()

        # Extract parameters from the header
        try:
            obsfreq = float(header['OBSFREQ'])
            obsbw = float(header['OBSBW'])
            obsnchan = int(header['OBSNCHAN'])
        except KeyError as e:
            raise KeyError(f"Missing key in GUPPI header: {e}")

        # Single channel bandwidth
        chan_bw = obsbw / obsnchan

        # Calculate the starting frequency of the band (can be lower or upper edge)
        # In GUPPI, OBSFREQ represents the exact center of the total band (OBSBW)
        f_start = obsfreq - (obsbw / 2.0)
        f_end = obsfreq + (obsbw / 2.0)

        # Check if the frequency is within the recorded band
        min_freq = min(f_start, f_end)
        max_freq = max(f_start, f_end)

        if not (min_freq <= target_freq_mhz <= max_freq):
            raise ValueError(
                f"Frequency {target_freq_mhz} MHz is out of the observed band "
                f"({min_freq:.2f} MHz - {max_freq:.2f} MHz)."
            )

        # Calculate the channel index based on the bandwidth sign
        if chan_bw > 0:
            # Ascending band
            channel = int((target_freq_mhz - f_start) // chan_bw)
        else:
            # Descending band (chan_bw < 0, f_start is the maximum frequency)
            channel = int((f_start - target_freq_mhz) // abs(chan_bw))

        # Safety check for extreme edges (prevent out-of-bounds)
        channel = max(0, min(channel, obsnchan - 1))

        print(f"Target Frequency : {target_freq_mhz} MHz")
        print(f"GUPPI Header Data: Center Freq = {obsfreq} MHz, Total BW = {obsbw} MHz, Channels = {obsnchan}")
        print(f"Calculated Channel: {channel}")

        return channel

    def get_band_start_freq(self, file_path: str) -> tuple[float, float]:
        """Return (f_start_hz, chan_bw_hz) for the recorded GUPPI band.

        Args:
            file_path: Path to the GUPPI raw file.

        Returns:
            (f_start_hz, chan_bw_hz): lower edge of the recorded band and
            the per-channel bandwidth, both in Hz.
        """
        raw_file = GuppiRaw(file_path)
        header, _ = raw_file.read_next_data_block()

        obsfreq = float(header['OBSFREQ']) * 1e6  # MHz -> Hz
        obsbw = float(header['OBSBW']) * 1e6       # MHz -> Hz
        obsnchan = int(header['OBSNCHAN'])

        chan_bw = obsbw / obsnchan
        f_start = obsfreq - obsbw / 2.0

        return f_start, chan_bw

    def load_data_from_guppi(
        self,
        file_path: str,
        channel: int,
        num_samples: int,
        polarization: int,
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
                except StopIteration:
                    print("get_data: end of file reached (StopIteration).")
                    break
                except Exception as exc:
                    raise RuntimeError(
                        f"read_next_data_block() failed after reading "
                        f"{len(blocks)} block(s) ({total} samples so far). "
                        f"Underlying error: {type(exc).__name__}: {exc}\n"
                        f"Common causes: channel={channel} or "
                        f"polarization={polarization} out of range for this "
                        f"file's actual data shape, or a corrupted/incomplete "
                        f"GUPPI file."
                    ) from exc

                if data is None:
                    print("get_data: no more data blocks (data is None).")
                    break

                if channel >= data.shape[0]:
                    raise IndexError(
                        f"channel={channel} is out of range: this file's "
                        f"data blocks have shape {data.shape} "
                        f"(channels, samples, polarizations), i.e. only "
                        f"{data.shape[0]} channel(s) available (0..{data.shape[0]-1})."
                    )
                if data.ndim >= 3 and polarization >= data.shape[2]:
                    raise IndexError(
                        f"polarization={polarization} is out of range: this "
                        f"file's data blocks have shape {data.shape}, i.e. "
                        f"only {data.shape[2]} polarization(s) available "
                        f"(0..{data.shape[2]-1})."
                    )

                block = data[channel, :, polarization].flatten()
                blocks.append(block)
                total += len(block)
                pbar.update(len(block))

        if not blocks:
            raise RuntimeError(
                f"No data blocks were read from '{file_path}'. The file may "
                f"be empty, truncated, or the header could not be parsed. "
                f"Check that the .raw file was generated/recorded successfully."
            )

        vector = np.concatenate(blocks)[:num_samples]
        vector = (vector - vector.mean()) / vector.std()  # zero-mean, unit-variance

        self.signal = vector

        #TODO: WHY THIS DOESN'T WORKS?
        # raw_data = np.concatenate(blocks)
        # raw_data = raw_data[:(len(raw_data) // 2) * 2]
        # raw_data = raw_data.astype(np.float32)
        # vector_complex = raw_data[0::2] + 1j * raw_data[1::2]
        # vector_complex = vector_complex[:num_samples]
        # vector_complex = (vector_complex - vector_complex.mean()) / vector_complex.std()
        
        # self.signal = vector_complex

        print(f"Loaded and normalised {len(self.signal):,} samples.")

    # -----------------------------------------------------------------------
    # KLT variants
    # -----------------------------------------------------------------------

    def apply_mrklt(self, n_eigenvectors: int = 1, method: str = "auto") -> None:
        """Multiple-Realization KLT (MR-KLT).

        The signal must be a 2-D array with shape (M realisations, N samples).

        Args:
            n_eigenvectors: Number of principal components kept for reconstruction.
            method:         'auto', 'cov', or 'svd'. 'auto' uses SVD if N > M and N > 5000.
        """
        self._require_signal()
        if self.signal.ndim != 2:
            raise ValueError("MR-KLT requires a 2-D signal matrix (M realisations × N samples).")

        data = self.signal if self.noise is None else self.signal + self.noise
        self._klt_input = data  # saved for plotting
        
        M, N = data.shape
        
        if method == "auto":
            method = "svd" if (N > M and N > 5000) else "cov"
            
        if method == "svd":
            self._compute_eigensystem_svd(data)
        elif method == "cov":
            self.covariance_matrix = np.cov(data, rowvar=False)
            self._compute_eigensystem()
        else:
            raise ValueError("method must be 'auto', 'cov', or 'svd'")

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
        self._klt_input = data  # saved for plotting

        matrix = data[: K * W].reshape(K, W)
        self.covariance_matrix = np.cov(matrix, rowvar=False)
        self._compute_eigensystem()

        col_means = matrix.mean(axis=0)         
        all_coefficients = (matrix - col_means) @ self.eigenvectors   

        V = self.eigenvectors[:, :n_eigenvectors]                      
        kept_coefficients = all_coefficients[:, :n_eigenvectors]       
        self.reconstructed_signal = (kept_coefficients @ V.conj().T + col_means).flatten()

    def apply_tklt(self, n_eigenvectors: int = 1) -> None:
        """Toeplitz KLT (T-KLT).

        The signal must be a 1-D vector. The covariance matrix is approximated
        by a Toeplitz matrix built from the signal's autocorrelation.

        Args:
            n_eigenvectors: Number of principal components kept for reconstruction.
        """
        self._require_signal(ndim=1)

        data = self.signal if self.noise is None else self.signal + self.noise
        self._klt_input = data  # saved for plotting

        autocorr = self._autocorrelation(data)
        self.covariance_matrix = self._toeplitz_matrix(autocorr)
        self._compute_eigensystem()

        mu = data.mean()
        # all_coefficients has shape (N,) — one coefficient per eigenvector
        all_coefficients = (data - mu) @ self.eigenvectors   # shape (N,)

        V = self.eigenvectors[:, :n_eigenvectors]            # shape (N, n)
        kept_coefficients = all_coefficients[:n_eigenvectors]  # shape (n,)

        self.reconstructed_signal = kept_coefficients[np.newaxis, :] @ V.conj().T
        self.reconstructed_signal = self.reconstructed_signal.squeeze() + mu

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

        # _klt_input is set by apply_cklt / apply_tklt / apply_mrklt.
        data = getattr(self, '_klt_input', None)
        if data is None:
            data = self.signal if self.noise is None else self.signal + self.noise
        recon = self.reconstructed_signal


        from scipy.signal import welch

        nperseg = min(1024, len(data))
        freqs_in, psd_in = welch(data, fs=fs, nperseg=nperseg, return_onesided=False)
        freqs_in = np.fft.fftshift(freqs_in)
        psd_in = np.fft.fftshift(psd_in)

        nperseg_r = min(1024, len(recon))
        freqs_rec, psd_rec = welch(recon, fs=fs, nperseg=nperseg_r, return_onesided=False)
        freqs_rec = np.fft.fftshift(freqs_rec)
        psd_rec = np.fft.fftshift(psd_rec)

        axes[0].plot(freqs_in, 10 * np.log10(psd_in + 1e-20), color="navy", label="Input (noisy)")
        axes[1].plot(freqs_rec, 10 * np.log10(psd_rec + 1e-20), color="red", label="Reconstructed")

        for ax, title in zip(axes, ("Input signal PSD (noisy)", "Reconstructed signal PSD (KLT)")):
            ax.set_title(title)
            ax.set_ylabel("PSD [dB/Hz]")
            ax.legend()
            ax.grid(True)
        axes[1].set_xlabel("Frequency [Hz]" if fs != 1.0 else "Frequency (normalized, cycles/sample)")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)

        return fig, axes

    def plot_waterfall_comparison(self, fft_length: int = 1024, cmap: str = 'viridis') -> None:
        """Plot the waterfall (dynamic spectrum) of the original and reconstructed signals.

        Calculates the spectrograms using a Short-Time Fourier Transform (STFT) approach.
        It divides the 1D complex voltage signals into chunks of size `fft_length`,
        computes the FFT for each chunk, and plots the side-by-side comparison.

        Args:
            fft_length: Length of the FFT window. Defines the frequency resolution
                        and the number of time bins (integrations).
            cmap:       Colormap to use for the spectrogram images.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if self.signal is None or self.reconstructed_signal is None:
            raise ValueError(
                "Both signal and reconstructed_signal must be loaded and computed "
                "before plotting."
            )

        def compute_waterfall(data: np.ndarray, nfft: int) -> np.ndarray:
            """Helper function to compute 2D power spectrum from 1D voltages."""
            data_1d = data.flatten()
            num_spectra = len(data_1d) // nfft
            truncated_data = data_1d[:num_spectra * nfft]
            reshaped_data = truncated_data.reshape((num_spectra, nfft))
            spectra = np.fft.fftshift(np.fft.fft(reshaped_data, axis=1), axes=1)
            power = 10 * np.log10(np.abs(spectra)**2 + 1e-10)
            return power

        input_data = getattr(self, '_klt_input', None)
        if input_data is None:
            input_data = self.signal if self.noise is None else self.signal + self.noise

        wf_original = compute_waterfall(input_data, fft_length)
        wf_reconstructed = compute_waterfall(self.reconstructed_signal, fft_length)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)

        im_kwargs = {
            'aspect': 'auto',
            'origin': 'upper',
            'cmap': cmap,
            'interpolation': 'nearest'
        }

        im1 = axes[0].imshow(wf_original, **im_kwargs)
        axes[0].set_title('Input Signal Waterfall (noisy)')
        axes[0].set_xlabel('Frequency Bins')
        axes[0].set_ylabel('Time (Integrations)')
        fig.colorbar(im1, ax=axes[0], label='Power (dB)')

        im2 = axes[1].imshow(wf_reconstructed, **im_kwargs)
        axes[1].set_title('Reconstructed Signal (KLT)')
        axes[1].set_xlabel('Frequency Bins')
        fig.colorbar(im2, ax=axes[1], label='Power (dB)')

        plt.tight_layout()

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

    def _compute_eigensystem_svd(self, data: np.ndarray) -> None:
        """Compute and sort eigenvalues / eigenvectors using SVD directly on data."""
        M = data.shape[0]
        mu_data = data.mean(axis=0)
        U, S, Vt = linalg.svd(data - mu_data, full_matrices=False)
        self.covariance_matrix = None  # Skipped to save memory
        self.eigenvalues = (S ** 2) / max(M - 1, 1)
        self.eigenvectors = Vt.conj().T

    def _autocorrelation(self, vector: np.ndarray) -> np.ndarray:
        """Return the non-negative-lag autocorrelation of *vector*."""
        full = np.correlate(vector, vector, mode="full")
        return full[full.size // 2:]

    def _toeplitz_matrix(self, autocorr: np.ndarray) -> np.ndarray:
        """Build a normalised Toeplitz matrix from an autocorrelation vector."""
        normalised = autocorr / autocorr[0]
        return toeplitz(normalised)