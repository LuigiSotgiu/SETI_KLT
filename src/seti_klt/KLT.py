"""
KLT.py
------
Karhunen-Loève Transform (KLT) utilities for radio-astronomy signal processing.

Supported variants
------------------
C-KLT   Covariance KLT   — 1-D signal vector, segmented into non-overlapping windows.
MR-KLT  Multiple-Realization KLT — 2-D signal matrix  (commented out, in progress).
T-KLT   Toeplitz KLT     — 1-D signal vector           (commented out, in progress).
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.linalg import toeplitz
from blimpy import GuppiRaw
from tqdm import tqdm
import matplotlib.pyplot as plt
import setigen as stg
import setigen.voltage.raw_utils as stg_raw_utils

from .utils.io import SimpleLogger, LogLevel


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class KLT:
    """Karhunen-Loève Transform with C-KLT, MR-KLT, and T-KLT variants.

    The class follows a load → apply → inspect workflow:

    1. Load a signal with :meth:`load_signal_from_array` or
       :meth:`load_data_from_guppi`.
    2. Run a KLT variant (e.g. :meth:`apply_cklt`).
    3. Retrieve results via the getter methods or generate diagnostic plots.

    A :class:`~seti_klt.utils.io.SimpleLogger` instance can be injected at
    construction time to control verbosity.  When no logger is provided the
    class creates an INFO-level console-only logger automatically, so all
    informational messages are still visible without any extra setup.

    Args:
        logger: Optional logger instance.  Pass ``None`` (default) to use a
                lightweight console logger at INFO level, or supply your own
                :class:`SimpleLogger` for fine-grained control over level and
                file output.

    Example::

        from seti_klt import KLT
        from seti_klt.utils import SimpleLogger, LogLevel

        logger = SimpleLogger(level=LogLevel.DEBUG)
        klt = KLT(logger=logger)
        klt.load_data_from_guppi("obs.raw", "obs", channel=1, num_samples=2**19, polarization=0)
        klt.apply_cklt(window_size=512, n_eigenvectors=1)
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(self, logger: SimpleLogger | None = None) -> None:
        # Internal state
        self.info: dict | None = None
        self.signal: np.ndarray | None = None
        self.noise: np.ndarray | None = None
        self.eigenvalues: np.ndarray | None = None
        self.eigenvectors: np.ndarray | None = None
        self.covariance_matrix: np.ndarray | None = None
        self.reconstructed_signal: np.ndarray | None = None
        self._klt_input: np.ndarray | None = None

        # Logger — fall back to a minimal console-only logger if none given
        if logger is None:
            self._logger = SimpleLogger(
                level=LogLevel.INFO,
                console_output=True,
                file_output=False,
            )
        else:
            self._logger = logger

        self._logger.debug("KLT instance created.")

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def load_signal_from_array(self, data: np.ndarray) -> None:
        """Load a signal directly from a NumPy array.

        Args:
            data: 1-D (C-KLT / T-KLT) or 2-D (MR-KLT) signal array.
        """
        self.signal = data
        self._logger.info(f"Signal loaded from array — shape: {data.shape}, dtype: {data.dtype}.")
        self._logger.debug(
            f"Signal stats — min: {data.min():.4f}, max: {data.max():.4f}, "
            f"mean: {data.mean():.4f}, std: {data.std():.4f}."
        )

    def load_noise_from_array(self, noise_data: np.ndarray) -> None:
        """Load a noise realisation from a NumPy array.

        When a noise array is present it is added to the signal before any
        KLT variant is applied, simulating the noisy observation.

        Args:
            noise_data: Noise array with the same shape as the signal.
        """
        self.noise = noise_data
        self._logger.info(
            f"Noise loaded from array — shape: {noise_data.shape}, dtype: {noise_data.dtype}."
        )
        self._logger.debug(
            f"Noise stats — std: {noise_data.std():.4f}, "
            f"min: {noise_data.min():.4f}, max: {noise_data.max():.4f}."
        )

    def find_channel_by_frequency(
        self,
        file_path: str,
        target_freq_mhz: float,
    ) -> int:
        """Return the index of the channel containing *target_freq_mhz*.

        Reads the GUPPI file header to extract the centre frequency
        (``OBSFREQ``), total bandwidth (``OBSBW``), and channel count
        (``OBSNCHAN``), then maps the target frequency to the corresponding
        channel index.

        Args:
            file_path:       Path to the GUPPI raw file.
            target_freq_mhz: Target frequency in MHz.

        Returns:
            Zero-based index of the channel containing the target frequency.

        Raises:
            KeyError:   If a required header keyword is missing.
            ValueError: If *target_freq_mhz* falls outside the recorded band.
        """
        self._logger.debug(f"Opening '{file_path}' to locate channel for {target_freq_mhz:.4f} MHz.")
        raw_file = GuppiRaw(file_path)
        header, _ = raw_file.read_next_data_block()

        try:
            obsfreq  = float(header["OBSFREQ"])
            obsbw    = float(header["OBSBW"])
            obsnchan = int(header["OBSNCHAN"])
        except KeyError as exc:
            raise KeyError(f"Missing key in GUPPI header: {exc}") from exc

        chan_bw  = obsbw / obsnchan
        f_start  = obsfreq - obsbw / 2.0
        f_end    = obsfreq + obsbw / 2.0
        min_freq = min(f_start, f_end)
        max_freq = max(f_start, f_end)

        self._logger.debug(
            f"Header — OBSFREQ: {obsfreq:.4f} MHz, OBSBW: {obsbw:.4f} MHz, "
            f"OBSNCHAN: {obsnchan}, chan_bw: {chan_bw:.6f} MHz."
        )
        self._logger.debug(f"Recorded band: [{min_freq:.4f}, {max_freq:.4f}] MHz.")

        if not (min_freq <= target_freq_mhz <= max_freq):
            raise ValueError(
                f"Frequency {target_freq_mhz} MHz is outside the recorded band "
                f"({min_freq:.2f} – {max_freq:.2f} MHz)."
            )

        if chan_bw > 0:
            channel = int((target_freq_mhz - f_start) // chan_bw)
        else:
            channel = int((f_start - target_freq_mhz) // abs(chan_bw))

        channel = max(0, min(channel, obsnchan - 1))

        self._logger.info(
            f"Target {target_freq_mhz:.4f} MHz → channel {channel} "
            f"(centre: {obsfreq + channel * chan_bw:.4f} MHz)."
        )
        return channel

    def get_band_start_freq(self, file_path: str) -> tuple[float, float]:
        """Return the lower-band edge and per-channel bandwidth for a GUPPI file.

        Args:
            file_path: Path to the GUPPI raw file.

        Returns:
            ``(f_start_hz, chan_bw_hz)``: lower edge of the recorded band and
            the per-channel bandwidth, both expressed in Hz.
        """
        self._logger.debug(f"Reading band parameters from '{file_path}'.")
        raw_file = GuppiRaw(file_path)
        header, _ = raw_file.read_next_data_block()

        obsfreq  = float(header["OBSFREQ"]) * 1e6   # MHz → Hz
        obsbw    = float(header["OBSBW"])   * 1e6   # MHz → Hz
        obsnchan = int(header["OBSNCHAN"])

        chan_bw = obsbw / obsnchan
        f_start = obsfreq - obsbw / 2.0

        self._logger.info(
            f"Band: f_start = {f_start / 1e6:.4f} MHz, "
            f"chan_bw = {chan_bw / 1e6:.6f} MHz ({obsnchan} channels)."
        )
        self._logger.debug(
            f"Raw header values — OBSFREQ: {obsfreq/1e6:.4f} MHz, "
            f"OBSBW: {obsbw/1e6:.4f} MHz, OBSNCHAN: {obsnchan}."
        )
        return f_start, chan_bw

    def load_data_from_guppi(
        self,
        file_path: str,
        file_stem: str,
        channel: int,
        num_samples: int,
        polarization: int,
        standard_scaler: bool = False,
    ) -> None:
        """Load, concatenate, and optionally normalise samples from a GUPPI raw file.

        Reads data blocks sequentially until *num_samples* samples have been
        collected from the specified *channel* and *polarization*.  The result
        is stored in :attr:`signal`.

        Args:
            file_path:       Path to the GUPPI ``.raw`` file.
            file_stem:       File stem passed to ``setigen`` to retrieve raw
                             parameters (e.g. ``"obs"`` for ``obs.0000.raw``).
            channel:         Zero-based channel index to extract.
            num_samples:     Total number of time samples to load.
            polarization:    Polarization index (0 or 1 for dual-pol files).
            standard_scaler: When ``True``, the concatenated vector is
                             zero-mean / unit-variance normalised before storage.

        Raises:
            IndexError:   If *channel* or *polarization* is out of range for
                          the file's data shape.
            RuntimeError: If the file is empty or no blocks could be read.
        """
        self._logger.section("Loading GUPPI data")
        self._logger.info(
            f"File: '{file_path}' | channel: {channel} | "
            f"polarization: {polarization} | requested samples: {num_samples:,}."
        )

        raw_file = GuppiRaw(file_path)
        blocks: list[np.ndarray] = []
        total  = 0
        header = None
        params = stg_raw_utils.get_raw_params(file_stem, start_chan=0)

        self._logger.debug(
            f"setigen raw params — fch1: {params.get('fch1', 'N/A')}, "
            f"chan_bw: {params.get('chan_bw', 'N/A')}."
        )

        with tqdm(total=num_samples, desc="Reading blocks", unit="samp", unit_scale=True) as pbar:
            while total < num_samples:
                try:
                    header, data = raw_file.read_next_data_block()
                except StopIteration:
                    self._logger.warning(
                        f"End of file reached after {len(blocks)} block(s) "
                        f"({total:,} samples). Requested {num_samples:,}."
                    )
                    break
                except Exception as exc:
                    raise RuntimeError(
                        f"read_next_data_block() failed after {len(blocks)} block(s) "
                        f"({total:,} samples). "
                        f"Underlying error: {type(exc).__name__}: {exc}\n"
                        f"Common causes: channel={channel} or polarization={polarization} "
                        f"out of range, or a corrupted / incomplete GUPPI file."
                    ) from exc

                if data is None:
                    self._logger.warning("read_next_data_block() returned None — no more data.")
                    break

                # Validate dimensions on the first block only
                if not blocks:
                    self._logger.debug(f"First data block shape: {data.shape} (channels, samples, pols).")
                    if channel >= data.shape[0]:
                        raise IndexError(
                            f"channel={channel} is out of range: data shape is {data.shape}, "
                            f"only {data.shape[0]} channel(s) available (0..{data.shape[0]-1})."
                        )
                    if data.ndim >= 3 and polarization >= data.shape[2]:
                        raise IndexError(
                            f"polarization={polarization} is out of range: data shape is {data.shape}, "
                            f"only {data.shape[2]} polarization(s) available (0..{data.shape[2]-1})."
                        )

                self.info = params
                block = data[channel, :, polarization].flatten()
                blocks.append(block)
                total += len(block)
                pbar.update(len(block))

        if not blocks:
            raise RuntimeError(
                f"No data blocks were read from '{file_path}'. "
                "The file may be empty, truncated, or unparseable."
            )

        vector = np.concatenate(blocks)[:num_samples]

        if standard_scaler:
            mu, sigma = vector.mean(), vector.std()
            vector = (vector - mu) / sigma
            self._logger.debug(f"Standard scaler applied — original mean: {mu:.4f}, std: {sigma:.4f}.")

        self.signal = vector

        self._logger.info(
            f"Loaded {len(self.signal):,} samples "
            f"(requested: {num_samples:,}, blocks read: {len(blocks)})."
        )
        if header is not None:
            npol = header.get("NPOL", "N/A")
            self._logger.info(f"File polarizations (NPOL): {npol}.")

        self._logger.debug(
            f"Signal stats — mean: {self.signal.mean():.4f}, "
            f"std: {self.signal.std():.4f}, "
            f"min: {self.signal.min():.4f}, "
            f"max: {self.signal.max():.4f}."
        )

    # -----------------------------------------------------------------------
    # KLT variants
    # -----------------------------------------------------------------------

    def apply_cklt(self, window_size: int, n_eigenvectors: int = 1) -> None:
        """Apply the Covariance KLT (C-KLT) to the loaded 1-D signal.

        The signal is partitioned into *K* non-overlapping windows of
        *window_size* samples, forming a realisation matrix of shape
        ``(K, window_size)``.  The sample covariance matrix is computed,
        diagonalised, and the *n_eigenvectors* dominant eigenvectors are used
        to project and reconstruct the signal.

        The reconstructed signal is stored in :attr:`reconstructed_signal` and
        the raw input (signal + noise, if present) is cached in
        ``_klt_input`` for use by plotting methods.

        Args:
            window_size:    Number of samples per window (W).
            n_eigenvectors: Number of dominant eigenvectors kept for
                            signal reconstruction.

        Raises:
            ValueError: If no 1-D signal is loaded, or if the signal is too
                        short to form at least one complete window.
        """
        self._logger.section("Applying C-KLT")
        self._require_signal(ndim=1)

        W = window_size
        N = len(self.signal)
        K = N // W

        self._logger.info(
            f"Window size: {W} samples | windows (K): {K} | "
            f"eigenvectors to keep: {n_eigenvectors}."
        )
        self._logger.debug(
            f"Signal length: {N:,} | samples used: {K * W:,} | "
            f"samples discarded (tail): {N - K * W:,}."
        )

        if K < 1:
            raise ValueError(
                f"Signal length ({N}) is shorter than window_size ({W}). "
                "Cannot form a single complete window."
            )

        data = self.signal if self.noise is None else self.signal + self.noise
        self._klt_input = data

        # Build realisation matrix and covariance
        matrix = data[: K * W].reshape(K, W)
        self._logger.debug(f"Realisation matrix shape: {matrix.shape}.")
        self.covariance_matrix = np.cov(matrix, rowvar=False)
        self._logger.debug(f"Covariance matrix shape: {self.covariance_matrix.shape}.")

        # Eigensystem
        self._compute_eigensystem()

        top2_ratio = self.eigenvalues[0] / self.eigenvalues[1] if len(self.eigenvalues) > 1 else float("inf")
        self._logger.info(
            f"Top eigenvalue: {self.eigenvalues[0]:.4f} | "
            f"2nd eigenvalue: {self.eigenvalues[1]:.4f} | "
            f"gap ratio λ₁/λ₂: {top2_ratio:.2f}x."
        )
        self._logger.debug(
            f"Full eigenspectrum (first 10): "
            f"{np.array2string(self.eigenvalues[:10], precision=4, suppress_small=True)}."
        )

        # Projection and reconstruction
        col_means         = matrix.mean(axis=0)
        all_coefficients  = (matrix - col_means) @ self.eigenvectors.conj()
        V                 = self.eigenvectors[:, :n_eigenvectors]
        kept_coefficients = all_coefficients[:, :n_eigenvectors]
        self.reconstructed_signal = (kept_coefficients @ V.T + col_means).flatten()

        self._logger.info(
            f"Reconstruction complete — output length: {len(self.reconstructed_signal):,} samples."
        )
        self._logger.debug(
            f"Reconstructed signal stats — "
            f"mean: {self.reconstructed_signal.mean():.4f}, "
            f"std: {self.reconstructed_signal.std():.4f}, "
            f"min: {self.reconstructed_signal.min():.4f}, "
            f"max: {self.reconstructed_signal.max():.4f}."
        )

    # def apply_mrklt(self, n_eigenvectors: int = 1, method: str = "auto") -> None:
    #     """Multiple-Realization KLT (MR-KLT).
    #
    #     The signal must be a 2-D array with shape (M realisations, N samples).
    #
    #     Args:
    #         n_eigenvectors: Number of principal components kept for reconstruction.
    #         method:         'auto', 'cov', or 'svd'. 'auto' selects SVD when N > M
    #                         and N > 5000, otherwise covariance decomposition.
    #     """
    #     self._require_signal()
    #     if self.signal.ndim != 2:
    #         raise ValueError("MR-KLT requires a 2-D signal matrix (M realisations × N samples).")
    #
    #     data = self.signal if self.noise is None else self.signal + self.noise
    #     self._klt_input = data
    #
    #     M, N = data.shape
    #     if method == "auto":
    #         method = "svd" if (N > M and N > 5000) else "cov"
    #     if method == "svd":
    #         self._compute_eigensystem_svd(data)
    #     elif method == "cov":
    #         self.covariance_matrix = np.cov(data, rowvar=False)
    #         self._compute_eigensystem()
    #     else:
    #         raise ValueError("method must be 'auto', 'cov', or 'svd'.")
    #
    #     mu = self.signal.mean(axis=0)
    #     V  = self.eigenvectors[:, :n_eigenvectors]
    #     coefficients = (self.signal - mu) @ V
    #     self.reconstructed_signal = coefficients @ V.conj().T + mu

    # def apply_tklt(self, n_eigenvectors: int = 1) -> None:
    #     """Toeplitz KLT (T-KLT).
    #
    #     The covariance matrix is approximated by a Toeplitz matrix built from
    #     the signal's autocorrelation.  Requires a 1-D signal.
    #
    #     Args:
    #         n_eigenvectors: Number of principal components kept for reconstruction.
    #     """
    #     self._require_signal(ndim=1)
    #
    #     data = self.signal if self.noise is None else self.signal + self.noise
    #     self._klt_input = data
    #
    #     autocorr = self._autocorrelation(data)
    #     self.covariance_matrix = self._toeplitz_matrix(autocorr)
    #     self._compute_eigensystem()
    #
    #     mu               = data.mean()
    #     all_coefficients = (data - mu) @ self.eigenvectors          # shape (N,)
    #     V                = self.eigenvectors[:, :n_eigenvectors]    # shape (N, n)
    #     kept_coefficients = all_coefficients[:n_eigenvectors]       # shape (n,)
    #     self.reconstructed_signal = (
    #         kept_coefficients[np.newaxis, :] @ V.conj().T
    #     ).squeeze() + mu

    # -----------------------------------------------------------------------
    # Getters
    # -----------------------------------------------------------------------

    def get_reconstructed_signal(self) -> np.ndarray | None:
        """Return the reconstructed signal, or ``None`` if not yet computed."""
        return self.reconstructed_signal

    def get_eigenvalues(self) -> np.ndarray | None:
        """Return the eigenvalues sorted in descending order, or ``None``."""
        return self.eigenvalues

    def get_eigenvectors(self) -> np.ndarray | None:
        """Return the eigenvectors sorted by descending eigenvalue, or ``None``."""
        return self.eigenvectors

    # -----------------------------------------------------------------------
    # Diagnostic methods
    # -----------------------------------------------------------------------

    def channel_power_scan(
        self,
        file_path: str,
        polarization: int = 0,
        expected_signal_freq_Mhz: float | None = None,
    ) -> tuple[plt.Figure, plt.Axes, int]:
        """Plot per-channel peak power and identify the dominant channel.

        Computes the FFT power spectrum for every coarse channel in the first
        data block of *file_path* and returns the index of the channel with
        the highest peak power (DC bin suppressed).  Optionally overlays a
        vertical marker at the expected signal frequency.

        Args:
            file_path:               Path to the GUPPI raw file.
            polarization:            Polarization index to read.
            expected_signal_freq_Mhz: When provided, a vertical dashed line is
                                     drawn at this frequency for visual
                                     cross-checking.

        Returns:
            ``(fig, axes, dominant_channel)``: the figure, axes, and the
            zero-based index of the channel with the highest peak power.

        Raises:
            ValueError: If :attr:`info` is not yet populated (i.e. no data
                        has been loaded via :meth:`load_data_from_guppi`).
        """
        if self.info is None:
            raise ValueError(
                "No file metadata available. Call load_data_from_guppi() before "
                "cross_check_channel_power_scan()."
            )

        self._logger.section("Channel power scan")
        self._logger.info(
            f"File: '{file_path}' | polarization: {polarization} | "
            f"expected signal: "
            f"{expected_signal_freq_Mhz:.4f} MHz." if expected_signal_freq_Mhz else "not specified."
        )

        raw_file = GuppiRaw(file_path)
        _, data  = raw_file.read_next_data_block()
        num_chans = data.shape[0]

        self._logger.debug(
            f"First data block shape: {data.shape} | channels: {num_chans}."
        )

        # Power spectrum per channel (FFT along the time axis).
        # DC bin (bin 0) is zeroed to suppress the PFB DC offset without
        # discarding the rest of channel 0.
        spectra = np.abs(
            np.fft.fft(data[:, :, polarization].astype(np.complex128), axis=1)
        ) ** 2
        spectra[:, 0] = 0.0

        peak_powers     = np.max(spectra, axis=1)
        dominant_channel = int(np.argmax(peak_powers))

        channel_centers_mhz = np.array([
            self.info["fch1"] / 1e6 + ch * self.info["chan_bw"] / 1e6
            for ch in range(num_chans)
        ])

        self._logger.info(
            f"Dominant channel: {dominant_channel} "
            f"(centre: {channel_centers_mhz[dominant_channel]:.4f} MHz, "
            f"peak power: {peak_powers[dominant_channel]:.3e})."
        )
        self._logger.debug(
            f"Peak power range — min: {peak_powers.min():.3e}, "
            f"max: {peak_powers.max():.3e}, "
            f"median: {np.median(peak_powers):.3e}."
        )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(channel_centers_mhz, peak_powers, "o-", color="orange",
                label="Peak power per channel")
        ax.set_yscale("log")
        ax.set_xlabel("Channel centre frequency [MHz]")
        ax.set_ylabel("Peak power (log scale)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

        if expected_signal_freq_Mhz is not None:
            ax.axvline(
                expected_signal_freq_Mhz, color="red", linestyle="--",
                label=f"Expected signal ({expected_signal_freq_Mhz:.4f} MHz)",
            )
            ax.legend()

        fig.tight_layout()
        return fig, ax, dominant_channel

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------

    def plot_eigenspectrum(
        self,
        n_components: int = 1,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the sorted eigenspectrum, highlighting the retained components.

        Args:
            n_components: Number of eigenvalues marked as *kept* (red).
            save_path:    Optional file path to save the figure.

        Returns:
            ``(fig, ax)``: the figure and axes objects.

        Raises:
            ValueError: If no eigenvalues are available (KLT not yet applied).
        """
        if self.eigenvalues is None:
            raise ValueError("No eigenvalues available. Apply a KLT variant first.")

        self._logger.debug(
            f"Plotting eigenspectrum — {len(self.eigenvalues)} eigenvalues, "
            f"{n_components} kept."
        )

        fig, ax = plt.subplots()
        ax.scatter(
            np.arange(n_components, len(self.eigenvalues)),
            self.eigenvalues[n_components:],
            color="black", marker=".", label="Rejected",
        )
        ax.scatter(
            np.arange(n_components),
            self.eigenvalues[:n_components],
            color="red", marker=".", label=f"Kept ({n_components})",
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
            self._logger.info(f"Eigenspectrum saved to '{save_path}'.")

        return fig, ax

    def plot_waterfall_comparison(
        self,
        channel: int,
        fftlength: int = 1024,
        int_factor: int = 1,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Side-by-side waterfall: noisy input vs. KLT-reconstructed signal.

        Uses ``setigen``'s PFB fine-channelisation (``get_pfb_waterfall``) to
        convert both the raw input and the KLT reconstruction into time–frequency
        waterfalls, then plots them side by side with a shared absolute RF
        frequency axis.

        Args:
            channel:    Coarse channel index (used to compute the absolute
                        frequency axis from :attr:`info`).
            fftlength:  FFT length for the PFB fine-channelisation.
            int_factor: Integration factor passed to ``get_pfb_waterfall``.

        Returns:
            ``(fig, axes)``: figure and the two-element axes array.

        Raises:
            ValueError: If :attr:`info` is ``None`` or the reconstructed signal
                        has not been computed yet.
        """
        if self.info is None:
            raise ValueError("No file metadata. Call load_data_from_guppi() first.")

        reconstructed = self.get_reconstructed_signal()
        if reconstructed is None:
            raise ValueError("No reconstructed signal. Call apply_cklt() first.")

        self._logger.debug(
            f"Waterfall comparison — channel: {channel}, "
            f"fftlength: {fftlength}, int_factor: {int_factor}."
        )

        chan_bw     = self.info["chan_bw"]
        center_freq = self.info["fch1"] + channel * chan_bw

        input_noisy = self._klt_input if self._klt_input is not None else (
            self.signal if self.noise is None else self.signal + self.noise
        )

        # Fine-channelise both signals via the reference PFB
        psd_noisy = stg.voltage.get_pfb_waterfall(
            input_noisy.reshape(-1, 1), fftlength=fftlength, int_factor=int_factor
        )
        psd_recon = stg.voltage.get_pfb_waterfall(
            reconstructed.reshape(-1, 1), fftlength=fftlength, int_factor=int_factor
        )

        self._logger.debug(
            f"PFB waterfall shapes — noisy: {np.asarray(psd_noisy).shape}, "
            f"reconstructed: {np.asarray(psd_recon).shape}."
        )

        # Absolute RF frequency axis
        fine_freqs    = np.fft.fftshift(np.fft.fftfreq(fftlength, d=1 / chan_bw))
        freq_axis_mhz = (center_freq + fine_freqs) / 1e6
        extent = [freq_axis_mhz[0], freq_axis_mhz[-1], np.asarray(psd_noisy).shape[0], 0]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, sharex=True)

        im0 = axes[0].imshow(
            stg.db(np.asarray(psd_noisy) + 1e-20),
            cmap="viridis", aspect="auto", interpolation="none", extent=extent,
        )
        axes[0].set_title(f"Input — Noisy Signal (PFB), channel {channel}")
        axes[0].set_xlabel("Frequency [MHz]")
        axes[0].set_ylabel("Time (integrations)")
        fig.colorbar(im0, ax=axes[0], label="Power [dB]")

        im1 = axes[1].imshow(
            stg.db(np.asarray(psd_recon) + 1e-20),
            cmap="viridis", aspect="auto", interpolation="none", extent=extent,
        )
        axes[1].set_title("KLT Reconstructed Signal (C-KLT + PFB)")
        axes[1].set_xlabel("Frequency [MHz]")
        fig.colorbar(im1, ax=axes[1], label="Power [dB]")

        fig.tight_layout()
        return fig, axes

    def plot_psd_comparison(
        self,
        channel: int,
        fftlength: int = 1024,
        int_factor: int = 1,
    ) -> tuple[plt.Figure, plt.Axes, float]:
        """Two-panel PSD comparison: noisy input (top) vs. KLT reconstruction (bottom).

        Time-integrates the PFB waterfalls to produce averaged power spectral
        densities, then plots them in stacked panels with a common absolute RF
        frequency axis.  The frequency of the peak in the reconstructed PSD is
        returned for external validation.

        Args:
            channel:    Coarse channel index.
            fftlength:  FFT length for the PFB fine-channelisation.
            int_factor: Integration factor passed to ``get_pfb_waterfall``.

        Returns:
            ``(fig, axes, peak_recon_mhz)``: figure, the two-element axes
            array, and the frequency (MHz) of the reconstructed signal peak.

        Raises:
            ValueError: If :attr:`info` is ``None`` or the reconstructed signal
                        is not available.
        """
        if self.info is None:
            raise ValueError("No file metadata. Call load_data_from_guppi() first.")

        reconstructed = self.get_reconstructed_signal()
        if reconstructed is None:
            raise ValueError("No reconstructed signal. Call apply_cklt() first.")

        chan_bw     = self.info["chan_bw"]
        center_freq = self.info["fch1"] + channel * chan_bw

        input_noisy = self._klt_input if self._klt_input is not None else (
            self.signal if self.noise is None else self.signal + self.noise
        )

        psd_noisy = stg.voltage.get_pfb_waterfall(
            input_noisy.reshape(-1, 1), fftlength=fftlength, int_factor=int_factor
        )
        psd_recon = stg.voltage.get_pfb_waterfall(
            reconstructed.reshape(-1, 1), fftlength=fftlength, int_factor=int_factor
        )

        integrated_noisy = np.asarray(np.sum(psd_noisy, axis=0))
        integrated_recon = np.asarray(np.sum(psd_recon, axis=0))

        fine_freqs    = np.fft.fftshift(np.fft.fftfreq(fftlength, d=1 / chan_bw))
        freq_axis_mhz = (center_freq + fine_freqs) / 1e6

        peak_noisy_mhz = freq_axis_mhz[np.argmax(integrated_noisy)]
        peak_recon_mhz = freq_axis_mhz[np.argmax(integrated_recon)]

        self._logger.info(
            f"PSD comparison — peak noisy: {peak_noisy_mhz:.4f} MHz, "
            f"peak reconstructed: {peak_recon_mhz:.4f} MHz."
        )
        self._logger.debug(
            f"Integrated PSD stats — "
            f"noisy peak power: {np.max(integrated_noisy):.3e}, "
            f"recon peak power: {np.max(integrated_recon):.3e}."
        )

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(freq_axis_mhz, stg.db(integrated_noisy + 1e-20),
                     color="navy", label="Input (noisy)")
        axes[0].axvline(peak_noisy_mhz, color="blue", linestyle="--", alpha=0.5,
                        label=f"Peak: {peak_noisy_mhz:.4f} MHz")
        axes[0].set_ylabel("Integrated power [dB]")
        axes[0].set_title(f"PSD comparison — channel {channel}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(freq_axis_mhz, stg.db(integrated_recon + 1e-20),
                     color="red", label="KLT reconstructed")
        axes[1].axvline(peak_recon_mhz, color="darkred", linestyle="--", alpha=0.7,
                        label=f"Peak: {peak_recon_mhz:.4f} MHz")
        axes[1].set_xlabel("Frequency [MHz]")
        axes[1].set_ylabel("Integrated power [dB]")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig, axes, peak_recon_mhz

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _require_signal(self, ndim: int | None = None) -> None:
        """Raise ``ValueError`` if no signal is loaded or its ndim is wrong.

        Args:
            ndim: Expected number of dimensions.  Skipped when ``None``.
        """
        if self.signal is None:
            raise ValueError(
                "No signal loaded. Call load_signal_from_array() or "
                "load_data_from_guppi() first."
            )
        if ndim is not None and self.signal.ndim != ndim:
            raise ValueError(
                f"Expected a {ndim}-D signal array, got {self.signal.ndim}-D."
            )
        if self.noise is not None and ndim is not None and self.noise.ndim != ndim:
            raise ValueError(
                "Signal and noise arrays must have the same number of dimensions."
            )

    def _compute_eigensystem(self) -> None:
        """Diagonalise the covariance matrix and sort by descending eigenvalue.

        Uses ``scipy.linalg.eigh`` (symmetric / Hermitian solver) for
        numerical stability.  Results are stored in :attr:`eigenvalues` and
        :attr:`eigenvectors`.
        """
        self._logger.debug(
            f"Running eigh on covariance matrix "
            f"({self.covariance_matrix.shape[0]}×{self.covariance_matrix.shape[1]})."
        )
        eigenvalues, eigenvectors = linalg.eigh(self.covariance_matrix)
        order = np.argsort(eigenvalues)[::-1]
        self.eigenvalues  = eigenvalues[order]
        self.eigenvectors = eigenvectors[:, order]
        self._logger.debug(
            f"Eigensystem computed — largest eigenvalue: {self.eigenvalues[0]:.4f}, "
            f"smallest: {self.eigenvalues[-1]:.4f}."
        )

    def _compute_eigensystem_svd(self, data: np.ndarray) -> None:
        """Compute the eigensystem from *data* via truncated SVD.

        Avoids forming the explicit covariance matrix; suitable for cases
        where the number of samples N greatly exceeds the number of
        realisations M.  Eigenvalues are derived from the singular values as
        ``λ = s² / (M − 1)``.

        Args:
            data: 2-D data matrix of shape ``(M, N)``.
        """
        M = data.shape[0]
        self._logger.debug(
            f"Running SVD on data matrix ({data.shape}) — "
            f"covariance matrix will not be stored."
        )
        mu_data = data.mean(axis=0)
        _, S, Vt = linalg.svd(data - mu_data, full_matrices=False)
        self.covariance_matrix = None   # Not stored to save memory
        self.eigenvalues  = (S ** 2) / max(M - 1, 1)
        self.eigenvectors = Vt.conj().T
        self._logger.debug(
            f"SVD eigensystem — largest eigenvalue: {self.eigenvalues[0]:.4f}, "
            f"smallest: {self.eigenvalues[-1]:.4f}."
        )

    def _autocorrelation(self, vector: np.ndarray) -> np.ndarray:
        """Return the non-negative-lag autocorrelation of *vector*.

        Args:
            vector: 1-D real or complex signal.

        Returns:
            Array of length ``len(vector)`` containing lags 0, 1, …, N-1.
        """
        full = np.correlate(vector, vector, mode="full")
        return full[full.size // 2:]

    def _toeplitz_matrix(self, autocorr: np.ndarray) -> np.ndarray:
        """Build a normalised symmetric Toeplitz matrix from *autocorr*.

        The first row / column is ``autocorr / autocorr[0]``, ensuring a
        unit diagonal (correlation matrix convention).

        Args:
            autocorr: Non-negative-lag autocorrelation vector.

        Returns:
            Symmetric Toeplitz matrix of shape ``(N, N)``.
        """
        normalised = autocorr / autocorr[0]
        return toeplitz(normalised)