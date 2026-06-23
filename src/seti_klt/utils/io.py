"""
utils/io.py
-----------
I/O utilities for the seti_klt package.

Provides:
    - Path helpers  : locate the project root, the data directory, and a
                      user-defined output tree.
    - Saving helpers: save figures and CSV files to canonical output paths.
    - SimpleLogger  : lightweight dual-destination logger (console + file)
                      with coloured output and structured formatting.
"""

from __future__ import annotations

import datetime
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def get_project_root() -> Path:
    """Return the absolute path to the project root.

    The root is defined as the directory that contains the ``seti_klt``
    package, regardless of the current working directory.

    Returns:
        Path: Absolute path to the project root.
    """
    # This file lives at  <root>/src/seti_klt/utils/io.py
    # parents: [0] utils/  [1] seti_klt/  [2] src/  [3] <root>
    return Path(__file__).resolve().parents[3]


def get_data_path() -> Path:
    """Return the absolute path to the ``data/`` directory.

    The directory is expected at ``<project_root>/data/``. It is the
    canonical location for GUPPI raw files and HDF5 data sets.

    Returns:
        Path: Absolute path to the data directory.

    Raises:
        FileNotFoundError: If the ``data/`` directory does not exist.
    """
    data_dir = get_project_root() / "data"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"'data' directory not found at '{data_dir}'. "
            "Please create it or adjust the path."
        )
    return data_dir


def get_notebooks_path() -> Path:
    """Return the absolute path to the ``notebooks/`` directory.

    Returns:
        Path: Absolute path to the notebooks directory.

    Raises:
        FileNotFoundError: If the ``notebooks/`` directory does not exist.
    """
    nb_dir = get_project_root() / "notebooks"
    if not nb_dir.exists():
        raise FileNotFoundError(
            f"'notebooks' directory not found at '{nb_dir}'."
        )
    return nb_dir


def prepare_output_dir(path: str | Path | None = None, name: str = "outputs") -> Path:
    """Create a directory if it does not exist and return its path.
    
    If path is None, the directory will be created EXACTLY where the 
    current script or notebook is being executed (Path.cwd()).
    """
    if path is None:
        resolved = Path.cwd() / name
    else:
        resolved = Path(path)
        
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def prepare_output_dirs(base_dir: str | Path | None = None) -> dict[str, Path]:
    """Create the standard output subdirectory tree and return all paths.
    
    The tree will be generated under the current execution directory.
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    
    if base_dir is None:
        outputs = prepare_output_dir(name="outputs")
    else:
        outputs = prepare_output_dir(base / "outputs")

    subdirs = ("results", "figures", "logs")
    paths: dict[str, Path] = {}
    for name in subdirs:
        p = outputs / name
        p.mkdir(parents=True, exist_ok=True)
        paths[name] = p

    return paths


# ---------------------------------------------------------------------------
# Saving utilities
# ---------------------------------------------------------------------------


def save_figure(
    fig: plt.Figure | None = None,
    filename: str = "figure.png",
    output_dir: str | Path | None = None,
    dpi: int = 300,
) -> Path:
    """Save a Matplotlib figure to the figures output directory.

    Args:
        fig:        Figure to save.  Uses the current active figure when
                    ``None``.
        filename:   Output filename, including extension (e.g. ``"plot.png"``).
        output_dir: Destination directory.  Defaults to
                    ``<project_root>/outputs/figures``.
        dpi:        Resolution in dots per inch.

    Returns:
        Path: Absolute path to the saved file.
    """
    out_dir = prepare_output_dir(
        output_dir or (prepare_output_dir() / "figures")
    )
    clean_filename = Path(filename).name
    filepath = out_dir / clean_filename

    if fig is None:
        fig = plt.gcf()

    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved → {filepath}")
    return filepath


def save_csv(
    df,
    filename: str = "data.csv",
    output_dir: str | Path | None = None,
) -> Path:
    """Save a pandas DataFrame as a CSV file.

    Args:
        df:         DataFrame to serialise.
        filename:   Output filename (e.g. ``"results.csv"``).
        output_dir: Destination directory.  Defaults to
                    ``<project_root>/outputs/results``.

    Returns:
        Path: Absolute path to the saved file.
    """
    out_dir = prepare_output_dir(
        output_dir or (prepare_output_dir() / "results")
    )
    filepath = out_dir / filename
    df.to_csv(filepath, index=False)
    print(f"CSV saved → {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class LogLevel(Enum):
    """Severity levels for :class:`SimpleLogger`."""

    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


# ANSI colour codes used for console output.
_COLOUR_RESET = "\033[0m"
_COLOUR_RED = "\033[91m"
_COLOUR_YELLOW = "\033[93m"


class SimpleLogger:
    """Lightweight dual-destination logger (console and/or file).

    Messages are formatted with a timestamp and level tag.  Console output
    uses ANSI colours to highlight warnings and errors.  All messages at or
    above the configured *level* threshold are processed; lower-priority
    messages are silently dropped.

    Example::

        logger = SimpleLogger(level=LogLevel.DEBUG)
        logger.info("Pipeline started")
        logger.warning("Low SNR detected")
        logger.close()
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        filename: str | None = None,
        level: LogLevel = LogLevel.INFO,
        console_output: bool = True,
        file_output: bool = True,
    ) -> None:
        """Initialise the logger and open the log file if requested.

        Args:
            log_dir:        Directory for the log file.  Defaults to
                            ``<project_root>/outputs/logs``.
            filename:       Log file name.  When ``None`` a timestamped name
                            such as ``log_20250101_120000.txt`` is generated.
            level:          Minimum severity to record.
            console_output: When ``True``, messages are printed to stdout.
            file_output:    When ``True``, messages are written to a file.
        """
        self.level = level
        self.console_output = console_output
        self.file_output = file_output
        self.log_file: Path | None = None

        if file_output:
            resolved_dir = Path(
                log_dir or (prepare_output_dir() / "logs")
            )
            resolved_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = resolved_dir / (filename or f"log_{timestamp}.txt")

            with open(self.log_file, "w", encoding="utf-8") as fh:
                fh.write(f"=== LOG STARTED {timestamp} ===\n")
                fh.write(f"Log Level: {self.level.name}\n\n")

            if console_output:
                print(f"Logging to: {self.log_file}")

    # ------------------------------------------------------------------
    # Public logging interface
    # ------------------------------------------------------------------

    def debug(self, message: str) -> None:
        """Log a DEBUG-level message."""
        self._write(message, LogLevel.DEBUG)

    def info(self, message: str) -> None:
        """Log an INFO-level message."""
        self._write(message, LogLevel.INFO)

    def warning(self, message: str) -> None:
        """Log a WARNING-level message."""
        self._write(message, LogLevel.WARNING)

    def error(self, message: str) -> None:
        """Log an ERROR-level message."""
        self._write(message, LogLevel.ERROR)

    def critical(self, message: str) -> None:
        """Log a CRITICAL-level message."""
        self._write(message, LogLevel.CRITICAL)

    def log(self, message: str, level: LogLevel = LogLevel.INFO) -> None:
        """Log *message* at an explicitly specified *level*.

        Args:
            message: Text to log.
            level:   Severity level.
        """
        self._write(message, level)

    # ------------------------------------------------------------------
    # Structured helpers
    # ------------------------------------------------------------------

    def section(self, title: str) -> None:
        """Write a visual section divider followed by *title*.

        Args:
            title: Section heading (automatically upper-cased).
        """
        separator = "=" * 60
        block = f"\n{separator}\n=== {title.upper()}\n{separator}"
        if self.console_output:
            print(block)
        if self.file_output and self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(block + "\n")

    def progress(
        self,
        current: int,
        total: int,
        prefix: str = "Progress",
        suffix: str = "",
    ) -> None:
        """Log a one-line progress update.

        Args:
            current: Current iteration index.
            total:   Total number of iterations.
            prefix:  Text shown before the fraction.
            suffix:  Optional text appended after the percentage.
        """
        pct = 100.0 * current / total
        self.info(f"{prefix}: {current}/{total} ({pct:.1f}%) {suffix}".rstrip())

    def summary_stats(self, stats: dict, title: str = "Summary") -> None:
        """Log a dictionary of key/value pairs under a section header.

        Floats are formatted to four decimal places; all other types use
        their default string representation.

        Args:
            stats: Mapping of statistic names to values.
            title: Section heading.
        """
        self.section(title)
        for key, value in stats.items():
            if isinstance(value, float):
                self.info(f"{key}: {value:.4f}")
            else:
                self.info(f"{key}: {value}")

    def set_level(self, level: LogLevel) -> None:
        """Change the minimum logging threshold at runtime.

        Args:
            level: New minimum severity level.
        """
        self.level = level
        self.info(f"Log level changed to: {level.name}")

    def close(self) -> None:
        """Finalise the log file by writing a closing timestamp."""
        if self.file_output and self.log_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(f"\n=== LOG ENDED {timestamp} ===\n")
            if self.console_output:
                print(f"Log closed: {self.log_file}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_log(self, level: LogLevel) -> bool:
        """Return ``True`` if *level* meets the current threshold."""
        return level.value >= self.level.value

    def _format(self, message: str, level: LogLevel) -> str:
        """Return a formatted log line with timestamp and level tag.

        Args:
            message: Raw message text.
            level:   Severity level.

        Returns:
            Formatted string ready for output.
        """
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        tag = f"[{level.name}]".ljust(10)
        return f"[{ts}] {tag} {message}"

    def _write(self, message: str, level: LogLevel) -> None:
        """Format and dispatch *message* to all enabled destinations.

        Args:
            message: Raw message text.
            level:   Severity level.
        """
        if not self._should_log(level):
            return

        line = self._format(message, level)

        if self.console_output:
            if level in (LogLevel.ERROR, LogLevel.CRITICAL):
                print(f"{_COLOUR_RED}{line}{_COLOUR_RESET}")
            elif level == LogLevel.WARNING:
                print(f"{_COLOUR_YELLOW}{line}{_COLOUR_RESET}")
            else:
                print(line)

        if self.file_output and self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")