"""
seti_klt.utils
--------------
Utility sub-package for the seti_klt library.

Public re-exports
-----------------
All commonly used symbols are available directly from this namespace::

    from seti_klt.utils import SimpleLogger, LogLevel, save_figure, get_data_path
"""

from .io import (
    LogLevel,
    SimpleLogger,
    get_data_path,
    get_notebooks_path,
    get_project_root,
    prepare_output_dir,
    prepare_output_dirs,
    save_csv,
    save_figure,
)

__all__ = [
    # Path helpers
    "get_project_root",
    "get_data_path",
    "get_notebooks_path",
    "prepare_output_dir",
    "prepare_output_dirs",
    # Saving helpers
    "save_figure",
    "save_csv",
    # Logger
    "LogLevel",
    "SimpleLogger",
]