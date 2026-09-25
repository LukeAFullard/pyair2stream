"""
pyair2stream package.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: `pyproject.toml`'s `[tool.poetry].version`, read via
    # installed package metadata rather than duplicated as a literal string here.
    __version__ = version("pyair2stream")
except PackageNotFoundError:
    # Running from a source checkout with no installed/editable metadata
    # (e.g. `python -c "import pyair2stream"` without `pip install -e .` first).
    __version__ = "0.3.0"

from .preprocessing import merge_timeseries, read_and_resample
from .pre_analysis import analyze_timeseries
