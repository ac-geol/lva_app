"""LVA core engine: pure numpy/scipy/pandas, no plotting and no file paths.

Kept deliberately free of matplotlib, multiprocessing and numba so the same
code can run under CPython, pytest, or Pyodide in a browser later.
"""
from .config import DEFAULT_CFG, make_cfg
from .pipeline import Dataset, prepare, run_estimator
from .estimators import ESTIMATORS, field_to_orientations

__all__ = ['DEFAULT_CFG', 'make_cfg', 'Dataset', 'prepare', 'run_estimator',
           'ESTIMATORS', 'field_to_orientations']
