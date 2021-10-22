"""
This package provides facilities to calculate the dynamic correlation function
G(r,t), also known as the van-Hove-function, by taking MDTraj trajectory
objects together with MDTraj groups as input, and returning an array containing
the function values at various times from t=0.
"""

try:
    from importlib.metadata import version as get_pkg_version
except ModuleNotFoundError:
    from importlib_metadata import version as get_pkg_version

__version__ = get_pkg_version('mylibrary')

from numba import get_num_threads

from .src.patches import get_patches
from .src.plotting import plot_grt, plot_map
from .src.time_resolved_rdf import grt
from .src.vanhove.Grt import Grt

print(f'\nNumba currently using {get_num_threads()} threads with shared memory! Reduce this number by passing \"numba.set_num_threads(x)\" with an apropriate integer value.\n')
