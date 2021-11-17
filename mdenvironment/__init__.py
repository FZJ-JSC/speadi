"""
This package provides the ability to both calculate time-resolved RDFs for single or groups of particles/atoms, as well
as the dynamic correlation function G(r,t), also known as the van Hove function. These functions take MDTraj trajectory
objects together with MDTraj groups as input, and return N*t*r-dimensional arrays containing the function values for N
reference groups at times t for distances of r.
"""

try:
    from importlib.metadata import version as get_pkg_version
except ModuleNotFoundError:
    from importlib_metadata import version as get_pkg_version

__version__ = get_pkg_version('MDEnvironment')

try:
    from numba import __version__ as __numba_version__
    from numba import get_num_threads
    print('Numba version ', __numba_version__, ' detected. MDEnvironment will default to optimization using Numba.')
    print(f'\nNumba currently using {get_num_threads()} threads with shared memory! Reduce this number by passing '
          f'\"numba.set_num_threads(x)\" with an appropriate integer value.\n')
except ImportError:
    print('Numba not detected in the current Python environment. Defaulting to numpy vectorization.')

from .patches import get_patches
from .time_resolved_rdf.tools.plotting import plot_grt, plot_map
from .vanhove.tools.plotting import plot_Grt, plot_dual_Grt
from .time_resolved_rdf.grt import grt
from .vanhove.Grt import Grt
