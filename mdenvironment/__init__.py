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

# Remove OpenMP warnings caused by the Numba threading layer
import os
os.environ['KMP_WARNINGS'] = 'off'

from .common_tools.check_acceleration import check_jax, check_numba
JAX_AVAILABLE = check_jax()
NUMBA_AVAILABLE = check_numba()
NUMBA_THREADS = 1
if JAX_AVAILABLE:
    from jax import __version__ as __jax_version__
    print('JAX version', __jax_version__, 'detected. MDEnvironment will default to optimization using JAX.')
else:
    print('JAX not detected in the current Python environment. Trying numba optimization next.')
    if NUMBA_AVAILABLE:
        from numba import __version__ as __numba_version__, get_num_threads
        NUMBA_THREADS = get_num_threads()
        print('Numba version', __numba_version__, 'detected. MDEnvironment will default to optimization using Numba.')
        print(f'\nMDEnvironment is currently using {get_num_threads()} threads with shared memory through Numba! '
              f'Reduce this number by changing the value of \"mdenvironment.NUMBA_THREADS\" to an appropriate integer '
              f'value.\n')
    else:
        print('Numba not detected in the current Python environment. Defaulting to numpy array calculations without '
              'additional acceleration.')

from .patches import get_patches
from .time_resolved_rdf.trrdf import trrdf
from .vanhove.vanhove import vanhove
