"""
This package provides facilities to calculate the dynamic correlation function G(r,t), also known as the van-Hove-function, by taking MDTraj trajectory objects together with MDTraj groups as input, and returning an array containing the function values at various times from t=0.
"""

from .src.vanhove import grt
from .src.vanhove_cython import avg_grt as grt_cython
from .src.plotting import plot_grt
from .src.plotting import plot_map

from ._version import __version__
