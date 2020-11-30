"""
This package provides facilities to calculate the dynamic correlation function G(r,t), also known as the van-Hove-function, by taking MDTraj trajectory objects together with MDTraj groups as input, and returning an array containing the function values at various times from t=0.
"""

from .vanhove import grt
from .vanhove_cython import avg_grt as grt_cython
from .plotting import plot_grt
from .plotting import plot_map
