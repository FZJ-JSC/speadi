"""
Simplified histogram function, that is able to be parallelised. Numpy's default 1d histogram function isn't able to run in numba's nopython mode during e.g. a prange.

Boost-histogram might be faster for single histograms, but also aren't able to run during a prange without additional modification.

Modified from the following sources:
https://github.com/numba/numba-examples/blob/master/examples/density_estimation/histogram/impl.py
https://numba.pydata.org/numba-examples/examples/density_estimation/histogram/results.html
"""

import numpy as np
from numba import njit, float32

@njit
def _compute_bin(x, bin_edges):
    """
    Compute the bin for a given value x.

    Parameters
    ----------
    x : float
        number to return bin for
    bin_edges : numpy.array
        array containing all bin edges

    Returns
    -------
    bin : float
    """

    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@njit
def _histogram(a, bin_edges):
    """
    Compute the histogram for an array a, given an array of bin edges.

    Parameters
    ----------
    a : numpy.array
        array to obtain a histogram for
    bin_edges : numpy.array
        array containing all bin edges

    Returns
    -------
    hist : numpy.array
    """
    # hist = np.zeros((bin_edges.shape[0],), dtype=float32)
    hist = np.zeros((bin_edges.shape[0],), dtype=np.float32)

    for x in a.flat:
        bin = _compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist[:-1]


