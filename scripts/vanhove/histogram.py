import numpy as np
from numba import njit, prange, float64

@njit
def _compute_bin(x, bin_edges):
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
    hist = np.zeros((bin_edges.shape[0],), dtype=float64)

    for x in a.flat:
        bin = _compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist[:-1]


