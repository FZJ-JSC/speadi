"""Numba-compatible histograms with NumPy's bin-edge conventions."""
import numpy as np
from numba import njit, int32

# Do not enable fastmath: range checks must also reject NaN and infinities.
opts = dict(parallel=False, fastmath=False, nogil=True, cache=False, debug=False)


@njit(['i4(f4,f4[:])'], **opts)
def _compute_bin(x, bin_edges):
    """Return a bin index, or -1 for a value outside the explicit edges."""
    if not (bin_edges[0] <= x <= bin_edges[-1]):
        return -1
    if x == bin_edges[-1]:
        return len(bin_edges) - 2
    return np.searchsorted(bin_edges, x, side='right') - 1


@njit(['i4[:](f4[:],f4[:])', 'i4[:](f4[:,:],f4[:])', 'i4[:](f4[:,:,:],f4[:])'], **opts)
def _histogram(a, bin_edges):
    """Count [left, right) bins, including the final right edge."""
    hist = np.zeros(len(bin_edges) - 1, dtype=int32)
    for x in a.flat:
        index = _compute_bin(x, bin_edges)
        if index >= 0:
            hist[index] += 1
    return hist
