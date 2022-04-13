import numpy as np
from numba import njit, float32

from ...common_tools.numba_histogram import _histogram

pi = float32(np.pi)
opts = dict(parallel=True, fastmath=True, nogil=True, cache=False, debug=False)


@njit(['f4[:,:](f4[:,:],f4[:,:],f4[:],f4[:])'], **opts)
def _compute_G_self(self_rt_array, G_self, window_unitcell_volumes, bin_edges):
    """
    Numba jitted and parallelised version of histogram of the time-distance
    matrix.

    Parameters
    ----------
    G_self
    self_rt_array : numpy.array
        Time-distance matrix from which to calculate the histogram.
    window_unitcell_volumes : numpy.array
        Array with volumes of each frame considered.
    r_range : tuple(float, float)
        Tuple over which r in G(r,t) is defined.
    nbins : integer
        Number of bins (points in r to consider) in G(r,t)

    Returns
    -------
    r : np.array
        bin centers of G(r,t)
    G_self : np.array
        function values of G(r,t) for each time from t=0 considered, not
    	averaged over whole trajectory.
    """
    Ni = self_rt_array.shape[1]
    Nj = self_rt_array.shape[1]
    n_frames = self_rt_array.shape[0]

    r_vol = 4.0/3.0 * pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    Nj_density = Nj / window_unitcell_volumes.mean()
    norm = Nj_density * r_vol * Ni

    for t in range(n_frames):
        G_self[t] += _histogram(self_rt_array[t], bin_edges) / norm

    return G_self.astype(np.float32)


@njit(['f4[:,:](f4[:,:,:],f4[:,:],f4[:],f4[:])'], **opts)
def _compute_G_distinct(distinct_rt_array, G_distinct, window_unitcell_volumes, bin_edges):
    """
    Numba jitted and parallelised version of histogram of the time-distance
    matrix.

    Parameters
    ----------
    G_distinct
    distinct_rt_array : numpy.array
        Time-distance matrix from which to calculate the histogram.
    window_unitcell_volumes : numpy.array
        Array with volumes of each frame considered.
    r_range : tuple(float, float)
        Tuple over which r in G(r,t) is defined.
    nbins : integer
        Number of bins (points in r to consider) in G(r,t)

    Returns
    -------
    r : np.array
        bin centers of G(r,t)
    G_distinct : np.array
        function values of G(r,t) for each time from t=0 considered, not
    	averaged over whole trajectory.
    """
    Ni = distinct_rt_array.shape[1]
    Nj = distinct_rt_array.shape[2]
    n_frames = distinct_rt_array.shape[0]

    r_vol = 4.0/3.0 * pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    Nj_density = Nj / window_unitcell_volumes.mean()
    norm = Nj_density * r_vol * Ni

    for t in range(n_frames):
        G_distinct[t] += _histogram(distinct_rt_array[t], bin_edges) / norm

    return G_distinct.astype(np.float32)
