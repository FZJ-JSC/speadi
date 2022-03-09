import numpy as np
from numba import njit, float32, prange

from ...histogram import _histogram

opts = dict(parallel=True, fastmath=True, nogil=True, cache=True, debug=False)

@njit
def _calculate_bin_edges(nbins, r_range):
    edges = np.linspace(r_range[0], r_range[1], nbins + 1)
    return edges


@njit(['f8[:,:](f4[:,:],f4[:],UniTuple(f8,2),i8)'], **opts)
def _compute_G_self(self_rt_array, window_unitcell_volumes, r_range, nbins):
    """
    Numba jitted and parallelised version of histogram of the time-distance
    matrix.

    Parameters
    ----------
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
    G_self = np.empty((n_frames, nbins), dtype=float32)
    # G_self = np.empty((n_frames, nbins), dtype=np.float32)
    edges = _calculate_bin_edges(nbins, r_range)
    for t in prange(n_frames):
        G_self[t] = _histogram(self_rt_array[t], edges)

    r = 0.5 * (edges[1:] + edges[:-1])
    r_vol = 4.0/3.0 * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    Nj_density = Nj / window_unitcell_volumes.mean()

    norm = Nj_density * r_vol * Ni
    G_self = G_self / norm

    return G_self


@njit(['f8[:,:](f4[:,:,:],f4[:],UniTuple(f8,2),i8)'], **opts)
def _compute_G_distinct(distinct_rt_array, window_unitcell_volumes, r_range, nbins):
    """
    Numba jitted and parallelised version of histogram of the time-distance
    matrix.

    Parameters
    ----------
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
    G_distinct = np.empty((n_frames, nbins), dtype=float32)
    # G_distinct = np.empty((n_frames, nbins), dtype=np.float32)
    edges = _calculate_bin_edges(nbins, r_range)
    for t in prange(n_frames):
        G_distinct[t] = _histogram(distinct_rt_array[t], edges)

    r = 0.5 * (edges[1:] + edges[:-1])
    r_vol = 4.0/3.0 * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    Nj_density = Nj / window_unitcell_volumes.mean()

    norm = Nj_density * r_vol * Ni
    G_distinct = G_distinct / norm

    return G_distinct
