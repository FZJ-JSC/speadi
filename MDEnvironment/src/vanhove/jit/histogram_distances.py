import numpy as np
from numba import njit, float64, prange

from MDEnvironment.src.histogram import _histogram


@njit(['f8[:,:](f4[:,:,:],f4[:],UniTuple(f8,2),i8)'], parallel=True, fastmath=True, nogil=True, cache=True)
def _compute_Grt(rt_array, window_unitcell_volumes, r_range, nbins):
    """
    Numba jitted and parallelised version of histogram of the time-distance
    matrix.

    Parameters
    ----------
    rt_array : numpy.array
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
    G_rt : np.array
        function values of G(r,t) for each time from t=0 considered, not
    	averaged over whole trajectory.
    """
    Ni = rt_array.shape[1]
    Nj = rt_array.shape[2]
    n_frames = rt_array.shape[0]
    G_rt = np.empty((n_frames, nbins), dtype=float64)
    edges = np.linspace(r_range[0], r_range[1], nbins+1)
    for t in prange(n_frames):
        G_rt[t] = _histogram(rt_array[t], edges)

    r = 0.5 * (edges[1:] + edges[:-1])
    r_vol = 4.0/3.0 * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    Nj_density = Nj / window_unitcell_volumes.mean()

    norm = Nj_density * r_vol * Ni
    G_rt = G_rt / norm

    return G_rt