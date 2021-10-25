import numpy as np
from numba import jit, float64

from ...histogram import _histogram


@jit
def _compute_grt_numba(rt_array, window_unitcell_volumes, r_range, nbins, raw_counts):
    """
    Numba jitted and parallelised version of histogram of the time-distance matrix.

    Parameters
    ----------
    rt_array : numpy.array
        Time-distance matrix from which to calculate the histogram.
    window_unitcell_volumes : numpy.array
        Array with volumes of each frame considered.
    r_range : tuple(float, float)
        Tuple over which r in g(r,t) is defined.
    nbins : integer
        Number of bins (points in r to consider) in g(r,t)

    Returns
    -------
    r : np.array
        bin centers of g(r,t)
    g_rt : np.array
        function values of g(r,t) for each time from t=0 considered, not averaged over whole trajectory.
    """
    Ni = rt_array.shape[1]
    Nj = rt_array.shape[2]
    n_frames = rt_array.shape[0]
    g_rt = np.empty(nbins, dtype=float64)
    edges = np.linspace(r_range[0], r_range[1], nbins + 1)
    # for t in prange(n_frames):
    # g_r, edges = np.histogram(rt_array[t], range=r_range, bins=bins)
    # g_rt[t] = g_r
    # g_rt[t] = _histogram(rt_array[t], edges)
    g_rt = _histogram(rt_array, edges)

    if raw_counts:
        # No normalisation w.r.t volume and particle density
        g_rt = g_rt / Ni / n_frames
    else:
        # r = 0.5 * (edges[1:] + edges[:-1])
        r_vol = 4.0 * np.pi * np.power(edges[1:], 2) * (edges[1:] - edges[:-1])
        Nj_density = Nj / window_unitcell_volumes.mean()

        # Use normal RDF norming for each time step
        norm = Nj_density * r_vol * Ni
        g_rt = g_rt / norm / n_frames

    return g_rt