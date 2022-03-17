import numpy as np


def _compute_grt(rt_array, window_unitcell_volumes, r_range, nbins, raw_counts):
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
    g_rt : np.array
        function values of g(r,t) for each time from t=0 considered, not averaged over whole trajectory.
    """

    Ni = rt_array.shape[1]
    Nj = rt_array.shape[2]
    n_frames = rt_array.shape[0]
    g_rt, edges = np.histogram(rt_array, range=r_range, bins=nbins)

    if raw_counts:
        # No normalisation w.r.t volume and particle density
        g_rt = g_rt / Ni / n_frames
    else:
        r_vol = 4.0 / 3.0 * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
        Nj_density = Nj / window_unitcell_volumes.mean()

        # Use normal RDF norming over each time step
        norm = Nj_density * r_vol * Ni
        g_rt = g_rt / norm / n_frames

    return g_rt
