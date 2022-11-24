import numpy as np


def _compute_nrt(rt_array, window_unitcell_volumes, r_range, nbins):
    """
    Numba jitted and parallelised version of histogram of the time-distance matrix.

    Parameters
    ----------
    rt_array : numpy.array
        Time-distance matrix from which to calculate the histogram.
    window_unitcell_volumes : numpy.array
        Array with volumes of each frame considered.
    r_range : tuple(float, float)
        Tuple over which r in n(r,t) is defined.
    nbins : integer
        Number of bins (points in r to consider) in n(r,t)

    Returns
    -------
    n_rt : np.array
        function values of n(r,t) for each time from t=0 considered, not averaged over whole trajectory.
    """

    Ni = rt_array.shape[1]
    n_frames = rt_array.shape[0]
    n_rt, edges = np.histogram(rt_array, range=r_range, bins=nbins)

    n_rt = n_rt / Ni / n_frames

    return n_rt
