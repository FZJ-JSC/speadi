from numba import njit
from ...common_tools.numba.numba_histogram import _histogram

opts = dict(parallel=True, fastmath=True, nogil=True, cache=False, debug=True)


@njit(['f4[:](f4[:,:,:],f4[:],f4[:],f4[:])'], **opts)
def _compute_nrt_numba(rt_array, n_rt, window_unitcell_volumes, bin_edges):
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

    n_rt = _histogram(rt_array, bin_edges)

    # Norming only over the number of particles in g1 and the frames
    n_rt = n_rt / Ni / n_frames

    # Cumulative sum represents the integral of $g(r,t)$
    n_rt = n_rt.cumsum()

    return n_rt.astype('float32')