import jax.numpy as np
from jax import jit

from ...common_tools.jax_histogram import _histogram


@jit
def _compute_nrt(rt_array, window_unitcell_volumes, bin_edges):
    """
    JAX/XLA jitted and parallelised version histogram of the time-distance matrix.

    Parameters
    ----------
    rt_array : numpy.ndarray
        Time-distance matrix from which to calculate the histogram.
    bin_edges : numpy.ndarray
        Array containing the bin edges of the histogram that results in n(r,t).
    window_unitcell_volumes : numpy.ndarray
        Array with volumes of each frame considered.

    Returns
    -------
    r : numpy.ndarray
        bin centers of n(r,t)
    n_rt : numpy.ndarray
        function values of n(r,t) for each time from t=0 considered, not averaged over whole trajectory.
    """
    Ni = rt_array.shape[1]
    n_frames = rt_array.shape[0]
    n_rt = _histogram(rt_array, bin_edges)

    # Norming only over the number of particles in g1 and the frames
    n_rt = n_rt / Ni / n_frames

    return n_rt
