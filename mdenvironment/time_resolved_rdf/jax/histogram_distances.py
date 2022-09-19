import jax.numpy as np
from jax import jit

from ...common_tools.jax_histogram import _histogram


@jit
def _compute_grt(rt_array, window_unitcell_volumes, bin_edges):
    """
    JAX/XLA jitted and parallelised version histogram of the time-distance matrix.

    Parameters
    ----------
    rt_array : numpy.ndarray
        Time-distance matrix from which to calculate the histogram.
    bin_edges : numpy.ndarray
        Array containing the bin edges of the histogram that results in g(r,t).
    window_unitcell_volumes : numpy.ndarray
        Array with volumes of each frame considered.

    Returns
    -------
    r : numpy.ndarray
        bin centers of g(r,t)
    g_rt : numpy.ndarray
        function values of g(r,t) for each time from t=0 considered, not averaged over whole trajectory.
    """
    Ni = rt_array.shape[1]
    Nj = rt_array.shape[2]
    n_frames = rt_array.shape[0]
    g_rt = _histogram(rt_array, bin_edges)

    r_vol = 4.0 / 3.0 * np.pi * (np.power(bin_edges[1:], 3) - np.power(bin_edges[:-1], 3))
    Nj_density = Nj / window_unitcell_volumes.mean()

    # Use normal RDF norming over each time step
    norm = Nj_density * r_vol * Ni
    g_rt = g_rt / norm / n_frames

    return g_rt
