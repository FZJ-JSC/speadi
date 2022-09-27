import jax.numpy as np
from jax import jit

from ...common_tools.jax.jax_histogram import _histogram


@jit
def _compute_Grt_distinct(rt_array, G_rt, window_unitcell_volumes, bin_edges):
    """
    JAX/XLA jitted and parallelised version of histogram of the time-distance matrix.

    Parameters
    ----------
    G_rt
    rt_array : numpy.ndarray
        Time-distance matrix from which to calculate the histogram.
    bin_edges : numpy.ndarray
        Array containing the bin edges of the histogram that results in G(r,t).
    window_unitcell_volumes : numpy.ndarray
        Array with volumes of each frame considered.

    Returns
    -------
    r : numpy.ndarray
        bin centers of G(r,t)
    G_rt : numpy.ndarray
        function values of G(r,t) for each time from t=0 considered, not averaged over whole trajectory.
    """
    Ni = rt_array.shape[1]
    Nj = rt_array.shape[2]
    n_frames = rt_array.shape[0]

    r_vol = 4.0 / 3.0 * np.pi * (np.power(bin_edges[1:], 3) - np.power(bin_edges[:-1], 3))
    Nj_density = Nj / window_unitcell_volumes.mean()

    # Use normal RDF norming over each time step
    norm = Nj_density * r_vol * Ni


    for t in range(n_frames):
        G_rt = G_rt.at[t].set(_histogram(rt_array[t], bin_edges) / norm / n_frames)

    # G_rt = G_rt / norm / n_frames


    return G_rt


@jit
def _compute_Grt_self(rt_array, G_rt, window_unitcell_volumes, bin_edges):
    """
    JAX/XLA jitted and parallelised version of histogram of the time-distance matrix.

    Parameters
    ----------
    G_rt
    rt_array : numpy.ndarray
        Time-distance matrix from which to calculate the histogram.
    bin_edges : numpy.ndarray
        Array containing the bin edges of the histogram that results in G(r,t).
    window_unitcell_volumes : numpy.ndarray
        Array with volumes of each frame considered.

    Returns
    -------
    r : numpy.ndarray
        bin centers of G(r,t)
    G_rt : numpy.ndarray
        function values of G(r,t) for each time from t=0 considered, not averaged over whole trajectory.
    """
    Ni = rt_array.shape[1]
    Nj = rt_array.shape[1]
    n_frames = rt_array.shape[0]

    r_vol = 4.0 / 3.0 * np.pi * (np.power(bin_edges[1:], 3) - np.power(bin_edges[:-1], 3))
    Nj_density = Nj / window_unitcell_volumes.mean()

    # Use normal RDF norming over each time step
    norm = Nj_density * r_vol * Ni
    # G_rt = G_rt / norm / n_frames

    for t in range(n_frames):
        G_rt = G_rt.at[t].set(_histogram(rt_array[t], bin_edges) / norm / n_frames)

    return G_rt
