import numpy as np


def _compute_G_distinct(distinct_rt_array, window_unitcell_volumes, r_range, nbins):
    Ni = distinct_rt_array.shape[1]
    Nj = distinct_rt_array.shape[2]
    n_frames = distinct_rt_array.shape[0]
    G_distinct = np.empty((n_frames, nbins), dtype=np.float32)
    for t in range(n_frames):
        g_r, edges = np.histogram(distinct_rt_array[t], range=r_range, bins=nbins)
        G_distinct[t] = g_r

    r = 0.5 * (edges[1:] + edges[:-1])
    r_vol = 4.0 / 3.0 * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    Nj_density = Nj / window_unitcell_volumes.mean()

    # Use normal RDF norming for each timestep
    norm = Nj_density * r_vol * Ni
    G_distinct = G_distinct / norm

    return r, G_distinct


def _compute_G_self(self_rt_array, window_unitcell_volumes, r_range, nbins):
    Ni = self_rt_array.shape[1]
    Nj = self_rt_array.shape[1]
    n_frames = self_rt_array.shape[0]
    G_self = np.empty((n_frames, nbins), dtype=np.float32)
    for t in range(n_frames):
        g_r, edges = np.histogram(self_rt_array[t], range=r_range, bins=nbins)
        G_self[t] = g_r

    r = 0.5 * (edges[1:] + edges[:-1])
    r_vol = 4.0 / 3.0 * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    Nj_density = Nj / window_unitcell_volumes.mean()

    # Use normal RDF norming for each timestep
    norm = Nj_density * r_vol * Ni
    G_self = G_self / norm

    return r, G_self