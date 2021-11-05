import numpy as np


def _compute_Grt(rt_array, window_unitcell_volumes, r_range, nbins):
    Ni = rt_array.shape[1]
    Nj = rt_array.shape[2]
    n_frames = rt_array.shape[0]
    G_rt = np.empty((n_frames, nbins), dtype=np.float64)
    for t in range(n_frames):
        g_r, edges = np.histogram(rt_array[t], range=r_range, bins=nbins)
        G_rt[t] = g_r

    r = 0.5 * (edges[1:] + edges[:-1])
    r_vol = 4.0 / 3.0 * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    Nj_density = Nj / window_unitcell_volumes.mean()

    # Use normal RDF norming for each timestep
    norm = Nj_density * r_vol * Ni
    G_rt = G_rt / norm

    return r, G_rt