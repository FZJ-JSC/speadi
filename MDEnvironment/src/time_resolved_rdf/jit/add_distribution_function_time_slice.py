from numba import jit, prange

from .time_distance_matrix import _compute_rt_mic_numba
from .histogram_distances import _compute_grt_numba


@jit(cache=True)
def _opt_append_grts(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, raw_counts):
    for i in prange(g1.shape[0]):
        for j in prange(g2.shape[0]):
            rt_array = _compute_rt_mic_numba(xyz, g1[i][:g1_lens[i]], g2[j][:g2_lens[j]], cuvec)
            g_rts[i, j, n] += _compute_grt_numba(rt_array, cuvol, r_range, nbins, raw_counts)
    return g_rts