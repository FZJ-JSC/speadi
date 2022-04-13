from numba import jit, prange

from .time_distance_matrix import _rt_ortho_mic, _rt_general_mic
from .histogram_distances import _compute_grt_numba

opts = dict(parallel=True, fastmath=True, nogil=True, cache=False, debug=False)


@jit(**opts)
def _append_grts_mic(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, raw_counts,
                     unions, bin_edges, orthogonal=False):
    for i in prange(g1.shape[0]):
        for j in prange(g2.shape[0]):
            if orthogonal:
                rt_array = _rt_ortho_mic(xyz, g1[i][:g1_lens[i]], g2[j][:g2_lens[j]], cuvec)
            else:
                rt_array = _rt_general_mic(xyz, g1[i][:g1_lens[i]], g2[j][:g2_lens[j]], cuvec)
            g_rts[i, j, n] = _compute_grt_numba(rt_array, g_rts[i, j, n], cuvol, bin_edges)
    return g_rts
