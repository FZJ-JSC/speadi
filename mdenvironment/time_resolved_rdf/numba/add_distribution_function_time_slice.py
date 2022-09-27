from numba import jit, prange

from ...common_tools.numba.time_distance_matrix import _rt_ortho_mic, _rt_general_mic
from .histogram_distances import _compute_grt_numba

opts = dict(parallel=True, fastmath=True, nogil=True, cache=False, debug=False)


@jit(**opts)
def _append_grts_mic(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins,
                     unions, bin_edges, orthogonal=False):
    for i in prange(g1_lens.shape[0]):
        for j in prange(g2_lens.shape[0]):
            if orthogonal:
                rt_array = _rt_ortho_mic(xyz, g1[i].astype('int32'), g2[j].astype('int32'), cuvec)
            else:
                rt_array = _rt_general_mic(xyz, g1[i].astype('int32'), g2[j].astype('int32'), cuvec)
            g_rts[i, j, n] = _compute_grt_numba(rt_array, g_rts[i, j, n], cuvol, bin_edges)
    return g_rts
