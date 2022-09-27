from ...common_tools.vectorized.time_distance_matrix import _rt_mic
from .histogram_distances import _compute_grt


def _append_grts_mic(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, unions,
                     bin_edges, orthogonal=False):
    for i in range(g1.shape[0]):
        for j in range(g2.shape[0]):
            rt_array = _rt_mic(xyz, g1[i][:g1_lens[i]], g2[j][:g2_lens[j]], cuvec, orthogonal=orthogonal)
            g_rts[i, j, n] += _compute_grt(rt_array, cuvol, r_range, nbins)
    return g_rts
