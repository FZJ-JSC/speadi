from jax import jit

from ...common_tools.jax.time_distance_matrix import _rt_ortho_mic, _rt_general_mic
from .histogram_distances import _compute_grt


def _append_grts_mic(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, unions,
                     bin_edges, orthogonal=False):
    if orthogonal:
        g_rts = _append_grts_ortho_mic(g_rts, n, xyz, g1, g2, cuvec, cuvol, unions, bin_edges)
    else:
        g_rts = _append_grts_general_mic(g_rts, n, xyz, g1, g2, cuvec, cuvol, unions, bin_edges)
    return g_rts


@jit
def _append_grts_ortho_mic(g_rts, n, xyz, g1, g2, cuvec, cuvol, unions, bin_edges):
    for i in range(len(g1)):
        for j in range(len(g2)):
            union = unions[str(i)][str(j)]
            rt_array = _rt_ortho_mic(xyz, g1[i], g2[j], union, cuvec)
            g_rts = g_rts.at[i, j, n].set(_compute_grt(rt_array, cuvol, bin_edges))

    return g_rts


@jit
def _append_grts_general_mic(g_rts, n, xyz, g1, g2, cuvec, cuvol, unions, bin_edges):
    for i in range(len(g1)):
        for j in range(len(g2)):
            union = unions[str(i)][str(j)]
            rt_array = _rt_general_mic(xyz, g1[i], g2[j], union, cuvec)
            g_rts = g_rts.at[i, j, n].set(_compute_grt(rt_array, cuvol, bin_edges))

    return g_rts
