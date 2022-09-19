import jax.numpy as np
from jax import jit

from .time_distance_matrix import _rt_ortho_mic, _rt_general_mic
from .histogram_distances import _compute_nrt


def _append_nrts_mic(n_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, unions,
                     bin_edges, orthogonal=False):
    if orthogonal:
        n_rts = _append_nrts_ortho_mic(n_rts, n, xyz, g1, g2, cuvec, cuvol, unions, bin_edges)
    else:
        n_rts = _append_nrts_general_mic(n_rts, n, xyz, g1, g2, cuvec, cuvol, unions, bin_edges)
    return n_rts


@jit
def _append_nrts_ortho_mic(n_rts, n, xyz, g1, g2, cuvec, cuvol, unions, bin_edges):
    for i in range(len(g1)):
        for j in range(len(g2)):
            union = unions[str(i)][str(j)]
            rt_array = _rt_ortho_mic(xyz, g1[i], g2[j], union, cuvec)
            n_rts = n_rts.at[i, j, n].set(_compute_nrt(rt_array, cuvol, bin_edges))

    return n_rts


@jit
def _append_nrts_general_mic(n_rts, n, xyz, g1, g2, cuvec, cuvol, unions, bin_edges):
    for i in range(len(g1)):
        for j in range(len(g2)):
            union = unions[str(i)][str(j)]
            rt_array = _rt_general_mic(xyz, g1[i], g2[j], union, cuvec)
            n_rts = n_rts.at[i, j, n].set(_compute_nrt(rt_array, cuvol, bin_edges))

    return n_rts
