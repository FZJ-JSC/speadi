from jax import jit
from functools import partial

from ...common_tools.jax.time_distance_matrix import _rt_ortho_mic, _rt_general_mic
from .histogram_distances import _compute_nrt


def _append_nrts_mic(n_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, unions,
                     bin_edges, orthogonal=False):
    if orthogonal:
        n_rts = _append_nrts_ortho_mic(n_rts, n, xyz, g1, g2, cuvec, cuvol, unions, bin_edges)
    else:
        n_rts = _append_nrts_general_mic(n_rts, n, xyz, g1, g2, cuvec, cuvol, unions, bin_edges)
    return n_rts


# @jit
def _append_nrts_ortho_mic(n_rts, n, xyz, g1, g2, cuvec, cuvol, unions, bin_edges):
    g1 = tuple([tuple(g) for g in g1])
    g2 = tuple([tuple(g) for g in g2])
    for i in range(len(g1)):
        for j in range(len(g2)):
            union1, union2 = unions[str(i)][str(j)]
            n_rts = _ortho_inner_func(bin_edges, cuvec, cuvol, g1, g2, n_rts, i, j, n, union1, union2, xyz)
    return n_rts


@partial(jit, static_argnames=['g1', 'g2', 'i', 'j'])
def _ortho_inner_func(bin_edges, cuvec, cuvol, g1, g2, n_rts, i, j, n, union1, union2, xyz):
    rt_array = _rt_ortho_mic(xyz, g1[i], g2[j], union1, union2, cuvec)
    n_rts = n_rts.at[i, j, n].set(_compute_nrt(rt_array, cuvol, bin_edges))
    return n_rts


# @jit
def _append_nrts_general_mic(n_rts, n, xyz, g1, g2, cuvec, cuvol, unions, bin_edges):
    g1 = tuple([tuple(g) for g in g1])
    g2 = tuple([tuple(g) for g in g2])
    for i in range(len(g1)):
        for j in range(len(g2)):
            union1, union2 = unions[str(i)][str(j)]
            n_rts = _general_inner_func(bin_edges, cuvec, cuvol, g1, g2, n_rts, i, j, n, union1, union2, xyz)

    return n_rts


@partial(jit, static_argnames=['g1', 'g2', 'i', 'j'])
def _general_inner_func(bin_edges, cuvec, cuvol, g1, g2, n_rts, i, j, n, union1, union2, xyz):
    rt_array = _rt_general_mic(xyz, g1[i], g2[j], union1, union2, cuvec)
    n_rts = n_rts.at[i, j, n].set(_compute_nrt(rt_array, cuvol, bin_edges))
    return n_rts
