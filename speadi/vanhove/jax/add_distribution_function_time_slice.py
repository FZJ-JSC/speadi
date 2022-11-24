import jax.numpy as np
from jax import jit

from .histogram_distances import _compute_Grt_distinct, _compute_Grt_self
from .time_distance_matrix import _rtau_general_mic, _rtau_ortho_mic, _rtau_ortho_mic_self


@jit
def _append_Grts_general_mic(G_self, G_distinct, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, unions,
                             bin_edges):
    for i in range(g1_lens.shape[0]):
        for j in range(g2_lens.shape[0]):
            union = unions[str(i)][str(j)]

            rt_self, rt_distinct = _rtau_general_mic(xyz, g1[i], g2[j], union, cuvec)
            G_distinct = G_distinct.at[i, j].add(_compute_Grt_distinct(rt_distinct, G_distinct[i, j], cuvol, bin_edges))
        G_self = G_self.at[i].add(_compute_Grt_self(rt_self, G_self[i], cuvol, bin_edges))
    return G_self, G_distinct


@jit
def _append_Grts_ortho_mic(G_self, G_distinct, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, unions,
                           bin_edges):
    for i in range(g1_lens.shape[0]):
        for j in range(g2_lens.shape[0]):
            union = unions[str(i)][str(j)]

            rt_self, rt_distinct = _rtau_ortho_mic(xyz, g1[i], g2[j], union, cuvec)
            G_distinct = G_distinct.at[i, j].add(_compute_Grt_distinct(rt_distinct, G_distinct[i, j], cuvol, bin_edges))
        G_self = G_self.at[i].add(_compute_Grt_self(rt_self, G_self[i], cuvol, bin_edges))
    return G_self, G_distinct


@jit
def _append_Grts_self(G_self, xyz, g1, g1_lens, cuvec, cuvol, r_range, nbins, bin_edges):
    for i in range(g1_lens.shape[0]):
        rt_self = _rtau_ortho_mic_self(xyz, g1[i], cuvec)

        G_self = G_self.at[i].add(_compute_Grt_self(rt_self, G_self[i], cuvol, bin_edges))
    return G_self