from .histogram_distances import _compute_G_distinct, _compute_G_self
from .time_distance_matrix import _compute_rt_ortho_mic, _compute_rt_general_mic


def _append_Grts_ortho_mic(G_self, G_distinct, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, unions, bin_edges):
    for i, sub_g1 in enumerate(g1):
        for j, sub_g2 in enumerate(g2):
            rt_self, rt_distinct = _compute_rt_ortho_mic(xyz, sub_g1, sub_g2, cuvec)
            r, distances = _compute_G_distinct(rt_distinct, cuvol, r_range, nbins)
            G_distinct[i, j] += distances
        r, self_distance = _compute_G_self(rt_self, cuvol, r_range, nbins)
        G_self[i] += self_distance

    return G_self, G_distinct


def _append_Grts_general_mic(G_self, G_distinct, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, unions, bin_edges):
    for i, sub_g1 in enumerate(g1):
        for j, sub_g2 in enumerate(g2):
            rt_self, rt_distinct = _compute_rt_general_mic(xyz, sub_g1, sub_g2, cuvec)
            r, distances = _compute_G_distinct(rt_distinct, cuvol, r_range, nbins)
            G_distinct[i, j] += distances
        r, self_distance = _compute_G_self(rt_self, cuvol, r_range, nbins)
        G_self[i] += self_distance

    return G_self, G_distinct
