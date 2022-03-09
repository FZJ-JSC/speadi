from .histogram_distances import _compute_G_distinct, _compute_G_self
from .time_distance_matrix import _compute_rt_mic, _compute_rt


def _mic_append_Grts(G_self, G_distinct, xyz, g1, g2, cuvec, cuvol, r_range, nbins):
    for i, sub_g1 in enumerate(g1):
        for j, sub_g2 in enumerate(g2):
            rt_self, rt_distinct = _compute_rt_mic(xyz, sub_g1, sub_g2, cuvec)
            r, distances = _compute_G_distinct(rt_distinct, cuvol, r_range, nbins)
            G_distinct[i, j] += distances
        r, self_distance = _compute_G_self(rt_self, cuvol, r_range, nbins)
        G_self[i] += self_distance

    return r, G_self, G_distinct


def _plain_append_Grts(G_self, G_distinct, xyz, g1, g2, cuvol, r_range, nbins):
    for i, sub_g1 in enumerate(g1):
        for j, sub_g2 in enumerate(g2):
            rt_self, rt_distinct = _compute_rt(xyz, sub_g1, sub_g2)
            r, distances = _compute_G_distinct(rt_distinct, cuvol, r_range, nbins)
            G_distinct[i, j] += distances
        r, self_distance = _compute_G_self(rt_self, cuvol, r_range, nbins)
        G_self[i] += self_distance

    return r, G_self, G_distinct