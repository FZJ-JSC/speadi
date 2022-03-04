from .histogram_distances import _compute_Grt
from .time_distance_matrix import _compute_rt_mic, _compute_rt


def _mic_append_Grts(G_self, G_distinct, xyz, g1, g2, cuvec, cuvol, r_range, nbins):
    for i, sub_g1 in enumerate(g1):
        for j, sub_g2 in enumerate(g2):
            rt_array = _compute_rt_mic(xyz, sub_g1, sub_g2, cuvec)
            r, G_rt_res = _compute_Grt(rt_array, cuvol, r_range, nbins)
            G_rts[i, j] += G_rt_res
    return r, G_self, G_distinct


def _plain_append_Grts(G_self, G_distinct, xyz, g1, g2, cuvol, r_range, nbins):
    for i, sub_g1 in enumerate(g1):
        for j, sub_g2 in enumerate(g2):
            rt_array = _compute_rt(xyz, sub_g1, sub_g2)
            r, G_rt_res = _compute_Grt(rt_array, cuvol, r_range, nbins)
            G_rts[i, j] += G_rt_res
    return r, G_self, G_distinct