import numpy as np

from ..jit.add_distribution_function_time_slice import _jit_append_Grts_general_mic, _jit_append_Grts_ortho_mic, _jit_append_Grts_self
from ..vectorized.add_distribution_function_time_slice import _plain_append_Grts_ortho_mic, _plain_append_Grts_general_mic


def _append_results(G_self, G_distinct, xyz, g1, g2, cuvec, cuvol, r_range, nbins, pbc, opt, g1_lens=None, g2_lens=None, self_only=False):
    if opt and pbc == 'ortho':
        if self_only:
            G_self = _jit_append_Grts_self(G_self, xyz, g1, g1_lens, cuvec, cuvol, r_range, nbins)
        else:
            G_self, G_distinct = _jit_append_Grts_ortho_mic(G_self, G_distinct, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins)

        edges = np.linspace(r_range[0], r_range[1], nbins + 1)
        r = 0.5 * (edges[1:] + edges[:-1])

    if opt and pbc != 'ortho':
        G_self, G_distinct = _jit_append_Grts_general_mic(G_self, G_distinct, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins)

        edges = np.linspace(r_range[0], r_range[1], nbins+1)
        r = 0.5 * (edges[1:] + edges[:-1])

    if not opt and pbc == 'ortho':
        r, G_self, G_distinct = _plain_append_Grts_ortho_mic(G_self, G_distinct, xyz, g1, g2, cuvec, cuvol, r_range, nbins)

    if not opt and pbc != 'ortho':
        r, G_self, G_distinct = _plain_append_Grts_general_mic(G_self, G_distinct, xyz, g1, g2, cuvec, cuvol, r_range, nbins)

    return r, G_self, G_distinct
