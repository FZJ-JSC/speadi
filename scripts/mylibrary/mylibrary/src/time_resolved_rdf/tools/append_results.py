import numpy as np

from ..jit.add_distribution_function_time_slice import _opt_append_grts


def _append_grts(g_rts, n, xyz, g1, g2, cuvec, cuvol,
                 r_range, nbins, pbc, opt, raw_counts,
                 g1_lens=None, g2_lens=None):
    if pbc == 'ortho':
        if opt:
            g_rts = _opt_append_grts(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, raw_counts)
            edges = np.linspace(r_range[0], r_range[1], nbins + 1)
            r = 0.5 * (edges[1:] + edges[:-1])
            return r, g_rts
    else:
        raise NotImplementedError('Currently, only orthogonal simulation cells are supported.')