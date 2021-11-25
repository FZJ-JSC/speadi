import mdtraj as md
import numpy as np
from tqdm import trange

from ..jit.add_distribution_function_time_slice import _jit_append_grts_ortho_mic, _jit_append_grts_general_mic


def _construct_results_array(g1, g2, n_windows, nbins):
    """
    Pre-allocates the array to store the results of each window, g_rts, according to the parameters supplied to `trrdf()`.
    Returns the group index arrays as lists of arrays if not given to `trrdf()` as such.

    Parameters
    ----------
    g1
    g2
    n_windows
    nbins

    Returns
    -------
    g_rts
    g1
    g2
    """
    if isinstance(g1, list) and isinstance(g2, list):
        g_rts = np.zeros((len(g1), len(g2), n_windows, nbins), dtype=np.float32)
    elif isinstance(g1, list) and not isinstance(g2, list):
        g_rts = np.zeros((len(g1), 1, n_windows, nbins), dtype=np.float32)
        g2 = [g2]
    elif not isinstance(g1, list) and isinstance(g2, list):
        g_rts = np.zeros((1, len(g2), n_windows, nbins), dtype=np.float32)
        g1 = [g1]
    else:
        g_rts = np.zeros((1, 1, n_windows, nbins), dtype=np.float32)
        g1 = [g1]
        g2 = [g2]
    return g_rts, g1, g2


def _append_grts(g_rts, n, xyz, g1, g2, cuvec, cuvol, r_range, nbins, pbc, opt, raw_counts, g1_lens=None, g2_lens=None):
    edges = np.linspace(r_range[0], r_range[1], nbins + 1)
    r = 0.5 * (edges[1:] + edges[:-1])

    if not opt:
        raise NotImplementedError('Vectorised TRRDFs have not been implemented yet! Try the jitted function with opt=True')

    if pbc == 'ortho':
        if opt:
            g_rts = _jit_append_grts_ortho_mic(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, raw_counts)
    elif pbc:
        if opt:
            g_rts = _jit_append_grts_general_mic(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, raw_counts)

    return r, g_rts


def _calculate_according_to_inputs(g1, g2, g_rt, n_windows, nbins, opt, pbc, r_range, raw_counts, skip, stride, top,
                                   traj, window_size):
    if isinstance(traj, str) and isinstance(top, md.core.topology.Topology):
        g1_lens = np.array([len(x) for x in g1], dtype=np.int64)
        g2_lens = np.array([len(x) for x in g2], dtype=np.int64)
        g1_array = np.zeros((len(g1), g1_lens.max()), dtype=np.int64)
        g2_array = np.zeros((len(g2), g2_lens.max()), dtype=np.int64)
        for i in range(g1_array.shape[0]):
            g1_array[i, :len(g1[i])] = g1[i]
        for i in range(g2_array.shape[0]):
            g2_array[i, :len(g2[i])] = g2[i]
        with md.open(traj) as f:
            f.seek(skip)
            for n in trange(n_windows, total=n_windows, desc='Progress over trajectory'):
                window = f.read_as_traj(top, n_frames=int(window_size / stride), stride=stride)
                r, g_rt = _append_grts(g_rt, n, window.xyz, g1_array, g2_array,
                                       window.unitcell_vectors, window.unitcell_volumes,
                                       r_range, nbins, pbc, opt, raw_counts,
                                       g1_lens=g1_lens, g2_lens=g2_lens)

    else:
        raise TypeError('You must input either the path to a trajectory together with a MDTraj topology instance, '
                        'or an MDTraj trajectory, or a generator of such.')
    return r, g_rt
