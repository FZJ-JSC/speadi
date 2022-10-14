from mdtraj.formats.hdf5 import HDF5TrajectoryFile
from ..common_tools.get_union import get_all_unions
from tqdm import trange

import mdtraj as md
import numpy as np


def _get_acc_functions(JAX_AVAILABLE, NUMBA_AVAILABLE):
    if JAX_AVAILABLE:
        from .jax.add_distribution_function_time_slice import _append_grts_mic
    else:
        if NUMBA_AVAILABLE:
            # Remove OpenMP warnings caused by the Numba threading layer
            import os
            os.environ['KMP_WARNINGS'] = 'off'

            from numba import set_num_threads
            from mdenvironment import NUMBA_THREADS
            set_num_threads(NUMBA_THREADS)

            from .numba.add_distribution_function_time_slice import _append_grts_mic
        else:
            from .vectorized.add_distribution_function_time_slice import _append_grts_mic

    return _append_grts_mic


def _append_grts(g_rts, n, xyz, g1, g2, cuvec, cuvol, r_range, nbins, pbc, unions, bin_edges,
                 _append_grts_mic, g1_lens=1, g2_lens=1):
    if pbc == 'ortho':
        g_rts = _append_grts_mic(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins,
                                 unions, bin_edges, orthogonal=True)
    else:
        g_rts = _append_grts_mic(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins,
                                 unions, bin_edges, orthogonal=False)

    return g_rts


def _calculate_according_to_inputs(g1, g2, g_rt, n_windows, nbins, pbc, r_range, skip, stride, top, traj,
                                   window_size):
    if isinstance(traj, str) and isinstance(top, md.core.topology.Topology):
        g1_lens = np.array([len(x) for x in g1], dtype=np.int32)
        g2_lens = np.array([len(x) for x in g2], dtype=np.int32)

        unions = get_all_unions(g1, g2, g1_lens, g2_lens)

        bin_edges = np.linspace(r_range[0], r_range[1], nbins + 1, dtype=np.float32)
        r = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        from mdenvironment import JAX_AVAILABLE, NUMBA_AVAILABLE
        _append_grts_mic = _get_acc_functions(JAX_AVAILABLE, NUMBA_AVAILABLE)

        with md.open(traj) as f:
            f.seek(skip)
            for n in trange(n_windows, total=n_windows, desc='Progress over trajectory'):
                if type(f) == HDF5TrajectoryFile:
                    window = f.read_as_traj(n_frames=window_size, stride=stride)
                else:
                    window = f.read_as_traj(top, n_frames=int(window_size / stride), stride=stride)
                g_rt = _append_grts(g_rt, n, window.xyz, g1, g2, window.unitcell_vectors,
                                    window.unitcell_volumes, r_range, nbins, pbc, unions, bin_edges,
                                    _append_grts_mic, g1_lens=g1_lens, g2_lens=g2_lens)

    else:
        raise TypeError('You must input either the path to a trajectory together with a MDTraj topology instance, '
                        'or an MDTraj trajectory, or a generator of such.')
    return r, g_rt
