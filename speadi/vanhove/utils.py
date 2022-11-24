from mdtraj.formats.hdf5 import HDF5TrajectoryFile
from typing import Generator
from ..common_tools.get_union import get_all_unions
from tqdm.auto import trange, tqdm

import mdtraj as md
import numpy as np


def _get_acc_functions(JAX_AVAILABLE, NUMBA_AVAILABLE):
    if JAX_AVAILABLE:
        from .jax.add_distribution_function_time_slice import _append_Grts_general_mic, _append_Grts_ortho_mic, \
            _append_Grts_self
        return _append_Grts_ortho_mic, _append_Grts_general_mic, _append_Grts_self
    else:
        if NUMBA_AVAILABLE:
            # Remove OpenMP warnings caused by the Numba threading layer
            import os
            os.environ['KMP_WARNINGS'] = 'off'

            from numba import set_num_threads
            from speadi import NUMBA_THREADS
            set_num_threads(NUMBA_THREADS)

            from .numba.add_distribution_function_time_slice import _append_Grts_general_mic, _append_Grts_ortho_mic, \
                _append_Grts_self
            return _append_Grts_ortho_mic, _append_Grts_general_mic, _append_Grts_self
        else:
            from .vectorized.add_distribution_function_time_slice import _append_Grts_ortho_mic, \
                _append_Grts_general_mic
            return _append_Grts_ortho_mic, _append_Grts_general_mic, None


def _calculate_according_to_inputs(G_self, G_distinct, g1, g2, n_windows, nbins, overlap, pbc, r_range, self_only, skip,
                                   stride, top, traj, window_size):
    nbins = np.int32(nbins)
    r_range = tuple([np.float32(r) for r in r_range])

    if isinstance(traj, str) and isinstance(top, md.core.topology.Topology):
        r, G_self, G_distinct = _fast_raw_trajectory_loop(G_self, G_distinct, g1, g2, n_windows, nbins, overlap, pbc,
                                                          r_range, self_only, skip, stride, top, traj, window_size)

    elif isinstance(traj, md.core.trajectory.Trajectory):
        r, G_self, G_distinct, n_windows = _trajectory_loop(G_self, G_distinct, g1, g2, n_windows, nbins, pbc, r_range,
                                                            stride, traj)

    elif isinstance(traj, Generator):
        r, G_self, G_distinct = _generator_loop(G_self, G_distinct, g1, g2, n_windows, nbins, pbc, r_range, stride,
                                                traj)

    else:
        raise TypeError('You must input either the path to a trajectory together with a MDTraj topology instance, '
                        'or an MDTraj trajectory, or a generator of such.')

    return r, G_self, G_distinct, n_windows


def _prepare_loop_inputs(g1, g2, nbins, r_range):
    g1_lens = np.array([len(x) for x in g1], dtype=np.int32)
    g2_lens = np.array([len(x) for x in g2], dtype=np.int32)
    g1_array = np.zeros((len(g1), g1_lens.max()), dtype=np.int32)
    g2_array = np.zeros((len(g2), g2_lens.max()), dtype=np.int32)

    for i in range(g1_array.shape[0]):
        g1_array[i, :len(g1_array[i])] = g1[i]
    for i in range(g2_array.shape[0]):
        g2_array[i, :len(g2_array[i])] = g2[i]

    unions = get_all_unions(g1_array, g2_array, g1_lens, g2_lens)
    from speadi import JAX_AVAILABLE, NUMBA_AVAILABLE
    append_functions = _get_acc_functions(JAX_AVAILABLE, NUMBA_AVAILABLE)
    if NUMBA_AVAILABLE and not JAX_AVAILABLE:
        unions = np.float32(0)

    bin_edges = np.linspace(r_range[0], r_range[1], nbins + 1, dtype=np.float32)
    r = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    return append_functions, bin_edges, g1_array, g1_lens, g2_array, g2_lens, r, unions


def _fast_raw_trajectory_loop(G_self, G_distinct, g1, g2, n_windows, nbins, overlap, pbc, r_range, self_only, skip,
                              stride, top, traj, window_size):
    append_functions, bin_edges, g1_array, g1_lens, g2_array, g2_lens, r, unions = _prepare_loop_inputs(g1, g2, nbins,
                                                                                                        r_range)

    with md.open(traj) as f:
        f.seek(skip)
        for n in trange(n_windows, total=n_windows, desc='Progress over trajectory'):
            if type(f) == HDF5TrajectoryFile:
                window = f.read_as_traj(n_frames=window_size, stride=stride)
            else:
                window = f.read_as_traj(top, n_frames=int(window_size / stride), stride=stride)
            G_self, G_distinct = _append_results(G_self, G_distinct, window.xyz, g1_array, g2_array,
                                                 window.unitcell_vectors, window.unitcell_volumes, r_range,
                                                 nbins, pbc, append_functions, unions, bin_edges,
                                                 g1_lens=g1_lens, g2_lens=g2_lens, self_only=self_only)
            if isinstance(overlap, int) and overlap >= 1:
                f.seek(-window_size + overlap, 1)
    return r, G_self, G_distinct


def _generator_loop(G_self, G_distinct, g1, g2, n_windows, nbins, pbc, r_range, stride, traj):
    append_functions, bin_edges, g1_array, g1_lens, g2_array, g2_lens, r, unions = _prepare_loop_inputs(g1, g2, nbins,
                                                                                                        r_range)
    n = 0

    for window in tqdm(traj, total=n_windows, desc='Progress over trajectory'):
        G_self, G_distinct = _append_results(G_self, G_distinct, window.xyz[::stride], g1_array, g2_array,
                                             window[::stride].unitcell_vectors, window[::stride].unitcell_volumes,
                                             r_range, nbins, pbc, append_functions, unions, bin_edges)
        if n >= n_windows - 1:
            break
        n += 1
    return r, G_self, G_distinct


def _trajectory_loop(G_self, G_distinct, g1, g2, n_windows, nbins, pbc, r_range, stride, traj):
    append_functions, bin_edges, g1_array, g1_lens, g2_array, g2_lens, r, unions = _prepare_loop_inputs(g1, g2, nbins,
                                                                                                        r_range)
    traj = traj[::stride]
    window_size = int(2.0 / traj.timestep)
    n_windows = int(np.floor(len(traj.time) // window_size))

    for n in trange(n_windows, total=n_windows, desc='Progress over trajectory'):
        window = traj[int(window_size * n):int(window_size * (1 + n))]
        G_self, G_distinct = _append_results(G_self, G_distinct, window.xyz, g1_array, g2_array,
                                             window.unitcell_vectors, window.unitcell_volumes, r_range, nbins, pbc,
                                             append_functions, unions, bin_edges)
    return r, G_self, G_distinct, n_windows


def _append_results(G_self, G_distinct, xyz, g1, g2, cuvec, cuvol, r_range, nbins, pbc, append_functions, unions,
                    bin_edges, g1_lens=None, g2_lens=None, self_only=False):
    _append_Grts_ortho_mic, _append_Grts_general_mic, _append_Grts_self = append_functions

    if pbc == 'ortho':
        if self_only:
            G_self = _append_Grts_self(G_self, xyz, g1, g1_lens, cuvec, cuvol, r_range, nbins, unions, bin_edges)
        else:
            G_self, G_distinct = _append_Grts_ortho_mic(G_self, G_distinct, xyz, g1, g2, g1_lens, g2_lens, cuvec,
                                                        cuvol, r_range, nbins, unions, bin_edges)

    if pbc != 'ortho':
        G_self, G_distinct = _append_Grts_general_mic(G_self, G_distinct, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol,
                                                      r_range, nbins, unions, bin_edges)

    return G_self, G_distinct
