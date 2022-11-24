from mdtraj.formats.hdf5 import HDF5TrajectoryFile
from typing import Generator

import mdtraj as md
import numpy as np
from tqdm import trange, tqdm

from .append_results_array import _append_results


def _calculate_according_to_inputs(G_self, G_distinct, g1, g2, n_windows, nbins, opt, overlap, pbc, r_range, self_only, skip, stride, top, traj,
                                   window_size):
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
                if type(f) == HDF5TrajectoryFile:
                    window = f.read_as_traj(n_frames=window_size, stride=stride)
                else:
                    window = f.read_as_traj(top, n_frames=int(window_size / stride), stride=stride)
                r, G_self, G_distinct = _append_results(G_self, G_distinct, window.xyz, g1_array, g2_array,
                                                window.unitcell_vectors, window.unitcell_volumes,
                                                r_range, nbins, pbc, opt,
                                                g1_lens=g1_lens, g2_lens=g2_lens, self_only=self_only)
                if isinstance(overlap, int) and overlap >= 1:
                    f.seek(-window_size + overlap, 1)

    elif isinstance(traj, md.core.trajectory.Trajectory):
        r, G_self, G_distinct, n_windows = _trajectory_loop(G_self, G_distinct, g1, g2, n_windows, nbins, opt, pbc, r_range, stride, traj)

    elif isinstance(traj, Generator):
        r, G_self, G_distinct = _generator_loop(G_self, G_distinct, g1, g2, n_windows, nbins, opt, pbc, r_range, stride, traj)

    else:
        raise TypeError('You must input either the path to a trajectory together with a MDTraj topology instance, '
                        'or an MDTraj trajectory, or a generator of such.')

    return r, G_self, G_distinct, n_windows


def _generator_loop(G_self, G_distinct, g1, g2, n_windows, nbins, opt, pbc, r_range, stride, traj):
    n = 0
    for window in tqdm(traj, total=n_windows, desc='Progress over trajectory'):
        r, G_self, G_distinct = _append_results(G_self, G_distinct, window.xyz[::stride], g1, g2,
                                window[::stride].unitcell_vectors, window[::stride].unitcell_volumes,
                                                r_range, nbins, pbc, opt, )
        if n >= n_windows - 1:
            break
        n += 1
    return r, G_self, G_distinct


def _trajectory_loop(G_self, G_distinct, g1, g2, n_windows, nbins, opt, pbc, r_range, stride, traj):
    traj = traj[::stride]
    window_size = int(2.0 / traj.timestep)
    n_windows = int(np.floor(len(traj.time) // window_size))
    for n in trange(n_windows, total=n_windows, desc='Progress over trajectory'):
        window = traj[int(window_size * n):int(window_size * (1 + n))]
        r, G_self, G_distinct = _append_results(G_self, G_distinct, window.xyz, g1, g2,
                                                window.unitcell_vectors, window.unitcell_volumes,
                                                r_range, nbins, pbc, opt, )
    return r, G_self, G_distinct, n_windows