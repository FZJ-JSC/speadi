"""
Provides functions to calculate the radial distribution function (RDF) between
two groups of particles for specified windows along a trajectory g(r,t).
Groups can also consist of single particles.
"""

import mdtraj as md
import numpy as np
from numba import (get_num_threads, set_num_threads)
from tqdm import trange

from .tools.append_results import _append_grts
from .tools.construct_arrays import _construct_results_array

set_num_threads(get_num_threads())


def grt(traj, g1, g2, top=None, pbc='ortho', opt=True,
        n_windows=100, window_size=200, skip=1, stride=10,
        r_range=(0.0, 2.0), nbins=400, raw_counts=False):
    """
    Calculate g(r,t) for two groups given in a trajectory. g(r) is
    calculated for each frame in the trajectory, then averaged over
    specified windows of time, returning g(r,t) (where t represents
    the window time along the trajectory).

    Parameters
    ----------
    traj : str
    	String pointing to the location of a trajectory that MDTraj is
    	able to load
    g1 : list
        List of numpy arrays of atom indices representing the group to
    	calculate G(r,t) for
    g2 : list
        List of numpy arrays of atom indices representing the group to
    	calculate G(r,t) with

    Other parameters
    ----------------
    top : mdtraj.topology
        Topology object. Needed if trajectory given as a path to lazy-load.
    pbc : {string, NoneType}
        String representing the periodic boundary conditions of the
    	simulation cell. Currently, only 'ortho' for orthogonal simulation cells
    	is implemented.
    n_windows : int
        Number of windows in which to split the trajectory (if a whole
    	trajectory is supplied).
    window_size : int
        Number of frames in each window.
    skip : int
        Number of frames to skip at the beginning if giving a path as
    	trajectory.
    stride : int
        Number of frames in the original trajectory to skip between each
        calculation. E.g. stride = 10 means calculate distances only every
    	10th frame.
    r_range : tuple(float, float)
        Tuple over which r in g(r,t) is defined.
    nbins : int
        Number of bins (points in r to consider) in g(r,t)

    Returns
    -------
    r : np.array
        bin centers of g(r,t)
    g_rt : np.array
        averaged function values of g(r,t) for each time from t=0 considered
    """
    g_rts, g1, g2 = _construct_results_array(g1, g2, n_windows, nbins)

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
                r, g_rts = _append_grts(g_rts, n, window.xyz, g1_array, g2_array,
                                        window.unitcell_vectors, window.unitcell_volumes,
                                        r_range, nbins, pbc, opt, raw_counts,
                                        g1_lens=g1_lens, g2_lens=g2_lens)

    else:
        raise TypeError('You must input either the path to a trajectory together with a MDTraj topology instance, '
                        'or an MDTraj trajectory, or a generator of such.')

    g_rt = g_rts
    return r, g_rt
