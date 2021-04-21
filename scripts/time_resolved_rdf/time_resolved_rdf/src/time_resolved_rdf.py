import mdtraj as md
from tqdm import trange

import numpy as np
import math
from numba import njit, prange, float32, float64, get_num_threads, set_num_threads, jit

from .histogram import _histogram

set_num_threads(get_num_threads())


def grt(traj, g1, g2, top=None, pbc='ortho', opt=True,
        n_chunks=100, chunk_size=200, skip=1, stride=10,
        r_range=(0.0, 2.0), nbins=400, raw_counts=False):
    """
    Calculate g(r,t) for two groups given in a trajectory.
    g(r) is calculated for each frame in the trajectory, then averaged over specified chunks
    of time, returning g(r,t) (t representing the time during the trajectory).

    Parameters
    ----------
    traj : string
        MDTraj trajectory, or Generator of trajectories (obtained using mdtraj.iterload).
    g1 : numpy.array
        Numpy array of atom indices representing the group to calculate g(r,t) for.
    g2 : numpy.array
        Numpy array of atom indices representing the group to calculate g(r,t) with.

    Other parameters
    ----------------
    top : mdtraj.topology
        Topology object. Needed if trajectory given as a path to lazy-load.
    pbc : {string, NoneType}
        String representing the periodic boundary conditions of the simulation cell.
        Currently, only 'ortho' for orthogonal simulation cells is implemented.
    n_chunks : integer
        Number of chunks in which to split the trajectory (if a whole trajectory is supplied).
    chunk_size : integer
        Number of frames in each chunk.
    skip : integer
        Number of frames to skip at the beginning if giving a path as trajectory.
    stride : integer
        Number of frames in the original trajectory to skip between each
        calculation. E.g. stride = 10 means calculate distances only every 10th frame.
    r_range : tuple(float, float)
        Tuple over which r in g(r,t) is defined.
    nbins : integer
        Number of bins (points in r to consider) in g(r,t)

    Returns
    -------
    r : np.array
        bin centers of g(r,t)
    g_rt : np.array
        averaged function values of g(r,t) for each time from t=0 considered
    """
    if isinstance(g1, list) and isinstance(g2, list):
        g_rts = np.zeros((len(g1), len(g2), n_chunks, nbins), dtype=np.float32)
    elif isinstance(g1, list) and not isinstance(g2, list):
        g_rts = np.zeros((len(g1), 1, n_chunks, nbins), dtype=np.float32)
        g2 = [g2]
    elif not isinstance(g1, list) and isinstance(g2, list):
        g_rts = np.zeros((1, len(g2), n_chunks, nbins), dtype=np.float32)
        g1 = [g1]
    else:
        g_rts = np.zeros((1, 1, n_chunks, nbins), dtype=np.float32)
        g1 = [g1]
        g2 = [g2]

    if isinstance(traj, str) and isinstance(top, md.core.topology.Topology):
        g1_lens = np.array([len(x) for x in g1], dtype=np.int64)
        g2_lens = np.array([len(x) for x in g2], dtype=np.int64)
        g1_array = np.empty((len(g1), g1_lens.max()), dtype=np.int64)
        g2_array = np.empty((len(g2), g2_lens.max()), dtype=np.int64)
        g1_array[:,:], g2_array[:,:] = np.nan, np.nan
        for i in range(g1_array.shape[0]):
            g1_array[i,:len(g1[i])] = g1[i]
        for i in range(g2_array.shape[0]):
            g2_array[i,:len(g2[i])] = g2[i]
        with md.open(traj) as f:
            f.seek(skip)
            for n in trange(n_chunks, total=n_chunks, desc='Progress over trajectory'):
                chunk = f.read_as_traj(top, n_frames=int(chunk_size / stride), stride=stride)
                r, g_rts = _append_grts(g_rts, n, chunk.xyz, g1_array, g2_array,
                                        chunk.unitcell_vectors, chunk.unitcell_volumes,
                                        r_range, nbins, pbc, opt, raw_counts,
                                        g1_lens=g1_lens, g2_lens=g2_lens)

    else:
        raise TypeError('You must input either the path to a trajectory together with a MDTraj topology instance, or an MDTraj trajectory, or a generator of such.')

    g_rt = g_rts
    return r, g_rt


def _append_grts(g_rts, n, xyz, g1, g2, cuvec, cuvol,
                 r_range, nbins, pbc, opt, raw_counts,
                 g1_lens=None, g2_lens=None):
    if pbc == 'ortho':
        if opt:
            g_rts = _opt_append_grts(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, raw_counts)
            edges = np.linspace(r_range[0], r_range[1], nbins+1)
            r = 0.5 * (edges[1:] + edges[:-1])
            return r, g_rts
    else:
        raise NotImplementedError('Currently, only orthogonal simulation cells are supported.')


@njit(['f4[:,:,:](f4[:,:,:],i8[:],i8[:],f4[:,:,:])'], parallel=True, fastmath=True, nogil=True)
def _compute_rt_mic_numba(chunk, g1, g2, bv):
    """
    Numba jitted and parallelised version of function to calculate
    the distance matrix between each atom in group 1 at time zero and
    each atom in group 2 at each frame supplied. Minimum image convention
    for orthogonal simulation boxes applied.

    Parameters
    ----------
    chunk : slice of mdtraj.trajectory
        Slice of trajectory or chunk from mdtraj.iterload of time length t_max
        to calculate g(r,t) over.
    g1 : numpy.array
        Numpy array of atom indices representing the group to calculate g(r,t) for.
    g2 : numpy.array
        Numpy array of atom indices representing the group to calculate g(r,t) with.
    bv : numpy.array
        simulation box vectors (which vary over time) as supplied by
        mdtraj.trajectory.unitcell_vectors.

    Returns
    -------
    rt : numpy.array
        Numpy array containing the time-distance matrix.
    """
    # rt0 = chunk[0]
    # r1 = rt0[g1]
    r1 = chunk[:, g1]
    xyz = chunk[:, g2]

    rt = np.empty((chunk.shape[0], g1.shape[0], g2.shape[0]), dtype=float32)
    rtd = np.empty((chunk.shape[0], g1.shape[0], g2.shape[0], 3), dtype=float32)

    frames = chunk.shape[0]

    for t in prange(frames):
        for i in prange(g1.shape[0]):
            for j in range(g2.shape[0]):
                rt[t][i][j] = 0
                for coord in range(3):
                    rtd[t][i][j][coord] = r1[t][i][coord] - xyz[t][j][coord]
                    rtd[t][i][j][coord] -= bv[t][coord][coord] * round(rtd[t][i][j][coord] / bv[t][coord][coord])
                    rtd[t][i][j][coord] = rtd[t][i][j][coord] ** 2
                    rt[t][i][j] += rtd[t][i][j][coord]
                rt[t][i][j] = math.sqrt(rt[t][i][j])
                rt[t][i][j]
                # remove self interaction part of g(r,t) by making the distance i == j large
                if i == j:
                    rt[t][i][j] = 99.0

    return rt


# @njit(['f8[:,:](f4[:,:,:],f4[:],UniTuple(f8,2),i8)'], parallel=True, fastmath=True, nogil=True)
@jit
def _compute_grt_numba(rt_array, chunk_unitcell_volumes, r_range, nbins, raw_counts):
    """
    Numba jitted and parallelised version of histogram of the time-distance matrix.

    Parameters
    ----------
    rt_array : numpy.array
        Time-distance matrix from which to calculate the histogram.
    chunk_unitcell_volumes : numpy.array
        Array with volumes of each frame considered.
    r_range : tuple(float, float)
        Tuple over which r in g(r,t) is defined.
    nbins : integer
        Number of bins (points in r to consider) in g(r,t)

    Returns
    -------
    r : np.array
        bin centers of g(r,t)
    g_rt : np.array
        function values of g(r,t) for each time from t=0 considered, not averaged over whole trajectory.
    """
    Ni = rt_array.shape[1]
    Nj = rt_array.shape[2]
    n_frames = rt_array.shape[0]
    g_rt = np.empty(nbins, dtype=float64)
    edges = np.linspace(r_range[0], r_range[1], nbins+1)
    # for t in prange(n_frames):
        # g_r, edges = np.histogram(rt_array[t], range=r_range, bins=bins)
        # g_rt[t] = g_r
        # g_rt[t] = _histogram(rt_array[t], edges)
    g_rt = _histogram(rt_array, edges)

    if raw_counts:
        # No normalisation w.r.t volume and particle density
        g_rt = g_rt / Ni / n_frames
    else:
        # r = 0.5 * (edges[1:] + edges[:-1])
        r_vol = 4.0 * np.pi * np.power(edges[1:], 2) * (edges[1:] - edges[:-1])
        Nj_density = Nj / chunk_unitcell_volumes.mean()

        # Use normal RDF norming for each time step
        norm = Nj_density * r_vol * Ni
        g_rt = g_rt / norm / n_frames

    return g_rt


# @njit(['f4[:,:,:,:](f4[:,:,:,:],i8,f4[:,:,:],i8[:,:],i8[:,:],i8[:],i8[:],f4[:,:,:],f4[:],UniTuple(f8,2),i8)'], parallel=True, fastmath=True, nogil=True)
@jit
def _opt_append_grts(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, raw_counts):
    for i in prange(g1.shape[0]):
        for j in prange(g2.shape[0]):
            rt_array = _compute_rt_mic_numba(xyz, g1[i][:g1_lens[i]], g2[j][:g2_lens[j]], cuvec)
            g_rts[i,j,n] += _compute_grt_numba(rt_array, cuvol, r_range, nbins, raw_counts)
    return g_rts
