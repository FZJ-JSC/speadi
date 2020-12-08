import mdtraj as md
from mdtraj.utils import ensure_type
from typing import Generator
from tqdm import tqdm, trange
from scipy.spatial.distance import cdist

import numpy as np
import math
import numba
from numba import njit, prange

from .histogram import _histogram


def grt(traj, g1, g2, top=None, pbc='ortho', opt=True,
        n_chunks=100, chunk_size=200, skip=1, stride=10,
        r_range=(0.0, 2.0), nbins=400):
    """
    Calculate G(r,t) for two groups given in a trajectory.
    G(r,t) is calculated for a smaller time frame (typically 2 ps). G(r,t) is
    then averaged over the whole trajectory supplied.

    Parameters
    ----------
    traj : {string, mdtraj.trajectory, Generator}
        MDTraj trajectory, or Generator of trajectories (obtained using mdtraj.iterload).
    g1 : numpy.array
        Numpy array of atom indices representing the group to calculate G(r,t) for.
    g2 : numpy.array
        Numpy array of atom indices representing the group to calculate G(r,t) with.

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
        Tuple over which r in G(r,t) is defined.
    nbins : integer
        Number of bins (points in r to consider) in G(r,t)

    Returns
    -------
    r : np.array
        bin centers of G(r,t)
    g_rt : np.array
        averaged function values of G(r,t) for each time from t=0 considered
    """
    if isinstance(g1, list) and isinstance(g2, list):
        g_rts = np.empty((len(g1), len(g2), n_chunks, int(chunk_size//stride), nbins), dtype=np.float32)
    elif isinstance(g1, list) and not isinstance(g2, list):
        g_rts = np.empty((len(g1), 1, n_chunks, int(chunk_size//stride), nbins), dtype=np.float32)
        g2 = [g2]
    elif not isinstance(g1, list) and isinstance(g2, list):
        g_rts = np.empty((1, len(g2), n_chunks, int(chunk_size//stride), nbins), dtype=np.float32)
        g1 = [g1]
    else:
        g_rts = np.empty((1, 1, n_chunks, int(chunk_size//stride), nbins), dtype=np.float32)
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
                                        r_range, nbins, pbc, opt,
                                        g1_lens=g1_lens, g2_lens=g2_lens)

    elif isinstance(traj, md.core.trajectory.Trajectory):
        traj = traj[::stride]
        chunk_size = int(2.0 / traj.timestep)
        n_chunks = int(np.floor(len(traj.time) // chunk_size))
        for n in trange(n_chunks, total=n_chunks, desc='Progress over trajectory'):
            chunk = traj[int(chunk_size * n):int(chunk_size * (1 + n))]
            r, g_rts = _append_grts(g_rts, n, chunk.xyz, g1, g2,
                                   chunk.unitcell_vectors, chunk.unitcell_volumes,
                                    r_range, nbins, pbc, opt,)

    elif isinstance(traj, Generator):
        n = 0
        for chunk in tqdm(traj, total=n_chunks, desc='Progress over trajectory'):
            r, g_rts = _append_grts(g_rts, n, chunk.xyz[::stride], g1, g2,
                                    chunk[::stride].unitcell_vectors, chunk[::stride].unitcell_volumes,
                                    r_range, nbins, pbc, opt,)
            if n >= n_chunks - 1:
                break
            n += 1

    else:
        raise TypeError('You must input either the path to a trajectory together with a MDTraj topology instance, or an MDTraj trajectory, or a generator of such.')

    g_rt = np.mean(np.array(g_rts), axis=2)
    return r, g_rt


def _append_grts(g_rts, n, xyz, g1, g2, cuvec, cuvol,
                 r_range, nbins, pbc, opt,
                 g1_lens=None, g2_lens=None):
    if pbc == 'ortho':
        if opt:
            g_rts = _opt_append_grts(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins)
            edges = np.linspace(r_range[0], r_range[1], nbins+1)
            r = 0.5 * (edges[1:] + edges[:-1])
        else:
            r, g_rts = _mic_append_grts(g_rts, n, xyz, g1, g2, cuvec, cuvol, r_range, nbins)
    else:
        r, g_rts = _plain_append_grts(g_rts, n, xyz, g1, g2, cuvec, cuvol, r_range, nbins)
    return r, g_rts


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
        to calculate G(r,t) over.
    g1 : numpy.array
        Numpy array of atom indices representing the group to calculate G(r,t) for.
    g2 : numpy.array
        Numpy array of atom indices representing the group to calculate G(r,t) with.
    bv : numpy.array
        simulation box vectors (which vary over time) as supplied by
        mdtraj.trajectory.unitcell_vectors.

    Returns
    -------
    rt : numpy.array
        Numpy array containing the time-distance matrix.
    """
    rt0 = chunk[0]
    r1 = rt0[g1]
    xyz = chunk[:, g2]

    rt = np.empty((chunk.shape[0], g1.shape[0], g2.shape[0]), dtype=numba.float32)
    rtd = np.empty((chunk.shape[0], g1.shape[0], g2.shape[0], 3), dtype=numba.float32)

    frames = chunk.shape[0]

    for t in prange(frames):
        for i in prange(g1.shape[0]):
            for j in prange(g2.shape[0]):
                rt[t][i][j] = 0
                for coord in range(3):
                    rtd[t][i][j][coord] = r1[i][coord] - xyz[t][j][coord]
                    rtd[t][i][j][coord] -= bv[t][coord][coord] * round(rtd[t][i][j][coord] / bv[t][coord][coord])
                    rtd[t][i][j][coord] = rtd[t][i][j][coord] ** 2
                    rt[t][i][j] += rtd[t][i][j][coord]
                rt[t][i][j] = math.sqrt(rt[t][i][j])
                rt[t][i][j]
                # remove self interaction part of G(r,t)
                if i == j:
                    rt[t][i][j] = 99.0

    return rt


@njit(['f8[:,:](f4[:,:,:],f4[:],UniTuple(f8,2),i8)'], parallel=True, fastmath=True, nogil=True)
def _compute_grt_numba(rt_array, chunk_unitcell_volumes, r_range, nbins):
    """
    Numba jitted and parallelised version of histogram of the time-distance matrix.

    Parameters
    ----------
    rt_array : numpy.array
        Time-distance matrix from which to calculate the histogram.
    chunk_unitcell_volumes : numpy.array
        Array with volumes of each frame considered.
    r_range : tuple(float, float)
        Tuple over which r in G(r,t) is defined.
    nbins : integer
        Number of bins (points in r to consider) in G(r,t)

    Returns
    -------
    r : np.array
        bin centers of G(r,t)
    g_rt : np.array
        function values of G(r,t) for each time from t=0 considered, not averaged over whole trajectory.
    """
    Ni = rt_array.shape[1]
    Nj = rt_array.shape[2]
    n_frames = rt_array.shape[0]
    g_rt = np.empty((n_frames, nbins), dtype=numba.float64)
    edges = np.linspace(r_range[0], r_range[1], nbins+1)
    for t in prange(n_frames):
        # g_r, edges = np.histogram(rt_array[t], range=r_range, bins=bins)
        # g_rt[t] = g_r
        g_rt[t] = _histogram(rt_array[t], edges)

    r = 0.5 * (edges[1:] + edges[:-1])
    r_vol = 4.0/3.0 * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    Nj_density = Nj / chunk_unitcell_volumes.mean()

    # Shinohara's funny norming function doesn't lead to recognisable results...
    # norm = 4 * np.pi * N_density * N * r**2

    # Use normal RDF norming for each timestep
    norm = Nj_density * r_vol * Ni
    g_rt = g_rt / norm

    return g_rt


def _compute_rt_vectorized(xyz, g1, g2):
    """
    Vectorised version of _compute_rt
    """
    rt = np.empty((xyz.shape[0], g1.shape[0], g2.shape[0]))
    r01 = xyz[0, g1]
    for t in range(rt.shape[0]):
        rt[t] = cdist(r01, xyz[t, g2])
        np.fill_diagonal(rt[t], np.inf)

    return rt


def _grid_sub(r1, r2):
    """
    
    """
    r12 = np.stack([r1]*len(r2), axis=0) - np.stack([r2]*len(r1), axis=1)
    return r12


def _compute_rt_mic_vectorized(xyz, g1, g2, box_vectors):
    bv = np.diag(box_vectors.mean(axis=0))

    rt = np.empty((xyz.shape[0], len(g1), len(g2)), dtype=np.float32)
    r1 = xyz[0, g1]
    for t in range(len(xyz)):
        r12 = _grid_sub(r1, xyz[t, g2])

        r12 -= bv * np.round(r12 / bv)

        r12 = r12**2
        r12 = r12.sum(axis=2)
        r12 = np.sqrt(r12)
        rt[t] = r12
        # remove self interaction part of G(r,t)
        np.fill_diagonal(rt[t], np.inf)

    return rt


def _compute_grt(rt_array, chunk_unitcell_volumes, r_range, nbins):
    Ni = rt_array.shape[1]
    Nj = rt_array.shape[2]
    n_frames = rt_array.shape[0]
    g_rt = np.empty((n_frames, nbins), dtype=np.float64)
    for t in prange(n_frames):
        g_r, edges = np.histogram(rt_array[t], range=r_range, bins=nbins)
        g_rt[t] = g_r

    r = 0.5 * (edges[1:] + edges[:-1])
    r_vol = 4.0/3.0 * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    Nj_density = Nj / chunk_unitcell_volumes.mean()

    # Shinohara's funny norming function doesn't lead to recognisable results...
    # norm = 4 * np.pi * N_density * N * r**2

    # Use normal RDF norming for each timestep
    norm = Nj_density * r_vol * Ni
    g_rt = g_rt / norm

    return r, g_rt


@njit(['f4[:,:,:,:,:](f4[:,:,:,:,:],i8,f4[:,:,:],i8[:,:],i8[:,:],i8[:],i8[:],f4[:,:,:],f4[:],UniTuple(f8,2),i8)'], parallel=True, fastmath=True, nogil=True)
def _opt_append_grts(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins):
    for i in prange(g1.shape[0]):
        for j in prange(g2.shape[0]):
            rt_array = _compute_rt_mic_numba(xyz, g1[i][:g1_lens[i]], g2[j][:g2_lens[j]], cuvec)
            g_rts[i,j,n] = _compute_grt_numba(rt_array, cuvol, r_range, nbins)
    return g_rts


def _mic_append_grts(g_rts, n, xyz, g1, g2, cuvec, cuvol, r_range, nbins):
    for i, sub_g1 in enumerate(g1):
        for j, sub_g2 in enumerate(g2):
            rt_array = _compute_rt_mic_vectorized(xyz, sub_g1, sub_g2, cuvec)
            r, g_rts[i,j,n] = _compute_grt(rt_array, cuvol, r_range, nbins)
    return r, g_rts


def _plain_append_grts(g_rts, n, xyz, g1, g2, cuvec, cuvol, r_range, nbins):
    for i, sub_g1 in enumerate(g1):
        for j, sub_g2 in enumerate(g2):
            rt_array = _compute_rt_vectorized(xyz, sub_g1, sub_g2)
            r, g_rts[i,j,n] = _compute_grt(rt_array, cuvol, r_range, nbins)
    return r, g_rts
