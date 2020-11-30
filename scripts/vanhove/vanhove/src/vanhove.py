import mdtraj as md
from mdtraj.utils import ensure_type
from typing import Generator
from tqdm import tqdm, trange
from scipy.spatial.distance import cdist

import numpy as np
import math
import numba
from numba import njit, prange

from histogram import _histogram


def grt(traj, g1, g2, pbc='ortho', opt=True, n_chunks=100, stride=10, r_range=(0.0, 2.0), nbins=400):
    g_rts = []
    if isinstance(traj, md.core.trajectory.Trajectory):
        traj = traj[::stride]
        chunk_size = int(2.0 / traj.timestep)
        n_chunks = int(np.floor(len(traj.time) // chunk_size))
        for n in trange(n_chunks, total=n_chunks, desc='Progress over trajectory'):
            chunk = traj[int(chunk_size * n):int(chunk_size * (1 + n))]
            if pbc == 'ortho':
                if opt:
                    rt_array = _compute_rt_mic_numba(chunk.xyz, g1, g2, chunk.unitcell_vectors)
                    r, g_rt = _compute_grt_numba(rt_array, chunk.unitcell_volumes, r_range, nbins)
                else:
                    rt_array = _compute_rt_mic_vectorized(chunk.xyz, g1, g2, chunk.unitcell_vectors)
                    r, g_rt = _compute_grt(rt_array, chunk.unitcell_volumes, r_range, nbins)
            else:
                rt_array = _compute_rt_vectorized(chunk.xyz, g1, g2)
                r, g_rt = _compute_grt(rt_array, chunk.unitcell_volumes, r_range, nbins)
            g_rts.append(g_rt)

    elif isinstance(traj, Generator):
        for chunk in tqdm(traj, total=n_chunks, desc='Progress over trajectory'):
            if pbc == 'ortho':
                if opt:
                    rt_array = _compute_rt_mic_numba(chunk.xyz[::stride], g1, g2, chunk[::stride].unitcell_vectors)
                    r, g_rt = _compute_grt_numba(rt_array, chunk.unitcell_volumes, r_range, nbins)
                else:
                    rt_array = _compute_rt_mic_vectorized(chunk.xyz[::stride], g1, g2, chunk[::stride].unitcell_vectors)
                    r, g_rt = _compute_grt(rt_array, chunk.unitcell_volumes, r_range, nbins)
            else:
                rt_array = _compute_rt_vectorized(chunk.xyz[::stride], g1, g2)
                r, g_rt = _compute_grt(rt_array, chunk.unitcell_volumes, r_range, nbins)
            g_rts.append(g_rt)

    else:
        raise TypeError('You must input either an MDTraj trajectory, or a generator of such.')

    g_rt = np.mean(np.array(g_rts), axis=0)
    return r, g_rt


@njit(['f4[:,:,:](f4[:,:,:],i8[:],i8[:],f4[:,:,:])'], parallel=True, fastmath=True, nogil=True)
def _compute_rt_mic_numba(chunk, g1, g2, bv):
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


@njit(['Tuple((f8[:],f8[:,:]))(f4[:,:,:],f4[:],UniTuple(f8,2),i8)'], parallel=True, fastmath=True, nogil=True)
def _compute_grt_numba(rt_array, chunk_unitcell_volumes, r_range, nbins):
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

    return r, g_rt


def _compute_rt_vectorized(xyz, g1, g2):
    rt = np.empty((xyz.shape[0], g1.shape[0], g2.shape[0]))
    r01 = xyz[0, g1]
    for t in range(rt.shape[0]):
        rt[t] = cdist(r01, xyz[t, g2])
        np.fill_diagonal(rt[t], np.inf)

    return rt


def _grid_sub(r1, r2):
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
