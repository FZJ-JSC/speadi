import math

import numpy as np
from numba import njit, float32, prange


@njit(['Tuple((f4[:,:],f4[:,:,:]))(f4[:,:,:],i8[:],i8[:],f4[:,:,:])'], parallel=True, fastmath=True, nogil=True, cache=True)
def _compute_rt_general_mic(window, g1, g2, bv):
    """
    Numba jitted and parallelised version of function to calculate
    the distance matrix between each atom in group 1 at time zero and
    each atom in group 2 at each frame supplied. Minimum image convention
    for all box geometries applied.

    Minimum image convention distance calculation follows the computation
    from scheme B.9 in Tuckerman, M. "Statistical Mechanics: Theory and
    Molecular Simulation", 2010.

    Parameters
    ----------
    window : slice of mdtraj.trajectory
        Slice of trajectory or window from mdtraj.iterload of time length t_max
        to calculate G(r,t) over.
    g1 : numpy.array
        Numpy array of atom indices representing the group to calculate G(r,t)
    	for.
    g2 : numpy.array
        Numpy array of atom indices representing the group to calculate G(r,t)
    	with.
    bv : numpy.array
        simulation box vectors (which vary over time) as supplied by
        mdtraj.trajectory.unitcell_vectors.

    Returns
    -------
    rt_distinct : numpy.array
        Numpy array containing the time-distance matrix.
    """
    rt0 = window[0]
    r1 = rt0[g1]
    xyz = window[:, g2]
    intersect = np.intersect1d(g1, g2)

    rt_distinct = np.empty((window.shape[0], g1.shape[0], g2.shape[0]), dtype=float32)
    rt_self = np.empty((window.shape[0], g1.shape[0]), dtype=float32)

    frames = window.shape[0]

    for t in prange(frames):
        bv_inv = np.linalg.inv(bv[t])
        for i in prange(g1.shape[0]):
            for j in prange(g2.shape[0]):
                s12 = bv_inv * r1[i] - bv_inv * xyz[t][j]
                s12 -= np.rint(s12)
                r12 = bv[t] * s12
                rt_distinct[t][i][j] = math.sqrt(r12[0,0]**2 + r12[1,1]**2 + r12[2,2]**2)

        # remove self interaction part of G(r,t)
        for i in prange(intersect.shape[0]):
            ix = intersect[i]
            rt_self[t][ix] = rt_distinct[t][ix][ix]
            rt_distinct[t][ix][ix] = 99.0

    return rt_self, rt_distinct


@njit(['Tuple((f4[:,:],f4[:,:,:]))(f4[:,:,:],i8[:],i8[:],f4[:,:,:])'], parallel=True, fastmath=True, nogil=True, cache=True)
def _compute_rt_ortho_mic(window, g1, g2, bv):
    """
    Numba jitted and parallelised version of function to calculate
    the distance matrix between each atom in group 1 at time zero and
    each atom in group 2 at each frame supplied. Minimum image convention
    for orthogonal simulation boxes applied.

    Minimum image convention distance calculation adapted from scheme B.9 in
    Tuckerman, M. "Statistical Mechanics: Theory and Molecular Simulation", 2010.

    Parameters
    ----------
    window : slice of mdtraj.trajectory
        Slice of trajectory or window from mdtraj.iterload of time length t_max
        to calculate G(r,t) over.
    g1 : numpy.array
        Numpy array of atom indices representing the group to calculate G(r,t)
    	for.
    g2 : numpy.array
        Numpy array of atom indices representing the group to calculate G(r,t)
    	with.
    bv : numpy.array
        simulation box vectors (which vary over time) as supplied by
        mdtraj.trajectory.unitcell_vectors.

    Returns
    -------
    rt_distinct : numpy.array
        Numpy array containing the time-distance matrix.
    """
    rt0 = window[0]
    r1 = rt0[g1]
    xyz = window[:, g2]
    intersect = np.intersect1d(g1, g2)

    rt_distances = np.zeros((window.shape[0], g1.shape[0], g2.shape[0], 3), dtype=float32)
    rt_distinct = np.zeros((window.shape[0], g1.shape[0], g2.shape[0]), dtype=float32)
    rt_self = np.zeros((window.shape[0], g1.shape[0]), dtype=float32)

    frames = window.shape[0]

    for t in prange(frames):
        for i in prange(g1.shape[0]):
            for j in prange(g2.shape[0]):
                for coord in range(3):
                    rt_distances[t][i][j][coord] = r1[i][coord] - xyz[t][j][coord]
                    rt_distances[t][i][j][coord] -= bv[t][coord][coord] * \
                                                    round(rt_distances[t][i][j][coord] / bv[t][coord][coord])
                    rt_distances[t][i][j][coord] = rt_distances[t][i][j][coord] ** 2
                    rt_distinct[t][i][j] += rt_distances[t][i][j][coord]
                rt_distinct[t][i][j] = math.sqrt(rt_distinct[t][i][j])

        # remove self interaction part of G(r,t)
        for i in prange(intersect.shape[0]):
            ix = intersect[i]
            rt_self[t][ix] = rt_distinct[t][ix][ix]
            rt_distinct[t][ix][ix] = 99.0

    return rt_self, rt_distinct


@njit(['f4[:,:](f4[:,:,:],i8[:],f4[:,:,:])'], parallel=True, fastmath=True, nogil=True, cache=True)
def _compute_rt_mic_self(window, g1, bv):
    """
    Numba jitted and parallelised version of function to calculate
    the distance matrix between each atom in group 1 at time zero and
    each atom in group 2 at each subsequent time (frame) supplied. Minimum
    image convention for orthogonal simulation boxes applied.

    Parameters
    ----------
    window : slice of mdtraj.trajectory
        Slice of trajectory or window from mdtraj.iterload of time length t_max
        to calculate G(r,t) over.
    g1 : numpy.array
        Numpy array of atom indices representing the group to calculate G(r,t)
    	for.
    bv : numpy.array
        simulation box vectors (which vary over time) as supplied by
        mdtraj.trajectory.unitcell_vectors.

    Returns
    -------
    rt_self : numpy.array
        Numpy array containing the time-distance matrix.
    """
    rt0 = window[0]
    r1 = rt0[g1]
    xyz = window[:, g1]

    rt_distances = np.zeros((window.shape[0], g1.shape[0], 3), dtype=float32)
    rt_self = np.zeros((window.shape[0], g1.shape[0]), dtype=float32)

    frames = window.shape[0]

    for t in prange(frames):
        for i in prange(g1.shape[0]):
            for coord in range(3):
                rt_distances[t][i][coord] = r1[i][coord] - xyz[t][i][coord]
                rt_distances[t][i][coord] -= bv[t][coord][coord] * round(rt_distances[t][i][coord] / bv[t][coord][coord])
                rt_distances[t][i][coord] = rt_distances[t][i][coord] ** 2
                rt_self[t][i] += rt_distances[t][i][coord]
            rt_self[t][i] = math.sqrt(rt_self[t][i])

    return rt_self
