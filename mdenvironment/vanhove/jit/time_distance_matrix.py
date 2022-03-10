import math

import numpy as np
from numba import njit, float32, prange

@njit(['f4[:,:,:,:](f4[:,:,:],f4[:,:,:])'], **opts)
def _grid_sub(s1, s2):
    l1 = s1.shape[0]
    l2 = s2.shape[0]

    s1 = np.repeat(s1, l2).reshape((-1, l2))
    s1 = np.swapaxes(s1, 0, 1)
    s1 = np.reshape(s1.ravel(), (l2, l1, 3, 3))
    s2 = np.repeat(s2, l1).reshape((-1, l1))
    s2 = np.swapaxes(s2, 0, 1)
    s2 = np.reshape(s2.ravel(), (l2, l1, 3, 3))

    s12 = s1 - s2

    return s12


@njit(['Tuple((f4[:,:],f4[:,:,:]))(f4[:,:,:],i8[:],i8[:],f4[:,:,:])'], **opts)
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

    rt_distinct = np.empty((window.shape[0], g1.shape[0], g2.shape[0]), dtype=float32)
    rt_self = np.empty((window.shape[0], g1.shape[0]), dtype=float32)
    rt_self[:] = 9999.0

    frames = window.shape[0]

    for t in prange(frames):
        bv_inv = np.linalg.inv(bv[t])
        # for i in prange(g1.shape[0]):
        #     for j in prange(g2.shape[0]):
        #         s12 = bv_inv * r1[i] - bv_inv * xyz[t][j]
        #         s12 -= np.rint(s12)
        #         r12 = bv[t] * s12
        #         rt_distinct[t][i][j] = math.sqrt(r12[0,0]**2 + r12[1,1]**2 + r12[2,2]**2)

        #         if g1[i] == g2[j]:
        #             rt_self[t][i] = math.sqrt(rt_distinct[t][i][j])
        #         else:
        #             rt_distinct[t][i][j] = math.sqrt(rt_distinct[t][i][j])

        # s1 = np.expand_dims(bv_inv, 0) * np.expand_dims(r1, -1)
        s1 = np.zeros((g1.shape[0], 3, 3), dtype=float32)
        for i in prange(g1.shape[0]):
            s1[i] = bv_inv * r1[i]

        # s2 = np.expand_dims(bv_inv, 0) * np.expand_dims(xyz[t], -1)
        s2 = np.zeros((g2.shape[0], 3, 3), dtype=float32)
        for i in prange(g2.shape[0]):
            s2[i] = bv_inv * xyz[t, i]

        s12 = _grid_sub(s1, s2)
        s12 -= np.rint(s12)

        # r12 = np.expand_dims(np.expand_dims(bv[t], 0), 0) * s12
        r12 = np.zeros((g2.shape[0], g1.shape[0], 3, 3), dtype=float32)
        for j in prange(g2.shape[0]):
            for i in range(g1.shape[0]):
                r12[j, i] = bv[t] * s12[j, i]

        rt_diag = np.zeros((g1.shape[0], g2.shape[0], 3))
        for i in prange(g1.shape[0]):
            for j in prange(g2.shape[0]):
                rt_diag[i, j] = np.diag(r12[j, i].T)
        rt_sq = np.sum(rt_diag**2, axis=2)
        rt_distinct[t] = np.sqrt(rt_sq)

        # remove self interaction part of G(r,t)
        rt_self[t] = np.diag(rt_distinct[t])
        np.fill_diagonal(rt_distinct[t], 9999.0)

    return rt_self, rt_distinct


@njit(['Tuple((f4[:,:],f4[:,:,:]))(f4[:,:,:],i8[:],i8[:],f4[:,:,:])'], **opts)
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

    rt_distances = np.zeros((window.shape[0], g1.shape[0], g2.shape[0], 3), dtype=float32)
    rt_distinct = np.zeros((window.shape[0], g1.shape[0], g2.shape[0]), dtype=float32)
    rt_self = np.zeros((window.shape[0], g1.shape[0]), dtype=float32)
    rt_self[:] = 9999.0

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

                if g1[i] == g2[j]:
                    rt_self[t][i] = math.sqrt(rt_distinct[t][i][j])
                else:
                    rt_distinct[t][i][j] = math.sqrt(rt_distinct[t][i][j])
        rt_distinct[t][i][i] = 9999.0

    return rt_self, rt_distinct


@njit(['f4[:,:](f4[:,:,:],i8[:],f4[:,:,:])'], **opts)
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
