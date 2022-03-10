import math

import numpy as np
from numba import njit, float32, prange, guvectorize

opts = dict(parallel=True, fastmath=True, nogil=True, cache=False, debug=False)
vopts = dict(boundscheck=False, fastmath=True, nopython=True, target='parallel')


@guvectorize(['f4[:,:,:],f4[:,:,:],f4[:,:,:,:]'],
             '(i,n,n),(j,n,n)->(j,i,n,n)', **vopts)
def _vec_grid_sub(s1, s2, s12):
    l1 = s1.shape[0]
    l2 = s2.shape[0]

    s1 = np.repeat(s1, l2).reshape((-1, l2))
    s1 = np.swapaxes(s1, 0, 1)
    s1 = np.reshape(s1.ravel(), (l2, l1, 3, 3))
    s2 = np.repeat(s2, l1).reshape((-1, l1))
    s2 = np.swapaxes(s2, 0, 1)
    s2 = np.reshape(s2.ravel(), (l2, l1, 3, 3))

    s12 = s1 - s2


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


# @njit(['Tuple((f4[:,:],f4[:,:,:]))(f4[:,:,:],i8[:],i8[:],f4[:,:,:])'])
# @njit
@njit(['Tuple((f4[:,:],f4[:,:,:]))(f4[:,:,:],i8[:],i8[:],f4[:,:,:])'], **opts)
def _compute_rt_general_mic(window, g1, g2, bvt):
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

    l1 = g1.shape[0]
    l2 = g2.shape[0]
    lw = window.shape[0]

    ri = np.zeros((4), dtype=float32)
    rj = np.zeros((4), dtype=float32)

    rt_distinct = np.empty((lw, l1, l2), dtype=float32)
    rt_distinct[:] = 9999.0
    rt_self = np.empty((lw, l1), dtype=float32)
    rt_self[:] = 9999.0

    frames = window.shape[0]

    for t in prange(frames):
        bv = bvt[t].flatten()
        bv1 = np.array([bv[0], bv[3], bv[6], 0], dtype=float32)
        bv2 = np.array([bv[1], bv[4], bv[7], 0], dtype=float32)
        bv3 = np.array([bv[2], bv[5], bv[8], 0], dtype=float32)
        bv3 -= bv2 * round(bv3[1] / bv2[1])
        bv3 -= bv1 * round(bv3[0] / bv1[0])
        bv2 -= bv1 * round(bv2[0] / bv1[0])
        recip_box_size = np.array([1.0 / bv1[0], 1.0 / bv2[1], 1.0 / bv3[2]])

        for i in prange(l1):
            for j in prange(l2):
                ri[:3] = r1[i]
                rj[:3] = xyz[t, j]
                r12 = rj - ri

                r12 -= bv3 * np.round(r12[2] * recip_box_size[2])
                r12 -= bv2 * np.round(r12[1] * recip_box_size[1])
                r12 -= bv1 * np.round(r12[0] * recip_box_size[0])

                min_dist2 = 9999.0
                min_r = r12
                for x in (-1, 0, 1):
                    ra = r12 + bv1 * x
                    for y in (-1, 0, 1):
                        rb = ra + bv2 * y
                        for z in (-1, 0, 1):
                            rc = rb + bv3 * z
                            dist2 = np.dot(rc, rc)
                            if dist2 <= min_dist2:
                                min_dist2 = dist2
                                min_r = rc

                if g1[i] == g2[j]:
                    rt_self[t, i] = math.sqrt(min_dist2)
                else:
                    rt_distinct[t, i, j] = math.sqrt(min_dist2)

    return rt_self, rt_distinct


# @njit
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


# @njit
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

    for t in range(frames):
        for i in range(g1.shape[0]):
            for coord in range(3):
                rt_distances[t][i][coord] = r1[i][coord] - xyz[t][i][coord]
                rt_distances[t][i][coord] -= bv[t][coord][coord] * round(rt_distances[t][i][coord] / bv[t][coord][coord])
                rt_distances[t][i][coord] = rt_distances[t][i][coord] ** 2
                rt_self[t][i] += rt_distances[t][i][coord]
            rt_self[t][i] = math.sqrt(rt_self[t][i])

    return rt_self
