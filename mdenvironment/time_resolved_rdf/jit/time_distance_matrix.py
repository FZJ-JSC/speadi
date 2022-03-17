import math

import numpy as np
from numba import njit, float32, prange

opts = dict(parallel=True, fastmath=True, nogil=True, cache=False, debug=False)


@njit(['f4[:,:,:](f4[:,:,:],i8[:],i8[:],f4[:,:,:])'], **opts)
def _compute_rt_ortho_mic(window, g1, g2, bv):
    """
    Numba jitted and parallelised version of function to calculate
    the distance matrix between each atom in group 1 at time zero and
    each atom in group 2 at each frame supplied. Minimum image convention
    for orthogonal simulation boxes applied.

    Parameters
    ----------
    window : slice of mdtraj.trajectory
        Slice of trajectory or window from mdtraj.iterload of time length t_max
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
    xyz = window[:, g2]

    l1 = g1.shape[0]
    l2 = g2.shape[0]
    lw = window.shape[0]

    rt = np.zeros((lw, l1, l2), dtype=float32)
    rtd = np.zeros((lw, l1, l2, 3), dtype=float32)

    for t in prange(lw):
        for i in prange(l1):
            for j in prange(l2):
                for coord in prange(3):
                    rtd[t,i,j,coord] = xyz[t,i,coord] - xyz[t,j,coord]
                    rtd[t,i,j,coord] -= bv[t,coord,coord] * round(rtd[t,i,j,coord] / bv[t,coord,coord])
                    rtd[t,i,j,coord] = rtd[t,i,j,coord] ** 2
                    rt[t,i,j] += rtd[t,i,j,coord]
                rt[t,i,j] = math.sqrt(rt[t,i,j])

                # remove self interaction part of g(r,t) by making the distance i == j large
                if g1[i] == g2[j]:
                    rt[t,i,i] = 9999.0

    return rt


@njit(['f4[:,:,:](f4[:,:,:],i8[:],i8[:],f4[:,:,:])'], **opts)
def _compute_rt_general_mic(window, g1, g2, bv):
    """
    Numba jitted and parallelised version of function to calculate
    the distance matrix between each atom in group 1 at time zero and
    each atom in group 2 at each frame supplied. Minimum image convention
    for all simulation boxes applied.

    Parameters
    ----------
    window : slice of mdtraj.trajectory
        Slice of trajectory or window from mdtraj.iterload of time length t_max
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
    xyz = window[:, g2]

    l1 = g1.shape[0]
    l2 = g2.shape[0]
    lw = window.shape[0]

    rt = np.zeros((lw, l1, l2), dtype=float32)
    rtd = np.zeros((lw, l1, l2, 3), dtype=float32)

    for t in prange(lw):
        bv_inv = np.linalg.inv(bv[t])
        for i in prange(l1):
            for j in prange(l2):
                s12 = bv_inv * xyz[t,i] - bv_inv * xyz[t,j]
                s12 -= np.rint(s12)
                r12 = bv[t] * s12
                # rt[t,i,j] = np.linalg.norm(r12)
                rt[t,i,j] = math.sqrt(r12[0, 0] ** 2 + r12[1, 1] ** 2 + r12[2, 2] ** 2)

                # remove self interaction part of g(r,t)
                if g1[i] == g2[j]:
                    rt[t,i,i] = 9999.0

    return rt
