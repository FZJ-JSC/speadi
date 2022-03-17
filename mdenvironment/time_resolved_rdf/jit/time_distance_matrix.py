import math

import numpy as np
from numba import njit, float32, prange

opts = dict(parallel=True, fastmath=True, nogil=True, cache=False, debug=False)


@njit(['f4[:,:,:](f4[:,:,:],i8[:],i8[:],f4[:,:,:])'], **opts)
def _rt_ortho_mic(window, g1, g2, bv):
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
    r1 = window[:, g1]
    r2 = window[:, g2]

    l1 = g1.shape[0]
    l2 = g2.shape[0]
    lw = window.shape[0]

    rt = np.zeros((lw, l1, l2), dtype=float32)
    rtd = np.zeros((lw, l1, l2, 3), dtype=float32)

    for t in prange(lw):
        for i in prange(l1):
            for j in prange(l2):
                for coord in prange(3):
                    rtd[t,i,j,coord] = r1[t,i,coord] - r2[t,j,coord]
                    rtd[t,i,j,coord] -= bv[t,coord,coord] * round(rtd[t,i,j,coord] / bv[t,coord,coord])
                    rtd[t,i,j,coord] = rtd[t,i,j,coord] ** 2
                    rt[t,i,j] += rtd[t,i,j,coord]
                rt[t,i,j] = math.sqrt(rt[t,i,j])

                # remove self interaction part of g(r,t) by making the distance i == j large
                if g1[i] == g2[j]:
                    rt[t,i,i] = 9999.0

    return rt


@njit(['f4[:,:,:](f4[:,:,:],i8[:],i8[:],f4[:,:,:])'], **opts)
def _rt_general_mic(window, g1, g2, bvt):
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
    r1 = window[:,g1]
    r2 = window[:,g2]

    l1 = g1.shape[0]
    l2 = g2.shape[0]
    lw = window.shape[0]

    rt = np.zeros((lw, l1, l2), dtype=float32)
    rtd = np.zeros((lw, l1, l2, 3), dtype=float32)

    for t in prange(lw):
        bv = bvt[t].flatten()
        bv1 = np.array([bv[0], bv[3], bv[6], 0], dtype=float32)
        bv2 = np.array([bv[1], bv[4], bv[7], 0], dtype=float32)
        bv3 = np.array([bv[2], bv[5], bv[8], 0], dtype=float32)

        bv3 -= bv2 * round(bv3[1] / bv2[1])
        bv3 -= bv1 * round(bv3[0] / bv1[0])
        bv2 -= bv1 * round(bv2[0] / bv1[0])
        recip_box_size = np.array([1.0 / bv1[0], 1.0 / bv2[1], 1.0 / bv3[2]], dtype=float32)

        for i in prange(l1):
            for j in prange(l2):
                for coord in prange(3):
                    rtd[t,i,j,coord] = r1[t,i,coord] - r2[t,j,coord]
                    rtd[t,i,j,coord] -= bv3[coord] * round(rtd[t,i,j,2] * recip_box_size[2])
                    rtd[t,i,j,coord] -= bv2[coord] * round(rtd[t,i,j,1] * recip_box_size[2])
                    rtd[t,i,j,coord] -= bv1[coord] * round(rtd[t,i,j,0] * recip_box_size[0])

                    min_dist2 = 9999.0
                    for x in prange(-1, 2):
                        ra = rtd[t,i,j,coord] + bv1[coord] * x
                        for y in prange(-1, 2):
                            rb = ra + bv2[coord] * y
                            for z in prange(-1, 2):
                                rc = rb + bv3[coord] * z
                                dist2 = rc * rc
                                if dist2 <= min_dist2:
                                    min_dist2 = dist2

                    rt[t,i,j] += min_dist2

                rt[t,i,j] = math.sqrt(rt[t,i,j])

                # remove self interaction part of g(r,t)
                if g1[i] == g2[j]:
                    rt[t,i,j] = 9999.0

    return rt
