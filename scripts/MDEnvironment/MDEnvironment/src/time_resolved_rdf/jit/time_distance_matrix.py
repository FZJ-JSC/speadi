import math

import numpy as np
from numba import njit, float32, prange


@njit(['f4[:,:,:](f4[:,:,:],i8[:],i8[:],f4[:,:,:])'], parallel=True, fastmath=True, nogil=True)
def _compute_rt_mic_numba(window, g1, g2, bv):
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
    # rt0 = window[0]
    # r1 = rt0[g1]
    r1 = window[:, g1]
    xyz = window[:, g2]

    rt = np.empty((window.shape[0], g1.shape[0], g2.shape[0]), dtype=float32)
    rtd = np.empty((window.shape[0], g1.shape[0], g2.shape[0], 3), dtype=float32)

    frames = window.shape[0]

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
                # remove self interaction part of g(r,t) by making the distance i == j large
                if i == j:
                    rt[t][i][j] = 99.0

    return rt