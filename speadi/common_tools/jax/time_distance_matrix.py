import jax.numpy as np
from jax import jit, vmap


@jit
def _rt_ortho_mic(window, g1, g2, union1, union2, bv):
    """
    JAX/XLA jitted and parallelised version of function to calculate
    the distance matrix between each atom in group 1 at time zero and
    each atom in group 2 at each frame supplied. Minimum image convention
    for orthogonal simulation boxes applied.

    Parameters
    ----------
    window : slice of mdtraj.trajectory
        Slice of trajectory or window from mdtraj.iterload of time length t_max
        to calculate g(r,t) over.
    g1 : numpy.ndarray
        Numpy array of atom indices representing the group to calculate g(r,t) for.
    g2 : numpy.ndarray
        Numpy array of atom indices representing the group to calculate g(r,t) with.
    union : numpy.ndarray
        2d array containing the indices of g1 and g2 that are the same.
    bv : numpy.ndarray
        simulation box vectors (which vary over time) as supplied by
        mdtraj.trajectory.unitcell_vectors.

    Returns
    -------
    r12 : numpy.ndarray
        Numpy array containing the distance matrix.
    """
    r1 = window[:, g1]
    r2 = window[:, g2]
    bv_diags = bv.diagonal(axis1=1, axis2=2)[:, np.newaxis, np.newaxis, :]

    r12 = r1[:, :, np.newaxis, :] - r2[:, np.newaxis, :, :]
    r12 -= bv_diags * np.round(r12 / bv_diags)
    r12 = np.power(r12, 2)
    r12 = np.sum(r12, axis=3)
    r12 = np.sqrt(r12)

    # remove self interaction part of g(r,t) by making the distance i == j large
    r12 = r12.at[:, union1, union2].set(9999.0)

    return r12


@jit
def _rt_general_mic(window, g1, g2, union1, union2, bv):
    """
    JAX/XLA jitted and parallelised version of function to calculate
    the distance matrix between each atom in group 1 at time zero and
    each atom in group 2 at each frame supplied. Minimum image convention
    for all simulation boxes applied. Follows the scheme set out by

    Minimum image convention distance calculation follows the computation
    from scheme B.9 in Tuckerman, M. "Statistical Mechanics: Theory and
    Molecular Simulation", 2010.

    Parameters
    ----------
    window : slice of mdtraj.trajectory
        Slice of trajectory or window from mdtraj.iterload of time length t_max
        to calculate g(r,t) over.
    g1 : numpy.ndarray
        Numpy array of atom indices representing the group to calculate g(r,t) for.
    g2 : numpy.ndarray
        Numpy array of atom indices representing the group to calculate g(r,t) with.
    union : numpy.ndarray
        2d array containing the indices of g1 and g2 that are the same.
    bv : numpy.ndarray
        simulation box vectors (which vary over time) as supplied by
        mdtraj.trajectory.unitcell_vectors.

    Returns
    -------
    r12 : numpy.ndarray
        Numpy array containing the time-distance matrix.
    """
    r1 = window[:, g1]
    r2 = window[:, g2]
    # bv_inv = np.linalg.inv(bv)

    ## Reduce box vectors to a `reduced' basis
    bv1 = bv[:, 0]
    bv2 = bv[:, 1]
    bv3 = bv[:, 2]

    bv3 -= bv2 * np.round(bv3[:, 1] / bv2[:, 1])[:, np.newaxis]
    bv3 -= bv1 * np.round(bv3[:, 0] / bv1[:, 0])[:, np.newaxis]
    bv2 -= bv1 * np.round(bv2[:, 0] / bv1[:, 0])[:, np.newaxis]
    bv_inv = np.linalg.inv(np.array([bv1.T, bv2.T, bv3.T]).T)

    s1 = bv_inv[:, np.newaxis, :, :] * r1[:, :, :, np.newaxis]
    s2 = bv_inv[:, np.newaxis, :, :] * r2[:, :, :, np.newaxis]
    s12 = s1[:, :, np.newaxis, :] - s2[:, np.newaxis, :, :]
    s12 -= np.rint(s12)
    r12 = bv[:, np.newaxis, np.newaxis, :, :] * s12
    r12 = np.diagonal(r12, axis1=3, axis2=4)
    r12 = np.power(r12, 2)
    r12 = np.sum(r12, axis=3)
    r12 = np.sqrt(r12)

    # remove self interaction part of g(r,t) by making the distance i == j large
    r12 = r12.at[:, union1, union2].set(9999.0)

    return r12
