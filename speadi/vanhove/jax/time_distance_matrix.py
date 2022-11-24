import jax.numpy as np
from jax import jit


@jit
def _rtau_general_mic(window, g1, g2, union, bv):
    """
    JAX/XLA jitted and parallelised version of function to calculate
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
    g1 : numpy.ndarray
        Numpy array of atom indices representing the group to calculate G(r,t)
    	for.
    g2 : numpy.ndarray
        Numpy array of atom indices representing the group to calculate G(r,t)
    	with.
    union : numpy.ndarray
        2d numpy array of overlapping indices between g1 and g2.
    bv : numpy.ndarray
        simulation box vectors (which vary over time) as supplied by
        mdtraj.trajectory.unitcell_vectors.

    Returns
    -------
    rtau_self : numpy.ndarray
        Numpy array containing the time-distance matrix between identical particles.
    rtau_distinct : numpy.ndarray
        Numpy array containing the time-distance matrix between distinct particles.
    """
    r01 = window[0, g1]
    rt2 = window[:, g2]
    # bv_inv = np.linalg.inv(bv)

    ## Reduce box vectors to a `reduced' basis
    bv1 = bv[:, 0]
    bv2 = bv[:, 1]
    bv3 = bv[:, 2]

    bv3 -= bv2 * np.round(bv3[:, 1] / bv2[:, 1])[:, np.newaxis]
    bv3 -= bv1 * np.round(bv3[:, 0] / bv1[:, 0])[:, np.newaxis]
    bv2 -= bv1 * np.round(bv2[:, 0] / bv1[:, 0])[:, np.newaxis]
    bv_inv = np.linalg.inv(np.array([bv1.T, bv2.T, bv3.T]).T)

    s01 = bv_inv[0, np.newaxis, :, :] * r01[:, :, np.newaxis]
    st2 = bv_inv[:, np.newaxis, :, :] * rt2[:, :, :, np.newaxis]
    s01t2 = s01[:, np.newaxis, :] - st2[:, np.newaxis, :, :]
    s01t2 -= np.rint(s01t2)

    r01t2 = bv[:, np.newaxis, np.newaxis, :, :] * s01t2
    r01t2 = np.diagonal(r01t2, axis1=3, axis2=4)
    r01t2 = np.power(r01t2, 2)
    r01t2 = np.sum(r01t2, axis=3)
    r01t2 = np.sqrt(r01t2)

    # separate the self interaction part of G_self from G_distinct
    rtau_self = np.diagonal(r01t2, axis1=1, axis2=2)
    rtau_distinct = r01t2.at[:, union[0], union[1]].set(9999.0)

    return rtau_self, rtau_distinct


@jit
def _rtau_ortho_mic(window, g1, g2, union, bv):
    """
    JAX/XLA jitted and parallelised version of function to calculate
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
    g1 : numpy.ndarray
        Numpy array of atom indices representing the group to calculate G(r,t)
    	for.
    g2 : numpy.ndarray
        Numpy array of atom indices representing the group to calculate G(r,t)
    	with.
    union : numpy.ndarray
        2d numpy array of overlapping indices between g1 and g2.
    bv : numpy.ndarray
        simulation box vectors (which vary over time) as supplied by
        mdtraj.trajectory.unitcell_vectors.

    Returns
    -------
    rtau_self : numpy.ndarray
        Numpy array containing the time-distance matrix between identical particles.
    rtau_distinct : numpy.ndarray
        Numpy array containing the time-distance matrix between distinct particles.
    """
    r01 = window[0, g1]
    rt2 = window[:, g2]
    bv_diags = bv.diagonal(axis1=1, axis2=2)[:, np.newaxis, np.newaxis, :]

    r01t2 = r01[:, np.newaxis, :] - rt2[:, np.newaxis, :, :]
    r01t2 -= bv_diags * np.round(r01t2 / bv_diags)
    r01t2 = np.power(r01t2, 2)
    r01t2 = np.sum(r01t2, axis=3)
    r01t2 = np.sqrt(r01t2)

    # separate the self interaction part of G_self from G_distinct
    rtau_self = np.diagonal(r01t2, axis1=1, axis2=2)
    rtau_distinct = r01t2.at[:, union[0], union[1]].set(9999.0)

    return rtau_self, rtau_distinct


@jit
def _rtau_ortho_mic_self(window, g1, bv):
    """
    JAX/XLA jitted and parallelised version of function to calculate
    the distance matrix between each atom in group 1 at time zero and
    each atom in group 2 at each subsequent time (frame) supplied. Minimum
    image convention for orthogonal simulation boxes applied.

    Parameters
    ----------
    window : slice of mdtraj.trajectory
        Slice of trajectory or window from mdtraj.iterload of time length t_max
        to calculate G(r,t) over.
    g1 : numpy.ndarray
        Numpy array of atom indices representing the group to calculate G(r,t)
    	for.
    bv : numpy.ndarray
        simulation box vectors (which vary over time) as supplied by
        mdtraj.trajectory.unitcell_vectors.

    Returns
    -------
    rtau_self : numpy.ndarray
        Numpy array containing the time-distance matrix between identical particles.
    """
    r01 = window[0, g1]
    rt1 = window[:, g1]
    bv_diags = bv.diagonal(axis1=1, axis2=2)[:, np.newaxis, np.newaxis, :]

    rtau_self = r01[:, :] - rt1[:, :, :]
    rtau_self -= bv_diags * np.round(rtau_self / bv_diags)
    rtau_self = np.power(rtau_self, 2)
    rtau_self = np.sum(rtau_self, axis=3)
    rtau_self = np.sqrt(rtau_self)

    rtau_self = np.diagonal(rtau_self, axis1=1, axis2=2)

    return rtau_self