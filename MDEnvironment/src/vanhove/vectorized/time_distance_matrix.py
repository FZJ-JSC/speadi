import numpy as np
from scipy.spatial.distance import cdist


def _compute_rt(xyz, g1, g2):
    rt = np.empty((xyz.shape[0], g1.shape[0], g2.shape[0]))
    r01 = xyz[0, g1]
    for t in range(rt.shape[0]):
        rt[t] = cdist(r01, xyz[t, g2])
        np.fill_diagonal(rt[t], np.inf)

    return rt


def _compute_rt_mic(xyz, g1, g2, box_vectors):
    bv = np.diag(box_vectors.mean(axis=0))

    rt = np.empty((xyz.shape[0], g1.shape[0], g2.shape[0]), dtype=np.float32)
    r1 = xyz[0, g1]
    for t in range(xyz.shape[0]):
        r12 = _grid_sub(r1, xyz[t, g2])

        r12 -= bv * np.round(r12 / bv)

        r12 = r12**2
        r12 = r12.sum(axis=2)
        r12 = np.sqrt(r12)
        rt[t] = r12.T
        # remove self interaction part of G(r,t)
        np.fill_diagonal(rt[t], np.inf)

    return rt


def _grid_sub(r1, r2):
    r12 = np.stack([r1]*len(r2), axis=0) - np.stack([r2]*len(r1), axis=1)
    return r12