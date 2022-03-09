import numpy as np
from scipy.spatial.distance import cdist


def _compute_rt(window, g1, g2):
    rt_distinct = np.empty((window.shape[0], g1.shape[0], g2.shape[0]), dtype=np.float32)
    rt_self = np.empty((window.shape[0], g1.shape[0]), dtype=np.float32)
    r01 = window[0, g1]
    for t in range(rt_distinct.shape[0]):
        rt_distinct[t] = cdist(r01, window[t, g2])
        rt_self[t] = np.diagonal(rt_distinct[t])
        np.fill_diagonal(rt_distinct[t], 9999.0)

    return rt_self, rt_distinct


def _compute_rt_mic(window, g1, g2, box_vectors):
    bv = np.diag(box_vectors.mean(axis=0))

    rt_distinct = np.empty((window.shape[0], g1.shape[0], g2.shape[0]), dtype=np.float32)
    rt_self = np.empty((window.shape[0], g1.shape[0]), dtype=np.float32)
    r1 = window[0, g1]
    for t in range(window.shape[0]):
        r12 = _grid_sub(r1, window[t, g2])

        r12 -= bv * np.round(r12 / bv)

        r12 = r12**2
        r12 = r12.sum(axis=2)
        r12 = np.sqrt(r12)
        rt_distinct[t] = r12.T
        # remove self interaction part of G(r,t)
        rt_self[t] = np.diagonal(rt_distinct[t])
        np.fill_diagonal(rt_distinct[t], 9999.0)

    return rt_self, rt_distinct


def _grid_sub(r1, r2):
    r12 = np.stack([r1]*len(r2), axis=0) - np.stack([r2]*len(r1), axis=1)
    return r12