import numpy as np


def _compute_rt_general_mic(window, g1, g2, box_vectors):
    rt_distinct = np.zeros((window.shape[0], g1.shape[0], g2.shape[0]), dtype=np.float32)
    rt_self = np.zeros((window.shape[0], g1.shape[0]), dtype=np.float32)
    r1 = window[0, g1]
    xyz = window[:, g2]

    for t in range(rt_distinct.shape[0]):
        bv_inv = np.linalg.inv(box_vectors[t])

        s1 = bv_inv * r1[:, :, np.newaxis]
        s2 = bv_inv * xyz[t, :, :, np.newaxis]
        s12 = _grid_sub(s1, s2)
        s12 -= np.rint(s12)
        r12 = box_vectors[t] * s12
        rt_diag = np.diagonal(r12.T)
        rt_sq = np.sum(rt_diag**2, axis=2)
        rt_distinct[t] = np.sqrt(rt_sq)

        # remove self interaction part of G(r,t)
        rt_self[t] = np.diagonal(rt_distinct[t])
        np.fill_diagonal(rt_distinct[t], 9999.0)

    return rt_self, rt_distinct


def _compute_rt_ortho_mic(window, g1, g2, box_vectors):
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