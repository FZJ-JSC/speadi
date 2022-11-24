import numpy as np
from mdtraj.geometry._geometry import _dist_mic


def _rt_mic(window, g1, g2, bv, orthogonal=False):
    pairs = np.array(np.meshgrid(g1, g2)).T.reshape(-1, 2)
    rt = np.empty((window.shape[0], pairs.shape[0]), dtype=np.float32).copy(order='C')
    _dist_mic(window.copy(order='C'), pairs.copy(order='C'), bv.copy(order='C'), rt, orthogonal)
    rt = rt.reshape((window.shape[0], len(g1), len(g2)))
    part_union, g1_union, g2_union = np.intersect1d(g1, g2, return_indices=True, assume_unique=True)

    for i, j in zip(g1_union, g2_union):
        rt[:, i, j] = 9999.0

    return rt