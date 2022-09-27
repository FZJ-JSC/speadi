import numpy as np
from mdtraj.geometry.distance import _distance_mic


def _rt_mic(window, g1, g2, bv, orthogonal=False):
    pairs = np.array(np.meshgrid(g1, g2)).T.reshape(-1, 2)
    rt = _distance_mic(window, pairs, bv, orthogonal)
    rt = rt.reshape((window.shape[0], g1.shape[0], g2.shape[0]))
    part_union, g1_union, g2_union = np.intersect1d(g1, g2, return_indices=True, assume_unique=True)

    for i, j in zip(g1_union, g2_union):
        rt[:, i, j] = 9999.0

    return rt