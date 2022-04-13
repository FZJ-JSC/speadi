import numpy as np


def get_union(g1, g2):
    part_union, g1_union, g2_union = np.intersect1d(g1, g2, return_indices=True, assume_unique=True)
    unions = np.array([g1_union, g2_union])

    return unions


def get_all_unions(g1, g2, g1_lens, g2_lens):
    Ng1 = g1_lens.shape[0]
    Ng2 = g2_lens.shape[0]

    unions = []
    if Ng1 > 1 and Ng2 > 1:
        unions = [[get_union(g1i, g2j) for g2j in g2] for g1i in g1]
    elif Ng1 > 1 and Ng2 == 1:
        unions = [[get_union(g1i, g2)] for g1i in g1]
    elif Ng1 == 1 and Ng2 > 1:
        unions = [[get_union(g1, g2j)] for g2j in g2]
    elif Ng1 == 1 and Ng2 == 1:
        unions = [[get_union(g1, g2)],]

    unions = np.array(unions, dtype=np.int32)

    return unions