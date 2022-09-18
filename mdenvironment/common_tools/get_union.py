import numpy as np


def get_union(ref, target):
    part_union, ref_indices, target_indices = np.intersect1d(ref, target, return_indices=True, assume_unique=True)
    union = [ref_indices, target_indices]

    return union


def get_all_unions(g1, g2, g1_lens, g2_lens):
    Ng1 = g1_lens.shape[0]
    Ng2 = g2_lens.shape[0]

    unions = {}
    for i in range(Ng1):
        unions[str(i)] = {}
        for j in range(Ng2):
            unions[str(i)][str(j)] = get_union(g1[i], g2[j])


    return unions