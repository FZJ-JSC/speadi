import numpy as np


def get_union(ref, target):
    """Gives the union of a single reference and a single target group, allowing $G(r,t)$ to be split
    into $G_s(r,t)$ and $G_t(r,t)$.

    Parameters
    ----------
    ref : np.ndarray, dtype=int
        The array containing the indices of the reference particles.
    target : np.ndarray, dtype=int
        The array containing the indices of the target particles.

    Returns
    -------
    union : np.ndarray, dtype=int
        List containing the indices contained in both the reference and the target group.

    """
    part_union, ref_indices, target_indices = np.intersect1d(ref, target, return_indices=True, assume_unique=True)
    union = [ref_indices, target_indices]

    return union


def get_all_unions(g1, g2, g1_lens, g2_lens):
    """Gives the union of all reference groups and all target groups, allowing $G(r,t)$ to be split
    into $G_s(r,t)$ and $G_t(r,t)$.

    Parameters
    ----------
    g1 : list
        List object containing arrays of reference groups.
    g2 : list
        List object containing arrays of target groups.
    g1_lens : np.ndarray, dtype=int
        Integer number of elements in each reference group contained in g1.
    g2_lens : np.ndarray, dtype=int
        Integer number of elements in each target group contained in g2.

    Returns
    -------
    unions : dict
        Dictionary containing the overlapping indices in each combination of groups in g1 and g2.

    """
    Ng1 = g1_lens.shape[0]
    Ng2 = g2_lens.shape[0]

    unions = {}
    for i in range(Ng1):
        unions[str(i)] = {}
        for j in range(Ng2):
            unions[str(i)][str(j)] = get_union(g1[i], g2[j])

    return unions
