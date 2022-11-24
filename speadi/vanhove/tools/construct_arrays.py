import numpy as np


def _construct_results_array(g1, g2, nbins, stride, window_size):
    """
    Pre-allocates the array to store the results of each window, G_distinct, according to the parameters supplied to `vanhove()`.
    Returns the group index arrays as lists of arrays if not given to `vanhove()` as such.

    Parameters
    ----------
    g1
    g2
    nbins
    stride
    window_size

    Returns
    -------
    G_self
    G_distinct
    g1
    g2
    """
    if isinstance(g1, list) and isinstance(g2, list):
        G_self = np.zeros((len(g1), int(window_size // stride), nbins), dtype=np.float32)
        G_distinct = np.zeros((len(g1), len(g2), int(window_size // stride), nbins), dtype=np.float32)
    elif isinstance(g1, list) and not isinstance(g2, list):
        G_self = np.zeros((len(g1), int(window_size // stride), nbins), dtype=np.float32)
        G_distinct = np.zeros((len(g1), 1, int(window_size // stride), nbins), dtype=np.float32)
        g2 = [g2]
    elif not isinstance(g1, list) and isinstance(g2, list):
        G_self = np.zeros((1, int(window_size // stride), nbins), dtype=np.float32)
        G_distinct = np.zeros((1, len(g2), int(window_size // stride), nbins), dtype=np.float32)
        g1 = [g1]
    else:
        G_self = np.zeros((1, int(window_size // stride), nbins), dtype=np.float32)
        G_distinct = np.zeros((1, 1, int(window_size // stride), nbins), dtype=np.float32)
        g1 = [g1]
        g2 = [g2]
    return G_self, G_distinct, g1, g2