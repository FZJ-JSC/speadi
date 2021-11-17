import numpy as np


def _construct_results_array(g1, g2, n_windows, nbins):
    """
    Pre-allocates the array to store the results of each window, g_rts, according to the parameters supplied to `grt()`.
    Returns the group index arrays as lists of arrays if not given to `grt()` as such.

    Parameters
    ----------
    g1
    g2
    n_windows
    nbins

    Returns
    -------
    g_rts
    g1
    g2
    """
    if isinstance(g1, list) and isinstance(g2, list):
        g_rts = np.zeros((len(g1), len(g2), n_windows, nbins), dtype=np.float32)
    elif isinstance(g1, list) and not isinstance(g2, list):
        g_rts = np.zeros((len(g1), 1, n_windows, nbins), dtype=np.float32)
        g2 = [g2]
    elif not isinstance(g1, list) and isinstance(g2, list):
        g_rts = np.zeros((1, len(g2), n_windows, nbins), dtype=np.float32)
        g1 = [g1]
    else:
        g_rts = np.zeros((1, 1, n_windows, nbins), dtype=np.float32)
        g1 = [g1]
        g2 = [g2]
    return g_rts, g1, g2