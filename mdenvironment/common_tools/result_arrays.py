import numpy as np

from .check_inputs import prepare_input_group


def _trrdf_results(g1, g2, n_windows, nbins):
    """
    Pre-allocates the array to store the results of each window, g_rts, according to the parameters supplied to `trrdf()`.
    Returns the group index arrays as lists of arrays if not given to `trrdf()` as such.

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
    g1 = prepare_input_group(g1)
    g2 = prepare_input_group(g2)

    g_rts = np.zeros((len(g1), len(g2), n_windows, nbins), dtype=np.float32)

    return g_rts, g1, g2


def _vanhove_results(g1, g2, nbins, stride, window_size):
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
    g1 = prepare_input_group(g1)
    g2 = prepare_input_group(g2)

    G_self = np.zeros((len(g1), int(window_size // stride), nbins), dtype=np.float32)
    G_distinct = np.zeros((len(g1), len(g2), int(window_size // stride), nbins), dtype=np.float32)

    return G_self, G_distinct, g1, g2
