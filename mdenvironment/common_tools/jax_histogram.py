"""
Simplified histogram function, that is able to be parallelised using JAX/XLA.
"""
import jax.numpy as np
from jax import jit, vmap


@jit
def _get_bin(x, bin_edges):
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    bin = np.array(n * (x - a_min) / (a_max - a_min), int)

    return bin


_mapped_get_bin = vmap(_get_bin, (0, None), 0)


@jit
def _histogram(rt_array, bin_edges):
    rt_array = np.ravel(rt_array)
    bins_map = _mapped_get_bin(rt_array, bin_edges)
    counts = np.bincount(bins_map, None, length=bin_edges.shape[0])[:-1]

    return counts
