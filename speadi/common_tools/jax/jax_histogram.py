"""JIT-compatible histograms with NumPy's bin-edge conventions."""
import jax.numpy as np
from jax import jit


@jit
def _histogram(rt_array, bin_edges):
    """Count [left, right) bins, including the final right edge.

    Explicit edges keep the same boundary definition as NumPy/MDTraj;
    out-of-range values are ignored, including for positive lower bounds.
    """
    counts, _ = np.histogram(rt_array, bins=bin_edges)
    return counts
