def check_jax():
    JAX_AVAILABLE = False
    try:
        from jax import __version__ as __jax_version
        from jaxlib import __version__ as __jaxlib_version
        JAX_AVAILABLE = True
    except ImportError:
        JAX_AVAILABLE = False

    return JAX_AVAILABLE


def check_numba():
    NUMBA_AVAILABLE = False
    try:
        from numba import __version__ as __numba_version__
        NUMBA_AVAILABLE = True
    except ImportError:
        NUMBA_AVAILABLE = False

    return NUMBA_AVAILABLE

