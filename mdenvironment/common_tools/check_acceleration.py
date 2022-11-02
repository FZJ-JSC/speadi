def check_jax():
    """Checks the availability of `jax` and the `jaxlib` library in the current python environment.

    Returns
    -------
    JAX_AVAILABLE : bool
        Boolean variable that other functions can use to import the correct accelerated versions of code.

    """
    JAX_AVAILABLE = False
    try:
        from jax import __version__ as __jax_version
        from jaxlib import __version__ as __jaxlib_version
        JAX_AVAILABLE = True
    except ImportError:
        JAX_AVAILABLE = False

    return JAX_AVAILABLE


def check_numba():
    """Checks the availability of `numba` in the current python environment.

    Returns
    -------
    NUMBA_AVAILABLE : bool
        Boolean variable that other functions can use to import the correct accelerated versions of code.

    """
    NUMBA_AVAILABLE = False
    try:
        from numba import __version__ as __numba_version__
        NUMBA_AVAILABLE = True
    except ImportError:
        NUMBA_AVAILABLE = False

    return NUMBA_AVAILABLE

