def check_numba():
    NUMBA_AVAILABLE = False
    try:
        from numba import __version__ as __numba_version__
        NUMBA_AVAILABLE = True
    except ImportError:
        NUMBA_AVAILABLE = False

    return NUMBA_AVAILABLE
