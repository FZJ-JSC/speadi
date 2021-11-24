"""
Provides functions to calculate the radial distribution function (RDF) between
two groups of particles for specified windows along a trajectory g(r,t).
Groups can also consist of single particles.
"""

from .tools.utils import _construct_results_array, _calculate_according_to_inputs
from ..common_tools.check_numba import check_numba

NUMBA_AVAILABLE = check_numba()
if NUMBA_AVAILABLE:
    from numba import (get_num_threads, set_num_threads)
    set_num_threads(get_num_threads())


def trrdf(traj, g1, g2, top=None, pbc='ortho', opt=NUMBA_AVAILABLE, n_windows=100, window_size=200, skip=1, stride=10,
          r_range=(0.0, 2.0), nbins=400, raw_counts=False):
    """
    Calculate g(r,t) for two groups given in a trajectory. g(r) is
    calculated for each frame in the trajectory, then averaged over
    specified windows of time, returning g(r,t) (where t represents
    the window time along the trajectory).

    Parameters
    ----------
    traj : str
    	String pointing to the location of a trajectory that MDTraj is
    	able to load
    g1 : list
        List of numpy arrays of atom indices representing the group to
    	calculate G(r,t) for
    g2 : list
        List of numpy arrays of atom indices representing the group to
    	calculate G(r,t) with

    Other parameters
    ----------------
    top : mdtraj.topology
        Topology object. Needed if trajectory given as a path to lazy-load.
    pbc : {string, NoneType}
        String representing the periodic boundary conditions of the
    	simulation cell. Currently, only 'ortho' for orthogonal simulation cells
    	is implemented.
    n_windows : int
        Number of windows in which to split the trajectory (if a whole
    	trajectory is supplied).
    window_size : int
        Number of frames in each window.
    skip : int
        Number of frames to skip at the beginning if giving a path as
    	trajectory.
    stride : int
        Number of frames in the original trajectory to skip between each
        calculation. E.g. stride = 10 means calculate distances only every
    	10th frame.
    r_range : tuple(float, float)
        Tuple over which r in g(r,t) is defined.
    nbins : int
        Number of bins (points in r to consider) in g(r,t)

    Returns
    -------
    r : np.array
        bin centers of g(r,t)
    g_rt : np.array
        averaged function values of g(r,t) for each time from t=0 considered
    """
    g_rt, g1, g2 = _construct_results_array(g1, g2, n_windows, nbins)

    r, g_rt = _calculate_according_to_inputs(g1, g2, g_rt, n_windows, nbins, opt, pbc, r_range, raw_counts, skip,
                                             stride, top, traj, window_size)

    return r, g_rt
