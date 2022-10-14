"""
Provides functions to calculate the vanhove dynamic correlation function
G(r,t) between two groups of particles. Groups can also consist of single
particles.
"""

from .utils import _calculate_according_to_inputs
from ..common_tools.result_arrays import _vanhove_results as _construct_results_array


def vanhove(traj, g1, g2, top=None, pbc='ortho', n_windows=100, window_size=200, overlap=False, skip=1, stride=10,
            r_range=(0.0, 2.0), nbins=400, self_only=False):
    """
    Calculate $G(r,t)$ for two groups given in a trajectory.
    $G(r,t)$ is calculated for a smaller time frame (typically 2 ps). $G(r,t)$ is
    then averaged over the whole trajectory supplied.

    $G(r,t)$ is calculated as:

    $$G_{ab}(r,t) = \frac{1}{4\pi\rho_{b}N_{a}r^2} \sum^{N_a}_{i=1} \sum^{N_b}_{j=1} \delta(|r_i(0) - r_j(t)| - r)$$

    following the formulation by Shinohara, Y. et al. "Identifying
    water-anion correlated motion in aqueous solutions through van hove
    functions, The Journal of Physical Chemistry Letters", 10(22), 7119–7125
    (2019).  http://dx.doi.org/10.1021/acs.jpclett.9b02891

    Originally published by Léon van Hove in 1954: van Hove, L., "Correlations
    in space and time and born approximation scattering in systems of
    interacting particles", Physical Review, 95(1), 249–262 (1954).
    http://dx.doi.org/10.1103/physrev.95.249

    Parameters
    ----------
    traj : {string, mdtraj.trajectory, Generator}
        MDTraj trajectory, or Generator of trajectories (obtained using
    	mdtraj.iterload).
    g1 : list
        List of numpy arrays of atom indices representing the group to
    	calculate G(r,t) for.
    g2 : list
        List of numpy arrays of atom indices representing the group to
    	calculate G(r,t) with.

    Other parameters
    ----------------
    top : mdtraj.topology
        Topology object. Needed if trajectory given as a path to lazy-load.
    pbc : {string, NoneType}
        String representing the periodic boundary conditions of the simulation
    	cell. Currently, only 'ortho' for orthogonal simulation cells is
    	implemented.
    n_windows : integer
        Number of windows in which to split the trajectory (if a whole
    	trajectory is supplied).
    window_size : integer
        Number of frames in each window.
    overlap : {int, False}
        Positive integer number of frames between overlapping windows.
    skip : integer
        Number of frames to skip at the beginning if giving a path as
    	trajectory.
    stride : integer
        Number of frames in the original trajectory to skip between each
        calculation. E.g. stride = 10 means calculate distances only every
    	10th frame.
    r_range : tuple(float, float)
        Tuple over which r in G(r,t) is defined.
    nbins : integer
        Number of bins (points in r to consider) in G(r,t)

    Returns
    -------
    r : np.array
        bin centers of G(r,t)
    G_self  : np.array
        averaged function values of $G_{s}(r,t)$ for each time from t=0 considered
    G_distinct  : np.array
        averaged function values of $G_{d}(r,t)$ for each time from t=0 considered
    """
    G_self, G_distinct, g1, g2 = _construct_results_array(g1, g2, nbins, stride, window_size)

    r, G_self, G_distinct, n_windows = _calculate_according_to_inputs(G_self, G_distinct, g1, g2, n_windows, nbins,
                                                                      overlap, pbc, r_range, self_only, skip, stride,
                                                                      top, traj, window_size)
    G_self = G_self / n_windows
    G_distinct = G_distinct / n_windows
    return r, G_self, G_distinct

