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

    $$G_{ab}(r,t) = \\frac{1}{4\\pi\\rho_{b}N_{a}r^2} \\sum^{N_a}_{i=1}
      \\sum^{N_b}_{j=1} \\delta(|r_i(0) - r_j(t)| - r)$$


    following the formulation by Shinohara, Y. et al. [1]

    The function was originally published by Léon van Hove in 1954. [2]

    Parameters
    ----------
    traj : {string, mdtraj.trajectory, Generator}
        MDTraj trajectory, or Generator of trajectories (obtained using
    	mdtraj.iterload).
    g1 : list
        List of numpy arrays of atom indices representing the group to
    	calculate $G(r,t)$ for.
    g2 : list
        List of numpy arrays of atom indices representing the group to
    	calculate $G(r,t)$ with.
    top : mdtraj.topology
        Topology object. Needed if trajectory given as a path to lazy-load.
    pbc : {string, NoneType}
        String representing the periodic boundary conditions of the simulation
    	cell. Currently, only 'ortho' for orthogonal simulation cells is
    	implemented.
    n_windows : int
        Number of windows in which to split the trajectory (if a whole
    	trajectory is supplied).
    window_size : int
        Number of frames in each window.
    overlap : {int, False}
        Positive integer number of frames between overlapping windows.
    skip : int
        Number of frames to skip at the beginning if giving a path as
    	trajectory.
    stride : integer
        Number of frames in the original trajectory to skip between each
        calculation. E.g. stride = 10 means calculate distances only every
    	10th frame.
    r_range : tuple(float, float)
        Tuple over which r in $G(r,t)$ is defined.
    nbins : int
        Number of bins (points in r to consider) in $G(r,t)$

    Returns
    -------
    r : np.array
        bin centers of $G(r,t)$
    G_self  : np.ndarray
        averaged function values of $G_{s}(r,t)$ for each time from t=0 considered
    G_distinct  : np.ndarray
        averaged function values of $G_{d}(r,t)$ for each time from t=0 considered

    Examples
    --------
    First, import both `MDTraj` and `MDEnvironment` together.
    >>> import mdtraj as md
    >>> import mdenvironment as mde

    Then, point to a particle simulation topology and trajectory (e.g. a Molecular Dynamics Simulation using `Gromacs`).
    >>> topology = './topology.gro'
    >>> trajectory = './trajectory.xtc'

    Next, load the topology file using `MDTraj` and start defining reference and target groups.
    >>> top = md.load_topology(topology)
    >>> na = top.select('name NA')
    >>> cl = top.select('name CL')
    >>> protein_by_atom = [top.select(f'index {ix}') for
    >>>                    ix in top.select('protein and not type H')]

    Finally, run the van Hove Function (VHF) by calling `vhf()`.
    >>> r, g_s, g_d = mde.vanhove(trajectory, protein_by_atom, [na, cl], top=top,
    >>>                           n_windows=1000, window_size=500, skip=0,
    >>>                           pbc='general', stride=1, nbins=400)

    The outputs are

     - the centre points of the radial bins `r`

     - the $G_s(r,t)$ self part of the correlation function with shape
      $N$(reference groups)$\\times N$(\\tau windows)$\\times N$(radial bins)

     - the $G_s(r,t)$ self part of the correlation function with shape
      $N$(reference groups)$\\times N$(target groups)$\\times N$(\\tau windows)$\\times N$(radial bins)

    References
    -------
    [1] Shinohara, Y., Matsumoto, R., Thompson, M. W. et al., "Identifying water-anion
           correlated motion in aqueous solutions through van Hove functions,"
           The Journal of Physical Chemistry Letters, 10(22), 7119–7125 (2019).
           http://dx.doi.org/10.1021/acs.jpclett.9b02891

    [2] van Hove, L., "Correlations in space and time and Born approximation scattering
           in systems of interacting particles", Physical Review, 95(1), 249–262 (1954).
           http://dx.doi.org/10.1103/physrev.95.249

    """
    G_self, G_distinct, g1, g2 = _construct_results_array(g1, g2, nbins, stride, window_size)

    r, G_self, G_distinct, n_windows = _calculate_according_to_inputs(G_self, G_distinct, g1, g2, n_windows, nbins,
                                                                      overlap, pbc, r_range, self_only, skip, stride,
                                                                      top, traj, window_size)
    G_self = G_self / n_windows
    G_distinct = G_distinct / n_windows
    return r, G_self, G_distinct

