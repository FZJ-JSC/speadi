"""
Provides the helper function `get_patches` to capture groups (of variable size)
of particles that are found in the vicinity of another group of particles along
an atomistic simulation trajectory loaded with `MDTraj`.
"""

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import xarray as xr


def get_patches(traj, g1, g2, size=4):
    """
    Gives the N-nearest particles in a second group with respect to the
    first group (g1) along an MDTraj trajectory.

    Parameters
    ----------
    traj : mdtraj.trajectory
        MDTraj trajectory along which to calculate distances
    g1 : numpy.array
        Numpy array of particle indices representing the group to calculate
    	the nearest particles for
    g2 : numpy.array
        Numpy array of particle indices representing the group to calculate
    	the nearest particles with

    Other parameters
    ----------------
    size : int
    	Integer number of nearest particles to return as a patch

    Returns
    -------
    patches : xarray.Dataset
        xarray Dataset of the N-nearest particles in g2 with respect to the
    	particles in g2 for each frame in an MDTraj trajectory
    """
    pairs = traj.top.select_pairs(g1, g2)
    atom_strings = np.array(list(traj.top.atoms), dtype=np.str_)[g2]

    distances = md.compute_distances(traj, pairs)
    distances = np.reshape(distances, (traj.n_frames, g1.shape[0], g2.shape[0]))

    shortest_distance_index = np.argpartition(distances, range(size), axis=2)[:,:,:size]
    shortest_distances = np.take_along_axis(distances, shortest_distance_index, -1)
    nearest_particles = atom_strings[shortest_distance_index]

    sorting_index = np.argsort(nearest_particles)
    sorted_nearest_particles = np.take_along_axis(nearest_particles, sorting_index, -1)
    sorted_patches = np.empty((traj.n_frames, g1.shape[0]), dtype=object)
    for t in range(traj.n_frames):
        for particle in range(g1.shape[0]):
            sorted_patches[t, particle] = '--'.join(sorted_nearest_particles[t, particle])

    sorted_distances = np.take_along_axis(shortest_distances, sorting_index, -1)

    patches = xr.Dataset(
        {
            'patch': (['t', 'particleID'], sorted_patches),
            'patch_particles': (['t', 'particleID', 'atom'], sorted_nearest_particles),
            'patch_distances': (['t', 'particleID', 'atom'], sorted_distances)
        },
        coords = {
            't': range(traj.n_frames),
            'particleID': g1,
            'atom': range(size),
        },
        attrs = {'patch': 'labels', 'patch_particles': 'labels', 'patch_distances': 'nm'}
    )

    return patches


default_shells = {1: (0, .4), 2: (.4, .6), 3: (.6, .8)}


def shell_occurance(ds, shell, shell_dict=default_shells):
    """
    Counts the occurance of patches with at least one particle within a
    specified shell

    Parameters
    ----------
    ds : xarray.Dataset
    	xarray Dataset containing the N-nearest particles between two groups
    	as returned by `get_patches`
    shell : int
        Integer specifying the shell to calculate occurance for

    Other parameters
    ----------------
    shell_dict : dict
    	Dictionary of shells corresponding to a given distance interval

	Returns
    -------
    patches : numpy.array
    	Numpy array containing the patches with particles present in the
    	specified shell
    counts : numpy.array
    	Numpy array containing the total count of the patches contained in
    	the `patches` array
    """
    subset_shell = ds.patch.where(
        (np.any(ds.patch_distances >= shell_dict[shell][0], axis=2) &
         np.any(ds.patch_distances < shell_dict[shell][1], axis=2)),
        other='None'
        )

    patches, counts = np.unique(subset_shell, return_counts=True)

    ix = np.argsort(counts)
    patches = patches[ix][-2::-1]
    counts = counts[ix][-2::-1]

    return patches, counts


def shell_by_time(ds, shell, shell_dict=default_shells):
    """
    Counts the number of particles in total surrounding a first group of
    particles as a function of time along a trajectory

    Parameters
    ----------
    ds : xarray.Dataset
    	xarray Dataset containing the N-nearest particles between two groups
    	as returned by `get_patches`
    shell : int
        Integer specifying the shell to calculate occurance for

    Other parameters
    ----------------
    shell_dict : dict
    	Dictionary of shells corresponding to a given distance interval

    Returns
    -------
    N_t : numpy.array
    """
    subset_shell = ds.patch.where(
        (np.any(ds.patch_distances >= shell_dict[shell][0], axis=2) &
         np.any(ds.patch_distances < shell_dict[shell][1], axis=2)),
        drop=True
        )

    N_t = subset_shell.count(axis=1)

    return N_t


def check_res(df, res):
    """

    Parameters
    ----------
    df : pandas.DataFrame
    res : int

    Returns
    -------
    res_df : numpy.array
    """
    res_df = df[df.patch.str.contains(res)].sort_values('count', ascending=False)
    return res_df


def check_res_sum_traj(df, res):
    """

    Parameters
    ----------
    df : pandas.DataFrame
    res : int

    Returns
    -------
    res_df : numpy.array
    """
    res_df = df[df.patch.str.contains(res)].groupby('traj').sum()
    return res_df


def plot_interaction(df, res_list, shell='1st', **kwds):
    """
    Plotting helper function

    Parameters
    ----------
    df : pandas.DataFrame
    res_list : numpy.array
    	Numpy array containing the integer number of the residues in a protein
    	to plot for

    Other parameters
    ----------------
    shell : string
    	String with which to label the plot
	kwds : matplotlib.pyplot.subplots keywords
    """
    fig, ax = plt.subplots(figsize=kwds['figsize'])
    df = df.loc[res_list, shell, :].droplevel(1)
    conf_dict = {x: [('--', 'none'), ('-', 'full')][i] for i, x in enumerate(df.columns.levels[0])}
    temp_dict = {x: [('o', 'C0'), ('s', 'C1')][i] for i, x in enumerate(df.columns.levels[1])}
    for conf in conf_dict.keys():
        for temp in temp_dict.keys():
            ax.plot(df.index, df[(conf, temp)], temp_dict[temp][0]+conf_dict[conf][0], c=temp_dict[temp][1], fillstyle=conf_dict[conf][1], label=temp)
    ax.set(ylabel=f'p(Chloride in {shell} shell)',
           ylim=(0, None))
    handles, labels = ax.get_legend_handles_labels()
    legend_handles = [plt.Rectangle((0,0), 0, 0, color='w')] + handles[:2] + [plt.Rectangle((0,0), 0, 0, color='w')] + handles[2:]
    legend_labels = [df.columns.levels[0][0]] + labels[:2] + [df.columns.levels[0][1]] + labels[2:]
    leg = ax.legend(legend_handles, legend_labels, ncol=2, frameon=False)
    for i in [0, 3]:
        leg.get_texts()[i].set_weight('bold')
