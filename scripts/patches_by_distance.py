import numpy as np
import mdtraj as md
import xarray as xr

def get_patches(traj, g1, g2, size=4):
    pairs = traj.top.select_pairs(g1, g2)
    atom_strings = np.array(list(traj.top.atoms), dtype=np.str_)[g2]

    distances = md.compute_distances(traj, pairs)
    distances = np.reshape(distances, (traj.n_frames, g1.shape[0], g2.shape[0]))

    shortest_distance_index = np.argpartition(distances, range(size), axis=2)[:,:,:size]
    shortest_distances = np.take_along_axis(distances, shortest_distance_index, -1)
    nearest_atoms = atom_strings[shortest_distance_index]

    sorting_index = np.argsort(nearest_atoms)
    sorted_nearest_atoms = np.take_along_axis(nearest_atoms, sorting_index, -1)
    sorted_patches = np.empty((traj.n_frames, g1.shape[0]), dtype=object)
    for t in range(traj.n_frames):
        for ion in range(g1.shape[0]):
            sorted_patches[t, ion] = '--'.join(sorted_nearest_atoms[t, ion])

    sorted_distances = np.take_along_axis(shortest_distances, sorting_index, -1)

    patches = xr.Dataset(
        {
            'patch': (['t', 'ionID'], sorted_patches),
            'patch_atoms': (['t', 'ionID', 'atom'], sorted_nearest_atoms),
            'patch_distances': (['t', 'ionID', 'atom'], sorted_distances)
        },
        coords = {
            't': range(traj.n_frames),
            'ionID': g1,
            'atom': range(size),
        },
        attrs = {'patch': 'labels', 'patch_atoms': 'labels', 'patch_distances': 'nm'}
    )

    return patches


def shell_occurance(ds, shell):
    shell_dict = {1: (0, .4), 2: (.4, .6), 3: (.6, .8)}
    subset_shell = ds.patch.where(
        	(np.any(ds.patch_distances >= shell_dict[shell][0], axis=2) &\
             np.any(ds.patch_distances < shell_dict[shell][1], axis=2)),
        	other='None'
        )

    patches, counts = np.unique(subset_shell, return_counts=True)

    ix = np.argsort(counts)
    patches = patches[ix][-2::-1]
    counts = counts[ix][-2::-1]

    return patches, counts


def shell_by_time(ds, shell):
    shell_dict = {1: (0, .4), 2: (.4, .6), 3: (.6, .8)}
    subset_shell = ds.patch.where(
        	(np.any(ds.patch_distances >= shell_dict[shell][0], axis=2) &\
             np.any(ds.patch_distances < shell_dict[shell][1], axis=2)),
        	drop=True
        )

    N_t = subset_shell.count(axis=1)

    return N_t


def check_res(df, res):
    res_df = df[df.patch.str.contains(res)].sort_values('count', ascending=False)
    return res_df


def check_res_sum_traj(df, res):
    res_df = df[df.patch.str.contains(res)].groupby('traj').sum()
    return res_df

# from matplotlib import rc
# rc('text', usetex=True)

def plot_interaction(df, res_list, shell='1st', **kwds):
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
