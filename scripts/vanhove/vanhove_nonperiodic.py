import mdtraj as md
from mdtraj.utils import ensure_type
import numpy as np
from typing import Generator
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiline import multiline


def dist(r1, r2, sum_axis=2):
    d = np.sqrt(np.sum((r1 - r2)**2, axis=sum_axis))
    return d


def rt(xyz, g1, g2):
    rt = np.empty((len(xyz[:,0,0]), len(g1), len(g2)))
    for i, atom in enumerate(g1):
        rt[:, i, :] = dist(xyz[0, atom], xyz[:, g2])

    for t in range(len(rt)):
        np.fill_diagonal(rt[t], np.inf)

    return rt


def compute_grt(rt_array, traj, r_range=(0.0, 2.0), bins=400):
    Ni = len(rt_array[0,:,0])
    Nj = len(rt_array[0,0,:])
    n_frames = len(rt_array[:,0,0])
    g_rt = np.empty((n_frames, bins))
    for t in range(n_frames):
        g_r, edges = np.histogram(rt_array[t], range=r_range, bins=bins)
        g_rt[t] = g_r

    r = 0.5 * (edges[1:] + edges[:-1])
    r_vol = 4.0/3.0 * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    Nj_density = Nj / traj.unitcell_volumes.mean()

    # Shinohara's funny norming function doesn't lead to recognisable results...
    # norm = 4 * np.pi * N_density * N * r**2

    # Use normal RDF norming for each timestep
    norm = Nj_density * r_vol * Ni
    g_rt = g_rt / norm

    return r, g_rt


def avg_grt(traj, g1, g2, n_chunks=100, stride=10):
    g_rts = []
    if isinstance(traj, md.core.trajectory.Trajectory):
        chunk_size = int(2.0 / traj.timestep)
        n_chunks = int(np.floor(len(traj.time) / chunk_size))
        chunks = np.array_split(traj.xyz, n_chunks)
        for chunk in tqdm(chunks, desc='Progress over trajectory'):
            rt_array = rt(chunk, g1, g2)
            r, g_rt = compute_grt(rt_array, traj)
            g_rts.append(g_rt)

    elif isinstance(traj, Generator):
        for chunk in tqdm(traj, total=n_chunks, desc='Progress over trajectory'):
            rt_array = rt(chunk.xyz[::stride], g1, g2)
            r, g_rt = compute_grt(rt_array, chunk)
            g_rts.append(g_rt)

    else:
        raise TypeError('You must input either an MDTraj trajectory, or a generator of such.')

    g_rt = np.mean(np.array(g_rts), axis=0)
    return r, g_rt


def plot_grt(r, g_rt, xmax=0.8, ymax='peak', save='grt.pdf', pair='', cmap='bwr'):
    fig, ax = plt.subplots()

    # colors = plt.cm.jet(np.linspace(0, 1, len(g_rt)))
    # for i, g_ry in enumerate(g_rt):
    #     ax.plot(r, g_ry, color=colors[i])

    c = np.linspace(0, 2.0, len(g_rt))
    rs = np.tile(r, (len(g_rt), 1))
    lc = multiline(rs, g_rt, c, cmap=cmap, ax=ax)
    axcb = fig.colorbar(lc)
    axcb.set_label('t / ps')

    if ymax == 'peak':
        ymax = g_rt[:, 25:].max()
    ax.set_ylim(0.0, ymax)
    ax.yaxis.grid(True, which='minor')
    ax.set_xlim(0.0, xmax)
    ax.xaxis.grid(True, which='minor')
    ax.set_ylabel('G(r,t)')
    ax.set_xlabel('r / Å')
    ax.set_title(f'van Hove dynamic correlation function {pair}')

    if save:
        plt.savefig(save)
    return fig, ax

