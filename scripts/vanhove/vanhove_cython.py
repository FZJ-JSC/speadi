import mdtraj as md
from mdtraj.utils import ensure_type
import numpy as np
from typing import Generator
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiline import multiline
from scipy.spatial.distance import cdist
from rt_mic_p import rt_mic_p

import boost_histogram as bh
import os

threads = os.cpu_count()
print(f'Using {threads} threads!')



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


def vrt(xyz, g1, g2):
    rt = np.empty((len(xyz[:,0,0]), len(g1), len(g2)))
    r01 = xyz[0, g1]
    for t in range(len(rt)):
        rt[t] = cdist(r01, xyz[t, g2])
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


def compute_grt_boost(rt_array, traj, r_range=(0.0, 2.0), bins=400):
    Ni = len(rt_array[0,:,0])
    Nj = len(rt_array[0,0,:])
    n_frames = len(rt_array[:,0,0])
    g_rt = np.empty((n_frames, bins))
    axis = bh.axis.Regular(bins, r_range[0], r_range[1], overflow=False, underflow=False)
    for t in range(n_frames):
        hist = bh.Histogram(axis, storage=bh.storage.Int64())
        hist.fill(rt_array[t].ravel(), threads=threads)
        g_r = hist.view()
        g_rt[t] = g_r

    edges = axis.edges
    r = 0.5 * (edges[1:] + edges[:-1])
    r_vol = 4.0/3.0 * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    Nj_density = Nj / traj.unitcell_volumes.mean()

    # Shinohara's funny norming function doesn't lead to recognisable results...
    # norm = 4 * np.pi * N_density * N * r**2

    # Use normal RDF norming for each timestep
    norm = Nj_density * r_vol * Ni
    g_rt = g_rt / norm

    return r, g_rt


def avg_grt(traj, g1, g2, pbc=None, omp=False, n_chunks=100, stride=10):
    g_rts = []
    if isinstance(traj, md.core.trajectory.Trajectory):
        traj = traj[::stride]
        chunk_size = int(2.0 / traj.timestep)
        n_chunks = int(np.floor(len(traj.time) / chunk_size))
        chunks = np.array_split(traj.xyz, n_chunks)
        unitcell_vectors = np.array_split(traj.unitcell_vectors, n_chunks)
        for chunk, unitcell_vector_chunk in zip(tqdm(chunks, total=n_chunks, desc='Progress over trajectory'), unitcell_vectors):
            if pbc == 'ortho':
                if omp:
                    rt_array = rt_mic_p(chunk, g1, g2, unitcell_vector_chunk)
                else:
                    rt_array = rt_mic(chunk, g1, g2, unitcell_vector_chunk)
            else:
                rt_array = vrt(chunk, g1, g2)
            r, g_rt = compute_grt_boost(rt_array, traj)
            g_rts.append(g_rt)

    elif isinstance(traj, Generator):
        for chunk in tqdm(traj, total=n_chunks, desc='Progress over trajectory'):
            if pbc == 'ortho':
                if omp:
                    rt_array = rt_mic_p(chunk.xyz[::stride], g1, g2, chunk[::stride].unitcell_vectors)
                else:
                    rt_array = rt_mic(chunk.xyz[::stride], g1, g2, chunk[::stride].unitcell_vectors)
            else:
                rt_array = vrt(chunk, g1, g2)
            r, g_rt = compute_grt_boost(rt_array, chunk)
            g_rts.append(g_rt)

    else:
        raise TypeError('You must input either an MDTraj trajectory, or a generator of such.')

    g_rt = np.mean(np.array(g_rts), axis=0)
    return r, g_rt


def plot_grt(r, g_rt, xmax=0.8, ymax='peak', save='grt.pdf', pair='', cmap='bwr'):
    fig, ax = plt.subplots()

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

def plot_map(r, g_rt, xmax=2.0, ymax=2.0, vlim=(0.90, 1.10), total_t=2.0, save='map.pdf', pair='', cmap='viridis'):
    fig, ax = plt.subplots()

    extent = (0, total_t, 0, r.max())

    image = plt.imshow(g_rt, origin='lower', vmin=vlim[0], vmax=vlim[1], extent=extent, aspect='auto', cmap=cmap)
    axcb = fig.colorbar(image, extend='both')
    axcb.set_label('G(r,t)')

    ax.set_ylim(0.0, ymax)
    ax.set_xlim(0.0, xmax)

    ax.set_ylabel('t / ps')
    ax.set_xlabel('r / Å')
    ax.set_title(f'van Hove dynamic correlation function {pair}')
    if save:
        plt.savefig(save)
    return fig, ax


def plot_both(r, g_rt, xmax=2.0, ymax=2.0, vlim=(0.90, 1.10), total_t=2.0, save='both.pdf', pair='', cmap='viridis'):
    _, ax1 = plot_grt(r, g_rt, xmax=xmax, ymax='peak', save=False, pair=None, cmap=cmap)
    _, ax2 = plot_map(r, g_rt, xmax=xmax, ymax=ymax, vlim=(0.90, 1.10),
                      total_t=2.0, save=False, pair=None, cmap=cmap)

    fig, axs = plt.subplots(1, 2, sharex=True)

    axs[0] = ax1
    axs[1] = ax2

    if save:
        plt.savefig(save)
    return fig, axs


def grid_sub(r1, r2):
    r12 = np.stack([r1]*len(r2), axis=0) - np.stack([r2]*len(r1), axis=1)
    return r12


def rt_mic(xyz, g1, g2, box_vectors):
    bv = np.diag(box_vectors.mean(axis=0))

    rt = np.empty((xyz.shape[0], len(g1), len(g2)), dtype=np.float32)
    r1 = xyz[0, g1]
    for t in range(len(xyz)):
        r12 = grid_sub(r1, xyz[t, g2])

        r12 -= bv * np.round(r12 / bv)

        r12 = r12**2
        r12 = r12.sum(axis=2)
        r12 = np.sqrt(r12)
        rt[t] = r12
        # remove self interaction part of G(r,t)
        np.fill_diagonal(rt[t], np.inf)

    return rt


