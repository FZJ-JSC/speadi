import matplotlib.pyplot as plt
import numpy as np

from ...multiline import multiline

def plot_Grt(r, g_rt, xmax=10.0, t_max=2.0, ymax='peak', title='pair', pair='', save='trrdf.pdf', cmap='bwr', ax=None, cbar=True, xlabel=True, ylabel=True, cax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    c = np.linspace(0, t_max, g_rt.shape[0])
    rs = np.tile(r * 10, (g_rt.shape[0], 1))
    lc = multiline(rs, g_rt, c, cmap=cmap, ax=ax)
    if cbar:
        if 'cax' in dir():
            axcb = plt.colorbar(lc, cax=cax)
        else:
            axcb = plt.colorbar(lc, ax=ax)
        axcb.set_label('t / ps')

    if ymax == 'peak':
        ymax = g_rt.max()
    ax.set_ylim(0.0, ymax)
    ax.yaxis.grid(True, which='minor')
    ax.set_xlim(0.0, xmax)
    ax.xaxis.grid(True, which='minor')
    if ylabel:
        ax.set_ylabel('G(r,t)')
    else:
        ax.set_yticks([])
    if xlabel:
        ax.set_xlabel('r / Å')
    else:
        ax.set_xticks([])
    if title == 'pair':
        ax.set_title(f'van Hove dynamic correlation function {pair}')
    else:
        ax.set_title(title)
    if save:
        plt.savefig(save)
    return fig, ax


def plot_map(r, g_rt, xmax=10.0, ymax=2.0, vlim=(0.90, 1.10), total_t=2.0, total_r=20.0, title='pair', save='map.pdf', pair='', cmap='viridis', ax=None, cbar=True, xlabel=True, ylabel=True, cax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    extent = (0, xmax, 0, ymax)
    xmix = int(g_rt.shape[1] / total_r * xmax)
    ymix = int(g_rt.shape[0] / total_t * ymax)

    assert xmax <= total_r, f'x-axis cannot extend beyond the maximum radial bin ({total_r} Å)'
    assert ymax <= total_t, f'y-axis cannot extend beyond the maximum time ({total_t} ps)'

    image = ax.imshow(g_rt[:ymix,:xmix], origin='lower', vmin=vlim[0], vmax=vlim[1], extent=extent, aspect='auto', cmap=cmap)
    if cbar:
        if 'cax' in dir():
            axcb = plt.colorbar(image, cax=cax, extend='both')
        else:
            axcb = plt.colorbar(image, ax=ax, extend='both')
        axcb.set_label('G(r,t)')

    # ax.set_ylim(0.0, ymax)
    # ax.set_xlim(0.0, xmax)

    if ylabel:
        ax.set_ylabel('t / ps')
    else:
        ax.set_yticks([])
    if xlabel:
        ax.set_xlabel('r / Å')
    else:
        ax.set_xticks([])
    if title == 'pair':
        ax.set_title(f'van Hove dynamic correlation function {pair}')
    else:
        ax.set_title(title)
    if save:
        plt.savefig(save)
    return fig, ax


def plot_dual_Grt(r, g_rt, xmax=2.0, ymax=2.0, vlim=(0.90, 1.10), total_t=2.0, save='both.pdf', pair='', cmap='viridis'):
    _, ax1 = plot_grt(r, g_rt, xmax=xmax, ymax='peak', save=False, pair=None, cmap=cmap)
    _, ax2 = plot_map(r, g_rt, xmax=xmax, ymax=ymax, vlim=(0.90, 1.10),
                      total_t=2.0, save=False, pair=None, cmap=cmap)

    fig, axs = plt.subplots(1, 2, sharex=True)

    axs[0] = ax1
    axs[1] = ax2

    if save:
        plt.savefig(save)
    return fig, axs
