import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

from MDEnvironment import Grt
from MDEnvironment import plot_grt, plot_map


def prepare(iterator=True, single=False):
    topology = '../simbox/npt.gro'
    trajectory = '../simbox/npt.xtc'
    if iterator:
        t_all = md.iterload(trajectory, top=topology, chunk=200, skip=1)
        if single:
            t_all = next(t_all)
    else:
        t_all = md.load(trajectory, top=topology)[1:]
        if single:
            t_all = t_all[-200::5]

    top = md.load_topology(topology)
    solPairs = top.select_pairs('name O', 'name O')
    sol = top.select('name O')
    globals().update(locals())


def save():
    with open('test.npy', 'wb') as f:
        np.save(f, r)
        np.save(f, g_rt)


def plot(xmax=0.8, save='Grt.pdf'):
    fig, ax = plot_grt(r, G_rt, pair='(O-O)', xmax=xmax, save=save)
    plt.show()


def pmap(xmax=0.8, cmap='viridis', save='map.pdf'):
    fig, ax = plot_map(r, g_rt, pair='(O-O)', xmax=xmax, cmap=cmap, save=save)
    plt.show()


def test(opt=True, pbc='ortho', stride=5):
    r, G_rt = Grt(t_all, sol, sol, n_windows=50, opt=opt, pbc=pbc, stride=stride)
    globals().update(locals())


def load(f='test.npy'):
    with open(f, 'rb') as f:
        r = np.load(f)
        g_rt = np.load(f)
    globals().update(locals())


if __name__ == '__main__':
    prepare(iterator=False)
    test(opt=True, pbc='ortho')
    # plot(xmax=0.8)
    save()
