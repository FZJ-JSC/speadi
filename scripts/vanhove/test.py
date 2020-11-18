import mdtraj as md
from mdtraj.utils import ensure_type
import numpy as np
import matplotlib.pyplot as plt

from vanhove_nonperiodic import avg_grt, plot_grt

def prepare(iterator=True, single=False):
    topology = '../simbox/npt.gro'
    trajectory = '../simbox/npt.xtc'
    if iterator:
        t_all = md.iterload(trajectory, top=topology, chunk=200, skip=1)
        if single:
            t_all = next(t_all)
    else:
        t_all = md.load(trajectory, top=topology)
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


def plot(xmax=0.8):
    fig, ax = plot_grt(r, g_rt, pair='(O-O)', xmax=xmax)
    plt.show()


def test(stride=2):
    r, g_rt = avg_grt(t_all, sol, sol, n_chunks=50, stride=stride)
    globals().update(locals())


def load(f='test.npy'):
    with open(f, 'rb') as f:
        r = np.load(f)
        g_rt = np.load(f)
    globals().update(locals())


if __name__ == '__main__':
    prepare()
    test()
    # plot(xmax=0.8)
    save()
