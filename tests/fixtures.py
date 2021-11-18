from os.path import dirname
import mdtraj as md
from numba import set_num_threads
set_num_threads(2)
import mdenvironment as mde
import numpy as np
import pytest


@pytest.fixture(scope='session')
def paths():
    paths = {'test_dir': dirname(__file__)}
    paths['top_path'] = paths['test_dir'] + '/data/nacl_box.gro'
    paths['traj_path'] = paths['test_dir'] + '/data/nacl_box.xtc'
    paths['gmx_rdf'] = paths['test_dir'] + '/data/rdf.xvg'
    paths['mdtraj_r'] = paths['test_dir'] + '/data/mdtraj_r.txt'
    paths['mdtraj_gr'] = paths['test_dir'] + '/data/mdtraj_gr.txt'
    return paths


@pytest.fixture(scope='session')
def nacl_top(paths):
    top = md.load_topology(paths['top_path'])
    return top


@pytest.fixture(scope='session')
def nacl_traj(paths, nacl_top):
    traj = md.load(paths['traj_path'], top=nacl_top)
    return traj


@pytest.fixture(scope='session')
def gmx_rdf(paths):
    gmx_r, gmx_gr = np.loadtxt(paths['gmx_rdf'], comments=['#', '@'], unpack=True)
    gmx_r += 0.005
    return gmx_r, gmx_gr


@pytest.fixture(scope='session')
def mdtraj_groups(nacl_top, nacl_traj):
    groups = {g: nacl_traj.top.select(f'name {g}') for g in ['O', 'NA', 'CL', 'H']}
    return groups


@pytest.fixture(scope='session')
def mdtraj_rdf(paths):
    r = np.loadtxt(paths['mdtraj_r'])
    gr = np.loadtxt(paths['mdtraj_gr'])
    return r, gr
