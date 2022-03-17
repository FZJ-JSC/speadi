from os.path import dirname
import mdtraj as md
from numba import set_num_threads
set_num_threads(4)
import mdenvironment as mde
import numpy as np
import pytest


@pytest.fixture(scope='session')
def paths():
    paths = {'test_dir': dirname(__file__)}
    paths['top_path'] = paths['test_dir'] + '/data/nacl_box.gro'
    paths['traj_path'] = paths['test_dir'] + '/data/nacl_box.xtc'

    for ref in ['O', 'NA', 'CL']:
        paths[ref] = {
            'gmx': paths['test_dir'] + f'/data/{ref}_rdf.xvg',
            'mdtraj_r': paths['test_dir'] + f'/data/{ref}_mdtraj_r.txt',
            'mdtraj_gr': paths['test_dir'] + f'/data/{ref}_mdtraj_gr.txt'
        }

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
def gmx_rdf(request, paths):
    gmx_r = {}
    gmx_gr = {}
    for ref in ['O', 'NA', 'CL']:
        r, gr = np.loadtxt(paths[ref]['gmx'], comments=['#', '@'], unpack=True)
        gmx_r[ref] = r + 0.005
        gmx_gr[ref] = gr
    return gmx_r, gmx_gr


@pytest.fixture(scope='session')
def mdtraj_groups(nacl_top, nacl_traj):
    groups = {g: nacl_traj.top.select(f'name {g}') for g in ['O', 'NA', 'CL', 'H']}
    return groups


@pytest.fixture(scope='session')
def mdtraj_rdf(paths):
    mdtraj_r = {}
    mdtraj_gr = {}
    for ref in ['O', 'NA', 'CL']:
        mdtraj_r[ref] = np.loadtxt(paths[ref]['mdtraj_r'])
        mdtraj_gr[ref] = np.loadtxt(paths[ref]['mdtraj_gr'])
    return mdtraj_r, mdtraj_gr
