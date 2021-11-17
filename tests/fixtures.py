from os.path import dirname
import mdtraj as md
from numba import set_num_threads
set_num_threads(2)
import pytest
from gromacs.formats import XVG


@pytest.fixture(scope='session')
def paths():
    paths = {'test_dir': dirname(__file__)}
    paths['top_path'] = paths['test_dir'] + '/data/nacl_box.gro'
    paths['traj_path'] = paths['test_dir'] + '/data/nacl_box.xtc'
    paths['gmx_rdf'] = paths['test_dir'] + '/data/rdf.xvg'
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
    gmx_rdf = XVG(paths['gmx_rdf']).to_df()
    return gmx_rdf


@pytest.fixture(scope='session')
def mdtraj_groups(nacl_top, nacl_traj):
    groups = {g: nacl_traj.top.select(f'name {g}') for g in ['O', 'NA', 'CL', 'H']}
    return groups
