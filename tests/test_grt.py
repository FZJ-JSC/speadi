from os.path import dirname
import mdtraj as md
import MDEnvironment as mde
import pytest

@pytest.fixture
def paths():
    paths = {'test_dir': dirname(__file__)}
    paths['top_path'] = paths['test_dir'] + '/data/nacl_box.gro'
    paths['traj_path'] = paths['test_dir'] + '/data/nacl_box.xtc'
    return paths


@pytest.fixture
def nacl_top(paths):
    top = md.load_topology(paths['top_path'])
    return top


@pytest.fixture
def nacl_traj(paths, nacl_top):
    traj = md.load(paths['traj_path'], top=nacl_top)
    return traj


def test_top(nacl_top):
    assert nacl_top == nacl_top


def test_traj(nacl_traj):
    assert nacl_traj == nacl_traj


def test_import_grt():
    assert mde.grt == mde.grt