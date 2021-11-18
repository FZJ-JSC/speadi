from fixtures import *
set_num_threads(2)


@pytest.fixture(scope='module')
def mde_vhf_rdf(paths, nacl_top, mdtraj_groups):
    r, Grt = mde.Grt(paths['traj_path'], mdtraj_groups['O'], mdtraj_groups['O'], pbc='ortho', opt=True, top=nacl_top,
                     n_windows=200, window_size=1, stride=1, skip=0, r_range=(0.0, 1.2), nbins=120)
    return r, Grt


@pytest.fixture(scope='module')
def mdtraj_rdf(nacl_top, nacl_traj, mdtraj_groups):
    r = np.loadtxt('data/mdtraj_r.txt')
    gr = np.loadtxt('data/mdtraj_gr.txt')
    return r, gr


def test_rdf_binning_gmx(mde_vhf_rdf, gmx_rdf):
    gmx_r, _ = gmx_rdf.T.values[0], gmx_rdf.T.values[1]
    r, _ = mde_vhf_rdf

    np.testing.assert_allclose(r, gmx_r[:-1], rtol=1e-4)


def test_rdf_binning_mdtraj(mde_vhf_rdf, mdtraj_rdf):
    mdtraj_r, _ = mdtraj_rdf[0], mdtraj_rdf[1]
    r, _ = mde_vhf_rdf

    np.testing.assert_allclose(r, mdtraj_r, rtol=1e-4)


@pytest.mark.skip('Binning differences with GROMACS causes small differences.')
def test_rdf_results_gmx(mde_vhf_rdf, gmx_rdf):
    _, gmx_gr = gmx_rdf.T.values[0], gmx_rdf.T.values[1]
    _, Grt = mde_vhf_rdf
    gr = np.mean(Grt, axis=(0,1,2))

    np.testing.assert_allclose(gr, gmx_gr, atol=1e-2)


def test_rdf_results_mdtraj(mde_vhf_rdf, mdtraj_rdf):
    _, mdtraj_gr = mdtraj_rdf[0], mdtraj_rdf[1]
    _, Grt = mde_vhf_rdf
    gr = np.mean(Grt, axis=(0,1,2))

    np.testing.assert_allclose(gr, mdtraj_gr, atol=1e-2)
