from fixtures import *
set_num_threads(2)


@pytest.fixture(scope='module', params=['ortho', True])
def mde_rdf(request, paths, nacl_top, mdtraj_groups):
    r, grt = mde.trrdf(paths['traj_path'], mdtraj_groups['O'], mdtraj_groups['O'], pbc=request.param, opt=True, top=nacl_top,
                       n_windows=20, window_size=10, stride=1, skip=0, r_range=(0.0, 1.2), nbins=120)
    return r, grt


def test_rdf_binning_gmx(mde_rdf, gmx_rdf):
    gmx_r, _ = gmx_rdf
    r, _ = mde_rdf

    np.testing.assert_allclose(r, gmx_r[:-1], rtol=1e-4)


def test_rdf_binning_mdtraj(mde_rdf, mdtraj_rdf):
    mdtraj_r, _ = mdtraj_rdf
    r, _ = mde_rdf

    np.testing.assert_allclose(r, mdtraj_r, rtol=1e-4)


@pytest.mark.skip('Binning differences with GROMACS causes small differences.')
def test_rdf_results_gmx(mde_rdf, gmx_rdf):
    _, gmx_gr = gmx_rdf
    _, grt = mde_rdf
    gr = np.mean(grt, axis=(0,1,2))

    np.testing.assert_allclose(gr, gmx_gr, atol=1e-2)


def test_rdf_results_mdtraj(mde_rdf, mdtraj_rdf):
    _, mdtraj_gr = mdtraj_rdf
    _, grt = mde_rdf
    gr = np.mean(grt, axis=(0,1,2))

    np.testing.assert_allclose(gr, mdtraj_gr, atol=1e-2)
