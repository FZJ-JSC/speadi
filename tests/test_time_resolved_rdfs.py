from fixtures import *
set_num_threads(2)


@pytest.fixture(scope='session')
def mde_rdf(paths, nacl_top, mdtraj_groups):
    r, grt = mde.grt(paths['traj_path'], mdtraj_groups['O'], mdtraj_groups['O'], pbc='ortho', opt=True, top=nacl_top,
                     n_windows=10, window_size=20, stride=1, skip=1, r_range=(0.0, 2.0), nbins=400)
    return r, grt


@pytest.fixture(scope='session')
def mdtraj_rdf(nacl_top, nacl_traj, mdtraj_groups):
    oxygen_pairs = nacl_traj.top.select_pairs(mdtraj_groups['O'], mdtraj_groups['O'])
    r, gr = md.compute_rdf(nacl_traj, pairs=oxygen_pairs, r_range=(0, 2.0), periodic=True, opt=True, n_bins=400)
    return r, gr


def test_rdf_binning_gmx(mde_rdf, gmx_rdf):
    gmx_r, _ = gmx_rdf.T.values[0], gmx_rdf.T.values[1]
    gmx_r += 0.0025  # Add half a bin width; GROMACS lists bins by starting radius, not bin centre.
    r, _ = mde_rdf

    np.testing.assert_allclose(r, gmx_r, rtol=1e-4)


def test_rdf_binning_mdtraj(mde_rdf, mdtraj_rdf):
    mdtraj_r, _ = mdtraj_rdf[0], mdtraj_rdf[1]
    r, _ = mde_rdf

    np.testing.assert_allclose(r, mdtraj_r, rtol=1e-4)


def test_rdf_results_gmx(mde_rdf, gmx_rdf):
    _, gmx_gr = gmx_rdf.T.values[0], gmx_rdf.T.values[1]
    _, grt = mde_rdf
    gr = np.mean(grt, axis=(0,1,2))

    np.testing.assert_allclose(gr, gmx_gr, rtol=1e-4)


def test_rdf_results_mdtraj(mde_rdf, mdtraj_rdf):
    _, mdtraj_gr = mdtraj_rdf[0], mdtraj_rdf[1]
    _, grt = mde_rdf
    gr = np.mean(grt, axis=(0,1,2))

    np.testing.assert_allclose(gr, mdtraj_gr, rtol=1e-4)


def test_mdtraj_vs_gmx(gmx_rdf, mdtraj_rdf):
    _, gmx_gr = gmx_rdf.T.values[0], gmx_rdf.T.values[1]
    _, mdtraj_gr = mdtraj_rdf[0], mdtraj_rdf[1]

    np.testing.assert_allclose(gmx_gr, mdtraj_gr, rtol=1e-4)
