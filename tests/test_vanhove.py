from fixtures import *
from itertools import product

set_num_threads(2)

refs = ['NA']
pbc = ['ortho', 'general']
JAX_AVAILABLE = [True, False]
NUMBA_AVAILABLE = [True, False]
parameter_sets = list(product(refs, pbc, JAX_AVAILABLE, NUMBA_AVAILABLE))


def idfn(args):
    return f'ref: {args[0]}, pbc: {args[1]}, JAX_AVAILABLE: {args[2]}, NUMBA_AVAILABLE: {args[3]}'


@pytest.fixture(scope='module', params=parameter_sets, ids=idfn)
def speadi_vhf_rdf(request, paths, nacl_top, mdtraj_groups):
    sp.JAX_AVAILABLE = request.param[2]
    sp.NUMBA_AVAILABLE = request.param[3]

    r, G_self, G_distinct = sp.vanhove(paths['traj_path'], mdtraj_groups[request.param[0]], mdtraj_groups['O'],
                                        top=nacl_top, pbc=request.param[1], n_windows=200, window_size=1, skip=0,
                                        stride=1, r_range=(0.0, 1.2), nbins=120)
    return r, G_self, G_distinct, request.param[0]


def test_rdf_binning_gmx(mde_vhf_rdf, gmx_rdf):
    r, _, _, ref = speadi_vhf_rdf
    gmx_r, _ = gmx_rdf

    np.testing.assert_allclose(r, gmx_r[ref][:-1], rtol=1e-4)


def test_rdf_binning_mdtraj(mde_vhf_rdf, mdtraj_rdf):
    r, _, _, ref = speadi_vhf_rdf
    mdtraj_r, _ = mdtraj_rdf

    np.testing.assert_allclose(r, mdtraj_r[ref], rtol=1e-4)


@pytest.mark.skip('Binning differences with GROMACS causes small differences.')
def test_rdf_results_gmx(mde_vhf_rdf, gmx_rdf):
    _, _, G_distinct, ref = speadi_vhf_rdf
    _, gmx_gr = gmx_rdf
    gr = np.mean(G_distinct, axis=(0,1,2))

    np.testing.assert_allclose(gr, gmx_gr[ref], rtol=5e-2)


def test_rdf_results_mdtraj(mde_vhf_rdf, mdtraj_rdf):
    _, _, G_distinct, ref = speadi_vhf_rdf
    _, mdtraj_gr = mdtraj_rdf
    gr = np.mean(G_distinct, axis=(0,1,2))

    np.testing.assert_allclose(gr, mdtraj_gr[ref], rtol=5e-2)
