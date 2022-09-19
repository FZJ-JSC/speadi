from fixtures import *
from itertools import product

set_num_threads(2)

ref = ['NA']
refs = [('NA', 'CL'), ('O', 'NA')]
pbc = ['ortho', 'general']
JAX_AVAILABLE = [True, False]
# NUMBA_AVAILABLE = [True, False]
NUMBA_AVAILABLE = [True]
single_ref_params = list(product(ref, pbc, JAX_AVAILABLE, NUMBA_AVAILABLE))
double_ref_params = list(product(refs, pbc, JAX_AVAILABLE, NUMBA_AVAILABLE))


def idfn(args):
    return f'ref: {args[0]}, pbc: {args[1]}, JAX_AVAILABLE: {args[2]}, NUMBA_AVAILABLE: {args[3]}'


@pytest.fixture(scope='module', params=single_ref_params, ids=idfn)
def mde_rdf(request, paths, nacl_top, mdtraj_groups):
    mde.JAX_AVAILABLE = request.param[2]
    mde.NUMBA_AVAILABLE = request.param[3]

    r, grt = mde.trrdf(paths['traj_path'], mdtraj_groups[request.param[0]], mdtraj_groups['O'], top=nacl_top,
                       pbc=request.param[1], n_windows=20, window_size=10, skip=0, stride=1, r_range=(0.0, 1.2),
                       nbins=120)
    return r, grt, request.param[0]


@pytest.fixture(scope='module', params=double_ref_params, ids=idfn)
def double_refs_mde_rdf(request, paths, nacl_top, mdtraj_groups):
    mde.JAX_AVAILABLE = request.param[2]
    mde.NUMBA_AVAILABLE = request.param[3]
    refs = request.param[0]
    ref_groups = [mdtraj_groups[ref] for ref in refs]

    r, grt = mde.trrdf(paths['traj_path'], ref_groups, mdtraj_groups['O'], top=nacl_top,
                       pbc=request.param[1], n_windows=20, window_size=10, skip=0, stride=1, r_range=(0.0, 1.2),
                       nbins=120)
    return r, grt, request.param[0]


@pytest.fixture(scope='module', params=double_ref_params, ids=idfn)
def quadruple_refs_mde_rdf(request, paths, nacl_top, mdtraj_groups):
    mde.JAX_AVAILABLE = request.param[2]
    mde.NUMBA_AVAILABLE = request.param[3]
    refs = request.param[0]
    ref_groups = [mdtraj_groups[ref] for ref in refs]

    r, grt = mde.trrdf(paths['traj_path'], ref_groups, [mdtraj_groups['O'],mdtraj_groups['NA']], top=nacl_top,
                       pbc=request.param[1], n_windows=20, window_size=10, skip=0, stride=1, r_range=(0.0, 1.2),
                       nbins=120)
    return r, grt, request.param[0]


def test_rdf_binning_gmx(mde_rdf, gmx_rdf):
    r, _, ref = mde_rdf
    gmx_r, _ = gmx_rdf

    np.testing.assert_allclose(r, gmx_r[ref][:-1], rtol=1e-4)


def test_rdf_binning_mdtraj(mde_rdf, mdtraj_rdf):
    r, _, ref = mde_rdf
    mdtraj_r, _ = mdtraj_rdf

    np.testing.assert_allclose(r, mdtraj_r[ref], rtol=1e-4)


@pytest.mark.skip('Binning differences with GROMACS causes small differences.')
def test_rdf_results_gmx(mde_rdf, gmx_rdf):
    _, grt, ref = mde_rdf
    _, gmx_gr = gmx_rdf
    gr = np.mean(grt, axis=(0,1,2))

    np.testing.assert_allclose(gr, gmx_gr[ref], rtol=5e-2)


def test_single_rdf_results_mdtraj(mde_rdf, mdtraj_rdf):
    _, grt, ref = mde_rdf
    _, mdtraj_gr = mdtraj_rdf
    gr = np.mean(grt, axis=(0,1,2))

    np.testing.assert_allclose(gr, mdtraj_gr[ref], rtol=5e-2)


def test_double_rdf_results_mdtraj(double_refs_mde_rdf, mdtraj_rdf):
    _, grt, refs = double_refs_mde_rdf
    _, mdtraj_gr = mdtraj_rdf
    gr1 = np.mean(grt[0], axis=(0,1))
    gr2 = np.mean(grt[1], axis=(0,1))

    np.testing.assert_allclose(gr1, mdtraj_gr[refs[0]], rtol=5e-2)
    np.testing.assert_allclose(gr2, mdtraj_gr[refs[1]], rtol=5e-2)


def test_quadruple_rdf_results_mdtraj(quadruple_refs_mde_rdf, mdtraj_rdf):
    _, grt, refs = quadruple_refs_mde_rdf
    _, mdtraj_gr = mdtraj_rdf
    gr1 = np.mean(grt[0][0], axis=(0))
    gr2 = np.mean(grt[1][0], axis=(0))

    np.testing.assert_allclose(gr1, mdtraj_gr[refs[0]], rtol=5e-2)
    np.testing.assert_allclose(gr2, mdtraj_gr[refs[1]], rtol=5e-2)
