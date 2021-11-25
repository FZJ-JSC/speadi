from fixtures import *
from itertools import product

set_num_threads(2)

refs = ['NA', 'CL']
pbc = ['ortho', 'general']
opt = [True, False]
parameter_sets = list(product(refs, pbc, opt))


def idfn(args):
    return f'ref: {args[0]}, pbc: {args[1]}, opt: {args[2]}'


@pytest.fixture(scope='module', params=parameter_sets, ids=idfn)
def mde_rdf(request, paths, nacl_top, mdtraj_groups):
    r, grt = mde.trrdf(paths['traj_path'], mdtraj_groups[request.param[0]], mdtraj_groups['O'], pbc=request.param[1],
                       opt=request.param[2], top=nacl_top, n_windows=20, window_size=10, stride=1, skip=0, r_range=(0.0, 1.2),
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


def test_rdf_results_mdtraj(mde_rdf, mdtraj_rdf):
    _, grt, ref = mde_rdf
    _, mdtraj_gr = mdtraj_rdf
    gr = np.mean(grt, axis=(0,1,2))

    np.testing.assert_allclose(gr, mdtraj_gr[ref], rtol=5e-2)
