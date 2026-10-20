"""Regression coverage for bounded device work and host TRRDF results."""
import importlib

import numpy as np
import pytest

pytest.importorskip('jax')
from speadi.common_tools.jax.time_distance_matrix import _rt_ortho_mic, _rt_general_mic
from speadi.time_resolved_rdf.jax.histogram_distances import _compute_grt
from speadi import trrdf

backend = importlib.import_module('speadi.time_resolved_rdf.jax.add_distribution_function_time_slice')


@pytest.mark.parametrize('orthogonal', [True, False])
@pytest.mark.parametrize('pair_chunk,frame_chunk', [(2, 2), (3, 1), (128, 16)])
def test_blocked_matches_dense(orthogonal, pair_chunk, frame_chunk):
    rng = np.random.default_rng(41)
    xyz = rng.uniform(0, 2, (5, 9, 3)).astype(np.float32)
    boxes = np.tile(np.diag([2., 2.5, 3.]), (5, 1, 1)).astype(np.float32)
    if not orthogonal:
        boxes[:, 1, 0] = .2
        boxes[:, 2, :2] = [.1, .3]
    # Changing volumes detects incorrectly normalizing individual frame blocks.
    boxes *= np.linspace(1, 1.2, 5, dtype=np.float32)[:, None, None]
    volumes = np.linalg.det(boxes)
    groups1 = [np.array([7, 1, 3, 1], dtype=np.int32), np.array([0], dtype=np.int32)]
    groups2 = [np.array([3, 2, 7, 4, 1], dtype=np.int32), np.array([8, 5], dtype=np.int32)]
    edges = np.linspace(.15, 1.5, 12, dtype=np.float32)
    output = np.full((2, 2, 3, 11), -1, dtype=np.float32)
    result = backend._append_grts_mic(output, 1, xyz, groups1, groups2, None, None,
                                     boxes, volumes, (.15, 1.5), 11, None, edges,
                                     orthogonal, pair_chunk, frame_chunk)
    assert result is output
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[:, :, 0], -1)
    np.testing.assert_array_equal(result[:, :, 2], -1)
    distance = _rt_ortho_mic if orthogonal else _rt_general_mic
    for i, g1 in enumerate(groups1):
        for j, g2 in enumerate(groups2):
            union = np.array(np.where(g1[:, None] == g2[None, :]), dtype=np.int32)
            expected = _compute_grt(distance(xyz, g1, g2, union, boxes), volumes, edges)
            np.testing.assert_allclose(result[i, j, 1], expected, rtol=2e-6, atol=1e-7)


def test_distance_dispatch_is_bounded(monkeypatch):
    calls = []
    def count_group(xyz1, xyz2, ids1, ids2, boxes, edges, orthogonal, tile1, tile2):
        calls.append((xyz1.shape, xyz2.shape))
        assert xyz1.shape[0] <= 2 and xyz2.shape[0] <= 2
        assert tile1 <= 3 and tile2 <= 3
        assert xyz1.shape[1] == len(ids1) == 9
        assert xyz2.shape[1] == len(ids2) == 6
        return np.zeros((2, len(edges) - 1), dtype=np.uint32)
    monkeypatch.setattr(backend, '_count_group_pair', count_group)
    xyz = np.zeros((5, 100, 3), dtype=np.float32)
    boxes = np.tile(np.eye(3, dtype=np.float32), (5, 1, 1))
    g1 = [np.arange(7, dtype=np.int32)]
    g2 = [np.arange(4, dtype=np.int32)]
    backend._append_grts_mic(np.zeros((1, 1, 1, 2), dtype=np.float32), 0, xyz, g1, g2,
                            None, None, boxes, np.ones(5), (0, 1), 2, None,
                            np.linspace(0, 1, 3), True, 3, 2)
    # Synchronize once per frame chunk, rather than once per atom tile.
    assert len(calls) == 3


def test_wide_device_counts():
    low = np.array([2**32 - 2, 17], dtype=np.uint32)
    high = np.array([0, 3], dtype=np.uint32)
    updated, carry = backend._add_counts((low, high), np.array([5, 2], dtype=np.uint32))
    np.testing.assert_array_equal(updated, [3, 19])
    np.testing.assert_array_equal(carry, [1, 3])


def test_public_trrdf_returns_host_arrays(tmp_path, monkeypatch):
    import mdtraj as md
    import speadi as sp
    monkeypatch.setattr(sp, 'JAX_AVAILABLE', True)
    top = md.Topology()
    residue = top.add_residue('TEST', top.add_chain())
    for _ in range(9):
        top.add_atom('C', md.element.carbon, residue)
    xyz = np.random.default_rng(12).uniform(0, 2, (6, 9, 3)).astype(np.float32)
    trajectory = md.Trajectory(xyz, top, unitcell_lengths=np.full((6, 3), 2.),
                               unitcell_angles=np.full((6, 3), 90.))
    path = str(tmp_path / 'blocks.xtc')
    trajectory.save_xtc(path)
    args = dict(top=top, n_windows=2, window_size=3, skip=0, stride=1, nbins=12,
                r_range=(0., 1.))
    g1 = np.array([0, 2, 4], dtype=np.int32)
    g2 = np.array([2, 3, 5, 7], dtype=np.int32)
    _, expected = trrdf(path, g1, g2, **args)
    def forbidden_union(*args):
        pytest.fail('JAX must not build full group intersections')
    utils = importlib.import_module('speadi.time_resolved_rdf.utils')
    monkeypatch.setattr(utils, 'get_all_unions', forbidden_union)
    original = backend._count_distance_block
    def bounded(xyz, ids1, ids2, *args):
        assert xyz.shape[0] <= 2
        assert len(ids1) <= 2 and len(ids2) <= 2
        return original(xyz, ids1, ids2, *args)
    monkeypatch.setattr(backend, '_count_distance_block', bounded)
    for _ in range(2):
        r, result = trrdf(path, g1, g2, pair_chunk_size=2, frame_chunk_size=2, **args)
        assert isinstance(r, np.ndarray) and isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize('name', ['pair_chunk_size', 'frame_chunk_size'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True, None])
def test_invalid_chunk_sizes(name, value):
    with pytest.raises(ValueError, match=name):
        trrdf(None, [], [], **{name: value})
