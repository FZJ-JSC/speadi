"""Exact-count regression tests, independent of RDF normalization."""
import importlib

import numpy as np
import pytest


@pytest.fixture(params=['jax', 'numba'])
def histogram(request):
    pytest.importorskip(request.param)
    module = importlib.import_module(
        f'speadi.common_tools.{request.param}.{request.param}_histogram')
    return module._histogram


@pytest.mark.parametrize('lower', [0.0, 0.2])
@pytest.mark.parametrize('shape', [(-1,), (1, -1), (1, 1, -1)])
def test_edges_and_adjacent_floats(histogram, lower, shape):
    edges = np.linspace(lower, 1.2, 121, dtype=np.float32)
    # XLA may flush subnormal values adjacent to zero to zero. Probe zero
    # itself and ordinary negative values, and adjacent floats at nonzero edges.
    nonzero_edges = edges[edges != 0]
    values = np.concatenate([
        edges,
        np.nextafter(nonzero_edges, np.float32(-np.inf)),
        np.nextafter(nonzero_edges, np.float32(np.inf)),
        np.array([lower - 0.005, -1, 9999, np.nan, -np.inf, np.inf], dtype=np.float32),
    ]).reshape(shape)
    expected, _ = np.histogram(values, bins=edges)
    np.testing.assert_array_equal(np.asarray(histogram(values, edges)), expected)


def test_empty_and_out_of_range(histogram):
    edges = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    for values in [[], [0.0, 0.195, 0.401, 9999]]:
        values = np.array(values, dtype=np.float32)
        np.testing.assert_array_equal(np.asarray(histogram(values, edges)), [0, 0])


def test_nonuniform_edges(histogram):
    edges = np.array([0, 0.1, 0.4, 1.2], dtype=np.float32)
    values = np.random.default_rng(42).uniform(-0.1, 1.3, 10000).astype(np.float32)
    expected, _ = np.histogram(values, bins=edges)
    np.testing.assert_array_equal(np.asarray(histogram(values, edges)), expected)
