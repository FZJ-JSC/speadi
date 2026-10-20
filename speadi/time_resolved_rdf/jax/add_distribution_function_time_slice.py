"""Bounded JAX distance blocks; trajectory results stay in host memory."""
from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, lax

from ...common_tools.jax.time_distance_matrix import _rt_ortho_mic, _rt_general_mic
from ...common_tools.jax.jax_histogram import _histogram


@partial(jit, static_argnames=('orthogonal',))
def _count_distance_block(xyz, ids1, ids2, cuvec, bin_edges, orthogonal):
    """Only compact coordinates, atom IDs and a small histogram cross devices."""
    n1, n2 = ids1.shape[0], ids2.shape[0]
    local1 = jnp.arange(n1, dtype=jnp.int32)
    local2 = jnp.arange(n1, n1 + n2, dtype=jnp.int32)
    empty_union = jnp.empty((2, 0), dtype=jnp.int32)
    distance = _rt_ortho_mic if orthogonal else _rt_general_mic
    rt = distance(xyz, local1, local2, empty_union, cuvec)
    valid = ((ids1[:, None] >= 0) & (ids2[None, :] >= 0)
             & (ids1[:, None] != ids2[None, :]))
    rt = jnp.where(valid[None], rt, jnp.inf)
    return _histogram(rt, bin_edges)


def _add_counts(state, counts):
    """Exact wide counts without enabling JAX's global float64 mode."""
    low, high = state
    updated = low + counts.astype(jnp.uint32)
    return updated, high + (updated < low).astype(jnp.uint32)


@partial(jit, static_argnames=('orthogonal', 'tile1', 'tile2'))
def _count_group_pair(xyz1, xyz2, ids1, ids2, cuvec, bin_edges, orthogonal, tile1, tile2):
    """Run the atom tile loop on device, with one transfer per frame chunk."""
    blocks2 = ids2.shape[0] // tile2
    def count_tile(index, state):
        a, b = (index // blocks2) * tile1, (index % blocks2) * tile2
        r1 = lax.dynamic_slice_in_dim(xyz1, a, tile1, axis=1)
        r2 = lax.dynamic_slice_in_dim(xyz2, b, tile2, axis=1)
        atom1 = lax.dynamic_slice_in_dim(ids1, a, tile1)
        atom2 = lax.dynamic_slice_in_dim(ids2, b, tile2)
        counts = _count_distance_block(jnp.concatenate((r1, r2), axis=1),
                                       atom1, atom2, cuvec, bin_edges, orthogonal)
        return _add_counts(state, counts)
    initial = (jnp.zeros(bin_edges.shape[0] - 1, dtype=jnp.uint32),) * 2
    low, high = lax.fori_loop(0, (ids1.shape[0] // tile1) * blocks2, count_tile, initial)
    return jnp.stack((low, high))


def _pad_group(xyz, group, tile):
    """Pad only the final atom tile; negative IDs exclude padding from counts."""
    padding = (-len(group)) % tile
    ids = np.pad(group, (0, padding), constant_values=-1)
    coords = np.pad(xyz[:, group], ((0, 0), (0, padding), (0, 0)))
    return coords, ids


def _append_grts_mic(g_rts, n, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, unions,
                     bin_edges, orthogonal=False, pair_chunk_size=128, frame_chunk_size=16):
    """Accumulate raw block counts, then normalize once for the whole window.

    Distance arrays have at most frame_chunk_size * pair_chunk_size**2 elements.
    Atom tiles run in a compiled loop and share uploaded group coordinates.
    Host conversion synchronizes once per frame chunk and group pair. The
    complete result array is never passed to JAX.
    """
    n_frames = xyz.shape[0]
    if n_frames == 0:
        raise ValueError('Cannot calculate a TRRDF from an empty trajectory window')
    shell_volumes = (4.0 / 3.0 * np.pi) * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    mean_volume = np.asarray(cuvol).mean()
    for i, group1 in enumerate(g1):
        for j, group2 in enumerate(g2):
            # Total counts can exceed int32 even though device blocks are small.
            counts = np.zeros(nbins, dtype=np.int64)
            if not len(group1) or not len(group2):
                g_rts[i, j, n] = np.nan
                continue
            tile1 = min(pair_chunk_size, len(group1))
            tile2 = min(pair_chunk_size, len(group2))
            for frame in range(0, n_frames, frame_chunk_size):
                frames = slice(frame, frame + frame_chunk_size)
                xyz1, ids1 = _pad_group(xyz[frames], group1, tile1)
                xyz2, ids2 = _pad_group(xyz[frames], group2, tile2)
                words = np.asarray(_count_group_pair(xyz1, xyz2, ids1, ids2, cuvec[frames],
                                                     bin_edges, orthogonal, tile1, tile2), dtype=np.int64)
                counts += words[0] + (words[1] << 32)
            norm = (len(group2) / mean_volume) * shell_volumes * len(group1)
            g_rts[i, j, n] = counts / norm / n_frames
    return g_rts
