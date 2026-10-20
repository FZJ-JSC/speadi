# Time-resolved Radial Distribution Functions (TRRDFs)

## GPU memory in notebooks and HPC jobs

JAX normally reserves 75% of GPU memory at its first operation and retains
its memory pool for reuse. Large, successively smaller allocation failures
can be allocator reservation retries; those messages alone do not establish
the size of the TRRDF working arrays or prove a memory leak.

Restart the notebook kernel, then run this as the **first cell**, before
importing JAX, SPEADI, or another library that initializes JAX:

```python
import os
os.environ.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import speadi as sp
```

This enables allocation on demand, but JAX still caches freed buffers. If
returning unused allocations to CUDA is required between calculations, also
set `os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"` in that first cell,
before the import. Use this only for memory debugging or when returning
memory is essential: the platform allocator can be dramatically slower,
particularly for tiled calculations. Keep the default pooling allocator
for normal analysis. The `pop` above removes a previous platform setting.
Changing either setting after GPU initialization requires a kernel restart.
See [JAX's GPU memory documentation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).

The JAX TRRDF path processes at most 16 frames and 128 atoms from each group
per distance block by default (262,144 distances, or 1 MiB for a float32
distance array; intermediates, compilation and allocator overhead are extra).
Completed histograms and the full returned `g_rt` array stay in NumPy host
memory. The atom tile loop runs inside compiled JAX code, with one upload
of each group's coordinates and one histogram download per frame chunk
and group pair. Tiles reuse their device buffers; each frame chunk completes
before another is submitted. Final tiles are padded and masked so ragged
groups do not require separate kernels for their remainder tiles.

The distance bound excludes group coordinates, whose device storage scales
linearly with the selected atoms and frame chunk size. It also does not cap
the JAX allocator's reserved pool or host memory: MDTraj still loads one
trajectory window, and the host output scales with group pairs, windows and bins.

Use `pair_chunk_size` and `frame_chunk_size` to tune the JAX working blocks:

```python
r, g_rt = sp.trrdf(trajectory, g1, g2, top=top,
                   pair_chunk_size=64, frame_chunk_size=8)
```

Chunk sizes change execution memory and dispatch overhead, not the averaging
window, stride or normalization. They are ignored by the NumPy and Numba
backends. This blocking currently applies to `trrdf`, not `int_trrdf` or
`vanhove`. Deleting NumPy results releases host memory; it does not flush
JAX's GPU pool. `jax.clear_caches()` clears compilation caches and is not a
GPU pool release API. Restarting the kernel terminates its GPU context.

## Pair-correlation function g(r,t)
::: speadi.time_resolved_rdf.trrdf
    options:
        members:
            - trrdf
        separate_signature: True
        heading_level: 3

## Integral of g(r,t): n(r,t)
::: speadi.int_trrdf.int_trrdf
    options:
        members:
            - int_trrdf
        separate_signature: True 
        heading_level: 3

