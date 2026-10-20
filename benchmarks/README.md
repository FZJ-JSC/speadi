# RDF benchmarks

Run from a checkout with SPEADI and the selected optional backend installed:

```sh
python benchmarks/benchmark_rdf.py --backend jax --case histogram --output histogram.json
python benchmarks/benchmark_rdf.py --backend jax --case trrdf --output trrdf.json
python benchmarks/benchmark_rdf.py --backend jax --case vanhove --window-size 1 --output vhf-rdf.json
python benchmarks/benchmark_rdf.py --backend jax --case vanhove --window-size 10 --output vhf-lags.json
```

Repeat with `--backend numpy` or `--backend numba`, or `--pbc general` for
trajectory benchmarks. Use `--window-size 1`, `10`, or `20` to examine shape
and compilation scaling on the first 200 frames of the 201-frame fixture. Histogram distances are
computed once with MDTraj; its `--pbc` option is not used.

Each invocation launches a fresh worker process. Reports include package
versions, input size, first-call time, all warmed samples, and their median.
JAX results are explicitly synchronized before stopping timers. Histogram
benchmarks put inputs on the device before timing, separate JAX lowering /
compilation from execution, and compare exact counts with NumPy. Numba's
explicit signatures compile on import, outside histogram execution timings.

TRRDF and VHF benchmarks measure the public API, including trajectory I/O,
input preparation, transfers and accumulation. Their first call includes any
lazy compilation; it is **not** a pure compilation measurement. A separate
MDTraj prefix-read time gives I/O context, but should not be subtracted as an exact
estimate of kernel cost. Imports and startup are outside these timings;
filesystem and persistent compiler caches may still be warm across processes.
`--threads` sets Numba and selected host-library thread limits, not JAX's total
CPU worker count. Keep hardware, environment and cache settings fixed when
comparing revisions. No fixed timing thresholds are imposed in unit tests.

The cases reuse `tests/data/nacl_box.{gro,xtc}` and the saved NA–O MDTraj RDF.
TRRDF uses the same first 200 frames as the existing tests and the existing test tolerances. VHF with one-frame
windows reproduces the RDF comparison in `tests/test_vanhove.py`; longer
windows exercise lag-dependent compilation but only check finite outputs,
not physical correctness. GROMACS reference values are not treated as
identical-bin ground truth (the existing tests skip that comparison).

With Numba 0.67, the existing TRRDF pipeline fails before histogram execution
because it passes a Python `dict` (`unions`) into a nopython function. This
also reproduces with the original histogram. The Numba histogram-only case
works; full-pipeline failures remain visible rather than being reported as
successful timings.

## Initial CPU measurement

[Recorded comparison](results/jax-histogram-cpu.json): on the local ARM Mac,
JAX 0.10.2, 106,950 NA–O distances and 120 bins, the warmed median over 30
samples was 0.162 ms for the original arithmetic histogram and 0.873 ms for
`jax.numpy.histogram` (about 5.4× slower). Both matched NumPy on these trajectory
distances; the original fails the synthetic boundary cases. This is one
histogram-only CPU comparison, not an end-to-end or GPU performance claim.
The built-in implementation establishes the binning baseline; a future
optimized implementation must retain the exact-count regression tests.

## Larger workloads

Use water oxygen–water oxygen TRRDFs to exercise the dense pair-distance
calculation and histogram together:

```sh
python benchmarks/benchmark_rdf.py --case trrdf --workload water --backend jax --repeats 3 --output water-jax.json
python benchmarks/benchmark_rdf.py --case trrdf --workload water --backend numpy --repeats 3 --output water-numpy.json
```

The bundled water case uses the saved O–O MDTraj reference and the existing
5% tolerance. The report includes reference/target atom counts, dense pair
counts (including subsequently excluded self pairs), analyzed frames, and
warmed frames/second. The histogram-only water case uses MDTraj's unique pair
list, so its distance count is different from SPEADI's dense O–O matrix.

For sustained throughput, supply a real longer trajectory:

```sh
python benchmarks/benchmark_rdf.py --case trrdf --workload water --trajectory /path/run.xtc --topology /path/topology.gro --frames 10000 --window-size 100 --repeats 3 --output long-water.json
```

`--frames` must be divisible by `--window-size`. For external trajectories, the default selections are
`water and element O` for water and `name NA` for sodium references; the target
is water oxygen. Override them with `--reference` and `--target` when needed.
The requested prefix is read in chunks before timing to check its length;
long trajectories are not loaded into memory at once. This preflight warms
filesystem caches, so these runs measure throughput with potentially cached
I/O, not cold-disk performance. `prefix_read_seconds` records that preflight.
External/custom selections or frame counts other than 200 check finite output
and bin coordinates only; they are not validated against the saved RDF.
Keep the 1.2 nm radial cutoff in mind when choosing a simulation box.

Increase frame count to test sustained throughput; increase window size to
test compilation and per-window memory scaling. These are different axes:
a longer trajectory does not inherently create a larger per-window workload.
The original NA–O histogram measurement above is a microbenchmark, not a
representative measure of full-system performance.

The bundled GRO uses the generic residue name `RES`, so its water preset
selects `name O`, matching the existing tests, rather than relying on residue
recognition. External trajectories use the water selection described above.

### Water TRRDF CPU comparison

Using 2,139 water oxygens, 200 frames, 20 windows of 10 frames, and 120 bins
produces 915,064,200 dense pair-frame evaluations per call. Sequential runs
on the same ARM Mac with three warmed samples gave:

| Histogram implementation | Warm median, full TRRDF | Frames/second |
| --- | ---: | ---: |
| [Original arithmetic](results/water-trrdf-original-histogram-cpu.json) | 3.01 s | 66.5 |
| [JAX built-in](results/water-trrdf-jax-cpu.json) | 13.27 s | 15.1 |

Both passed the saved O–O MDTraj comparison. This larger workload shows a
roughly 4.4× end-to-end slowdown from the histogram replacement on this CPU,
so the microbenchmark slowdown does matter here. The original still fails
boundary tests and is not a correct replacement without fixes. These are
single-machine measurements with only three warm samples (visible timing
variation in the JSON), not GPU results or a long-trajectory scaling study.
The baseline changes only the histogram implementation, loaded from the
recorded Git revision, while using the current benchmark harness.
