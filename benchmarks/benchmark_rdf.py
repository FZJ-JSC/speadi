"""Run each benchmark in a fresh process; see benchmarks/README.md."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def synchronize(value):
    if isinstance(value, (tuple, list)):
        for item in value:
            synchronize(item)
    elif hasattr(value, 'block_until_ready'):
        value.block_until_ready()


def timed(call):
    start = perf_counter()
    result = call()
    synchronize(result)
    return perf_counter() - start, result


def worker(args):
    import numpy as np
    import mdtraj as md
    import speadi as sp

    sp.JAX_AVAILABLE = args.backend == 'jax'
    sp.NUMBA_AVAILABLE = args.backend == 'numba'
    sp.NUMBA_THREADS = args.threads
    if args.backend == 'numba':
        import numba
        numba.set_num_threads(args.threads)

    data = ROOT / 'tests' / 'data'
    external = args.trajectory is not None
    trajectory_path = args.trajectory or data / 'nacl_box.xtc'
    topology_path = args.topology or data / 'nacl_box.gro'
    topology = md.load_topology(str(topology_path))
    # The bundled GRO labels solvent as RES rather than a recognized water residue.
    water_selection = 'water and element O' if external else 'name O'
    reference_selection = args.reference or (water_selection if args.workload == 'water' else 'name NA')
    target_selection = args.target or water_selection
    refs = topology.select(reference_selection)
    targets = topology.select(target_selection)
    if not len(refs) or not len(targets):
        raise ValueError('Both atom selections must be nonempty')
    analyzed_frames = args.frames
    needed_frames = args.window_size if args.case == 'histogram' else analyzed_frames

    # Preflight the requested prefix without retaining a long trajectory in RAM.
    start = perf_counter()
    available = 0
    first_window = None
    for chunk in md.iterload(str(trajectory_path), top=topology, chunk=args.window_size):
        if first_window is None:
            first_window = chunk
        available += len(chunk)
        if available >= needed_frames:
            break
    load_seconds = perf_counter() - start
    if available < needed_frames:
        raise ValueError(f'Requested {needed_frames} frames, but only {available} are available')
    edges = np.linspace(0, 1.2, 121, dtype=np.float32)
    expected_r = 0.5 * (edges[1:] + edges[:-1])
    use_saved_reference = (not external and args.reference is None and args.target is None
                           and analyzed_frames == 200)
    reference_name = 'O' if args.workload == 'water' else 'NA'
    expected_g = np.loadtxt(data / f'{reference_name}_mdtraj_gr.txt') if use_saved_reference else None
    report = {'case': args.case, 'backend': args.backend, 'pbc': args.pbc,
              'threads': args.threads, 'prefix_read_seconds': load_seconds,
              'workload': args.workload, 'trajectory': str(trajectory_path),
              'reference_selection': reference_selection, 'target_selection': target_selection,
              'reference_atoms': len(refs), 'target_atoms': len(targets),
              'dense_pairs_per_frame': len(refs) * len(targets),
              'python': platform.python_version(), 'platform': platform.platform(),
              'versions': {name: importlib.metadata.version(name)
                           for name in ['numpy', 'mdtraj', 'jax', 'numba']
                           if importlib.util.find_spec(name) is not None}}

    if args.case == 'histogram':
        pairs = topology.select_pairs(refs, targets)
        distances = md.compute_distances(first_window, pairs).astype(np.float32)
        expected_counts = np.histogram(distances, bins=edges)[0]
        report['distance_shape'] = list(distances.shape)
        if args.backend == 'numpy':
            call = lambda: np.histogram(distances, bins=edges)[0]
        else:
            module = __import__(f'speadi.common_tools.{args.backend}.{args.backend}_histogram',
                                fromlist=['_histogram'])
            if args.backend == 'jax':
                import jax
                report['devices'] = [str(d) for d in jax.devices()]
                start = perf_counter()
                distances, edges = jax.device_put((distances, edges))
                synchronize((distances, edges))
                report['device_transfer_seconds'] = perf_counter() - start
                start = perf_counter()
                executable = module._histogram.lower(distances, edges).compile()
                report['lower_and_compile_seconds'] = perf_counter() - start
                call = lambda: executable(distances, edges)
            else:
                # Explicit-signature Numba functions compile on module import.
                call = lambda: module._histogram(distances, edges)
        def validate(result):
            np.testing.assert_array_equal(np.asarray(result), expected_counts)
    else:
        windows = analyzed_frames // args.window_size
        if windows < 1:
            raise ValueError('window-size exceeds requested frame count')
        if windows * args.window_size != analyzed_frames:
            raise ValueError('window-size must divide requested frame count')
        report.update(n_windows=windows, window_size=args.window_size,
                      analyzed_frames=analyzed_frames)
        function = sp.trrdf if args.case == 'trrdf' else sp.vanhove
        call = lambda: function(str(trajectory_path), refs, targets, top=topology,
                                pbc=args.pbc, n_windows=windows, window_size=args.window_size,
                                skip=0, stride=1, r_range=(0.0, 1.2), nbins=120)
        def validate(result):
            np.testing.assert_allclose(np.asarray(result[0]), expected_r, rtol=1e-4)
            if not all(np.isfinite(np.asarray(x)).all() for x in result):
                raise AssertionError('Nonfinite output')
            if expected_g is None:
                return
            if args.case == 'trrdf':
                rdf = np.asarray(result[1]).mean(axis=(0, 1, 2))
            elif args.window_size == 1:
                rdf = np.asarray(result[2]).mean(axis=(0, 1, 2))
            else:
                # The saved RDF covers all frames, not the subset of VHF origins.
                return
            np.testing.assert_allclose(rdf, expected_g, rtol=5e-2)

    first, result = timed(call)
    validate(result)
    samples = []
    for _ in range(args.repeats):
        elapsed, result = timed(call)
        validate(result)
        samples.append(elapsed)
    report.update(first_call_seconds=first, warm_seconds=samples,
                  warm_median_seconds=statistics.median(samples),
                  validation='saved MDTraj RDF' if use_saved_reference and (args.case == 'trrdf' or
                  (args.case == 'vanhove' and args.window_size == 1)) else
                  'NumPy counts' if args.case == 'histogram' else 'finite output only')
    if args.case != 'histogram':
        report['warm_frames_per_second'] = analyzed_frames / report['warm_median_seconds']
        report['dense_pair_frame_evaluations'] = analyzed_frames * len(refs) * len(targets)
    print('BENCHMARK_JSON=' + json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['jax', 'numba', 'numpy'], default='jax')
    parser.add_argument('--case', choices=['histogram', 'trrdf', 'vanhove'], default='histogram')
    parser.add_argument('--workload', choices=['sodium', 'water'], default='sodium')
    parser.add_argument('--trajectory', type=Path)
    parser.add_argument('--topology', type=Path)
    parser.add_argument('--frames', type=int, default=200)
    parser.add_argument('--reference', help='MDTraj reference atom selection')
    parser.add_argument('--target', help='MDTraj target atom selection')
    parser.add_argument('--pbc', choices=['ortho', 'general'], default='ortho')
    parser.add_argument('--window-size', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.window_size, args.repeats, args.threads, args.frames) < 1:
        parser.error('window-size, frames, repeats and threads must be positive')
    if (args.trajectory is None) != (args.topology is None):
        parser.error('--trajectory and --topology must be supplied together')
    if args.case != 'histogram' and args.frames % args.window_size:
        parser.error('--frames must be divisible by --window-size')
    if args.worker:
        worker(args)
        return
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads))
    process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                              *sys.argv[1:], '--worker'], env=env, text=True, capture_output=True)
    if process.returncode:
        sys.stderr.write(process.stdout + process.stderr)
        raise SystemExit(process.returncode)
    line = next(line for line in process.stdout.splitlines() if line.startswith('BENCHMARK_JSON='))
    report = json.loads(line.split('=', 1)[1])
    rendered = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
