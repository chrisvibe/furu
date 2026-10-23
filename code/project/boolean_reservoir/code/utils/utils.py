import numpy as np
from hashlib import sha256
from pathlib import Path
import networkx as nx
import time
from contextlib import contextmanager
import socket
import logging
import signal
import sys
from os import getpid, replace, environ

logger = logging.getLogger(__name__)

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(process)d - %(filename)s - %(message)s',
        stream=sys.stdout,
        force=True,
    )
    logging.Formatter.converter = time.gmtime

def generate_unique_seed(*args):
    combined_str = ','.join(map(str, args))
    hash_digest = sha256(combined_str.encode()).hexdigest()
    return int(hash_digest, 16) % (2**31 - 1)

def make_rng(seed=None, *purpose) -> np.random.Generator:
    """Local Generator derived from an explicit seed — never the global RNG.

    `purpose` derives an INDEPENDENT stream from the same base seed
    (e.g. make_rng(R.seed, 'graph') vs make_rng(R.seed, 'lut')), via
    generate_unique_seed. Independent streams are the point, not a nicety:
    they are what stops one component's consumption from shifting another's
    draw. The bug this replaces was exactly that — a load-vs-generate branch
    consuming different amounts of global RNG before a later shuffle, so the
    result depended on whether a cache file happened to exist.

    seed=None -> fresh OS entropy (non-reproducible), keeping the convention that
    an unset seed means "don't pin this". Every config that produces results sets
    its seeds; if a run needs to be reproducible and is not, an unset seed is the
    first thing to check.

    This replaced set_seed(), which seeded the process-global numpy and stdlib
    RNGs. Do not reintroduce it: global seeding fixes only where a stream starts,
    not how much of it something else consumed first, which is the failure mode
    above. Randomness that should NOT be pinned — scheduler backoff jitter, worker
    recycling — must keep using the unseeded global RNG so it does not run in
    lockstep across workers.
    """
    if seed is None:
        return np.random.default_rng()
    if purpose:
        seed = generate_unique_seed(seed, *purpose)
    return np.random.default_rng(seed)

@contextmanager
def flush_subnormals():
    """Round subnormal floats (|x| < ~1e-308 f64 / ~1e-38 f32) to zero on this thread.

    Wrap LAPACK calls (SVD / lstsq / matrix_rank) on rank-deficient matrices.
    Single-threaded, the SVD's null-space part decays into subnormals and x86
    handles those ~10x slower: a 2048x2048 rank-27 reservoir state matrix took
    23.7s for matrix_rank vs 2.1s with this on (identical rank; lstsq weights
    bit-identical). Multi-threaded BLAS hides it, which is why it only showed
    on the cluster's one-core-per-job workers.

    Only speed changes: those values sit ~300 orders of magnitude below the
    rank/lstsq cutoff (~1e-12 x largest singular value), so they are treated
    as zero either way. The switch is the CPU's FTZ/DAZ mode (per-thread),
    flipped through Julia's set_zero_subnormals because juliacall runs on the
    calling Python thread; it is restored on exit so nothing else is affected.
    """
    from project.boolean_reservoir.code.reservoir import _get_jl
    jl = _get_jl()
    previous = bool(jl.get_zero_subnormals())
    jl.set_zero_subnormals(True)
    try:
        yield
    finally:
        jl.set_zero_subnormals(previous)

def print_pretty_binary_matrix(data, input_nodes=None, reservoir_nodes=None, print_str=True, return_str=False):
    QUADRANT_COLORS = {
        'II': '\033[95m',
        'IR': '\033[93m',
        'RI': '\033[91m',
        'RR': '\033[92m',
        'default': '\033[94m',
    }

    def infer_quadrant(u, v):
        if u in input_nodes and v in input_nodes:
            return 'II'
        elif u in input_nodes and v in reservoir_nodes:
            return 'IR'
        elif u in reservoir_nodes and v in input_nodes:
            return 'RI'
        elif u in reservoir_nodes and v in reservoir_nodes:
            return 'RR'
        return 'default'

    if isinstance(data, np.ndarray):
        array = (data != 0).astype(int)
        color_matrix = np.full(array.shape, 'default', dtype=object)

    elif isinstance(data, (nx.Graph, nx.DiGraph)):
        nodes = list(data.nodes())
        node_index = {node: i for i, node in enumerate(nodes)}
        array = nx.to_numpy_array(data, nodelist=nodes, dtype=bool).astype(int)
        color_matrix = np.full(array.shape, 'default', dtype=object)

        for u, v, attrs in data.edges(data=True):
            i, j = node_index[u], node_index[v]
            if 'quadrant' in attrs:
                q = attrs['quadrant']
            elif input_nodes is not None and reservoir_nodes is not None:
                q = infer_quadrant(u, v)
            else:
                q = 'default'
            color_matrix[i, j] = q
            if not data.is_directed():
                color_matrix[j, i] = q
    else:
        raise TypeError("Input must be a np.ndarray or networkx Graph/DiGraph.")

    lines = []
    for i in range(array.shape[0]):
        row = ''
        for j in range(array.shape[1]):
            val = str(array[i, j])
            color = QUADRANT_COLORS.get(color_matrix[i, j], QUADRANT_COLORS['default'])
            row += f"{color}{val}\033[0m"
        lines.append(row)

    result = '\n'.join(lines)
    if print_str:
        print(result)
    if return_str:
        return result

def override_symlink(source: Path, link: Path = None):
    if link is None:
        link = Path(source.name)
    try:
        temp_link = Path(f"{link}.tmp.{getpid()}.{time.time_ns()}")
        temp_link.symlink_to(source)
        replace(str(temp_link), str(link))
    except Exception:
        if 'temp_link' in locals() and temp_link.exists():
            try:
                temp_link.unlink()
            except Exception:
                pass

def l2_distance(arr):
    return np.sqrt((arr ** 2).sum(axis=1))

def l2_distance_squared(arr):
    return (arr ** 2).sum(axis=1)

def manhattan_distance(arr):
    return np.abs(arr).sum(axis=1)

def balance_dataset(dataset, num_bins=100, distance_fn=l2_distance, labels_are_classes=False,
                    target_mode='samples_over_bins', verbose=False, rng=None):
    # rng: pass an explicit Generator (make_rng(D.seed, 'balance')). Balancing runs
    # AFTER dataset construction, so relying on the global RNG made the result depend
    # on how much the dataset build had consumed — i.e. on cache state.
    if rng is None:
        rng = make_rng(getattr(dataset.D, 'seed', None), 'balance')
    x = dataset.data['x']
    y = dataset.data['y']

    distances = distance_fn(y)
    bins = np.linspace(distances.min(), distances.max(), num_bins + 1)
    if labels_are_classes:
        bin_indices = y.ravel().astype(int)
    else:
        bin_indices = np.digitize(distances, bins) - 1  # 0-indexed

    if target_mode == 'samples_over_bins':
        target_per_bin = max(len(y) // num_bins, 1)
    elif target_mode == 'minimum_bin':
        target_per_bin = int(np.bincount(y.ravel().astype(int)).min())

    balanced_indices = []
    for i in range(num_bins):
        bin_samples = np.where(bin_indices == i)[0]
        if len(bin_samples) > 0:
            if len(bin_samples) <= target_per_bin:
                balanced_indices.append(bin_samples)
            else:
                sampled = bin_samples[rng.permutation(len(bin_samples))[:target_per_bin]]
                balanced_indices.append(sampled)

    balanced_indices = np.concatenate(balanced_indices) if balanced_indices else np.arange(len(y))

    if verbose:
        if labels_are_classes:
            print("Class distribution:")
            unique_labels = np.unique(y)
            for label in unique_labels:
                before_count = (y == label).sum()
                after_count = (y[balanced_indices] == label).sum()
                print(f" Class {int(label)}: {before_count} → {after_count} samples")
        else:
            print("Value distribution (quartiles):")
            for q in [0, 25, 50, 75, 100]:
                before = np.percentile(y, q)
                after  = np.percentile(y[balanced_indices], q)
                print(f" {q}%: {before:.4f} → {after:.4f}")

    balanced_indices = balanced_indices[rng.permutation(len(balanced_indices))]
    from benchmark.utils.base_dataset import _ROW_KEYS  # local: base_dataset imports this module
    dataset.set_data({k: dataset.data[k][balanced_indices] for k in ('x', 'y', *_ROW_KEYS)
                      if dataset.data[k] is not None})

    if verbose:
        n_before, n_after = len(x), len(balanced_indices)
        print(f'Balanced dataset from {n_before} to {n_after} ({(n_before-n_after)/n_before*100:.2f}% reduction)')

    return dataset


def run_on_node(configs, node_job_assignments, run_fn, **kwargs):
    def handle_exit(signum, frame):
        logger.warning(f"Received signal {signum}. Shutting down node runner...")
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    node = environ.get("SLURMD_NODENAME") or socket.gethostname()
    if "hpc" in node:
        logger.info(f"HPC node: {node}")
        node_id = int(node[3:])
    else:
        logger.warning(f"Non-HPC node: {node}")
        node_id = 'unknown'

    indices = node_job_assignments.get(node_id, node_job_assignments.get('unknown', []))
    selected = [configs[i] for i in indices]

    for c in selected:
        logger.info(f"Starting config for index {indices[selected.index(c)]}")
        run_fn(c, **kwargs)

    print('done!')


def run_on_nodes(configs, run_fn, **kwargs):
    # Every run cooperates through the grid's state.db, so nothing here needs the node
    # count: identity is per process (train_model_parallel's hostname_pid_time) and the
    # ETA counts the live nodes in the DB.
    node = environ.get('SLURMD_NODENAME', socket.gethostname())

    array_task_id = environ.get('SLURM_ARRAY_TASK_ID')
    if array_task_id is not None:
        idx = int(array_task_id)
        if idx >= len(configs):
            raise ValueError(
                f"SLURM_ARRAY_TASK_ID={idx} is out of range — only {len(configs)} config(s) "
                f"(valid: 0..{len(configs)-1}). Check for a stale env var."
            )
        logger.info(f"Array mode: task {idx+1}/{len(configs)}, config: {configs[idx]}")
        configs = [configs[idx]]
    else:
        logger.info(f"Node {node}: {len(configs)} config(s), joined through each grid's state.db")

    for yaml_path in configs:
        logger.info(f"Starting config: {yaml_path}")
        run_fn(yaml_path, **kwargs)
