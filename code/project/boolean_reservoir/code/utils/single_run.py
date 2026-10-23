"""Single-run weight precompute — deterministic, (i, j)-addressable checkpoints.

Library half of PLAN_graph_viz.md (§1 + §2). No dash/dashboard imports here: this
module is shared by `out/analysis/test.py` and the `wsgi.precompute_weights` CLI.

Why this exists
---------------
The grid search trains each (i, j) row but saves no weights (`save_keys: null`),
and its logging paths are timestamped per call (`init_logging`), not addressable
by (i, j). To look at a row's readout weights we retrain the row from its OWN
serialised job params — the parquet's `params_json` carries the seeds applied by
`apply_seed` (`train_model_parallel.py`), so retraining is deterministic in
principle. Do NOT re-derive params from the combination index: that skips
`apply_seed` and yields a different random wiring/LUT than the grid row.

Paths follow one convention (the ONE replace below):

    <config L.out_path with 'grid_search' -> 'single_run/weights'>
        i{i:05d}_j{j}/
            layout.npy                                            (seeded 3D layout)
            runs/last_run/checkpoints/last_checkpoint/
                parameters.yaml, weights.npz                      (last_run/last_checkpoint
                                                                   are override_symlink's)
"""
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import orjson
import pyarrow.parquet as pq

from project.boolean_reservoir.code.parameter import Params, load_yaml_config
from project.boolean_reservoir.code.train_model import train_single_model
from project.boolean_reservoir.code.utils.reservoir_utils import SaveAndLoadModel
from project.boolean_reservoir.code.utils.utils import generate_unique_seed


class SingleRunError(RuntimeError):
    """A row could not produce usable weights (e.g. all-zero readout).

    Raised, never swallowed: silent skips hide failure. Callers that batch
    (the CLI) own the try/except boundary and decide skip vs abort.
    """


# ── Path convention ───────────────────────────────────────────────────────────

_SINGLE_RUN_REPLACE = ('grid_search', 'single_run/weights')
SAVE_KEYS = ['weights', 'parameters']
_IJ_RE = re.compile(r'i(\d+)_j(\d+)$')


def single_run_base(config_out_path) -> Path:
    """Config-level base dir for a config's single runs.

    The ONE place holding the 'grid_search' -> 'single_run/weights' replace
    convention. Idempotent: an already-replaced base passes through unchanged.
    """
    return Path(str(config_out_path).replace(*_SINGLE_RUN_REPLACE))


def single_run_out_path(row_params_or_base, i: int, j: int) -> Path:
    """Run dir `<base>/i{i:05d}_j{j}` for a grid-search row.

    Accepts a row Params (uses its `L.out_path`, i.e. the grid-search out_path a
    loader extracts as the `L_out_path` column) or an already-extracted base
    path/str.
    """
    if isinstance(row_params_or_base, Params):
        base = row_params_or_base.L.out_path
    else:
        base = row_params_or_base
    return single_run_base(base) / f'i{i:05d}_j{j}'


def checkpoint_dir(base_out_path, i: int, j: int) -> Path:
    """`<base>/i{i:05d}_j{j}/runs/last_run/checkpoints/last_checkpoint` (symlinked)."""
    return (single_run_out_path(base_out_path, i, j)
            / 'runs' / 'last_run' / 'checkpoints' / 'last_checkpoint')


def weights_path(base_out_path, i: int, j: int) -> Path:
    return checkpoint_dir(base_out_path, i, j) / 'weights.npz'


def states_path(base_out_path, i: int, j: int) -> Path:
    """`<run>/runs/last_run/history` -- where recorded states land.

    The recorded history is written under the run's timestamped dir by
    `model.save()`/flush (`L.history.save_path` is re-resolved relative to the
    run the model created), but `<run>/runs/last_run` is a symlink to that latest
    run, so this resolves to the current history dir deterministically. Read back
    with `BatchedTensorHistoryWriter(...).reload_history()`. Mirrors
    `checkpoint_dir` and `weights_path` so all three artifacts of a row resolve
    through the same `last_run` symlink.
    """
    return single_run_out_path(base_out_path, i, j) / 'runs' / 'last_run' / 'history'


def layout_path(base_out_path, i: int, j: int) -> Path:
    return single_run_out_path(base_out_path, i, j) / 'layout.npy'


def ij_from_out_path(out_path):
    """Parse (i, j) from a run dir's trailing `i{i:05d}_j{j}` component, else None."""
    m = _IJ_RE.search(str(out_path))
    return (int(m.group(1)), int(m.group(2))) if m else None


# ── Params handling ───────────────────────────────────────────────────────────

def rehydrate_params(params_json: str) -> Params:
    """Deserialize a row's `params_json` into a real Params tree.

    MUST be the validating branch (`Params.model_validate(orjson.loads(...))`,
    cf. load_save.py's non-fast branch) — NOT the DotDict fast branch. DotDict
    is attribute-read sugar over the dumped dict with no pydantic properties or
    validators; `train_single_model` needs properties (`P.dataset_init_obj`,
    `P.accuracy_obj`) and `BooleanReservoir.__init__` reads
    `self.I.ticks_expanded` unconditionally. Properties are never serialised, so
    a DotDict params crashes at model construction.
    """
    return Params.model_validate(orjson.loads(params_json))


def make_single_run_params(row_params: Params, i: int, j: int, base_out_path,
                           record_states: bool = False) -> Params:
    """Row params -> params for the single-run retrain of that row.

    Deepcopy + deterministic out_path + save_keys. NO seed logic: row params
    already carry the applied seeds (`apply_seed` ran in the grid search).

    `record_states=True` turns on reservoir dynamics recording
    (the states probe): `L.history.record` + `persist_to_disk` are enabled and
    `history.save_path` is pointed at the (i, j)-addressable `<run>/history` dir
    (a sibling of `weights.npz`), so the recorded `(row, node)` state tensor + its
    expanded meta are discoverable by `BatchedTensorHistoryWriter.reload_history`
    the same way `weights_path` locates the weights. `save_keys` is unchanged --
    states travel via the history writer, not via a new save_key.
    """
    p = deepcopy(row_params)
    p.L.out_path = str(single_run_out_path(base_out_path, i, j))
    p.L.save_keys = list(SAVE_KEYS)
    if record_states:
        p.L.history.record = True
        p.L.history.persist_to_disk = True
        p.L.history.save_path = Path(p.L.out_path) / 'history'
    return p


# ── Load-or-train ─────────────────────────────────────────────────────────────

def load_or_train_weights(row_params: Params, i: int, j: int, base_out_path,
                          force: bool = False) -> dict:
    """Load a row's trained readout from its checkpoint, else retrain and save.

    Returns {'W', 'b', 'model', 'trained'}: 'model' is the freshly trained
    BooleanReservoir on the train path (None on the load path) — the CLI needs
    it in hand to write the seeded layout.

    An all-zero readout means the ridge solve never ran: raise (SingleRunError),
    do not skip — the caller's try/except boundary decides.
    """
    wpath = weights_path(base_out_path, i, j)
    if force or not wpath.exists():
        p = make_single_run_params(row_params, i, j, base_out_path)
        _, model, _, _ = train_single_model(parameter_override=p, save_model=True)
        W = np.asarray(model.readout_W)
        b = np.asarray(model.readout_b)
        trained = True
    else:
        load_dict = SaveAndLoadModel.load_from_path_dict_or_checkpoint_folder(
            checkpoint_path=checkpoint_dir(base_out_path, i, j),
            load_key_include_set={'weights'},
        )
        w = load_dict['weights']
        W = np.asarray(w['W'])
        b = np.asarray(w['b'])
        model, trained = None, False
    if np.allclose(W, 0.0) and np.allclose(b, 0.0):
        raise SingleRunError(
            f'row (i={i}, j={j}): all-zero readout — ridge solve never ran')
    return {'W': W, 'b': b, 'model': model, 'trained': trained}


def record_states_probe(row_params: Params, i: int, j: int, base_out_path,
                        max_samples: int | None = None) -> Path:
    """Train one (i, j) row with states recording and persist the dynamics.

    States probe: the ridge-training path (`get_states`) passes
    `None` for the history buffer, so merely flipping `L.history.record` during
    training records NOTHING. To persist real reservoir states you must run a
    recorded `model.forward(...)` on a probe input AFTER training. That is what
    this does:

      1. build single-run params with `record_states=True` (writer exists);
      2. ridge-train + save the checkpoint (weights as usual);
      3. run a recorded `forward()` on a probe slice of the train inputs
         (`max_samples` cap, default = all train samples);
      4. `flush_history()` -> writes `tensor_*.npy` + `meta_*.csv` into
         `states_path(base, i, j)`.

    Returns the `states_path` so callers can hand it to
    `BatchedTensorHistoryWriter(...).reload_history()`. Raises SingleRunError on
    an all-zero readout (the ridge solve never ran), mirroring load_or_train.
    """
    p = make_single_run_params(row_params, i, j, base_out_path, record_states=True)
    _, model, dataset, _ = train_single_model(parameter_override=p, save_model=True)
    if np.allclose(model.readout_W, 0.0) and np.allclose(model.readout_b, 0.0):
        raise SingleRunError(
            f'row (i={i}, j={j}): all-zero readout — ridge solve never ran')

    x = dataset.data['x']
    probe = x[:max_samples] if max_samples is not None else x
    # Recorded forward: populates `model.history` via _forward_julia. Must run
    # AFTER training (get_states records nothing) and BEFORE flush.
    model.forward(probe)
    model.flush_history()

    sp = states_path(base_out_path, i, j)
    return sp


# ── Seeded layout (§4 fix 7) ──────────────────────────────────────────────────

_LAYOUT_SEED_DOMAIN = 'graph_viz_layout'


def layout_seed_for_run(run_out_path) -> int:
    """Per-run layout seed, derived from the run dir's (i, j) — never the global RNG.

    networkx random generators draw from the GLOBAL numpy/stdlib rng unless
    seeded explicitly (the hazard `_nx_seed_kwargs` documents); a seeded layout
    makes `layout.npy` a speedup, not a correctness crutch. A run dir without an
    `i{i:05d}_j{j}` suffix falls back to a seed hashed from the path itself —
    still deterministic.
    """
    ij = ij_from_out_path(run_out_path)
    if ij is not None:
        return generate_unique_seed(_LAYOUT_SEED_DOMAIN, *ij)
    return generate_unique_seed(_LAYOUT_SEED_DOMAIN, 'path', str(run_out_path))


def seeded_layout_3d(graph, run_out_path) -> dict:
    """Deterministic 3D spring layout for one run dir."""
    import networkx as nx
    return nx.spring_layout(graph, dim=3, seed=layout_seed_for_run(run_out_path))


def save_layout(base_out_path, i: int, j: int, graph) -> Path:
    """Write the seeded layout as an (N, 3) array in node-id order (CLI, at
    precompute time — never from a dashboard request, which sits outside the
    write gate's safe roots)."""
    pos = seeded_layout_3d(graph, single_run_out_path(base_out_path, i, j))
    arr = np.stack([np.asarray(pos[n], dtype=float).reshape(3) for n in sorted(graph.nodes())])
    path = layout_path(base_out_path, i, j)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return path


def load_layout_for_run(run_out_path, n_nodes: int):
    """Return the cached (N, 3) layout array in node-id order, or None.

    `run_out_path` is the run dir itself (what `model.L.out_path` holds for a
    constructed single run). A cache whose node count no longer matches the
    graph is ignored — with the seed pinned the recompute is identical anyway.
    """
    path = Path(run_out_path) / 'layout.npy'
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape != (n_nodes, 3):
        return None
    return arr


# ── Two-pass params_json fetch (§2 step 3) ────────────────────────────────────

def fetch_params_json(config_paths, ij_pairs) -> pd.DataFrame:
    """Fetch `params_json` for surviving (i, j) pairs only, by batch-scanning
    each config's raw parquet.

    Two-pass load, pass 2. `params_json` is ~3.1 KB/row (~1.9 GB in memory for
    the 2D continuous config's 628,992 rows), so the single-pass
    `keep_params_json=True` route is an opt-in, not the default. Do NOT use the
    loader's `df_filter_mask` for view filters instead: it is hashed into the
    extraction cache key (load_save.py), so a per-view mask forks the warm cache.

    Returns a df with columns [L_out_path, i, j, params_json]; `L_out_path` is
    the constant str(config L.out_path) so rows join on (L_out_path, i, j) — an
    (i, j)-only join is wrong across configs, since i restarts at 0 per config.
    Duplicates follow the loader's convention: drop_duplicates(subset=['i','j'],
    keep='last').
    """
    from project.boolean_reservoir.code.utils.load_save import get_data_path

    want = {(int(i), int(j)) for i, j in ij_pairs}
    cols = ['L_out_path', 'i', 'j', 'params_json']
    if not want:
        return pd.DataFrame(columns=cols)
    i_values = sorted({i for i, _ in want})
    frames = []
    for cp in config_paths:
        data_path = get_data_path(cp)
        if not data_path.exists():
            continue
        config_out = load_yaml_config(cp).L.out_path
        tbl = pq.read_table(data_path, columns=['i', 'j', 'params_json'],
                            filters=[('i', 'in', i_values)])
        df = tbl.to_pandas()
        if df.empty:
            continue
        df = df[[(int(i), int(j)) in want for i, j in zip(df['i'], df['j'])]]
        if df.empty:
            continue
        df = df.drop_duplicates(subset=['i', 'j'], keep='last')
        df = df.assign(L_out_path=str(config_out))
        frames.append(df[cols])
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


# ── Scalar weight summaries (§6 sidecar manifest) ─────────────────────────────

_W_SPARSITY_EPS = 1e-12


def summarize_weights(L_out_path, i: int, j: int, W: np.ndarray, b: np.ndarray) -> dict:
    """One manifest row of scalar weight summaries (a scatter row cannot hold a vector)."""
    rec = {
        'L_out_path': str(L_out_path),
        'i': int(i), 'j': int(j),
        'W_l2': float(np.linalg.norm(W)),
        'W_max': float(np.abs(W).max()) if np.asarray(W).size else 0.0,
        'W_sparsity': float((np.abs(W) < _W_SPARSITY_EPS).mean()),
        'b_l2': float(np.linalg.norm(b)),
    }
    for dim, b_val in enumerate(np.asarray(b).flatten()):
        rec[f'b_dim{dim}'] = float(b_val)
    return rec


def rebuild_weights_summary(base_out_path, config_out_path) -> pd.DataFrame:
    """Full idempotent rebuild of `<base>/weights_summary.parquet`.

    Scans the base's `i*_j*` run dirs, summarises each existing weights.npz and
    rewrites the manifest. Cheap (hundreds of rows); called by the CLI after
    each trained batch so progress is durable. `L_out_path` is the config's
    grid-search out_path — the constant join key against the loader's extracted
    `L_out_path` column.
    """
    base = single_run_base(base_out_path)
    records = []
    if base.is_dir():
        for run_d in sorted(base.glob('i*_j*')):
            ij = ij_from_out_path(run_d.name)
            if ij is None:
                continue
            wp = checkpoint_dir(base, *ij) / 'weights.npz'
            if not wp.exists():
                continue
            w = np.load(wp)
            records.append(summarize_weights(config_out_path, ij[0], ij[1], w['W'], w['b']))
    df = pd.DataFrame(records)
    df.to_parquet(base / 'weights_summary.parquet', index=False)
    return df
