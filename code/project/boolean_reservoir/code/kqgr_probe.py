"""KQ/GR probe construction — task-agnostic, axis-agnostic.

Works purely on the (m, s, f, b) tensor shape shared by every dataset type
(temporal, path integration, neurogym). Replaces the former
KQGRInit._adjust_free_bits / _get_kq_cached_dataset / kqgr(),
prepare_kqgr_model_params, and BooleanTransformer._apply_tau_bits /
_apply_tau_steps.

Key idea: 'steal' vs 'augment' is not a branch inside the freeze operation —
it's entirely expressed by how wide the pool was generated (native width, or
native+tau under augment). Freezing is always an in-place overwrite of a
tau-sized slice with one shared reference row, never a concatenation/growth.
This mirrors gr_steps_probe's existing overwrite mechanic (kq_and_gr_metric.py),
generalized to either axis and to full pool construction.
"""
import copy
import json
import warnings
from collections import OrderedDict
from typing import Literal, Optional
import numpy as np

from project.boolean_reservoir.code.parameter import Params, InputParams

POOL_STREAMS = 2 ** 22   # a multi-channel exhaustive pool's cap, in streams (samples x channels)


def get_supply_width(P: Params, axis: Literal['resolution', 'steps']) -> int:
    """Current native width along the freeze axis (of P as the pool sees it —
    build_kqgr_probe hands in P with the dataset already at kqgr_pool())."""
    if axis == 'resolution':
        return P.M.I.resolution
    field = P.dataset.step_length_field()
    if field is None:
        raise NotImplementedError(
            f"tau_axis='steps' requires a dataset exposing 'bits' or 'steps'; "
            f"{type(P.dataset).__name__} has neither."
        )
    return getattr(P.dataset, field)


def set_supply_width(P: Params, axis: Literal['resolution', 'steps'], width: int) -> Params:
    """Deep-copy P with the freeze axis widened/narrowed to `width`.

    resolution: reconstructs InputParams (nulls derived bits/chunk_size/n_nodes
    so the validator chain recomputes them) — a genuine model-architecture
    change, since more resolution means more input nodes.
    steps: plain setattr via step_length_field() — no architecture change,
    sequence length is a runtime property of the forward pass, not the model.
    """
    P = copy.deepcopy(P)
    if axis == 'resolution':
        I = P.M.I
        updated = I.model_dump()
        updated.update({'resolution': width, 'bits': None, 'chunk_size': None, 'n_nodes': None})
        P.M.input_layer = InputParams(**updated)
        if hasattr(P.dataset, 'bits'):
            P.dataset.bits = P.M.I.features * P.M.I.resolution
    else:
        field = P.dataset.step_length_field()
        if field is None:
            raise NotImplementedError(
                f"tau_axis='steps' requires a dataset exposing 'bits' or 'steps'; "
                f"{type(P.dataset).__name__} has neither."
            )
        setattr(P.dataset, field, width)
    return P


def freeze(x: np.ndarray, tau: int, axis: Literal['resolution', 'steps'],
           evaluation: str, rng: np.random.Generator, ref_idx: Optional[int] = None,
           start: int = 0) -> np.ndarray:
    """Overwrite a tau-sized slice along `axis` with one shared reference row,
    drawn from x itself. Never grows the array — width in, width out.

    ref_idx: which row of x supplies the frozen reference. Defaults to a
    random draw (rng.integers(0, m)); pass an explicit index (e.g. 0) for
    callers that need a fixed, deterministic reference row.
    start: the first freezable position (a temporal washout leads the steps:
    first/last/random are taken over the scored span [start, width) only).
    """
    if tau == 0:
        return x.copy()
    dim = 1 if axis == 'steps' else x.ndim - 1
    width = x.shape[dim]
    if evaluation == 'last':
        positions = np.arange(width - tau, width)
    elif evaluation == 'first':
        positions = np.arange(start, start + tau)
    elif evaluation == 'random':
        positions = start + rng.permutation(width - start)[:tau]
    else:
        return x.copy()

    m = x.shape[0]
    if ref_idx is None:
        ref_idx = int(rng.integers(0, m))
    out = x.copy()
    idx_out = [slice(None)] * x.ndim
    idx_out[dim] = positions
    idx_ref = list(idx_out)
    idx_ref[0] = slice(ref_idx, ref_idx + 1)
    out[tuple(idx_out)] = x[tuple(idx_ref)]
    return out


def dedupe(x: np.ndarray) -> tuple:
    """np.unique over flattened rows, preserving original row order.
    Returns (unique_rows, source_indices) — source_indices are positions into
    the ORIGINAL x, sorted to keep relative order.
    """
    flat = x.reshape(x.shape[0], -1)
    _, first_idx = np.unique(flat, axis=0, return_index=True)
    index = np.sort(first_idx)
    return x[index], index


def trim_to_n(x: np.ndarray, index: np.ndarray, n: int, rng: np.random.Generator) -> tuple:
    """Warn-and-keep if short of n, random-subsample if over n. Index is
    carried along so callers can map surviving rows back to a shared pool.
    """
    m = x.shape[0]
    if m < n:
        warnings.warn(
            f"Only {m} distinct patterns available for {n} samples "
            f"— inputs will repeat. Reduce tau or increase resolution/steps.",
            UserWarning,
        )
        return x, index
    if m > n:
        sel = rng.permutation(m)[:n]
        return x[sel], index[sel]
    return x, index


def build_kqgr_probe(P: Params, dataset_init, n_nodes: int, rng: np.random.Generator,
                     fast: bool = True) -> tuple:
    """Returns (kq_x, gr_x, P_model). P_model already has the correct
    (possibly widened) architecture baked in — construct BooleanReservoir(P_model)
    directly, no separate "prepare model params" step needed.
    """
    tau, axis, mode = P.D.tau, P.D.tau_axis, P.D.tau_mode
    start = getattr(P.D, 'washout', 0) if axis == 'steps' else 0   # freeze within the scored steps
    # The pool is the task's plain input stream: the dataset strips its own
    # training-only framing (e.g. PI steps_mix / query_mix) — both freeze axes.
    P = copy.copy(P)
    P.dataset = P.dataset.kqgr_pool()
    native = get_supply_width(P, axis)
    width = native + (tau if mode == 'augment' else 0)

    P_pool = set_supply_width(P, axis, width)
    P_pool.dataset.tau = 0
    P_pool.dataset.split.train = 1
    P_pool.dataset.split.dev = 0
    P_pool.dataset.split.test = 0
    if getattr(P_pool.dataset, 'sampling_mode', '') == 'exhaustive':
        P_pool.dataset.samples = 2 ** width
        dims = getattr(P_pool.dataset, 'dimensions', 1)
        if isinstance(dims, int) and dims > 1:
            # c channels draw 2**width samples x c streams: 40 GB at c 256, width 20 (furu capacity). The probe keeps
            # <= n_nodes distinct rows, so cap the pool's streams; c = 1 pools are unchanged.
            P_pool.dataset.samples = min(2 ** width, POOL_STREAMS // dims)

    def encode(ds, rows):
        """Run the normal encode/split path over just the rows handed in."""
        sub = copy.copy(ds)
        sub.x = rows
        sub.y = None if ds.y is None else ds.y[:rows.shape[0]]
        return dataset_init._process_dataset(sub, P_pool).data['x']

    # ---- which representation the probe steps run on ---------------------------
    # The probe is always: [freeze ->] dedupe -> trim -> (KQ taken from GR's pool).
    # Those steps can run on the ENCODED pool or on the RAW one, because:
    #   * freeze(axis='steps') rewrites the STEP axis while the encoder only expands
    #     the LAST axis, so the two commute;
    #   * dedupe keys on whole-row identity and returns rows in original order, and
    #     the encoder is applied per row, so duplicate classes match -- PROVIDED the
    #     encoding is injective;
    #   * trim_to_n draws rng.permutation(m) from the row COUNT alone.
    # Running on the raw pool means only the <= n_nodes survivors are ever encoded,
    # so the pool is never materialised at full redundancy.
    # Injectivity holds when the input is ALREADY BINARY (bool) -- encoding merely
    # replicates bits.  Continuous input (path integration) is QUANTISED to
    # `resolution` bits, so two distinct rows can collapse into one and dedupe would
    # disagree; those keep the encoded pool, and are cheap anyway since their pool is
    # `samples` rows rather than 2**width.  axis='resolution' freezes the last axis,
    # which the encoder rewrites, so it is excluded too.
    # Byte-identical either way -- pinned by test_kqgr_probe_fast_path.py.
    # A cache hit (see PROBE_CACHE_SIZE) skips the pool build too; only fast-path pools are stored.
    key = _probe_key(P_pool, P.D, n_nodes, start, rng) if fast and axis == 'steps' and not P.D.independent else None
    if key is not None and key in _PROBE_CACHE:
        pool_ds, kq_rows, gr_rows, rng_end = _PROBE_CACHE[key]
        _PROBE_CACHE.move_to_end(key)
        rng.bit_generator.state = rng_end
        return encode(pool_ds, kq_rows), encode(pool_ds, gr_rows), P_pool

    pool_ds = dataset_init._create_raw_dataset(P_pool)
    encode_late = fast and axis == 'steps' and pool_ds.x.dtype == np.bool_

    def carrier(ds):
        """The pool in whichever representation the probe steps run on."""
        return ds.x if encode_late else dataset_init._process_dataset(ds, P_pool).data['x']

    def finish(ds, rows):
        return encode(ds, rows) if encode_late else rows

    ds_kq, pool = pool_ds, carrier(pool_ds)

    if P.D.independent:
        # KQ and GR drawn from two independent pools (Snyder 2012 methodology).
        kq_rows, kq_idx = dedupe(pool)
        kq_rows, _ = trim_to_n(kq_rows, kq_idx, n_nodes, rng)
        ds_gr = dataset_init._create_raw_dataset(P_pool)
        gr_rows, gr_idx = dedupe(freeze(carrier(ds_gr), tau, axis, P.D.evaluation_mode, rng, start=start))
        gr_rows, _ = trim_to_n(gr_rows, gr_idx, n_nodes, rng)
    else:
        # Paired: GR's surviving rows also select KQ, so both see identical samples.
        ds_gr = ds_kq
        gr_rows, gr_idx = dedupe(freeze(pool, tau, axis, P.D.evaluation_mode, rng, start=start))
        gr_rows, gr_idx = trim_to_n(gr_rows, gr_idx, n_nodes, rng)
        kq_rows = pool[gr_idx]

    if key is not None and encode_late and PROBE_CACHE_SIZE > 0:
        _PROBE_CACHE[key] = (pool_ds, kq_rows, gr_rows, rng.bit_generator.state)
        while len(_PROBE_CACHE) > PROBE_CACHE_SIZE:
            _PROBE_CACHE.popitem(last=False)
    return finish(ds_kq, kq_rows), finish(ds_gr, gr_rows), P_pool


# ---- probe reuse across reservoirs ------------------------------------------------
# On the fast path the raw probe rows depend only on the pool (the dataset as the
# probe sees it), the freeze setting, n_nodes and the rng's state, never on the
# reservoir; only the late encode does. So many reservoirs measured under one setting
# can share the raw rows: dedupe of a 2^20-stream augment pool is ~90% of a probe's
# time. Off by default (a grid worker would hold pools it rarely reuses); the
# re-measure tool turns it on. A hit restores the rng's end state, so a cached call is
# byte-identical to an uncached one, rng included (test_kqgr_probe_cache.py).
PROBE_CACHE_SIZE = 0
_PROBE_CACHE: 'OrderedDict' = OrderedDict()


def _probe_key(P_pool: Params, D, n_nodes: int, start: int, rng: np.random.Generator) -> str:
    # path is derived and refreshed lazily (dataset_init.update_path), so it would split equal pools
    return json.dumps([P_pool.dataset.model_dump_json(exclude={'path'}), D.tau, D.tau_axis, D.evaluation_mode, start, n_nodes,
                       rng.bit_generator.state], sort_keys=True, default=str)
