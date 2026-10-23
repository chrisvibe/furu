from project.boolean_reservoir.code.reservoir import BooleanReservoir, BatchedTensorHistoryWriter
from project.boolean_reservoir.code.utils.utils import override_symlink, flush_subnormals
from project.boolean_reservoir.code.kqgr_probe import freeze
from pathlib import Path
import copy
import numpy as np


def _distinct_columns(states: np.ndarray) -> np.ndarray:
    """states' distinct nonzero columns, each scaled by sqrt(its multiplicity): with U the
    distinct columns and c their counts, states @ states.T == (U*sqrt(c)) @ (U*sqrt(c)).T,
    so this has states' nonzero singular values on a far smaller matrix."""
    cols = np.ascontiguousarray(states.T)
    keys = cols.view(np.dtype((np.void, cols.dtype.itemsize * cols.shape[1]))).ravel()
    _, first, counts = np.unique(keys, return_index=True, return_counts=True)
    u = cols[first].T * np.sqrt(counts).astype(states.dtype)
    return u[:, np.any(u != 0, axis=0)]


def spectral_norm(states: np.ndarray) -> float:
    """np.linalg.norm(states, 2), the largest singular value, on the distinct columns only."""
    u = _distinct_columns(np.asarray(states, dtype=np.float64))
    if u.size == 0:
        return 0.0
    with flush_subnormals():
        return float(np.linalg.svd(u, compute_uv=False).max())


def state_rank(states: np.ndarray, scale: float | None = None) -> int:
    """np.linalg.matrix_rank(states), computed on the distinct columns only.

    Reservoir state matrices (samples x nodes) repeat columns heavily: frozen
    nodes and redundancy copies are identical columns (a 2048-node PI job had 35
    distinct ones). With U the distinct columns and c their multiplicities,
    states @ states.T == (U*sqrt(c)) @ (U*sqrt(c)).T, so both matrices have the
    same nonzero singular values; all-zero columns add nothing and are dropped.
    The tolerance is matrix_rank's default for the ORIGINAL shape and dtype,
    so the rank decision is the same as before — just on a far smaller SVD.
    scale: judge the singular values against this magnitude instead of the matrix's own
    largest one — for a residual (e.g. an interaction term) whose "zero" is rounding noise
    of a larger matrix.
    """
    states = np.asarray(states)
    if not np.issubdtype(states.dtype, np.floating):
        states = states.astype(np.float64)   # what matrix_rank's svd would use
    u = _distinct_columns(states)
    if u.size == 0:
        return 0
    with flush_subnormals():
        s = np.linalg.svd(u, compute_uv=False)
    tol = (s.max() if scale is None else scale) * max(states.shape) * np.finfo(s.dtype).eps
    return int(np.count_nonzero(s > tol))


def compute_rank(model: BooleanReservoir, x, metric: str = '', reset_reservoir: bool = True) -> int:
    """Compute reservoir rank from the final state after a forward pass.

    Reads states_parallel directly — no history recording, no disk I/O.
    `metric` is accepted for call-site compatibility but not used.

    Forwards in chunks of T.batch_size and collects each chunk's final states:
    forward() overwrites states_parallel per batch, so a single post-hoc read
    would keep only the last batch's states and cap the rank at batch_size.
    """
    m = x.shape[0]
    bs = model.T.batch_size
    record = model.record
    model.record = False
    states_chunks = []
    try:
        model.eval()
        for ka in range(0, m, bs):
            xb = x[ka:ka + bs]
            _ = model(xb)
            states_chunks.append(
                model.states_parallel[:xb.shape[0], model.res_slice].astype(np.float32))
    finally:
        model.record = record
    states = np.concatenate(states_chunks, axis=0)
    if reset_reservoir:
        model.reset_reservoir(hard_reset=True)
    return state_rank(states)


def gr_steps_probe(x, margin_bits: int = 2) -> tuple:
    """Build a steps-axis GR probe from a KQ probe set.

    Overwrites the trailing tau_steps steps of every stream with stream 0's
    tail, so streams differ only in their earliest steps. The rank of the
    resulting final/readout states counts distinctions RETAINED from old
    inputs across tau_steps further updates — retention rank. This is
    Snyder's GR construction on the time axis, for datasets whose native
    KQGR runs on the resolution axis (PI) and therefore never probes
    retention.

    tau_steps is auto-sized: the differing head keeps at least
    log2(m)+margin_bits bits so the probe set can stay distinct. Returns
    (x_shared, tau_steps), or (None, 0) when no tau >= 1 leaves enough head
    bits (e.g. 1-bit-per-step temporal streams, where the native GR already
    probes the steps axis).
    """
    m, s = x.shape[0], x.shape[1]
    bits_per_step = int(np.prod(x.shape[2:]))
    need = int(np.ceil(np.log2(max(m, 2)))) + margin_bits
    head_steps = max(1, int(np.ceil(need / bits_per_step)))
    tau = s - head_steps
    if tau < 1:
        return None, 0
    x_shared = freeze(x, tau, axis='steps', evaluation='last', rng=np.random.default_rng(), ref_idx=0)
    return x_shared, tau


def collect_states(model: BooleanReservoir, x, heads=None, reset_reservoir: bool = True) -> tuple:
    """One chunked forward pass -> ({head or None: readout states}, final reservoir states),
    float32. The readout states are model.get_states(x) — what the ridge readout trains on
    (time-series: per-step states summed; classification: the final state)."""
    m = x.shape[0]
    bs = model.T.batch_size
    keys = list(heads) if heads is not None else [None]
    readout_chunks, last_chunks = {k: [] for k in keys}, []
    model.eval()
    for ka in range(0, m, bs):
        xb = x[ka:ka + bs]
        per_key = model.get_states(xb, heads=heads) if heads is not None else {None: model.get_states(xb)}
        for k, (states, _) in per_key.items():
            readout_chunks[k].append(np.asarray(states, dtype=np.float32))
        last_chunks.append(
            model.states_parallel[:xb.shape[0], model.res_slice].astype(np.float32))
    if reset_reservoir:
        model.reset_reservoir(hard_reset=True)
    return {k: np.concatenate(v, axis=0) for k, v in readout_chunks.items()}, np.concatenate(last_chunks, axis=0)


def compute_rank_pair(model: BooleanReservoir, x, reset_reservoir: bool = True, heads=None):
    """Rank of the pre-readout representation AND of the final reservoir state,
    both collected from one chunked forward pass.

    readout rank (`kqgr`): ranks model.get_states(x) — the representation the
    ridge readout actually trains on. In time-series output mode that is the
    per-step states summed over steps; in classification mode it is the final
    state, so the two ranks coincide there (up to the bipolar affine shift).

    last rank (`kqgr_last`): ranks the final reservoir state — Snyder's
    original (samples x nodes) construction. Kept alongside so the Snyder 2012
    replication stays comparable to the published definition.

    heads=[modes]: every head's readout rank from the SAME pass (the last rank is
    mode-independent) -> {head: (readout_rank, last_rank)}; else one tuple.
    """
    readouts, last = collect_states(model, x, heads, reset_reservoir)
    last_rank = None
    out = {}
    for k, readout in readouts.items():
        last_rank = state_rank(last) if last_rank is None else last_rank
        # classification with binary encoding: the readout IS the final state
        readout_rank = last_rank if np.array_equal(readout, last) else state_rank(readout)
        out[k] = (readout_rank, last_rank)
    return out if heads is not None else out[None]


def walk_query_split(S: np.ndarray) -> dict:
    """Two-way split of a crossed state table S (L, m, N) — m inputs (walks) each under L
    levels of a second input (queries), N nodes:
        S[l, w] = grand + A[w] + B[l] + E[l, w]
    A = the walk main effect, B = the level main effect (one fixed offset per level),
    E = the INTERACTION: how much the level's effect on the state depends on the walk. A
    linear readout of additive states (E = 0) can only output f(walk) + c_level, so it
    cannot pick a different function of the walk per level; that part must come from E.
    The parts are orthogonal, so |S - grand|^2 = L|A|^2 + m|B|^2 + |E|^2. Returns
        kq                  rank of S                 (raw — rises by the L-1 level directions for free)
        rank_walk, rank_query, rank_interaction       ranks of A, B, E
        interaction_energy  |E|^2 / |S - grand|^2     (interaction share of all variation)
        interaction_share   |E|^2 / (m|B|^2 + |E|^2)  (interaction share of the level-dependent part)
    """
    S = S.astype(np.float64)
    L, m, N = S.shape
    grand = S.mean(axis=(0, 1))
    A = S.mean(axis=0) - grand
    B = S.mean(axis=1) - grand
    E = S - grand - A[None] - B[:, None]
    e2, b2 = float((E ** 2).sum()), float((B ** 2).sum()) * m
    total = float(((S - grand) ** 2).sum())
    scale = spectral_norm(S.reshape(-1, N) - grand)   # the residuals' zero is S's rounding
    return dict(kq=state_rank(S.reshape(-1, N)), rank_walk=state_rank(A, scale), rank_query=state_rank(B, scale),
                rank_interaction=state_rank(E.reshape(-1, N), scale),
                interaction_energy=e2 / total if total else 0.0, interaction_share=e2 / (b2 + e2) if b2 + e2 else 0.0)


def compute_walk_query_split(model: BooleanReservoir, x, levels: int, heads=None) -> dict:
    """walk_query_split of the readout states of x, a crossed probe of `levels` blocks
    (level-major: block l is every walk under level l) -> {head or None: split}."""
    readouts, _ = collect_states(model, x, heads)
    return {k: walk_query_split(r.reshape(levels, -1, r.shape[1])) for k, r in readouts.items()}


def compute_rank_flexible(model: BooleanReservoir, x, metric: str, reset_reservoir: bool = True) -> int:
    """Compute reservoir rank via full history recording (flexible but slower).

    Records all phases, filters to the final output_layer step, and returns the
    rank of the resulting (m × N_R) state matrix.  Use when you need to inspect
    intermediate states or save history artefacts alongside the rank.
    """
    nested_out = model.L.save_path / 'history' / metric
    new_save_path = nested_out / 'history'

    record = model.record
    try:
        model.record = True
        if model.history:
            model.history = BatchedTensorHistoryWriter(
                save_path=new_save_path,
                persist_to_disk=model.history.persist_to_disk,
                buffer_size=model.history.buffer_size
            )
        else:
            model.history = BatchedTensorHistoryWriter(save_path=new_save_path, persist_to_disk=False)
        model.eval()
        _ = model(x)
        model.flush_history()
    finally:
        model.record = record

    override_symlink(Path('../../checkpoint'), new_save_path / 'checkpoint')
    load_dict, history, expanded_meta, meta = model.history.reload_history()
    output_layer = expanded_meta[expanded_meta['phase'] == 'output_layer']
    final_s = output_layer['s'].max()
    df_filter = output_layer[output_layer['s'] == final_s]
    filtered_history = history[df_filter.index].astype(np.float32)
    reservoir_node_history = filtered_history[:, model.res_slice]
    if reset_reservoir:
        model.reset_reservoir(hard_reset=True)
    return state_rank(reservoir_node_history)