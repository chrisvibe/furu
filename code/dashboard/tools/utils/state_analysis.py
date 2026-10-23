"""Pure state-dimension reduction for recorded reservoir history.

DRY home beside `readout_analysis.py` in dashboard/tools/utils/ (the single-(i,j)
model introspection cluster weight_explorer -> readout_analysis -> state_analysis
-> weights_dash / trace). None of these functions touches Julia or
boots a server; they reduce a recorded history tensor + its expanded meta into
clean, axes-separated embeddings.

Grounding (the bug this replaces): `visualization.py::plot_dynamics_history`
(OUTDATED, unused) zscores the whole history tensor and fits a 2-D PCA/tSNE over
a FLATTENED view that pools the sample axis AND the node axis, then colours every
point by a `(phase, s, f)` tuple. That confounds "state over time" with "state
over space": it is one colour blob, not a trajectory. This module separates them:

  - TIME axis: the reservoir-state trajectory over sequence positions for a single
    probe sample -- does the dynamics sweep an interpretable manifold?
  - SPACE axis: the node (feature) dimension at a fixed position across samples --
    the low-rank-vs-space-filling question (the "high-dim projection trick").

Layout contract (from `BatchedTensorHistoryWriter.reload_history` +
`BooleanReservoir._build_history_metadata`): the recorded tensor is
`(n_rows, N_total)` uint8, one row per (history-entry, sample); the expanded meta
has exactly one row per history row carrying `phase`, `s`, `f`, `t` and `sample_id`
(0..m-1 within that entry). Row i of the tensor <-> row i of the meta. All the
reducer functions below index into that pair, so the probe (Part B) can persist
one canonical `(recorded-states, expanded-meta)` pair and this module never has to
reverse-guess shape.

Seam vs `readout_analysis.normalize`: the
reducer deliberately does NOT consume `readout_analysis.normalize`. That pandas
function is a constant-column-fearing *colour/rescale* to [0,1] built for the
pandas colourbar consumer (the graph tool); `reduce*` here feed an (n_rows,
n_feat) numpy uint8 matrix into sklearn, and rescaling states to [0,1] would
distort the geometry PCA/UMAP operate on. The two outward guards share intent,
not code; do not "unify" them. (Also note the reducer path performs no zscore,
so the div-by-zero the old `plot_dynamics_history` hit is absent here.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PHASES = ('init', 'input_layer', 'reservoir_layer', 'output_layer')
DEFAULT_PHASE = 'reservoir_layer'
# Short reducer names the UI/CLI can offer; `_make_reducer` is the single
# constructor behind them (any sklearn transformer is also accepted directly).
REDUCERS = ('pca', 'tsne', 'umap')
# Ordinal columns that order the forward pass within a phase, most significant
# first. Reservoir states for a single sample trace through (s, f, t).
_POSITION_COLS = ('s', 'f', 't')


def available_phases(expanded_meta: pd.DataFrame, min_rows: int = 1) -> tuple[str, ...]:
    """Phases actually present in the recording, in canonical `PHASES` order.

    Real path-integration recordings carry NO 'reservoir_layer' rows (the
    record points are config-dependent) — so UIs and
    defaults must build their phase choices from THIS, never from `PHASES`, or
    the default selects zero rows on exactly the models under review.
    """
    counts = expanded_meta['phase'].value_counts()
    return tuple(p for p in PHASES if counts.get(p, 0) >= min_rows)


def pick_default_phase(
    expanded_meta: pd.DataFrame,
    prefer: tuple[str, ...] = ('reservoir_layer', 'output_layer', 'input_layer', 'init'),
) -> str:
    """Best available phase: the first `prefer` entry the recording actually has.

    Falls back to the first present phase in canonical order; raises only when
    the meta carries no known phase at all (a shape/contract problem worth
    failing loudly on).
    """
    present = available_phases(expanded_meta)
    for p in prefer:
        if p in present:
            return p
    if not present:
        found = sorted(set(expanded_meta['phase'].unique()))
        raise ValueError(f'no known phases in recording (found {found}; expected {PHASES})')
    return present[0]


@dataclass(frozen=True)
class RecordedStates:
    """Recorded state tensor + expanded meta, validated and phase-restricted.

    `states` is `(n_rows, n_feat)` uint8 (one row per recorded sample).
    `meta` is the matching expanded-meta sub-frame (same row order), carrying at
    least `phase`, `s`, `f`, `t` and `sample_id`.
    """

    states: np.ndarray
    meta: pd.DataFrame
    phase: str
    node_offset: int  # absolute node id of column 0 (for labelling)

    @property
    def n_samples(self) -> int:
        return int(self.meta['sample_id'].nunique())

    def time_trajectory(self, sample_id: int) -> tuple[np.ndarray, pd.DataFrame]:
        """Rows for one sample, ordered by `(s, f, t)`: the TIME axis view.

        Returns `(states, meta)` where `states` is `(n_positions, n_feat)` and
        `meta` is the matching rows (so a caller can label each point by s/f/t).
        """
        m = self.meta['sample_id'] == sample_id
        return self.states[m], self.meta[m]

    def space_spread(self, position: dict[str, int]) -> tuple[np.ndarray, pd.DataFrame]:
        """Rows across SAMPLES at one fixed `(s, f, t)` position: the SPACE axis.

        `position` is e.g. `{'s': 3, 'f': 1, 't': 1}`. Returns `(states, meta)` of
        shape `(n_samples, n_feat)` / matching rows.
        """
        m = pd.Series(True, index=self.meta.index)
        for col, val in position.items():
            m &= self.meta[col] == val
        mm = self.meta[m]
        # keep stable sample order
        return self.states[mm.index], mm


def reshape_recorded_states(
    history: np.ndarray,
    expanded_meta: pd.DataFrame,
    phase: str = DEFAULT_PHASE,
    node_slice: slice | None = None,
) -> RecordedStates:
    """Restrict the recorded states to one phase (+ optional node slice).

    `history` is `(n_rows, N_total)` uint8 and `expanded_meta` is its row-for-row
    expanded meta (from `BatchedTensorHistoryWriter.reload_history`). Returns a
    `RecordedStates` view whose rows are the history rows of `phase`, with `s`,
    `f`, `t`, `sample_id` provenance intact. Recordings that name the
    sequence-time column `time` instead of `t` (real PI recordings; see the
    guard below) are normalised so `t` is always present.
    """
    if phase not in PHASES:
        raise ValueError(f'unknown phase {phase!r}; expected one of {PHASES}')
    if 'phase' not in expanded_meta.columns or 'sample_id' not in expanded_meta.columns:
        raise ValueError('expanded_meta must carry `phase` and `sample_id` columns')
    if history.ndim != 2:
        raise ValueError(f'history must be (rows, nodes); got {history.shape}')
    if len(history) != len(expanded_meta):
        raise ValueError(
            f'history rows ({len(history)}) and expanded_meta rows ({len(expanded_meta)}) '
            'must match (reload_history grows them in lockstep)')

    keep = expanded_meta['phase'].to_numpy() == phase
    states = history[keep]
    meta = expanded_meta[keep].reset_index(drop=True)
    # Real PI recordings name the sequence-time column `time` and carry no
    # reservoir_layer rows (measured on i374120:
    # time = 2,4,6.. per sample). Older recorders instead emit a `t` HEADER
    # that is entirely NaN (measured on i327601) with the real sequence time in
    # `time`. Normalise both shapes to the documented contract here -- the ONE
    # gate every consumer flows through -- so points frames, hover labels and
    # `space_spread` position dicts can rely on a populated `t`. Non-destructive
    # to recorded data: `time` is kept alongside, and a `t` column carrying
    # values is never touched.
    if 'time' in meta.columns:
        t_unusable = 't' not in meta.columns or meta['t'].isna().all()
        if t_unusable:
            meta['t'] = meta['time']
    node_offset = node_slice.start if (node_slice is not None and node_slice.start is not None) else 0
    if node_slice is not None:
        states = states[:, node_slice]
    return RecordedStates(states=states, meta=meta, phase=phase, node_offset=node_offset)


def reduce(
    history: np.ndarray,
    expanded_meta: pd.DataFrame,
    reducer='pca',
    n_components: int = 2,
    *,
    phase: str = DEFAULT_PHASE,
    node_slice: slice | None = None,
    **kwargs,
) -> tuple[np.ndarray, pd.DataFrame]:
    """One-DRY-reducer entry point: restrict, then embed the phase's state rows.

    `history`/`expanded_meta` are the reload_history pair (see
    `reshape_recorded_states`). `reducer` is a short name (pca|tsne|umap) or an
    sklearn transformer. Returns `(embedding, provenance_meta)` where
    `embedding` is `(n_rows, n_components)` and `provenance_meta` is the
    matching rows (phase/s/f/t/sample_id) so a caller can colour points.

    This is a plain PCA/UMAP/tSNE over ALL state rows of the phase. For the two
    deliberately separated axes use `time_trajectory`/`space_spread` and reduce
    those sub-matrices; this helper is the shared low-level fit the callers use.
    """
    rs = reshape_recorded_states(history, expanded_meta, phase=phase, node_slice=node_slice)
    if rs.states.shape[0] < 2:
        raise ValueError('not enough recorded states to reduce (phase has < 2 rows)')
    emb = reduce_inline(rs.states, reducer, n_components, **kwargs)
    return emb, rs.meta.reset_index(drop=True)


def reduce_inline(states: np.ndarray, reducer='pca', n_components: int = 2,
                  **kwargs) -> np.ndarray:
    """Fit a reducer on an already-shaped state matrix `stats` (n_rows, n_feat).

    The shared low-level fit both `reduce` (whole-phase) and the viz builders
    (a single trajectory / a fixed-position spread) call, so PEMd/UMAP/tsne are
    constructed in exactly one place. `reducer` is a short name (pca|tsne|umap)
    or an sklearn transformer.
    """
    X = _as_float(states)
    # tSNE perplexity default must be < n_samples (a tiny trajectory would
    # otherwise fail with PREC:PERP scikit errors). Only default it here, where
    # the row count is known; never override an explicit perplexity.
    if isinstance(reducer, str) and reducer.strip().lower() in ('tsne', 't-sne'):
        kwargs.setdefault('perplexity', min(30, max(5, len(states) - 1)))
    est = _make_reducer(reducer, n_components, **kwargs) \
        if isinstance(reducer, str) else reducer
    return est.fit_transform(X)


def _as_float(states: np.ndarray) -> np.ndarray:
    if states.dtype == np.float64:
        return states
    if states.dtype != np.uint8:
        raise ValueError(f'unexpected state dtype {states.dtype}; expected uint8 or float64')
    return states.astype(np.float64)


def _make_reducer(reducer, n_components: int, **kwargs):
    """Lazily construct an sklearn manifold reducer by short name or return as-is.

    UMAP only if installed (optional; PCA/tSNE from sklearn always exist). Unknown
    strings and foreign callables are rejected loudly, not silently downgraded.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if not isinstance(reducer, str):
        return reducer
    n = reducer.strip().lower()
    if n == 'pca':
        return PCA(n_components=n_components, **kwargs)
    if n in ('tsne', 't-sne'):
        # fallback when no explicit perplexity reached here (reduce_inline sets
        # a sample-aware default; this guards the reduce()-with-dire-transfer path)
        kwargs.setdefault('perplexity', 30)
        return TSNE(n_components=n_components, **kwargs)
    if n == 'umap':
        try:
            from umap import UMAP as _UMAP
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError("UMAP requested but `umap` is not installed") from exc
        return _UMAP(n_components=n_components, **kwargs)
    raise ValueError(f'unknown reducer {reducer!r}; expected {"|".join(REDUCERS)} '
                     'or an sklearn transformer')