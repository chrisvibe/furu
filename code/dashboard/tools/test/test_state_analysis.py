"""pytest tests for `tools/utils/state_analysis`.

Lives in dashboard/tools/test/ (the repo's co-located test-folder convention).
Tests the PURE reduction module only (reshape_recorded_states, reduce,
time_trajectory, space_spread) against a synthetic `(rows, nodes)` history tensor
+ expanded meta -- no Julia, no server, no project.* import. Uses the same
dashboard-on-sys.path import seam as the sibling test so `tools.*` resolves here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# parents[2] == <repo>/code/dashboard (dashboard on sys.path only -- never
# drags in project.* / juliacall).
_DASH_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DASH_DIR))

from tools.utils.state_analysis import (  # noqa: E402
    PHASES,
    RecordedStates,
    available_phases,
    pick_default_phase,
    reduce,
    reshape_recorded_states,
)


def _synthetic(phase='reservoir_layer', entries=3, samples=4, nodes=8,
               s_vals=(1, 2, 3), f_vals=(1,), t_vals=(1, 2)):
    """Build a reload_history-shaped (rows, nodes) tensor + expanded meta.

    Each history entry is one metadata row expanded over `samples`. sample_id is
    0..samples-1 within a (phase, s, f) group, steps s/f repeat per batch, t only
    on reservoir_layer rows. Row i of the tensor <-> row i of expanded_meta.
    """
    rows = []
    meta_rows = []
    rng = np.random.default_rng(0)
    for si in s_vals:
        for fi in f_vals:
            for ti in t_vals if phase == 'reservoir_layer' else [None]:
                mrow = {'phase': phase, 's': si, 'f': fi}
                if phase == 'reservoir_layer':
                    mrow['t'] = ti
                for j in range(samples):
                    meta_rows.append({**mrow, 'sample_id': j})
                    rows.append(1000 * (si * 10 + (ti or 0)) + rng.integers(0, 10, nodes))
    states = np.stack(rows).astype(np.uint8)
    expanded_meta = pd.DataFrame(meta_rows)
    return states, expanded_meta


def test_reshape_restricts_phase_and_nodes():
    states, meta = _synthetic(phase='reservoir_layer')
    # mixed phases -> only reservoir kept
    both = _synthetic(phase='reservoir_layer')
    extra_input = pd.DataFrame({**meta.iloc[0].to_dict(), 'phase': 'input_layer'},
                               index=[99999])
    meta_mixed = pd.concat([meta, extra_input], ignore_index=True)
    states_mixed = np.vstack([states, np.zeros((1, 8), dtype=np.uint8)])
    rs = reshape_recorded_states(states_mixed, meta_mixed, phase='reservoir_layer')
    assert len(rs.states) == len(states)
    assert (rs.meta['phase'] == 'reservoir_layer').all()

    # node slice restricts the columns and pins node_offset
    rs2 = reshape_recorded_states(states, meta, phase='reservoir_layer', node_slice=slice(2, 6))
    assert rs2.states.shape[1] == 4
    assert rs2.node_offset == 2


def test_reshape_shape_mismatch_raises():
    states, meta = _synthetic()
    with pytest.raises(ValueError, match='must match'):
        reshape_recorded_states(states[:-1], meta)


def test_reshape_bad_phase_and_3d_raises():
    states, meta = _synthetic()
    with pytest.raises(ValueError, match='unknown phase'):
        reshape_recorded_states(states, meta, phase='nope')
    # a 3-D tensor (the naive reading of "history is (entries, samples, nodes)")
    # must be rejected, since the recorded layout is actually 2-D (row, node).
    bad3d = np.zeros((2, 3, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match='must be .rows, nodes'):
        reshape_recorded_states(bad3d, meta)


def test_time_trajectory_orders_by_position():
    states, meta = _synthetic(entries=3, samples=4, nodes=8)
    rs = reshape_recorded_states(states, meta, phase='reservoir_layer')
    # a trajectory for one sample must visit every (s, f, t) once, in position order
    sample0_states, sample0_meta = rs.time_trajectory(0)
    assert len(sample0_states) == 3 * 1 * 2          # s x f x t for reservoir
    assert (sample0_meta['sample_id'] == 0).all()
    assert list(sample0_meta['s']) == [1, 1, 2, 2, 3, 3]      # s-major, then t
    assert list(sample0_meta['t']) == [1, 2, 1, 2, 1, 2]      # t-minor within s


def test_space_spread_fixed_position_across_samples():
    states, meta = _synthetic(entries=3, samples=4, nodes=8)
    rs = reshape_recorded_states(states, meta, phase='reservoir_layer')
    sp, sp_meta = rs.space_spread({'s': 2, 'f': 1, 't': 1})
    assert len(sp) == 4                                     # all samples at that position
    assert sorted(sp_meta['sample_id']) == [0, 1, 2, 3]


def test_reduce_pca_embedding_shape_and_dtype():
    states, meta = _synthetic(entries=3, samples=4, nodes=8)
    emb, out_meta = reduce(states, meta, reducer='pca', n_components=2)
    assert emb.shape == (len(states), 2)
    assert len(out_meta) == len(states)
    assert (out_meta['sample_id'] == meta['sample_id']).all()


def test_reduce_pools_all_phase_rows_consistently():
    # reduce() embeds every recorded row of the phase; provenance must line up
    states, meta = _synthetic(entries=2, samples=4, nodes=8)
    emb, out_meta = reduce(states, meta, reducer='pca', phase='reservoir_layer')
    assert emb.shape[0] == len(states) == len(out_meta)


def test_reduce_unknown_reducer_raises():
    states, meta = _synthetic()
    with pytest.raises(ValueError, match='unknown reducer'):
        reduce(states, meta, reducer='lle')


def test_known_phases():
    assert set(PHASES) == {'init', 'input_layer', 'reservoir_layer', 'output_layer'}


# ── Phase helpers (the #71 PI-recording bug: no reservoir_layer rows) ────────

def test_available_phases_lists_only_present_in_canonical_order():
    # PI-shaped recording: init + input_layer + output_layer, NO reservoir_layer
    rows = []
    meta_rows = []
    rng = np.random.default_rng(0)
    for ph, n in (('init', 2), ('input_layer', 6), ('output_layer', 6)):
        for j in range(2):
            for _ in range(n):
                meta_rows.append({'phase': ph, 's': 1, 'f': 1, 't': 1,
                                  'sample_id': j})
                rows.append(rng.integers(0, 2, 8))
    meta = pd.DataFrame(meta_rows)
    assert available_phases(meta) == ('init', 'input_layer', 'output_layer')


def test_pick_default_phase_skips_absent_reservoir_layer():
    meta = pd.DataFrame({'phase': ['input_layer'] * 4 + ['output_layer'] * 4})
    # the historical hardcoded default 'reservoir_layer' selects zero rows here;
    # the picker must fall through to the first phase actually recorded
    assert pick_default_phase(meta) == 'output_layer'


def test_pick_default_phase_prefers_reservoir_layer_when_present():
    meta = pd.DataFrame({'phase': ['input_layer'] * 2 + ['reservoir_layer'] * 2})
    assert pick_default_phase(meta) == 'reservoir_layer'


def test_pick_default_phase_raises_on_no_known_phase():
    meta = pd.DataFrame({'phase': ['weird', 'weird']})
    with pytest.raises(ValueError, match='no known phases'):
        pick_default_phase(meta)


# ── time->t normalisation (real PI recordings carry `time`, never `t`) ───────

def test_reshape_maps_time_column_to_t():
    # PI-shaped recording: output_layer rows with `time`, NO explicit `t`
    # (measured on i374120: time = 2,4,6.. per sample). reshape must normalise
    # so the documented contract (`t` always present) holds for consumers.
    states, meta = _synthetic(phase='output_layer')
    assert 't' not in meta.columns          # precondition: no explicit t
    meta['time'] = meta['s'] * 2
    rs = reshape_recorded_states(states, meta, phase='output_layer')
    assert 't' in rs.meta.columns
    assert rs.meta['t'].tolist() == (rs.meta['s'] * 2).tolist()
    assert 'time' in rs.meta.columns        # non-destructive: kept alongside


def test_reshape_fills_all_nan_t_header():
    # i327601 shape: an older recorder emits a `t` HEADER that is entirely NaN
    # in every phase, with the real sequence time in `time`. An all-NaN t must
    # be treated as absent, not passed through (int(NaN) killed the TIME hover).
    states, meta = _synthetic(phase='output_layer')
    meta['t'] = np.nan
    meta['time'] = meta['s'] * 2
    rs = reshape_recorded_states(states, meta, phase='output_layer')
    assert rs.meta['t'].notna().all()
    assert rs.meta['t'].tolist() == (rs.meta['s'] * 2).tolist()
    assert 'time' in rs.meta.columns


def test_reshape_keeps_partially_populated_t():
    # a t column carrying ANY values is recorded data -- never overwritten
    states, meta = _synthetic(phase='output_layer')
    meta['t'] = np.nan
    meta.loc[0, 't'] = 7                    # one real value amid NaN
    meta['time'] = meta['s'] * 2
    rs = reshape_recorded_states(states, meta, phase='output_layer')
    assert rs.meta.loc[0, 't'] == 7         # untouched
    assert rs.meta['t'].isna().sum() == len(meta) - 1


def test_reshape_never_touches_explicit_t():
    # synthetic reservoir_layer shape: explicit `t`, no `time` -- guard silent
    states, meta = _synthetic(phase='reservoir_layer')
    assert 'time' not in meta.columns
    rs = reshape_recorded_states(states, meta, phase='reservoir_layer')
    assert 't' in rs.meta.columns and 'time' not in rs.meta.columns
    assert rs.meta['t'].tolist() == meta['t'].tolist()