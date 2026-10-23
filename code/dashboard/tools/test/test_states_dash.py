"""pytest for `tools.states_dash` figure-data builders.

Tests only the PURE builders (`build_trajectory`, `build_space_spread`) which
operate on a `state_analysis.RecordedStates` and numpy in -- no plotly, no Dash,
no Julia, no project.* import. Mirrors test_state_analysis.py / the sibling
readout_analysis test seam (dashboard on sys.path).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DASH_DIR = Path(__file__).resolve().parents[2]      # <repo>/code/dashboard
sys.path.insert(0, str(_DASH_DIR))

from tools.utils.state_analysis import reshape_recorded_states  # noqa: E402
from tools.states_dash import build_space_spread, build_trajectory  # noqa: E402


def _synthetic(entries=3, samples=4, nodes=8):
    rows, meta_rows = [], []
    rng = np.random.default_rng(0)
    for si in (1, 2, 3):
        for fi in (1,):
            for ti in (1, 2):
                for j in range(samples):
                    meta_rows.append({'phase': 'reservoir_layer', 's': si, 'f': fi,
                                      't': ti, 'sample_id': j})
                    rows.append(1000 * (si * 10 + ti) + rng.integers(0, 10, nodes))
    states = np.stack(rows).astype(np.uint8)
    return reshape_recorded_states(states, pd.DataFrame(meta_rows))


def test_build_trajectory_reduces_one_sample():
    rs = _synthetic()
    data = build_trajectory(rs, reducer='pca', sample_id=0)
    # one sample's trajectory: 3 s-positions x 2 t-positions = 6 points
    assert len(data.x) == len(data.y) == len(data.colour) == 6
    assert len(data.labels) == 6
    assert 'Trajectory' in data.title
    assert 'sample 0' in data.title


def test_build_space_spread_reduces_fixed_position():
    rs = _synthetic()
    data = build_space_spread(rs, reducer='pca')
    # all samples at a fixed (s,f,t): samples=4 points
    assert len(data.x) == len(data.y) == 4
    assert len(data.labels) == 4
    assert all('sample' in lb for lb in data.labels)
    assert 'Space spread' in data.title


def test_build_space_spread_explicit_position():
    rs = _synthetic()
    data = build_space_spread(rs, reducer='pca', position={'s': 2, 'f': 1, 't': 1})
    assert len(data.x) == 4  # 4 samples at that exact position


def test_build_trajectory_tsne():
    rs = _synthetic()
    data = build_trajectory(rs, reducer='tsne', sample_id=0)
    assert len(data.x) == 6


# ── The transposed view + points frames + figure assembly ──────

from tools.states_dash import (  # noqa: E402
    build_node_space,
    node_metadata,
    space_points_df,
    state_figure,
    time_points_df,
)
from tools.utils.readout_analysis import normalize  # noqa: E402


class _FakeSlice:  # reservoir 0..7, output 8..9 (synthetic has 8 nodes)
    pass


def _fake_model():
    from types import SimpleNamespace
    import networkx as nx
    g = nx.DiGraph()
    g.add_nodes_from(range(10))
    for k in range(2, 8):                 # inputs feed reservoir
        g.add_edge(k - 2, k)
    for k in range(8):                    # reservoir -> output
        g.add_edge(k, 8 + (k % 2))
    return SimpleNamespace(
        N_total=10,
        input_nodes_mask=np.array([n < 2 for n in range(10)]),
        out_slice=slice(8, 10),
        res_slice=slice(0, 8),
        readout_W=rng2.normal(size=(2, 8)) * 0.1,
        graph=g,
    )


rng2 = np.random.default_rng(3)


def test_build_node_space_points_are_nodes():
    rs = _synthetic()
    data = build_node_space(rs, reducer='pca', sample_id=0)
    # one point per NODE (transposed trajectory), labels name nodes
    assert len(data.x) == len(data.y) == len(data.labels) == 8
    assert data.labels[0] == 'node 0'
    assert 'transposed' in data.title and 'TIME' in data.title


def test_time_points_df_carries_provenance_and_coords():
    rs = _synthetic()
    emb = np.arange(12, dtype=float).reshape(6, 2)
    df = time_points_df(rs, 0, emb)
    assert list(df.columns) == ['s', 'f', 't', 'sample_id', 'position', 'x', 'y']
    assert df['position'].tolist() == list(range(6))
    assert df['x'].tolist() == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def test_space_points_df_joins_metadata_and_state_stats():
    rs = _synthetic(nodes=8)
    model = _fake_model()
    nm = node_metadata(model, rs)
    assert set(nm.columns) >= {'node', 'node_rel', 'part', 'depth', 'w0', 'w1',
                               'n_in', 'n_out'}
    # node 0/1 are inputs -> part I, depth 0
    assert nm.loc[0, 'part'] == 'I' and nm.loc[0, 'depth'] == 0
    # readout columns index the reservoir slice -> w0 defined for 0..7
    assert nm['w0'].notna().all() and len(nm) == 8

    emb = np.arange(16, dtype=float).reshape(8, 2)
    pts = space_points_df(rs, 0, emb, nm)
    assert {'mean', 'flips', 'x', 'y'} <= set(pts.columns)
    assert len(pts) == 8 and pts.index.tolist() == list(range(8))


def test_state_figure_time_vs_space_modes():
    rs = _synthetic()
    model = _fake_model()
    nm = node_metadata(model, rs)
    emb = np.arange(16, dtype=float).reshape(8, 2)
    pts = space_points_df(rs, 0, emb, nm)
    norm = normalize(pts)
    fig = state_figure(pts, norm.loc[pts.index], 'space', 'w0', 'Viridis',
                       uirevision='k')
    assert fig.data[0].mode == 'markers'
    assert fig.data[0].marker.colorbar.title.text == 'w0'
    assert fig.layout.uirevision == 'k'
    assert len(fig.data[0].hovertext) == 8

    tpts = time_points_df(rs, 0, np.arange(12, dtype=float).reshape(6, 2))
    tnorm = normalize(tpts)
    tfig = state_figure(tpts, tnorm.loc[tpts.index], 'time', 'position', 'Viridis')
    assert tfig.data[0].mode == 'lines+markers'
    assert tfig.data[0].marker.colorbar.title.text == 'position'


def test_state_figure_rejects_unknown_colour_attr():
    rs = _synthetic()
    tpts = time_points_df(rs, 0, np.arange(12, dtype=float).reshape(6, 2))
    norm = normalize(tpts)
    with pytest.raises(KeyError):
        state_figure(tpts, norm.loc[tpts.index], 'time', 'nope', 'Viridis')


def test_time_points_df_survives_real_pi_shape():
    # Real PI recordings carry `time` (no `t`) and NO reservoir_layer rows --
    # the interactive TIME view used to die KeyError('t') at app build. The
    # time->t normalisation in state_analysis.reshape_recorded_states (the one
    # gate every consumer flows through) must keep the provenance contract.
    rows, meta_rows = [], []
    rng = np.random.default_rng(1)
    for si in (1, 2, 3):
        for j in range(4):
            meta_rows.append({'phase': 'output_layer', 's': si, 'f': 1,
                              'time': si * 2, 'sample_id': j})
            rows.append(rng.integers(0, 10, 8))
    states, meta = np.stack(rows).astype(np.uint8), pd.DataFrame(meta_rows)
    rs = reshape_recorded_states(states, meta, phase='output_layer')
    df = time_points_df(rs, 0, np.arange(6, dtype=float).reshape(3, 2))
    assert list(df.columns) == ['s', 'f', 't', 'sample_id', 'position', 'x', 'y']
    assert df['t'].tolist() == [2, 4, 6]       # normalised from `time`
    assert df['position'].tolist() == [0, 1, 2]


def test_time_figure_survives_all_nan_t_header():
    # The i327601 crash: recording ships an all-NaN `t` header; reshape fills t
    # from `time`, and the TIME hover must format provenance without int(NaN).
    rows, meta_rows = [], []
    rng = np.random.default_rng(2)
    for si in (1, 2, 3):
        for j in range(4):
            meta_rows.append({'phase': 'output_layer', 's': si, 'f': 1,
                              't': np.nan, 'time': si * 2, 'sample_id': j})
            rows.append(rng.integers(0, 10, 8))
    states, meta = np.stack(rows).astype(np.uint8), pd.DataFrame(meta_rows)
    rs = reshape_recorded_states(states, meta, phase='output_layer')
    tpts = time_points_df(rs, 0, np.arange(6, dtype=float).reshape(3, 2))
    norm = normalize(tpts)
    fig = state_figure(tpts, norm.loc[tpts.index], 'time', 'position', 'Viridis')
    assert fig.data[0].mode == 'lines+markers'
    # hover carries the normalised t, not NaN
    assert 't=2' in fig.data[0].hovertext[0]