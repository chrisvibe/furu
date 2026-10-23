"""pytest tests for `tools/utils/readout_analysis` (dashboard readout-strategy stats).

Lives in dashboard/tools/test/ (the repo's co-located test-folder convention,
matching code/project/*/test/ and dashboard/wsgi/apps/test/). Purposely tests the
PURE analysis module only
(`per_output_row_diagnostics`, `fuse_with_depth`, `global_concentration`,
`cascade_depth_from_inputs`, `apply_pandas_query`) -- not
`tools/weights_dash.py` (the wrapper that consumes this module), which pulls in
dash/plotly and builds a server.
This is the intended seam: the strategy diagnostics are unit-testable without
booting the app.

These are NECESSARY-not-
sufficient signals for compute-vs-hash; the tests pin the geometry math, not an
interpretation verdict.
"""
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
import pytest

# parents[2] == <repo>/code/dashboard. dashboard on sys.path only -- never
# drags in project.* / juliacall.
_DASH_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DASH_DIR))

from tools.utils.readout_analysis import (  # noqa: E402
    apply_pandas_query,
    cascade_depth_from_inputs,
    fuse_with_depth,
    global_concentration,
    normalize,
    per_output_row_diagnostics,
)


def test_normalize_ranges():
    df = pd.DataFrame({'a': [0.0, 0.5, 1.0], 'b': ['x', 'y', 'x']})
    out = normalize(df)
    assert out['a_normalized_'].tolist() == pytest.approx([0.0, 0.5, 1.0])
    # categorical factorized -> in range, distinct unless repeat
    assert out['b_normalized_'].max() <= 1.0 and out['b_normalized_'].min() >= 0.0
    assert out['b_normalized_'].iloc[0] == out['b_normalized_'].iloc[2]  # repeated cat
    assert out['b_normalized_'].iloc[0] != out['b_normalized_'].iloc[1]


def test_normalize_constant_column_is_zero_not_nan():
    # constant numeric (was div-by-zero -> NaN) and constant categorical
    out = normalize(pd.DataFrame({'c': [5, 5, 5], 'd': ['LOOP', 'LOOP', 'LOOP']}))
    assert out['c_normalized_'].notna().all()
    assert out['c_normalized_'].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert not out['d_normalized_'].isna().any()
    assert out['d_normalized_'].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_normalize_single_row_and_suffix():
    single = normalize(pd.DataFrame({'v': [7.0]}))
    assert single['v_normalized_'].tolist() == pytest.approx([0.0])
    custom = normalize(pd.DataFrame({'w': [1.0, 3.0]}), suffix='_scaled')
    assert 'w_scaled' in custom.columns and 'w_normalized_' not in custom.columns


def test_per_output_single_node_pointer():
    W = np.zeros((1, 5))
    W[0] = np.r_[np.array([10.0]), np.zeros(4)]
    rows = per_output_row_diagnostics(W)
    assert rows[0]['pr'] == pytest.approx(1.0)
    assert rows[0]['support'] == 1
    assert rows[0]['single_node'] is True
    assert rows[0]['top_node'] == 0
    assert rows[0]['top_share'] == pytest.approx(1.0)


def test_per_output_uniform_hash():
    W = np.full((1, 200), 0.05)
    rows = per_output_row_diagnostics(W)
    assert rows[0]['pr'] == pytest.approx(200.0)
    assert rows[0]['support'] == 200
    assert rows[0]['single_node'] is False


def test_per_output_allzero_output_handled():
    W = np.zeros((2, 50))
    rows = per_output_row_diagnostics(W)
    for r in rows:
        assert r['top_node'] is None
        assert r['single_node'] is False


def test_per_output_1d_wrapped():
    # a 1-D readout (single output) should be treated as one row
    rows = per_output_row_diagnostics(np.array([1.0, 0.0, 0.0]))
    assert len(rows) == 1


def test_global_gini_endpoints_and_constant():
    # all-equal -> 0 (uniform)
    assert global_concentration(np.full(10, 3.0)) == pytest.approx(0.0)
    # single non-zero among N nodes -> (N-1)/N (Gini reaches 1 only as N->large;
    # with N=10 a single-hot vector is 0.9).
    v = np.zeros(10); v[0] = 1.0
    assert global_concentration(v) == pytest.approx(0.9)
    # all-zero -> 0 (the constant-colour path must not divide by zero)
    assert global_concentration(np.zeros(10)) == 0.0


def test_cascade_depth_multi_source():
    G = nx.DiGraph()
    G.add_nodes_from(range(7))
    G.add_edges_from([(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6)])
    mask = np.zeros(7, dtype=bool); mask[0] = mask[1] = True
    depth = cascade_depth_from_inputs(G, mask)
    assert depth[0] == 0 and depth[1] == 0       # input nodes
    assert depth[2] == 1                          # directly input-shadowing
    assert depth[3] == 2 and depth[4] == 3        # deeper -> needs recurrence
    assert depth[5] == 4 and depth[6] == 4


def test_cascade_depth_unreachable_is_none():
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1)  # node 2 isolated -> unreachable from input 0
    mask = np.zeros(3, dtype=bool); mask[0] = True
    depth = cascade_depth_from_inputs(G, mask)
    assert depth[2] is None


def test_apply_pandas_query_valid_and_error():
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'part': ['I', 'R', 'R', 'I']})
    ok, err = apply_pandas_query(df, 'a > 2')
    assert len(ok) == 2 and err == ''
    # bad query: unchanged frame + surfaced message (NOT silent fallback)
    kept, err2 = apply_pandas_query(df, 'nonexistent > 0')
    assert len(kept) == len(df)
    assert 'Pandas query failed' in err2
    # empty string: unchanged, no error
    same, err3 = apply_pandas_query(df, '   ')
    assert len(same) == len(df) and err3 == ''


def test_fuse_deep_node_gives_compute_verdict():
    # top_node reads a DEEP node (depth 2) -> "computed" verdict (+1)
    depth = {100: 2}                                    # abs node 100 is deep
    geoms = [dict(pr=1.0, support=1, top_share=1.0, top_node=3, single_node=True)]
    fused = fuse_with_depth(geoms, depth, res_slice_start=97)  # abs = 97+3 = 100
    assert fused[0]['top_depth'] == 2
    assert fused[0]['depth_lbl'] == 'deep'
    assert fused[0]['verdict_int'] == +1
    assert 'DEEP node' in fused[0]['verdict']
    # input list must be untouched: the source dict gains no new keys
    assert 'verdict' not in geoms[0] and 'verdict_int' not in geoms[0]


def test_fuse_shallow_node_gives_hash_verdict():
    depth = {5: 1}
    geoms = [dict(pr=1.0, support=1, top_share=1.0, top_node=0, single_node=True)]
    fused = fuse_with_depth(geoms, depth, res_slice_start=5)  # abs = 5
    assert fused[0]['depth_lbl'] == 'shallow'
    assert fused[0]['verdict_int'] == -1
    assert 'SHALLOW' in fused[0]['verdict']


def test_fuse_spread_hash_verdict():
    # spread over the readout -> readout doing the work -> hash verdict, regardless of depth
    depth = {7: 2}
    geoms = [dict(pr=50.0, support=50, top_share=0.02, top_node=2, single_node=False)]
    fused = fuse_with_depth(geoms, depth, res_slice_start=5)
    assert fused[0]['verdict_int'] == -1
    assert 'spread' in fused[0]['verdict'].lower()


def test_fuse_unused_output():
    depth = {}
    geoms = [dict(pr=0.0, support=0, top_share=0.0, top_node=None, single_node=False)]
    fused = fuse_with_depth(geoms, depth, 0)
    assert fused[0]['verdict_int'] == 0
    assert 'unused' in fused[0]['verdict']