"""pytest for `tools.dash_components` -- the shared dashboard chrome.

Tests the pure pandas/numpy layer (ComputedFields, colorbar_ticks, clamp_top_n,
fmt_val, part_labels) without plotly or Dash; the dash-dependent query-help
builders are covered by the live-server probes and the dashboards' own wiring.
Mirrors the sibling test seam (dashboard on sys.path, no project.* import).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_DASH_DIR = Path(__file__).resolve().parents[2]      # <repo>/code/dashboard
sys.path.insert(0, str(_DASH_DIR))

from tools.utils.dash_components import (  # noqa: E402
    ComputedFields,
    clamp_top_n,
    colorbar_ticks,
    fmt_val,
    part_labels,
)
from tools.utils.readout_analysis import normalize  # noqa: E402


# ── clamp_top_n ───────────────────────────────────────────────────────────────

def test_clamp_top_n_zero_and_none_mean_uncapped():
    assert clamp_top_n(None, 100) == 0
    assert clamp_top_n(0, 100) == 0
    assert clamp_top_n(-5, 100) == 0


def test_clamp_top_n_caps_at_population():
    assert clamp_top_n(50, 32) == 32
    assert clamp_top_n(10, 100) == 10


# ── fmt_val ───────────────────────────────────────────────────────────────────

def test_fmt_val_float_vs_other():
    assert fmt_val(0.12345678) == '0.1235'
    assert fmt_val(np.float64(2.0)) == '2'
    assert fmt_val('abc') == 'abc'
    assert fmt_val(7) == '7'


# ── colorbar_ticks ────────────────────────────────────────────────────────────

def test_colorbar_ticks_numeric_maps_back_to_raw():
    df = pd.DataFrame({'v': [10.0, 20.0, 30.0]})
    tickvals, ticktext = colorbar_ticks(df, 'v')
    assert tickvals == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert ticktext[0] == '10' and ticktext[-1] == '30'


def test_colorbar_ticks_constant_column_single_tick():
    df = pd.DataFrame({'v': [5.0, 5.0, 5.0]})
    tickvals, ticktext = colorbar_ticks(df, 'v')
    assert tickvals == [0.0]
    assert ticktext == ['5']


def test_colorbar_ticks_categorical_lists_every_category():
    df = pd.DataFrame({'p': ['I', 'R', 'O', 'R']})
    tickvals, ticktext = colorbar_ticks(df, 'p')
    assert ticktext == ['I', 'R', 'O']
    assert len(tickvals) == 3 and tickvals[0] == 0.0 and tickvals[-1] == 1.0


# ── ComputedFields ────────────────────────────────────────────────────────────

def test_computed_parse_register_and_clear():
    cf = ComputedFields()
    assert cf.parse('hw = abs(w0)') == 'Computed field: hw = abs(w0)'
    assert cf.fields == {'hw': 'abs(w0)'}
    assert cf.names() == ['hw']
    # empty box clears everything (the graph tool's long-standing semantics)
    assert cf.parse('') == ''
    assert cf.fields == {}


def test_computed_parse_invalid_inputs():
    cf = ComputedFields()
    assert 'Invalid' in cf.parse('no_equals_sign')
    assert cf.fields == {}
    assert 'Invalid' in cf.parse('   = abs(w0)')   # empty name
    assert 'Invalid' in cf.parse('name =    ')     # empty expr


def test_computed_apply_evals_into_df():
    cf = ComputedFields()
    cf.parse('wsum = w0 + w1')
    df = pd.DataFrame({'w0': [1.0, 2.0], 'w1': [3.0, 4.0]})
    applied = cf.apply(df)
    assert applied == [('wsum', 'w0 + w1')]
    assert df['wsum'].tolist() == [4.0, 6.0]


def test_computed_apply_swallows_bad_expr_and_reports():
    cf = ComputedFields()
    cf.parse('bad = nonexistent_col * 2')
    df = pd.DataFrame({'w0': [1.0]})
    applied = cf.apply(df)          # must not raise
    assert applied == []
    assert 'bad' not in df.columns


def test_computed_normalize_into_uses_shared_guard():
    cf = ComputedFields()
    cf.parse('c = a * 0')           # constant result -> 0.0, not NaN/div-zero
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
    df['c'] = 0.0
    norm = normalize(df[['a']])
    cf.normalize_into(df, norm)
    assert 'c_normalized_' in norm.columns
    assert norm['c_normalized_'].tolist() == [0.0, 0.0, 0.0]
    # idempotent: an existing _normalized_ column is not recomputed
    norm['c_normalized_'] = 42.0
    cf.normalize_into(df, norm)
    assert (norm['c_normalized_'] == 42.0).all()


def test_computed_options_merge_base_attrs():
    cf = ComputedFields()
    cf.parse('z = w0')
    assert cf.options(['w0', 'part']) == ['part', 'w0', 'z']


# ── part_labels (the I/O/R semantics) ─────────────────────────────────────────

def test_part_labels_uses_masks_not_graph_attrs():
    model = SimpleNamespace(
        N_total=10,
        input_nodes_mask=np.array([n < 2 for n in range(10)]),
        out_slice=slice(8, 10),
    )
    labels = part_labels(model)
    assert labels.tolist() == ['I', 'I', 'R', 'R', 'R', 'R', 'R', 'R', 'O', 'O']
