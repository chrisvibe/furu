"""
Tests for the polar dashboard's selection-highlight wedge math
(dashboards.polar._make_selection_overlays).

Covers the invariants of the highlight sectors:
- every direction owns an equal step-wide slice of the circle, whether isolated or
  part of a run, so total highlighted coverage always matches the true fraction of
  matching directions (e.g. half matching -> half the circle highlighted)
- 0 and 360 are NOT the same row (each combo owns exactly one slot, step=360/n_combos)
  but ARE step-apart neighbours through the seam, so a run touching one end merges
  with a run touching the other end when they're wrap-adjacent
"""
import sys
from pathlib import Path

# dashboards.polar lives under code/dashboard — mirror run.py's sys.path setup so
# this test can run standalone (pytest also picks up code/conftest.py for the
# project.* imports polar.py needs).
_DASHBOARD_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_DASHBOARD_DIR))

import pandas as pd
import pytest

from dashboards.polar import _make_selection_overlays, make_polar_figure


def _table(n, matching_indices, step=None):
    """n evenly-spaced directions (step=360/n); rows in `matching_indices` get mode='match'."""
    step = step if step is not None else 360 / n
    dirs = [round(i * step, 1) for i in range(n)]
    return [
        {'Direction': f'{d}°', 'mode': ('match' if i in matching_indices else 'other')}
        for i, d in enumerate(dirs)
    ]


_CONSTRAINT = [{'column_id': 'mode', 'value': 'match'}]


def _spans(overlays):
    """[(left, right), ...] sorted by left edge, for easy assertions."""
    return sorted(
        (o['theta'][0] - o['width'][0] / 2, o['theta'][0] + o['width'][0] / 2)
        for o in overlays
    )


def test_no_constraints_returns_empty():
    assert _make_selection_overlays([], _table(8, {0}), {}) == []


def test_no_match_returns_empty():
    table = _table(8, {0})
    assert _make_selection_overlays([{'column_id': 'mode', 'value': 'nope'}], table, {}) == []


def test_isolated_match_padded_half_step_both_sides():
    n = 8  # step = 45
    table = _table(n, {3})  # 135 degrees, no neighbours matching
    overlays = _make_selection_overlays(_CONSTRAINT, table, {})
    assert len(overlays) == 1
    left, right = _spans(overlays)[0]
    assert left == pytest.approx(135 - 22.5)
    assert right == pytest.approx(135 + 22.5)


def test_contiguous_run_padded_only_at_outer_edges():
    n = 8  # step = 45
    table = _table(n, {1, 2, 3})  # 45, 90, 135 degrees, one run
    overlays = _make_selection_overlays(_CONSTRAINT, table, {})
    assert len(overlays) == 1
    left, right = _spans(overlays)[0]
    assert left == pytest.approx(45 - 22.5)
    assert right == pytest.approx(135 + 22.5)


@pytest.mark.parametrize("matching_indices", [
    {0, 2, 4, 6, 8, 10},   # scattered, alternating
    {3, 4, 5, 6, 7, 8},    # one contiguous block, away from the seam
])
def test_half_matching_directions_covers_half_the_circle(matching_indices):
    n = 12  # step = 30
    table = _table(n, matching_indices)
    overlays = _make_selection_overlays(_CONSTRAINT, table, {})
    total_width = sum(o['width'][0] for o in overlays)
    assert total_width == pytest.approx(180.0)


def test_wraparound_run_merges_into_one_wedge():
    n = 12  # step = 30
    # last two + first two directions match, wrapping through the seam
    table = _table(n, {10, 11, 0, 1})
    overlays = _make_selection_overlays(_CONSTRAINT, table, {})
    assert len(overlays) == 1
    left, right = _spans(overlays)[0]
    assert right - left == pytest.approx(4 * 30)  # 4 directions, one step each
    # centre should fall on the seam (between the run's two middle directions, 330 and 0)
    center = (left + right) / 2
    assert center % 360 == pytest.approx(345.0) or center == pytest.approx(-15.0)


def test_isolated_match_at_seam_is_not_clamped():
    """Regression: an isolated match at exactly 0 degrees used to be clamped to [0, step/2]
    instead of getting its rightful symmetric half-step padding through the seam."""
    n = 12  # step = 30
    table = _table(n, {0})  # only direction 0 matches; its neighbours (330, 30) don't
    overlays = _make_selection_overlays(_CONSTRAINT, table, {})
    assert len(overlays) == 1
    left, right = _spans(overlays)[0]
    assert right - left == pytest.approx(30.0)
    assert left == pytest.approx(-15.0)
    assert right == pytest.approx(15.0)


def test_full_circle_match_covers_360():
    n = 10
    table = _table(n, set(range(n)))
    overlays = _make_selection_overlays(_CONSTRAINT, table, {})
    assert len(overlays) == 1
    total_width = sum(o['width'][0] for o in overlays)
    assert total_width == pytest.approx(360.0)


# ── center_seam (make_polar_figure) ─────────────────────────────────────────────

def _combo_df(n):
    return pd.DataFrame({
        'accuracy': range(n),
        'R_k_avg':  [1] * n,
        'F':        [f'combo{i}' for i in range(n)],
    })


def test_center_seam_default_offsets_by_half_step():
    """36 combos, step=10: seam-centered mode should start at 5 deg, not 0 deg,
    so the 0/360 reference sits in the middle of the wrap gap."""
    df = _combo_df(36)
    _, mapping_df, _ = make_polar_figure(df, factors=['F'], metric_col='accuracy', k_avg_col='R_k_avg')
    labels = mapping_df['Direction'].tolist()
    assert labels[0] == '5.0°'
    assert labels[-1] == '355.0°'


def test_center_seam_false_keeps_legacy_start_at_zero():
    df = _combo_df(36)
    _, mapping_df, _ = make_polar_figure(
        df, factors=['F'], metric_col='accuracy', k_avg_col='R_k_avg', center_seam=False,
    )
    labels = mapping_df['Direction'].tolist()
    assert labels[0] == '0.0°'
    assert labels[-1] == '350.0°'


# ── k̄ colours (make_polar_figure) ──────────────────────────────────────────────

def _k_df(ks):
    return pd.DataFrame({'accuracy': [0.5] * 2 * len(ks), 'R_k_avg': [k for k in ks for _ in range(2)],
                         'F': ['a', 'b'] * len(ks)})


def _line_colors(fig):
    return {t.name: t.line.color for t in fig.data if t.mode == 'lines'}


def test_k_colours_come_from_the_site_presets(tmp_path, monkeypatch):
    """A grid's k̄ (1.0 ... 8.0) takes the presets' colour for that k̄, whatever subset the view shows."""
    import dashboards.polar as polar
    from plotly.colors import sample_colorscale
    (tmp_path / 'figure_presets.yaml').write_text(
        'factors:\n  R_k_avg: {scale: Plasma_r, range: [0, 0.7778], levels: [1, 2, 3, 4, 5, 6, 7, 8]}\n')
    monkeypatch.setattr(polar, '_VIEWS_DIR', tmp_path / 'polar')
    full = _line_colors(make_polar_figure(_k_df([float(k) for k in range(1, 9)]), factors=['F'])[0])
    subset = _line_colors(make_polar_figure(_k_df([3.0, 8.0]), factors=['F'])[0])
    assert full['3.0'] == subset['3.0'] == sample_colorscale('Plasma_r', [0.7778 * 2 / 7])[0]
    assert full['8.0'] == subset['8.0']


def test_k_colours_without_presets_follow_k_order(tmp_path, monkeypatch):
    import dashboards.polar as polar
    import plotly.express as px
    monkeypatch.setattr(polar, '_VIEWS_DIR', tmp_path / 'polar')
    colors = _line_colors(make_polar_figure(_k_df([3.0, 8.0]), factors=['F'])[0])
    assert colors == {'3.0': px.colors.sequential.Plasma_r[0], '8.0': px.colors.sequential.Plasma_r[1]}


def test_a_contour_takes_its_preset_dash_and_hides_as_one_group(tmp_path, monkeypatch):
    """furu Fig 1: a head's contour takes its preset dash (its delay); the contour, its reference circle and its start
    mark share a legend group and a name, so one legend click or a saved trace_visibility hides all three."""
    import dashboards.polar as polar
    from dashboards.utils import apply_trace_vis
    (tmp_path / 'figure_presets.yaml').write_text('series:\n  h_d10: {color: 1, dash: dot, label: h d10}\n')
    monkeypatch.setattr(polar, '_VIEWS_DIR', tmp_path / 'polar')
    df = pd.DataFrame({'acc': [0.6, 0.7] * 2, 'head': ['h_d10'] * 2 + ['h_d5'] * 2, 'F': ['a', 'b'] * 2,
                       'floor': 0.5})
    fig, _, _ = make_polar_figure(df, factors=['F'], metric_col='acc', k_avg_col='head', reference_col='floor')
    contours = {t.name: t.line.dash for t in fig.data if t.mode == 'lines' and not t.name.endswith('floor')}
    assert contours == {'h d10': 'dot', 'h_d5': 'solid'}
    assert {t.legendgroup for t in fig.data if t.name.startswith('h d10')} == {'h d10'}
    apply_trace_vis(fig, {'h d10|lines': 'legendonly', 'h d10 floor|lines': 'legendonly',
                          'h d10|markers': 'legendonly'})
    assert [t.visible for t in fig.data if t.legendgroup == 'h d10'] == ['legendonly'] * 3


def test_direction_labels_are_compact_where_the_presets_spell_every_factor(tmp_path, monkeypatch):
    """furu Fig 2: 'O64' / 'X4' in place of 'override · r = 64' (the long labels were clipped at the frame); the metric
    names itself from the presets, under the circle, and the circle keeps its size with or without a legend."""
    import dashboards.polar as polar
    from dashboards.polar import _figure
    from dashboards.theme import register_templates
    register_templates()
    (tmp_path / 'figure_presets.yaml').write_text(
        'factors:\n  I_perturbation: {tick: {override: O, xor: X}}\n  r: {label: "r = {}", tick: "{}"}\n'
        'axis_titles:\n  "acc (max)": best design accuracy\n')
    monkeypatch.setattr(polar, '_VIEWS_DIR', tmp_path / 'polar')
    df = pd.DataFrame([dict(I_perturbation=p, r=r, k=2.0, acc=0.6) for p in ('override', 'xor') for r in (1, 64)])
    fig, _, _ = _figure(df, ['I_perturbation', 'r'], 'max', 'acc', 'k', ['light'], r_min=0.5, r_max=1,
                        direction_labels=True)
    assert sorted(fig.layout.polar.angularaxis.ticktext) == ['O1', 'O64', 'X1', 'X64']
    assert [a.text for a in fig.layout.annotations] == ['best design accuracy'] and fig.layout.title.text is None
    assert fig.layout.polar.domain.y[0] > 0 and fig.layout.legend.orientation == 'h'
    (tmp_path / 'figure_presets.yaml').write_text('factors:\n  r: {label: "r = {}"}\n')
    fig, _, _ = _figure(df, ['I_perturbation', 'r'], 'max', 'acc', 'k', ['light'], direction_labels=True)
    assert 'override · r = 64' in fig.layout.polar.angularaxis.ticktext
