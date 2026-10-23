"""pytest tests for the site figure presets (figure_presets.yaml): one colour per series across a
site's figures, shared export axis titles and figure defaults, resolved into the scatter build
(the hegg stages, then furu's heads).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # code/dashboard, as the other tests

from dashboards.utils import (  # noqa: E402
    load_figure_presets, preset_series, preset_series_symbol, preset_color_offset, resolve_figure_options,
)

PRESETS = """
series:
  integration: {color: 0, label: integration}
  retention:   {color: 1}
  selection:   {color: {h: 4}}
axis_titles: {R_k_avg: "k\\u0304", log10_nmse: NMSE}
figure: {legend: per-series, title: false}
"""


@pytest.fixture
def site(tmp_path):
    (tmp_path / 'figure_presets.yaml').write_text(PRESETS)
    views = tmp_path / 'scatter' / 'app'
    views.mkdir(parents=True)
    return views


def test_presets_are_found_above_the_views_dir(site):
    p = load_figure_presets(site)
    assert set(p['series']) == {'integration', 'retention', 'selection'}


def test_no_presets_file_gives_empty(tmp_path):
    assert load_figure_presets(tmp_path) == {}


def test_presets_are_reread_when_the_file_changes(site):
    f = site.parents[1] / 'figure_presets.yaml'
    assert load_figure_presets(site)['series']['retention']['color'] == 1
    import os
    f.write_text(PRESETS.replace('{color: 1}', '{color: 5}'))
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 5))
    assert load_figure_presets(site)['series']['retention']['color'] == 5


@pytest.mark.parametrize('name, series', [
    ('integration', 'integration'),
    ('retention [dev_test]', 'retention'),        # '<group> [<y>]'
    ('2D | selection [log10_nmse]', 'selection'),  # one part of a combined group
    ('linear [log10_nmse]', None),
])
def test_series_match(site, name, series):
    assert preset_series(load_figure_presets(site), name) == series



def test_a_symbol_comes_from_any_part_of_a_combined_group():
    """Colour and dash come from the first matching part (the task), the marker from whichever part has one
    (the injection), so one figure can carry three factors."""
    p = {'series': {'density_w5_d5': {'color': 0, 'dash': 'solid'}, 'xor': {'color': 1, 'symbol': 'circle-open'}}}
    assert preset_series(p, 'density_w5_d5 | xor') == 'density_w5_d5'
    assert preset_series_symbol(p, 'density_w5_d5 | xor [acc]') == 'circle-open'
    assert preset_series_symbol(p, 'density_w5_d5') is None and preset_series_symbol({}, 'xor') is None


def test_series_offsets_resolve_to_palette_offsets(site):
    p = load_figure_presets(site)
    assert preset_color_offset(p, {'series': 'retention'}) == {'d': 1}
    assert preset_color_offset(p, {'series': 'selection'}) == {'h': 4}
    assert preset_color_offset(p, {'series': 'nope'}) is None
    assert preset_color_offset(p, {'d': 3}) == {'d': 3}


def test_figure_options_take_the_site_defaults_and_titles(site):
    p = load_figure_presets(site)
    out = resolve_figure_options({'legend_swatch': 'line', 'title': True,
                                  'names': {'linear [log10_nmse]': 'integration', 'x': 'free text', 'y': None}},
                                 p, 'R_k_avg', ['log10_nmse', 'log10_nmse_steps'])
    assert out['legend'] == 'per-series' and out['title'] is True      # the view's own keys win
    assert out['x_title'] == 'k̄' and out['y_title'] == 'NMSE'
    assert out['names'] == {'linear [log10_nmse]': 'integration', 'x': 'free text', 'y': None}


def test_views_without_a_figure_block_are_left_alone(site):
    assert resolve_figure_options(None, load_figure_presets(site), 'R_k_avg', ['log10_nmse']) is None


def test_scatter_colours_traces_by_series_unless_the_view_overrides(site):
    from dashboards.scatter import ScatterDashboard
    from dashboards.theme import register_templates, resolve as resolve_theme
    register_templates()
    df = pd.DataFrame({'x': [1, 2] * 3, 'y': range(6), 'stage': ['integration'] * 2 + ['retention'] * 2 + ['other'] * 2})
    d = ScatterDashboard(df, ['stage'], renderer='svg', initial_views_dir=str(site))
    palette = resolve_theme(['light']).palette
    view = {'x-axis': 'x', 'y-axis': 'y', 'color-by-factor': ['stage'], 'overlay-dots-toggle': ['on'],
            '_extras': {'color_offsets': {'other': {'series': 'integration'}, 'retention': {'d': 7}}}}
    fig = d._reconstruct_figure(view, theme_override=['light'])
    colour = {t.name: t.marker.color for t in fig.data}
    assert colour['integration'] == palette[0]
    assert colour['retention'] == palette[7 % len(palette)]   # explicit offset beats the series
    assert colour['other'] == palette[0]                       # a view can point a trace at a series


def test_hegg_views_resolve_against_the_hegg_presets():
    """Every hegg view that names a series names one the hegg presets define."""
    root = Path(__file__).resolve().parents[4].parent / 'out' / 'dashboard' / 'hegg'
    if not (root / 'figure_presets.yaml').exists():
        pytest.skip('no hegg views here')
    series = set(load_figure_presets(root / 'scatter')['series'])
    for f in root.rglob('*.json'):
        offsets = (json.loads(f.read_text()).get('_extras') or {}).get('color_offsets') or {}
        for off in offsets.values():
            if isinstance(off, dict) and 'series' in off:
                assert off['series'] in series, f


def test_series_labels_rename_unlisted_traces_on_export(site):
    import plotly.graph_objects as go
    from dashboards.utils import apply_figure_options
    fig = go.Figure()
    for name in ('integration', 'integration [dev_test]', 'retention', 'other'):
        fig.add_scatter(x=[1], y=[1], name=name)
    opts = resolve_figure_options({'names': {'retention': 'kept'}}, load_figure_presets(site), 'x', ['y'])
    apply_figure_options(fig, opts)
    assert [t.name for t in fig.data] == ['integration', 'integration [dev_test]', 'kept', 'other']
    opts = resolve_figure_options({}, {'series': {'integration': {'color': 0, 'label': 'integr.'}}}, 'x', ['y']) or {}
    assert opts == {}   # an empty figure block is no figure block: views opt in


# ── ordered colour factors and export text ratios ───────────────────────────────────
FACTOR = {'factors': {'r': {'scale': 'Viridis', 'range': [0, 0.85], 'levels': [1, 4, 16, 64], 'label': 'r = {}'}}}


def test_a_level_keeps_its_colour_whatever_levels_a_view_shows(tmp_path):
    from plotly.colors import sample_colorscale
    from dashboards.utils import preset_factor_colors
    colours = preset_factor_colors(FACTOR, 'r')
    assert list(colours) == ['1', '4', '16', '64']
    assert colours['1'] == sample_colorscale('Viridis', [0.0])[0]
    assert colours['64'] == sample_colorscale('Viridis', [0.85])[0]
    assert preset_factor_colors(FACTOR, 'other') == {} and preset_factor_colors({}, 'r') == {}

    from dashboards.scatter import ScatterDashboard
    from dashboards.theme import register_templates
    import yaml
    register_templates()
    (tmp_path / 'figure_presets.yaml').write_text(yaml.safe_dump(FACTOR))
    df = pd.DataFrame({'x': [1, 2] * 4, 'y': range(8), 'r': [1, 1, 16, 16, 64, 64, 4, 4]})
    d = ScatterDashboard(df, ['r'], renderer='svg', initial_views_dir=str(tmp_path))
    for extras, expect_16 in (({}, colours['16']), ({'color_offsets': {'16': {'hsv': [0, 100, 100]}}}, '#ff0000')):
        view = {'x-axis': 'x', 'y-axis': 'y', 'color-by-factor': ['r'], 'overlay-dots-toggle': ['on'], '_extras': extras}
        fig = d._reconstruct_figure(view, theme_override=['light'])
        got = {t.name: t.marker.color for t in fig.data}
        assert got['1'] == colours['1'] and got['16'] == expect_16    # the view's explicit offset wins


def test_combined_groups_take_the_first_factors_colour_and_a_later_factors_symbol(tmp_path):
    """Colour by s and mark by ticks in one scatter: a combined group '16 | 2' takes r = 16's colour and ticks 2's
    marker from the factor presets."""
    from dashboards.scatter import ScatterDashboard
    from dashboards.theme import register_templates
    from dashboards.utils import preset_factor_colors, preset_factor_symbol
    import yaml
    presets = {'factors': {**FACTOR['factors'], 't': {'symbols': {1: 'circle', 2: 'circle-open'}}}}
    assert preset_factor_symbol(presets, ['r', 't'], '16 | 2 [y]') == 'circle-open'
    assert preset_factor_symbol(presets, ['r', 't'], '16') is None and preset_factor_symbol(presets, 't', '1') == 'circle'
    register_templates()
    (tmp_path / 'figure_presets.yaml').write_text(yaml.safe_dump(presets))
    df = pd.DataFrame({'x': range(8), 'y': range(8), 'r': [1, 1, 16, 16] * 2, 't': [1, 2] * 4})
    d = ScatterDashboard(df, ['r', 't'], renderer='svg', initial_views_dir=str(tmp_path))
    fig = d._reconstruct_figure({'x-axis': 'x', 'y-axis': 'y', 'color-by-factor': ['r', 't']}, theme_override=['light'])
    got = {t.name.rsplit(' [', 1)[0]: (t.marker.color, t.marker.symbol) for t in fig.data}
    assert got['16 | 2'] == (preset_factor_colors(presets, 'r')['16'], 'circle-open')
    assert got['1 | 1'] == (preset_factor_colors(presets, 'r')['1'], 'circle')


def test_factor_levels_export_with_their_labels_in_level_order():
    out = resolve_figure_options({'title': False}, FACTOR, 'x', ['y'], color_col='r')
    assert out['names'] == {'1': 'r = 1', '4': 'r = 4', '16': 'r = 16', '64': 'r = 64'}
    out = resolve_figure_options({'names': {'4': 'four', 'z': 'zed'}}, FACTOR, 'x', ['y'], color_col='r')
    assert list(out['names'].items())[:2] == [('1', 'r = 1'), ('4', 'four')] and out['names']['z'] == 'zed'
    assert 'names' not in resolve_figure_options({'title': False}, FACTOR, 'x', ['y'], color_col='other')


def test_text_ratios_pin_ticks_on_every_axis_and_the_legend():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from dashboards.utils import scale_figure_fonts
    fig = make_subplots(rows=1, cols=2)
    fig.add_scatter(x=[1], y=[1], row=1, col=1)
    fig.add_scatter(x=[1], y=[1], row=1, col=2)
    out = scale_figure_fonts(fig, 30, tick_scale=0.85, legend_scale=0.7)
    for axis in ('xaxis', 'yaxis', 'xaxis2', 'yaxis2'):
        assert out.layout[axis].tickfont.size == pytest.approx(25.5), axis
    assert out.layout.legend.font.size == pytest.approx(21)
    plain = scale_figure_fonts(fig, 30)            # no ratios: the furniture scale, as before
    assert plain.layout.xaxis.tickfont.size == pytest.approx(12 * (30 / 12) ** 0.6)


def test_the_export_route_takes_text_ratios_from_the_views_site(tmp_path, monkeypatch):
    import dash
    import plotly.graph_objects as go
    from dash import html
    from dashboards import utils
    views = tmp_path / 'scatter' / 'app'
    views.mkdir(parents=True)
    (views / 'v.json').write_text('{}')
    (tmp_path / 'figure_presets.yaml').write_text('text: {ticks: 0.85, axis_title: 0.9}\n')
    seen = []
    monkeypatch.setattr(utils, 'render_figure_image', lambda fig, w, h, fmt='png': seen.append(fig) or b'png')
    app = dash.Dash('text-test')
    app.layout = html.Div('x')
    utils.register_export_route(app, lambda v, w, h, t: go.Figure(go.Scatter(x=[1], y=[1])),
                                safe_roots=[str(tmp_path)], default_search_dirs=[str(views)],
                                title_scale=1.05, axis_title_scale=1.0)
    assert app.server.test_client().get('/?v&format=png&font=20').status_code == 200
    layout = seen[-1].layout
    assert layout.xaxis.tickfont.size == pytest.approx(17) and layout.xaxis.title.font.size == pytest.approx(18)
    assert layout.title.font.size == pytest.approx(21)   # not in the presets: the route's own default


def test_a_shift_click_line_offset_recolours_the_fitted_line_of_a_single_y(tmp_path):
    """With legend: per-series and a line swatch the legend entry is the fitted line, so a shift-click stores
    'name::line'; the single-Y build must honour it (it did only on the multi-Y overlay before)."""
    from dashboards.scatter import ScatterDashboard
    from dashboards.theme import register_templates
    register_templates()
    df = pd.DataFrame({'x': [1, 2, 3] * 2, 'y': range(6), 'g': ['a'] * 3 + ['b'] * 3})
    d = ScatterDashboard(df, ['g'], renderer='svg', initial_views_dir=str(tmp_path))

    def line_colours(offsets):
        view = {'x-axis': 'x', 'y-axis': 'y', 'color-by-factor': ['g'], 'overlay-line-toggle': ['on'],
                '_extras': {'color_offsets': offsets}}
        fig = d._reconstruct_figure(view, theme_override=['light'])
        return {t.name: t.line.color for t in fig.data if t.mode == 'lines'}

    before, after = line_colours({}), line_colours({'a::line': {'hsv': [0, 100, 100]}})
    assert after['a'] == '#ff0000' and after['b'] == before['b']


def test_point_details_are_a_tree_with_the_rows_values_and_collapsed_params():
    """A clicked reservoir shows its metrics (accuracy_dev too, not only the factors) and its params as a tree."""
    from dash import html
    from dashboards.scatter import details_tree
    tree = details_tree({'point': {'kq': 3}, 'values': {'accuracy_dev': 0.8125}, 'factors': {},
                         'params': {'M': {'R': {'k_avg': 2.0}}, 'seeds': [1, 2]}}, open_sections=('point', 'values'))
    assert [t.children[0].children for t in tree] == ['point', 'values', 'params']   # an empty section is left out
    assert [t.open for t in tree] == [True, True, False]
    assert tree[1].children[1].children == 'accuracy_dev: 0.8125'
    m = tree[2].children[1]
    assert isinstance(m, html.Details) and m.children[0].children == 'M'
    assert m.children[1].children[1].children == 'k_avg: 2.0'
