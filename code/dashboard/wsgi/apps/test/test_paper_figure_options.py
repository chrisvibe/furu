"""pytest tests for the figure options behind the hegg paper's remaining data figures: facets, legend
names from a column, shared legend entries, error bars, swapped axes, arrows and reference marks, through the scatter
build as a saved view reaches it (_reconstruct_figure), plus the export's font scaling of facet axes and legends.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # code/dashboard, as the other tests

from dashboards.scatter import ScatterDashboard  # noqa: E402
from dashboards.theme import register_templates  # noqa: E402
from dashboards.utils import (apply_figure_options, facet_cells, resolve_figure_options, scale_figure_fonts,  # noqa: E402
                              view_name_from_query)

register_templates()


def _view(x, y, color=None, figure=None, **fields):
    return {'x-axis': x, 'y-axis': y, 'color-by-factor': [color] if color else [], 'overlay-dots-toggle': ['on'],
            '_extras': {'figure': figure or {}}, **fields}


@pytest.fixture
def curves(tmp_path):
    """Fig. collapse_illustration's shape: per column (factor) and row (panel) one curve per level, a label per level,
    and a y that is empty where unsolved."""
    rows = []
    for factor in ('a', 'b'):
        for level in (0, 1):
            for panel in ('designed', 'shifted'):
                for k in range(1, 5):
                    nmse = 0.1 * k + level
                    rows.append(dict(factor=factor, series=f'level {level}', label=f'{factor}{level}: δ {level}',
                                     panel=panel, k=k + 0.5 * level * (panel == 'shifted'), nmse=nmse,
                                     nmse_solved=nmse if nmse < 1 else np.nan))
    df = pd.DataFrame(rows)
    return ScatterDashboard(df, ['factor', 'series', 'label', 'panel'], renderer='svg', initial_views_dir=str(tmp_path))


FACET = {'row': 'panel', 'col': 'factor', 'order': {'panel': ['designed', 'shifted']}, 'title': '{factor}',
         'title_on': 'top'}


def test_facets_put_each_panel_on_its_own_axes(curves):
    fig = curves._reconstruct_figure(_view('k', 'nmse', 'series', {'facet': FACET}), theme_override=['light'])
    cells = facet_cells(fig)
    assert [(c['rv'], c['cv']) for c in cells] == [('designed', 'a'), ('designed', 'b'), ('shifted', 'a'), ('shifted', 'b')]
    for c in cells:   # two levels per panel, each on the panel's axes
        assert len([t for t in fig.data if (t.xaxis, t.yaxis) == (c['x'], c['y'])]) == 2
    assert [a.text for a in fig.layout.annotations] == ['a', 'b']   # titles on the top row only
    assert fig.layout.xaxis2.matches == 'x' and fig.layout.yaxis4.matches == 'y'


def test_a_series_in_several_panels_has_one_legend_entry(curves):
    fig = curves._reconstruct_figure(_view('k', 'nmse', 'series', {'facet': FACET}), theme_override=['light'])
    shown = [t.name for t in fig.data if t.showlegend is not False]
    assert sorted(shown) == ['level 0', 'level 1']
    groups = {t.name: {u.legendgroup for u in fig.data if u.name == t.name} for t in fig.data}
    assert all(len(g) == 1 for g in groups.values())   # the panels' copies toggle together


def test_axis_titles_by_facet_row_and_first_column(curves):
    figure = {'facet': FACET, 'x_titles': {'designed': 'k', 'shifted': 'k_eff'}, 'y_title': 'NMSE'}
    fig = curves._reconstruct_figure(_view('k', 'nmse', 'series', figure), theme_override=['light'])
    titles = {(c['row'], c['col']): (fig.layout[f'xaxis{c["x"][1:]}'].title.text, fig.layout[f'yaxis{c["y"][1:]}'].title.text)
              for c in facet_cells(fig)}
    assert titles == {(0, 0): ('k', 'NMSE'), (0, 1): ('k', ''), (1, 0): ('k_eff', 'NMSE'), (1, 1): ('k_eff', '')}


def test_names_from_a_column_and_one_entry_per_column_legend(curves):
    """Two y columns (solved: solid, all: dotted) named from the label column: one entry per level, in the legend of
    its facet column, and the generated name kept for saved colours and visibility."""
    figure = {'facet': FACET, 'names_from': 'label', 'facet_legend': 'bottom right', 'trace_mode': 'lines+markers'}
    fig = curves._reconstruct_figure(_view('k', ['nmse_solved', 'nmse'], 'series', figure), theme_override=['light'])
    shown = sorted((t.legend or 'legend', t.name) for t in fig.data if t.showlegend is not False)
    assert shown == [('legend', 'a0: δ 0'), ('legend', 'a1: δ 1'), ('legend2', 'b0: δ 0'), ('legend2', 'b1: δ 1')]
    t = next(t for t in fig.data if t.name == 'a1: δ 1')
    assert t.meta['name0'] in ('level 1 [nmse_solved]', 'level 1 [nmse]') and t.meta['mode0'] == 'markers'
    assert fig.layout.legend2.xanchor == 'right' and fig.layout.legend2.yanchor == 'bottom'


def test_a_group_name_names_every_y_of_the_group():
    fig = go.Figure()
    for name in ('1 [frequency]', '1 [binomial]', '2 [frequency]'):
        fig.add_scatter(x=[0, 1], y=[0, 1], mode='markers', name=name)
    apply_figure_options(fig, {'names': {'1': 'k = 1', '2': 'k = 2'}})
    assert [t.name for t in fig.data] == ['k = 1', 'k = 1', 'k = 2']
    assert [t.showlegend for t in fig.data] == [None, False, None]


def test_error_bars_then_swapped_axes():
    """Fig. optimum_shift: measured shift with its CI as error bars, categories on the plotted x, then transposed."""
    df = pd.DataFrame({'row': [0, 1], 'measured': [0.5, -0.2], 'lo': [0.3, -0.5], 'hi': [0.6, 0.1], 'task': 'a'})
    d = ScatterDashboard(df, ['task'], renderer='svg')
    figure = {'error_bars': [{'lo': 'lo', 'hi': 'hi'}], 'swap_axes': True, 'x_title': 'shift', 'y_title': '',
              'vlines': [{'x': 0}]}
    fig = d._reconstruct_figure(_view('row', 'measured', figure=figure), theme_override=['light'])
    t = fig.data[0]
    assert list(t.x) == [0.5, -0.2] and list(t.y) == [0, 1]
    assert np.allclose(t.error_x.array, [0.1, 0.3]) and np.allclose(t.error_x.arrayminus, [0.2, 0.3])
    assert fig.layout.xaxis.title.text == 'shift' and fig.layout.yaxis.title.text == ''
    assert [(s.x0, s.yref) for s in fig.layout.shapes] == [(0, 'y domain')]


def test_rings_circle_the_matching_points_once_in_the_legend():
    """furu Fig 3: the perfect reservoirs (accuracy 1) ringed among the rest, one legend entry for both panels."""
    df = pd.DataFrame({'kq': [1, 2, 3, 4], 'delta': [0, 1, 2, 3], 'acc': [0.6, 1.0, 0.7, 1.0],
                       'head': ['a', 'a', 'b', 'b']})
    d = ScatterDashboard(df, ['head'], renderer='svg')
    figure = {'rings': [{'where': 'acc >= 0.9995', 'label': 'accuracy 1'}]}
    fig = d._reconstruct_figure(_view('kq', 'delta', 'head', figure=figure), theme_override=['light'])
    rings = [t for t in fig.data if t.name == 'accuracy 1']
    assert sorted(x for t in rings for x in t.x) == [2, 4]
    assert [t.marker.symbol for t in rings] == ['circle-open'] * 2
    assert [t.showlegend for t in rings] == [True, False]


def test_rings_around_webgl_dots_are_webgl_too():
    """An svg ring sits under every WebGL trace (furu Fig 4's png showed none), so a ring takes its dots' type."""
    df = pd.DataFrame({'kq': [1, 2, 3, 4], 'delta': [0, 1, 2, 3], 'acc': [0.6, 1.0, 0.7, 1.0], 'head': 'a'})
    d = ScatterDashboard(df, ['head'], renderer='webgl')
    fig = d._reconstruct_figure(_view('kq', 'delta', figure={'rings': [{'where': 'acc >= 0.9995', 'label': 'acc 1'}]}),
                                theme_override=['light'])
    assert {t.type for t in fig.data} == {'scattergl'}


def test_x_dodge_spaces_the_levels_evenly_and_sets_the_shown_traces_apart():
    """furu Fig 3: r levels 1 .. 512 evenly spaced, the heads side by side so their quartile bars don't overlap."""
    df = pd.DataFrame({'r': [1, 4, 512] * 3, 'k': range(9), 'head': ['a'] * 3 + ['b'] * 3 + ['c'] * 3})
    d = ScatterDashboard(df, ['head'], renderer='svg')
    view = _view('r', 'k', 'head', figure={'x_dodge': 0.1})
    view['_extras']['trace_visibility'] = {'c|markers': 'legendonly'}
    fig = d._reconstruct_figure(view, theme_override=['light'])
    xs = {t.name: list(t.x) for t in fig.data}
    assert np.allclose(xs['a'], [-0.05, 0.95, 1.95]) and np.allclose(xs['b'], [0.05, 1.05, 2.05])
    assert list(fig.layout.xaxis.tickvals) == [0, 1, 2] and list(fig.layout.xaxis.ticktext) == ['1', '4', '512']


def test_bin2d_draws_the_median_per_bin_in_place_of_the_dots():
    """furu Fig 4b: 90k reservoirs as the median accuracy per (KQ, Δ) bin; a thin bin is blank."""
    df = pd.DataFrame({'kq': [0, 1, 2, 10, 11], 'delta': [0, 0, 1, 0, 0], 'acc': [0.5, 0.7, 0.9, 0.6, 0.6], 'head': 'a'})
    d = ScatterDashboard(df, ['head'], renderer='svg')
    figure = {'bin2d': {'x': 'kq', 'y': 'delta', 'z': 'acc', 'dx': 10, 'dy': 10, 'min_n': 3}}
    fig = d._reconstruct_figure(_view('kq', 'delta', 'acc', figure=figure), theme_override=['light'])
    heat = [t for t in fig.data if t.type == 'heatmap']
    assert len(heat) == 1 and all(t.visible is False for t in fig.data if t.type != 'heatmap')
    assert list(heat[0].x) == [5, 15] and np.allclose(heat[0].z[0][0], 0.7) and np.isnan(heat[0].z[0][1])
    titles = {'axis_titles': {'acc': 'dev accuracy'}}
    assert resolve_figure_options(figure, titles, 'kq', ['delta'], 'acc')['color_title'] == 'median dev accuracy'


def test_the_heatmap_control_bins_the_axes_by_the_colour_column():
    """The GUI's Heatmap checkbox and bin sizes, as a saved view holds them, give the bin2d heatmap; off, or a
    colour that is not continuous, leaves the dots."""
    df = pd.DataFrame({'kq': [0, 1, 2, 10, 11], 'delta': [0, 0, 1, 0, 0], 'acc': [0.5, 0.7, 0.9, 0.6, 0.6], 'head': 'a'})
    d = ScatterDashboard(df, ['head'], renderer='svg')
    on = {'heatmap-toggle': ['on'], 'heatmap-dx': 10, 'heatmap-dy': 10, 'heatmap-min-n': 3}
    fig = d._reconstruct_figure(_view('kq', 'delta', 'acc', **on), theme_override=['light'])
    heat = [t for t in fig.data if t.type == 'heatmap']
    assert len(heat) == 1 and list(heat[0].x) == [5, 15] and np.allclose(heat[0].z[0][0], 0.7)
    for view in (_view('kq', 'delta', 'acc'), _view('kq', 'delta', 'head', **on)):
        assert not any(t.type == 'heatmap' for t in d._reconstruct_figure(view, theme_override=['light']).data)


def test_colorbar_props_move_the_colour_scale_into_a_corner():
    """furu Fig 4b: a short horizontal colour bar in a panel's empty corner, so the panels keep the full width."""
    df = pd.DataFrame({'kq': [0, 1, 2], 'delta': [0, 1, 2], 'acc': [0.5, 0.7, 0.9], 'head': 'a'})
    d = ScatterDashboard(df, ['head'], renderer='svg')
    figure = {'colorbar': {'orientation': 'h', 'x': 1, 'xanchor': 'right', 'len': 0.3}}
    fig = d._reconstruct_figure(_view('kq', 'delta', 'acc', figure=figure, **{'color-as-continuous': ['on']}),
                                theme_override=['light'])
    bar = fig.layout.coloraxis.colorbar
    assert (bar.orientation, bar.x, bar.xanchor, bar.len) == ('h', 1, 'right', 0.3)


def test_a_series_takes_its_preset_dash(tmp_path):
    """furu heads: colour = task, dash = delay, from the site presets; a series without one keeps the default."""
    (tmp_path / 'figure_presets.yaml').write_text('series:\n  a_d10: {color: 1, dash: dot}\n  a_d5: {color: 1}\n')
    df = pd.DataFrame({'r': [1, 2, 1, 2], 'k': [3, 2, 4, 3], 'head': ['a_d5', 'a_d5', 'a_d10', 'a_d10']})
    d = ScatterDashboard(df, ['head'], renderer='svg', initial_views_dir=str(tmp_path))
    fig = d._reconstruct_figure(_view('r', 'k', 'head', {'trace_mode': 'lines+markers'}), theme_override=['light'])
    dash = {t.name: t.line.dash for t in fig.data}
    assert dash['a_d10'] == 'dot' and dash['a_d5'] in (None, 'solid')   # None: plotly's solid
    assert fig.data[0].line.color == fig.data[1].line.color


def test_arrows_from_columns_and_numbers():
    df = pd.DataFrame({'x': [0.0, 0.2, 0.1], 'step': [0, 1, 2], 'x_prev': [np.nan, 0.0, 0.2],
                       'step_prev': [np.nan, 0, 1], 'vx': [0.0, 0.3, -0.2], 'walk': 0})
    d = ScatterDashboard(df, ['walk'], renderer='svg')
    figure = {'arrows': {'x0': 'x_prev', 'y0': 'step_prev', 'dx': 'vx', 'dy': 1}}
    fig = d._reconstruct_figure(_view('x', 'step', figure=figure), theme_override=['light'])
    arrows = [(a.ax, a.ay, a.x, a.y) for a in fig.layout.annotations]
    assert np.allclose(arrows, [(0.0, 0, 0.3, 1), (0.2, 1, 0.0, 2)])   # step 0 has no input


def test_reference_marks_in_every_panel(curves):
    figure = {'facet': FACET, 'hbands': [{'y0': 1, 'y1': 2}], 'dlines': [{}], 'axes': {'x': {'range': [0, 5]}}}
    fig = curves._reconstruct_figure(_view('k', 'nmse', 'series', figure), theme_override=['light'])
    assert sorted(s.yref for s in fig.layout.shapes) == ['y', 'y2', 'y3', 'y4']
    lines = [t for t in fig.data if t.hoverinfo == 'skip']
    assert len(lines) == 4 and all(list(t.x) == [0, 5] for t in lines)   # the identity over the set x range


def test_export_fonts_reach_facet_axes_and_column_legends():
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2)
    fig.update_layout(legend2={'x': 0.5})
    out = scale_figure_fonts(fig, 40, tick_scale=0.8, legend_scale=0.7).layout
    assert out.xaxis2.tickfont.size == out.xaxis.tickfont.size == 32
    assert out.legend2.font.size == out.legend.font.size == 28


def test_every_view_control_has_a_layout_default(curves):
    """A view need only hold what it changes: every control it can hold starts with a value in the layout, which
    loading the view (live) and the export both use for a key the view leaves out."""
    from dashboards.scatter_layout import VIEW_FIELDS
    assert set(curves._view_defaults) == {fid for fid, _ in VIEW_FIELDS}
    minimal = {'x-axis': 'k', 'y-axis': 'nmse', '_extras': {}}
    full = {**curves._view_defaults, **minimal}
    a = curves._reconstruct_figure(minimal, theme_override=['light'])
    b = curves._reconstruct_figure(full, theme_override=['light'])
    assert a.to_json() == b.to_json() and a.data[0].marker.size == 5


def test_facet_titles_take_the_axis_title_size_in_the_export(curves):
    """A facet title is an axis-level label: the export's ?axislabel= sets it (not the raw ?font=), lifted by half the
    y tick label (whose top half stands above the panel), and a figure with no title gets a top margin that just
    holds the top row's facet titles."""
    fig = curves._reconstruct_figure(_view('k', 'nmse', 'series', {'facet': FACET, 'title': False}),
                                     theme_override=['light'])
    out = scale_figure_fonts(fig, 116, axislabel_px=47).layout
    titles = [a for a in out.annotations if a.name == 'facet_title']
    assert [a.font.size for a in titles] == [47, 47]
    lift = round(0.5 * out.yaxis.tickfont.size)
    assert [a.yshift for a in titles] == [lift, lift]
    assert out.xaxis.title.font.size == 47
    assert 47 * 1.4 + lift < out.margin.t < 47 * 1.4 + lift + 10


def test_a_figure_without_title_or_facet_titles_keeps_a_small_top_margin(curves):
    fig = curves._reconstruct_figure(_view('k', 'nmse', 'series', {'title': False}), theme_override=['light'])
    assert fig.layout.margin.t < 20
    tick = scale_figure_fonts(fig, 116).layout.yaxis.tickfont.size
    assert scale_figure_fonts(fig, 116).layout.margin.t == round(0.7 * tick)


def test_inner_rows_show_x_tick_labels_only_with_an_x_title_of_their_own(curves):
    fig = curves._reconstruct_figure(_view('k', 'nmse', 'series', {'facet': FACET}), theme_override=['light'])
    shown = {c['row']: fig.layout[f'xaxis{c["x"][1:]}'].showticklabels for c in facet_cells(fig)}
    assert shown == {0: False, 1: None}
    figure = {'facet': FACET, 'x_titles': {'designed': 'k', 'shifted': 'k_eff'}}
    fig = curves._reconstruct_figure(_view('k', 'nmse', 'series', figure), theme_override=['light'])
    assert all(fig.layout[f'xaxis{c["x"][1:]}'].showticklabels is not False for c in facet_cells(fig))


def test_column_legends_below_the_grid(curves):
    figure = {'facet': FACET, 'facet_legend': 'below'}
    fig = curves._reconstruct_figure(_view('k', 'nmse', 'series', figure), theme_override=['light'])
    for legend, col in (('legend', 0), ('legend2', 1)):
        lg = fig.layout[legend]
        x = next(c['x'] for c in facet_cells(fig) if c['col'] == col)
        assert (lg.yref, lg.y, lg.yanchor, lg.xanchor) == ('container', 0, 'bottom', 'left')
        assert lg.x == fig.layout[f'xaxis{x[1:]}'].domain[0]


def test_column_legends_below_the_grid_hang_from_one_line_in_the_export(curves):
    """At the export's size each column's legend hangs from the top of the tallest one, so a short legend starts
    right under the x titles too (bottom-anchored, it would float below them)."""
    figure = {'facet': FACET, 'facet_legend': 'below', 'names_from': 'label'}
    fig = curves._reconstruct_figure(_view('k', 'nmse', 'series', figure), theme_override=['light'])
    fig.data[0].name += '<br>a second line'   # column 0's legend, one line taller
    out = scale_figure_fonts(fig, 116, height=1500).layout
    tops = {(out[k].y, out[k].yanchor) for k in ('legend', 'legend2')}
    assert len(tops) == 1 and tops.pop()[1] == 'top'
    assert 0 < out.legend.y < 0.5


def test_an_exported_x_title_at_the_foot_gets_room_for_its_descenders(curves):
    fig = curves._reconstruct_figure(_view('k', 'nmse', 'series', {'x_title': 'position'}), theme_override=['light'])
    out = scale_figure_fonts(fig, 116, axislabel_px=47, height=700).layout
    assert out.margin.b >= round(1.25 * out.xaxis.tickfont.size + 1.6 * 47)


def test_one_y_title_for_the_grid(curves):
    """y_title_span: the grid's rows share one y title, centred on its height, left of the first column's ticks."""
    figure = {'facet': FACET, 'y_title': 'share', 'y_title_span': True, 'axes': {'y': {'range': [0, 1], 'dtick': 0.2}}}
    fig = curves._reconstruct_figure(_view('k', 'nmse', 'series', figure), theme_override=['light'])
    assert not any(fig.layout[f'yaxis{c["y"][1:]}'].title.text for c in facet_cells(fig))
    (title,) = [a for a in fig.layout.annotations if a.name == 'grid_y_title']
    assert (title.text, title.textangle, title.y) == ('share', -90, 0.5)
    out = scale_figure_fonts(fig, 116, axislabel_px=47).layout
    (title,) = [a for a in out.annotations if a.name == 'grid_y_title']
    labels = 3 * 0.56 * out.yaxis.tickfont.size     # '0.2' ... '0.8'
    assert title.font.size == 47 and title.xshift < -labels and out.margin.l > labels + 47


@pytest.mark.parametrize('query, name', [('pi_x&format=png', 'pi_x'), ('?pi%5Fx&width=9', 'pi_x'),
                                         ('view=pi_x&format=png', None), ('format=png', None)])
def test_a_url_names_its_view_by_a_bare_key_only(query, name):
    assert view_name_from_query(query) == name
    assert view_name_from_query('expand&pi_y', skip={'expand'}) == 'pi_y'
