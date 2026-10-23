"""pytest tests for the `?font=` export font override (`dashboards.utils`).

Tests `scale_figure_fonts` (the recursive walker + the furniture cap +
automargin-on-export), the polar `_table_fit` `max_scale` cap contract, and the
export route's `?font=` parsing / dispatch / validation.


Section 4 covers the title/axis-title pinning: plotly renders an inherited figure title at ~1.4x
the base font and inherited axis titles at ~1.2x; `scale_figure_fonts` can now
pin explicit sizes at caller-chosen multiples (scatter family: caption size)
and the route forwards the ratios plus per-request ?title=/?axislabel= px
overrides. Callers passing none of the new args (polar, trace) are unchanged.

Note on `dashboards.polar`: it is NOT importable here — importing it pulls
`wsgi.base -> project -> juliacall`, which needs a writable `julia_env` dir (fails
with PermissionError on julia_env/lock.pid in the sandbox and in read-only
deployments). The pure table-fit functions are therefore exec'd from a source
slice of polar.py (same approach the original harness used), and the polar
`font_px` contract is checked via AST instead of import.
"""
import ast
import re
import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest
import pandas as pd

# dashboards.* live under code/dashboard (mirrors run.py / test_polar_highlight.py).
# parents[3] == <repo>/code/dashboard.
_DASHBOARD_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_DASHBOARD_DIR))

from dashboards import utils  # noqa: E402


# ── 1. scale_figure_fonts: the walker ─────────────────────────────────────────

def _base_fig():
    """A figure exercising every font kind the walker must scale (and nothing else)."""
    return go.Figure(
        data=[
            go.Scatter(y=[1, 2], marker=dict(size=6, colorbar=dict(tickfont=dict(size=10))),
                       textfont=dict(size=9), hoverlabel=dict(font=dict(size=8))),
            go.Table(header=dict(font=dict(size=10), height=28),
                     cells=dict(font=dict(size=10), height=24),
                     columnwidth=[40, 60]),
        ],
        layout=dict(
            title=dict(text='T', font=dict(size=14)),
            xaxis=dict(tickfont=dict(size=10), title=dict(font=dict(size=11))),
            yaxis=dict(tickfont=dict(size=10)),
            legend=dict(font=dict(size=12), title=dict(font=dict(size=12))),
            annotations=[dict(text='a', font=dict(size=11))],
            font=dict(color='red'),          # color only — no size (inherits 12)
            template='plotly_dark',
            margin=dict(l=80, r=80),
            width=1200, height=800,
        ),
    )


def _kf(k):
    """Furniture scale factor: 12 -> data grows as k, furniture as k^0.6."""
    return k ** 0.6


def test_input_figure_untouched():
    fig = _base_fig()
    orig = fig.to_plotly_json()
    utils.scale_figure_fonts(fig, 36)  # k = 3
    assert fig.to_plotly_json() == orig


def test_walker_scales_every_font_kind_at_k3():
    j = utils.scale_figure_fonts(_base_fig(), 36).to_plotly_json()  # k = 3
    approx = lambda v, t: abs(v - t) < 1e-6
    kf = _kf(3)
    # base font
    assert j['layout']['font']['size'] == 36
    assert j['layout']['font']['color'] == 'red'                 # color kept
    # data text: full k
    assert j['layout']['title']['font']['size'] == 42            # 14*3
    assert j['layout']['annotations'][0]['font']['size'] == 33   # 11*3
    assert j['data'][0]['textfont']['size'] == 27                # 9*3
    assert j['data'][0]['hoverlabel']['font']['size'] == 24      # 8*3
    assert j['data'][0]['marker']['colorbar']['tickfont']['size'] == 30  # 10*3
    assert j['data'][1]['header']['font']['size'] == 30          # 10*3
    assert j['data'][1]['cells']['font']['size'] == 30           # 10*3
    # furniture: k^0.6 (ticks, legends, axis-subtree titles)
    assert approx(j['layout']['xaxis']['tickfont']['size'], 10 * kf)
    assert approx(j['layout']['xaxis']['title']['font']['size'], 11 * kf)
    assert approx(j['layout']['legend']['font']['size'], 12 * kf)
    assert approx(j['layout']['legend']['title']['font']['size'], 12 * kf)


def test_non_font_properties_preserved_at_k3():
    j = utils.scale_figure_fonts(_base_fig(), 36).to_plotly_json()
    assert j['data'][0]['marker']['size'] == 6                   # markers untouched
    assert j['data'][1]['columnwidth'] == [40, 60]               # colwidth untouched
    assert j['layout']['margin'] == {'l': 80, 'r': 80}           # margins untouched
    assert (j['layout']['width'], j['layout']['height']) == (1200, 800)
    # template preserved.  plotly serializes a named template by EXPANDING it
    # to its full dict and dropping the name — the raw input's own
    # to_plotly_json() already carries the expansion, no 'plotly_dark' string —
    # so compare against the input's own serialization, not the name.
    templ = j['layout']['template']
    assert templ == _base_fig().to_plotly_json()['layout']['template']
    assert 'rgb(17,17,17)' in repr(templ)        # plotly_dark bg survived


def test_automargin_on_for_override():
    j = utils.scale_figure_fonts(_base_fig(), 36).to_plotly_json()
    assert j['layout']['title']['automargin'] is True
    assert j['layout']['xaxis']['automargin'] is True
    assert j['layout']['yaxis']['automargin'] is True


def test_k1_leaves_explicit_sizes_unchanged():
    j12 = utils.scale_figure_fonts(_base_fig(), 12).to_plotly_json()
    assert [j12['layout']['title']['font']['size'],
            j12['layout']['xaxis']['tickfont']['size'],
            j12['data'][1]['header']['font']['size']] == [14, 10, 10]
    # automargin still on for any ?font= request, even k=1
    assert j12['layout']['xaxis']['automargin'] is True


def test_inherited_furniture_pinned_not_scaled_base():
    # no explicit tickfont / legend font: furniture must be pinned to 12*k^0.6,
    # not silently inherit the scaled base font.
    fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[1, 2], name='s')],
                    layout=dict(width=800, height=600))
    j = utils.scale_figure_fonts(fig, 36).to_plotly_json()
    assert j['layout']['font']['size'] == 36                     # injected base
    assert _kf(3) * 12 - j['layout']['legend']['font']['size'] < 1e-6
    assert _kf(3) * 12 - j['layout']['xaxis']['tickfont']['size'] < 1e-6


# ── 2. polar._table_fit cap semantics (source-extracted, see module docstring) ─

@pytest.fixture(scope='module')
def polar_table():
    """Source-extract polar's pure table-fit funcs + constants into a namespace."""
    polar_src = (_DASHBOARD_DIR / 'dashboards' / 'polar.py').read_text()
    tree = ast.parse(polar_src)

    def fn_src(name):
        # walk, not tree.body: polar's reconstruct fns are nested inside the
        # app factory since the polar refactor, so a top-level scan misses them
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef,)) and node.name == name:
                return ast.get_source_segment(polar_src, node)
        raise AssertionError(f'{name} not found in polar.py')

    ns = {'pd': pd}
    consts = dict(re.findall(r'^(_TBL_\w+)\s*=\s*([0-9.]+)', polar_src, re.M))
    ns.update({k: float(v) for k, v in consts.items()})
    for name in ('_centred_span', '_table_natural_widths', '_table_fit'):
        exec(fn_src(name), ns)
    rec_sig = fn_src('_reconstruct_table').split('\n')[0]
    return dict(_table_fit=ns['_table_fit'], _TBL_FONT_PX=ns['_TBL_FONT_PX'],
                _reconstruct_sig=rec_sig, _reconstruct_src=fn_src('_reconstruct_table'))


def test_polar_reconstruct_declares_font_px(polar_table):
    # the exact mechanism register_export_route uses to detect font-aware fns
    assert 'font_px=None' in polar_table['_reconstruct_sig']
    assert 'font_px / _TBL_FONT_PX' in polar_table['_reconstruct_src']


def test_table_cap_binds_scale(polar_table):
    _fit, _px = polar_table['_table_fit'], polar_table['_TBL_FONT_PX']
    df = pd.DataFrame({'param': ['k.avg', 'R.seed'], 'value': ['0.5', '1']})
    _, s_free, _ = _fit(df, 1000, 820)
    _, s_cap, _ = _fit(df, 1000, 820, max_scale=1.0)
    assert s_free > 1                      # free fit enlarges (fills frame)
    assert abs(s_cap - 1.0) < 1e-12        # cap binds scale to 1.0
    assert abs(_px * s_cap - _px) < 1e-9   # font at cap == _TBL_FONT_PX


def test_table_cap_keeps_domain_centred(polar_table):
    _fit = polar_table['_table_fit']
    df = pd.DataFrame({'param': ['k.avg', 'R.seed'], 'value': ['0.5', '1']})
    _, _, dom = _fit(df, 1000, 820, max_scale=1.0)
    assert abs(dom['x'][0] + dom['x'][1] - 1) < 1e-9
    assert dom['x'][1] < 1


def test_table_cap_ignored_when_fit_smaller(polar_table):
    _fit = polar_table['_table_fit']
    df = pd.DataFrame({'param': ['k.avg', 'R.seed'], 'value': ['0.5', '1']})
    _, s_small, _ = _fit(df, 300, 200)
    _, s_smallcap, _ = _fit(df, 300, 200, max_scale=5.0)
    assert abs(s_small - s_smallcap) < 1e-12


# ── 3. the export route: param parsing + dispatch + validation ────────────────

def _mk_view(tmp_path):
    # the route resolves <view_name>.json under the search dirs; the tests
    # request ?_font_test_view, so the file must carry that stem.
    view = tmp_path / '_font_test_view.json'
    view.write_text('{"anything": true}', encoding='utf-8')
    return view


@pytest.fixture()
def route_apps(tmp_path, monkeypatch):
    """Build a plain-fn app and an aware-fn app; capture rendered figures.

    Returns (plain_app_client, aware_app_client, captured).
    """
    import json as _json
    import dash
    from dash import html

    captured = {}

    def fake_render(fig, width, height, fmt='png'):
        captured['fig'] = fig
        return b'IMG'

    monkeypatch.setattr(utils, 'render_figure_image', fake_render)

    plain_fig = go.Figure(layout=dict(font=dict(size=12), width=10, height=10))

    def plain_fn(view_data, width, height, theme_override):
        return plain_fig

    def aware_fn(view_data, width, height, theme_override, font_px=None):
        return ('AWARE_FIG', font_px)

    view = _mk_view(tmp_path)
    d = str(view.parent)

    app = dash.Dash('font-test-plain')
    app.layout = html.Div('x')
    utils.register_export_route(app, plain_fn, safe_roots=[str(tmp_path)],
                                default_search_dirs=[d])

    # separate app: a Flask before_request hook that returns a response
    # short-circuits later hooks, so the two fns must not share one app.
    app2 = dash.Dash('font-test-aware')
    app2.layout = html.Div('x')
    utils.register_export_route(app2, {'polar': plain_fn, 'table': aware_fn},
                                safe_roots=[str(tmp_path)],
                                default_search_dirs=[d])

    return app.server.test_client(), app2.server.test_client(), captured, plain_fig


def test_no_font_param_passes_original_fig(route_apps):
    c, _c2, captured, plain_fig = route_apps
    r = c.get('/?_font_test_view&format=png&width=10&height=10')
    assert r.status_code == 200
    assert captured['fig'] is plain_fig          # same object, not rebuilt


def test_font_param_scales_plain_fn_fig(route_apps):
    c, _c2, captured, plain_fig = route_apps
    r = c.get('/?_font_test_view&format=png&width=10&height=10&font=29')
    assert r.status_code == 200
    cap = captured['fig']
    assert cap is not plain_fig                  # rebuilt
    assert cap.layout.font.size == 29            # base scaled to 29
    assert plain_fig.layout.font.size == 12      # original untouched


def test_aware_fn_receives_font_px(route_apps):
    c, c2, captured, _plain = route_apps
    r = c2.get('/?_font_test_view&format=png&width=10&height=10&font=29&target=table')
    assert r.status_code == 200
    assert captured['fig'] == ('AWARE_FIG', 29)
    r = c2.get('/?_font_test_view&format=png&width=10&height=10&target=table')
    assert r.status_code == 200
    assert captured['fig'] == ('AWARE_FIG', None)


def test_font_validation_400(route_apps):
    c, _c2, _captured, _plain = route_apps
    assert c.get('/?_font_test_view&format=png&font=abc').status_code == 400
    assert c.get('/?_font_test_view&format=png&font=5').status_code == 400
    assert c.get('/?_font_test_view&format=png&font=200').status_code == 400


def test_interactive_page_untouched_by_font(route_apps):
    c, _c2, _captured, _plain = route_apps
    r = c.get('/')
    assert r.status_code == 200
    assert b'IMG' not in r.data