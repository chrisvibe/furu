from pathlib import Path

from dash import dcc, html

from .utils import _view_dropdown_options, _AGG_OPTIONS_WITH_RAW, admin_token_store

# ── Style constants ───────────────────────────────────────────────────────────
# Every colour here is a CSS custom property resolved in the browser, so these
# style dicts are theme-agnostic — the token values live in dashboards/theme.py.
from .theme import (
    SURFACE as _SURFACE, PANEL as _PANEL, CONTROL as _CONTROL,
    BORDER as _BORDER, TEXT as _TEXT, TEXT_DIM as _TEXT_DIM,
    ACCENT as _ACCENT, OK as _OK, DELETE as _DELETE, DANGER as _DANGER,
    ON_SOLID as _ON_SOLID, make_theme_toggle, theme_pref_store,
)

input_style = {
    'width': '60px', 'display': 'inline-block', 'fontSize': '11px',
    'padding': '2px 4px', 'marginLeft': '4px',
    'background': _PANEL, 'color': _TEXT, 'border': f'1px solid {_BORDER}', 'borderRadius': '3px',
}
label_style = {'fontSize': '11px', 'color': _TEXT_DIM, 'marginLeft': '4px'}

_collapse_btn_style = {
    'padding': '4px 10px', 'fontSize': '12px',
    'cursor': 'pointer', 'border': f'1px solid {_BORDER}',
    'borderRadius': '4px', 'background': _CONTROL, 'color': _TEXT,
}

btn_sm = {
    'fontSize': '10px', 'padding': '2px 6px', 'cursor': 'pointer',
    'background': _CONTROL, 'color': _TEXT, 'border': f'1px solid {_BORDER}', 'borderRadius': '3px',
}

_COLOR_SCALE_OPTIONS = [
    {'label': 'Viridis', 'value': 'Viridis'},
    {'label': 'Plasma',  'value': 'Plasma'},
    {'label': 'RdBu',    'value': 'RdBu'},
    {'label': 'Hot',     'value': 'Hot'},
    {'label': 'Turbo',   'value': 'Turbo'},
]

SELECTION_VALUE = '_selection_'

# The controls a saved view holds, (id, property): save_view writes them, loading a view sets them
VIEW_FIELDS = [
    ('x-axis',                  'value'),
    ('y-axis',                  'value'),
    ('x-jitter-toggle',         'value'),
    ('x-jitter-min',            'value'),
    ('x-jitter-max',            'value'),
    ('x-bin-width',             'value'),
    ('y-jitter-toggle',         'value'),
    ('y-jitter-min',            'value'),
    ('y-jitter-max',            'value'),
    ('color-by-factor',         'value'),
    ('color-by-values',         'value'),
    ('color-scale-picker',      'value'),
    ('color-scale-reverse',     'value'),
    ('agg-group-by',            'value'),
    ('agg-y-mode',              'value'),
    ('filter-expr',             'value'),
    ('fields-expr',             'value'),
    ('setting-display-points',  'value'),
    ('setting-discrete-threshold', 'value'),
    ('setting-marker-size',     'value'),
    ('setting-marker-opacity',  'value'),
    ('color-as-continuous',     'value'),
    ('heatmap-toggle',          'value'),
    ('heatmap-dx',              'value'),
    ('heatmap-dy',              'value'),
    ('heatmap-min-n',           'value'),
    ('legend-show-factor-name', 'value'),
    ('skip-sparse-factors',    'value'),
    ('overlay-dots-toggle',  'value'),
    ('overlay-line-toggle',  'value'),
    ('overlay-line-width',   'value'),
    ('overlay-line-opacity', 'value'),
    ('overlay-spline-k',          'value'),
    ('overlay-spline-s',          'value'),
    ('overlay-per-line-toggle',   'value'),
    ('overlay-spline-ks',         'data'),
    ('overlay-anchor-left',       'value'),
    ('overlay-anchor-right',      'value'),
    ('overlay-y-styles',          'data'),
    ('theme-toggle',         'value'),
]


def control_defaults(layout, fields=VIEW_FIELDS) -> dict:
    """{id: value} each control of fields starts with in layout: what a saved view that leaves it out gets, so a
    view need only hold what it changes."""
    wanted, out = dict(fields), {}

    def walk(c):
        cid = getattr(c, 'id', None)
        if isinstance(cid, str) and cid in wanted:
            out[cid] = getattr(c, wanted[cid], None)
        children = getattr(c, 'children', None)
        for child in children if isinstance(children, (list, tuple)) else [children]:
            if hasattr(child, 'to_plotly_json'):
                walk(child)
    walk(layout)
    return out


# ── Layout component builders ─────────────────────────────────────────────────

def make_collapsible_panel(panel_id, label, children, start_open=True):
    arrow_open, arrow_closed = '▼', '▶'
    return html.Div([
        html.Button(
            f'{arrow_open if start_open else arrow_closed} {label}',
            id=f'{panel_id}-toggle', n_clicks=0, style=_collapse_btn_style,
        ),
        html.Div(children, id=f'{panel_id}-body',
                 style={'display': 'block' if start_open else 'none'}),
    ], style={'marginBottom': '10px'})


def make_jitter_controls(axis):
    jitter_row = html.Div([
        dcc.Checklist(
            id=f'{axis}-jitter-toggle',
            options=[{'label': f' Jitter {axis.upper()}', 'value': 'on'}],
            value=[],
            style={'display': 'inline-block', 'fontSize': '11px'},
            inputStyle={'marginRight': '3px'}, labelStyle={'color': _TEXT_DIM},
        ),
        html.Span('min:', style=label_style),
        dcc.Input(id=f'{axis}-jitter-min', type='number', value=-0.5, step='any', style=input_style),
        html.Span('max:', style=label_style),
        dcc.Input(id=f'{axis}-jitter-max', type='number', value=0.5,  step='any', style=input_style),
    ])
    return html.Div([jitter_row], style={'marginTop': '4px'})


def build_layout(
    discrete_factors: list,
    factor_values: dict,
    axis_options: list,
    default_x,
    default_y,
    color_by_options: list,
    initial_display_points: int,
    discrete_threshold: int,
    has_params_json: bool,
    cols_hint: str,
    initial_views_dir: str = '/out/dashboard',
) -> html.Div:
    _section = {'fontWeight': 'bold', 'color': _TEXT_DIM, 'fontSize': '11px',
                'marginBottom': '4px', 'textTransform': 'uppercase', 'letterSpacing': '0.05em'}
    return html.Div([
        dcc.Location(id='url', refresh=False),
        theme_pref_store(),

        # ── Plot ─────────────────────────────────────────────────────────────
        make_collapsible_panel('controls', 'Plot', start_open=True, children=[
            # Row 1: Axis + color + lazy/refresh
            html.Div([
                html.Div([
                    html.Label('X-axis:'),
                    dcc.Dropdown(id='x-axis',
                                 options=[{'label': c, 'value': c} for c in axis_options],
                                 value=default_x, clearable=True, placeholder='Select X axis...'),
                ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),

                html.Div([
                    html.Label('Y-axis:'),
                    dcc.Dropdown(id='y-axis',
                                 options=[{'label': c, 'value': c} for c in axis_options],
                                 value=default_y, clearable=True, multi=True,
                                 placeholder='Select Y axis(es)...'),
                ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),

                html.Div([
                    html.Label('Color by:'),
                    dcc.Dropdown(id='color-by-factor', options=color_by_options,
                                 value=[SELECTION_VALUE], clearable=True, multi=True,
                                 placeholder='None (uniform color)'),
                ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    html.Label('Color values:'),
                    dcc.Dropdown(id='color-by-values', options=[], value=[], multi=True,
                                 placeholder='Select a color factor first...'),
                ], id='color-values-container',
                   style={'width': '18%', 'display': 'none', 'marginRight': '2%'}),

                html.Div([
                    dcc.Checklist(id='color-as-continuous',
                                  options=[{'label': ' Continuous', 'value': 'on'}], value=[],
                                  inputStyle={'marginRight': '3px'},
                                  labelStyle={'color': _TEXT_DIM, 'fontSize': '11px'}),
                ], id='color-continuous-toggle-container',
                   style={'display': 'none', 'width': '9%', 'marginRight': '1%', 'verticalAlign': 'bottom'}),

                html.Div([
                    html.Label('Color scale:'),
                    dcc.Dropdown(id='color-scale-picker', options=_COLOR_SCALE_OPTIONS,
                                 value='Viridis', clearable=False),
                    dcc.Checklist(id='color-scale-reverse',
                                  options=[{'label': ' Reverse', 'value': 'on'}], value=[],
                                  style={'fontSize': '11px', 'marginTop': '3px'},
                                  inputStyle={'marginRight': '3px'}, labelStyle={'color': _TEXT_DIM}),
                ], id='color-scale-container',
                   style={'width': '18%', 'display': 'none', 'marginRight': '2%'}),

                html.Div([
                    dcc.Checklist(id='heatmap-toggle',
                                  options=[{'label': ' Heatmap (median per bin)', 'value': 'on'}], value=[],
                                  inputStyle={'marginRight': '3px'},
                                  labelStyle={'color': _TEXT_DIM, 'fontSize': '11px'}),
                    html.Div([
                        item
                        for label, cid in (('x bin', 'heatmap-dx'), ('y bin', 'heatmap-dy'), ('min n', 'heatmap-min-n'))
                        for item in (
                            html.Label(label, style={'fontSize': '11px', 'marginRight': '3px', 'color': _TEXT_DIM}),
                            dcc.Input(id=cid, type='number', value=None, min=0, step='any', placeholder='1',
                                      style={**input_style, 'width': '48px', 'marginRight': '6px'}))
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '3px'}),
                ], id='heatmap-container',
                   style={'width': '22%', 'display': 'none', 'marginRight': '2%'}),

                html.Div([
                    html.Button('Lazy Mode (ON)', id='lazy-mode-toggle', n_clicks=0, style={
                        'padding': '6px 12px', 'fontSize': '12px', 'cursor': 'pointer',
                        'border': f'2px solid {_BORDER}', 'borderRadius': '4px',
                        'background': _ACCENT, 'color': _TEXT, 'fontWeight': 'bold', 'marginRight': '8px',
                    }),
                    html.Button('Refresh', id='refresh-btn', n_clicks=0, style={
                        'padding': '6px 12px', 'fontSize': '12px', 'cursor': 'pointer',
                        'border': f'2px solid {_OK}', 'borderRadius': '4px',
                        'background': _OK, 'color': _ON_SOLID, 'fontWeight': 'bold',
                    }),
                ], style={'width': '18%', 'display': 'inline-block', 'verticalAlign': 'bottom'}),
            ], style={'marginBottom': '20px'}),

            # Row 2: Aggregation
            html.Div([
                html.Div(['Aggregation:'], style={'marginBottom': '10px', 'fontStyle': 'italic', 'color': _TEXT_DIM}),
                html.Div([
                    html.Div([
                        html.Label('Group by:'),
                        html.Div([
                            html.Button('Select All', id='agg-group-select-all', n_clicks=0,
                                        style={**btn_sm, 'marginRight': '4px'}),
                            html.Button('Clear All', id='agg-group-clear-all', n_clicks=0, style=btn_sm),
                        ], style={'marginBottom': '4px'}),
                        dcc.Dropdown(id='agg-group-by',
                                     options=[{'label': f, 'value': f} for f in discrete_factors],
                                     value=[], multi=True, placeholder='Select columns to group by...'),
                    ], style={'width': '36%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                    html.Div([
                        html.Label('Y Aggregation:'),
                        dcc.Dropdown(id='agg-y-mode', value='raw', clearable=False,
                                     options=_AGG_OPTIONS_WITH_RAW),
                    ], style={'width': '18%', 'display': 'inline-block', 'verticalAlign': 'top',
                              'marginRight': '2%'}),
                    html.Div([
                        html.Label('Bin width:'),
                        dcc.Input(id='x-bin-width', type='number', value=1, min=0, step='any',
                                  style=input_style),
                    ], id='x-bin-width-container',
                       style={'width': '10%', 'display': 'none', 'verticalAlign': 'top'}),
                ]),
            ], style={'marginBottom': '20px'}),
        ]),

        # ── Filters ───────────────────────────────────────────────────────────
        make_collapsible_panel('custom', 'Filters', start_open=True, children=[
            html.Div([
                html.Div(['Factor filters:'],
                         style={'marginBottom': '10px', 'fontStyle': 'italic', 'color': _TEXT_DIM}),
                html.Div([
                    html.Div([
                        html.Label(factor),
                        html.Div([
                            html.Button('Select All', id={'type': 'select-all', 'factor': factor},
                                        n_clicks=0, style={**btn_sm, 'marginRight': '4px'}),
                            html.Button('Clear All', id={'type': 'clear-all', 'factor': factor},
                                        n_clicks=0, style=btn_sm),
                        ], style={'marginBottom': '4px'}),
                        dcc.Dropdown(id={'type': 'dropdown', 'factor': factor},
                                     options=[{'label': v, 'value': v} for v in factor_values[factor]],
                                     value=[], multi=True),
                    ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'})
                    for factor in discrete_factors
                ]),
                html.Div(id='derived-filter-dropdowns', style={'display': 'inline'}),
            ], style={'marginBottom': '20px'}),
            html.Div([
                html.Div(['Filter (df.query):'],
                         style={'fontStyle': 'italic', 'color': _TEXT_DIM, 'fontSize': '12px'}),
                dcc.Textarea(id='filter-expr', value='',
                             placeholder=f'e.g. (T_loss > 0.1) & (T_accuracy <= 0.95)\nColumns: {cols_hint}',
                             style={'width': '100%', 'fontFamily': 'monospace', 'fontSize': '12px',
                                    'minHeight': '40px', 'marginTop': '4px', 'padding': '6px',
                                    'border': f'1px solid {_BORDER}', 'borderRadius': '4px',
                                    'background': _PANEL, 'color': _TEXT}),
                html.Div(id='filter-error',
                         style={'color': _DANGER, 'fontSize': '11px', 'marginTop': '4px', 'fontFamily': 'monospace'}),
                html.Div(id='filter-count', style={'color': _TEXT_DIM, 'fontSize': '11px', 'marginTop': '2px'}),
            ], style={'marginTop': '6px', 'marginBottom': '12px'}),
            html.Div([
                html.Div(['Computed fields (name = expr, one per line):'],
                         style={'fontStyle': 'italic', 'color': _TEXT_DIM, 'fontSize': '12px'}),
                html.Div('columns as Series · np.* · pandas methods (.clip .astype .min …) · abs round min max',
                         style={'color': _TEXT_DIM, 'fontSize': '11px', 'marginTop': '2px',
                                'fontFamily': 'monospace'}),
                dcc.Textarea(id='fields-expr', value='',
                             placeholder=(
                                 'e.g.:\n  loss_sq = T_loss ** 2\n  acc_pct = T_accuracy * 100\n'
                                 '  ratio = T_loss / (T_accuracy + 1e-8)\n  clipped = T_accuracy.clip(0, 0.95)\n'
                                 '  better = np.minimum(kqgr_kq, kqgr_gr)\n  norm = T_loss - T_loss.min()\n'
                                 '  k_int = R_k_avg.astype(int)'
                             ),
                             style={'width': '100%', 'fontFamily': 'monospace', 'fontSize': '12px',
                                    'minHeight': '50px', 'marginTop': '4px', 'padding': '6px',
                                    'border': f'1px solid {_BORDER}', 'borderRadius': '4px',
                                    'background': _PANEL, 'color': _TEXT}),
                html.Div(id='fields-error',
                         style={'color': _DANGER, 'fontSize': '11px', 'marginTop': '4px', 'fontFamily': 'monospace'}),
                html.Div(id='fields-info', style={'color': _TEXT_DIM, 'fontSize': '11px', 'marginTop': '2px'}),
            ]),
        ]),

        # ── Appearance ────────────────────────────────────────────────────────
        make_collapsible_panel('appearance', 'Appearance', start_open=True, children=[
            html.Div([
                html.Div([
                    make_theme_toggle(),
                ], style={'marginBottom': '12px'}),

                html.Div('Markers', style=_section),
                html.Div([
                    html.Label('Size:', style={'fontSize': '12px', 'marginRight': '6px'}),
                    dcc.Input(id='setting-marker-size', type='number', value=5,
                              min=1, max=30, step=1, style={**input_style, 'width': '55px'}),
                    html.Label('Opacity:', style={'fontSize': '12px', 'marginLeft': '12px', 'marginRight': '6px'}),
                    dcc.Input(id='setting-marker-opacity', type='number', value=0.7,
                              min=0.05, max=1.0, step=0.05, style={**input_style, 'width': '55px'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '6px'}),
                html.Div([
                    dcc.Checklist(
                        id='legend-show-factor-name',
                        options=[{'label': ' Factor name', 'value': 'on'}], value=['on'],
                        style={'display': 'inline-block', 'fontSize': '11px', 'marginRight': '16px'},
                        inputStyle={'marginRight': '3px'}, labelStyle={'color': _TEXT_DIM},
                    ),
                    dcc.Checklist(
                        id='skip-sparse-factors',
                        options=[{'label': ' Skip single-value factors', 'value': 'on'}], value=['on'],
                        style={'display': 'inline-block', 'fontSize': '11px', 'marginRight': '16px'},
                        inputStyle={'marginRight': '3px'}, labelStyle={'color': _TEXT_DIM},
                    ),
                    dcc.Checklist(
                        id='setting-sticky-colors',
                        options=[{'label': ' Sticky colors', 'value': 'on'}], value=['on'],
                        style={'display': 'inline-block', 'fontSize': '11px'},
                        inputStyle={'marginRight': '3px'}, labelStyle={'color': _TEXT_DIM},
                    ),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),

                html.Div('Jitter & Binning', style=_section),
                make_jitter_controls('x'),
                make_jitter_controls('y'),
                html.Div(style={'marginBottom': '12px'}),

                html.Div('Lines', style=_section),
                html.Div([
                    html.Label('Width:', style={'fontSize': '12px', 'marginRight': '6px'}),
                    dcc.Input(id='overlay-line-width', type='number', value=2,
                              min=0.5, max=20, step=0.5, style={**input_style, 'width': '55px'}),
                    html.Label('Opacity:', style={'fontSize': '12px', 'marginLeft': '12px', 'marginRight': '6px'}),
                    dcc.Input(id='overlay-line-opacity', type='number', value=1.0,
                              min=0.05, max=1.0, step=0.05, style={**input_style, 'width': '55px'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '6px'}),
                html.Div([
                    dcc.Checklist(
                        id='overlay-dots-toggle',
                        options=[{'label': ' Dots', 'value': 'on'}],
                        value=['on'],
                        style={'display': 'inline-block', 'fontSize': '11px', 'marginRight': '12px'},
                        inputStyle={'marginRight': '3px'}, labelStyle={'color': _TEXT_DIM},
                    ),
                    dcc.Checklist(
                        id='overlay-line-toggle',
                        options=[{'label': ' Line', 'value': 'on'}],
                        value=[],
                        style={'display': 'inline-block', 'fontSize': '11px', 'marginRight': '12px'},
                        inputStyle={'marginRight': '3px'}, labelStyle={'color': _TEXT_DIM},
                    ),
                    html.Div(
                        dcc.Checklist(
                            id='overlay-per-line-toggle',
                            options=[{'label': ' Per line', 'value': 'on'}],
                            value=[],
                            style={'display': 'inline-block', 'fontSize': '11px',
                                   'marginRight': '8px'},
                            inputStyle={'marginRight': '3px'},
                            labelStyle={'color': _TEXT_DIM},
                        ),
                        id='overlay-per-line-container',
                        style={'display': 'none'},
                    ),
                    html.Div([
                        html.Span('k:', style=label_style),
                        dcc.Input(id='overlay-spline-k', type='number', value=1,
                                  min=1, max=5, step=1,
                                  style={**input_style, 'width': '36px'}),
                        html.Span('s:', style={**label_style, 'marginLeft': '6px'}),
                        dcc.Input(id='overlay-spline-s', type='number', value=None,
                                  min=0, step='any', placeholder='auto',
                                  style={**input_style, 'width': '48px'}),
                    ], id='overlay-spline-controls', style={'display': 'none'}),
                    html.Div([
                        html.Span('anchor:', style={**label_style, 'marginLeft': '8px'}),
                        dcc.Checklist(
                            id='overlay-anchor-left',
                            options=[{'label': ' L', 'value': 'on'}],
                            value=[],
                            style={'display': 'inline-block', 'fontSize': '11px'},
                            inputStyle={'marginRight': '2px'}, labelStyle={'color': _TEXT_DIM},
                        ),
                        dcc.Checklist(
                            id='overlay-anchor-right',
                            options=[{'label': ' R', 'value': 'on'}],
                            value=[],
                            style={'display': 'inline-block', 'fontSize': '11px',
                                   'marginLeft': '4px'},
                            inputStyle={'marginRight': '2px'}, labelStyle={'color': _TEXT_DIM},
                        ),
                    ], id='overlay-anchor-controls', style={'display': 'none'}),
                    html.Div(id='overlay-spline-per-line',
                             style={'display': 'none', 'flexWrap': 'wrap',
                                    'gap': '4px', 'marginTop': '4px'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap'}),
                html.Div(id='overlay-y-styles-container',
                         style={'display': 'none'}),
            ], style={'marginTop': '6px'}),
        ]),

        # ── Settings ──────────────────────────────────────────────────────────
        make_collapsible_panel('settings', '⚙ Settings', start_open=True, children=[
            html.Div([
                html.Div('Sampling', style=_section),
                html.Div([
                    html.Label('Max display points:', style={'fontSize': '12px', 'marginRight': '6px'}),
                    dcc.Input(id='setting-display-points', type='number', value=initial_display_points,
                              min=1000, step=1000, style={**input_style, 'width': '80px'}),
                    html.Span('sampled when data exceeds this', style={**label_style, 'marginLeft': '8px'}),
                ], style={'marginBottom': '4px', 'display': 'flex', 'alignItems': 'center'}),
                html.Div([
                    dcc.Checklist(
                        id='setting-stratified-sample',
                        options=[{'label': ' Stratified sampling (preserve color groups)', 'value': 'on'}],
                        value=['on'],
                        style={'display': 'inline-block', 'fontSize': '11px'},
                        inputStyle={'marginRight': '3px'}, labelStyle={'color': _TEXT_DIM},
                    ),
                ], style={'marginBottom': '4px'}),
                html.Div([
                    html.Label('Discrete threshold:', style={'fontSize': '12px', 'marginRight': '6px'}),
                    dcc.Input(id='setting-discrete-threshold', type='number', value=discrete_threshold,
                              min=2, step=1, style={**input_style, 'width': '60px'}),
                    html.Span('unique values below this → discrete (base factors: restart required)',
                              style={**label_style, 'marginLeft': '8px'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),
            ], style={'marginTop': '6px'}),
        ]),

        # ── Stores, graph, detail panel ───────────────────────────────────────
        dcc.Store(id='lazy-mode-active', data=True),
        dcc.Store(id='trace-row-map', data=None),
        dcc.Store(id='color-offsets', data={}),
        dcc.Store(id='saved-viewport', data=None),
        dcc.Store(id='saved-trace-visibility', data=None),
        dcc.Store(id='live-trace-visibility', data={}),
        dcc.Store(id='overlay-spline-ks', data=[]),
        dcc.Store(id='overlay-y-styles', data=[]),
        dcc.Store(id='legend-shift-click', data=''),
        dcc.Store(id='active-view-name', data=None),
        dcc.Store(id='view-figure', data=None),   # the loaded view's _extras.figure (edited in the JSON only)
        dcc.Store(id='derived-factors', data=[]),
        dcc.Store(id='process-trigger', data=0),
        dcc.Store(id='pending-derived-filters', data={}),
        dcc.Graph(id='scatter-plot', style={'height': '60vh'}),
        html.Div(id='point-details', children='Click a point to see details.', style={
            'padding': '10px', 'background': _PANEL, 'border': f'1px solid {_BORDER}',
            'borderRadius': '4px', 'fontSize': '12px', 'fontFamily': 'monospace', 'maxHeight': '320px',
            'overflowY': 'auto', 'whiteSpace': 'pre-wrap', 'marginTop': '10px', 'color': _TEXT,
        }),

        # ── YAML export (hidden when no params_json column) ───────────────────
        html.Div([
            html.Div([
                html.Button('Export YAML', id='export-yaml-btn', n_clicks=0, disabled=True, style={
                    'padding': '2px 12px', 'fontSize': '12px',
                    'background': _CONTROL, 'color': _TEXT, 'border': f'1px solid {_BORDER}',
                    'borderRadius': '4px', 'cursor': 'pointer',
                }),
            ], style={'marginTop': '6px', 'display': 'flex', 'alignItems': 'center'}),
            html.Div(id='export-status', style={'fontSize': '11px', 'color': _TEXT_DIM, 'marginTop': '4px'}),
        ], hidden=not has_params_json),
        dcc.Download(id='yaml-download'),

        # ── Views ─────────────────────────────────────────────────────────────
        make_collapsible_panel('views', 'Views', start_open=True, children=[
            html.Div([
                html.Div([
                    dcc.Input(id='view-name', type='text', placeholder='View name...',
                              style={'fontSize': '12px', 'padding': '4px 8px', 'width': '180px',
                                     'marginRight': '8px', 'border': f'1px solid {_BORDER}', 'borderRadius': '4px',
                                     'background': _PANEL, 'color': _TEXT}),
                    html.Button('Save', id='view-save-btn', n_clicks=0,
                                style={**btn_sm, 'marginRight': '8px', 'background': _OK,
                                       'color': _ON_SOLID, 'border': f'1px solid {_OK}', 'borderRadius': '4px'}),
                    html.Button('Delete', id='view-delete-btn', n_clicks=0,
                                style={**btn_sm, 'marginRight': '16px', 'background': _DELETE,
                                       'color': _ON_SOLID, 'border': f'1px solid {_DELETE}', 'borderRadius': '4px'}),
                    dcc.Dropdown(id='view-picker',
                                 options=_view_dropdown_options(Path(initial_views_dir)),
                                 value=None, clearable=True,
                                 placeholder='Load a saved view...',
                                 style={'width': '220px', 'display': 'inline-block',
                                        'verticalAlign': 'middle', 'fontSize': '12px'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '4px'}),
                html.Div([
                    html.Span('Dir:', style={'fontSize': '11px', 'color': _TEXT_DIM, 'marginRight': '4px'}),
                    dcc.Input(id='views-dir', type='text', value=initial_views_dir, debounce=True,
                              style={'fontSize': '11px', 'padding': '2px 6px', 'width': '300px',
                                     'border': f'1px solid {_BORDER}', 'borderRadius': '4px',
                                     'fontFamily': 'monospace', 'background': _PANEL, 'color': _TEXT}),
                    admin_token_store(),
                ], style={'marginTop': '4px'}),
                html.Div(id='view-status', style={'color': _TEXT_DIM, 'fontSize': '11px', 'marginTop': '4px'}),
            ], style={'marginTop': '6px'}),
        ]),
        dcc.Store(id='selected-row-idx', data=None),
    ], style={'background': _SURFACE, 'color': _TEXT, 'padding': '16px', 'minHeight': '100vh'})
