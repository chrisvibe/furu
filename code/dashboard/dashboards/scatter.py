from pathlib import Path
import colorsys
import functools
import plotly.io as pio
from dash import Dash, dcc, html, callback, Input, Output, State, ALL, ctx, MATCH, no_update
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from hashlib import md5
from dash.dependencies import Input, Output, State
import yaml
import orjson
import traceback as _traceback
import time as _time

from project.boolean_reservoir.code.utils.single_run import fetch_params_json
from .utils import categorize_object_columns

from .scatter_layout import (
    build_layout, make_collapsible_panel, VIEW_FIELDS, control_defaults,
    _SURFACE, _CONTROL, _BORDER, _TEXT, _TEXT_DIM,
    btn_sm, SELECTION_VALUE,
)
from .theme import (
    DARK as _DARK_THEME, resolve as resolve_theme, register_templates,
    register_theme_attribute, index_string as _theme_index_string,
    HUE_STOPS as _HUE_STOPS,
    ACCENT as _ACCENT, PANEL as _PANEL, FAINT as _FAINT, DANGER as _DANGER,
)
from .utils import (
    register_export_route, make_combo_column,
    _views_dir, _list_views, _view_dropdown_options,
    parse_and_apply_fields, _fmt, apply_trace_vis, apply_figure_options, group_key as _group_key,
    load_figure_presets, preset_series, preset_series_dash, preset_series_symbol, preset_factor_symbol, preset_color_offset,
    preset_factor_colors, level_name,
    resolve_figure_options,
    guard_view_write, DEFAULT_SAFE_ROOTS, register_admin_token_capture,
    admin_token_state,
    save_view_json, load_view_json, delete_view_json,
    aggregate_df, _AGG_OPTIONS_WITH_RAW, ViewDataMismatch, _FONT_BASE_PX, view_name_from_query,
)
from .view_filters import apply_view_filters, missing_filter_values


def _integer_tick_kwargs(v_min: int, v_max: int, max_ticks: int = 20) -> dict:
    """Ticks for a jittered integer axis: every integer on a short range, else plotly's own (round) ticks as integers
    (a forced dtick from v_min gave -23, 3, 29, ... and crowded small panels)."""
    if v_max - v_min <= max_ticks:
        return {'tickmode': 'array', 'tickvals': list(range(v_min, v_max + 1))}
    return {'tickformat': 'd'}

def _present(v) -> bool:
    """A value worth a details line: not missing (a list or dict always is)."""
    if isinstance(v, (list, dict, tuple)):
        return True
    try:
        return bool(pd.notna(v))
    except (TypeError, ValueError):
        return True


def _out_path(params, row):
    """A row's run folder: its params' out path, else the lean boot's L_out_path column."""
    out = ((params.get('logging') or {}).get('out_path') or (params.get('L') or {}).get('out_path')) if params else None
    out = out or row.get('L_out_path')
    return str(out) if _present(out) and str(out).strip() else None


def _detail_value(v) -> str:
    if isinstance(v, (float, np.floating)):   # as written when short (2.0 stays a float), else 6 digits
        short = repr(float(v))
        return short if len(short) <= 8 else f'{v:.6g}'
    return str(v)


def details_tree(sections: dict, open_sections=(), note=None) -> list:
    """A clicked point's details as a collapsible tree: one <details> per section (open if listed), a nested dict or
    list (a run's params) one level per key, a scalar one 'key: value' line. Empty sections are left out."""
    def node(key, v, is_open=False, depth=0):
        if isinstance(v, dict) or isinstance(v, list):
            items = v.items() if isinstance(v, dict) else enumerate(v)
            kids = [node(k, c, depth=depth + 1) for k, c in items]
            return html.Details([html.Summary(str(key), style={'cursor': 'pointer'}), *kids], open=is_open,
                                style={'marginLeft': f'{12 if depth else 0}px'})
        return html.Div(f'{key}: {_detail_value(v)}', style={'marginLeft': '12px'})
    out = [html.Div(note, style={'color': _TEXT_DIM})] if note else []
    for key, v in sections.items():
        if v:
            out.append(node(key, v, is_open=key in open_sections))
    return out


def _sorted_groups(series) -> list:
    groups = sorted((v for v in series.unique() if v != 'Other'), key=_group_key)
    if (series == 'Other').any():
        groups.append('Other')
    return groups

def _drop_constant_cols(df, cols):
    return [c for c in cols if df[c].nunique() > 1]



def _detect_gpu():
    """Detect if GPU-accelerated WebGL is likely available on the client."""
    import os, subprocess
    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        return True
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.lower().split('\n'):
                if 'vga' in line or '3d controller' in line or 'display' in line:
                    return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False

_COLOR_SPECIAL = frozenset({SELECTION_VALUE, '_header_discrete', '_header_continuous', '_header_derived'})

def _stratified_sample(df, n, color_factors):
    """Sample n rows from df, stratified by color group so small groups aren't dropped."""
    strat_cols = [f for f in (color_factors or [])
                  if f and f not in _COLOR_SPECIAL and f in df.columns]
    if not strat_cols:
        return df.sample(n, random_state=None)
    key = (df[strat_cols[0]].astype(str) if len(strat_cols) == 1
           else df[strat_cols[0]].astype(str).str.cat(
               [df[c].astype(str) for c in strat_cols[1:]], sep='_'))
    per_group = max(1, n // key.nunique())
    sampled = (df.assign(_strat=key)
               .groupby('_strat', group_keys=False)
               .apply(lambda g: g.sample(min(len(g), per_group), random_state=None))
               .drop(columns='_strat', errors='ignore'))
    if len(sampled) < n:
        remainder = df.loc[~df.index.isin(sampled.index)]
        if len(remainder) > 0:
            sampled = pd.concat([sampled,
                                 remainder.sample(min(n - len(sampled), len(remainder)),
                                                  random_state=None)])
    return sampled

def _get_overlay_ks(view_data: dict) -> list:
    """The spline k/s per line: the per-line list when the per-line toggle is on, else the one k/s for every line."""
    ks = view_data.get('overlay-spline-ks')
    if ks and 'on' in (view_data.get('overlay-per-line-toggle') or []):
        return ks
    k = view_data.get('overlay-spline-k')
    s = view_data.get('overlay-spline-s')
    return [[k, s]] if k is not None or s is not None else [[1, None]]

def _add_line_overlay(fig, df, x_col, y_col, group_col, color_map, overlay_ks=None,
                      use_webgl=False, anchor_left=False, anchor_right=False, ks_offset=0,
                      line_width=2, line_opacity=1.0, dash_of=None):
    """Add one smoothed line trace per group onto fig, matching dot colors (and dash_of(group)'s dash, if any)."""
    from scipy.interpolate import UnivariateSpline
    ScatterType = go.Scattergl if use_webgl else go.Scatter
    ks_list  = list(overlay_ks or [])
    default  = ks_list[0] if ks_list else [1, None]
    for i, (group_label, color) in enumerate((color_map or {}).items()):
        if i >= _MAX_OVERLAY_GROUPS:
            break
        ks = ks_list[i + ks_offset] if (i + ks_offset) < len(ks_list) else default
        k  = max(1, min(5, int(ks[0]))) if ks[0] is not None else 1
        s  = float(ks[1]) if ks[1] is not None else 0.0
        sub = (df[df[group_col] == group_label] if group_col is not None else df)
        sub = sub.dropna(subset=[x_col, y_col]).drop_duplicates(subset=[x_col]).sort_values(x_col)
        if len(sub) < k + 1:
            continue
        x_vals = sub[x_col].values.astype(float)
        y_vals = sub[y_col].values.astype(float)
        try:
            w = np.ones(len(x_vals))
            if anchor_left:
                w[0] = 1e6
            if anchor_right:
                w[-1] = 1e6
            spl = UnivariateSpline(x_vals, y_vals, k=k, s=s, w=w)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
            y_line = spl(x_line)
        except Exception:
            x_line, y_line = x_vals, y_vals  # fall back to connect-dots
        fig.add_trace(ScatterType(
            x=x_line, y=y_line,
            mode='lines',
            line=dict(color=color, width=line_width, dash=(dash_of(group_label) if dash_of else None)),
            opacity=line_opacity,
            name=group_label,
            showlegend=True,
            hoverinfo='skip',
        ))

_COSMETIC_TRIGGERS = frozenset({
    'color-offsets', 'legend-show-factor-name',
    'overlay-dots-toggle', 'overlay-line-toggle',
    'overlay-line-width', 'overlay-line-opacity',
    'overlay-spline-k', 'overlay-spline-s',
    'overlay-per-line-toggle', 'overlay-spline-ks',
    'overlay-anchor-left', 'overlay-anchor-right',
    'overlay-y-styles',
    'color-as-continuous',
    'heatmap-toggle', 'heatmap-dx', 'heatmap-dy', 'heatmap-min-n',
    'color-by-values',
    'skip-sparse-factors',
    'theme-toggle',
})

_MAX_OVERLAY_GROUPS = 15

_Y_STYLES = [
    {'symbol': 'circle',        'dash': 'solid'},
    {'symbol': 'square',        'dash': 'dash'},
    {'symbol': 'diamond',       'dash': 'dot'},
    {'symbol': 'cross',         'dash': 'dashdot'},
    {'symbol': 'x',             'dash': 'longdash'},
    {'symbol': 'triangle-up',   'dash': 'solid'},
    {'symbol': 'triangle-down', 'dash': 'dash'},
    {'symbol': 'star',          'dash': 'dot'},
]

class ScatterDashboard:
    """
    Encapsulates a Plotly/Dash scatter dashboard with view save/load,
    server-side PNG export, and a two-stage data/figure callback pipeline.

    Usage:
        app = ScatterDashboard(df, factors).build()
        app.run(debug=False)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        factors: list[str],
        discrete_threshold: int = 100,
        renderer: str = 'auto',
        initial_display_points: int = 10_000,
        safe_roots: list[str] = None,
        initial_views_dir: str = '/out/dashboard',
        url_prefix: str = '/',
        config_by_lout: dict = None,
    ):
        if safe_roots is None:
            safe_roots = list(DEFAULT_SAFE_ROOTS)
        self._initial_views_dir = initial_views_dir
        self._url_prefix = url_prefix

        if renderer == 'auto':
            self.use_webgl = _detect_gpu()
        elif renderer == 'webgl':
            self.use_webgl = True
        else:
            self.use_webgl = False
        print(f"Scatter dashboard renderer: {'WebGL (Scattergl)' if self.use_webgl else 'SVG (Scatter)'}")

        self.df = df.copy()
        # params_json may be absent (lean boot, wsgi/base.py) — in that case a
        # per-row lazy fetch via _fetch_params_row covers the export/hover needs.
        self._config_by_lout = config_by_lout or {}
        self.has_params_json = 'params_json' in df.columns or bool(self._config_by_lout)
        self.initial_display_points = initial_display_points
        self.discrete_threshold = discrete_threshold
        self._safe_roots = [Path(r) for r in safe_roots]

        # Factor analysis
        self.df['design'], self.factors_subset = make_combo_column(self.df, factors, return_as_str=True)
        # The freshly built design column is a high-cardinality object string —
        # same str-first category cast as base.py (value contract preserved).
        categorize_object_columns(self.df)
        candidate_factors = list(self.factors_subset)
        self.discrete_factors   = [f for f in candidate_factors if self.df[f].nunique() < discrete_threshold]
        # continuous colours: the many-valued factors, and every numeric measurement column (accuracy, KQ, ...);
        # colour only, never a filter
        measures = [c for c in self.df.columns
                    if c not in candidate_factors and c != 'design' and not str(c).startswith('_')
                    and pd.api.types.is_numeric_dtype(self.df[c]) and not pd.api.types.is_bool_dtype(self.df[c])]
        self.continuous_factors = [f for f in candidate_factors if self.df[f].nunique() >= discrete_threshold] + measures
        self.numeric_discrete_factors = [
            f for f in self.discrete_factors if pd.api.types.is_numeric_dtype(self.df[f])
        ]
        self.factor_values = {
            f: [str(v) for v in sorted(v for v in self.df[f].unique() if pd.notna(v))]
            for f in self.discrete_factors
        }

        # Axis options
        # the numeric columns, then the text factors (categories on an axis, e.g. one point per head)
        numeric_cols = self.df.select_dtypes(include='number').columns.tolist()
        text_factors = [f for f in self.discrete_factors if not pd.api.types.is_numeric_dtype(self.df[f])]
        self.axis_options = list(dict.fromkeys(numeric_cols + text_factors))
        self.default_x = 'R_k_avg'   if 'R_k_avg'   in self.axis_options else (self.axis_options[0] if self.axis_options else None)
        self.default_y = 'T_accuracy' if 'T_accuracy' in self.axis_options else (self.axis_options[1] if len(self.axis_options) > 1 else None)

        # Color — the active palette comes from the Theme threaded into each render.
        self.color_by_options = [
            {'label': '\u2b21 Filter Selection', 'value': SELECTION_VALUE},
            {'label': '\u2500\u2500 Discrete \u2500\u2500', 'value': '_header_discrete', 'disabled': True},
        ]
        for f in self.discrete_factors:
            self.color_by_options.append({'label': f'  {f}', 'value': f})
        if self.continuous_factors:
            self.color_by_options.append(
                {'label': '\u2500\u2500 Continuous \u2500\u2500', 'value': '_header_continuous', 'disabled': True}
            )
            for f in self.continuous_factors:
                self.color_by_options.append({'label': f'  {f}', 'value': f})

        self.ScatterType = go.Scattergl if self.use_webgl else go.Scatter
        self.app = None  # set in build()
        self._pipeline_cache_key: object = None
        self._pipeline_cache_val: tuple | None = None

    # ── Color helpers ─────────────────────────────────────────────────────────
    # A Theme is threaded through explicitly rather than stashed on self: the
    # instance is shared by every request a worker serves, so per-instance theme
    # state made one user's render depend on the previous request's theme.

    def _axis_dropdown_options(self, df_with_fields=None) -> list:
        """x/y dropdown options: the numeric columns, then under a '── computed ──' header the numeric
        columns computed fields added to df_with_fields."""
        opts = [{'label': c, 'value': c} for c in self.axis_options]
        new = [] if df_with_fields is None else [
            c for c in df_with_fields.select_dtypes(include='number').columns if c not in self.axis_options]
        if new:
            opts += ([{'label': '── computed ──', 'value': '_h_computed', 'disabled': True}]
                     + [{'label': f'  {c}', 'value': c} for c in new])
        return opts

    def _view_axis_options(self, fields_expr) -> list:
        """The dropdown options a saved view needs before its values land. Dash 4's dcc.Dropdown drops a
        value that is not among its options, so a view whose y is a computed field (e.g. dev_train) opened
        with no y at all. The fields run on a few rows: only the column names matter."""
        if not (fields_expr and fields_expr.strip()):
            return self._axis_dropdown_options()
        sample, _, _ = parse_and_apply_fields(self.df.head(256).copy(), fields_expr)
        return self._axis_dropdown_options(sample)

    @property
    def _figure_presets(self):
        return load_figure_presets(self._initial_views_dir)

    def _series_dash(self, name):
        """The line dash the site presets give the series a group name belongs to, else None."""
        return preset_series_dash(self._figure_presets, name)

    def _series_symbol(self, name):
        """The marker symbol the site presets give a part of a group name, else None."""
        return preset_series_symbol(self._figure_presets, name)

    def _default_offset(self, name, fallback):
        """The colour offset of a trace with no explicit one: its preset series' colour, else fallback."""
        series = preset_series(self._figure_presets, name)
        if series is None:
            return fallback
        resolved = preset_color_offset(self._figure_presets, {'series': series})
        return fallback if resolved is None else resolved

    def _get_stable_color(self, theme, combo_str, offset=0):
        if combo_str == 'Other':
            return theme.other
        palette = theme.palette
        offset = preset_color_offset(self._figure_presets, offset)
        if offset is None:   # a {"series": ...} the presets don't define
            offset = 0
        if isinstance(offset, dict):
            if offset.get('fg'):    # the theme's text colour: black on white, light on dark (a reference series)
                return theme.text
            if 'hsv' in offset:
                h, s, v = offset['hsv']
                r, g, b = colorsys.hsv_to_rgb(h / 360.0, s / 100.0, v / 100.0)
                return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
            if 'h' in offset:
                h_deg = _HUE_STOPS[offset['h'] % len(_HUE_STOPS)]
                s, v = theme.hsv_sv
                r, g, b = colorsys.hsv_to_rgb(h_deg / 360.0, s, v)
                return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
            return palette[offset['d'] % len(palette)]
        hash_value = int(md5(f'{combo_str}_{offset}'.encode()).hexdigest(), 16)
        return palette[hash_value % len(palette)]

    def _make_color_map(self, theme, unique_groups, color_offsets, factor=None):
        """Each group's colour: the view's explicit offset, else its level's colour when the first colour factor
        (factor: a column, or the columns of combined groups) is an ordered colour factor of the presets, else its
        series' colour or a stable hash."""
        offsets = color_offsets or {}
        first = factor if isinstance(factor, str) else (factor or [None])[0]
        levels = preset_factor_colors(self._figure_presets, first)
        level = lambda g: level_name(str(g).split(' | ')[0])
        return {
            g: self._get_stable_color(theme, g, offsets[g]) if g in offsets
               else levels[level(g)] if level(g) in levels
               else self._get_stable_color(theme, g, self._default_offset(g, 0))
            for g in unique_groups
        }

    def _line_color(self, theme, name, color_offsets, fallback):
        """A fitted line's colour: its own shift-click offset ('name::line'), else the trace's, else fallback."""
        offsets = color_offsets or {}
        offset = offsets.get(f'{name}::line', offsets.get(name))
        return fallback if offset is None else self._get_stable_color(theme, name, offset)

    # ── Trace visibility helper ───────────────────────────────────────────────

    @staticmethod
    def _apply_trace_vis(fig, trace_visibility):
        apply_trace_vis(fig, trace_visibility)

    # ── Figure sub-builders ───────────────────────────────────────────────────

    def _make_discrete_figure(self, theme, filtered_df, x_col, y_col, col_name, label, hover_base, color_offsets, cd_cols, show_label=True,
                              factor=None):
        unique_groups = _sorted_groups(filtered_df[col_name])
        color_map = self._make_color_map(theme, unique_groups, color_offsets, factor)
        fig = px.scatter(
            filtered_df, x=x_col, y=y_col,
            color=col_name, color_discrete_map=color_map,
            custom_data=cd_cols,
            category_orders={col_name: unique_groups},
            labels={col_name: label if show_label else ''},
            render_mode='webgl' if self.use_webgl else 'svg',
        )
        fig.update_traces(hovertemplate=hover_base)
        for t in fig.data:
            if self._series_dash(t.name):
                t.line.dash = self._series_dash(t.name)
            symbol = self._series_symbol(t.name) or preset_factor_symbol(self._figure_presets, factor, t.name)
            if symbol:
                t.marker.symbol = symbol
        return fig

    def _make_uniform_figure(self, theme, filtered_df, x_col, y_col, hover_base, cd_cols):
        return go.Figure(self.ScatterType(
            x=filtered_df[x_col], y=filtered_df[y_col],
            mode='markers', marker=dict(color=theme.uniform), name='',
            customdata=filtered_df[cd_cols].values,
            hovertemplate=hover_base,
        ))

    # ── Main figure builder ───────────────────────────────────────────────────

    def _build_figure(self, filtered_df, x_col, y_col, color_cfg=None, color_offsets=None,
                      x_jitter=None, y_jitter=None, y_agg_label=None,
                      x_label=None, marker_size=5, marker_opacity=0.7,
                      overlay_line=False, overlay_ks=None, overlay_dots=True,
                      anchor_left=False, anchor_right=False,
                      theme=None, show_factor_name=True,
                      skip_sparse=True, line_width=2, line_opacity=1.0,
                      _overlay_info_out=None):
        theme = resolve_theme(theme)
        if not x_col or not y_col:
            fig = go.Figure()
            fig.update_layout(title='Select X and Y axes to display data',
                              xaxis_title='X', yaxis_title='Y', template=theme.template)
            return fig
        if filtered_df is None or filtered_df.empty:
            fig = go.Figure()
            fig.update_layout(title='No data for selected filters', xaxis_title=x_col,
                              yaxis_title=y_col, template=theme.template)
            return fig
        filtered_df = filtered_df.copy()
        filtered_df['combo_str'] = filtered_df['design'].astype(str) if 'design' in filtered_df.columns else ''

        if '_row_idx' not in filtered_df.columns:
            filtered_df['_row_idx'] = -1 if y_agg_label else filtered_df.index

        cd_cols = ['_row_idx'] + (['_agg_count'] if '_agg_count' in filtered_df.columns else [])
        for f in self.factors_subset:
            if f not in filtered_df.columns:
                filtered_df[f] = None

        plot_x_col, plot_y_col = x_col, y_col
        if x_jitter:
            jit_col = x_col + '_jit'
            filtered_df[jit_col] = filtered_df[x_col] + np.random.uniform(x_jitter[0], x_jitter[1], size=len(filtered_df))
            plot_x_col = jit_col
        if y_jitter:
            jit_col = y_col + '_jit'
            filtered_df[jit_col] = filtered_df[y_col] + np.random.uniform(y_jitter[0], y_jitter[1], size=len(filtered_df))
            plot_y_col = jit_col

        hover_base = f'<b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra>%{{fullData.name}}</extra>'

        def _selection_color_groups(fdf, filter_dict):
            active = {k: set(v) for k, v in (filter_dict or {}).items() if v and k in fdf.columns}
            if skip_sparse and active:
                varying = _drop_constant_cols(fdf, list(active))
                if varying:  # only narrow down when some factors actually vary
                    active = {k: active[k] for k in varying}
            if not active:
                return None
            is_other = np.zeros(len(fdf), dtype=bool)
            label_parts = []
            for factor, selected in active.items():
                col_str = fdf[factor].astype(str)
                is_other |= ~col_str.isin(selected)
                label_parts.append((factor + ':') + col_str if show_factor_name else col_str)
            group_labels = (label_parts[0] if len(label_parts) == 1
                            else label_parts[0].str.cat(label_parts[1:], sep=', '))
            fdf['color_group'] = group_labels
            fdf.loc[is_other, 'color_group'] = 'Other'
            return 'color_group'

        _overlay_group_col = None
        _overlay_color_map = None

        if color_cfg and color_cfg['type'] == 'continuous':
            col = color_cfg['column']
            fig = px.scatter(
                filtered_df, x=plot_x_col, y=plot_y_col, color=col,
                color_continuous_scale=color_cfg.get('scale', 'Viridis'),
                custom_data=cd_cols,
                render_mode='webgl' if self.use_webgl else 'svg',
            )
            fig.update_traces(hovertemplate=hover_base)
            # no discrete groups — overlay not supported for continuous coloring
        elif color_cfg and color_cfg['type'] == 'precomputed':
            col = color_cfg['column']
            if col in filtered_df.columns:
                unique_groups = _sorted_groups(filtered_df[col])
                fig = self._make_discrete_figure(theme, filtered_df, plot_x_col, plot_y_col, col,
                                                 color_cfg.get('label', 'Group'), hover_base, color_offsets, cd_cols, show_label=show_factor_name)
                _overlay_group_col = col
                _overlay_color_map = self._make_color_map(theme, unique_groups, color_offsets)
            else:
                fig = self._make_uniform_figure(theme, filtered_df, plot_x_col, plot_y_col, hover_base, cd_cols)
        elif color_cfg and color_cfg['type'] == 'columns':
            cols = [c for c in color_cfg['columns'] if c in filtered_df.columns]
            if cols:
                varying_cols = (_drop_constant_cols(filtered_df, cols) if skip_sparse else cols) or cols
                _str_cols = [filtered_df[c].astype(str) for c in varying_cols]
                filtered_df['color_group'] = (
                    _str_cols[0].str.cat(_str_cols[1:], sep=' | ')
                    if len(_str_cols) > 1 else _str_cols[0]
                )
                unique_groups = _sorted_groups(filtered_df['color_group'])
                _factor = varying_cols[0] if len(varying_cols) == 1 else varying_cols
                fig = self._make_discrete_figure(theme, filtered_df, plot_x_col, plot_y_col,
                                                 'color_group', ' | '.join(cols), hover_base, color_offsets, cd_cols, show_label=show_factor_name,
                                                 factor=_factor)
                _overlay_group_col = 'color_group'
                _overlay_color_map = self._make_color_map(theme, unique_groups, color_offsets, _factor)
            else:
                fig = self._make_uniform_figure(theme, filtered_df, plot_x_col, plot_y_col, hover_base, cd_cols)
        elif color_cfg and color_cfg['type'] == 'selection':
            col_name = _selection_color_groups(filtered_df, color_cfg['filter_dict'])
            if col_name:
                unique_groups = _sorted_groups(filtered_df[col_name])
                fig = self._make_discrete_figure(theme, filtered_df, plot_x_col, plot_y_col,
                                                 col_name, 'Selection', hover_base, color_offsets, cd_cols, show_label=show_factor_name)
                _overlay_group_col = col_name
                _overlay_color_map = self._make_color_map(theme, unique_groups, color_offsets)
            else:
                fig = self._make_uniform_figure(theme, filtered_df, plot_x_col, plot_y_col, hover_base, cd_cols)
        elif color_cfg and color_cfg['type'] == 'discrete':
            col = color_cfg['column']
            filtered_df['color_group'] = filtered_df[col].astype(str).where(
                filtered_df[col].astype(str).isin(color_cfg['values']), other='Other'
            )
            unique_groups = _sorted_groups(filtered_df['color_group'])
            fig = self._make_discrete_figure(theme, filtered_df, plot_x_col, plot_y_col,
                                             'color_group', col, hover_base, color_offsets, cd_cols, show_label=show_factor_name,
                                             factor=col)
            _overlay_group_col = 'color_group'
            _overlay_color_map = self._make_color_map(theme, unique_groups, color_offsets, col)
        else:
            fig = self._make_uniform_figure(theme, filtered_df, plot_x_col, plot_y_col, hover_base, cd_cols)
            _overlay_color_map = {'__all__': theme.uniform}

        if _overlay_info_out is not None:
            _overlay_info_out.append((plot_x_col, plot_y_col, filtered_df,
                                      _overlay_group_col, _overlay_color_map))

        hide_dots = not overlay_dots

        if not hide_dots:
            fig.update_traces(marker=dict(size=marker_size, opacity=marker_opacity))

        if overlay_line and _overlay_color_map and _overlay_info_out is None:
            line_colors = {g: c if g == '__all__' else self._line_color(theme, g, color_offsets, c)
                           for g, c in _overlay_color_map.items()}
            _add_line_overlay(fig, filtered_df, plot_x_col, plot_y_col,
                              _overlay_group_col, line_colors, overlay_ks,
                              use_webgl=self.use_webgl,
                              anchor_left=anchor_left, anchor_right=anchor_right,
                              ks_offset=0, line_width=line_width, line_opacity=line_opacity,
                              dash_of=self._series_dash)

        if hide_dots:
            for trace in fig.data:
                if (getattr(trace, 'mode', None) or 'markers') != 'lines':
                    trace.visible = False
                    trace.showlegend = False

        y_label = f'{y_agg_label}({y_col})' if y_agg_label and y_agg_label != 'raw' else y_col
        _x_display = x_label or x_col
        fig.update_layout(
            template=theme.template,
            title=f'{y_label} vs {_x_display}', title_x=0.5,
            xaxis_title=_x_display + (' (jittered)' if x_jitter else ''),
            yaxis_title=y_label  + (' (jittered)' if y_jitter else ''),
            uirevision='scatter-plot', clickmode='event+select',
        )

        if x_jitter and x_col in filtered_df.columns:
            try:
                col = filtered_df[x_col].dropna()
                if pd.api.types.is_integer_dtype(col):
                    fig.update_xaxes(**_integer_tick_kwargs(int(col.min()), int(col.max())))
            except (ValueError, TypeError):
                pass
        if y_jitter and y_col in filtered_df.columns:
            try:
                col = filtered_df[y_col].dropna()
                if pd.api.types.is_integer_dtype(col):
                    fig.update_yaxes(**_integer_tick_kwargs(int(col.min()), int(col.max())))
            except (ValueError, TypeError):
                pass
        if not y_jitter and y_col in filtered_df.columns and y_agg_label != 'std':
            y_min, y_max = filtered_df[y_col].min(), filtered_df[y_col].max()
            if y_min >= 0 and y_max <= 1:
                fig.update_yaxes(range=[0, 1])

        _has_line_traces = any((getattr(t, 'mode', None) or '') == 'lines' for t in fig.data)
        if not _has_line_traces and len({t.name for t in fig.data if t.name}) <= 1:
            fig.update_layout(showlegend=False)

        return fig

    # ── Multi-Y overlay figure builder ────────────────────────────────────────

    def _build_overlay_figure(self, filtered_df, x_col, y_cols, *,
                              filter_expr='', x_bin_width=1,
                              agg_group_by=None, agg_y_mode='raw',
                              color_factor=None, factor_filters=None,
                              y_styles=None,
                              **figure_kwargs):
        """Build one figure with traces from each Y column overlaid on a shared axis.

        Pass 1 (Y-major): scatter traces, each with a unique legendgroup so every
        (group, Y) entry is independently toggleable in the legend.
        Pass 2 (group-major): line overlay traces so their legend order and k/s panel
        indices mirror the scatter ordering — index 0 = first group × first Y, etc.
        """
        fig = go.Figure()
        theme           = resolve_theme(figure_kwargs.get('theme'))
        single_fig = None
        no_color_factor = figure_kwargs.get('color_cfg') is None
        y_styles_list   = y_styles or []
        overlay_line    = figure_kwargs.pop('overlay_line', False)
        overlay_ks      = list(figure_kwargs.pop('overlay_ks', None) or [])
        line_width      = figure_kwargs.pop('line_width', 2)
        line_opacity    = figure_kwargs.pop('line_opacity', 1.0)
        anchor_left     = figure_kwargs.get('anchor_left', False)
        anchor_right    = figure_kwargs.get('anchor_right', False)
        per_y_infos     = []  # per-Y: None or (px_col, py_col, sub_df, group_col, color_map)
        _co             = figure_kwargs.get('color_offsets') or {}
        y_ranges        = []  # per-Y y-axis range _build_figure chose (None = autorange)

        # Pass 1: collect scatter traces per-Y (deferred — added group-major below)
        scatter_traces = {}  # (orig_trace_name, i_y) → (trace, style, y_color, y)
        for i_y, y in enumerate(y_cols):
            sub_df, x_plot, x_bin_lbl, y_agg_lbl, _ = self._apply_data_pipeline(
                filtered_df, x_col, y,
                filter_expr='',
                x_bin_width=x_bin_width,
                agg_group_by=agg_group_by, agg_y_mode=agg_y_mode,
                color_factor=color_factor, factor_filters=factor_filters,
            )
            if sub_df.empty:
                per_y_infos.append(None)
                continue
            tmp = []
            single_fig = self._build_figure(
                sub_df, x_plot, y,
                y_agg_label=y_agg_lbl,
                x_label=x_bin_lbl,
                overlay_line=False,
                _overlay_info_out=tmp,
                **figure_kwargs,
            )
            per_y_infos.append(tmp[0] if tmp else None)
            y_ranges.append(single_fig.layout.yaxis.range)
            # GUI-selected style takes priority; then the group's preset dash and the symbol the figure gave it
            # (series or factor preset, below); auto-cycle is the fallback
            saved  = (y_styles_list[i_y] if i_y < len(y_styles_list) else None) or {}
            style  = {'symbol': saved.get('symbol'), 'auto_symbol': _Y_STYLES[i_y % len(_Y_STYLES)]['symbol'],
                      'dash':   saved.get('dash'), 'auto_dash': _Y_STYLES[i_y % len(_Y_STYLES)]['dash']}
            # Per-Y color: use shift-click stored offset if present; else fall back to index
            y_offset = (_co[y] if y in _co else self._default_offset(y, i_y)) if no_color_factor else None
            y_color  = self._get_stable_color(theme, y, y_offset) if no_color_factor else None
            for trace in single_fig.data:
                scatter_traces[(trace.name, i_y)] = (trace, style, y_color, y)

        if single_fig is None:
            fig.update_layout(title='No data for selected filters')
            return fig

        # Shared group ordering — drives both scatter (group-major) and line (Pass 2) addition
        # Discrete coloring: groups are color_map keys; uniform: single trace with orig_name=''
        all_groups = list(dict.fromkeys(
            g for info in per_y_infos if info is not None
            for g in (info[4] or {})
            if g != '__all__'
        ))
        scatter_orig_names = all_groups if all_groups else ['']

        # Add scatter traces group-major so scatter and line legends share the same ordering
        for orig_name in scatter_orig_names:
            for i_y, y in enumerate(y_cols):
                key = (orig_name, i_y)
                if key not in scatter_traces:
                    continue
                trace, style, y_color, y_val = scatter_traces[key]
                marker_upd = dict(symbol=style['symbol'] or trace.marker.symbol or style['auto_symbol'])
                line_upd   = dict(dash=style['dash'] or self._series_dash(orig_name) or style['auto_dash'])
                if y_color is not None:
                    marker_upd['color'] = y_color
                    line_upd['color']   = y_color
                trace.update(marker=marker_upd, line=line_upd)
                trace.name = f'{orig_name} [{y_val}]' if orig_name else y_val
                # Unique legendgroup per (group, Y) so each entry is independently toggleable
                trace.legendgroup = trace.name
                # Per-(group,Y) shift-click override (only for discrete color case)
                if y_color is None:
                    per_trace_off = _co.get(trace.name)
                    if per_trace_off is not None:
                        oc = self._get_stable_color(theme, trace.name, per_trace_off)
                        trace.update(marker={'color': oc}, line={'color': oc})
                fig.add_trace(trace)

        # Pass 2: line overlay in group-major order (all_groups shared from above)
        if overlay_line:
            ks_offset = 0
            from scipy.interpolate import UnivariateSpline
            ScatterType = go.Scattergl if self.use_webgl else go.Scatter
            for group in all_groups:
                for i_y, y in enumerate(y_cols):
                    info = per_y_infos[i_y] if i_y < len(per_y_infos) else None
                    if info is None or group not in (info[4] or {}):
                        continue
                    px_col, py_col, sub_df, gcol, cmap = info
                    trace_name = f'{group} [{y}]'
                    color     = self._line_color(theme, trace_name, _co, cmap[group])
                    saved     = (y_styles_list[i_y] if i_y < len(y_styles_list) else None) or {}
                    line_dash = saved.get('dash') or self._series_dash(group) or _Y_STYLES[i_y % len(_Y_STYLES)]['dash']
                    ks = overlay_ks[ks_offset] if ks_offset < len(overlay_ks) else (overlay_ks[0] if overlay_ks else [1, None])
                    k  = max(1, min(5, int(ks[0]))) if ks[0] is not None else 1
                    s  = float(ks[1]) if ks[1] is not None else 0.0
                    sub = (sub_df[sub_df[gcol] == group] if gcol else sub_df)
                    sub = sub.dropna(subset=[px_col, py_col]).drop_duplicates(subset=[px_col]).sort_values(px_col)
                    if len(sub) >= k + 1:
                        x_vals = sub[px_col].values.astype(float)
                        y_vals = sub[py_col].values.astype(float)
                        try:
                            w = np.ones(len(x_vals))
                            if anchor_left: w[0] = 1e6
                            if anchor_right: w[-1] = 1e6
                            spl = UnivariateSpline(x_vals, y_vals, k=k, s=s, w=w)
                            x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
                            y_line = spl(x_line)
                        except Exception:
                            x_line, y_line = x_vals, y_vals
                        fig.add_trace(ScatterType(
                            x=x_line, y=y_line,
                            mode='lines',
                            line=dict(color=color, width=line_width, dash=line_dash),
                            opacity=line_opacity,
                            name=trace_name,
                            legendgroup=f'{trace_name}::line',
                            showlegend=True,
                            hoverinfo='skip',
                        ))
                        ks_offset += 1

        fig.update_layout(single_fig.layout)
        # single_fig is only the LAST Y column, so its y-range must not govern the shared
        # axis: _build_figure pins [0, 1] for a column that fits the unit interval, which
        # clips any other series a computed field moved outside it (`T_loss = T_loss + 1.1`).
        # Keep a range only where every Y column agreed on it, else autorange over them all.
        fig.layout.yaxis.range = y_ranges[0] if y_ranges and len(set(y_ranges)) == 1 else None
        y_title = ' | '.join(y_cols)
        fig.update_layout(
            title=f'{y_title} vs {x_col}', title_x=0.5,
            yaxis_title=y_title,
            uirevision='scatter-plot', clickmode='event+select',
            showlegend=True,
        )
        _has_line_traces = any((getattr(t, 'mode', None) or '') == 'lines' for t in fig.data)
        if not _has_line_traces and len({t.name for t in fig.data if t.name}) <= 1:
            fig.update_layout(showlegend=False)
        return fig

    # ── Shared helpers (used by live render and PNG export) ──────────────────

    def _build_color_cfg(self, color_factor, filter_dict,
                         color_values, color_scale, color_scale_reverse, as_continuous):
        """Resolve color_factor + settings into a color_cfg dict for _build_figure."""
        _non_special = [f for f in (color_factor or [])
                        if f and f not in (SELECTION_VALUE, None,
                                           '_header_discrete', '_header_continuous', '_header_derived')]
        if SELECTION_VALUE in (color_factor or []):
            return {'type': 'selection', 'filter_dict': filter_dict or {}}
        if len(_non_special) == 1:
            f = _non_special[0]
            continuous_override = as_continuous and 'on' in (as_continuous or [])
            is_cont = f in self.continuous_factors or (f in self.numeric_discrete_factors and continuous_override)
            if is_cont:
                scale = (color_scale or 'Viridis') + ('_r' if color_scale_reverse and 'on' in color_scale_reverse else '')
                return {'type': 'continuous', 'column': f, 'scale': scale}
            if color_values:
                return {'type': 'discrete', 'column': f, 'values': color_values}
        if _non_special:
            return {'type': 'columns', 'columns': _non_special}
        return None

    # ── Shared data pipeline (used by live render and PNG export) ────────────

    def _apply_data_pipeline(self, filtered_df, x_col, y_col, *,
                              filter_expr='', x_bin_width=1,
                              agg_group_by=None, agg_y_mode='raw', color_factor=None,
                              factor_filters=None):
        """Canonical query → x-binning → aggregation pipeline.

        Returns (filtered_df, x_plot_col, x_bin_label, y_agg_label, filter_error).
        filter_error is None on success or a string describing the query failure.
        """
        _factor_filters = factor_filters or {}
        _color_factor   = color_factor or []
        _agg_group_by   = agg_group_by or []

        filter_error = None
        if filter_expr and filter_expr.strip():
            try:
                filtered_df = filtered_df.query(filter_expr)
            except Exception as e:
                filter_error = str(e)

        x_plot_col, x_bin_label = x_col, None
        if agg_y_mode and agg_y_mode != 'raw' and x_bin_width and float(x_bin_width) > 0:
            bw = float(x_bin_width)
            _bin_col = x_col + '_binned'
            filtered_df = filtered_df.copy()
            filtered_df[_bin_col] = (filtered_df[x_col] / bw).round() * bw
            x_plot_col  = _bin_col
            x_bin_label = f'{x_col} (bin={bw})'

        x_is_binned = x_plot_col != x_col
        y_agg_label = agg_y_mode if (agg_y_mode and agg_y_mode != 'raw' and x_is_binned) else None
        if y_agg_label:
            effective_group = list(_agg_group_by)
            extra_agg_cols, str_collapsed = [], []
            if SELECTION_VALUE in _color_factor:
                for factor, values in _factor_filters.items():
                    if values and factor not in effective_group and factor in filtered_df.columns:
                        effective_group.append(factor)
            for cf in _color_factor:
                if not cf or cf in (SELECTION_VALUE, None) or cf == y_col:
                    continue
                if cf in effective_group or cf not in filtered_df.columns:
                    continue
                if pd.api.types.is_numeric_dtype(self.df[cf]):
                    extra_agg_cols.append(cf)
                else:
                    str_collapsed.append(cf)
            n_before = len(filtered_df)
            filtered_df = aggregate_df(filtered_df, x_plot_col, y_col, effective_group, agg_y_mode,
                                       x_is_binned=x_is_binned, extra_agg_cols=extra_agg_cols)
            for cf in str_collapsed:
                if cf in filtered_df.columns:
                    filtered_df[cf] = 'Other'
            # Groups so fine-grained that every row is its own group — no actual
            # aggregation happened. Fall back to raw so _row_idx stays valid.
            if len(filtered_df) == n_before:
                y_agg_label = None

        return filtered_df, x_plot_col, x_bin_label, y_agg_label, filter_error

    # ── Server-side figure reconstruction (PNG export) ────────────────────────

    def _row_params(self, row_idx):
        """A row's params dict: its params_json column, else (the lean boot, no such column) fetched on demand,
        cached and bounded (_fetch_params_row); None when neither has it."""
        if 'params_json' in self.df.columns:
            pjson = self.df.at[row_idx, 'params_json']
            try:
                return orjson.loads(pjson) if isinstance(pjson, str) and pjson.strip() else None
            except (ValueError, TypeError):
                return None
        return self._fetch_params_row(row_idx)

    def _fetch_params_row(self, row_idx):
        """Lazily fetch one row's params dict when the params_json column is absent.

        Uses fetch_params_json (predicate-pushdown read of the source parquet,
        the same reader the weight tools use) with this row's config path,
        resolved from the row's L_out_path via self._config_by_lout. Result is
        cached per row index (bounded); returns None on any failure so callers
        degrade gracefully (hover drops a line, export shows a warning).
        """
        if getattr(self, '_params_row_cache', None) is None:
            self._params_row_cache = {}
        cache = self._params_row_cache
        if row_idx in cache:
            return cache[row_idx]
        try:
            lout = self.df.at[row_idx, 'L_out_path']
            lout = str(lout) if pd.notna(lout) else ''
            cp = self._config_by_lout.get(lout) if lout else None
            if cp is None:
                cache[row_idx] = None
                return None
            pairs = [(int(self.df.at[row_idx, 'i']), int(self.df.at[row_idx, 'j']))]
            rows = fetch_params_json([cp], ij_pairs=pairs)
            if rows.empty:
                out = None
            else:
                pjson = rows.iloc[0]['params_json']
                out = (orjson.loads(pjson)
                       if isinstance(pjson, str) and pjson.strip() else None)
        except Exception:
            out = None
        if len(cache) >= 128:
            cache.clear()
        cache[row_idx] = out
        return out

    def _reconstruct_figure(self, view_data, theme_override=None):
        """Rebuild a figure from a saved view dict without a browser. Keys the view leaves out take the controls'
        defaults, as loading it on the live page does (the axes and theme excepted)."""
        view_data = {**{k: v for k, v in self._view_defaults.items() if k not in ('x-axis', 'y-axis', 'theme-toggle')},
                     **view_data}
        theme_val = theme_override if theme_override is not None else (view_data.get('theme-toggle') or [])
        theme     = resolve_theme(theme_val)
        x_col = view_data.get('x-axis')
        _y_raw = view_data.get('y-axis')
        y_cols = ([_y_raw] if isinstance(_y_raw, str) else list(_y_raw or []))
        _primary_y = y_cols[0] if y_cols else None
        if not x_col or not y_cols:
            fig = go.Figure()
            fig.update_layout(title='View has no axes configured', template=theme.template)
            return fig

        _extras = view_data.get('_extras') or {}

        # 1–4. View filters via the shared, order-critical helper (§3). A failing
        # filter-expr keeps the step-1..3 result — the error was previously
        # computed by the pipeline and discarded here, so behaviour is unchanged.
        try:
            filtered_df, _factor_filters = apply_view_filters(self.df, view_data)
        except ViewDataMismatch:
            raise   # the export route answers 409: no figure beats a wrong one
        except Exception:
            filtered_df, _factor_filters = apply_view_filters(
                self.df, {**view_data, 'filter-expr': ''})

        # 5–6. Binning → aggregation (shared pipeline; the query ran above — the
        # CLI calls apply_view_filters directly and skips this section entirely,
        # since it needs raw rows with i/j)
        color_factor = view_data.get('color-by-factor') or []
        _agg_y_mode  = view_data.get('agg-y-mode') or 'raw'
        _agg_group_by = view_data.get('agg-group-by')
        _pipeline_kwargs = dict(
            filter_expr='',
            x_bin_width=view_data.get('x-bin-width') or 1,
            agg_group_by=_agg_group_by,
            agg_y_mode=_agg_y_mode,
            color_factor=color_factor,
            factor_filters=_factor_filters,
        )
        if len(y_cols) == 1:
            filtered_df, x_plot_col, x_bin_label, y_agg_label, _ = self._apply_data_pipeline(
                filtered_df, x_col, _primary_y, **_pipeline_kwargs,
            )
        else:
            filtered_df, x_plot_col, x_bin_label, _, _ = self._apply_data_pipeline(
                filtered_df, x_col, _primary_y,
                **{**_pipeline_kwargs, 'agg_group_by': None, 'agg_y_mode': 'raw'},
            )
            y_agg_label = None

        # 7–8. Build figure via the step shared with the live render
        return self._view_figure(filtered_df, {**view_data, 'y-axis': y_cols}, theme=theme,
                                 factor_filters=_factor_filters, x_plot_col=x_plot_col,
                                 x_bin_label=x_bin_label, y_agg_label=y_agg_label,
                                 viewport=_extras.get('viewport') or view_data.get('_viewport'))

    def _view_figure(self, filtered_df, view, *, theme, factor_filters, x_plot_col, x_bin_label, y_agg_label,
                     viewport=None):
        """The build step shared by the live render and the export: rows that went through the view's filters and
        aggregation, and the view itself (control id -> value, as saved; '_extras' holds color_offsets,
        trace_visibility and the figure block), to the figure. viewport ({xaxis, yaxis} ranges) comes before the
        figure block, so a range the figure block sets wins, live as in the export."""
        x_col, y_cols = view['x-axis'], view['y-axis']
        extras = view.get('_extras') or {}
        color_cols = [f for f in (view.get('color-by-factor') or []) if f in filtered_df.columns]
        figure = extras.get('figure')
        heatmap = self._heatmap_spec(view, x_col, y_cols)
        if heatmap:   # the Heatmap control, drawn by the figure options' bin2d
            figure = {**(figure or {}), 'bin2d': heatmap}
        options = resolve_figure_options(figure, self._figure_presets, x_col, y_cols,
                                         color_col=color_cols[0] if len(color_cols) == 1 else None)

        def build(rows):
            return self._build_panel(rows, view, theme=theme, factor_filters=factor_filters, x_plot_col=x_plot_col,
                                     x_bin_label=x_bin_label, y_agg_label=y_agg_label)

        facet = (options or {}).get('facet')
        if facet and any(facet.get(k) in filtered_df.columns for k in ('row', 'col')):
            fig = self._facet_figure(filtered_df, facet, build, theme, options.get('series_labels'))
        else:
            fig = build(filtered_df)
        viewport = viewport or {}
        if viewport.get('xaxis'):
            fig.update_xaxes(range=viewport['xaxis'])
        if viewport.get('yaxis'):
            fig.update_yaxes(range=viewport['yaxis'])
        self._apply_trace_vis(fig, extras.get('trace_visibility'))
        apply_figure_options(fig, options, rows=filtered_df)
        return fig

    def _heatmap_spec(self, view, x_col, y_cols):
        """The Heatmap control as a bin2d spec (apply_figure_options): the shown points binned on x and the first y
        (bin widths heatmap-dx, heatmap-dy; a bin with fewer than heatmap-min-n points blank), coloured by the median
        of the continuous colour column. None when off or the colour is not one continuous column."""
        if 'on' not in (view.get('heatmap-toggle') or []):
            return None
        color = self._build_color_cfg(view.get('color-by-factor'), None, color_values=None, color_scale=None,
                                      color_scale_reverse=None, as_continuous=view.get('color-as-continuous'))
        if not color or color['type'] != 'continuous':
            return None
        sizes = {'dx': view.get('heatmap-dx'), 'dy': view.get('heatmap-dy'), 'min_n': view.get('heatmap-min-n')}
        return {'x': x_col, 'y': y_cols[0], 'z': color['column'], **{k: v for k, v in sizes.items() if v}}

    @staticmethod
    def _facet_figure(filtered_df, facet, build, theme, series_labels=None):
        """One panel per value of facet['row'] x facet['col'] (either may be absent), each drawn by build(panel_df),
        on one subplot grid. Levels come in table order unless facet['order'] lists them ({column: [values]}).
        facet['title']: a panel title, formatted with the panel's first row ('{dims} {stage}'; a value that is a series
        of the presets as its label, series_labels), on every panel or
        with facet['title_on'] = 'top' only on the top row. facet['share']: {x, y} shared ranges (default both;
        y tick labels then only on the first column, x tick labels only on the bottom row). facet['spacing']: [horizontal, vertical] (paper fractions).
        The panels' axis references go into layout.meta['facet'] for the figure options (facet_cells)."""
        from plotly.subplots import make_subplots

        def levels(col):
            if not col:
                return [None]
            seen = list(dict.fromkeys(filtered_df[col].astype(str)))
            wanted = [str(v) for v in (facet.get('order') or {}).get(col) or []]
            return [v for v in wanted if v in seen] + [v for v in seen if v not in wanted]

        row_col, col_col = facet.get('row'), facet.get('col')
        rows, cols = levels(row_col), levels(col_col)
        h, v = facet.get('spacing') or (0.2 / len(cols), 0.3 / len(rows))
        fig = make_subplots(rows=len(rows), cols=len(cols), horizontal_spacing=h, vertical_spacing=v)
        share = {'x': True, 'y': True, **(facet.get('share') or {})}
        cells, panel_fig = [], None
        for i, rv in enumerate(rows):
            for j, cv in enumerate(cols):
                mask = np.ones(len(filtered_df), dtype=bool)
                if row_col:
                    mask &= (filtered_df[row_col].astype(str) == rv).to_numpy()
                if col_col:
                    mask &= (filtered_df[col_col].astype(str) == cv).to_numpy()
                sp = fig.get_subplot(i + 1, j + 1)
                xref, yref = sp.xaxis.plotly_name.replace('axis', ''), sp.yaxis.plotly_name.replace('axis', '')
                cells.append({'row': i, 'col': j, 'rv': rv, 'cv': cv, 'x': xref, 'y': yref})
                panel = filtered_df[mask]
                if panel.empty:
                    continue
                panel_fig = build(panel)
                for t in panel_fig.data:
                    fig.add_trace(t, row=i + 1, col=j + 1)
                for src, dst in ((panel_fig.layout.xaxis, sp.xaxis), (panel_fig.layout.yaxis, sp.yaxis)):
                    dst.update(title_text=src.title.text, range=src.range, type=src.type,
                               tickmode=src.tickmode, tickvals=src.tickvals, dtick=src.dtick, tick0=src.tick0)
                if facet.get('title') and (facet.get('title_on') != 'top' or i == 0):
                    first = {k: (series_labels or {}).get(str(v), v) for k, v in panel.iloc[0].to_dict().items()}
                    fig.add_annotation(text=facet['title'].format(**first), showarrow=False, font_size=_FONT_BASE_PX,
                                       xref=f'{xref} domain', yref=f'{yref} domain', x=0, y=1,
                                       xanchor='left', yanchor='bottom', name='facet_title')
        if panel_fig is None:
            return go.Figure(layout=dict(title='No data for selected filters', template=theme.template))
        for c in cells:   # axis titles once: x on the bottom row, y on the first column
            if c['row'] != len(rows) - 1:
                fig.layout[f'xaxis{c["x"][1:]}'].title.text = ''
                if share['x']:
                    fig.layout[f'xaxis{c["x"][1:]}'].showticklabels = False
            if c['col'] != 0:
                fig.layout[f'yaxis{c["y"][1:]}'].title.text = ''
                if share['y']:
                    fig.layout[f'yaxis{c["y"][1:]}'].showticklabels = False
        if share['x']:
            fig.update_xaxes(matches='x')
            fig.layout.xaxis.matches = None
        if share['y']:
            fig.update_yaxes(matches='y')
            fig.layout.yaxis.matches = None
        layout = {k: v for k, v in panel_fig.layout.to_plotly_json().items()
                  if not k.startswith(('xaxis', 'yaxis')) and k not in ('annotations', 'shapes', 'meta')}
        fig.update_layout(layout)
        fig.update_layout(meta={'facet': {'row': row_col, 'col': col_col, 'cells': cells}})
        return fig

    def _build_panel(self, filtered_df, view, *, theme, factor_filters, x_plot_col, x_bin_label, y_agg_label):
        """One plot of the view's rows, its settings read from the view (control id -> value, as saved): the
        single-y figure or the multi-y overlay. _view_figure calls it once, or per panel when the view facets."""
        def on(fid, default=()):
            return 'on' in (view.get(fid, default) or [])

        def jitter(axis):
            lo, hi = view.get(f'{axis}-jitter-min'), view.get(f'{axis}-jitter-max')
            return (float(lo), float(hi)) if on(f'{axis}-jitter-toggle') and lo is not None and hi is not None else None

        def setting(fid, default):
            v = view.get(fid)
            return default if v is None else v

        x_col, y_cols = view['x-axis'], view['y-axis']
        color_factor = view.get('color-by-factor') or []
        color_cfg = self._build_color_cfg(
            color_factor, factor_filters,
            color_values=view.get('color-by-values') or [],
            color_scale=view.get('color-scale-picker') or 'Viridis',
            color_scale_reverse=view.get('color-scale-reverse') or [],
            as_continuous=view.get('color-as-continuous') or [],
        )
        _figure_kwargs = dict(
            color_cfg=color_cfg, color_offsets=(view.get('_extras') or {}).get('color_offsets'),
            x_jitter=jitter('x'), y_jitter=jitter('y'),
            marker_size=setting('setting-marker-size', 5),
            marker_opacity=setting('setting-marker-opacity', 0.7),
            overlay_dots=on('overlay-dots-toggle', ['on']),
            overlay_line=on('overlay-line-toggle'),
            overlay_ks=_get_overlay_ks(view),
            anchor_left=on('overlay-anchor-left'),
            anchor_right=on('overlay-anchor-right'),
            theme=theme,
            show_factor_name=on('legend-show-factor-name', ['on']),
            skip_sparse='on' in (view.get('skip-sparse-factors') or ['on']),
            line_width=setting('overlay-line-width', 2),
            line_opacity=setting('overlay-line-opacity', 1.0),
        )
        if len(y_cols) == 1:
            return self._build_figure(
                filtered_df, x_plot_col, y_cols[0],
                y_agg_label=y_agg_label, x_label=x_bin_label,
                **_figure_kwargs,
            )
        return self._build_overlay_figure(
            filtered_df, x_col, y_cols,
            filter_expr='',
            x_bin_width=view.get('x-bin-width') or 1,
            agg_group_by=view.get('agg-group-by'), agg_y_mode=view.get('agg-y-mode') or 'raw',
            color_factor=color_factor, factor_filters=factor_filters,
            y_styles=view.get('overlay-y-styles') or [],
            **_figure_kwargs,
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _register_callbacks(self):
        # Capture shared state into locals so callbacks are plain closures
        df                    = self.df
        discrete_factors      = self.discrete_factors
        factors_subset        = self.factors_subset
        factor_values         = self.factor_values
        axis_options          = self.axis_options
        continuous_factors    = self.continuous_factors
        numeric_discrete_factors = self.numeric_discrete_factors
        has_params_json       = self.has_params_json
        initial_display_points = self.initial_display_points
        color_by_options      = self.color_by_options
        _safe_roots           = self._safe_roots
        app                   = self.app

        # ── Derived-factor helper closures ────────────────────────────────────
        def _get_factor_values_dict(derived):
            combined = dict(factor_values)
            for entry in (derived or []):
                combined[entry['col']] = entry['values']
            return combined

        def _get_all_discrete_factors(derived):
            return discrete_factors + [entry['col'] for entry in (derived or [])]

        def _build_color_by_options(derived):
            opts = list(color_by_options)
            derived_cols = [entry['col'] for entry in (derived or [])]
            if derived_cols:
                opts.append({'label': '\u2500\u2500 Derived \u2500\u2500', 'value': '_header_derived', 'disabled': True})
                for col in derived_cols:
                    opts.append({'label': f'  {col}', 'value': col})
            return opts

        def _build_agg_options(derived):
            return [{'label': f, 'value': f} for f in _get_all_discrete_factors(derived)]

        # ── Collapsible panel toggles ─────────────────────────────────────────
        _panel_ids = ['controls', 'custom', 'appearance', 'views', 'settings']
        _panel_labels = {
            'controls':   'Plot',
            'custom':     'Filters',
            'appearance': 'Appearance',
            'views':      'Views',
            'settings':   '\u2699 Settings',
        }

        def _register_panel_toggle(panel_id, label):
            @callback(
                Output(f'{panel_id}-body',   'style',    allow_duplicate=True),
                Output(f'{panel_id}-toggle', 'children', allow_duplicate=True),
                Input(f'{panel_id}-toggle', 'n_clicks'),
                State(f'{panel_id}-body', 'style'),
                prevent_initial_call=True,
            )
            def toggle(n_clicks, current_style):
                hidden = (current_style or {}).get('display') == 'none'
                if hidden:
                    return {'display': 'block'}, f'\u25bc {label}'
                return {'display': 'none'}, f'\u25b6 {label}'

        for pid, lbl in _panel_labels.items():
            _register_panel_toggle(pid, lbl)

        # Collapse all panels when ?expand=false is in the URL
        @callback(
            *[Output(f'{pid}-body',   'style',    allow_duplicate=True) for pid in _panel_ids],
            *[Output(f'{pid}-toggle', 'children', allow_duplicate=True) for pid in _panel_ids],
            Input('url', 'search'),
            prevent_initial_call='initial_duplicate',
        )
        def _init_panel_expand(search):
            raw = (search or '').lstrip('?').lower()
            params = {k: v for part in raw.split('&') if '=' in part
                      for k, v in [part.split('=', 1)]}
            if params.get('expand') == 'false':
                closed = {'display': 'none'}
                return (
                    *[closed for _ in _panel_ids],
                    *[f'\u25b6 {_panel_labels[pid]}' for pid in _panel_ids],
                )
            raise PreventUpdate

        # ── View save/load ────────────────────────────────────────────────────
        import json

        @callback(
            Output('view-status',       'children', allow_duplicate=True),
            Output('view-picker',       'options',  allow_duplicate=True),
            Output('active-view-name',  'data',     allow_duplicate=True),
            Output('views-dir',         'value',    allow_duplicate=True),
            Input('view-save-btn', 'n_clicks'),
            State('view-name',  'value'),
            State('views-dir',  'value'),
            *[State(fid, prop) for fid, prop in VIEW_FIELDS],
            State({'type': 'dropdown', 'factor': ALL}, 'value'),
            State('color-offsets',          'data'),
            State('live-trace-visibility',  'data'),
            State('scatter-plot',           'figure'),
            admin_token_state(),
            prevent_initial_call=True,
        )
        def save_view(n_clicks, view_name, views_dir_str, *args):
            if not view_name or not view_name.strip():
                return '\u26a0 Enter a view name first.', no_update, no_update, no_update
            view_name = view_name.strip()
            vdir = _views_dir(views_dir_str)
            n_fields  = len(VIEW_FIELDS)
            # admin-token is the last State — see the callback signature above.
            blocked = guard_view_write(vdir, self._initial_views_dir, _safe_roots,
                                       args[n_fields + 4], 4)
            if blocked is not None:
                return blocked
            view_data = {fid: val for (fid, _), val in zip(VIEW_FIELDS, args[:n_fields])}

            discrete_factors_set = set(discrete_factors)
            static_filters, derived_filters = {}, {}
            for item in ctx.states_list[2 + n_fields]:
                factor = item['id']['factor']
                val = item.get('value') or []
                (static_filters if factor in discrete_factors_set else derived_filters)[factor] = val
            color_offsets     = args[n_fields + 1] or {}
            # live-trace-visibility reflects actual browser state (legend clicks included),
            # unlike State('scatter-plot', 'figure') which only has the last server-sent figure.
            live_trace_vis    = args[n_fields + 2] or {}
            figure            = args[n_fields + 3] or {}
            # a trace the figure block renamed keeps its generated name (the one colours and
            # visibility are keyed by) in meta.name0 (utils._rename)
            traces            = [{**t, 'name': t['meta'].get('name0', t.get('name')),
                                  'mode': t['meta'].get('mode0', t.get('mode'))}
                                 if isinstance(t.get('meta'), dict) else t
                                 for t in figure.get('data') or []]
            visible_names     = {t['name'] for t in traces if t.get('name')}
            # Use live visibility; fall back to figure data for any missing entries.
            trace_visibility  = {
                f"{t.get('name', '')}|{t.get('mode') or 'markers'}":
                    live_trace_vis.get(
                        f"{t.get('name', '')}|{t.get('mode') or 'markers'}",
                        t.get('visible', True)
                    )
                for t in traces if t.get('name')
            }
            layout = figure.get('layout') or {}
            xr = (layout.get('xaxis') or {}).get('range')
            yr = (layout.get('yaxis') or {}).get('range')
            view_data['_extras'] = {
                'factor_filters':         static_filters,
                'derived_filter_filters': derived_filters,
                'color_offsets':          {k: v for k, v in color_offsets.items()
                                           if k in visible_names
                                           or (k.endswith('::line') and k[:-6] in visible_names)},
                'viewport':               {'xaxis': xr, 'yaxis': yr} if (xr or yr) else None,
                'trace_visibility':       trace_visibility,
            }
            # the figure block (paper labels, legend placement) is edited in the JSON, never in the UI:
            # carry it over, or a browser save silently drops it
            prev_figure = ((load_view_json(view_name, vdir) or {}).get('_extras') or {}).get('figure')
            if prev_figure:
                view_data['_extras']['figure'] = prev_figure

            save_view_json(view_name, view_data, vdir)
            return f'\u2713 Saved "{view_name}"', _view_dropdown_options(vdir), view_name, str(vdir)

        def _load_view_data(view_name, views_dir_str, trigger):
            """Shared helper: load view file and build callback outputs list."""
            view_data = load_view_json(view_name, _views_dir(views_dir_str))
            if view_data is None:
                fail = *([no_update] * (len(VIEW_FIELDS) + len(discrete_factors))), \
                       f'\u26a0 View "{view_name}" not found.', *([no_update] * 9)
                return fail
            extras = view_data.get('_extras', {})
            # backwards compat: old saves used top-level keys
            factor_filters = extras.get('factor_filters') or view_data.get('_factor_filters', {})
            # Theme is a user preference, not view state: it is still *written* to the
            # JSON (no key changes shape, and the export route still reads it), but
            # loading a view must never stomp the theme the user is looking at.
            # a key the view leaves out loads as the control's default, not as what the last view left
            defaults = self._view_defaults
            outputs = [no_update if fid == 'theme-toggle' else view_data.get(fid, defaults.get(fid, no_update))
                       for fid, _ in VIEW_FIELDS]
            # a factor the view does not filter loads unfiltered (as the export reads it), not as the last view left it
            outputs += [factor_filters.get(f, []) for f in discrete_factors]
            missing = missing_filter_values(self.df, factor_filters)
            outputs += [
                f'\u26a0 Loaded "{view_name}", but its filters name values this app has no data for '
                f'(restart the app if the grids finished after it booted): {missing}' if missing
                else f'\u2713 Loaded "{view_name}"',
                view_name,
                (trigger or 0) + 1,
                extras.get('derived_filter_filters') or view_data.get('_derived_filter_filters', {}),
                extras.get('color_offsets') or view_data.get('_color_offsets', {}),
                extras.get('viewport') or view_data.get('_viewport', None),
                extras.get('trace_visibility') or None,
                extras.get('figure'),
            ]
            axis_opts = self._view_axis_options(view_data.get('fields-expr'))
            outputs += [axis_opts, axis_opts]
            return tuple(outputs)

        @callback(
            *[Output(fid, prop, allow_duplicate=True) for fid, prop in VIEW_FIELDS],
            *[Output({'type': 'dropdown', 'factor': f}, 'value', allow_duplicate=True) for f in discrete_factors],
            Output('view-status',           'children',   allow_duplicate=True),
            Output('active-view-name',      'data',       allow_duplicate=True),
            Output('process-trigger',       'data',       allow_duplicate=True),
            Output('pending-derived-filters', 'data',     allow_duplicate=True),
            Output('color-offsets',           'data',       allow_duplicate=True),
            Output('saved-viewport',          'data',       allow_duplicate=True),
            Output('saved-trace-visibility',  'data',       allow_duplicate=True),
            Output('view-figure',             'data',       allow_duplicate=True),
            Output('x-axis',                  'options',    allow_duplicate=True),
            Output('y-axis',                  'options',    allow_duplicate=True),
            Output('views-dir',               'value',      allow_duplicate=True),
            Input('view-picker', 'value'),
            State('views-dir',        'value'),
            State('process-trigger',  'data'),
            prevent_initial_call=True,
        )
        def load_view(view_name, views_dir_str, trigger):
            if not view_name:
                raise PreventUpdate
            vdir = _views_dir(views_dir_str) if views_dir_str and views_dir_str.strip() \
                else Path(self._initial_views_dir)
            actual_dir = str(vdir) if (vdir / f'{view_name}.json').exists() else no_update
            return (*_load_view_data(view_name, str(vdir), trigger), actual_dir)

        @callback(
            Output('view-status',      'children', allow_duplicate=True),
            Output('view-picker',      'options',  allow_duplicate=True),
            Output('view-picker',      'value',    allow_duplicate=True),
            Output('active-view-name', 'data',     allow_duplicate=True),
            Input('view-delete-btn', 'n_clicks'),
            State('view-picker', 'value'),
            State('views-dir',   'value'),
            admin_token_state(),
            prevent_initial_call=True,
        )
        def delete_view(n_clicks, view_name, views_dir_str, admin_token):
            if not view_name:
                return '\u26a0 Select a view to delete.', no_update, no_update, no_update
            vdir = _views_dir(views_dir_str)
            blocked = guard_view_write(vdir, self._initial_views_dir, _safe_roots,
                                       admin_token, 4)
            if blocked is not None:
                return blocked
            delete_view_json(view_name, vdir)
            return f'\u2713 Deleted "{view_name}"', _view_dropdown_options(vdir), None, None

        @callback(
            *[Output(fid, prop, allow_duplicate=True) for fid, prop in VIEW_FIELDS],
            *[Output({'type': 'dropdown', 'factor': f}, 'value', allow_duplicate=True) for f in discrete_factors],
            Output('view-status',           'children',   allow_duplicate=True),
            Output('active-view-name',      'data',       allow_duplicate=True),
            Output('process-trigger',       'data',       allow_duplicate=True),
            Output('pending-derived-filters', 'data',     allow_duplicate=True),
            Output('color-offsets',           'data',       allow_duplicate=True),
            Output('saved-viewport',          'data',       allow_duplicate=True),
            Output('saved-trace-visibility',  'data',       allow_duplicate=True),
            Output('view-figure',             'data',       allow_duplicate=True),
            Output('x-axis',                  'options',    allow_duplicate=True),
            Output('y-axis',                  'options',    allow_duplicate=True),
            Output('views-dir',               'value',      allow_duplicate=True),
            Input('url', 'search'),
            State('views-dir',        'value'),
            State('process-trigger',  'data'),
            State('active-view-name', 'data'),
            prevent_initial_call='initial_duplicate',
        )
        def load_view_from_url(search, views_dir_str, trigger, active_name):
            view_name = view_name_from_query(search)
            if not view_name:
                raise PreventUpdate
            if view_name == active_name:
                raise PreventUpdate
            vdir = _views_dir(views_dir_str) if views_dir_str and views_dir_str.strip() \
                else Path(self._initial_views_dir)
            actual_dir = str(vdir) if (vdir / f'{view_name}.json').exists() else no_update
            return (*_load_view_data(view_name, str(vdir), trigger), actual_dir)

        # Update URL without reload when a view is activated
        app.clientside_callback(
            """
            function(view_name) {
                if (!view_name) return window.dash_clientside.no_update;
                window.history.replaceState(null, '', window.location.pathname + '?' + view_name);
                return window.dash_clientside.no_update;
            }
            """,
            Output('active-view-name', 'data', allow_duplicate=True),
            Input('active-view-name', 'data'),
            prevent_initial_call=True,
        )

        @callback(
            Output('view-name', 'value'),
            Input('active-view-name', 'data'),
        )
        def sync_view_name(name):
            return name or ''

        # ── Lazy mode toggle ──────────────────────────────────────────────────
        @callback(
            Output('lazy-mode-active',   'data'),
            Output('lazy-mode-toggle',   'children'),
            Output('lazy-mode-toggle',   'style'),
            Input('lazy-mode-toggle',    'n_clicks'),
            State('lazy-mode-active',    'data'),
            prevent_initial_call=True,
        )
        def toggle_lazy_mode(n_clicks, is_active):
            new_state = not is_active
            style = {
                'padding': '6px 12px', 'fontSize': '12px', 'cursor': 'pointer',
                'border': f'2px solid {_BORDER}', 'borderRadius': '4px', 'color': _TEXT,
                'background': _ACCENT if new_state else _CONTROL,
                'fontWeight': 'bold' if new_state else 'normal', 'marginRight': '8px',
            }
            return new_state, f'Lazy Mode ({"ON" if new_state else "OFF"})', style

        # ── Overlay spline controls visibility ───────────────────────────────
        @callback(
            Output('overlay-spline-controls',    'style'),
            Output('overlay-spline-per-line',    'style'),
            Output('overlay-per-line-container', 'style'),
            Output('overlay-anchor-controls',    'style'),
            Input('overlay-line-toggle',         'value'),
            Input('overlay-per-line-toggle',     'value'),
        )
        def toggle_spline_visibility(line_on, per_line_on):
            if not (line_on and 'on' in line_on):
                return ({'display': 'none'}, {'display': 'none'},
                        {'display': 'none'}, {'display': 'none'})
            per_line_active = per_line_on and 'on' in per_line_on
            controls_style  = ({'display': 'none'} if per_line_active
                               else {'display': 'flex', 'alignItems': 'center'})
            per_line_style  = ({'display': 'flex', 'flexWrap': 'wrap', 'gap': '4px'}
                               if per_line_active else {'display': 'none'})
            anchor_style    = {'display': 'flex', 'alignItems': 'center'}
            return controls_style, per_line_style, {'display': 'inline-block'}, anchor_style

        # ── Build per-line k/s inputs when per-line mode is active ────────────
        @callback(
            Output('overlay-spline-per-line', 'children'),
            Output('overlay-spline-ks',       'data', allow_duplicate=True),
            Input('overlay-line-toggle',      'value'),
            Input('overlay-per-line-toggle',  'value'),
            Input('scatter-plot',             'figure'),
            State('overlay-spline-ks',        'data'),
            prevent_initial_call=True,
        )
        def build_per_line_inputs(line_on, per_line_on, figure, ks_data):
            if not (line_on and 'on' in line_on and per_line_on and 'on' in per_line_on):
                return no_update, no_update
            if not figure or not figure.get('data'):
                return [], no_update
            line_traces = [t for t in figure['data'] if (t.get('mode') or '') == 'lines']
            n = max(1, len(line_traces))
            old = list(ks_data or [])
            new_ks = old[:n] + [[1, None]] * max(0, n - len(old))
            children = []
            for i, (k, s) in enumerate(new_ks):
                label = (line_traces[i].get('name', '') if i < len(line_traces) else '') or str(i)
                children.append(html.Div([
                    html.Span(label, style={'color': _TEXT_DIM, 'fontSize': '11px',
                                            'fontFamily': 'monospace', 'marginRight': '3px',
                                            'maxWidth': '120px', 'overflow': 'hidden',
                                            'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'}),
                    html.Span('k:', style={'fontSize': '11px', 'color': _TEXT_DIM,
                                          'marginRight': '2px'}),
                    dcc.Input(id={'type': 'spline-k', 'index': i}, type='number',
                              value=k, min=1, max=5, step=1,
                              style={'width': '34px', 'fontSize': '11px', 'padding': '2px 3px',
                                     'background': _PANEL, 'color': _TEXT,
                                     'border': f'1px solid {_BORDER}', 'borderRadius': '3px'}),
                    html.Span('s:', style={'fontSize': '11px', 'color': _TEXT_DIM,
                                          'margin': '0 2px 0 5px'}),
                    dcc.Input(id={'type': 'spline-s', 'index': i}, type='number',
                              value=s, min=0, step='any', placeholder='auto',
                              style={'width': '46px', 'fontSize': '11px', 'padding': '2px 3px',
                                     'background': _PANEL, 'color': _TEXT,
                                     'border': f'1px solid {_BORDER}', 'borderRadius': '3px'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '6px'}))
            store_out = new_ks if new_ks != old else no_update
            return children, store_out

        # ── Collect per-line k/s inputs → store ──────────────────────────────
        @callback(
            Output('overlay-spline-ks', 'data', allow_duplicate=True),
            Input({'type': 'spline-k', 'index': ALL}, 'value'),
            Input({'type': 'spline-s', 'index': ALL}, 'value'),
            prevent_initial_call=True,
        )
        def collect_per_line_ks(k_vals, s_vals):
            if not k_vals:
                raise PreventUpdate
            return [[k if k is not None else 1, s] for k, s in zip(k_vals, s_vals)]

        # ── Per-Y style (symbol + dash) ───────────────────────────────────────
        _SYMBOL_OPTIONS = [{'label': s, 'value': s}
                           for s in ['circle', 'square', 'diamond', 'cross', 'x',
                                     'triangle-up', 'triangle-down', 'star', 'circle-open', 'diamond-open']]
        _DASH_OPTIONS   = [{'label': d, 'value': d}
                           for d in ['solid', 'dash', 'dot', 'dashdot', 'longdash']]
        _dd_style = {'width': '100px', 'fontSize': '11px',
                     'background': _PANEL, 'color': _TEXT}

        def _with_value(options, value):
            return options if any(o['value'] == value for o in options) else options + [{'label': value, 'value': value}]

        @callback(
            Output('overlay-y-styles-container', 'style'),
            Input('y-axis', 'value'),
        )
        def toggle_y_styles_visibility(y_val):
            y_cols = [y_val] if isinstance(y_val, str) else list(y_val or [])
            if len(y_cols) > 1:
                return {'display': 'flex', 'flexDirection': 'column', 'gap': '6px',
                        'marginBottom': '12px'}
            return {'display': 'none'}

        @callback(
            Output('overlay-y-styles-container', 'children'),
            Output('overlay-y-styles', 'data', allow_duplicate=True),
            Input('y-axis', 'value'),
            State('overlay-y-styles', 'data'),
            prevent_initial_call=True,
        )
        def build_y_style_inputs(y_val, styles_data):
            y_cols = [y_val] if isinstance(y_val, str) else list(y_val or [])
            if len(y_cols) <= 1:
                return no_update, no_update
            old = list(styles_data or [])
            new_styles = (old + [None] * len(y_cols))[:len(y_cols)]
            children = []
            for i, y in enumerate(y_cols):
                saved   = new_styles[i] or {}
                sym_val  = saved.get('symbol', _Y_STYLES[i % len(_Y_STYLES)]['symbol'])
                dash_val = saved.get('dash',   _Y_STYLES[i % len(_Y_STYLES)]['dash'])
                children.append(html.Div([
                    html.Span(y, style={'color': _TEXT_DIM, 'fontSize': '11px',
                                        'fontFamily': 'monospace', 'marginRight': '4px',
                                        'maxWidth': '80px', 'overflow': 'hidden',
                                        'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'}),
                    # a view's own style is offered even when not listed (e.g. 'diamond-open'): Dash 4's
                    # dropdown drops a value that is not among its options, and collect_y_styles saves the loss
                    dcc.Dropdown(id={'type': 'y-symbol', 'index': i},
                                 options=_with_value(_SYMBOL_OPTIONS, sym_val), value=sym_val,
                                 clearable=False, style=_dd_style),
                    dcc.Dropdown(id={'type': 'y-dash', 'index': i},
                                 options=_with_value(_DASH_OPTIONS, dash_val), value=dash_val,
                                 clearable=False, style={**_dd_style, 'marginLeft': '4px'}),
                ], style={'display': 'flex', 'alignItems': 'center'}))
            store_out = new_styles if new_styles != old else no_update
            return children, store_out

        @callback(
            Output('overlay-y-styles', 'data', allow_duplicate=True),
            Input({'type': 'y-symbol', 'index': ALL}, 'value'),
            Input({'type': 'y-dash',   'index': ALL}, 'value'),
            prevent_initial_call=True,
        )
        def collect_y_styles(symbols, dashes):
            if not symbols:
                raise PreventUpdate
            return [{'symbol': sym or 'circle', 'dash': dash or 'solid'}
                    for sym, dash in zip(symbols, dashes)]

        # ── Factor dropdown select-all / clear-all ────────────────────────────
        @callback(
            Output({'type': 'dropdown', 'factor': MATCH}, 'value'),
            Input({'type': 'select-all', 'factor': MATCH}, 'n_clicks'),
            Input({'type': 'clear-all',  'factor': MATCH}, 'n_clicks'),
            State({'type': 'dropdown',   'factor': MATCH}, 'id'),
            State('derived-factors', 'data'),
            prevent_initial_call=True,
        )
        def update_dropdown_selection(select_clicks, clear_clicks, dropdown_id, derived):
            triggered_id = ctx.triggered_id
            if triggered_id is None:
                raise PreventUpdate
            factor = triggered_id['factor']
            if triggered_id['type'] == 'select-all':
                return _get_factor_values_dict(derived).get(factor, [])
            return []

        @callback(
            Output('agg-group-by', 'value'),
            Input('agg-group-select-all', 'n_clicks'),
            Input('agg-group-clear-all',  'n_clicks'),
            State('derived-factors', 'data'),
            prevent_initial_call=True,
        )
        def update_agg_group_selection(select_clicks, clear_clicks, derived):
            if ctx.triggered_id == 'agg-group-select-all':
                return _get_all_discrete_factors(derived)
            return []

        # ── Bin width visibility — only shown when aggregation is active ─────
        @callback(
            Output('x-bin-width-container', 'style'),
            Input('agg-y-mode', 'value'),
        )
        def toggle_bin_width_visibility(agg_mode):
            if agg_mode and agg_mode != 'raw':
                return {'width': '10%', 'display': 'inline-block', 'verticalAlign': 'top'}
            return {'display': 'none'}

        # ── Color control visibility ──────────────────────────────────────────
        @callback(
            Output('color-by-values',              'options'),
            Output('color-by-values',              'value'),
            Output('color-values-container',       'style'),
            Output('color-scale-container',        'style'),
            Output('color-continuous-toggle-container', 'style'),
            Output('heatmap-container',            'style'),
            Input('color-by-factor',  'value'),
            Input('color-as-continuous', 'value'),
            State('derived-factors',  'data'),
        )
        def update_color_controls(color_factor, as_continuous, derived):
            show_vals  = {'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}
            hide_vals  = {'width': '18%', 'display': 'none',         'marginRight': '2%'}
            show_scale = {'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}
            hide_scale = {'width': '18%', 'display': 'none',         'marginRight': '2%'}
            show_toggle = {'display': 'inline-block', 'width': '9%', 'marginRight': '1%', 'verticalAlign': 'bottom'}
            hide_toggle = {'display': 'none',         'width': '9%', 'marginRight': '1%', 'verticalAlign': 'bottom'}
            show_heat = {'width': '22%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}
            hide_heat = {'width': '22%', 'display': 'none',         'marginRight': '2%'}

            continuous_override = as_continuous and 'on' in as_continuous
            fv = _get_factor_values_dict(derived)
            derived_numeric = {entry['col'] for entry in (derived or []) if entry.get('is_numeric')}

            factors = [f for f in (color_factor or []) if f and f != SELECTION_VALUE]
            if len(factors) == 1:
                f = factors[0]
                if f in continuous_factors:
                    return [], [], hide_vals, show_scale, hide_toggle, show_heat
                is_numeric = f in numeric_discrete_factors or f in derived_numeric
                vals = fv.get(f, [])
                opts = [{'label': v, 'value': v} for v in vals]
                if is_numeric and continuous_override:
                    return opts, vals, hide_vals, show_scale, show_toggle, show_heat
                return opts, vals, show_vals, hide_scale, (show_toggle if is_numeric else hide_toggle), hide_heat
            return [], [], hide_vals, hide_scale, hide_toggle, hide_heat

        # ── Legend shift-click color offset ──────────────────────────────────
        app.clientside_callback(
            """
            function(figure) {
                if (!window._shiftKeyState) {
                    window._shiftKeyState = {pressed: false, alt: false, pendingDigits: '', pendingActions: []};
                    document.addEventListener('keydown', function(e) {
                        if (e.key === 'Shift') window._shiftKeyState.pressed = true;
                        else if (e.key === 'Alt') window._shiftKeyState.alt = true;
                        else if (window._shiftKeyState.pressed && /^Digit[0-9]$/.test(e.code))
                            window._shiftKeyState.pendingDigits += e.code.slice(-1);
                        else if (window._shiftKeyState.pressed && e.code === 'Comma')
                            window._shiftKeyState.pendingDigits += ',';
                    });
                    document.addEventListener('keyup', function(e) {
                        if (e.key === 'Shift') {
                            window._shiftKeyState.pressed = false;
                            window._shiftKeyState.pendingDigits = '';
                            var actions = window._shiftKeyState.pendingActions;
                            window._shiftKeyState.pendingActions = [];
                            if (actions.length > 0)
                                dash_clientside.set_props('legend-shift-click', {data: JSON.stringify(actions)});
                        } else if (e.key === 'Alt') {
                            window._shiftKeyState.alt = false;
                        }
                    });
                }
                setTimeout(function() {
                    var gd = document.querySelector('#scatter-plot .js-plotly-plot');
                    if (!gd) return;
                    gd.removeAllListeners('plotly_legendclick');
                    gd.on('plotly_legendclick', function(eventData) {
                        if (!window._shiftKeyState.pressed) return;
                        var traceData = eventData.data[eventData.curveNumber];
                        var traceName = (traceData.meta && traceData.meta.name0) || traceData.name;
                        var traceMode = traceData.mode || '';
                        var action;
                        if (window._shiftKeyState.pendingDigits) {
                            var pd = window._shiftKeyState.pendingDigits;
                            window._shiftKeyState.pendingDigits = '';
                            if (pd.indexOf(',') >= 0) {
                                var parts = pd.split(',').map(function(p) {
                                    return p === '' ? null : parseFloat(p);
                                });
                                action = {t: traceName, m: traceMode, action: 'hsv', value: parts};
                            } else {
                                action = {t: traceName, m: traceMode, action: 'hsv_h', value: parseInt(pd, 10)};
                            }
                        } else {
                            action = {t: traceName, m: traceMode, action: window._shiftKeyState.alt ? 'dec' : 'inc'};
                        }
                        window._shiftKeyState.pendingActions.push(action);
                        return false;
                    });
                }, 300);
                return window.dash_clientside.no_update;
            }
            """,
            Output('scatter-plot', 'id'),
            Input('scatter-plot', 'figure'),
        )

        @callback(
            Output('color-offsets', 'data'),
            Input('legend-shift-click', 'data'),
            State('color-offsets', 'data'),
            prevent_initial_call=True,
        )
        def bump_color_offset(shift_click_data, offsets):
            if not shift_click_data:
                raise PreventUpdate
            # Both palettes are the same length, so a stored offset means the same
            # palette index in either theme — no theme Input is needed here.
            n = len(_DARK_THEME.palette)
            for item in orjson.loads(shift_click_data):
                # Line traces store under 'name::line' to allow independent colors from scatter.
                # Scatter stores under plain 'name'. Pass 2 checks ::line first, then falls back.
                trace_name = item['t']
                if item.get('m') == 'lines':
                    trace_name = trace_name + '::line'
                action  = item['action']
                current = offsets.get(trace_name, 0)
                if isinstance(current, dict) and 'series' in current:   # cycle from the series' colour
                    current = preset_color_offset(self._figure_presets, current) or 0
                if action == 'set':
                    offsets[trace_name] = {'d': int(item['value']) % n}
                elif action == 'hsv_h':
                    offsets[trace_name] = {'h': int(item['value']) % 10}
                elif action == 'hsv':
                    parts = item['value']
                    h = float(parts[0]) if (len(parts) > 0 and parts[0] is not None) else 0.0
                    s = float(parts[1]) if (len(parts) > 1 and parts[1] is not None) else 100.0
                    v = float(parts[2]) if (len(parts) > 2 and parts[2] is not None) else 100.0
                    offsets[trace_name] = {'hsv': [h, s, v]}
                elif isinstance(current, dict):
                    if 'h' in current:
                        offsets[trace_name] = {'h': (current['h'] + (1 if action == 'inc' else -1)) % 10}
                    elif 'hsv' in current:
                        h, s, v = current['hsv']
                        offsets[trace_name] = {'hsv': [(h + (36 if action == 'inc' else -36)) % 360, s, v]}
                    else:
                        offsets[trace_name] = {'d': (current['d'] + (1 if action == 'inc' else -1)) % n}
                elif action == 'inc':
                    offsets[trace_name] = (current + 1) % n
                else:
                    offsets[trace_name] = (current - 1) % n
            return offsets

        # ── Sticky colors toggle — clear palette when turned off ──────────────
        @callback(
            Output('color-offsets', 'data', allow_duplicate=True),
            Input('setting-sticky-colors', 'value'),
            prevent_initial_call=True,
        )
        def handle_sticky_colors_toggle(sticky):
            if not sticky or 'on' not in sticky:
                return {}
            raise PreventUpdate

        # ── Point click details ───────────────────────────────────────────────
        @callback(
            Output('point-details',    'children'),
            Output('selected-row-idx', 'data'),
            Output('export-yaml-btn',  'disabled'),
            Input('scatter-plot', 'clickData'),
            State('x-axis',      'value'),
            State('y-axis',      'value'),
            State('agg-group-by', 'value'),
            State('agg-y-mode',   'value'),
            State('trace-row-map', 'data'),
            prevent_initial_call=True,
        )
        def show_point_details(click_data, x_col, y_col, agg_group_by, agg_y_mode, trace_row_map):
            if not click_data or not click_data['points']:
                raise PreventUpdate
            pt = click_data['points'][0]
            curve_num  = pt.get('curveNumber', 0)
            point_num  = pt.get('pointNumber', 0)
            if trace_row_map and curve_num < len(trace_row_map):
                point_cd = trace_row_map[curve_num]
                customdata = point_cd[point_num] if point_num < len(point_cd) else []
            else:
                customdata = pt.get('customdata') or []
            row_idx_val = int(customdata[0]) if customdata else -1
            agg_count = int(customdata[1]) if len(customdata) > 1 else None
            y_cols = [y_col] if isinstance(y_col, str) else list(y_col or [])
            selected = row_idx_val if row_idx_val >= 0 else None
            row = df.loc[row_idx_val] if row_idx_val >= 0 and row_idx_val in df.index else None
            if row is None:
                # an aggregated marker (customdata [-1, n]), or a sampled point on an overlay line
                y_label = (f'{agg_y_mode}({", ".join(y_cols)})' if agg_y_mode and agg_y_mode != 'raw' and agg_group_by
                           else ', '.join(y_cols) or 'y')
                if not customdata:
                    note = '(interpolated line — not a data point)'
                elif agg_count is None:
                    note = '(aggregated point)'
                elif agg_y_mode and agg_y_mode != 'raw':
                    note = f'({agg_y_mode} of {agg_count} points)'
                else:
                    note = f'(aggregated from {agg_count} points)'
                point = {x_col: pt.get('x'), y_label: pt.get('y'), 'grouped by': ', '.join(agg_group_by or []) or None}
                return details_tree({'point': point}, note=note), selected, not has_params_json
            # a data point: the row's own values (the plotted x/y may be jittered), then its factors, identity, params
            hidden = {'params_json', 'design', 'i', 'j', 'L_out_path'}
            shown = {k: v for k, v in row.items() if k not in hidden and _present(v)}
            point = {x_col: shown.get(x_col, pt.get('x')),   # a computed column isn't in the frame: as plotted
                     **{c: shown.get(c, pt.get('y') if len(y_cols) == 1 else '(computed)') for c in y_cols}}
            params = self._row_params(row_idx_val)
            identity = {'config': row.get('i'), 'sample': row.get('j'), 'out': _out_path(params, row)}
            sections = {
                'point':    point,
                'values':   {k: v for k, v in shown.items() if k not in factors_subset},
                'factors':  {k: v for k, v in shown.items() if k in factors_subset},
                'identity': {k: v for k, v in identity.items() if _present(v)},
                'params':   params,
            }
            return details_tree(sections, open_sections=('point', 'values', 'factors')), selected, not has_params_json

        # ── YAML export ───────────────────────────────────────────────────────
        @callback(
            Output('yaml-download',  'data'),
            Output('export-status',  'children'),
            Input('export-yaml-btn', 'n_clicks'),
            State('selected-row-idx',  'data'),
            prevent_initial_call=True,
        )
        def export_yaml(n_clicks, row_idx):
            if not n_clicks:
                raise PreventUpdate
            if row_idx is None:
                return no_update, html.Span('⚠ Not available for aggregated points',
                                            style={'color': _DANGER})
            params_data = self._row_params(row_idx)
            if not params_data:
                return no_update, html.Span('⚠ Params not available for this row',
                                            style={'color': _DANGER})
            if 'logging' in params_data and params_data['logging'] and 'out_path' in params_data['logging']:
                params_data['logging']['out_path'] = str(params_data['logging']['out_path']).replace('grid_search', 'single_run')
            yaml_str = yaml.dump(params_data, default_flow_style=False, sort_keys=False)
            return dcc.send_string(yaml_str, filename='params.yaml'), ''

        # ── Derived filter dropdowns ──────────────────────────────────────────
        @callback(
            Output('derived-filter-dropdowns', 'children'),
            Output('pending-derived-filters',  'data'),
            Input('derived-factors', 'data'),
            State('pending-derived-filters', 'data'),
            prevent_initial_call=True,
        )
        def render_derived_filters(derived_factors_data, pending):
            children = []
            for entry in (derived_factors_data or []):
                col, values = entry['col'], entry['values']
                selected = (pending or {}).get(col, [])
                children.append(html.Div([
                    html.Label(col),
                    html.Div([
                        html.Button('Select All', id={'type': 'select-all', 'factor': col},
                                    n_clicks=0, style={**btn_sm, 'marginRight': '4px'}),
                        html.Button('Clear All', id={'type': 'clear-all', 'factor': col},
                                    n_clicks=0, style=btn_sm),
                    ], style={'marginBottom': '4px'}),
                    dcc.Dropdown(id={'type': 'dropdown', 'factor': col},
                                 options=[{'label': v, 'value': v} for v in values],
                                 value=selected, multi=True),
                ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}))
            return children, {}

        # ── Render: data pipeline + figure (fires on any data-changing input) ──
        @callback(
            Output('scatter-plot',           'figure'),
            Output('saved-viewport',         'data', allow_duplicate=True),
            Output('saved-trace-visibility', 'data', allow_duplicate=True),
            Output('filter-error',    'children'),
            Output('filter-count',    'children'),
            Output('fields-error',    'children'),
            Output('fields-info',     'children'),
            Output('x-axis',          'options'),
            Output('y-axis',          'options'),
            Output('derived-factors', 'data'),
            Output('color-by-factor', 'options', allow_duplicate=True),
            Output('agg-group-by',    'options', allow_duplicate=True),
            Output('trace-row-map',   'data'),
            Input('refresh-btn',                'n_clicks'),
            Input({'type': 'dropdown', 'factor': ALL}, 'value'),
            Input('agg-group-by',               'value'),
            Input('agg-y-mode',                 'value'),
            Input('x-bin-width',                'value'),
            Input('setting-display-points',      'value'),
            Input('setting-stratified-sample',   'value'),
            Input('setting-discrete-threshold',  'value'),
            Input('x-axis',                     'value'),
            Input('y-axis',                     'value'),
            Input('color-by-factor',            'value'),
            Input('process-trigger',            'data'),
            Input('color-by-values',     'value'),
            Input('color-offsets',       'data'),
            Input('color-as-continuous', 'value'),
            Input('heatmap-toggle',      'value'),
            Input('heatmap-dx',          'value'),
            Input('heatmap-dy',          'value'),
            Input('heatmap-min-n',       'value'),
            Input('x-jitter-toggle',     'value'),
            Input('x-jitter-min',        'value'),
            Input('x-jitter-max',        'value'),
            Input('y-jitter-toggle',     'value'),
            Input('y-jitter-min',        'value'),
            Input('y-jitter-max',        'value'),
            Input('legend-show-factor-name', 'value'),
            Input('skip-sparse-factors',     'value'),
            Input('overlay-dots-toggle', 'value'),
            Input('overlay-line-toggle', 'value'),
            Input('overlay-spline-k',          'value'),
            Input('overlay-spline-s',          'value'),
            Input('overlay-per-line-toggle',   'value'),
            Input('overlay-spline-ks',         'data'),
            Input('overlay-anchor-left',       'value'),
            Input('overlay-anchor-right',      'value'),
            Input('overlay-y-styles',          'data'),
            Input('overlay-line-width',        'value'),
            Input('overlay-line-opacity',      'value'),
            # An Input, not a State: flipping the theme must rebuild the figure —
            # the palette itself changes, which no client-side patch could do.
            Input('theme-toggle',             'value'),

            State('lazy-mode-active',         'data'),
            State('filter-expr',              'value'),
            State('fields-expr',              'value'),
            State('scatter-plot',             'relayoutData'),
            State('derived-factors',          'data'),
            State('setting-marker-size',      'value'),
            State('setting-marker-opacity',   'value'),
            State('saved-viewport',           'data'),
            State('saved-trace-visibility',   'data'),
            State('color-scale-picker',       'value'),
            State('color-scale-reverse',      'value'),
            State('view-figure',              'data'),
            prevent_initial_call='initial_duplicate',
        )
        def render_figure(refresh_clicks, filter_values, agg_group_by, agg_y_mode,
                          x_bin_width, display_pts, stratified_sample, dynamic_discrete_threshold,
                          x_col, y_col, color_factor, process_trigger,
                          color_values, color_offsets, as_continuous,
                          heatmap_on, heatmap_dx, heatmap_dy, heatmap_min_n,
                          x_jit_on, x_jit_min, x_jit_max,
                          y_jit_on, y_jit_min, y_jit_max,
                          legend_show_factor, skip_sparse,
                          overlay_dots_toggle, overlay_toggle, overlay_k, overlay_s,
                          overlay_per_line, overlay_ks,
                          anchor_left, anchor_right, overlay_y_styles,
                          line_width, line_opacity, theme,
                          lazy_mode, filter_expr, fields_expr, relayout_data, current_derived,
                          marker_size, marker_opacity, saved_viewport, saved_trace_vis,
                          color_scale, color_scale_reverse, view_figure):
            _lazy_inputs = {
                'x-axis', 'y-axis',
                'agg-group-by', 'agg-y-mode',
                'x-bin-width',
                'setting-display-points',
            }
            if lazy_mode and ctx.triggered_id is not None:
                def _lazy(pid):
                    cid = pid.rpartition('.')[0]
                    return cid in _lazy_inputs or (cid.startswith('{') and '"dropdown"' in cid)
                if all(_lazy(p) for p in ctx.triggered_prop_ids):
                    raise PreventUpdate

            theme = resolve_theme(theme)

            # Normalise y_col: multi-select dropdown may return a string or list
            y_cols = ([y_col] if isinstance(y_col, str) else list(y_col or []))
            _primary_y = y_cols[0] if y_cols else None

            filter_error = ''
            fields_error = ''
            fields_info  = ''
            current_axis_opts = [{'label': c, 'value': c} for c in axis_options]
            _pipeline = []

            def _empty(title, x=None, y=None):
                fig = go.Figure()
                fig.update_layout(
                    title=title, xaxis_title=x or 'X', yaxis_title=y or 'Y',
                    template=theme.template,
                )
                return (fig, no_update, no_update,
                        filter_error, '', fields_error, fields_info,
                        current_axis_opts, current_axis_opts,
                        no_update, no_update, no_update, no_update)

            if not x_col or not y_cols:
                return _empty('Select X and Y axes to display data')

            # ── Pipeline cache: skip filter/agg/sample on pure cosmetic triggers ──
            _pipe_key = (
                x_col or '', tuple(y_cols),
                tuple(sorted(str(g) for g in (agg_group_by or []))),
                agg_y_mode or 'raw', str(x_bin_width or 1), str(display_pts),
                tuple(str(v) if v else '' for v in (filter_values or [])),
                filter_expr or '', fields_expr or '',
                tuple(str(f) for f in (color_factor or [])),
                str(process_trigger), str(dynamic_discrete_threshold),
                str(stratified_sample),
            )
            _has_vp = bool(relayout_data and any(
                relayout_data.get(k) is not None
                for k in ('xaxis.range[0]', 'yaxis.range[0]')
            ))
            _use_cache = (
                isinstance(ctx.triggered_id, str)
                and ctx.triggered_id in _COSMETIC_TRIGGERS
                and not _has_vp
                and self._pipeline_cache_key == _pipe_key
            )

            if _use_cache:
                (filtered_df, x_plot_col, x_bin_label, y_agg_label, _pipeline,
                 derived_discrete, current_axis_opts, filter_error, fields_error,
                 fields_info, _ff_dict) = self._pipeline_cache_val
                filtered_df = filtered_df.copy()
            else:
                # 1. Factor filters
                mask = np.ones(len(df), dtype=bool)
                for item in ctx.inputs_list[1]:
                    factor = item['id']['factor']
                    values = item.get('value') or []
                    if values and factor in df.columns:
                        mask &= df[factor].astype(str).isin(values)
                filtered_df = df[mask].copy()
                _pipeline.append((len(df), ''))
                if len(filtered_df) != len(df):
                    _pipeline.append((len(filtered_df), 'filter'))

                # 2. Computed fields
                derived_discrete = []
                if fields_expr and fields_expr.strip():
                    filtered_df, fields_info, fields_error = parse_and_apply_fields(filtered_df, fields_expr)
                    current_axis_opts = self._axis_dropdown_options(filtered_df)

                # 2b. Dynamic discrete detection
                eff_threshold = dynamic_discrete_threshold if dynamic_discrete_threshold is not None else self.discrete_threshold
                static_discrete_set = set(discrete_factors)
                orig_cols = set(df.columns)
                candidates = {col: filtered_df[col] for col in filtered_df.columns if col not in orig_cols}
                candidates.update({col: df[col] for col in df.columns
                                   if col not in static_discrete_set and col not in candidates})
                for col, series in candidates.items():
                    n_uniq = series.nunique()
                    if not (1 < n_uniq < eff_threshold):
                        continue
                    is_numeric = pd.api.types.is_numeric_dtype(series)
                    vals = sorted(str(v) for v in series.dropna().unique())
                    derived_discrete.append({'col': col, 'values': vals, 'is_numeric': is_numeric})

                # 3. Validate axes
                missing = [c for c in ([x_col] + y_cols) if c not in filtered_df.columns]
                if missing:
                    if not fields_error:
                        fields_error = f'⚠ Column(s) not in data: {", ".join(missing)}. Define in Computed Fields.'
                    return _empty(f"Column(s) not found: {', '.join(missing)}", x_col, _primary_y)

                # 4–5. Query → binning → aggregation (shared pipeline)
                _ff_dict = {f: v for f, v in zip(discrete_factors, filter_values) if v}
                if len(y_cols) == 1:
                    # Single-Y: full pipeline including aggregation
                    filtered_df, x_plot_col, x_bin_label, y_agg_label, _query_err = self._apply_data_pipeline(
                        filtered_df, x_col, _primary_y,
                        filter_expr=filter_expr,
                        x_bin_width=x_bin_width,
                        agg_group_by=agg_group_by, agg_y_mode=agg_y_mode,
                        color_factor=color_factor, factor_filters=_ff_dict,
                    )
                else:
                    # Multi-Y: apply query filter inline; aggregation + binning run per-Y
                    # inside _build_overlay_figure, so avoid a redundant outer pipeline call.
                    _query_err = None
                    if filter_expr and filter_expr.strip():
                        try:
                            filtered_df = filtered_df.query(filter_expr)
                        except Exception as _e:
                            _query_err = str(_e)
                    x_plot_col  = x_col
                    x_bin_label = None
                    if agg_y_mode and agg_y_mode != 'raw' and x_bin_width and float(x_bin_width) > 0:
                        x_plot_col  = x_col + '_binned'
                        x_bin_label = f'{x_col} (bin={float(x_bin_width)})'
                    y_agg_label = agg_y_mode if x_bin_label else None
                if _query_err:
                    filter_error = f'⚠ {_query_err}'
                elif filter_expr and filter_expr.strip():
                    _pipeline.append((len(filtered_df), 'query'))
                if y_agg_label:
                    _pipeline.append((len(filtered_df), 'agg'))

                # 5.5. Viewport-aware sampling
                total_points = len(filtered_df)
                if relayout_data and x_plot_col and _primary_y:
                    x0 = relayout_data.get('xaxis.range[0]')
                    x1 = relayout_data.get('xaxis.range[1]')
                    y0 = relayout_data.get('yaxis.range[0]')
                    y1 = relayout_data.get('yaxis.range[1]')
                    if x0 is not None or y0 is not None:
                        vp_mask = np.ones(len(filtered_df), dtype=bool)
                        _vp_x = x_plot_col if x_plot_col in filtered_df.columns else x_col
                        if x0 is not None and x1 is not None and _vp_x in filtered_df.columns:
                            vp_mask &= (filtered_df[_vp_x] >= x0) & (filtered_df[_vp_x] <= x1)
                        # Y-range viewport only valid in single-Y mode (multi-Y overlays share one axis)
                        if len(y_cols) == 1 and y0 is not None and y1 is not None and _primary_y in filtered_df.columns:
                            vp_mask &= (filtered_df[_primary_y] >= y0) & (filtered_df[_primary_y] <= y1)
                        viewport_df = filtered_df[vp_mask]
                        if len(viewport_df) > 0:
                            filtered_df = viewport_df
                            _pipeline.append((len(filtered_df), 'viewport'))
                _display_limit = display_pts if display_pts is not None else initial_display_points
                # Skip sampling in multi-Y + agg mode: aggregation collapses rows inside
                # _build_overlay_figure, so sampling here would give statistically wrong means.
                _multi_y_agg = len(y_cols) > 1 and agg_y_mode and agg_y_mode != 'raw'
                if (not _multi_y_agg
                        and len(filtered_df) == total_points
                        and _display_limit and total_points > _display_limit):
                    _use_strat = bool(stratified_sample and 'on' in stratified_sample)
                    filtered_df = (_stratified_sample(filtered_df, _display_limit, color_factor)
                                   if _use_strat else filtered_df.sample(_display_limit, random_state=None))
                    _pipeline.append((len(filtered_df), 'sample'))

                # 6. Row index — always set so customdata stays slim (single int per point)
                filtered_df = filtered_df.copy()
                _is_agg = bool(y_agg_label) or (len(y_cols) > 1 and agg_y_mode != 'raw')
                filtered_df['_row_idx'] = -1 if _is_agg else filtered_df.index

                # Store pipeline result so cosmetic-only re-renders can skip it
                self._pipeline_cache_key = _pipe_key
                self._pipeline_cache_val = (
                    filtered_df.copy(), x_plot_col, x_bin_label, y_agg_label,
                    list(_pipeline), list(derived_discrete), list(current_axis_opts),
                    filter_error, fields_error, fields_info, dict(_ff_dict),
                )

            pipeline_parts = [_fmt(_pipeline[0][0])] if _pipeline else []
            for n, label in (_pipeline[1:] if _pipeline else []):
                pipeline_parts.append(f'{_fmt(n)} ({label})')

            current_cols = [e['col'] for e in (current_derived or [])]
            new_cols     = [e['col'] for e in derived_discrete]
            derived_store_out = (no_update if _use_cache
                                 else (derived_discrete if new_cols != current_cols else no_update))

            try:
                if filtered_df.empty:
                    return _empty('No data for selected filters', x_col, _primary_y)

                filter_dict = {item['id']['factor']: item.get('value') for item in ctx.inputs_list[1]}
                # the controls keyed by id ('<id>.<prop>' in ctx), as a saved view holds them, built by the
                # step the export uses
                view = {pid.rpartition('.')[0]: v for pid, v in {**ctx.states, **ctx.inputs}.items()}
                view.update({'y-axis': y_cols, '_extras': {
                    'color_offsets': color_offsets, 'trace_visibility': saved_trace_vis, 'figure': view_figure}})
                fig = self._view_figure(filtered_df, view, theme=theme, factor_filters=filter_dict,
                                        x_plot_col=x_plot_col, x_bin_label=x_bin_label, y_agg_label=y_agg_label,
                                        viewport=saved_viewport)
                # saved viewport and visibility apply once (None clears their stores); a new uirevision makes
                # Plotly drop the current zoom for the saved ranges
                vp_out = tv_out = no_update
                if saved_viewport:
                    fig.update_layout(uirevision=_time.time())
                    vp_out = None
                if saved_trace_vis:
                    tv_out = None
                trace_row_map = [
                    (t.customdata.tolist() if t.customdata is not None and len(t.customdata) > 0 else [])
                    for t in fig.data
                ]
                if _use_cache:
                    return (fig, vp_out, tv_out,
                            no_update, ' → '.join(pipeline_parts), no_update, no_update,
                            no_update, no_update,
                            no_update, no_update, no_update, trace_row_map)
                return (fig, vp_out, tv_out,
                        filter_error, ' → '.join(pipeline_parts), fields_error, fields_info,
                        current_axis_opts, current_axis_opts,
                        derived_store_out,
                        _build_color_by_options(derived_discrete),
                        _build_agg_options(derived_discrete),
                        trace_row_map)
            except Exception:
                err = _traceback.format_exc()
                print(f'[render_figure ERROR]\n{err}', flush=True, file=__import__('sys').stderr)
                return _empty('render_figure error — check server log')

        # ── Stage 3: Clientside colorscale update (continuous color mode only) ─
        # color-scale-picker / color-scale-reverse are no longer Inputs to
        # render_figure; this callback handles them without a server round-trip.
        # It only acts when layout.coloraxis exists (continuous coloring mode).
        app.clientside_callback(
            """
            function(scale, reverse, figure) {
                if (!figure || !figure.layout || !figure.layout.coloraxis)
                    return window.dash_clientside.no_update;
                var rev = reverse && reverse.includes('on');
                var fullscale = (scale || 'Viridis') + (rev ? '_r' : '');
                var layout = Object.assign({}, figure.layout, {
                    coloraxis: Object.assign({}, figure.layout.coloraxis, {colorscale: fullscale})
                });
                return Object.assign({}, figure, {layout: layout});
            }
            """,
            Output('scatter-plot', 'figure', allow_duplicate=True),
            Input('color-scale-picker',  'value'),
            Input('color-scale-reverse', 'value'),
            State('scatter-plot', 'figure'),
            prevent_initial_call=True,
        )

        # ── Stage 4: Clientside marker update (zero server round-trip) ────────
        app.clientside_callback(
            """
            function(size, opacity, figure) {
                if (!figure || !figure.data || !figure.data.length) return window.dash_clientside.no_update;
                var traces = figure.data.map(function(t) {
                    return Object.assign({}, t, {marker: Object.assign({}, t.marker || {}, {size: size, opacity: opacity})});
                });
                return Object.assign({}, figure, {data: traces});
            }
            """,
            Output('scatter-plot', 'figure', allow_duplicate=True),
            Input('setting-marker-size',    'value'),
            Input('setting-marker-opacity', 'value'),
            State('scatter-plot', 'figure'),
            prevent_initial_call=True,
        )

        # ── Stage 5: Track client-side trace visibility (legend clicks) ─────────
        # State('scatter-plot', 'figure') only returns the last server-sent figure;
        # legend clicks toggle visibility in the browser but never propagate back.
        # This callback (a) snapshots visibility when the server pushes a new figure
        # and (b) re-registers a plotly_restyle listener so any legend click updates
        # the store. save_view reads from this store instead of from the figure State.
        app.clientside_callback(
            """
            function(figure, prevVis) {
                function captureVis(data) {
                    var vis = {};
                    (data || []).forEach(function(t) {
                        var name = (t.meta && t.meta.name0) || t.name;
                        if (name) {
                            var v = t.visible;
                            vis[name + '|' + ((t.meta && t.meta.mode0) || t.mode || 'markers')] =
                                (v === undefined || v === true) ? true : v;
                        }
                    });
                    return vis;
                }
                setTimeout(function() {
                    var gd = document.querySelector('#scatter-plot .js-plotly-plot');
                    if (!gd) return;
                    gd.removeAllListeners('plotly_restyle');
                    gd.on('plotly_restyle', function() {
                        dash_clientside.set_props('live-trace-visibility',
                            {data: captureVis(gd.data)});
                    });
                }, 150);
                var newVis = captureVis((figure || {}).data);
                var prev = prevVis || {};
                var merged = {};
                for (var k in newVis) {
                    // Server doesn't know about legend clicks — preserve legendonly
                    // from previous state when the server only says true.
                    merged[k] = (newVis[k] === true && prev[k] === 'legendonly')
                        ? 'legendonly' : newVis[k];
                }
                return merged;
            }
            """,
            Output('live-trace-visibility', 'data'),
            Input('scatter-plot', 'figure'),
            State('live-trace-visibility', 'data'),
            prevent_initial_call=True,
        )

    # ── PNG route ─────────────────────────────────────────────────────────────

    # Title print ratios for the scatter export family (figure1, temporal,
    # path-integration — all share this module, so one constant each governs
    # every scatter view).  plotly renders an inherited figure title at ~1.4x
    # the base font and inherited axis titles at ~1.2x; the paper needs both at
    # caption size while ticks stay at the furniture scale: axis-label em = 1.0x font, title em = 1.0–1.1x font
    # ("slight hierarchy above labels is fine").  Only active under ?font=;
    # per-figure ?title=<px> / ?axislabel=<px> override these as absolute px.
    EXPORT_TITLE_SCALE = 1.05
    EXPORT_AXIS_TITLE_SCALE = 1.0

    def _register_export_route(self):
        """
        Serve a PNG snapshot of any saved view via GET request.

        Usage:
            curl 'http://127.0.0.1:8051/?my_view&format=png&width=1200&height=800' -o my_view.png
        """
        from .utils import register_export_route

        def _reconstruct(view_data, width, height, theme_override):
            return self._reconstruct_figure(view_data, theme_override=theme_override)

        register_export_route(
            self.app,
            reconstruct_fn=_reconstruct,
            safe_roots=self._safe_roots,
            default_search_dirs=[Path(self._initial_views_dir), Path('/out/dashboard'), Path('/tmp/dashboard')],
            title_scale=self.EXPORT_TITLE_SCALE,
            axis_title_scale=self.EXPORT_AXIS_TITLE_SCALE,
        )

    # ── Assembly ──────────────────────────────────────────────────────────────

    def _layout(self):
        sample_cols = self.axis_options[:6]
        cols_hint = ', '.join(sample_cols) + (', ...' if len(self.axis_options) > 6 else '')
        return build_layout(
            discrete_factors=self.discrete_factors,
            factor_values=self.factor_values,
            axis_options=self.axis_options,
            default_x=self.default_x,
            default_y=self.default_y,
            color_by_options=self.color_by_options,
            initial_display_points=self.initial_display_points,
            discrete_threshold=self.discrete_threshold,
            has_params_json=self.has_params_json,
            cols_hint=cols_hint,
            initial_views_dir=self._initial_views_dir,
        )

    @functools.cached_property
    def _view_defaults(self) -> dict:
        """The controls' starting values: a saved view's missing keys, live (on load) and in the export."""
        return control_defaults(self._layout())

    def build(self) -> Dash:
        register_templates()
        app = Dash(__name__, suppress_callback_exceptions=True,
                   url_base_pathname=self._url_prefix)
        app.index_string = _theme_index_string(app.index_string)
        app.layout = self._layout()
        self.app = app
        self._register_callbacks()
        self._register_export_route()
        register_admin_token_capture(app)
        register_theme_attribute(app)
        from wsgi.base import register_view_refresh
        register_view_refresh(
            app,
            open_trigger_id='views-toggle',
            list_options_fn=lambda p: _view_dropdown_options(_views_dir(p)),
            body_state_id='views-body',
        )
        return app

def create_scatter_dashboard(
    df: pd.DataFrame,
    factors: list[str],
    discrete_threshold: int = 100,
    renderer: str = 'auto',
    initial_display_points: int = 10_000,
    safe_roots: list[str] = None,
    initial_views_dir: str = '/out/dashboard',
    url_prefix: str = '/',
    config_by_lout: dict = None,
) -> Dash:
    return ScatterDashboard(
        df, factors,
        discrete_threshold=discrete_threshold,
        renderer=renderer,
        initial_display_points=initial_display_points,
        # None ⇒ ScatterDashboard applies DEFAULT_SAFE_ROOTS. This used to read
        # `safe_roots or ['/out', '/tmp']`, a second, wider spelling of the write
        # boundary that silently won for every deployed app (nothing passes
        # safe_roots), so views could be written anywhere under /out — including
        # the results directories — and anywhere under /tmp. The boundary is
        # defined once, in DEFAULT_SAFE_ROOTS; do not restate it here.
        safe_roots=safe_roots,
        initial_views_dir=initial_views_dir,
        url_prefix=url_prefix,
        config_by_lout=config_by_lout,
    ).build()

