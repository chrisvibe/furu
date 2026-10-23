import json
from pathlib import Path

import dash_ag_grid as dag
from dash import Dash, dcc, html, Input, Output, State, ctx, \
    clientside_callback, no_update, ALL
from dash.exceptions import PreventUpdate

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from wsgi.base import views_dir as _views_dir_fn, list_view_options, is_light_theme, register_view_refresh, register_config_file_route
from project.boolean_reservoir.code.utils.load_save import custom_load_grid_search_data
from project.boolean_reservoir.code.utils.categorical_ordering import grayish_sort
from .utils import (
    register_export_route, make_combo_column,
    parse_and_apply_fields, _fmt, apply_trace_vis,
    guard_view_write, DEFAULT_SAFE_ROOTS, admin_token_store,
    register_admin_token_capture, admin_token_state,
    save_view_json, load_view_json, delete_view_json,
    _AGG_OPTIONS, _DEFAULT_AGG, _DEFAULT_CACHE_DIR as _CACHE_DIR, VIEWS_ROOT,
    load_figure_presets, preset_factor_colors, preset_level_label, preset_level_tick, preset_series_labels,
    preset_color_offset, preset_series_dash, preset_axis_title,
    level_name, view_name_from_query,
)
from .scatter_layout import btn_sm
from .theme import (
    DARK as _DARK_THEME, LIGHT as LIGHT_THEME, resolve as resolve_theme,
    register_templates, register_theme_attribute, theme_pref_store,
    make_theme_toggle, index_string as _theme_index_string,
    SURFACE as _SURFACE, PANEL as _PANEL, CONTROL as _CONTROL,
    BORDER as _BORDER, TEXT as _TEXT, TEXT_DIM as _TEXT_DIM,
    FAINT as _FAINT, DANGER as _DANGER, ACCENT as _ACCENT,
    GRID_HIT_BG, GRID_HIT_FG, ROW_HIT_BG, ROW_HIT_FG,
)

_VIEWS_DIR  = VIEWS_ROOT / 'polar'   # the site's polar views; create_polar_dashboard(views_dir=...) moves it
_SAFE_ROOTS = [Path(r) for r in DEFAULT_SAFE_ROOTS]
# bare URL keys that are flags, not view names
_URL_FLAGS = {'expand', 'format', 'width', 'height', 'theme', 'dir'}


def _vdir(path_str):
    return _views_dir_fn(path_str, default=_VIEWS_DIR)

# ── Config label helper ──────────────────────────────────────────────────────

def _config_label(path):
    skip = {'config', 'grid_search', 'design_choices'}
    return ' · '.join(p for p in Path(path).with_suffix('').parts if p not in skip)

# ── Column helpers ────────────────────────────────────────────────────────────

_DISCRETE_THRESHOLD = 20
_COLOR_BY_MAX       = 100


def _classify_columns(df):
    skip = {'params_json'}
    numeric, factors = [], []
    for col in df.columns:
        if col in skip or col.startswith('_'):
            continue
        if df[col].dtype.kind in ('f', 'i', 'u', 'b'):
            numeric.append(col)
            if df[col].nunique() <= _DISCRETE_THRESHOLD:
                factors.append(col)
        else:
            factors.append(col)
    return numeric, factors


def _color_by_cols(df):
    """Contour candidates: numeric columns (k̄) and text factors (e.g. a head) with 2 to _COLOR_BY_MAX levels."""
    skip = {'params_json'}
    return [c for c in df.columns
            if c not in skip and not c.startswith('_')
            and df[c].dtype.kind in ('f', 'i', 'u', 'O', 'U')
            and 1 < df[c].nunique() <= _COLOR_BY_MAX]


def _defaults(df):
    numeric, _ = _classify_columns(df)
    k_avg_cands = _color_by_cols(df)
    metric = next((c for c in ('kqgr_kq', 'T_accuracy', 'accuracy') if c in numeric),
                  numeric[0] if numeric else None)
    k_avg  = next((c for c in ('R_k_avg',) if c in k_avg_cands),
                  k_avg_cands[0] if k_avg_cands else None)
    return metric, k_avg, []


def _options(cols):
    return [{'label': c, 'value': c} for c in cols]

# ── In-process df cache ───────────────────────────────────────────────────────

_df_cache: dict = {}
# Bound the cache: each cached combined df is ~0.5-1 GB resident. FIFO eviction
# (dicts keep insertion order) caps worst-case growth at the limit below instead
# of letting one browsing session balloon the container.
_df_cache_limit = 4
_registry: list = []   # set by create_polar_dashboard

# ── Selection overlay helpers (module-level, framework-agnostic) ──────────────

_OVERLAY_META = 'selection-overlay'


def _strip_overlays(fig):
    return [dict(t) for t in (fig or {}).get('data', [])
            if t.get('meta') != _OVERLAY_META]


def _get_rmax(fig):
    try:
        rng = fig.get('layout', {}).get('polar', {}).get('radialaxis', {}).get('range')
        if rng and rng[1] is not None:
            return float(rng[1])
    except (TypeError, IndexError):
        pass
    vals = [r for t in fig.get('data', [])
            if t.get('type') == 'scatterpolar' and t.get('meta') != _OVERLAY_META
            for r in (t.get('r') if t.get('r') is not None else []) if r is not None]
    return float(max(vals)) if vals else 1.0


def _make_selection_overlays(constraints, table_data, fig):
    """Derive AND constraints from [{column_id, value}] list; return barpolar wedge dicts.

    Each constraint entry directly carries the (column_id, value) pair — no row-index lookup.

    Every direction owns an equal step-wide slice of the circle — from the midpoint with its
    left neighbour to the midpoint with its right neighbour — whether it's alone or part of a
    run. Matching directions are grouped into contiguous runs (directions exactly one angular
    step apart) and each run is padded by half a step on both ends, i.e. the union of its
    members' slices. This holds regardless of run length: an isolated match still needs the
    padding to have any width at all, and a multi-point run needs it too, or coverage falls
    short by half a step at every boundary between differently-valued runs — invisible with
    one boundary, but a real gap wherever two groups meet (e.g. a clean 50/50 split should
    highlight exactly half the circle, not just first-to-last).

    The circle is genuinely periodic (make_polar_figure gives each combo its own slot, step =
    360/n_combos, none duplicated at both 0° and 360°) — so the last direction and the first
    direction are real step-apart neighbours through the seam, same as any other pair. A run
    touching one end is merged with a run touching the other end when they're wrap-adjacent,
    and edges are allowed to go negative / past 360 (Plotly's angular axis is periodic) rather
    than being clamped, since clamping would clip real coverage rightfully owed to that combo.
    """
    if not constraints or not table_data:
        return []
    df = pd.DataFrame(table_data)
    # Each constraint independently matches rows; union them (additive highlights).
    mask = pd.Series(False, index=df.index)
    for c in constraints:
        col, val = c.get('column_id'), c.get('value')
        if col and col in df.columns:
            mask |= df[col].astype(str) == str(val if val is not None else '')
    matching_dirs = df.loc[mask, 'Direction'].dropna() if 'Direction' in df.columns else pd.Series(dtype=str)
    if matching_dirs.empty:
        return []
    r_max = _get_rmax(fig)
    step = 360 / max(len(df), 1)
    thetas = sorted(pd.to_numeric(matching_dirs.str.rstrip('°'), errors='coerce').dropna().tolist())

    # Direction labels are rounded to 1 decimal (make_polar_figure's `_label`), so two
    # rounded values can differ from the true (unrounded) step by up to ~0.1° — tolerance
    # must cover that. n_combos is capped at 1000 elsewhere, so step is always >= ~0.36°,
    # leaving comfortable margin before this could misjudge true non-adjacency as adjacent.
    _ADJACENT_TOL = 0.15
    runs = [[thetas[0]]]
    for th in thetas[1:]:
        if abs(th - runs[-1][-1] - step) < _ADJACENT_TOL:
            runs[-1].append(th)
        else:
            runs.append([th])

    # Wrap-around: the last direction and the first are `step` apart through the seam. If
    # both ends are matched, merge them into one run (shifting the tail run below 0 so the
    # combined run is a single contiguous ascending sequence).
    if len(runs) > 1:
        wrap_gap = (runs[0][0] + 360) - runs[-1][-1]
        if abs(wrap_gap - step) < _ADJACENT_TOL:
            tail = [t - 360 for t in runs.pop()]
            runs[0] = tail + runs[0]

    overlays = []
    for run in runs:
        left, right = run[0] - step / 2, run[-1] + step / 2
        overlays.append({
            'type': 'barpolar', 'r': [r_max], 'theta': [(left + right) / 2], 'width': [right - left],
            'base': 0, 'marker': {'color': _DARK_THEME.polar_mark, 'line': {'width': 0}},
            'meta': _OVERLAY_META, 'showlegend': False,
        })
    return overlays



def _factors_ascending_from_state(column_state, factors):
    """Derive (ordered_factors, ascending_list) from AG Grid columnState.

    Column order in the table drives factor order in the polar plot.
    Sort direction (asc/desc) per column drives ascending flags.
    Factors not present in columnState keep their original order at the end,
    defaulting to ascending.
    """
    if not column_state or not factors:
        return factors, [True] * len(factors)
    field_pos  = {c['colId']: i for i, c in enumerate(column_state)}
    sort_dir   = {c['colId']: c.get('sort') for c in column_state}
    in_state   = sorted([f for f in factors if f in field_pos], key=lambda f: field_pos[f])
    not_in     = [f for f in factors if f not in field_pos]
    ordered    = in_state + not_in
    ascending  = [sort_dir.get(f) != 'desc' for f in ordered]
    return ordered, ascending


def _load_df(config_path):
    # config_path may be a string or list — custom_load_grid_search_data handles both.
    # Same pattern as scatter dashboard (wsgi/base.py: load_scatter_dashboard).
    paths = config_path if isinstance(config_path, list) else [config_path]
    cache_key = tuple(sorted(paths))
    if cache_key not in _df_cache:
        entry = next((e for e in _registry if e['path'] in paths), None)
        if entry is None:
            return None
        # an entry with its own load() -> (df, factors) is a ready frame (furu's one row per job x readout x
        # head, as the scatter apps read it); the rest are grids read through their extractions
        if 'load' in entry:
            df, _ = entry['load']()
        else:
            df, _ = custom_load_grid_search_data(
                config_paths=paths,
                extractions=entry['extractions'],
                cache_dir=_CACHE_DIR,
            )
        _df_cache[cache_key] = df
        while len(_df_cache) > _df_cache_limit:
            _df_cache.pop(next(iter(_df_cache)))
    return _df_cache[cache_key]

# ── Data helpers ──────────────────────────────────────────────────────────────

def _prepare_df(df, fields_expr, filter_expr):
    """Stage 1: apply computed fields, then str filter. Returns (df, error)."""
    error = ''
    if fields_expr and fields_expr.strip():
        df, _, err = parse_and_apply_fields(df.copy(), fields_expr)
        if err:
            error = err
    if filter_expr and filter_expr.strip():
        try:
            df = df.query(filter_expr)
        except Exception as e:
            error = f'Filter: {e}'
    return df, error


def _build_polar(df, ordered_factors, ascending, k_avg_col, metric_col,
                 agg_mode, theme, r_min, r_max, sort_mode,
                 constraints, trace_visibility, direction_filter=None, center_seam=True,
                 reference_col=None, direction_labels=False):
    """Single rendering pipeline shared by the render callback and export reconstruct."""
    fig, mapping_df, n_combos = _figure(
        df, ordered_factors, agg_mode, metric_col, k_avg_col, theme,
        direction_filter=direction_filter, ascending=ascending,
        r_min=r_min, r_max=r_max, sort_mode=sort_mode, center_seam=center_seam,
        reference_col=reference_col, direction_labels=direction_labels,
    )
    records  = mapping_df.to_dict('records')
    fig_dict = fig.to_dict()
    overlays = _make_selection_overlays(constraints or [], records, fig_dict)
    if overlays:
        fig_dict['data'] = list(fig_dict['data']) + overlays
        # Barpolar overlays cause Plotly to add padding and auto-expand the radial
        # axis beyond the data range. Pin the range to prevent zoom-out on highlight.
        radial = fig_dict.get('layout', {}).get('polar', {}).get('radialaxis', {})
        if not radial.get('range'):
            rmax = _get_rmax(fig_dict)
            fig_dict.setdefault('layout', {}).setdefault('polar', {}).setdefault('radialaxis', {})['range'] = [0, rmax]
    if trace_visibility:
        apply_trace_vis(fig_dict, trace_visibility)
    return fig_dict, mapping_df, n_combos


def _resolve(config_path, fields_expr, filter_expr, factors, k_avg_col):
    """Load df, apply transforms, return (raw_df, df, valid_factors, err).

    Returns (None, None, None, None) if config not available.
    valid_factors filters `factors` to columns that actually exist in df.
    """
    raw_df = _load_df(config_path)
    if raw_df is None:
        return None, None, None, None
    df, err = _prepare_df(raw_df, fields_expr, filter_expr)
    valid_factors = [f for f in (factors or []) if f in df.columns]
    return raw_df, df, valid_factors, err

# ── View helpers ──────────────────────────────────────────────────────────────

def _list_view_options(path_str=None):
    return list_view_options(path_str, default=_VIEWS_DIR)


def _pipeline_status(stages, error=''):
    """stages: list of (n, label) — same convention as scatter dashboard."""
    parts = [_fmt(stages[0][0])] if stages else []
    for n, label in stages[1:]:
        parts.append(f'{_fmt(n)} ({label})')
    text = ' → '.join(parts)
    children = [html.Span(text, style={'color': _FAINT})]
    if error:
        children.append(html.Span(f'  {error}', style={'color': _DANGER}))
    return children


def make_polar_figure(df: pd.DataFrame, factors: list,
                      agg_mode: str = 'mean',
                      metric_col: str = 'accuracy', k_avg_col: str = 'R_k_avg',
                      ascending: list = None,
                      direction_filter: set = None,
                      r_min=None, r_max=None,
                      sort_mode: str = 'gray',
                      center_seam: bool = True,
                      reference_col: str = None,
                      palette=None,
                      direction_labels: bool = False) -> tuple:
    """Build the polar figure and direction-mapping DataFrame.

    reference_col: optional; each contour also gets a dashed circle, in its colour, at that column's mean over the
    contour's rows (e.g. a head's chance floor).
    palette: the theme's palette, for contours the site presets list as series (a head takes its palette colour).
    direction_labels: label each direction by its factor levels ('override · r = 64', a level spelled as the site
    presets' factor label) instead of its angle; for a figure with few directions.

    center_seam: if True (default), the whole circle is rotated by half a step so the
    0°/360° reference sits in the middle of the wrap-around gap between the last and
    first direction, rather than exactly on the first direction — the shape's own
    symmetries line up with the axis instead of being split by it. If False, the first
    direction sits exactly at 0° (legacy behaviour).

    Returns (fig, mapping_df, n_combos) where mapping_df has columns
    [Direction, *factors_no_k].
    """
    factors_no_k = [f for f in factors if f != k_avg_col]
    if reference_col in (metric_col, k_avg_col, *factors_no_k) or reference_col not in df.columns:
        reference_col = None
    needed = [metric_col, k_avg_col] + factors_no_k + ([reference_col] if reference_col else [])
    df = df[[c for c in needed if c in df.columns]].copy()

    df['_combo'], _ = make_combo_column(df, factors_no_k)
    cols = {metric_col: agg_mode}
    cols.update(dict.fromkeys(factors_no_k, 'first'))
    if reference_col:
        cols[reference_col] = 'mean'
    df_grouped = df.groupby(['_combo', k_avg_col]).agg(cols).reset_index()

    # Build per-factor ascending flags aligned to factors_no_k.
    # `ascending` comes from _factors_ascending_from_state which operates on all
    # valid_factors (including k_avg_col), so it may be longer than factors_no_k.
    if isinstance(ascending, list):
        factor_asc = {f: a for f, a in zip(factors, ascending)}
        asc = [factor_asc.get(f, True) for f in factors_no_k]
    else:
        asc = [True] * len(factors_no_k)
    sort_mode = sort_mode or 'gray'
    if factors_no_k:
        if sort_mode == 'gray':
            df_grouped = grayish_sort(df_grouped, factors_no_k, ascending=asc)
        else:
            df_grouped = df_grouped.sort_values(factors_no_k, ascending=asc).reset_index(drop=True)

    codes, uniques = pd.factorize(df_grouped['_combo'])
    n_combos = len(uniques)
    if n_combos > 1000:
        empty = go.Figure()
        empty.update_layout(template=_DARK_THEME.template)
        return empty, pd.DataFrame(columns=['Direction'] + factors_no_k), n_combos

    # Each combo gets its own slot evenly tiling the full circle (step = 360/n_combos) —
    # no combo duplicated at both 0° and 360°. The line trace below re-closes the loop
    # visually without a second combo occupying that seam.
    step = 360 / max(n_combos, 1)
    seam_offset = step / 2 if center_seam else 0
    df_grouped['_n']     = codes
    df_grouped['_deg']   = (df_grouped['_n'] * step + seam_offset) % 360
    df_grouped['_label'] = df_grouped['_deg'].round(1).astype(str) + '°'
    presets = load_figure_presets(_VIEWS_DIR)
    compact = False
    if direction_labels and factors_no_k:
        # compact ('O64') when the presets spell every level of every factor so, else 'override · r = 64'
        rows = df_grouped[factors_no_k]
        ticks = rows.apply(lambda row: [preset_level_tick(presets, f, v) for f, v in row.items()], axis=1)
        compact = all(t is not None for tt in ticks for t in tt)
        df_grouped['_tick'] = (ticks.map(''.join) if compact else rows.apply(
            lambda row: ' · '.join(preset_level_label(presets, f, v) for f, v in row.items()), axis=1))
    else:
        df_grouped['_tick'] = df_grouped['_label']

    if direction_filter:
        df_grouped = df_grouped[df_grouped['_label'].isin(direction_filter)]

    unique_dirs = (df_grouped[['_deg', '_label', '_tick']]
                   .drop_duplicates('_deg').sort_values('_deg'))

    polar_axis = dict(
        angularaxis=dict(
            tickmode='array',
            tickvals=unique_dirs['_deg'].tolist(),
            ticktext=unique_dirs['_tick'].tolist(),
            rotation=90, direction='clockwise',
            showticklabels=True, ticks='outside',
            tickfont=dict(size=10),
        ),
        radialaxis=dict(
            **(dict(range=[r_min, r_max]) if r_min is not None and r_max is not None else {}),
            tickfont=dict(size=10),
        ),
    )

    # a contour level the site presets list takes its colour there: an ordered colour factor's level (k̄, as in
    # every scatter view coloured by it) or a series (a head: its palette colour and label); else Plasma_r's
    # colours in level order
    colors      = px.colors.sequential.Plasma_r
    k_values    = sorted(df_grouped[k_avg_col].unique())
    preset      = preset_factor_colors(presets, k_avg_col)
    labels      = preset_series_labels(presets)

    def _series_color(v):
        offset = preset_color_offset(presets, {'series': str(v)})
        return palette[offset['d'] % len(palette)] if palette and offset and 'd' in offset else None

    color_scale = {v: preset.get(level_name(v)) or _series_color(v) or colors[i % len(colors)]
                   for i, v in enumerate(k_values)}

    fig = go.Figure()
    for k in k_values:
        sub   = df_grouped[df_grouped[k_avg_col] == k]
        r     = list(sub[metric_col])
        theta = list(sub['_deg'])
        color = color_scale[k]
        name  = labels.get(str(k), str(k))
        # Close the polygon by re-drawing the first point at the end — purely visual,
        # not an extra combo (each combo still owns exactly one slot).
        r_closed     = r + r[:1]
        theta_closed = theta + theta[:1]
        # the contour, its reference circle and its start mark share a legend group, so one legend click (or a
        # saved view's trace_visibility) shows or hides all three
        fig.add_trace(go.Scatterpolar(
            r=r_closed, theta=theta_closed, mode='lines', name=name, legendgroup=name,
            line=dict(color=color, width=2, dash=preset_series_dash(presets, str(k)) or 'solid'),
        ))
        if reference_col and len(sub):
            # the contour's reference as a dashed circle (the full turn, not only the shown directions)
            fig.add_trace(go.Scatterpolar(
                r=[sub[reference_col].mean()] * 73, theta=[i * 5 for i in range(73)], mode='lines',
                name=f'{name} {reference_col}', legendgroup=name, line=dict(color=color, width=1, dash='dash'),
                showlegend=False, hoverinfo='skip',
            ))
        if r:
            # Marks where the line starts — the first direction's own position, which
            # is 0° only when center_seam is off.
            fig.add_trace(go.Scatterpolar(
                r=[r[0]], theta=[theta[0]],
                mode='markers', name=name, legendgroup=name,
                marker=dict(color=color, symbol='x', size=6,
                            line=dict(width=1.5)),
                showlegend=False, hoverinfo='skip',
            ))

    fig.update_layout(
        polar=polar_axis,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=True, legend=dict(title=preset_axis_title(presets, k_avg_col, k_avg_col)),
        uirevision='polar',
        meta={'compact_directions': compact},
    )

    mapping_df = (df_grouped.groupby('_label', sort=False).first()
                  .reset_index()[['_label'] + factors_no_k]
                  .rename(columns={'_label': 'Direction'}))
    return fig, mapping_df, n_combos


_LEGEND_STRIP = 0.26   # a compact (paper) polar's frame below the circle (its bottom labels, the metric's title, the legend)


def _figure(df, factors, agg_mode, metric_col, k_avg_col, theme,
            direction_filter=None, ascending=None, r_min=None, r_max=None,
            sort_mode='gray', center_seam=True, reference_col=None, direction_labels=False):
    """Returns (polar_fig, mapping_df, n_combos)."""
    th = resolve_theme(theme)
    fig, mapping_df, n_combos = make_polar_figure(
        df, factors,
        agg_mode=agg_mode,
        metric_col=metric_col, k_avg_col=k_avg_col,
        direction_filter=direction_filter,
        ascending=ascending,
        r_min=r_min, r_max=r_max,
        sort_mode=sort_mode,
        center_seam=center_seam,
        reference_col=reference_col,
        palette=th.palette,
        direction_labels=direction_labels,
    )
    presets = load_figure_presets(_VIEWS_DIR)
    compact = (fig.layout.meta or {}).get('compact_directions')
    fig.update_layout(
        # paper_bgcolor is deliberately omitted — the template already carries it.
        template=th.template,
        title=dict(
            text=preset_axis_title(presets, f'{metric_col} ({agg_mode})', f'{metric_col}  <i>({agg_mode})</i>'),
            font=dict(size=11, color=th.fig_title_font),
            x=0.98, xanchor='right',
            y=0.01, yanchor='bottom',
        ),
        # level labels are wider than angles: room for them beside the circle
        margin=dict(l=110, r=110, t=40, b=50) if direction_labels and not compact else dict(l=40, r=40, t=30, b=50),
    )
    if compact:
        # a paper figure: the circle keeps the top of the frame whether or not the panel shows a legend (two panels
        # side by side keep one size); below it the metric, as an axis title, then the legend in one row
        radial = fig.layout.polar.radialaxis
        fig.add_annotation(text=fig.layout.title.text, x=0.5, y=_LEGEND_STRIP / 2, xref='paper', yref='paper',
                           xanchor='center', yanchor='middle', showarrow=False, font=fig.layout.title.font.to_plotly_json())
        fig.update_layout(
            title=None,
            polar=dict(domain=dict(x=[0, 1], y=[_LEGEND_STRIP, 1]),
                       radialaxis=dict(tickangle=0, **(dict(dtick=0.1) if radial.range is not None else {}))),
            legend=dict(orientation='h', x=0.5, y=0, xanchor='center', yanchor='bottom', title=None),
        )
    return fig, mapping_df, n_combos

# ── AG Grid helpers ───────────────────────────────────────────────────────────

def _col_def(name):
    return {'field': name, 'colId': name, 'headerName': name, 'sortable': True,
            'filter': True, 'resizable': True, 'minWidth': 60}

def _table_columns(mapping_df):
    return [_col_def(c) for c in mapping_df.columns]

def _table_columns_ordered(names):
    return [_col_def(c) for c in names]



def _grid_class(light=False):
    return (LIGHT_THEME if light else _DARK_THEME).grid_class

# ── Table export sizing ───────────────────────────────────────────────────────
# Plotly clips a go.Table to the canvas it is handed, so a table taller (or wider)
# than the requested image silently loses its bottom rows and truncates cell text.
# The export therefore measures the table's natural size from its own content and
# scales fonts, row heights and column widths by a single factor, so ?width/?height
# are the size of the exported image and the whole table always fits inside it.

_TBL_FONT_PX  = 11     # natural font size
_TBL_CHAR_W   = 0.65   # monospace advance width, as a fraction of the font size
_TBL_CELL_PAD = 18     # horizontal padding per cell, at _TBL_FONT_PX
_TBL_HEADER_H = 28
_TBL_ROW_H    = 24
_TBL_MARGIN   = 10     # figure margin, per side
_TBL_SLACK    = 1.05   # domain headroom, so cell-metric estimates can't clip a row

def _table_natural_widths(mapping_df):
    """Per-column natural width in px, from the widest header/cell string."""
    widths = []
    for col in mapping_df.columns:
        chars = max([len(str(col))] + [len(str(v)) for v in mapping_df[col]])
        widths.append(chars * _TBL_CHAR_W * _TBL_FONT_PX + _TBL_CELL_PAD)
    return widths

def _centred_span(fraction):
    """Domain edges for a centred span covering `fraction` of an axis.

    Maps [0, 1] onto a pair inside [0, 1]: correctly-rounded arithmetic can take
    neither 1 - f below 0 nor 1 + f above 2, so the result is always a domain
    plotly accepts and never needs clamping back into range.
    """
    return [(1 - fraction) / 2, (1 + fraction) / 2]

def _table_fit(mapping_df, width, height, max_scale=None):
    """Scale factor + trace domain that fit the whole table into a width x height image.

    Returns (column_widths, scale, domain). Column widths are relative, so plotly
    normalises them across the domain; the domain is sized to the table's own aspect
    ratio and centred, which keeps the scaling uniform instead of stretching cells.

    `max_scale` (optional) caps the scale so the table's font never exceeds
    _TBL_FONT_PX * max_scale — used by the paper's ?font= export override, which
    wants text at a chosen physical size rather than a filled frame.  When the
    cap binds, the table is smaller than the frame and stays centred.
    """
    col_w   = _table_natural_widths(mapping_df) or [1.0]
    nat_w   = max(sum(col_w), 1.0) * _TBL_SLACK
    nat_h   = max(_TBL_HEADER_H + len(mapping_df) * _TBL_ROW_H, 1.0) * _TBL_SLACK
    avail_w = max(width  - 2 * _TBL_MARGIN, 1)
    avail_h = max(height - 2 * _TBL_MARGIN, 1)
    # Per-axis fit ratio; the smaller one is the uniform scale, so the axis it came
    # from is filled edge to edge and the other keeps the table's own aspect ratio.
    fit_w, fit_h = avail_w / nat_w, avail_h / nat_h
    scale        = min(fit_w, fit_h, max_scale) if max_scale else min(fit_w, fit_h)
    # The fraction of an axis the scaled table occupies is scale / fit. On the binding
    # axis that divides a float by itself, which IEEE-754 guarantees is exactly 1.0;
    # the algebraically equal nat * scale / avail instead round-trips through a
    # multiply and lands an ulp above 1.0 for ~5% of sizes, putting the centred edge
    # at -1.1e-16 and making plotly reject the domain outright.
    domain = dict(x=_centred_span(scale / fit_w), y=_centred_span(scale / fit_h))
    return col_w, scale, domain

# ── App factory ───────────────────────────────────────────────────────────────

def create_polar_dashboard(available_configs: list, url_prefix='/', views_dir=None):
    global _registry, _VIEWS_DIR
    _registry = available_configs
    if views_dir is not None:
        _VIEWS_DIR = Path(views_dir)
    config_options = [{'label': e['label'], 'value': e['path']} for e in available_configs]
    initial_path   = available_configs[0]['path']
    initial_df     = _load_df(initial_path)
    init_metric, init_k_avg, init_factors = _defaults(initial_df)
    init_numeric, init_factor_cols = _classify_columns(initial_df)
    init_k_avg_cols = _color_by_cols(initial_df)

    register_templates()
    app = Dash(__name__, url_base_pathname=url_prefix)
    app.index_string = _theme_index_string(app.index_string)

    btn = btn_sm
    lbl = {'color': _TEXT_DIM, 'fontSize': '11px', 'marginBottom': '2px'}
    sec = {'display': 'flex', 'flexDirection': 'column', 'gap': '4px'}
    inp = {'width': '100%', 'fontSize': '11px', 'background': _PANEL,
           'color': _TEXT, 'border': f'1px solid {_BORDER}', 'borderRadius': '3px',
           'padding': '3px 6px', 'boxSizing': 'border-box'}
    ctrl_style = {
        'width': '270px', 'flexShrink': '0', 'padding': '10px',
        'background': _SURFACE, 'overflowY': 'auto',
        'display': 'flex', 'flexDirection': 'column', 'gap': '10px',
    }

    def _hdr(title):
        return html.Div(title, style={
            'color': _BORDER, 'fontSize': '9px', 'letterSpacing': '1.5px',
            'textTransform': 'uppercase', 'paddingBottom': '3px',
            'borderBottom': f'1px solid {_BORDER}',
        })

    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        theme_pref_store(),
        dcc.Store(id='panel-expanded', data=True),
        dcc.Store(id='active-view', data=None),
        dcc.Store(id='lazy-mode', data=False),
        dcc.Store(id='view-loaded', data=0),
        dcc.Store(id='url-side-effect', data=None),
        dcc.Store(id='selected-constraints', data=[]),
        dcc.Store(id='cell-style-refresh', data=0),
        dcc.Store(id='live-polar-trace-visibility', data={}),
        dcc.Store(id='saved-polar-trace-visibility', data=None),

        # ── Controls ──────────────────────────────────────────────────────────
        html.Div(id='ctrl-panel', style=ctrl_style, children=[

            html.Button('◀ Hide', id='panel-collapse-btn',
                        style={**btn, 'width': '100%', 'marginBottom': '2px'}),

            html.Div([
                _hdr('Config'),
                html.Div([
                    html.Span('Grid search', style=lbl),
                    html.A('yaml', id='config-file-link', href='#', target='_blank',
                           style={'fontSize': '10px', 'color': _FAINT,
                                  'textDecoration': 'none', 'marginLeft': '6px'}),
                ], style={'display': 'flex', 'alignItems': 'baseline'}),
                dcc.Dropdown(id='config-picker', options=config_options,
                             value=[initial_path], clearable=False, multi=True,
                             style={'fontSize': '11px'}),
            ], style=sec),

            html.Div([
                _hdr('Data'),
                html.Div([
                    html.Div('Computed fields  (name = expr)', style=lbl),
                    dcc.Textarea(id='fields-expr', value='',
                                 placeholder='delta = kqgr_kq - kqgr_gr',
                                 style={**inp, 'height': '52px', 'resize': 'vertical',
                                        'fontFamily': 'monospace'}),
                ], style=sec),
                html.Div([
                    html.Div('Filter  (pandas query)', style=lbl),
                    dcc.Textarea(id='filter-expr', value='',
                                 placeholder='R_k_avg > 2',
                                 style={**inp, 'height': '36px', 'resize': 'vertical',
                                        'fontFamily': 'monospace'}),
                ], style=sec),
                html.Div([
                    html.Button('Clear selection', id='clear-selection-btn',
                                style={**btn, 'width': '100%'}),
                ], style={'marginTop': '2px'}),
                html.Div(id='data-status',
                         style={'fontSize': '10px', 'minHeight': '12px'}),
            ], style=sec),

            html.Div([
                _hdr('Plot'),
                html.Div([
                    html.Div('Metric', style=lbl),
                    dcc.Dropdown(id='metric-col', options=_options(init_numeric),
                                 value=init_metric, clearable=False,
                                 style={'fontSize': '11px'}),
                ], style=sec),
                html.Div([
                    html.Div('Aggregation', style=lbl),
                    dcc.Dropdown(id='agg-mode', clearable=False,
                                 value=_DEFAULT_AGG, options=_AGG_OPTIONS,
                                 style={'fontSize': '11px'}),
                ], style=sec),
                html.Div([
                    html.Div('Radial range', style=lbl),
                    html.Div([
                        dcc.Input(id='radial-min', type='number', debounce=True,
                                  placeholder='auto', style={**inp, 'width': '45%'}),
                        html.Span('–', style={'color': _FAINT, 'padding': '0 4px',
                                              'lineHeight': '26px'}),
                        dcc.Input(id='radial-max', type='number', debounce=True,
                                  placeholder='auto', style={**inp, 'width': '45%'}),
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                ], style=sec),
                html.Div([
                    html.Div('Contours', style=lbl),
                    dcc.Dropdown(id='k-avg-col', options=_options(init_k_avg_cols),
                                 value=init_k_avg, clearable=False,
                                 style={'fontSize': '11px'}),
                ], style=sec),
                html.Div([
                    html.Div('Reference  (dashed circle per contour)', style=lbl),
                    dcc.Dropdown(id='reference-col', options=_options(init_numeric),
                                 value=None, clearable=True, placeholder='none',
                                 style={'fontSize': '11px'}),
                ], style=sec),
                html.Div([
                    html.Div([
                        html.Span('Design factors', style=lbl),
                        html.Div([
                            html.Button('All',    id='factors-all',   style=btn),
                            html.Button('None',   id='factors-none',  style={**btn, 'marginLeft': '3px'}),
                            html.Button('Unique', id='factors-unique',
                                        title='Keep only factors with >1 distinct value',
                                        style={**btn, 'marginLeft': '3px'}),
                        ]),
                    ], style={'display': 'flex', 'justifyContent': 'space-between',
                              'alignItems': 'center'}),
                    dcc.Dropdown(id='factor-cols', options=_options(init_factor_cols),
                                 value=init_factors, multi=True,
                                 style={'fontSize': '11px'}),
                ], style=sec),
                html.Div([
                    html.Div('Ordering', style=lbl),
                    dcc.Dropdown(id='sort-mode',
                                 options=[{'label': 'Gray code',    'value': 'gray'},
                                          {'label': 'Truth table',  'value': 'lex'}],
                                 value='gray', clearable=False,
                                 style={'fontSize': '11px'}),
                    dcc.Checklist(id='center-seam-toggle',
                                  options=[{'label': ' Center 0° in the wrap gap',
                                            'value': 'on'}],
                                  value=['on'],
                                  labelStyle={'fontSize': '11px', 'color': _TEXT_DIM},
                                  style={'marginTop': '4px'}),
                    dcc.Checklist(id='direction-labels-toggle',
                                  options=[{'label': ' Label directions by their levels',
                                            'value': 'on'}],
                                  value=[],
                                  labelStyle={'fontSize': '11px', 'color': _TEXT_DIM}),
                ], style=sec),
            ], style=sec),

            html.Div([
                _hdr('Appearance'),
                make_theme_toggle(
                    labelStyle={'fontSize': '11px', 'color': _TEXT_DIM},
                    style={'lineHeight': '1.6'}),
                html.Button('Lazy mode: OFF', id='lazy-toggle',
                            style={**btn, 'width': '100%', 'marginTop': '4px'}),
            ], style=sec),

            html.Div([
                _hdr('Views'),
                dcc.Input(id='views-dir', value='', debounce=True,
                          placeholder='save directory',
                          style={**inp, 'fontSize': '10px', 'background': _PANEL,
                                 'color': _FAINT, 'border': f'1px solid {_BORDER}'}),
                admin_token_store(),
                html.Div([
                    dcc.Input(id='view-name', placeholder='view name', type='text',
                              debounce=False, style={**inp, 'flex': '1'}),
                    html.Button('Save', id='view-save-btn',
                                style={**btn, 'marginLeft': '4px'}),
                ], style={'display': 'flex', 'alignItems': 'center'}),
                html.Div([
                    dcc.Dropdown(id='view-picker', options=_list_view_options(),
                                 placeholder='Load view…', clearable=True,
                                 style={'flex': '1', 'fontSize': '11px'}),
                    html.Button('Del', id='view-delete-btn',
                                style={**btn, 'marginLeft': '4px'}),
                ], style={'display': 'flex', 'alignItems': 'center'}),
                html.Div(id='view-status',
                         style={'color': _TEXT_DIM, 'fontSize': '10px', 'minHeight': '14px'}),
            ], style=sec),
        ]),

        # ── Right panel: table + graph ─────────────────────────────────────────
        html.Div([
            html.Button('▶ Show', id='panel-expand-btn',
                        style={**btn, 'position': 'absolute', 'top': '8px', 'left': '8px',
                               'zIndex': '10', 'display': 'none'}),

            # Direction mapping table — horizontally resizable
            html.Div([
                dag.AgGrid(
                    id='direction-table',
                    rowData=[],
                    columnDefs=[],
                    defaultColDef={'resizable': True, 'sortable': True,
                                   'filter': True, 'minWidth': 60},
                    dangerously_allow_code=True,
                    dashGridOptions={
                        'animateRows': False,
                        'headerHeight': 28,
                        'rowHeight': 24,
                        'suppressColumnVirtualisation': True,
                    },
                    className=_grid_class(light=False),
                    style={'height': '100%', 'width': '100%'},
                ),
            ], style={'resize': 'horizontal', 'overflow': 'auto',
                      'width': '320px', 'minWidth': '120px', 'maxWidth': '60vw',
                      'borderRight': f'1px solid {_BORDER}',
                      'boxSizing': 'border-box'}),

            # Polar graph — overflow hidden so polar traces stay within bounds
            dcc.Graph(id='polar-graph',
                      style={'flex': '1', 'minWidth': '0', 'height': '100%',
                             'overflow': 'hidden'},
                      config={'displayModeBar': True}),
        ], style={'flex': '1', 'minWidth': '0', 'position': 'relative',
                  'display': 'flex', 'flexDirection': 'row'}),

    ], style={'display': 'flex', 'height': '100vh',
              'background': _SURFACE, 'fontFamily': 'monospace'})

    # ── Callbacks ─────────────────────────────────────────────────────────────

    @app.callback(
        Output('factor-cols',   'value', allow_duplicate=True),
        Input('factors-all',    'n_clicks'),
        Input('factors-none',   'n_clicks'),
        Input('factors-unique', 'n_clicks'),
        State('factor-cols',    'options'),
        State('config-picker',  'value'),
        prevent_initial_call=True,
    )
    def _toggle_factors(_, __, ___, options, config_path):
        if ctx.triggered_id == 'factors-all':
            return [o['value'] for o in (options or [])]
        if ctx.triggered_id == 'factors-none':
            return []
        df = _load_df(config_path)
        if df is None:
            raise PreventUpdate
        return [o['value'] for o in (options or [])
                if o['value'] in df.columns and df[o['value']].nunique() > 1]

    @app.callback(
        Output('ctrl-panel',        'style'),
        Output('panel-expand-btn',  'style'),
        Output('panel-expanded',    'data'),
        Input('panel-collapse-btn', 'n_clicks'),
        Input('panel-expand-btn',   'n_clicks'),
        Input('url',                'search'),
        State('panel-expanded',     'data'),
        prevent_initial_call=False,
    )
    def _toggle_panel(_, __, search, expanded):
        trigger = ctx.triggered_id
        if trigger == 'panel-collapse-btn':
            expanded = False
        elif trigger == 'panel-expand-btn':
            expanded = True
        else:
            raw    = (search or '').lstrip('?').lower()
            params = {k: v for part in raw.split('&') if '=' in part
                      for k, v in [part.split('=', 1)]}
            if params.get('expand') == 'false':
                expanded = False
            else:
                raise PreventUpdate
        panel_style = {**ctrl_style, **({}  if expanded else {'display': 'none'})}
        strip_style = {**btn, 'position': 'absolute', 'top': '8px', 'left': '8px',
                       'zIndex': '10', 'display': 'block' if not expanded else 'none'}
        return panel_style, strip_style, expanded

    # ── Lazy mode toggle ──────────────────────────────────────────────────────

    @app.callback(
        Output('lazy-mode',   'data'),
        Output('lazy-toggle', 'children'),
        Output('lazy-toggle', 'style'),
        Input('lazy-toggle',  'n_clicks'),
        State('lazy-mode',    'data'),
        prevent_initial_call=True,
    )
    def _toggle_lazy(_, active):
        active = not active
        style = {**btn, 'width': '100%', 'marginTop': '4px',
                 'background': _ACCENT if active else _CONTROL,
                 'fontWeight': 'bold' if active else 'normal'}
        return active, f'Lazy mode: {"ON" if active else "OFF"}', style


    # ── Dropdown options from fields ──────────────────────────────────────────

    @app.callback(
        Output('metric-col',  'options', allow_duplicate=True),
        Output('factor-cols', 'options', allow_duplicate=True),
        Output('k-avg-col',   'options', allow_duplicate=True),
        Output('reference-col', 'options', allow_duplicate=True),
        Input('fields-expr',   'n_blur'),
        State('config-picker', 'value'),
        State('fields-expr',   'value'),
        State('filter-expr',   'value'),
        prevent_initial_call=True,
    )
    def _update_options_from_fields(_, config_path, fields_expr, filter_expr):
        _, df, _, _ = _resolve(config_path, fields_expr, filter_expr, [], None)
        if df is None:
            raise PreventUpdate
        numeric, factor_cols = _classify_columns(df)
        return _options(numeric), _options(factor_cols), _options(_color_by_cols(df)), _options(numeric)

    # ── Config switch ─────────────────────────────────────────────────────────

    @app.callback(
        Output('metric-col',  'options', allow_duplicate=True),
        Output('factor-cols', 'options', allow_duplicate=True),
        Output('k-avg-col',   'options', allow_duplicate=True),
        Output('reference-col', 'options', allow_duplicate=True),
        Input('config-picker', 'value'),
        prevent_initial_call=True,
    )
    def _update_dropdowns(config_path):
        df = _load_df(config_path)
        if df is None:
            raise PreventUpdate
        numeric, factor_cols = _classify_columns(df)
        return _options(numeric), _options(factor_cols), _options(_color_by_cols(df)), _options(numeric)

    # ── Main render (stage 1: fields + str filter → table + polar) ────────────

    @app.callback(
        Output('polar-graph',                  'figure'),
        Output('direction-table',              'rowData'),
        Output('direction-table',              'columnDefs'),
        Output('direction-table',              'columnSize'),
        Output('data-status',                  'children'),
        Output('saved-polar-trace-visibility', 'data',     allow_duplicate=True),
        Input('config-picker', 'value'),
        Input('fields-expr',   'n_blur'),
        Input('filter-expr',   'n_blur'),
        Input('metric-col',    'value'),
        Input('agg-mode',      'value'),
        Input('factor-cols',   'value'),
        Input('k-avg-col',       'value'),
        Input('theme-toggle',    'value'),
        Input('view-loaded',     'data'),
        Input('radial-min',      'value'),
        Input('radial-max',      'value'),
        Input('direction-table',      'columnState'),
        Input('sort-mode',            'value'),
        Input('center-seam-toggle',   'value'),
        Input('reference-col',        'value'),
        Input('direction-labels-toggle', 'value'),
        State('fields-expr',                   'value'),
        State('filter-expr',                   'value'),
        State('lazy-mode',                     'data'),
        State('selected-constraints',          'data'),
        State('saved-polar-trace-visibility',  'data'),
        State('live-polar-trace-visibility',   'data'),
        State('url',                           'search'),
        prevent_initial_call='initial_duplicate',
    )
    def _render(config_path, _fb, _flb,
                metric_col, agg_mode, factors, k_avg_col, theme, _vl,
                r_min, r_max, column_state, sort_mode, center_seam_value, reference_col, direction_labels_value,
                fields_expr, filter_expr, lazy_mode, constraints,
                saved_trace_vis, live_trace_vis, url_search):
        center_seam = 'on' in (center_seam_value or [])
        # Skip the initial default render when a URL view is about to load.
        # view-loaded=0 means no view has been explicitly loaded yet; if the URL
        # already names a view, _load_view_from_url→_load_view will re-fire
        # _render with the correct config (and view-loaded=1), so rendering with
        # the default (first) config here is pure waste.
        if _vl == 0:
            name = view_name_from_query(url_search, skip=_URL_FLAGS)
            if name and (_vdir(None) / f'{name}.json').exists():
                raise PreventUpdate
        triggered_props = {t['prop_id'] for t in (ctx.triggered or [])}
        triggered_ids   = {p.split('.')[0] for p in triggered_props}
        _lazy_ids = {'config-picker', 'fields-expr', 'filter-expr',
                     'metric-col', 'agg-mode', 'factor-cols', 'k-avg-col'}
        if lazy_mode and triggered_ids and triggered_ids.issubset(_lazy_ids):
            raise PreventUpdate
        # AG Grid emits an empty columnState as it initialises, before any data exists —
        # ignore that, but only when columnState is the *sole* trigger. Loading a view
        # fires columnState together with every other control in one batch, so keying off
        # its mere presence made a view saved without a column order (column_state: [])
        # abort its own first render and leave the page empty until something else fired.
        col_state_only = triggered_props == {'direction-table.columnState'}
        if col_state_only and not column_state:
            raise PreventUpdate
        raw_df, df, valid_factors, err = _resolve(config_path, fields_expr, filter_expr, factors, k_avg_col)
        if raw_df is None:
            raise PreventUpdate
        pipeline = [(len(raw_df), '')]
        if len(df) != len(raw_df):
            pipeline.append((len(df), 'filter'))
            if df.empty:
                err = err or 'no rows match filter'
        if not metric_col or metric_col not in df.columns:
            return no_update, [], [], no_update, _pipeline_status(pipeline, 'metric not set'), no_update
        if not k_avg_col or k_avg_col not in df.columns:
            return no_update, [], [], no_update, _pipeline_status(pipeline, 'contours not set'), no_update
        if not valid_factors:
            return no_update, [], [], no_update, _pipeline_status(pipeline, err), no_update
        ordered_factors, asc = _factors_ascending_from_state(column_state, valid_factors)
        tv_out = no_update
        fig_dict, mapping_df, n_combos = _build_polar(
            df, ordered_factors, asc, k_avg_col, metric_col,
            agg_mode, theme, r_min, r_max, sort_mode,
            constraints, saved_trace_vis or live_trace_vis,
            center_seam=center_seam, reference_col=reference_col,
            direction_labels='on' in (direction_labels_value or []),
        )
        if saved_trace_vis:
            tv_out = None  # clear one-shot store after applying
        warn = ''
        if len(mapping_df) == 0 and n_combos > 1000:
            pipeline.append((n_combos, 'grouped'))
            warn = f'too many combinations ({n_combos:,}) — reduce factors'
        else:
            pipeline.append((len(mapping_df), 'grouped'))
        records = mapping_df.to_dict('records')
        # Skip columnDefs/columnSize for that same sole-trigger case — avoids feeding
        # AG Grid's own state back and causing a render loop.
        col_defs = no_update if col_state_only else _table_columns(mapping_df)
        col_size = no_update if col_state_only else 'autoSize'
        return fig_dict, records, col_defs, col_size, _pipeline_status(pipeline, err or warn), tv_out

    # ── AG Grid column filter → re-render polar with direction subset ─────────
    # virtualRowData fires when AG Grid's client-side filter changes visible rows.

    @app.callback(
        Output('polar-graph', 'figure', allow_duplicate=True),
        Input('direction-table', 'virtualRowData'),
        State('direction-table', 'rowData'),
        State('config-picker',   'value'),
        State('fields-expr',     'value'),
        State('filter-expr',     'value'),
        State('metric-col',      'value'),
        State('agg-mode',        'value'),
        State('factor-cols',     'value'),
        State('k-avg-col',       'value'),
        State('theme-toggle',    'value'),
        State('radial-min',      'value'),
        State('radial-max',      'value'),
        State('selected-constraints',        'data'),
        State('direction-table',             'columnState'),
        State('sort-mode',                   'value'),
        State('center-seam-toggle',          'value'),
        State('reference-col',               'value'),
        State('direction-labels-toggle',     'value'),
        State('live-polar-trace-visibility', 'data'),
        prevent_initial_call=True,
    )
    def _filter_by_table(virtual_rows, all_rows, config_path,
                         fields_expr, filter_expr, metric_col, agg_mode,
                         factors, k_avg_col, theme, r_min, r_max, constraints, column_state,
                         sort_mode, center_seam_value, reference_col, direction_labels_value, trace_vis):
        # No filter active (virtual == full) — _render already produced the correct figure.
        if not all_rows or virtual_rows is None or len(virtual_rows) == len(all_rows):
            raise PreventUpdate
        _, df, valid_factors, _ = _resolve(config_path, fields_expr, filter_expr, factors, k_avg_col)
        if df is None or not valid_factors:
            raise PreventUpdate
        if not metric_col or metric_col not in df.columns or not k_avg_col or k_avg_col not in df.columns:
            raise PreventUpdate
        ordered_factors, asc = _factors_ascending_from_state(column_state, valid_factors)
        direction_filter = {r['Direction'] for r in virtual_rows if 'Direction' in r} or None
        fig_dict, _, _ = _build_polar(
            df, ordered_factors, asc, k_avg_col, metric_col,
            agg_mode, theme, r_min, r_max, sort_mode,
            constraints, None, direction_filter=direction_filter,
            center_seam='on' in (center_seam_value or []), reference_col=reference_col,
            direction_labels='on' in (direction_labels_value or []),
        )
        if trace_vis:
            apply_trace_vis(fig_dict, trace_vis)
        return fig_dict

    # ── Constraints change → update columnDefs + trigger cell refresh ────────────

    @app.callback(
        Output('direction-table',    'columnDefs',       allow_duplicate=True),
        Output('cell-style-refresh', 'data',             allow_duplicate=True),
        Input('selected-constraints', 'data'),
        State('direction-table',      'columnDefs'),
        State('direction-table',      'columnState'),
        prevent_initial_call=True,
    )
    def _update_cell_styles(constraints, col_defs, column_state):
        if not col_defs:
            raise PreventUpdate
        # columnDefs (State) is frozen in its original order — AG Grid tracks the user's
        # drag-reordering in columnState instead. Re-sort to the current on-screen order
        # before writing columnDefs back, or a click after a drag snaps columns back.
        if column_state:
            field_pos = {c['colId']: i for i, c in enumerate(column_state)}
            col_defs = sorted(col_defs, key=lambda cd: field_pos.get(cd.get('field', ''), len(field_pos)))
        by_col: dict = {}
        for c in (constraints or []):
            col, val = c.get('column_id'), c.get('value')
            if col:
                by_col.setdefault(col, []).append(str(val) if val is not None else '')
        result = []
        for cd in col_defs:
            cd = {k: v for k, v in cd.items() if k != 'cellStyle'}
            field = cd.get('field', '')
            if field in by_col:
                vals_json = json.dumps(by_col[field], ensure_ascii=False)
                cd['cellStyle'] = {'function':
                    f'{vals_json}.includes(String(params.value))'
                    f'?{{fontWeight:"bold",color:"{GRID_HIT_FG}",backgroundColor:"{GRID_HIT_BG}"}}:{{}}'}
            result.append(cd)
        import random as _r
        return result, _r.random()

    clientside_callback(
        """
        function(trigger) {
            console.log('[refresh] cell-style-refresh triggered, trigger=', trigger);
            setTimeout(function() {
                window.dash_ag_grid.getApiAsync('direction-table').then(function(api) {
                    console.log('[refresh] calling redrawRows, api=', !!api);
                    if (api) api.redrawRows();
                });
            }, 50);
            return window.dash_clientside.no_update;
        }
        """,
        Output('url-side-effect',   'data', allow_duplicate=True),
        Input('cell-style-refresh', 'data'),
        prevent_initial_call=True,
    )

    # ── Cell selection → radial wedge overlays ───────────────────────────────

    @app.callback(
        Output('polar-graph', 'figure', allow_duplicate=True),
        Input('selected-constraints', 'data'),
        State('direction-table',      'rowData'),
        State('polar-graph',          'figure'),
        State('live-polar-trace-visibility', 'data'),
        prevent_initial_call=True,
    )
    def _highlight_selected(constraints, row_data, fig, trace_vis):
        if fig is None:
            raise PreventUpdate
        base   = _strip_overlays(fig)
        result = dict(fig, data=base + _make_selection_overlays(constraints, row_data, fig))
        if trace_vis:
            apply_trace_vis(result, trace_vis)
        return result

    @app.callback(
        Output('selected-constraints',  'data',        allow_duplicate=True),
        Output('direction-table',       'getRowStyle', allow_duplicate=True),
        Output('polar-graph',           'figure',      allow_duplicate=True),
        Output('direction-table',       'columnDefs',  allow_duplicate=True),
        Output('cell-style-refresh',    'data',        allow_duplicate=True),
        Input('clear-selection-btn', 'n_clicks'),
        State('polar-graph', 'figure'),
        State('direction-table', 'columnDefs'),
        prevent_initial_call=True,
    )
    def _clear_selection(_, fig, col_defs):
        import random as _r
        plain_defs = [{k: v for k, v in cd.items() if k != 'cellStyle'} for cd in (col_defs or [])]
        return ([], {'function': 'null'},
                dict(fig, data=_strip_overlays(fig)) if fig else no_update,
                plain_defs if plain_defs else no_update,
                _r.random())

    # ── Click polar point → highlight matching table row via getRowStyle ─────

    @app.callback(
        Output('direction-table', 'getRowStyle', allow_duplicate=True),
        Input('polar-graph', 'clickData'),
        prevent_initial_call=True,
    )
    def _highlight_direction(click_data):
        if not click_data:
            raise PreventUpdate
        theta = click_data['points'][0].get('theta')
        if theta is None:
            raise PreventUpdate
        direction = f'{round(float(theta), 1)}°'
        return {'function': f'params.data.Direction === "{direction}" '
                            f'? {{backgroundColor: "{ROW_HIT_BG}", color: "{ROW_HIT_FG}"}} : {{}}'}

    # ── Theme → AG Grid class ─────────────────────────────────────────────────

    @app.callback(
        Output('direction-table', 'className', allow_duplicate=True),
        Input('theme-toggle', 'value'),
        prevent_initial_call=True,
    )
    def _update_table_theme(theme):
        return _grid_class(light=is_light_theme(theme))

    # ── Config file link ──────────────────────────────────────────────────────

    @app.callback(
        Output('config-file-link', 'href'),
        Input('config-picker', 'value'),
    )
    def _update_config_link(config_path):
        p = config_path[0] if isinstance(config_path, list) else config_path
        return f'config-file?path={p}'

    register_config_file_route(app, url_prefix)

    # ── Views: save ───────────────────────────────────────────────────────────

    @app.callback(
        Output('view-status',   'children',  allow_duplicate=True),
        Output('view-picker',   'options',   allow_duplicate=True),
        Output('active-view',   'data',      allow_duplicate=True),
        Output('views-dir',     'value',     allow_duplicate=True),
        Input('view-save-btn',  'n_clicks'),
        State('view-name',      'value'),
        State('views-dir',      'value'),
        State('config-picker',  'value'),
        State('fields-expr',    'value'),
        State('filter-expr',    'value'),
        State('metric-col',     'value'),
        State('agg-mode',       'value'),
        State('factor-cols',    'value'),
        State('k-avg-col',      'value'),
        State('theme-toggle',   'value'),
        State('selected-constraints',  'data'),
        State('direction-table',       'columnState'),
        State('radial-min',                    'value'),
        State('radial-max',                    'value'),
        State('live-polar-trace-visibility',   'data'),
        State('sort-mode',                     'value'),
        State('center-seam-toggle',            'value'),
        State('reference-col',                 'value'),
        State('direction-labels-toggle',       'value'),
        admin_token_state(),
        prevent_initial_call=True,
    )
    def _save_view(_, name, views_dir, config_path,
                   fields_expr, filter_expr, metric_col, agg_mode,
                   factors, k_avg_col, theme, constraints, column_state,
                   r_min, r_max, trace_visibility, sort_mode, center_seam_value, reference_col,
                   direction_labels_value, admin_token):
        if not name or not name.strip():
            return '⚠ Enter a view name.', no_update, no_update, no_update
        name = name.strip()
        vdir = _vdir(views_dir)
        blocked = guard_view_write(vdir, _VIEWS_DIR, _SAFE_ROOTS, admin_token, 4)
        if blocked is not None:
            return blocked
        save_view_json(name, {
            'config_path':          config_path,
            'fields_expr':          fields_expr or '',
            'filter_expr':          filter_expr or '',
            'metric':               metric_col,
            'agg_mode':             agg_mode or _DEFAULT_AGG,
            'factors':              factors or [],
            'contours':             k_avg_col,
            'theme':                theme or [],
            'selected_constraints': constraints or [],
            'column_state':         column_state or [],
            'radial_min':           r_min,
            'radial_max':           r_max,
            'trace_visibility':     trace_visibility or {},
            'sort_mode':            sort_mode or 'gray',
            'center_seam':          center_seam_value if center_seam_value is not None else ['on'],
            'reference':            reference_col,
            'direction_labels':     direction_labels_value or [],
        }, vdir)
        return f'Saved "{name}".', _list_view_options(views_dir), name, str(vdir)

    # ── Views: load ───────────────────────────────────────────────────────────

    @app.callback(
        Output('config-picker',   'value',       allow_duplicate=True),
        Output('fields-expr',     'value',       allow_duplicate=True),
        Output('filter-expr',     'value',       allow_duplicate=True),
        Output('metric-col',      'value',       allow_duplicate=True),
        Output('agg-mode',        'value',       allow_duplicate=True),
        Output('factor-cols',     'options',     allow_duplicate=True),
        Output('factor-cols',     'value',       allow_duplicate=True),
        Output('k-avg-col',       'options',     allow_duplicate=True),
        Output('k-avg-col',       'value',       allow_duplicate=True),
        Output('metric-col',      'options',     allow_duplicate=True),
        Output('theme-toggle',          'value',       allow_duplicate=True),
        Output('selected-constraints',  'data',        allow_duplicate=True),
        Output('direction-table',       'columnState', allow_duplicate=True),
        Output('radial-min',      'value',       allow_duplicate=True),
        Output('radial-max',      'value',       allow_duplicate=True),
        Output('view-status',           'children',     allow_duplicate=True),
        Output('active-view',     'data',        allow_duplicate=True),
        Output('views-dir',       'value',       allow_duplicate=True),
        Output('view-name',                    'value',       allow_duplicate=True),
        Output('view-loaded',                  'data',        allow_duplicate=True),
        Output('saved-polar-trace-visibility', 'data',        allow_duplicate=True),
        Output('live-polar-trace-visibility',  'data',        allow_duplicate=True),
        Output('sort-mode',                    'value',       allow_duplicate=True),
        Output('center-seam-toggle',           'value',       allow_duplicate=True),
        Output('reference-col',                'options',     allow_duplicate=True),
        Output('reference-col',                'value',       allow_duplicate=True),
        Output('direction-labels-toggle',      'value',       allow_duplicate=True),
        Input('view-picker',  'value'),
        State('views-dir',    'value'),
        State('view-loaded',  'data'),
        prevent_initial_call=True,
    )
    def _load_view(name, views_dir, loaded_counter):
        if not name:
            raise PreventUpdate
        vdir = _vdir(views_dir)
        vd = load_view_json(name, vdir)
        if vd is None:
            return (*([no_update] * 15), f'⚠ "{name}" not found.', *([no_update] * 11))
        config_path = vd.get('config_path', [initial_path])
        if isinstance(config_path, str):
            config_path = [config_path]
        valid_paths = {e['path'] for e in available_configs}
        config_path = [p for p in config_path if p in valid_paths] or [initial_path]
        raw_df = _load_df(config_path)
        # Apply saved fields_expr so computed columns appear in options
        df, _ = _prepare_df(raw_df, vd.get('fields_expr', ''), vd.get('filter_expr', ''))
        m, k, f = _defaults(df)
        numeric, factor_cols_avail = _classify_columns(df)
        return (config_path,
                vd.get('fields_expr',  ''),
                vd.get('filter_expr',  ''),
                vd.get('metric')   or m,
                vd.get('agg_mode') or _DEFAULT_AGG,
                _options(factor_cols_avail),
                vd.get('factors') if vd.get('factors') is not None else f,
                _options(_color_by_cols(df)),
                vd.get('contours') or k,
                _options(numeric),
                no_update,   # theme is a user preference, not view state — never restored
                vd.get('selected_constraints', []),
                vd.get('column_state', None),
                vd.get('radial_min'),
                vd.get('radial_max'),
                f'Loaded "{name}".', name, str(vdir), name,
                (loaded_counter or 0) + 1,
                vd.get('trace_visibility') or None,
                vd.get('trace_visibility') or None,
                vd.get('sort_mode', 'gray'),
                vd.get('center_seam', ['on']),
                _options(numeric),
                vd.get('reference'),
                vd.get('direction_labels', []))

    @app.callback(
        Output('view-picker', 'options', allow_duplicate=True),
        Output('view-picker', 'value',   allow_duplicate=True),
        Input('url', 'search'),
        State('active-view', 'data'),
        State('views-dir',   'value'),
        prevent_initial_call='initial_duplicate',
    )
    def _load_view_from_url(search, active, views_dir):
        name = view_name_from_query(search, skip=_URL_FLAGS)
        if not name or name == active:
            raise PreventUpdate
        if (_vdir(views_dir) / f'{name}.json').exists():
            return _list_view_options(views_dir), name
        raise PreventUpdate

    @app.callback(
        Output('view-status', 'children', allow_duplicate=True),
        Output('view-picker', 'options',  allow_duplicate=True),
        Output('view-picker', 'value',    allow_duplicate=True),
        Output('active-view', 'data',     allow_duplicate=True),
        Input('view-delete-btn', 'n_clicks'),
        State('view-picker',     'value'),
        State('views-dir',       'value'),
        admin_token_state(),
        prevent_initial_call=True,
    )
    def _delete_view(_, name, views_dir, admin_token):
        if not name:
            return '⚠ Select a view to delete.', no_update, no_update, no_update
        vdir = _vdir(views_dir)
        blocked = guard_view_write(vdir, _VIEWS_DIR, _SAFE_ROOTS, admin_token, 4)
        if blocked is not None:
            return blocked
        delete_view_json(name, vdir)
        return f'Deleted "{name}".', _list_view_options(views_dir), None, None

    clientside_callback(
        """
        function(view_name) {
            if (view_name) {
                window.history.replaceState(
                    null, '', window.location.pathname + '?' + view_name);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('url-side-effect', 'data', allow_duplicate=True),
        Input('active-view', 'data'),
        prevent_initial_call=True,
    )

    # Track legend-click visibility state client-side so save_view captures it.
    # Also re-applies visibility when a server render resets traces to visible —
    # if merged says a trace should be 'legendonly' but newVis has it as true,
    # call Plotly.restyle directly to restore it (suppressing the restyle listener
    # during the programmatic call to avoid a feedback loop).
    clientside_callback(
        """
        function(figure, prevVis) {
            function captureVis(data) {
                var vis = {};
                (data || []).forEach(function(t) {
                    if (t.name !== undefined && t.name !== null) {
                        var v = t.visible;
                        vis[String(t.name) + '|' + (t.mode || 'markers')] =
                            (v === undefined || v === true) ? true : v;
                    }
                });
                return vis;
            }
            function attachListener(gd) {
                gd.removeAllListeners('plotly_restyle');
                gd.on('plotly_restyle', function() {
                    dash_clientside.set_props('live-polar-trace-visibility',
                        {data: captureVis(gd.data)});
                });
            }
            setTimeout(function() {
                var gd = document.querySelector('#polar-graph .js-plotly-plot');
                if (gd) attachListener(gd);
            }, 150);

            var newVis = captureVis((figure || {}).data);
            var prev = prevVis || {};
            var merged = {};
            for (var k in newVis) {
                merged[k] = (newVis[k] === true && prev[k] === 'legendonly')
                    ? 'legendonly' : newVis[k];
            }

            // If the new figure reset any trace that should stay hidden, fix it now.
            var needsFix = Object.keys(merged).some(function(k) {
                return merged[k] === 'legendonly' && newVis[k] !== 'legendonly';
            });
            if (needsFix) {
                setTimeout(function() {
                    var gd = document.querySelector('#polar-graph .js-plotly-plot');
                    if (!gd || !gd.data) return;
                    var update = {visible: gd.data.map(function(t) {
                        var key = String(t.name || '') + '|' + (t.mode || 'markers');
                        return merged[key] !== undefined ? merged[key] : true;
                    })};
                    gd.removeAllListeners('plotly_restyle');
                    Plotly.restyle(gd, update).then(function() { attachListener(gd); });
                }, 0);
            }

            return merged;
        }
        """,
        Output('live-polar-trace-visibility', 'data'),
        Input('polar-graph', 'figure'),
        State('live-polar-trace-visibility', 'data'),
        prevent_initial_call=True,
    )

    # ── Image export ──────────────────────────────────────────────────────────

    def _resolve_for_export(view_data, theme_override):
        """Shared setup for both export targets: load df, defaults, theme."""
        config_path  = view_data.get('config_path', [initial_path])
        if isinstance(config_path, str):
            config_path = [config_path]
        factor_cols  = view_data.get('factors', [])
        k_avg_col    = view_data.get('contours')
        fields_expr  = view_data.get('fields_expr', '')
        filter_expr  = view_data.get('filter_expr', '')
        column_state = view_data.get('column_state') or []
        _, df, valid_factors, _ = _resolve(config_path, fields_expr, filter_expr, factor_cols, k_avg_col)
        if df is None:
            raise ValueError(f'Config not available: {config_path}')
        m, k, f = _defaults(df)
        factors = valid_factors or f
        ordered_factors, ascending = _factors_ascending_from_state(column_state, factors)
        theme   = theme_override if theme_override is not None else view_data.get('theme', [])
        return (df,
                ordered_factors,
                ascending,
                k_avg_col or k,
                view_data.get('metric', m),
                view_data.get('agg_mode', _DEFAULT_AGG),
                view_data.get('radial_min'),
                view_data.get('radial_max'),
                theme)

    def _reconstruct(view_data, width, height, theme_override):
        df, factors, ascending, k_col, metric_col, agg_mode, r_min, r_max, theme = \
            _resolve_for_export(view_data, theme_override)
        fig_dict, _, _ = _build_polar(
            df, factors, ascending, k_col, metric_col,
            agg_mode, theme, r_min, r_max, view_data.get('sort_mode'),
            view_data.get('selected_constraints'), view_data.get('trace_visibility'),
            center_seam='on' in (view_data.get('center_seam') or ['on']),
            reference_col=view_data.get('reference'),
            direction_labels='on' in (view_data.get('direction_labels') or []),
        )
        fig = go.Figure(fig_dict)
        fig.update_layout(width=width, height=height)
        return fig

    def _reconstruct_table(view_data, width, height, theme_override, font_px=None):
        df, factors, ascending, k_col, metric_col, agg_mode, r_min, r_max, theme = \
            _resolve_for_export(view_data, theme_override)
        _, mapping_df, _ = make_polar_figure(
            df,
            factors    = factors,
            ascending  = ascending,
            agg_mode   = agg_mode,
            metric_col = metric_col,
            k_avg_col  = k_col,
            r_min      = r_min,
            r_max      = r_max,
            sort_mode  = view_data.get('sort_mode'),
            center_seam = 'on' in (view_data.get('center_seam') or ['on']),
        )
        th          = resolve_theme(theme)
        bg          = th.paper
        header_fill = th.table_header_fill
        cell_fill   = th.table_cell_fill
        font_color  = th.table_font
        # ?font= override: cap the fit so the table's font lands at font_px
        # (shrinking only if the table genuinely doesn't fit at that size).
        max_scale  = font_px / _TBL_FONT_PX if font_px else None
        col_w, scale, domain = _table_fit(mapping_df, width, height, max_scale=max_scale)
        font_px  = max(_TBL_FONT_PX * scale, 1)
        header_h = _TBL_HEADER_H * scale
        row_h    = _TBL_ROW_H * scale
        fig = go.Figure(data=[go.Table(
            columnwidth=col_w,
            domain=domain,
            header=dict(
                values=list(mapping_df.columns),
                fill_color=header_fill, line_color=th.table_line,
                font=dict(color=font_color, size=font_px, family='monospace'),
                align='left',
                height=header_h,
            ),
            cells=dict(
                values=[mapping_df[c].tolist() for c in mapping_df.columns],
                fill_color=cell_fill, line_color=th.table_line,
                font=dict(color=font_color, size=font_px, family='monospace'),
                align='left',
                height=row_h,
            ),
        )])
        fig.update_layout(
            width=width, height=height, template=th.template,
            paper_bgcolor=bg,
            margin=dict(l=_TBL_MARGIN, r=_TBL_MARGIN, t=_TBL_MARGIN, b=_TBL_MARGIN),
        )
        return fig

    # ── Cell click → toggle constraint + refresh styles (all clientside) ────────
    # Clientside ensures State is always fresh — eliminates the race condition
    # that caused rapid successive clicks to produce stale accumulated constraints.

    clientside_callback(
        """
        function(cell_clicked, current) {
            console.log('[toggle] cellClicked:', cell_clicked);
            console.log('[toggle] current constraints:', JSON.stringify(current));
            if (!cell_clicked) return window.dash_clientside.no_update;
            var col_id = cell_clicked.colId;
            var value  = cell_clicked.value != null ? String(cell_clicked.value) : '';
            if (!col_id) return window.dash_clientside.no_update;
            current = current || [];
            var idx = current.findIndex(function(c) {
                return c.column_id === col_id && String(c.value) === value;
            });
            console.log('[toggle] idx found:', idx, 'col_id:', col_id, 'value:', value);
            var updated = idx >= 0
                ? current.filter(function(_, i) { return i !== idx; })
                : current.concat([{column_id: col_id, value: value}]);
            console.log('[toggle] updated constraints:', JSON.stringify(updated));
            return updated;
        }
        """,
        Output('selected-constraints', 'data', allow_duplicate=True),
        Input('direction-table', 'cellClicked'),
        State('selected-constraints', 'data'),
        prevent_initial_call=True,
    )


    register_export_route(app, {'polar': _reconstruct, 'table': _reconstruct_table},
                       safe_roots=_SAFE_ROOTS,
                       default_search_dirs=[_VIEWS_DIR])

    register_view_refresh(app, 'panel-expand-btn', _list_view_options)
    register_theme_attribute(app)
    register_admin_token_capture(app)

    return app

