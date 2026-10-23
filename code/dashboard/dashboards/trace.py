"""
Interactive activity-trace dashboard for Boolean Reservoir history data.

Layout
------
  Config ▼  Run ▼  [Load]           ← data selector
  Filter Nodes: ___  Phases: ☑☑☑☑   ← node filter + per-phase toggles
  Select Cells: ___  [Select] [Deselect]
  FILTER ☑ Hide isolated  ☑ Hide static
  PLOT   ☑ Light  ☑ State legend  ☑ Phase legend  ☑ Concat cols
         ☑ Parents  ☑ Children  ☑ Renum X  ☑ Renum Y
  ─── heatmap ───────────────────────────────────────
  ─── adjacency + LUT info panel ──────────────────────
  ▶ Views
"""
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
import orjson

from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from .theme import (
    resolve as _resolve_theme, register_templates, register_theme_attribute,
    theme_pref_store, index_string as _theme_index_string,
    SURFACE as _SURFACE, PANEL as _PANEL, CONTROL as _CONTROL,
    TEXT as _TEXT, BORDER as _BORDER, TEXT_DIM as _TEXT_DIM,
    FAINT as _FAINT, DANGER as _DANGER,
)
from .utils import (
    register_export_route, apply_trace_vis,
    _views_dir, _view_dropdown_options, is_dataset_hidden,
    guard_view_write, DEFAULT_SAFE_ROOTS, admin_token_store,
    register_admin_token_capture, admin_token_state,
    delete_view_json, VIEWS_ROOT, view_name_from_query,
)

# ── Styling ───────────────────────────────────────────────────────────────────

_S       = {'fontFamily': 'monospace', 'fontSize': '13px', 'color': _TEXT}
_INP     = {**_S, 'background': _PANEL, 'border': f'1px solid {_BORDER}',
            'borderRadius': '3px', 'padding': '4px 6px'}
_BTN     = {**_S, 'background': _CONTROL, 'border': f'1px solid {_BORDER}',
            'borderRadius': '3px', 'padding': '4px 10px', 'cursor': 'pointer'}
_LBL     = {**_S, 'color': _TEXT_DIM, 'marginRight': '12px'}
_SECTION = {'fontFamily': 'monospace', 'fontSize': '11px', 'fontWeight': 'bold',
            'color': _TEXT_DIM, 'letterSpacing': '0.5px', 'marginRight': '6px'}

# Phase colors matching visualization.py's sns.color_palette("husl", 4)
_PHASE_COLORS = {
    'init':            '#4fa8d5',  # blue
    'input_layer':     '#5cb87a',  # green
    'reservoir_layer': '#f0a030',  # orange
    'output_layer':    '#d95f5f',  # red
}
_EXTRA_COLORS = ['#9b59b6', '#20b2aa', '#e67e22', '#1abc9c', '#e74c3c']

# Toggle group membership
_FILTER_TOGGLE_IDS    = frozenset({'no_isolated', 'no_static'})
_DEFAULT_FILTER_TOGGLES: list = []
_DEFAULT_PLOT_TOGGLES  = ['state_legend', 'phase_legend', 'concat',
                          'highlight_row', 'highlight_col',
                          'highlight_parents', 'highlight_children']


def _phase_color(phase: str, unique_phases: list) -> str:
    if phase in _PHASE_COLORS:
        return _PHASE_COLORS[phase]
    idx = unique_phases.index(phase)
    return _EXTRA_COLORS[idx % len(_EXTRA_COLORS)]


def _binary_colorscale(theme):
    """Sharp step at 0.5. Dark: 0→black, 1→white. Light: 0→white, 1→black."""
    lo, hi = theme.binary_lo, theme.binary_hi
    return [[0.000, lo], [0.499, lo], [0.501, hi], [1.000, hi]]


def _merge_toggles(filter_toggles, plot_toggles) -> list:
    return list(filter_toggles or []) + list(plot_toggles or [])


def _split_toggles(ui_toggles) -> tuple[list, list]:
    """Split a flat ui_toggles list back into (filter_toggles, plot_toggles)."""
    if not ui_toggles:
        return list(_DEFAULT_FILTER_TOGGLES), list(_DEFAULT_PLOT_TOGGLES)
    ft = [v for v in ui_toggles if v in _FILTER_TOGGLE_IDS]
    pt = [v for v in ui_toggles if v not in _FILTER_TOGGLE_IDS]
    return ft, pt


# ── Dashboard class ───────────────────────────────────────────────────────────

class TracePlotDashboard:
    def __init__(
        self,
        config_base_path: str = '/code/config',
        out_base: str = '/out',
        safe_roots: list = None,
        initial_views_dir: str = str(VIEWS_ROOT / 'trace'),
        url_prefix: str = '/',
    ):
        self._config_base  = Path(config_base_path)
        self._out_base     = Path(out_base)
        self._safe_roots   = [Path(r) for r in (safe_roots or DEFAULT_SAFE_ROOTS)]
        self._initial_vdir = initial_views_dir
        self._url_prefix   = url_prefix
        self._cache: dict  = {}

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _list_configs(self):
        if not self._config_base.exists():
            return []
        from project.boolean_reservoir.code.parameter import load_yaml_config
        result = []
        for f in sorted(self._config_base.rglob('*.yaml')):
            if is_dataset_hidden(f.relative_to(self._config_base)):
                continue
            try:
                params   = load_yaml_config(str(f))
                run_base = self._resolve_run_base(str(params.logging.out_path))
                if self._find_run_dirs(run_base):
                    result.append({
                        'label': str(f.relative_to(self._config_base)),
                        'value': str(f.relative_to(self._config_base)),
                    })
            except Exception:
                pass
        return result

    def _resolve_run_base(self, out_path_str: str) -> Path:
        p = Path(out_path_str)
        if p.parts and p.parts[0] == 'out':
            p = Path(*p.parts[1:])
        return self._out_base / p

    def _find_run_dirs(self, run_base: Path) -> list:
        def _has_history(p):
            h = p / 'history'
            return h.exists() and any(h.glob('*.npy'))
        candidates = []
        if not run_base.exists():
            return candidates
        if _has_history(run_base):
            candidates.append(run_base)
        for sub in sorted(run_base.iterdir()):
            if not sub.is_dir():
                continue
            if _has_history(sub):
                candidates.append(sub)
            if sub.name == 'runs':
                for run in sorted(sub.iterdir()):
                    if run.is_dir() and _has_history(run):
                        candidates.append(run)
        return sorted(set(candidates), key=lambda x: x.stat().st_mtime, reverse=True)

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _build_query_grid(self, node_ids, col_indices, cached) -> pd.DataFrame:
        """Build a (node_id × time) grid with node_meta + phase for querying.

        Columns: node_id, time, phase, + all node_meta columns.
        Both filter expressions and selection expressions query this grid.
        """
        if not node_ids or not col_indices:
            return pd.DataFrame()
        phases = [cached['phases'][i] for i in col_indices]
        grid = pd.DataFrame({
            'node_id': np.repeat(node_ids, len(col_indices)),
            'time':    np.tile(col_indices, len(node_ids)),
            'phase':   np.tile(phases, len(node_ids)),
        })
        return grid.merge(cached['node_meta'], on='node_id', how='left')

    def _get_visible_data(self, run_path, node_filter, active_phases,
                          active_node_types=None, ui_toggles=None):
        """Returns (data_dict | None, error_str). node_meta owns the node ID list."""
        cached = self._cache.get(run_path)
        if not cached:
            return None, ''
        toggles = set(ui_toggles or [])

        all_node_ids = cached['node_meta']['node_id'].tolist()
        col_indices  = [i for i, p in enumerate(cached['phases']) if p in active_phases]

        filter_error = ''
        if node_filter and node_filter.strip():
            try:
                grid   = self._build_query_grid(all_node_ids, col_indices, cached)
                result = grid.query(node_filter)
                hit_nids = set(result['node_id'].unique())
                hit_cols = set(result['time'].unique())
                node_ids    = [n for n in all_node_ids if n in hit_nids]
                col_indices = [i for i in col_indices  if i in hit_cols]
            except Exception as e:
                filter_error = str(e)
                node_ids = all_node_ids
        else:
            node_ids = all_node_ids

        # Filter by node type (checklist)
        if active_node_types:
            type_set = set(active_node_types)
            type_col = cached['node_meta'].set_index('node_id')['node_type']
            node_ids = [n for n in node_ids if type_col.get(n, 'reservoir') in type_set]

        if 'no_isolated' in toggles:
            iso = set(cached['node_meta']
                      .loc[cached['node_meta']['degree_in'] == 0, 'node_id'])
            node_ids = [n for n in node_ids if n not in iso]

        if not col_indices or not node_ids:
            reason = 'no nodes match filter' if not node_ids else 'no phases selected'
            return None, filter_error or reason

        z = cached['z'][node_ids, :][:, col_indices]

        if 'no_static' in toggles and z.shape[1] > 1:
            active_mask = z.max(axis=1) != z.min(axis=1)
            node_ids = [n for n, keep in zip(node_ids, active_mask) if keep]
            z = z[active_mask, :]

        if not node_ids:
            return None, filter_error or 'all nodes filtered out'

        return {
            'run_path':          run_path,
            'z':                 z,
            'node_ids':          node_ids,
            'times':             col_indices,
            'all_phases_stream': [cached['phases'][i] for i in col_indices],
            'unique_phases':     cached['unique_phases'],
        }, filter_error

    def _get_selection_mask(self, data, select_expr):
        if not select_expr:
            return []
        try:
            cached = self._cache[data['run_path']]
            grid   = self._build_query_grid(data['node_ids'], data['times'], cached)
            return grid.query(select_expr)[['node_id', 'time']].values.tolist()
        except Exception as e:
            import sys
            print(f'[selection] query failed ({select_expr!r}): {e}', file=sys.stderr, flush=True)
            return []

    # ── Cache loading ─────────────────────────────────────────────────────────

    def _load_to_cache(self, run_path_str: str, config_path: str = None) -> str:
        from project.boolean_reservoir.code.utils.reservoir_utils import (
            BatchedTensorHistoryWriter, SaveAndLoadModel)
        from project.boolean_reservoir.code.reservoir import BooleanReservoir

        run_dir = Path(run_path_str)
        ckpt    = run_dir / 'history' / 'checkpoint'

        # History: always loaded directly from the run directory
        _, history, expanded_meta, combined_meta = BatchedTensorHistoryWriter(
            run_dir / 'history').reload_history()

        from project.boolean_reservoir.code.parameter import load_yaml_config
        ld = {}
        if ckpt.exists():
            ld = SaveAndLoadModel.load_from_path_dict_or_checkpoint_folder(
                checkpoint_path=ckpt, load_key_include_set={'graph', 'wiring', 'lut', 'parameters'})
        params = ld.pop('parameters', None) or (load_yaml_config(config_path) if config_path else None)
        if params is None:
            raise ValueError(f"No parameters: select a config or ensure the checkpoint has parameters.yaml")
        model = BooleanReservoir(params=params, load_dict=ld)

        G   = model.graph
        lut = model.lut

        n_nodes  = history.shape[1]
        node_ids = range(n_nodes)

        ni_end = model.input_slice.stop
        nr_end = model.res_slice.stop
        def _ntype(n):
            if n < ni_end:  return 'input'
            if n < nr_end:  return 'reservoir'
            return 'output'

        node_meta = pd.DataFrame({
            'node_id':       list(node_ids),
            'node_type':     [_ntype(n) for n in node_ids],
            'degree_in':     [G.in_degree(n)  for n in node_ids],
            'degree_out':    [G.out_degree(n) for n in node_ids],
            'has_self_loop': [G.has_edge(n, n) for n in node_ids],
        })
        adj = {n: {'p': list(G.predecessors(n)), 's': list(G.successors(n))}
               for n in G.nodes()}

        stride        = int(combined_meta['samples'].iloc[0]) if len(combined_meta) else 1
        agg           = expanded_meta.iloc[0::stride]
        z             = history[agg.index.values].T.astype(np.float32)
        phases        = agg['phase'].values.tolist()
        unique_phases = list(dict.fromkeys(phases))

        self._cache[run_path_str] = {
            'z': z, 'n_nodes': n_nodes, 'phases': phases,
            'unique_phases': unique_phases, 'node_meta': node_meta,
            'adj': adj, 'lut': lut,
        }
        cols = list(node_meta.columns)
        return f'Loaded: {n_nodes} nodes × {z.shape[1]} steps | filter cols: {cols}'

    # ── Figure builder ────────────────────────────────────────────────────────

    def _build_figure(self, data: dict, ui_toggles: list,
                      click, selection: list, trace_visibility: dict = None):
        """Returns (go.Figure, info_panel_children)."""
        toggles       = set(ui_toggles or [])
        tv            = trace_visibility or {}
        hidden_phases = {ph for ph in data.get('unique_phases', [])
                         if tv.get(f'{ph}|lines') == 'legendonly'}
        th            = _resolve_theme(toggles)
        # trace deliberately paints its figure canvas the PAGE colour (`surface`),
        # not the plotly canvas colour (`paper`) — do not "fix" this.
        bg            = th.surface
        text_color    = th.overlay_text
        show_state         = 'state_legend'       in toggles
        show_ph_leg        = 'phase_legend'       in toggles
        concat_mode        = 'concat'             in toggles
        highlight_row      = 'highlight_row'      in toggles
        highlight_col      = 'highlight_col'      in toggles
        highlight_parents  = 'highlight_parents'  in toggles
        highlight_children = 'highlight_children' in toggles
        renumber_x         = 'renumber_x'         in toggles
        renumber_y         = 'renumber_y'         in toggles

        # ── X coordinate mapping ─────────────────────────────────────────────
        n_cols = len(data['times'])
        if concat_mode:
            x_pos    = [str(i) for i in range(n_cols)]
            x_labels = [str(t) for t in data['times']]  # original time values
            cat_arr  = x_pos
        else:
            x_pos    = [str(t) for t in data['times']]
            max_t    = max(data['times']) if data['times'] else 0
            cat_arr  = [str(i) for i in range(max_t + 1)]
            x_labels = None

        x_cat_idx = [int(xp) for xp in x_pos]  # Plotly internal index per column

        # Tick labels before thinning
        if renumber_x:
            tick_texts = [str(i) for i in range(n_cols)]
        elif concat_mode and x_labels:
            tick_texts = x_labels
        else:
            tick_texts = x_pos  # category string == time value

        # Auto-thin: pick the smallest step from a nice sequence so ≤20 ticks show
        _STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        step = next((s for s in _STEPS if n_cols <= s * 20), _STEPS[-1])
        x_tick_kw = dict(tickmode='array',
                         tickvals=x_pos[::step],
                         ticktext=tick_texts[::step])

        # ── Y coordinate mapping (always compact sequential) ─────────────────
        node_ids    = data['node_ids']
        n_nodes_vis = len(node_ids)
        step_y      = max(1, n_nodes_vis // 10)
        y_pos       = list(range(n_nodes_vis))
        orig_to_y   = {n: i for i, n in enumerate(node_ids)}

        if renumber_y:
            t_vals    = y_pos[::step_y]
            y_tick_kw = dict(tickmode='array', tickvals=t_vals,
                             ticktext=[str(v) for v in t_vals])
        else:
            y_tick_kw = dict(tickmode='array',
                             tickvals=y_pos[::step_y],
                             ticktext=[str(node_ids[i]) for i in y_pos[::step_y]])

        # ── Colorscale ───────────────────────────────────────────────────────
        unique_vals = set(np.unique(data['z']))
        is_binary   = unique_vals <= {0.0, 1.0}
        if is_binary:
            colorscale = _binary_colorscale(th)
            colorbar   = dict(tickvals=[0, 1], ticktext=['0', '1'],
                              title='State', thickness=14, len=0.45)
        else:
            # Light: 0→white, 1→black ('Greys').  Dark: 0→black, 1→white ('Greys_r').
            colorscale = th.greys
            colorbar   = dict(title='Value', thickness=14, len=0.45)

        # ── Heatmap ──────────────────────────────────────────────────────────
        fig = go.Figure()
        node_ids_col = np.array(node_ids, dtype=int)[:, None]
        fig.add_trace(go.Heatmap(
            z=data['z'],
            x=x_pos,
            y=y_pos,
            customdata=np.broadcast_to(node_ids_col, data['z'].shape),
            coloraxis='coloraxis',
            showlegend=False,
            hovertemplate='<b>Node %{customdata}</b>  t=%{x}<br>State: %{z:.0f}<extra></extra>',
            name='states',
        ))

        # ── Shapes + annotations ─────────────────────────────────────────────
        shapes      = []
        annotations = []
        uph         = data['unique_phases']

        if show_ph_leg:
            for j, ph in enumerate(data['all_phases_stream']):
                if ph in hidden_phases:
                    continue
                xi = x_cat_idx[j]
                shapes.append(dict(
                    type='rect', xref='x', yref='paper',
                    x0=xi - 0.5, x1=xi + 0.5,
                    y0=1.01, y1=1.05,
                    fillcolor=_phase_color(ph, uph),
                    line=dict(width=0), layer='above',
                ))

        if show_ph_leg:
            sq_w, sq_h = 0.012, 0.020
            gap        = 0.006
            char_w     = 0.0075
            leg_y      = 1.10
            cursor_x   = 0.02
            for ph in uph:
                if ph in hidden_phases:
                    continue
                color = _phase_color(ph, uph)
                shapes.append(dict(
                    type='rect', xref='paper', yref='paper',
                    x0=cursor_x, x1=cursor_x + sq_w,
                    y0=leg_y - sq_h / 2, y1=leg_y + sq_h / 2,
                    fillcolor=color, line=dict(width=0), layer='above',
                ))
                annotations.append(dict(
                    xref='paper', yref='paper',
                    x=cursor_x + sq_w + gap, y=leg_y,
                    text=ph, showarrow=False,
                    font=dict(size=11, color=text_color),
                    xanchor='left', yanchor='middle',
                ))
                cursor_x += sq_w + gap + len(ph) * char_w + 0.018

        # ── Click highlights + info panel ────────────────────────────────────
        # X: extend to cell edges (ordinal ± 0.5)
        x_row_min = x_cat_idx[0]  - 0.5
        x_row_max = x_cat_idx[-1] + 0.5
        # Y: Voronoi bounds match Plotly's own cell-sizing rule exactly
        y_bounds  = self._voronoi_y_bounds(y_pos)

        def _add_row_highlight(node, color, width, fill):
            dy = orig_to_y.get(node)
            if dy is None:
                return
            shapes.append(self._row_rect(*y_bounds[dy], x_row_min, x_row_max,
                                         color, width, fill))

        info = 'Click a node to see adjacency and LUT.'
        if click:
            c_y_disp   = int(click['points'][0]['y'])   # ensure int — Plotly may send float
            c_time     = click['points'][0]['x']
            c_time_str = str(c_time)
            c_node     = node_ids[c_y_disp]
            c_entry    = self._cache[data['run_path']]
            adj        = c_entry['adj'].get(c_node, {'p': [], 's': []})

            phase_at_click = next(
                (data['all_phases_stream'][j]
                 for j, xp in enumerate(x_pos) if xp == c_time_str),
                None)
            clicked_col_j  = next(
                (j for j, xp in enumerate(x_pos) if xp == c_time_str), None)

            if highlight_row:
                _add_row_highlight(c_node, th.hl_row, 2, th.hl_row_fill)
            if highlight_col and clicked_col_j is not None:
                all_b  = list(y_bounds.values())
                col_y0 = min(b[0] for b in all_b)
                col_y1 = max(b[1] for b in all_b)
                shapes.append(self._cell_rect(x_cat_idx[clicked_col_j], col_y0, col_y1,
                                              th.hl_row, 1, th.hl_col_fill))
            if highlight_parents:
                for p in adj['p']:
                    _add_row_highlight(p, th.hl_parent, 1, th.hl_parent_fill)
            if highlight_children:
                for s in adj['s']:
                    _add_row_highlight(s, th.hl_child, 1, th.hl_child_fill)

            nm        = c_entry['node_meta'].set_index('node_id')
            node_type = nm.at[c_node, 'node_type'] if c_node in nm.index else '?'
            lut_text  = self._build_lut_info(c_node, clicked_col_j, data, c_entry)

            lines = [
                f'node:     {c_node}',
                f'type:     {node_type}',
                f'time:     {c_time}',
                f'phase:    {phase_at_click or "?"}',
                f'parents:  {adj["p"]}',
                f'children: {adj["s"]}',
            ]
            if lut_text:
                lines += ['', lut_text]
            info = html.Pre('\n'.join(lines),
                            style={**_S, 'margin': '0', 'whiteSpace': 'pre-wrap'})

        # ── Selection highlights (blob outline) ──────────────────────────────
        time_to_x_idx = {data['times'][j]: x_cat_idx[j] for j in range(len(data['times']))}
        sel_set = set()
        sel_cells = []
        for s_node, s_time in (selection or []):
            s_xi = time_to_x_idx.get(s_time)
            s_dy = orig_to_y.get(s_node)
            if s_xi is not None and s_dy is not None:
                sel_set.add((s_xi, s_dy))
                sel_cells.append((s_xi, s_dy))

        _SC = th.sel_stroke
        _SF = th.sel_fill
        for xi, dy in sel_cells:
            y0, y1 = y_bounds[dy]
            shapes.append(dict(type='rect', xref='x', yref='y',
                               x0=xi-0.5, x1=xi+0.5, y0=y0, y1=y1,
                               line=dict(width=0), fillcolor=_SF, layer='above'))
            if (xi-1, dy) not in sel_set:
                shapes.append(dict(type='line', xref='x', yref='y',
                                   x0=xi-0.5, x1=xi-0.5, y0=y0, y1=y1,
                                   line=dict(color=_SC, width=1), layer='above'))
            if (xi+1, dy) not in sel_set:
                shapes.append(dict(type='line', xref='x', yref='y',
                                   x0=xi+0.5, x1=xi+0.5, y0=y0, y1=y1,
                                   line=dict(color=_SC, width=1), layer='above'))
            if (xi, dy-1) not in sel_set:
                shapes.append(dict(type='line', xref='x', yref='y',
                                   x0=xi-0.5, x1=xi+0.5, y0=y0, y1=y0,
                                   line=dict(color=_SC, width=1), layer='above'))
            if (xi, dy+1) not in sel_set:
                shapes.append(dict(type='line', xref='x', yref='y',
                                   x0=xi-0.5, x1=xi+0.5, y0=y1, y1=y1,
                                   line=dict(color=_SC, width=1), layer='above'))

        # ── Ghost traces for phase legend (one per phase, no data) ───────────
        if show_ph_leg:
            for ph in data['unique_phases']:
                color = _phase_color(ph, data['unique_phases'])
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='lines',
                    line=dict(color=color, width=4),
                    name=ph, legendgroup=ph, showlegend=True,
                ))
            apply_trace_vis(fig, tv or None)

        # ── Layout ────────────────────────────────────────────────────────────
        top_margin = 80 if show_ph_leg else 30
        fig.update_layout(
            template=th.template, paper_bgcolor=bg, plot_bgcolor=bg,
            coloraxis=dict(
                colorscale=colorscale, cmin=0, cmax=1,
                showscale=show_state, colorbar=colorbar,
            ),
            showlegend=show_ph_leg,
            xaxis=dict(title='Time step', type='category', showgrid=False,
                       categoryorder='array', categoryarray=cat_arr,
                       **x_tick_kw),
            yaxis=dict(title='Node ID',
                       range=[n_nodes_vis - 0.5, -0.5],
                       **y_tick_kw),
            shapes=shapes,
            annotations=annotations,
            margin=dict(t=top_margin, b=70, l=60, r=20),
            uirevision='constant',
        )
        return fig, info

    # ── LUT truth table ───────────────────────────────────────────────────────

    def _build_lut_info(self, c_node: int, clicked_col_j,
                        data: dict, cache_entry: dict) -> str:
        """Return a plain-text LUT section string, or '' if no LUT."""
        lut = cache_entry.get('lut')
        if lut is None:
            return ''

        preds = cache_entry['adj'].get(c_node, {'p': [], 's': []})['p']
        k     = len(preds)

        # Predecessor states at the clicked column.
        # Use the full cached z (predecessors may be filtered from the visible view).
        pred_states: list = []
        for p in preds:
            if clicked_col_j is not None:
                t_global = data['times'][clicked_col_j]   # index into cached z columns
                pred_states.append(int(cache_entry['z'][p, t_global]))
            else:
                pred_states.append(None)

        all_known = all(s is not None for s in pred_states)

        header = f'LUT  node={c_node}  k={k}'

        if k == 0:
            body = f'   → {int(lut.row(c_node)[0])}'
        elif not all_known:
            body = '  (some predecessors not in view)'
        else:
            from project.boolean_reservoir.code.lut import lut_lookup
            bits = ''.join(str(s) for s in pred_states)
            body = f'  {bits} → {lut_lookup(lut, c_node, pred_states)}'

        return f'{header}\n{body}'

    # ── Shape helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _cell_rect(x_idx, y0, y1, color, width, fill=None):
        """x_idx: ordinal category index. y0, y1: Voronoi cell bounds (same as _row_rect)."""
        x_idx = int(x_idx)
        return dict(type='rect', xref='x', yref='y',
                    x0=x_idx - 0.5, x1=x_idx + 0.5,
                    y0=y0, y1=y1,
                    line=dict(color=color, width=width), fillcolor=fill, layer='above')

    @staticmethod
    def _row_rect(y0, y1, x_min, x_max, color, width, fill=None):
        return dict(type='rect', xref='x', yref='y',
                    x0=x_min, x1=x_max,
                    y0=y0, y1=y1,
                    line=dict(color=color, width=width), fillcolor=fill, layer='above')

    @staticmethod
    def _voronoi_y_bounds(y_positions: list) -> dict:
        """Map each display y position to (lo, hi) matching Plotly's Voronoi cell rule."""
        s = sorted(y_positions)
        n = len(s)
        bounds = {}
        for i, y in enumerate(s):
            if n == 1:
                lo, hi = y - 0.5, y + 0.5
            else:
                step_before = (y - s[i - 1]) if i > 0     else (s[1] - s[0])
                step_after  = (s[i + 1] - y) if i < n - 1 else (s[-1] - s[-2])
                lo = y - step_before / 2
                hi = y + step_after  / 2
            bounds[y] = (lo, hi)
        return bounds

    # ── Figure reconstruction (PNG export) ───────────────────────────────────

    def _reconstruct_figure(self, view_data: dict, width: int = 1200,
                             height: int = 800, theme_override=None):
        run_path         = view_data.get('run')
        config_path      = view_data.get('config_path')
        node_filter      = view_data.get('node_filter', '')
        active_ph        = view_data.get('active_phases') or []
        active_node_types = view_data.get('active_node_types') or None
        ui_toggles       = list(view_data.get('ui_toggles') or [])
        if theme_override is not None:
            # ?theme= wins over whatever the saved view stored. resolve()'s only
            # contract is membership of 'light', so trace keeps its theme flag
            # inside ui_toggles and no saved-view JSON changes shape.
            ui_toggles = [t for t in ui_toggles if t != 'light']
            if _resolve_theme(theme_override).is_light:
                ui_toggles.append('light')

        if run_path and run_path not in self._cache:
            try:
                self._load_to_cache(run_path, config_path)
            except Exception as e:
                import sys, traceback
                print(f'[export] failed to load run "{run_path}": {e}\n'
                      + traceback.format_exc(), file=sys.stderr, flush=True)

        data, _ = self._get_visible_data(run_path, node_filter, active_ph,
                                          active_node_types, ui_toggles)
        if not data:
            fig = go.Figure()
            fig.update_layout(width=width, height=height)
            return fig

        tv          = view_data.get('trace_visibility') or None
        click       = view_data.get('click_data')
        select_expr = view_data.get('select_expr', '')
        selection   = self._get_selection_mask(data, select_expr) if select_expr else []
        fig, _ = self._build_figure(data, ui_toggles, click=click, selection=selection,
                                    trace_visibility=tv)
        fig.update_layout(width=width, height=height)
        return fig

    # ── Public export API ─────────────────────────────────────────────────────

    def export_image(self, run_path: str, width: int = 1200, height: int = 800,
                     fmt: str = 'png', active_phases: list = None) -> bytes:
        from .utils import render_figure_image
        run_path = str(run_path)
        if run_path not in self._cache:
            self._load_to_cache(run_path)
        cached = self._cache[run_path]
        view_data = {
            'run':               run_path,
            'config_path':       None,
            'node_filter':       '',
            'active_phases':     active_phases if active_phases is not None else cached['unique_phases'],
            'active_node_types': [],
            'ui_toggles':        _DEFAULT_PLOT_TOGGLES,
            'select_expr':       '',
            'select_active':     [],
            'click_data':        None,
        }
        fig = self._reconstruct_figure(view_data, width, height)
        return render_figure_image(fig, width, height, fmt=fmt)

    # ── Build ─────────────────────────────────────────────────────────────────
    def build(self) -> Dash:
        register_templates()
        app = Dash(__name__, suppress_callback_exceptions=True,
                   url_base_pathname=self._url_prefix)
        app.index_string = _theme_index_string(app.index_string)
        app.layout = self._build_layout()
        self._register_callbacks(app)
        register_theme_attribute(app, 'tp-plot-toggles')
        register_export_route(
            app,
            reconstruct_fn=self._reconstruct_figure,
            safe_roots=self._safe_roots,
            default_search_dirs=[Path(self._initial_vdir), Path('/tmp/dashboard')],
        )
        from wsgi.base import register_view_refresh, register_config_file_route
        register_view_refresh(
            app,
            open_trigger_id='tp-views-toggle',
            list_options_fn=lambda p: _view_dropdown_options(_views_dir(p)),
            body_state_id='tp-views-panel',
            view_picker_id='tp-view-select',
            views_dir_id='tp-views-dir',
        )
        # Only the url id is tp-prefixed; the token store deliberately shares
        # ADMIN_STORE_ID with every other dashboard (see utils.py for why).
        register_admin_token_capture(app, url_id='tp-url')
        url_prefix = os.environ.get('URL_PREFIX', '/')
        register_config_file_route(app, url_prefix)
        return app

    def _build_layout(self):
        return html.Div(style={'padding': '12px'}, children=[
            dcc.Location(id='tp-url'),
            theme_pref_store(),
            dcc.Store(id='tp-run-store'),
            dcc.Store(id='tp-selection-store', data=[]),
            dcc.Store(id='tp-active-view-name'),
            dcc.Store(id='tp-escape-store', data=0),
            dcc.Store(id='tp-live-trace-visibility', data={}),

            # ── Data selector ─────────────────────────────────────────────────
            html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '10px',
                            'flexWrap': 'wrap', 'alignItems': 'center'}, children=[
                html.Span('Config:', style=_S),
                dcc.Dropdown(id='tp-config-select', options=self._list_configs(),
                             placeholder='Select config…',
                             style={**_INP, 'width': '380px', 'padding': '0'}),
                html.A('yaml', id='tp-config-file-link', href='#', target='_blank',
                       style={'fontSize': '11px', 'color': _FAINT,
                              'textDecoration': 'none', 'marginLeft': '4px'}),
                html.Span('Run:', style=_S),
                dcc.Dropdown(id='tp-run-select', placeholder='Select run…',
                             style={**_INP, 'width': '260px', 'padding': '0'}),
                html.Span(id='tp-load-status', style={**_S, 'color': _DANGER}),
            ]),

            # ── Controls ──────────────────────────────────────────────────────
            html.Div(style={'display': 'flex', 'flexDirection': 'column',
                            'gap': '8px', 'marginBottom': '10px'}, children=[

                # Row 1: node filter + phase checklist + node type checklist
                html.Div(style={'display': 'flex', 'gap': '10px',
                                'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
                    html.Span('Filter:', style=_LBL),
                    dcc.Input(id='tp-filter-expr',
                              placeholder='e.g. degree_in > 0 | time > 3',
                              debounce=False,
                              style={**_INP, 'width': '240px'}),
                    html.Button('Apply', id='tp-filter-btn', n_clicks=0, style=_BTN),
                    html.Span('Phases:', style=_LBL),
                    dcc.Checklist(id='tp-phase-select',
                                  options=[], value=[],
                                  inline=True, labelStyle=_LBL),
                    html.Span('Types:', style=_LBL),
                    dcc.Checklist(id='tp-node-type-select',
                                  options=[], value=[],
                                  inline=True, labelStyle=_LBL),
                ]),

                # Row 2: cell selection
                html.Div(style={'display': 'flex', 'gap': '10px', 'alignItems': 'center'}, children=[
                    html.Span('Select Cells:', style=_LBL),
                    dcc.Input(id='tp-select-expr',
                              placeholder='e.g. time == 5 | node_id < 10',
                              style={**_INP, 'flex': '1'}),
                    dcc.Checklist(id='tp-select-active',
                                  options=[{'label': '  Show', 'value': 'show'}],
                                  value=[], inline=True, labelStyle=_LBL),
                ]),

                # Row 3: FILTER toggles
                html.Div(style={'display': 'flex', 'gap': '6px', 'alignItems': 'center',
                                'flexWrap': 'wrap'}, children=[
                    html.Span('FILTER', style=_SECTION),
                    dcc.Checklist(id='tp-filter-toggles', options=[
                        {'label': '  Hide isolated', 'value': 'no_isolated'},
                        {'label': '  Hide static',   'value': 'no_static'},
                    ], value=list(_DEFAULT_FILTER_TOGGLES),
                    inline=True, labelStyle=_LBL),
                ]),

                # Row 4: PLOT toggles
                html.Div(style={'display': 'flex', 'gap': '6px', 'alignItems': 'center',
                                'flexWrap': 'wrap'}, children=[
                    html.Span('PLOT', style=_SECTION),
                    dcc.Checklist(id='tp-plot-toggles', options=[
                        {'label': '  Light',          'value': 'light'},
                        {'label': '  State legend',   'value': 'state_legend'},
                        {'label': '  Phase legend',   'value': 'phase_legend'},
                        {'label': '  Concat cols',    'value': 'concat'},
                        {'label': '  Row',            'value': 'highlight_row'},
                        {'label': '  Col',            'value': 'highlight_col'},
                        {'label': '  Parents',        'value': 'highlight_parents'},
                        {'label': '  Children',       'value': 'highlight_children'},
                        {'label': '  Renum X',        'value': 'renumber_x'},
                        {'label': '  Renum Y',        'value': 'renumber_y'},
                    ], value=list(_DEFAULT_PLOT_TOGGLES),
                    inline=True, labelStyle=_LBL),
                ]),
            ]),

            # ── Heatmap ───────────────────────────────────────────────────────
            dcc.Graph(id='tp-heatmap', style={'height': '65vh'},
                      config={'scrollZoom': True}),

            # ── Info panel ────────────────────────────────────────────────────
            html.Div(id='tp-info-panel', style={
                'marginTop': '8px', 'padding': '10px',
                'background': _PANEL, 'border': f'1px solid {_BORDER}',
                'borderRadius': '4px', 'minHeight': '48px', **_S,
            }),

            # ── Views panel (collapsible) ─────────────────────────────────────
            html.Div(style={'marginTop': '10px', 'border': f'1px solid {_BORDER}',
                            'borderRadius': '4px'}, children=[
                html.Button('▶ Views', id='tp-views-toggle', n_clicks=0,
                            style={**_BTN, 'width': '100%', 'textAlign': 'left',
                                   'borderRadius': '4px'}),
                html.Div(id='tp-views-panel',
                         style={'display': 'none', 'padding': '10px'}, children=[
                    html.Div(style={'display': 'flex', 'gap': '8px',
                                    'flexWrap': 'wrap', 'alignItems': 'center'}, children=[
                        dcc.Input(id='tp-view-name', type='text',
                                  placeholder='View name…',
                                  style={**_INP, 'width': '150px'}),
                        html.Button('Save',   id='tp-save-btn',         n_clicks=0, style=_BTN),
                        dcc.Dropdown(id='tp-view-select',
                                     options=_view_dropdown_options(Path(self._initial_vdir)),
                                     placeholder='Load view…',
                                     style={**_INP, 'width': '190px', 'padding': '0'},
                                     clearable=True),
                        html.Button('Delete', id='tp-delete-view-btn', n_clicks=0,
                                    style={**_BTN, 'color': _DANGER}),
                        dcc.Input(id='tp-views-dir', type='text',
                                  value=self._initial_vdir,
                                  style={**_INP, 'width': '270px'}),
                        admin_token_store(),
                        html.Span(id='tp-view-status', style={**_S, 'color': _TEXT_DIM}),
                    ]),
                ]),
            ]),
        ])

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _register_callbacks(self, app):

        # ── Views panel toggle ────────────────────────────────────────────────
        @app.callback(
            Output('tp-views-panel',  'style'),
            Output('tp-views-toggle', 'children'),
            Input('tp-views-toggle',  'n_clicks'),
        )
        def toggle_views(n):
            open_ = bool(n and n % 2 == 1)
            return ({'display': 'block', 'padding': '10px'} if open_
                    else {'display': 'none',  'padding': '10px'},
                    '▼ Views' if open_ else '▶ Views')

        # ── Config yaml link ──────────────────────────────────────────────────
        @app.callback(
            Output('tp-config-file-link', 'href'),
            Input('tp-config-select', 'value'),
        )
        def _update_config_link(config_rel):
            if not config_rel:
                return '#'
            try:
                rel = (self._config_base / config_rel).relative_to('/code')
                return f'config-file?path={rel}'
            except ValueError:
                return '#'

        # ── Config → run dropdown ─────────────────────────────────────────────
        @app.callback(
            Output('tp-run-select', 'options'),
            Output('tp-run-select', 'value'),
            Input('tp-config-select', 'value'),
        )
        def on_config_change(config_rel):
            if not config_rel:
                return [], None
            try:
                from project.boolean_reservoir.code.parameter import load_yaml_config
                params   = load_yaml_config(str(self._config_base / config_rel))
                run_dirs = self._find_run_dirs(
                    self._resolve_run_base(str(params.logging.out_path)))
                opts = [{'label': f'{p.name} ({_fmt_mtime(p)})', 'value': str(p)}
                        for p in run_dirs]
                return opts, (opts[0]['value'] if opts else None)
            except Exception:
                return [], None

        # ── Config/run change → cache + phase + node-type checklists ───────────
        @app.callback(
            Output('tp-run-store',          'data'),
            Output('tp-load-status',        'children'),
            Output('tp-phase-select',       'options'),
            Output('tp-phase-select',       'value'),
            Output('tp-node-type-select',   'options'),
            Output('tp-node-type-select',   'value'),
            Input('tp-run-select',    'value'),
            Input('tp-config-select', 'value'),
        )
        def load_data(run_path, config_rel):
            if not run_path:
                raise PreventUpdate
            config_path = (str(self._config_base / config_rel)
                           if config_rel else None)
            msg    = self._load_to_cache(run_path, config_path=config_path)
            cached = self._cache[run_path]
            phases = cached['unique_phases']
            p_opts = [{'label': ph, 'value': ph} for ph in phases]
            # Node types in fixed order; output off by default
            all_types  = ['input', 'reservoir', 'output']
            have_types = sorted(cached['node_meta']['node_type'].unique(),
                                key=lambda t: all_types.index(t) if t in all_types else 99)
            t_opts     = [{'label': t.capitalize(), 'value': t} for t in have_types]
            t_val      = [t for t in have_types if t != 'output']
            return ({'path': run_path, 'config': config_rel},
                    msg, p_opts, phases, t_opts, t_val)

        # ── Selection store ───────────────────────────────────────────────────
        @app.callback(
            Output('tp-selection-store', 'data', allow_duplicate=True),
            Input('tp-select-active',    'value'),
            Input('tp-select-expr',      'n_submit'),
            State('tp-select-expr',       'value'),
            State('tp-run-store',         'data'),
            State('tp-filter-expr',       'value'),
            State('tp-phase-select',      'value'),
            State('tp-node-type-select',  'value'),
            State('tp-filter-toggles',    'value'),
            State('tp-plot-toggles',      'value'),
            prevent_initial_call=True,
        )
        def handle_selection(active, _submit, expr, store, node_filter,
                             active_phases, active_node_types, filter_toggles, plot_toggles):
            if not active or 'show' not in active or not store:
                return []
            ui_toggles = _merge_toggles(filter_toggles, plot_toggles)
            data, _ = self._get_visible_data(store['path'], node_filter,
                                             active_phases or [], active_node_types,
                                             ui_toggles)
            if not data:
                return []
            return self._get_selection_mask(data, expr)

        # ── Escape key → clear click (one-time listener via set_props) ──────────
        app.clientside_callback(
            """
            function(pathname) {
                if (window._tp_esc_init) return window.dash_clientside.no_update;
                window._tp_esc_init = true;
                document.addEventListener('keyup', function(e) {
                    if (e.key === 'Escape')
                        window.dash_clientside.set_props('tp-heatmap', {clickData: null});
                });
                return window.dash_clientside.no_update;
            }
            """,
            Output('tp-escape-store', 'data'),
            Input('tp-url', 'pathname'),
            prevent_initial_call=False,
        )

        # ── Track legend-click visibility (with merge to preserve legendonly) ───
        app.clientside_callback(
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
                setTimeout(function() {
                    var gd = document.querySelector('#tp-heatmap .js-plotly-plot');
                    if (!gd) return;
                    gd.removeAllListeners('plotly_restyle');
                    gd.on('plotly_restyle', function() {
                        dash_clientside.set_props('tp-live-trace-visibility',
                            {data: captureVis(gd.data)});
                    });
                }, 150);
                var newVis = captureVis((figure || {}).data);
                var prev = prevVis || {};
                var merged = {};
                for (var k in newVis) {
                    merged[k] = (newVis[k] === true && prev[k] === 'legendonly')
                        ? 'legendonly' : newVis[k];
                }
                return merged;
            }
            """,
            Output('tp-live-trace-visibility', 'data'),
            Input('tp-heatmap', 'figure'),
            State('tp-live-trace-visibility', 'data'),
            prevent_initial_call=True,
        )

        # ── Main heatmap ──────────────────────────────────────────────────────
        @app.callback(
            Output('tp-heatmap',    'figure'),
            Output('tp-info-panel', 'children'),
            Input('tp-run-store',        'data'),
            Input('tp-filter-btn',       'n_clicks'),
            Input('tp-filter-expr',      'n_submit'),
            Input('tp-phase-select',     'value'),
            Input('tp-node-type-select', 'value'),
            Input('tp-filter-toggles',   'value'),
            Input('tp-plot-toggles',     'value'),
            Input('tp-heatmap',          'clickData'),
            Input('tp-selection-store',  'data'),
            State('tp-filter-expr',      'value'),
            State('tp-live-trace-visibility', 'data'),
        )
        def update_heatmap(store, _filter_btn, _filter_submit,
                           active_phases, active_node_types,
                           filter_toggles, plot_toggles, click, selection,
                           node_filter, trace_vis):
            if not store or not active_phases:
                return go.Figure(), 'Load a run to begin.'
            ui_toggles = _merge_toggles(filter_toggles, plot_toggles)
            data, ferr = self._get_visible_data(
                store['path'], node_filter, active_phases or [],
                active_node_types, ui_toggles)
            if not data:
                cached = self._cache.get(store['path'], {})
                cols   = list(cached.get('node_meta', pd.DataFrame()).columns)
                hint   = f'Filter cols: {cols}' if cols else ''
                return go.Figure(), f'{ferr or "No data matches filter."}  {hint}'
            fig, info = self._build_figure(data, ui_toggles, click, selection or [],
                                           trace_visibility=trace_vis)
            if ferr:
                info = html.Div([
                    html.Span(f'Filter error: {ferr}',
                              style={'color': _DANGER, 'fontFamily': 'monospace',
                                     'fontSize': '12px'}),
                    html.Br(), info,
                ])
            return fig, info

        # ── Save view ─────────────────────────────────────────────────────────
        @app.callback(
            Output('tp-view-status',      'children'),
            Output('tp-view-select',      'options',  allow_duplicate=True),
            Output('tp-active-view-name', 'data',     allow_duplicate=True),
            Input('tp-save-btn', 'n_clicks'),
            State('tp-view-name',             'value'),
            State('tp-views-dir',             'value'),
            State('tp-config-select',         'value'),
            State('tp-run-store',             'data'),
            State('tp-filter-expr',           'value'),
            State('tp-phase-select',          'value'),
            State('tp-node-type-select',      'value'),
            State('tp-filter-toggles',        'value'),
            State('tp-plot-toggles',          'value'),
            State('tp-select-expr',           'value'),
            State('tp-select-active',         'value'),
            State('tp-heatmap',               'clickData'),
            State('tp-live-trace-visibility', 'data'),
            admin_token_state(),
            prevent_initial_call=True,
        )
        def save_view(_, name, vdir_str, config, store, node_filter,
                      active_phases, active_node_types, filter_toggles, plot_toggles,
                      select_expr, select_active, click_data, live_tv, admin_token):
            if not name or not name.strip():
                return 'Enter a view name.', no_update, no_update
            name = name.strip()
            vdir = _views_dir(vdir_str)
            blocked = guard_view_write(vdir, self._initial_vdir, self._safe_roots,
                                       admin_token, 3)
            if blocked is not None:
                return blocked
            vdir.mkdir(parents=True, exist_ok=True)
            view = {
                'config':             config,
                'run':                store['path'] if store else None,
                'config_path':        str(self._config_base / config) if config else None,
                'node_filter':        node_filter or '',
                'active_phases':      active_phases or [],
                'active_node_types':  active_node_types or [],
                'ui_toggles':         _merge_toggles(filter_toggles, plot_toggles),
                'select_expr':        select_expr or '',
                'select_active':      select_active or [],
                'click_data':         click_data,
                'trace_visibility':   live_tv or {},
            }
            (vdir / f'{name}.json').write_bytes(orjson.dumps(view))
            return f'✓ Saved "{name}"', _view_dropdown_options(vdir), name

        # ── Load view (button) ────────────────────────────────────────────────
        @app.callback(
            Output('tp-config-select',        'value',    allow_duplicate=True),
            Output('tp-run-store',            'data',     allow_duplicate=True),
            Output('tp-filter-expr',          'value',    allow_duplicate=True),
            Output('tp-phase-select',         'options',  allow_duplicate=True),
            Output('tp-phase-select',         'value',    allow_duplicate=True),
            Output('tp-node-type-select',     'options',  allow_duplicate=True),
            Output('tp-node-type-select',     'value',    allow_duplicate=True),
            Output('tp-filter-toggles',       'value',    allow_duplicate=True),
            Output('tp-plot-toggles',         'value',    allow_duplicate=True),
            Output('tp-select-expr',          'value',      allow_duplicate=True),
            Output('tp-select-active',        'value',      allow_duplicate=True),
            Output('tp-selection-store',      'data',       allow_duplicate=True),
            Output('tp-heatmap',              'clickData',  allow_duplicate=True),
            Output('tp-view-status',          'children', allow_duplicate=True),
            Output('tp-active-view-name',     'data',     allow_duplicate=True),
            Output('tp-live-trace-visibility','data',     allow_duplicate=True),
            Input('tp-view-select', 'value'),
            State('tp-views-dir',   'value'),
            prevent_initial_call=True,
        )
        def load_view(view_name, vdir_str):
            if not view_name:
                raise PreventUpdate
            return self._apply_view(view_name, vdir_str)

        # ── Load view from URL (?view_name) ───────────────────────────────────
        @app.callback(
            Output('tp-config-select',        'value',    allow_duplicate=True),
            Output('tp-run-store',            'data',     allow_duplicate=True),
            Output('tp-filter-expr',          'value',    allow_duplicate=True),
            Output('tp-phase-select',         'options',  allow_duplicate=True),
            Output('tp-phase-select',         'value',    allow_duplicate=True),
            Output('tp-node-type-select',     'options',  allow_duplicate=True),
            Output('tp-node-type-select',     'value',    allow_duplicate=True),
            Output('tp-filter-toggles',       'value',    allow_duplicate=True),
            Output('tp-plot-toggles',         'value',    allow_duplicate=True),
            Output('tp-select-expr',          'value',      allow_duplicate=True),
            Output('tp-select-active',        'value',      allow_duplicate=True),
            Output('tp-selection-store',      'data',       allow_duplicate=True),
            Output('tp-heatmap',              'clickData',  allow_duplicate=True),
            Output('tp-view-status',          'children', allow_duplicate=True),
            Output('tp-active-view-name',     'data',     allow_duplicate=True),
            Output('tp-live-trace-visibility','data',     allow_duplicate=True),
            Input('tp-url', 'search'),
            State('tp-views-dir', 'value'),
            prevent_initial_call='initial_duplicate',
        )
        def load_view_from_url(search, vdir_str):
            name = view_name_from_query(search)
            if not name:
                raise PreventUpdate
            return self._apply_view(name, vdir_str)

        # ── Delete view ───────────────────────────────────────────────────────
        @app.callback(
            Output('tp-view-status', 'children', allow_duplicate=True),
            Output('tp-view-select', 'options',  allow_duplicate=True),
            Input('tp-delete-view-btn', 'n_clicks'),
            State('tp-view-select', 'value'),
            State('tp-views-dir',   'value'),
            admin_token_state(),
            prevent_initial_call=True,
        )
        def delete_view(_, view_name, vdir_str, admin_token):
            if not view_name:
                raise PreventUpdate
            vdir = _views_dir(vdir_str)
            blocked = guard_view_write(vdir, self._initial_vdir, self._safe_roots,
                                       admin_token, 2)
            if blocked is not None:
                return blocked
            delete_view_json(view_name, vdir)
            return f'✓ Deleted "{view_name}"', _view_dropdown_options(vdir)

        # ── Refresh view dropdown — handled by register_view_refresh below ───

        # ── Update URL when view activates (clientside) ───────────────────────
        app.clientside_callback(
            """
            function(name) {
                if (!name) return window.dash_clientside.no_update;
                window.history.replaceState(
                    null, '', window.location.pathname + '?' + name);
                return window.dash_clientside.no_update;
            }
            """,
            Output('tp-active-view-name', 'data', allow_duplicate=True),
            Input('tp-active-view-name', 'data'),
            prevent_initial_call=True,
        )

        @app.callback(
            Output('tp-view-name', 'value'),
            Input('tp-active-view-name', 'data'),
        )
        def sync_view_name(name):
            return name or ''

    # ── View apply helper ─────────────────────────────────────────────────────

    def _apply_view(self, view_name: str, vdir_str: str):
        path = _views_dir(vdir_str) / f'{view_name}.json'
        if not path.exists():
            raise PreventUpdate
        v        = orjson.loads(path.read_bytes())
        run_path = v.get('run')
        store    = ({'path': run_path, 'config': v.get('config')}
                    if run_path else None)
        if run_path and run_path not in self._cache:
            try:
                self._load_to_cache(run_path, config_path=v.get('config_path'))
            except Exception:
                pass
        cached = self._cache.get(run_path, {}) if run_path else {}

        phases     = cached.get('unique_phases', [])
        phase_opts = [{'label': ph, 'value': ph} for ph in phases]

        all_types  = ['input', 'reservoir', 'output']
        have_types = sorted(
            cached.get('node_meta', pd.DataFrame(columns=['node_type']))['node_type'].unique(),
            key=lambda t: all_types.index(t) if t in all_types else 99)
        t_opts = [{'label': t.capitalize(), 'value': t} for t in have_types]
        # Saved value → fallback to input+reservoir default
        t_val  = v.get('active_node_types') or [t for t in have_types if t != 'output']

        ft, pt   = _split_toggles(v.get('ui_toggles'))
        ui_toggles    = v.get('ui_toggles', [])
        active_phases = v.get('active_phases') or phases
        node_filter   = v.get('node_filter', '')
        select_expr   = v.get('select_expr', '')

        selection = []
        if select_expr and run_path in self._cache:
            data, _ = self._get_visible_data(run_path, node_filter, active_phases,
                                             t_val, ui_toggles)
            if data:
                selection = self._get_selection_mask(data, select_expr)

        return (
            v.get('config'),
            store,
            node_filter,
            phase_opts,
            active_phases,
            t_opts,
            t_val,
            ft,
            pt,
            select_expr,
            v.get('select_active', ['show'] if selection else []),
            selection,
            v.get('click_data'),
            f'✓ Loaded "{view_name}"',
            view_name,
            v.get('trace_visibility') or {},
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_mtime(p: Path) -> str:
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))


# ── Public factory ────────────────────────────────────────────────────────────

def create_trace_plot_dashboard(
    config_base_path: str = '/code/config',
    out_base: str = '/out',
    safe_roots: list = None,
    initial_views_dir: str = str(VIEWS_ROOT / 'trace'),
    url_prefix: str = '/',
) -> Dash:
    return TracePlotDashboard(
        config_base_path=config_base_path,
        out_base=out_base,
        safe_roots=safe_roots,
        initial_views_dir=initial_views_dir,
        url_prefix=url_prefix,
    ).build()
