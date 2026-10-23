"""State-viz front-end for the modernized time/space reducer.

Sibling of `readout_analysis.py` / `weights_dash.py`: the pure
reduction lives in `utils/state_analysis.py` (no Julia, no server); this module turns
its `RecordedStates` / `reduce_inline` output into concrete figures for ONE
recorded (i, j) model.

Three surfaces, from the same building blocks:

  * pure figure-data builders (`build_trajectory`, `build_space_spread`,
    `build_node_space`) -- numpy only, unit-testable without booting anything;
  * `export` -- write the time-and-space figures to `--out` (headless, e.g. for
    a paper panel or a quick look). This is the mode that runs in CI / a sandbox;
  * `serve` -- an interactive Dash dashboard over the SAME points-frame
    machinery as `weights_dash` (shared chrome from
    `tools/utils/dash_components.py`: pandas-query filter, computed fields, colour
    attribute + colorscale dropdowns). It exposes the two deliberately
    separated axes of `state_analysis`:

      * TIME -- one sample's state trajectory over sequence positions
        (points = (s, f, t) steps, reduced over nodes): does the dynamics sweep
        an interpretable manifold?
      * SPACE (the TRANSPOSE) -- the (positions x nodes) matrix flipped to
        (nodes x positions) and reduced over TIME (points = nodes): how do
        nodes arrange by how their state evolves? Colourable by the graph
        tool's node metadata (readout weights w0/w1, cascade depth, part,
        n_in/n_out) when the checkpoint is loaded, plus per-node state stats
        (mean activity, flip rate) always.

    The reducer is swappable at the dropdown (pca|tsne|umap via
    `state_analysis.REDUCERS`, one `_make_reducer` behind them) and embeddings
    are cached per (view, reducer, phase, sample) so tSNE/UMAP don't refit on
    every Apply click.

Typical flow (after the states probe has recorded the row):

    python -m tools.weight_explorer explore --config <cfg> --i X --j Y --states

then

    python -m tools.states_dash --config <cfg> --i X --j Y            # serve
    python -m tools.states_dash --config <cfg> --i X --j Y --out /out/states

Phase note: real path-integration recordings emit NO
`reservoir_layer` rows, so the default phase is always picked from the phases
actually present (`state_analysis.pick_default_phase`), and the dashboard's
phase dropdown lists `available_phases` -- never the hardcoded PHASES tuple.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ── Pure figure-data builders (no plotly/dash -- testable) ───────────────────

@dataclass
class TrajectoryData:
    """Reduced-time-axis points for one sample: `x`, `y` coordinates + a colour
    variable (sequence position index) and a label per point."""

    x: list[float]
    y: list[float]
    colour: list[float]
    labels: list[str]
    title: str


@dataclass
class ScatterData:
    """Reduced-space-axis points: coordinates + per-node colour + labels."""

    x: list[float]
    y: list[float]
    colour: list[float]
    labels: list[str]
    title: str


@dataclass
class NodeSpaceData:
    """Reduced TRANSPOSED points ('time over space'): one point per node."""

    x: list[float]
    y: list[float]
    colour: list[float]
    labels: list[str]
    title: str


def _int_or_dash(v):
    """Provenance-safe int formatting: NaN provenance renders '-', never raises.

    A partially-NaN `t`/`s`/`f` column must degrade the hover text, not kill
    the whole dashboard (the i327601 recording ships an all-NaN `t` header)."""
    return '-' if pd.isna(v) else str(int(v))


def build_trajectory(rs, reducer='pca', sample_id=0,
                     n_components=2, **kwargs) -> TrajectoryData:
    """Reduce one sample's state trajectory (TIME axis) and label its points.

    `rs` is a `state_analysis.RecordedStates`. Uses a `reduce_inline`-style fit
    on the single sample's trajectory sub-matrix so the reduced manifold is about
    THAT sample's evolution, not the whole phase.
    """
    from .utils.state_analysis import reduce_inline

    traj, traj_meta = rs.time_trajectory(sample_id)
    emb = reduce_inline(traj, reducer, n_components, **kwargs)
    x = emb[:, 0].tolist()
    y = emb[:, 1].tolist()
    # colour by sequence position (index), so you can read the direction of travel
    colour = list(range(len(traj)))
    if 't' in traj_meta.columns:
        labels = [f's={_int_or_dash(r.s)} t={_int_or_dash(r.t)}'
                  for _, r in traj_meta.iterrows()]
    else:
        labels = [f's={_int_or_dash(r.s)}' for _, r in traj_meta.iterrows()]
    return TrajectoryData(x=x, y=y, colour=colour, labels=labels,
                          title=f'Trajectory (TIME) sample {sample_id}, {reducer}')


def build_space_spread(rs, reducer='pca', position=None,
                       n_components=2, **kwargs) -> ScatterData:
    """Reduce the SPACE axis at one fixed (s, f, t): all samples, that step.

    `position` defaults to the first recorded reservoir position. Answers "at a
    fixed moment, do the reservoir states across samples form a low-rank cluster
    or a space-filling spread?" (the projection-trick candidacy).
    """
    from .utils.state_analysis import reduce_inline

    if position is None:
        pos = dict(s=int(rs.meta['s'].iloc[0]), f=int(rs.meta['f'].iloc[0]),
                   t=int(rs.meta['t'].iloc[0]))
    else:
        pos = position
    spread, _meta = rs.space_spread(pos)
    emb = reduce_inline(spread, reducer, n_components, **kwargs)
    colour = emb[:, 2].tolist() if (n_components > 2 and emb.shape[1] > 2) \
        else list(range(len(emb)))
    return ScatterData(x=emb[:, 0].tolist(), y=emb[:, 1].tolist(),
                       colour=colour,
                       labels=[f'sample {s}' for s in _meta['sample_id']],
                       title=f'Space spread at s={pos["s"]} t={pos["t"]}, {reducer}')


def build_node_space(rs, reducer='pca', sample_id=0,
                     n_components=2, **kwargs) -> NodeSpaceData:
    """Reduce the TRANSPOSE of one sample's trajectory: nodes as points.

    The TIME view's `(positions x nodes)` state matrix is flipped to
    `(nodes x positions)` and reduced over TIME, so each node lands at
    coordinates describing HOW ITS STATE EVOLVED -- nodes with similar temporal
    activity cluster. This is the 'time over space' axis: same reducer seam
    (`reduce_inline`), transposed input, points = nodes.
    """
    from .utils.state_analysis import reduce_inline

    traj, _meta = rs.time_trajectory(sample_id)
    emb = reduce_inline(traj.T, reducer, n_components, **kwargs)
    node_ids = list(range(rs.node_offset, rs.node_offset + traj.shape[1]))
    return NodeSpaceData(
        x=emb[:, 0].tolist(), y=emb[:, 1].tolist(),
        colour=list(range(len(node_ids))),
        labels=[f'node {n}' for n in node_ids],
        title=f'Node space over TIME (transposed) sample {sample_id}, {reducer}')


# ── Plotly figures (optional import kept local so the pure builders above stay
# ── importable without plotly, matching the readout_analysis separation seam) ─

def _two_panel_figure(trajectory: TrajectoryData, scatter: ScatterData):
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(trajectory.title, scatter.title))
    fig.add_trace(go.Scatter(
        x=trajectory.x, y=trajectory.y, mode='lines+markers',
        hovertext=trajectory.labels,
        marker=dict(color=trajectory.colour, colorscale='Viridis', size=6),
        name='trajectory'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=scatter.x, y=scatter.y, mode='markers',
        hovertext=scatter.labels,
        marker=dict(color=scatter.colour, colorscale='Viridis', size=6),
        name='space'), row=1, col=2)
    fig.update_layout(title='Reservoir state: TIME trajectory vs SPACE spread')
    return fig


def export_figures(trajectory: TrajectoryData, scatter: ScatterData,
                   out: Path, fmt='png'):
    """Write the two panels to `<out>/state_trajectory_space.<fmt>`. Returns the
    written paths. (plotly is imported lazily so the pure builders above work
    without it.)"""
    import plotly.io as pio

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig = _two_panel_figure(trajectory, scatter)
    p = out / f'state_trajectory_space.{fmt}'
    pio.write_image(fig, p)
    return [p]


# ── Points-frame builders for the interactive dashboard (pandas, testable) ───

def time_points_df(rs, sample_id, emb) -> pd.DataFrame:
    """One sample's TIME-axis points: provenance columns + embedding coords.

    Index = sequence position; columns `s, f, t, sample_id, position, x, y` --
    exactly the frame the dashboard's query/computed-field/colour chrome
    operates on.
    """
    traj, meta = rs.time_trajectory(sample_id)
    df = meta[['s', 'f', 't', 'sample_id']].reset_index(drop=True).copy()
    df['position'] = range(len(df))
    df['x'] = emb[:, 0]
    df['y'] = emb[:, 1]
    return df


def space_points_df(rs, sample_id, emb, node_meta) -> pd.DataFrame:
    """SPACE (transposed) points: one row per node + embedding + state stats.

    `sample_id=None` uses the WHOLE phase (all samples' rows) as the temporal
    signature; otherwise that single sample's trajectory. `node_meta` is the
    (optional) per-node attribute frame from `node_metadata`, indexed by
    absolute node id and aligned to `rs`'s columns. Adds two state-only stats:
    `mean` (mean activity over the view's rows) and `flips` (fraction of
    adjacent rows where the node's state changed).
    """
    mat = rs.states if sample_id is None else rs.time_trajectory(sample_id)[0]
    pts = node_meta.copy()
    pts['mean'] = mat.mean(axis=0)
    pts['flips'] = (mat[:-1] != mat[1:]).mean(axis=0) if len(mat) >= 2 else 0.0
    pts['x'] = emb[:, 0]
    pts['y'] = emb[:, 1]
    return pts


def state_figure(points, norm, view, color_attr, colorscale,
                 uirevision=None, skip_hover=('x', 'y')):
    """Assemble the 2-D dashboard figure from a (filtered) points frame.

    `norm` is `readout_analysis.normalize(points)` computed against the FULL
    frame so the colourbar stays stable across filter changes (the graph tool's
    full-vs-filtered lesson). TIME view draws the trajectory as lines+markers;
    SPACE draws markers. `uirevision` pins zoom/pan across colour/query-only
    changes and resets it when the embedding itself changes.
    """
    import plotly.graph_objs as go

    from .utils.dash_components import colorbar_ticks, fmt_val

    if color_attr not in points.columns:
        raise KeyError(f'color attribute {color_attr!r} not among {list(points.columns)}')
    colour = norm.loc[points.index, color_attr + '_normalized_']
    tickvals, ticktext = colorbar_ticks(points, color_attr)

    if view == 'time':
        hover = [
            f"pos {_int_or_dash(r.position)}: s={_int_or_dash(r.s)} "
            f"f={_int_or_dash(r.f)} t={_int_or_dash(r.t)}"
            for r in points.itertuples()
        ]
        trace = go.Scatter(
            x=points['x'], y=points['y'], mode='lines+markers',
            marker=dict(color=colour, colorscale=colorscale, size=8,
                        colorbar=dict(title=color_attr, tickvals=tickvals,
                                      ticktext=ticktext),
                        line=dict(width=1, color='rgba(0,0,0,0.4)')),
            hovertext=hover, hoverinfo='text', name='trajectory')
    else:
        hover = [
            f'node {int(r.node)} ({getattr(r, "part", "?")}): ' + ', '.join(
                f'{c}: {fmt_val(v)}' for c, v in zip(points.columns, row)
                if c not in skip_hover)
            for r, row in zip(points.itertuples(), points.itertuples(index=False))
        ]
        trace = go.Scatter(
            x=points['x'], y=points['y'], mode='markers',
            marker=dict(color=colour, colorscale=colorscale, size=7,
                        colorbar=dict(title=color_attr, tickvals=tickvals,
                                      ticktext=ticktext),
                        line=dict(width=1, color='rgba(0,0,0,0.4)')),
            hovertext=hover, hoverinfo='text', name='nodes')

    fig = go.Figure(trace)
    fig.update_layout(
        template='plotly_white',
        uirevision=uirevision,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"{'TIME trajectory' if view == 'time' else 'SPACE (nodes)'}"
              f" -- coloured by {color_attr}")
    return fig


# ── Load a recorded (i, j) row's states + per-node metadata ──────────────────

def load_history(config, i: int, j: int):
    """Ensure + load the raw `(history, expanded_meta)` pair for one (i, j) row.

    Auto-provisions via `ensure_states` (records the #71 probe if missing).
    Callers reshape per phase with `state_analysis.reshape_recorded_states`.
    """
    from .utils.ensure_data import ensure_states
    from project.boolean_reservoir.code.utils.reservoir_utils import (
        BatchedTensorHistoryWriter)

    sp = ensure_states(config, i, j)
    _ld, history, expanded_meta, _meta = BatchedTensorHistoryWriter(sp).reload_history()
    return history, expanded_meta


def load_states(base_out_path, i: int, j: int):
    """Load + reshape the recorded states for one row into a `RecordedStates`.

    The phase is picked from what the recording actually contains
    (`pick_default_phase`): real PI recordings have no `reservoir_layer` rows,
    so the historical hardcoded default selected zero rows.
    """
    from project.boolean_reservoir.code.utils.reservoir_utils import (
        BatchedTensorHistoryWriter)
    from project.boolean_reservoir.code.utils.single_run import states_path
    from .utils.state_analysis import pick_default_phase, reshape_recorded_states

    sp = states_path(base_out_path, i, j)
    if not (sp / 'tensor_0.npy').exists():
        raise FileNotFoundError(
            f"No recorded states for (i={i}, j={j}) at {sp} -- run the #71 states "
            "probe first: python -m tools.weight_explorer explore --i "
            f"{i} --j {j} --states")
    _ld, history, expanded_meta, _meta = BatchedTensorHistoryWriter(sp).reload_history()
    return reshape_recorded_states(history, expanded_meta,
                                   phase=pick_default_phase(expanded_meta))


def load_model(config, i: int, j: int):
    """Checkpoint -> BooleanReservoir, auto-provisioned (mirrors the graph tool)."""
    from .utils.ensure_data import ensure_weights
    from project.boolean_reservoir.code.reservoir import BooleanReservoir

    return BooleanReservoir(load_path=ensure_weights(config, i, j))


def node_metadata(model, rs) -> pd.DataFrame:
    """Per-node attributes for the SPACE view's points, indexed by absolute id.

    Reuses the graph tool's structural seam (`readout_analysis`): readout
    weights `w0..w{k-1}` (NaN outside the reservoir slice), cascade `depth`,
    `part` (I/O/R via `dash_components.part_labels`), `n_in`/`n_out`. Joined on
    absolute node id = `rs.node_offset` + column, so it aligns with the
    reducer's column space exactly.
    """
    from .utils.dash_components import part_labels
    from .utils.readout_analysis import cascade_depth_from_inputs

    n_cols = rs.states.shape[1]
    abs_ids = list(range(rs.node_offset, rs.node_offset + n_cols))
    df = pd.DataFrame({'node': abs_ids,
                       'node_rel': [a - rs.node_offset for a in abs_ids]},
                      index=abs_ids)
    df['part'] = pd.Series(part_labels(model), index=np.arange(model.N_total)) \
        .loc[abs_ids].to_numpy()
    depth = cascade_depth_from_inputs(model.graph, model.input_nodes_mask)
    df['depth'] = [depth.get(a) for a in abs_ids]
    res_start = model.res_slice.start
    for k in range(model.readout_W.shape[0]):
        w = pd.Series(model.readout_W[k],
                      index=range(res_start, model.res_slice.stop))
        df[f'w{k}'] = [w.get(a, np.nan) for a in abs_ids]
    deg_in = dict(model.graph.in_degree())
    deg_out = dict(model.graph.out_degree())
    df['n_in'] = [deg_in.get(a, 0) for a in abs_ids]
    df['n_out'] = [deg_out.get(a, 0) for a in abs_ids]
    return df


# ── Interactive dashboard (the revived review tool) ──────────────────────────

STATE_QUERY_HELP = """Query (filters points):
  TIME view:   s, f, t, position, sample_id, x, y
    t == 1            (one sub-step)
    position > 3      (late trajectory only)
  SPACE view:  node, node_rel, part, depth, w0.., n_in, n_out,
               mean, flips, x, y
    part == 'R'            (reservoir nodes only)
    depth > 2 & abs(w0) > 0.05
    flips > 0.4            (restless nodes)

Computed Field (new column to colour by):
  hw = abs(w0) * depth        (SPACE view)
  active = mean > 0.5

TIME view ignores Top-N / Node types (they filter nodes, not steps).
SPACE 'ALL probe inputs' reduces over every recorded row of the phase."""


def build_state_app(history, expanded_meta, config, i, j, model=None,
                    node_slice=None, default_reducer='pca', default_sample=0):
    """Construct (not run) the interactive review dashboard for one (i, j).

    Split from `serve_dashboard` so tests / probes can build the app and assert
    on layout + callback wiring without booting a server.
    """
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State

    from .utils.dash_components import (COLORSCALE_OPTIONS, DEFAULT_COLORSCALE,
                                  ComputedFields, clamp_top_n,
                                  query_help_components)
    from .utils.readout_analysis import apply_pandas_query, normalize
    from .utils.state_analysis import (REDUCERS, available_phases, pick_default_phase,
                                  reduce_inline, reshape_recorded_states)

    phases = available_phases(expanded_meta)
    phase0 = pick_default_phase(expanded_meta)
    sample_ids = sorted(int(s) for s in expanded_meta['sample_id'].unique())
    default_sample = default_sample if default_sample in sample_ids else sample_ids[0]

    _computed = ComputedFields()
    _emb_cache: dict = {}
    _node_meta: list = [None]

    def _rs(phase):
        return reshape_recorded_states(history, expanded_meta, phase=phase,
                                       node_slice=node_slice)

    def _node_meta_for(rs):
        if _node_meta[0] is None:
            n_cols = rs.states.shape[1]
            base = pd.DataFrame({'node': range(rs.node_offset, rs.node_offset + n_cols),
                                 'node_rel': list(range(n_cols))},
                                index=list(range(rs.node_offset, rs.node_offset + n_cols)))
            _node_meta[0] = node_metadata(model, rs) if model is not None else base
        return _node_meta[0]

    def _fit(view, rs, reducer, sample):
        key = (view, reducer, rs.phase, sample)
        if key not in _emb_cache:
            if view == 'time':
                mat = rs.time_trajectory(sample)[0]
            else:
                mat = rs.states if sample is None else rs.time_trajectory(sample)[0]
            _emb_cache[key] = reduce_inline(mat if view == 'time' else mat.T,
                                            reducer, 2)
        return _emb_cache[key]

    def _default_color(view, points):
        if view == 'time':
            return 'position'
        return 'w0' if 'w0' in points.columns else 'mean'

    def _render(view, reducer, phase, sample, color_attr, colorscale,
                top_n, parts, query):
        """Full render: reshape -> (cached) fit -> points frame -> chrome. Returns
        `(fig, query_status, count_text, color_attr, points_full)`; raises on a
        bad fit."""
        rs = _rs(phase)
        sample_key = None if (view == 'space' and sample == 'all') else int(sample)
        emb = _fit(view, rs, reducer, sample_key)
        if view == 'time':
            points = time_points_df(rs, sample_key, emb)
        else:
            points = space_points_df(rs, sample_key, emb, _node_meta_for(rs))

        _computed.apply(points)
        # Normalise against the FULL points frame (before filtering) so the
        # colourbar keeps its meaning across query/part changes, as in the
        # graph tool.
        norm = normalize(points)

        if view == 'space':
            mask = pd.Series(True, index=points.index)
            if 'part' in points.columns:
                mask &= points['part'].isin(tuple(parts or ()))
            top_n = clamp_top_n(top_n, len(points))
            if top_n:
                visible = points.index[mask][:top_n]
                mask &= points.index.isin(visible)
            points = points[mask]

        filtered, status = apply_pandas_query(points, query)
        if not color_attr or color_attr not in points.columns:
            color_attr = _default_color(view, points)
        norm = norm.loc[points.index]
        fig = state_figure(filtered, norm, view, color_attr, colorscale,
                           uirevision=str((view, reducer, phase, sample_key)))
        count = f'points: {len(filtered)}/{len(points)}'
        return fig, status, count, color_attr, points

    def _sample_options(view):
        opts = [{'label': f'probe input {s}', 'value': s} for s in sample_ids]
        if view == 'space':
            opts.append({'label': 'ALL probe inputs (whole phase)', 'value': 'all'})
        return opts

    def _color_options(points):
        return [{'label': c, 'value': c} for c in sorted(points.columns)]

    # ── Initial render (TIME view, default sample); seeds the colour options ─
    init_fig, _, init_count, init_color, init_points = _render(
        'time', default_reducer, phase0, default_sample,
        '', DEFAULT_COLORSCALE, 0, ('I', 'R', 'O'), '')

    app = dash.Dash(__name__)
    app.title = 'State over time / space'

    sidebar = html.Div([
        html.Pre(f'{Path(config).name}  i={i} j={j}\n'
                 f'phases: {", ".join(phases)}\n'
                 f'{"node metadata: ON (checkpoint loaded)" if model else "node metadata: OFF (--no-model)"}',
                 style={'margin': '4px 0', 'whiteSpace': 'pre-wrap',
                        'fontSize': 12}),
        html.Label('View:'),
        dcc.Dropdown(
            id='view_selector',
            options=[{'label': 'TIME (points = steps)',
                      'value': 'time'},
                     {'label': 'SPACE (points = nodes)',
                      'value': 'space'}],
            value='time', clearable=False),
        html.Label('Reducer:'),
        dcc.Dropdown(
            id='reducer_selector',
            options=[{'label': r, 'value': r} for r in REDUCERS],
            value=default_reducer, clearable=False),
        html.Label('Phase:'),
        dcc.Dropdown(
            id='phase_selector',
            options=[{'label': p, 'value': p} for p in phases],
            value=phase0, clearable=False),
        html.Label('Probe input (sample_id):'),
        dcc.Dropdown(
            id='sample_selector',
            options=_sample_options('time'),
            value=default_sample, clearable=False),
        html.Br(),
        html.Label('Colour attribute:'),
        dcc.Dropdown(id='color_selector',
                     options=_color_options(init_points),
                     value=init_color, clearable=False),
        html.Label('Colorscale:'),
        dcc.Dropdown(id='colorscale_selector',
                     options=[{'label': cs, 'value': cs} for cs in COLORSCALE_OPTIONS],
                     value=DEFAULT_COLORSCALE, clearable=False),
        html.Label('Top-N nodes (SPACE; 0 = all):'),
        dcc.Input(id='top_n_input', type='number', value=0, min=0, step=1),
        html.Label('Node types (SPACE):'),
        dcc.Checklist(
            id='parts_selector',
            options=[{'label': ' Input (I)', 'value': 'I'},
                     {'label': ' Reservoir (R)', 'value': 'R'},
                     {'label': ' Output (O)', 'value': 'O'}],
            value=['I', 'R', 'O'], labelStyle={'display': 'block'}),
        html.Br(),
                html.Label('Filter:'),
        *query_help_components(STATE_QUERY_HELP),
        dcc.Input(id='query_input', type='text', placeholder='e.g. depth > 2',
                  value=''),
        html.Div(id='query_status', style={'color': 'crimson', 'fontSize': 12,
                                           'marginTop': '4px',
                                           'whiteSpace': 'pre-wrap'}),
        html.Div(id='point_count', style={'color': '#555', 'fontSize': 12,
                                          'marginTop': '2px'}, children=init_count),
        html.Br(),
        html.Label('Computed Field (name = expression):'),
        dcc.Input(id='computed_field_input', type='text',
                  placeholder='e.g. hw = abs(w0)', value=''),
        html.Div(id='computed_field_status',
                 style={'color': 'green', 'fontSize': 12, 'marginTop': '4px',
                        'whiteSpace': 'pre-wrap'}),
        html.Button('Apply Changes', id='apply_filter_button', n_clicks=0),
    ], style={'position': 'fixed', 'top': '10px', 'left': '10px',
              'width': '320px', 'background-color': 'rgba(255,255,255,0.85)',
              'padding': '16px', 'border-radius': '10px',
              'box-shadow': '0px 0px 10px rgba(0,0,0,0.1)',
              'z-index': '1000', 'fontSize': 13,
              'maxHeight': '95vh', 'overflowY': 'auto'})

    app.layout = html.Div([
        sidebar,
        dcc.Graph(id='graph',
                  style={'height': '100vh', 'width': '100vw',
                         'position': 'absolute', 'top': '0', 'left': '0',
                         'z-index': '1'},
                  figure=init_fig),
    ], style={'position': 'relative', 'height': '100vh', 'width': '100vw',
              'overflow': 'hidden'})

    from .utils.dash_components import register_query_help_toggle
    register_query_help_toggle(app)

    @app.callback(
        Output('graph', 'figure'),
        Output('query_status', 'children'),
        Output('color_selector', 'options'),
        Output('computed_field_status', 'children'),
        Output('point_count', 'children'),
        Output('sample_selector', 'options'),
        Input('apply_filter_button', 'n_clicks'),
        State('view_selector', 'value'),
        State('reducer_selector', 'value'),
        State('phase_selector', 'value'),
        State('sample_selector', 'value'),
        State('color_selector', 'value'),
        State('colorscale_selector', 'value'),
        State('top_n_input', 'value'),
        State('parts_selector', 'value'),
        State('query_input', 'value'),
        State('computed_field_input', 'value'),
    )
    def update(n_clicks, view, reducer, phase, sample, color_attr, colorscale,
               top_n, parts, query, computed_expr):
        cf_status = _computed.parse(computed_expr)
        try:
            fig, status, count, color_attr, pts = _render(
                view, reducer, phase, sample, color_attr, colorscale,
                top_n, parts, query)
            # colour options follow the view's points frame (time: s/f/t/...,
            # space: node metadata + stats + computed fields)
            return (fig, status, _color_options(pts), cf_status, count,
                    _sample_options(view))
        except Exception as exc:  # noqa: BLE001 -- surfaced in the UI
            return (dash.no_update, f'render failed: {exc}', dash.no_update,
                    cf_status, dash.no_update, dash.no_update)

    # §4 fix 2: run_server was removed in Dash 3.x (env pins dash 4.2.0)
    return app


def serve_dashboard(history, expanded_meta, config, i, j, model=None,
                    node_slice=None, default_reducer='pca', default_sample=0):
    """Build the review dashboard and serve it (dev server)."""
    app = build_state_app(history, expanded_meta, config, i, j, model=model,
                          node_slice=node_slice, default_reducer=default_reducer,
                          default_sample=default_sample)
    app.run(debug=True, use_reloader=False)


def _cli():
    import argparse

    from .utils.state_analysis import REDUCERS, available_phases, pick_default_phase, \
        reshape_recorded_states

    parser = argparse.ArgumentParser(
        description='Review reservoir states over TIME and (transposed) over '
                    'SPACE for one recorded (i,j) row.')
    parser.add_argument('--config', required=True,
                        help='grid-search config yaml whose L.out_path feeds the '
                             'single_run replace convention')
    parser.add_argument('--i', type=int, required=True)
    parser.add_argument('--j', type=int, default=0)
    parser.add_argument('--reducer', default='pca', choices=list(REDUCERS))
    parser.add_argument('--phase', default=None,
                        help='phase to reduce (default: best available -- real PI '
                             'recordings have no reservoir_layer rows)')
    parser.add_argument('--sample', type=int, default=0,
                        help="probe input (sample_id) for the TIME trajectory -- "
                             "the recording's own input axis, NOT the grid-search "
                             "j seed copy (default 0)")
    parser.add_argument('--node-slice', type=str, default=None,
                        help="node slice e.g. '0:512' to restrict the reservoir nodes")
    parser.add_argument('--no-model', action='store_true',
                        help='skip loading the checkpoint (SPACE colours fall back '
                             'to state-only stats: mean/flips)')
    parser.add_argument('--out', type=Path, default=None,
                        help='write figures here (headless export); else serve the app')
    args = parser.parse_args()

    node_slice = None
    if args.node_slice:
        a, b = args.node_slice.split(':')
        node_slice = slice(int(a) if a else None, int(b) if b else None)

    history, expanded_meta = load_history(args.config, args.i, args.j)
    phase = args.phase or pick_default_phase(expanded_meta)
    print(f'phase {phase!r} (available: '
          f'{", ".join(available_phases(expanded_meta))})', flush=True)

    rs = reshape_recorded_states(history, expanded_meta, phase=phase,
                                 node_slice=node_slice)

    if args.out:
        trajectory = build_trajectory(rs, reducer=args.reducer,
                                      sample_id=args.sample)
        scatter = build_space_spread(rs, reducer=args.reducer)
        written = export_figures(trajectory, scatter, args.out)
        for p in written:
            print(f'wrote {p}')
        return 0

    model = None
    if not args.no_model:
        model = load_model(args.config, args.i, args.j)
    serve_dashboard(history, expanded_meta, args.config, args.i, args.j,
                    model=model, node_slice=node_slice,
                    default_reducer=args.reducer, default_sample=args.sample)
    return 0


if __name__ == '__main__':
    raise SystemExit(_cli())
