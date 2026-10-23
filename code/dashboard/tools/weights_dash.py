"""Graph weight visualisation for one precomputed single run (localhost tool).

PLAN_graph_viz.md §4 + §5. Takes the WHOLE BooleanReservoir — the structural
filter, the weight mapping and the (i, j)-seeded layout all need `res_slice`,
the masks and the run's out_path, none of which are derivable from
`(graph, W, b)` alone.

Weights first: `python -m wsgi.precompute_weights --view <name> --limit N`
trains rows into (i, j)-addressable checkpoints, then:

    python -m tools.weights_dash --i 374120 --j 0

constructs the model from the checkpoint (graph/lut/initial states regenerate
from the checkpoint's own seeded parameters.yaml, so the graph IS the run's
graph) and serves the 3D weight view. NOTE: rows with `random-warmup` init spin
the Julia engine at construction (`reset_reservoir(hard_reset=True)` runs
`warmup` when 'warmup' is in R.init); `zeros`/`random` rows stay Julia-free.

The model carries `res_slice`/`input_nodes_mask`/`out_slice`; node part for the
structural filter follows the `_part()` semantics (I / O / else R), NOT the 'O'
attribute — labels are multi-label ('O' means "read by the output layer", which
currently coincides with 'R' — both True for the reservoir slice). The 'part'
column is the only way to distinguish reservoir (R) from output (O) nodes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

from tools.utils.dash_components import (
    COLORSCALE_OPTIONS,
    DEFAULT_COLORSCALE,
    ComputedFields,
    clamp_top_n,
    colorbar_ticks,
    fmt_val,
    part_labels,
    query_help_components,
    register_query_help_toggle,
)
from tools.utils.readout_analysis import (
    apply_pandas_query,
    cascade_depth_from_inputs,
    fuse_with_depth,
    global_concentration,
    normalize,
    per_output_row_diagnostics,
)

DEFAULT_TOP_N = 3000
DEFAULT_CONFIG = 'config/path_integration/2D/grid_search/design_choices/continuous.yaml'

QUERY_HELP_TEXT = """Query (filters rows):
  abs(w0) > 0.1
  I                       (inputs only)
  R & ~O                  (reservoir, non-output)
  O                       (output nodes)
  R | O                   (all non-input nodes)
  n_in == 0               (no incoming edges, stranded)
  n_out == 0              (no outgoing edges, dead end)
  depth > 5               (deep cascade)
  depth.isna()            (unreachable from any input)
  p_flip < 0.01           (stable nodes, needs --states)

Computed Field (new column to color by):
  my_name = abs(w0)
  wsum = w0 + w1
  stable = p_flip < 0.01   (boolean column, needs --states)"""


def _compute_flip_probability(base_out_path, i, j):
    """Load recorded reservoir states and compute per-node flip probability.

    Expects states recorded by `record_states_probe` at `states_path`:
      {base_out_path}/i{i:05d}_j{j}/runs/last_run/history

    Flip probability = fraction of adjacent-timestep transitions where the
    node's state differs (0→1 or 1→0), computed over all samples in the
    input_layer phase (where the reservoir is actively processing). Returns a
    dict {node_id: float} or None if the states path doesn't exist or can't
    be loaded.
    """
    from project.boolean_reservoir.code.utils.single_run import states_path
    from project.boolean_reservoir.code.utils.reservoir_utils import BatchedTensorHistoryWriter

    sp = states_path(base_out_path, i, j)
    if not sp.exists():
        return None

    try:
        writer = BatchedTensorHistoryWriter(save_path=str(sp), persist_to_disk=True)
        _load_dict, _tensor, _expanded, _meta = writer.reload_history()
    except Exception:
        return None

    from dashboard.tools.utils.state_analysis import reshape_recorded_states
    # The states probe records input_layer and output_layer (reservoir runs
    # during both), but the reservoir slice is all N_total nodes. Use
    # input_layer as the primary processing phase.
    for phase in ('input_layer', 'output_layer', 'init'):
        rs = reshape_recorded_states(_tensor, _expanded, phase=phase)
        if rs.states.shape[0] >= 2:
            break
    else:
        return None

    # Flip = mean of adjacent-row transitions where state changed, per node.
    flips = (rs.states[:-1] != rs.states[1:]).mean(axis=0)
    # rs.states columns are offset by rs.node_offset
    node_ids = range(rs.node_offset, rs.node_offset + rs.states.shape[1])
    return dict(zip(node_ids, flips.tolist()))


def plot_graph_with_weight_coloring_3D(model, layout=None, metadata=None,
                                       draw_edges=True, draw_arrow=False,
                                       run_info=None, default_query=''):
    """Serve the weight-coloured 3D graph of one BooleanReservoir.

    model:   a constructed BooleanReservoir (checkpoint or freshly trained).
    layout:  optional callable graph -> {node: coords}. Default None uses the
             (i, j)-seeded spring layout, preferring the CLI-written
             `layout.npy` cache — positions are then comparable across filter
             changes and identical on recompute.
    run_info: optional dict with test metrics from the grid search evaluation
              (e.g. {'acc': 0.997, 'loss': 0.0037, 'kq': 182, 'gr': 113}).
              Overrides the checkpoint's own L.train values.
    """
    graph = model.graph

    # ── Readout-strategy metadata ───────────────
    # cascade depth from inputs: structural surrogate for "how much recurrence
    # must the reservoir do before the readout sees this node" (pure graph +
    # input mask, no states / no Julia). depth 0 = input, 1 = directly
    # input-shadowing, deep = requires propagation.
    _depth = cascade_depth_from_inputs(graph, model.input_nodes_mask)
    # fused per-output geometry (readout row axis) × depth of the node it reads
    # hardest -> one hypothesis label per output dim.
    _geoms = per_output_row_diagnostics(model.readout_W)
    _fused = fuse_with_depth(_geoms, _depth, model.res_slice.start)
    _gini = global_concentration(model.readout_W)
    # run identity: config i / sample j / metrics, read from the checkpoint's
    # logging params.  The grid search test accuracy is loaded separately in
    # __main__ and passed via run_info.
    _L = getattr(model, 'L', None) or getattr(getattr(model, 'P', None), 'L', None)
    _i = _L.train.config if _L and _L.train else None
    _j = _L.train.sample if _L and _L.train else None
    _acc = _L.train.accuracy_dev if _L and _L.train else None
    _loss = None
    _kqgr = _L.kqgr if _L and hasattr(_L, 'kqgr') else None
    # Override with grid search test metrics if provided
    if run_info:
        _acc = run_info.get('acc', _acc)
        _loss = run_info.get('loss', _loss)
        if 'kq' in run_info or 'gr' in run_info or 'delta' in run_info:
            if _kqgr is None:
                from types import SimpleNamespace
                _kqgr = SimpleNamespace(
                    kq=run_info.get('kq'),
                    gr=run_info.get('gr'),
                    delta=run_info.get('delta'),
                )
            else:
                if run_info.get('kq') is not None:
                    _kqgr.kq = run_info['kq']
                if run_info.get('gr') is not None:
                    _kqgr.gr = run_info['gr']
                if run_info.get('delta') is not None:
                    _kqgr.delta = run_info['delta']

    def _part_series():
        return part_labels(model)

    def add_readout_metadata(model):
        # §4 fix 5: readout_W columns index the RESERVOIR SLICE, not node ids.
        # (rows = output dims, columns = model.res_slice). The old version zipped
        # range(R_n_nodes) against weight rows — painting the INPUT nodes
        # 0..N_I-1 and leaving the real reservoir uncoloured.
        for k in range(model.readout_W.shape[0]):
            nx.set_node_attributes(graph, dict(zip(
                range(model.res_slice.start, model.res_slice.stop),
                model.readout_W[k])), f'w{k}')

    def split_coordinates(d):
        attrs = {}
        for key, arr in d.items():
            shp = arr.shape
            arr = np.resize(arr, 3)
            arr[shp[0]:] = 0
            attrs[key] = {"x": arr[0], "y": arr[1], "z": arr[2]}
        return attrs

    # ── Node metadata ────────────────────────────────────────────────────────
    add_readout_metadata(model)
    if metadata:
        for label, dict_data in metadata:
            nx.set_node_attributes(graph, dict_data, label)

    if layout is not None:
        pos_d = layout(graph)
    else:
        from project.boolean_reservoir.code.utils.single_run import (
            load_layout_for_run, seeded_layout_3d,
        )
        arr = load_layout_for_run(model.L.out_path, graph.number_of_nodes())
        if arr is not None:
            pos_d = {node: arr[k] for k, node in enumerate(sorted(graph.nodes()))}
        else:
            pos_d = seeded_layout_3d(graph, model.L.out_path)
    nx.set_node_attributes(graph, split_coordinates(pos_d))

    # collect all data in df. 'id' is the node id set by add_graph_labels — the
    # old add_readout_metadata overwrote it here; it must not.
    df = pd.DataFrame.from_dict(dict(graph.nodes(data=True))).T.set_index('id', drop=False)
    df = df.infer_objects()
    # part = one label per node from the model's own masks (I / O / else R) —
    # NOT the 'O' attribute, which is multi-label and coincides with 'R'.
    df['part'] = pd.Series(_part_series(), index=df.index)
    # cascade depth from inputs (NaN = unreachable from any input; normally only
    # ever applies to I/O nodes, and they're excluded by the part filter anyway).
    df['depth'] = pd.Series([_depth.get(n) for n in df.index],
                            index=df.index, dtype='float64')
    # incoming and outgoing edge counts
    in_deg = dict(graph.in_degree())
    out_deg = dict(graph.out_degree())
    df['n_in'] = pd.Series(in_deg, index=df.index, dtype='int64')
    df['n_out'] = pd.Series(out_deg, index=df.index, dtype='int64')
    df_normalized = normalize(df)
    available_attributes = sorted(df.columns)

    # Pick a sensible default for the color dropdown: prefer w0
    # (the first readout weight set), falling back to the first column.
    _default_color_attr = 'w0' if 'w0' in available_attributes else available_attributes[0]
    print(f'Available attributes: {available_attributes}',
          f'  default color: {_default_color_attr}', file=sys.stderr, flush=True)

    n_res = model.res_slice.stop - model.res_slice.start
    N_total = model.N_total

    def _default_mask(show_parts, top_n):
        mask = df['part'].isin(show_parts)
        top_n = clamp_top_n(top_n, N_total)
        if top_n and 0 < top_n < N_total:
            # Performance cap: keep at most top_n nodes (no ordering).
            visible = df.index[mask][:top_n]
            mask = pd.Series(df.index.isin(visible), index=df.index)
        return mask

    def _colors_for(attr, index):
        series = df_normalized.loc[index][attr + '_normalized_']
        if 'color' in df.columns:
            return series.where(df.loc[index]['color'].isna(), df.loc[index]['color'])
        return series

    # ── Computed field support (shared chrome, tools/utils/dash_components.py) ─
    # Persisted across callbacks so derived columns survive re-renders.
    _computed = ComputedFields()

    def _all_attr_options():
        return _computed.options(available_attributes)

    def create_figure(query='', color_attr='', colorscale=DEFAULT_COLORSCALE,
                      top_n=DEFAULT_TOP_N,
                      show_parts=('I', 'R', 'O'), relayout_data=None):
        # Apply computed fields on the full df, then normalise only the new
        # columns so they're available for coloring.
        _computed.apply(df)
        _computed.normalize_into(df, df_normalized)

        filtered_df = df[_default_mask(show_parts, int(top_n or 0))]
        filtered_df, status = apply_pandas_query(filtered_df, query)
        if status:
            print(status, file=sys.stderr, flush=True)
            print(status, file=sys.stderr, flush=True)

        n_shown = len(filtered_df)
        n_total = len(df)

        # Count edges that connect two visible nodes
        index = filtered_df.index
        n_edges_total = graph.number_of_edges()
        n_edges_shown = sum(1 for e in graph.edges() if e[0] in index and e[1] in index)

        if not color_attr:
            color_attr = available_attributes[0]

        edge_x, edge_y, edge_z = [], [], []
        index = filtered_df.index
        for edge in graph.edges():
            if edge[0] in index and edge[1] in index:
                x0, y0, z0 = df.loc[edge[0]][['x', 'y', 'z']]
                x1, y1, z1 = df.loc[edge[1]][['x', 'y', 'z']]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                edge_z += [z0, z1, None]

        arrow_trace = None
        if draw_arrow:  # TODO fix looks terrible. Need data for edges not just nodes
            edge_start_x = np.array(edge_x[::2])
            edge_start_y = np.array(edge_y[::2])
            edge_start_z = np.array(edge_z[::2])
            arrow_u = np.array(edge_x[1::2])
            arrow_v = np.array(edge_y[1::2])
            arrow_w = np.array(edge_z[1::2])
            arrow_trace = go.Cone(
                x=edge_start_x, y=edge_start_y, z=edge_start_z,
                u=arrow_u, v=arrow_v, w=arrow_w,
                sizemode='absolute', sizeref=0.25, colorscale='Blues', showscale=False,
            )

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=0.5, color='black') if draw_edges else dict(width=0, color='rgba(0,0,0,0)'),
            hoverinfo='none', mode='lines')

        tickvals, ticktext = colorbar_ticks(df, color_attr)
        node_trace = go.Scatter3d(
            x=filtered_df['x'], y=filtered_df['y'], z=filtered_df['z'],
            mode='markers',
            marker=dict(
                size=5,
                colorscale=colorscale,
                colorbar=dict(title=color_attr,
                              tickvals=tickvals,
                              ticktext=ticktext),
                color=_colors_for(color_attr, filtered_df.index),
                line=dict(width=2),
            ),
            hoverinfo='text',
        )
        node_trace.text = [
            f"Node {i} ({row['part']}): " + ', '.join(
                f'{col}: {fmt_val(val)}' for col, val in row.items()
                if col not in ('x', 'y', 'z', 'part'))
            for i, row in filtered_df.iterrows()
        ]
        fig = go.Figure(data=[edge_trace, node_trace, arrow_trace] if arrow_trace else [edge_trace, node_trace])
        fig.update_layout(showlegend=False)
        fig.update_layout(scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ))
        if relayout_data:
            fig.update_layout(scene_camera=relayout_data.get('scene.camera', {}))
        return fig, status, n_shown, n_total, n_edges_shown, n_edges_total

    app = dash.Dash(__name__)

    # Pre-compute initial figure and node count for the layout
    init_fig, _, init_n_shown, init_n_total, init_e_shown, init_e_total = create_figure(color_attr=_default_color_attr)
    init_count = f'nodes: {init_n_shown}/{init_n_total}  edges: {init_e_shown}/{init_e_total}'

    def _run_identity():
        parts = []
        if _i is not None:
            parts.append(f"i={_i}")
        if _j is not None:
            parts.append(f"j={_j}")
        # Test metrics from grid search evaluation
        if _acc is not None:
            parts.append(f"acc={_acc:.4g}")
        if _loss is not None:
            parts.append(f"loss={_loss:.4g}")
        # KQGR generalization metrics
        if _kqgr is not None:
            kq = getattr(_kqgr, 'kq', None)
            gr = getattr(_kqgr, 'gr', None)
            delta = getattr(_kqgr, 'delta', None)
            if kq is not None:
                parts.append(f"kq={kq}")
            if gr is not None:
                parts.append(f"gr={gr}")
            if delta is not None:
                parts.append(f"delta={delta}")
        return '  '.join(parts) if parts else '(run id not in params)'

    def _strategy_panel():
        """Placeholder for readout/strategy analysis.

        Kept as an empty-but-reserved box so the UI slot is stable. The real
        readout diagnostics (PR, support, depth-fused verdicts, global Gini)
        live in tools/utils/readout_analysis.py and were deliberately pulled OUT of the
        figure: they are necessary-not-sufficient and we are not ready to trust
        them in the view yet. Fill this in once we know what to display.
        """
        return html.Div(
            [html.Pre(f"Run  [{_run_identity()}]",
                      style={'margin': '4px 0', 'whiteSpace': 'pre-wrap'}),
             html.Small("(analysis pending - see tools/utils/readout_analysis.py)",
                        style={'color': '#666'})],
            style={'fontSize': 12})

    app.layout = html.Div([
        dcc.Store(id='relayout_data_store'),
        html.Div([
            html.Label('Node types (part):'),
            dcc.Checklist(
                id='parts_selector',
                options=[{'label': ' Input (I)', 'value': 'I'},
                         {'label': ' Reservoir (R)', 'value': 'R'},
                         {'label': ' Output (O)', 'value': 'O'}],
                value=['I', 'R', 'O'],
                labelStyle={'display': 'block'},
            ),
            html.Label('Top-N nodes (0 = all):'),
            dcc.Input(id='top_n_input', type='number',
                      value=clamp_top_n(DEFAULT_TOP_N, N_total),
                      min=0, max=N_total, step=1),
            html.Br(),
            html.Label('Filter:'),
            *query_help_components(QUERY_HELP_TEXT),
            dcc.Input(
                id='query_input',
                type='text',
                placeholder='e.g. abs(w0) > 0.1',
                value=default_query,
            ),
            html.Div(id='query_status', style={'color': 'crimson', 'fontSize': 12,
                                               'marginTop': '4px', 'whiteSpace': 'pre-wrap'}),
            html.Div(id='node_count', style={'color': '#555', 'fontSize': 12,
                                              'marginTop': '2px'}, children=init_count),
            html.Br(),
            html.Label('Computed Field (name = expression):'),
            dcc.Input(
                id='computed_field_input',
                type='text',
                placeholder='e.g. my_field = abs(w0)',
                value='',
            ),
            html.Div(id='computed_field_status', style={'color': 'green', 'fontSize': 12,
                                                         'marginTop': '4px', 'whiteSpace': 'pre-wrap'}),
            html.Br(),
            html.Label('Colour attribute:'),
            dcc.Dropdown(
                id='color_selector',
                options=[{'label': attr, 'value': attr} for attr in _all_attr_options()],
                value=_default_color_attr,
                clearable=False,
            ),
            html.Label('Colorscale:'),
            dcc.Dropdown(
                id='colorscale_selector',
                options=[{'label': cs, 'value': cs} for cs in COLORSCALE_OPTIONS],
                value=DEFAULT_COLORSCALE,
                clearable=False,
            ),
            html.Button('Apply Changes', id='apply_filter_button', n_clicks=0),
            html.Hr(),
            html.Div(id='strategy_panel', children=_strategy_panel(), style={'fontSize': 12}),
        ], style={'position': 'fixed', 'top': '10px', 'left': '10px', 'width': '300px', 'background-color': 'rgba(255,255,255,0.8)', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0,0,0,0.1)', 'z-index': '1000'}),
        dcc.Graph(id='graph', style={'height': '100vh', 'width': '100vw', 'position': 'absolute', 'top': '0', 'left': '0', 'z-index': '1'}, figure=init_fig),
    ], style={'position': 'relative', 'height': '100vh', 'width': '100vw', 'overflow': 'hidden'})

    @app.callback(
        Output('relayout_data_store', 'data'),
        Input('graph', 'relayoutData'),
    )
    def store_relayout_data(relayoutData):
        return relayoutData or {}

    register_query_help_toggle(app)

    @app.callback(
        Output('graph', 'figure'),
        Output('query_status', 'children'),
        Output('color_selector', 'options'),
        Output('computed_field_status', 'children'),
        Output('node_count', 'children'),
        Input('apply_filter_button', 'n_clicks'),
        State('query_input', 'value'),
        State('computed_field_input', 'value'),
        State('color_selector', 'value'),
        State('colorscale_selector', 'value'),
        State('top_n_input', 'value'),
        State('parts_selector', 'value'),
        State('relayout_data_store', 'data'),
    )
    def update_graph(n_clicks, query, computed_expr, color_attr, colorscale,
                     top_n, show_parts, relayout_data):
        # ── Parse and register computed field (shared chrome) ───────────────
        cf_status = _computed.parse(computed_expr)

        # ── Rebuild color dropdown with computed fields ─────────────────────
        all_opts = [{'label': attr, 'value': attr} for attr in _all_attr_options()]

        # ── Render figure ───────────────────────────────────────────────────
        # colour is always normalised against the FULL graph so the colorbar is
        # stable and interpretable across filter changes (the toggleable
        # full-vs-filtered scale was misleading: the same node changed meaning).
        fig, status, n_shown, n_total, e_shown, e_total = create_figure(query=query, color_attr=color_attr,
                                    colorscale=colorscale,
                                    top_n=int(top_n) if top_n is not None else 0,
                                    show_parts=tuple(show_parts or ()),
                                    relayout_data=relayout_data)
        node_count = f'nodes: {n_shown}/{n_total}  edges: {e_shown}/{e_total}'
        return fig, status, all_opts, cf_status, node_count

    # §4 fix 2: run_server was removed in Dash 3.x (env pins dash 4.2.0)
    app.run(debug=True, use_reloader=False)


if __name__ == '__main__':
    # §5: construct the model from the row's (i, j)-addressable checkpoint.
    # Data provisioning (weights, states) happens automatically via ensure_data.
    #
    # NOTE: ensure_weights retrains the model to get the graph/weights.  The
    # training is seeded deterministically, so the model (graph, LUT, readout
    # weights) is a bit-exact replica of the grid search original.  The benefit
    # is that we never store full checkpoints in the single_run tree — only the
    # lightweight weights.npz + parameters.yaml.
    #
    # HOWEVER, the checkpoint's own L.train.accuracy is the ridge training loss
    # (near 0 for MSE), NOT the test accuracy shown in the scatter view.  Test
    # metrics live only in the grid search data/ parquet, written once during
    # the original grid search evaluation.  We load them separately from
    # fetch_params_json to populate the run identity panel — so the numbers
    # (acc, loss, kq, gr) match the scatter view exactly, even though the
    # checkpoint was retrained.
    import argparse
    from project.boolean_reservoir.code.parameter import load_yaml_config
    from project.boolean_reservoir.code.reservoir import BooleanReservoir
    from project.boolean_reservoir.code.utils.single_run import single_run_base
    from dashboard.tools.utils.ensure_data import ensure_states, ensure_weights

    parser = argparse.ArgumentParser(description='Graph weight visualisation for one precomputed single run')
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                        help='grid-search config; its L.out_path feeds the single_run replace convention')
    parser.add_argument('--i', type=int, required=True, help='grid-search combination index')
    parser.add_argument('--j', type=int, default=0, help='sample index (default 0)')
    parser.add_argument('--states', action='store_true',
                        help='load recorded states probe and add p_flip attribute')
    args = parser.parse_args()

    base = single_run_base(load_yaml_config(args.config).L.out_path)

    # Auto-provision: ensures weights exist (trains if missing), then model loads.
    cp = ensure_weights(args.config, args.i, args.j)
    model = BooleanReservoir(load_path=cp)

    # Load grid search test metrics for the run identity panel.
    # The checkpoint's own L.train.accuracy is the training fit, not the
    # test accuracy shown in the scatter view.  We read the grid search
    # params_json to get the real test metrics.
    run_info = {}
    try:
        from project.boolean_reservoir.code.utils.single_run import (
            fetch_params_json, rehydrate_params,
        )
        from wsgi import precompute_weights
        _app_name, config_paths = precompute_weights.resolve_config_app(args.config)
        pjson = fetch_params_json(config_paths, [(args.i, args.j)])
        if not pjson.empty:
            params = rehydrate_params(pjson.iloc[0]['params_json'])
            if params and hasattr(params, 'L') and params.L:
                if hasattr(params.L, 'train') and params.L.train:
                    _a = getattr(params.L.train, 'accuracy_dev', None)
                    _l = getattr(params.L.train, 'loss_dev', None)
                    if _a is not None:
                        run_info['acc'] = _a
                    if _l is not None:
                        run_info['loss'] = _l
                if hasattr(params.L, 'kqgr') and params.L.kqgr:
                    for k in ('kq', 'gr', 'delta'):
                        v = getattr(params.L.kqgr, k, None)
                        if v is not None:
                            run_info[k] = v
    except Exception:
        pass

    metadata = None
    default_query = 'n_in != 0'
    if args.states:
        # Auto-provision states too (records probe if missing).
        ensure_states(args.config, args.i, args.j)
        fp = _compute_flip_probability(base, args.i, args.j)
        if fp is not None:
            metadata = [('p_flip', fp)]
            default_query = '(p_flip != 0) & (n_in != 0)'
            print(f'Loaded p_flip for {len(fp)} nodes', flush=True)
        else:
            print('Warning: --states requested but p_flip could not be computed.',
                  file=sys.stderr, flush=True)

    plot_graph_with_weight_coloring_3D(model, metadata=metadata, run_info=run_info,
                                       default_query=default_query)
