"""pytest for `tools/weights_dash.py` — the 3D weight-graph wrapper.

Construct the app WITHOUT serving (Dash.run
monkeypatched to a no-op — Flask debug=True cannot boot in this devcontainer's
/dev/shm), assert the layout + initial figure, then drive the multi-output
update callback through the flask test client (`app.server.test_client()`; dash
4.2 needs the modern `outputs` id/property list alongside the packed `output`
key).

A fake model + an explicit layout callable keep this offline: no checkpoint, no
out/ data, no Julia, no filesystem. The fake graph nodes carry `id` (what the
reconstructed checkpoint graph provides and `set_index('id')` requires).
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

_DASH_DIR = Path(__file__).resolve().parents[2]      # <repo>/code/dashboard
sys.path.insert(0, str(_DASH_DIR))

import dash  # noqa: E402
from tools.weights_dash import plot_graph_with_weight_coloring_3D  # noqa: E402


def _fake_model():
    from types import SimpleNamespace
    g = nx.DiGraph()
    g.add_nodes_from(range(10))
    for k in range(2, 8):                 # inputs feed reservoir
        g.add_edge(k - 2, k)
    for k in range(8):                    # reservoir -> output
        g.add_edge(k, 8 + (k % 2))
    for n in g.nodes():
        g.nodes[n]['id'] = n              # the attr `set_index('id')` needs
    rng = np.random.default_rng(3)
    return SimpleNamespace(
        N_total=10,
        input_nodes_mask=np.array([n < 2 for n in range(10)]),
        out_slice=slice(8, 10),
        res_slice=slice(0, 8),
        readout_W=rng.normal(size=(2, 8)) * 0.1,
        graph=g,
    )


def _linear_layout(graph):
    """Deterministic 3-D coords; skips the seeded-layout out_path seam."""
    return {n: np.array([float(n), 0.0, 0.0]) for n in graph.nodes()}


@pytest.fixture()
def graph_app(monkeypatch):
    """Build the app via plot_graph_with_weight_coloring_3D without serving."""
    captured = []
    orig_init = dash.Dash.__init__

    def _capture_init(self, *a, **k):
        orig_init(self, *a, **k)
        captured.append(self)

    monkeypatch.setattr(dash.Dash, '__init__', _capture_init)
    monkeypatch.setattr(dash.Dash, 'run', lambda self, *a, **k: None)
    plot_graph_with_weight_coloring_3D(_fake_model(), layout=_linear_layout,
                                       default_query='n_in != 0')
    return captured[-1]


def _drive(app, query='', computed='', color='w0'):
    """POST the packed multi-output callback; returns the response dict."""
    key = [k for k in app.callback_map if '..' in k and 'graph.figure' in k][0]
    outputs = [{'id': s.split('.')[0].strip('.'), 'property': s.split('.', 1)[1]}
               for s in key.strip('.').split('...')]
    body = {
        'output': key,
        'outputs': outputs,
        'inputs': [{'id': 'apply_filter_button', 'property': 'n_clicks',
                    'value': 1}],
        'state': [
            {'id': 'query_input', 'property': 'value', 'value': query},
            {'id': 'computed_field_input', 'property': 'value', 'value': computed},
            {'id': 'color_selector', 'property': 'value', 'value': color},
            {'id': 'colorscale_selector', 'property': 'value', 'value': 'Viridis'},
            {'id': 'top_n_input', 'property': 'value', 'value': 3000},
            {'id': 'parts_selector', 'property': 'value', 'value': ['I', 'R', 'O']},
            {'id': 'relayout_data_store', 'property': 'data', 'value': {}},
        ],
        'changedPropIds': ['apply_filter_button.n_clicks'],
    }
    r = app.server.test_client().post('/_dash-update-component', json=body)
    assert r.status_code == 200, f'{r.status_code}: {r.data[:300]}'
    return r.get_json()['response']


def _node_count(resp):
    return int(resp['node_count']['children']
               .split('nodes: ')[1].split('/')[0])


def test_app_builds_without_serving(graph_app):
    # layout: graph + full shared-chrome sidebar, seeded with the init figure
    fig = graph_app.layout.children[-1].figure
    assert len(fig.data) == 2               # edge trace + node trace
    assert fig.data[1].mode == 'markers'
    for cid in ('query_input', 'computed_field_input', 'color_selector',
                'colorscale_selector', 'top_n_input', 'parts_selector',
                'apply_filter_button'):
        assert cid in str(graph_app.layout), f'{cid} missing from layout'
    # the callback wiring exists and is the multi-output form
    assert any('..graph.figure' in k for k in graph_app.callback_map)


def test_callback_updates_figure_and_computed_field(graph_app):
    resp = _drive(graph_app, query='n_in != 0')
    # inputs 0/1 have in-degree 0; the other 8 nodes pass
    assert _node_count(resp) == 8
    assert len(resp['graph']['figure']['data']) == 2      # edges + nodes

    resp = _drive(graph_app, query='n_in != 0', computed='myw = abs(w0) * 2',
                  color='myw')
    opts = [o['value'] for o in resp['color_selector']['options']]
    assert 'myw' in opts


def test_callback_part_filter_narrows_to_reservoir(graph_app):
    resp = _drive(graph_app, query='part == "R"')
    assert _node_count(resp) == 6       # the 8 reservoir nodes minus inputs 0/1