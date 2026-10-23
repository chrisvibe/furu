import pytest
import yaml
import numpy as np
import networkx as nx
from pathlib import Path
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.parameter import Params

CONFIG = Path(__file__).parent / 'config' / 'networkx_wiring.yaml'


@pytest.fixture(scope='module')
def model():
    return BooleanReservoir(load_path=CONFIG)


def test_builds(model):
    """NetworkXWiring with barabasi_albert_graph constructs without error."""
    assert model is not None


def test_adj_list_shape(model):
    """adj_list rows == N_total; columns <= k_max + I->R identity contribution."""
    P = model.P
    N_total = P.M.I.n_nodes + P.M.R.n_nodes + P.M.O.n_nodes
    assert model.adj_list.shape[0] == N_total
    assert model.adj_list.shape[1] <= P.M.wiring[2].k_max + 1


def test_unknown_fn_raises():
    """Unknown same-layer function name raises ValueError with nx guidance."""
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg['model']['wiring'][2]['graph']['name'] = 'no_such_graph_fn'
    params = Params(**cfg)
    with pytest.raises(ValueError, match="nx has no function"):
        BooleanReservoir(params=params)


def test_unknown_bipartite_fn_raises():
    """Unknown cross-layer function name raises ValueError with nx.bipartite guidance."""
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    # Replace the R->R networkx entry with I->R and drop the existing I->R pattern
    # entry so there's no duplicate wiring for (I, R).
    cfg['model']['wiring'] = [
        cfg['model']['wiring'][0],  # B -> I pattern
        {**cfg['model']['wiring'][2], 'source': 'I', 'graph': {'name': 'no_such_bipartite_fn', 'params': {}}},
    ]
    params = Params(**cfg)
    with pytest.raises(ValueError, match="nx.bipartite has no function"):
        BooleanReservoir(params=params)


def test_explicit_wiring_builds(tmp_path):
    """ExplicitWiring loads a graph from an edge list file."""
    n_r = 32
    g = nx.barabasi_albert_graph(n_r, 3)
    edge_file = tmp_path / 'graph.edgelist'
    nx.write_edgelist(g, edge_file, data=False)

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg['model']['wiring'][2] = {
        'type': 'explicit', 'source': 'R', 'target': 'R',
        'graph': {'name': 'read_edgelist', 'params': {'path': str(edge_file)}},
        'directed': True,
    }
    model = BooleanReservoir(params=Params(**cfg))
    N_total = model.P.M.I.n_nodes + n_r + model.P.M.O.n_nodes
    assert model.adj_list.shape[0] == N_total


def test_explicit_wiring_node_count_mismatch_raises(tmp_path):
    """Wrong node count raises ValueError with a clear message."""
    edge_file = tmp_path / 'small.edgelist'
    nx.write_edgelist(nx.path_graph(5), edge_file, data=False)  # 5 nodes ≠ 32

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg['model']['wiring'][2] = {
        'type': 'explicit', 'source': 'R', 'target': 'R',
        'graph': {'name': 'read_edgelist', 'params': {'path': str(edge_file)}},
    }
    with pytest.raises(ValueError, match="expected"):
        BooleanReservoir(params=Params(**cfg))
