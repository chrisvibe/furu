"""The wiring table (adj_list + mask + lut) is the model; the graph is drawn from it.

Covers wiring.py's conversions and the bug they fix: install_wiring used to leave
model.graph (what save() stored) describing the old wiring, so a saved CA checkpoint
reloaded as the random patchwork with the CA LUT.
"""
import networkx as nx
import numpy as np

from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.wiring import wiring_from_matrix, wiring_from_graph, graph_from_wiring


def _random_matrix(n=40, p=0.15, seed=0):
    return (np.random.default_rng(seed).random((n, n)) < p).astype(np.uint8)


def test_matrix_inputs_ascending_like_networkx():
    """Same table the old path built: nx.from_numpy_array, then predecessors."""
    W = _random_matrix()
    adj, mask = wiring_from_matrix(W)
    g = nx.from_numpy_array(W, create_using=nx.DiGraph)
    for v in range(len(W)):
        assert adj[v][mask[v]].tolist() == list(g.predecessors(v)) == sorted(np.nonzero(W[:, v])[0])
    assert adj.shape[1] == max(d for _, d in g.in_degree())


def test_graph_round_trip_keeps_input_order():
    # rows deliberately NOT ascending: the order is the LUT bit order and must survive
    adj = np.array([[2, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.int64)
    mask = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 0]], dtype=np.bool_)
    g = graph_from_wiring(adj, mask)
    assert list(g.predecessors(0)) == [2, 0, 1]
    adj2, mask2 = wiring_from_graph(g)
    assert np.array_equal(adj2[mask2], adj[mask]) and np.array_equal(mask2, mask)


def test_no_edges_gives_zero_width():
    adj, mask = wiring_from_matrix(np.zeros((5, 5), dtype=np.uint8))
    assert adj.shape == (5, 0) and mask.shape == (5, 0)


def _ca_model(tmp_path):
    from project.boolean_reservoir.code.ca import ca_model, install_rule150

    def into_tmp(P):
        P.L.out_path = str(tmp_path)
        P.L.history.record = False
    return ca_model('config/test/boolean_reservoir/ca/rule150_linear_transducer.yaml',
                    lambda m: install_rule150(m, 'input'), edit=into_tmp)


def test_graph_follows_install_wiring(tmp_path):
    model = _ca_model(tmp_path)
    adj, mask = wiring_from_graph(model.graph)
    assert np.array_equal(adj[mask], model.adj_list[model.adj_list_mask])
    cell0 = model.res_slice.start
    assert list(model.graph.predecessors(cell0))[0] == model.input_nodes[0]  # input is cell 0's MSB
    assert model.graph.edges[model.input_nodes[0], cell0]['quadrant'] == 'IR'


def test_ca_checkpoint_reloads_to_same_model(tmp_path):
    model = _ca_model(tmp_path)
    model.save()
    loaded = BooleanReservoir(load_path=model.L.last_checkpoint)

    assert np.array_equal(loaded.adj_list, model.adj_list)
    assert np.array_equal(loaded.adj_list_mask, model.adj_list_mask)
    assert loaded.lut == model.lut
    assert np.array_equal(loaded.initial_states, model.initial_states)

    x = np.random.default_rng(0).integers(0, 2, (1, 20, 1, 1), dtype=np.uint8)
    model(x); loaded(x)
    assert np.array_equal(loaded.states_parallel, model.states_parallel)


def test_checkpoint_without_wiring_file_rebuilds_from_graph(tmp_path):
    """Checkpoints saved before the table was stored: same model, from the graph."""
    P = load_yaml_config('config/test/boolean_reservoir/ca/rule90.yaml')
    P.L.out_path = str(tmp_path)
    P.L.history.record = False
    P.L.save_keys = ['parameters', 'graph', 'lut', 'init_state']
    model = BooleanReservoir(P)
    model.save()
    loaded = BooleanReservoir(load_path=model.L.last_checkpoint)
    assert np.array_equal(loaded.adj_list[loaded.adj_list_mask], model.adj_list[model.adj_list_mask])
    assert loaded.lut == model.lut
