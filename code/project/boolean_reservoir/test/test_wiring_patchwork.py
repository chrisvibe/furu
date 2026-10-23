"""The config patchwork (wiring.build_patchwork): validation, selectors, joint layers."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from project.boolean_reservoir.code.parameter import Params
from project.boolean_reservoir.code.wiring import build_patchwork

CONFIG = Path(__file__).parent / 'config' / 'networkx_wiring.yaml'  # I=2 nodes, R=32 nodes


def _params(*blocks, n_r=32):
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg['model']['reservoir_layer']['n_nodes'] = n_r
    cfg['model']['wiring'] = [cfg['model']['wiring'][0], *blocks]  # keep B → I
    return Params(**cfg)


def _gnm(source, target, k_avg=2, **kw):
    return {'type': 'gnm', 'source': source, 'target': target, 'k_avg': k_avg, 'k_min': 0, **kw}


def _rr(P):
    """The R → R quadrant of W (rows: senders, columns: receivers)."""
    _, W = build_patchwork(P)
    r = slice(P.M.I.n_nodes, P.M.I.n_nodes + P.M.R.n_nodes)
    return W[r, r]


# ── validation ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('source,target', [('B', 'R'), ('R', 'B'), ('B+I', 'I'), ('X', 'R'), ('R+R', 'R')])
def test_bad_layers_rejected(source, target):
    with pytest.raises(ValueError):
        _params(_gnm(source, target))


def test_duplicate_pair_rejected():
    with pytest.raises(ValueError, match='Duplicate wiring'):
        build_patchwork(_params(_gnm('R', 'R'), _gnm('I+R', 'R')))


# ── selectors: the block is drawn on the selected nodes only ───────────────────────────────

def test_no_selector_gives_k_avg():
    assert _rr(_params(_gnm('R', 'R'))).sum(axis=0).mean() == pytest.approx(2)


def test_source_selector_keeps_k_avg_from_selected_senders():
    """A worked example: the old draw-then-erase gave mean in-degree ~1 here."""
    rr = _rr(_params(_gnm('R', 'R', source_selector='S :16')))
    assert rr[16:].sum() == 0                                  # only R0–R15 send
    assert rr.sum(axis=0).mean() == pytest.approx(2)           # every receiver still gets 2 on average


def test_target_selector_wires_only_selected_receivers():
    rr = _rr(_params(_gnm('R', 'R', target_selector='S :16')))
    deg = rr.sum(axis=0)
    assert deg[16:].sum() == 0 and deg[:16].mean() == pytest.approx(2)


def test_source_selector_indexes_layers_as_written():
    """source I+R: the chain counts I's nodes first, although the block lays R out first."""
    P = _params(_gnm('I+R', 'R', k_avg=1, source_selector='S :2'))  # = the 2 I nodes
    _, W = build_patchwork(P)
    n_i = P.M.I.n_nodes
    receivers = slice(n_i, n_i + P.M.R.n_nodes)
    assert W[n_i:, receivers].sum() == 0 and W[:n_i, receivers].sum() > 0


def test_self_loops_with_selector_rejected():
    with pytest.raises(ValueError, match='self_loops with a selector'):
        build_patchwork(_params(_gnm('R', 'R', self_loops=0.0, source_selector='S :16')))


def test_same_layer_graph_needs_one_node_set():
    nxw = {'type': 'networkx', 'source': 'R', 'target': 'R',
           'graph': {'name': 'cycle_graph', 'params': {}}, 'source_selector': 'S :8'}
    with pytest.raises(ValueError, match='must select the same nodes'):
        build_patchwork(_params(nxw))
    rr = _rr(_params({**nxw, 'target_selector': 'S :8'}))
    assert rr[:8, :8].sum() == 16 and rr.sum() == 16           # a directed 8-cycle among R0–R7


# ── joint layers: shared layers sit on the block's diagonal ───────────────────────────────

def test_joint_layers_align_self_loops_whatever_the_written_order():
    P = _params(_gnm('R+I', 'I+R', k_avg=4, self_loops=0.0))
    _, W = build_patchwork(P)
    n = P.M.I.n_nodes + P.M.R.n_nodes
    assert np.trace(W[:n, :n]) == 0 and W[:n, :n].sum() > 0
