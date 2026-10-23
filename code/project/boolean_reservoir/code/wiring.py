"""The wiring table: what a reservoir IS.

Node n reads its inputs adj_list[n, :k] (k = mask[n].sum(), valid columns first) and
looks up its next state in its truth table at idx = sum(input_c << (k-1-c)): column 0 is
the most significant bit. So the ORDER of a node's inputs is part of the model, which a
plain graph does not record. Everything that builds a reservoir writes this table; the
networkx graph is a picture drawn from it (graph_from_wiring) and is never edited.

Builders:
  - the config patchwork fills a block matrix W (W[u, v] = 1: u feeds v) and lists each
    node's inputs in ascending node id (wiring_from_matrix). With joint I+R sources the
    input nodes have the lowest ids, so a query edge is always the MSB — which is what
    keeps a silent query channel neutral (see BooleanReservoir.find_clamped_nodes).
  - CA rules write rows directly (BooleanReservoir.install_wiring).
  - checkpoints saved before the table was stored rebuild it from their graph
    (wiring_from_graph): predecessor order, which for those graphs is ascending node id.
"""
from dataclasses import dataclass
from inspect import signature

import networkx as nx
import numpy as np

from project.boolean_reservoir.code.graph import generate_adjacency_matrix
from project.boolean_reservoir.code.parameter import GNMWiring, GNPWiring, NetworkXWiring, ExplicitWiring, PatternWiring
from project.boolean_reservoir.code.utils.param_utils import ExpressionEvaluator
from project.boolean_reservoir.code.utils.reservoir_utils import ChainedSelector, BipartiteMappingStrategy
from project.boolean_reservoir.code.utils.utils import make_rng


def _pad(rows: list[np.ndarray], n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    width = max((len(r) for r in rows), default=1)
    adj_list = np.zeros((n_nodes, width), dtype=np.int64)
    mask = np.zeros((n_nodes, width), dtype=np.bool_)
    for n, r in enumerate(rows):
        adj_list[n, :len(r)] = r
        mask[n, :len(r)] = True
    return adj_list, mask


def wiring_from_matrix(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(adj_list, mask) from a block matrix, each node's inputs in ascending node id."""
    v, u = np.nonzero(np.asarray(W).T)  # row-major over W.T: grouped by target, sources ascending
    splits = np.cumsum(np.bincount(v, minlength=W.shape[1]))[:-1]
    return _pad(np.split(u.astype(np.int64), splits), W.shape[1])


def wiring_from_graph(graph: nx.DiGraph) -> tuple[np.ndarray, np.ndarray]:
    """(adj_list, mask) from a graph, inputs in predecessor order."""
    n = graph.number_of_nodes()
    return _pad([np.fromiter(graph.predecessors(v), dtype=np.int64) for v in range(n)], n)


def graph_from_wiring(adj_list: np.ndarray, mask: np.ndarray) -> nx.DiGraph:
    """The picture of a table. Predecessor order follows column order, so
    wiring_from_graph(graph_from_wiring(a, m)) == (a, m) — except that a node reading the
    same source twice shows one edge (a DiGraph has no parallel edges)."""
    g = nx.DiGraph()
    g.add_nodes_from(range(len(adj_list)))
    g.add_edges_from((int(u), v, {'weight': 1})
                     for v in range(len(adj_list)) for u in adj_list[v][mask[v]])
    return g


# ── The config patchwork ─────────────────────────────────────────────────────────────────
#
# P.M.wiring is a list of blocks, each wiring source layer(s) → target layer(s), e.g.
# 'R' → 'R' or 'I+R' → 'R'. Layers: B (input bits, only ever B → I), I, R, O. Each block
# fills its own slice of the block matrix from its own random stream, keyed by its index
# in the list, so adding a block does not shift the ones before it. Unwired pairs are 0.
#
# Selectors restrict a block to some source rows / target columns, indexed over the
# layers in the order written (source: I+R → I's nodes first). The block is drawn ON
# the selected rows × columns only, so k_avg is the in-degree each selected receiver
# gets, whichever side is selected.

GRAPH_LAYERS = 'IRO'


@dataclass
class _Block:
    """What a builder sees: one block of the patchwork, already narrowed by its selectors."""
    w: object
    P: object
    n_rows: int        # sources (after selection)
    n_cols: int        # targets (after selection)
    same_layer: bool   # one layer onto itself: the block is square and its diagonal is self-loops
    rng: np.random.Generator

    def evaluator(self, **extra):
        numeric = {k: v for k, v in self.P.M.variables.items() if not isinstance(v, str)}
        return ExpressionEvaluator({**self.structural, **numeric, **extra})

    @property
    def structural(self) -> dict:
        P = self.P
        return {'I_n_nodes': P.M.I.n_nodes, 'R_n_nodes': P.M.R.n_nodes,
                'src_size': self.n_rows, 'tgt_size': self.n_cols, 'n_nodes': self.n_cols}


def _resolve_random_wiring_params(b: _Block):
    w, variables = b.w, b.P.M.variables
    numeric = {k: v for k, v in variables.items() if not isinstance(v, str)}
    k_avg = float(ExpressionEvaluator({**numeric, **b.structural}).eval(w.k_avg))
    pre_ev = b.evaluator(k_avg=k_avg)
    all_vars = {}
    for key, val in variables.items():
        if isinstance(val, str):
            try:
                all_vars[key] = float(pre_ev.eval(val))
            except Exception:
                all_vars[key] = val
        else:
            all_vars[key] = val
    ev = ExpressionEvaluator({**b.structural, **all_vars, 'k_avg': k_avg})
    k_min = int(round(ev.eval(w.k_min)))
    k_max = int(round(ev.eval(w.k_max)))
    self_loops = float(ev.eval(w.self_loops)) if w.self_loops is not None else None
    if str(all_vars.get(w.mode, w.mode)) == 'homogeneous':
        k_min = k_max = int(k_avg)
    return k_min, k_avg, k_max, self_loops


def _random_block(b: _Block) -> np.ndarray:
    k_min, k_avg, k_max, self_loops = _resolve_random_wiring_params(b)
    fixed_edges = isinstance(b.w, GNMWiring)
    if b.w.degree == 'out':
        return generate_adjacency_matrix(
            n_nodes=b.n_rows, k_min=k_min, k_avg=k_avg, k_max=min(k_max, b.n_cols),
            self_loops=self_loops, rows=b.n_cols, fixed_edges=fixed_edges, rng=b.rng).T
    return generate_adjacency_matrix(
        n_nodes=b.n_cols, k_min=k_min, k_avg=k_avg, k_max=min(k_max, b.n_rows),
        self_loops=self_loops, rows=b.n_rows, fixed_edges=fixed_edges, rng=b.rng)


def _nx_seed_kwargs(fn, w, rng):
    """networkx random generators are decorated @np_random_state / @py_random_state:
    called without an explicit `seed` they draw from the GLOBAL numpy/stdlib RNG.
    Inject a seed drawn from this block's stream — unless the config pins one,
    which stays authoritative."""
    if 'seed' not in signature(fn).parameters:
        return {}
    if 'seed' in {k for k, _ in w.graph.params}:
        return {}
    return {'seed': int(rng.integers(0, 2**31 - 1))}


def _nx_fn(module, name):
    fn = getattr(module, name, None)
    if fn is None:
        raise ValueError(f"{'nx.bipartite' if module is nx.bipartite else 'nx'} has no function {name!r}.")
    return fn


def _nx_graph_to_block(g, b: _Block) -> np.ndarray:
    if b.w.directed and not g.is_directed():
        g = g.to_directed()
    if b.same_layer:
        return np.array(nx.to_numpy_array(g, nodelist=range(b.n_cols)), dtype=np.uint8)
    return np.array(nx.bipartite.biadjacency_matrix(g, row_nodes=range(b.n_rows)).toarray(), dtype=np.uint8)


def _networkx_block(b: _Block) -> np.ndarray:
    w, ev = b.w, b.evaluator()
    if b.same_layer:
        fn = _nx_fn(nx, w.graph.name)
        g = w.graph.call(fn, evaluator=ev, n=b.n_cols, **_nx_seed_kwargs(fn, w, b.rng))
    else:
        fn = _nx_fn(nx.bipartite, w.graph.name)
        g = w.graph.call(fn, evaluator=ev, n=b.n_rows, m=b.n_cols, **_nx_seed_kwargs(fn, w, b.rng))
    block = _nx_graph_to_block(g, b)
    if w.k_max is not None:  # generators know no degree cap: drop a random excess per node
        for col in np.where(block.sum(axis=0) > w.k_max)[0]:
            excess = np.where(block[:, col])[0]
            b.rng.shuffle(excess)
            block[excess[w.k_max:], col] = 0
    return block


def _explicit_block(b: _Block) -> np.ndarray:
    g = nx.convert_node_labels_to_integers(b.w.graph.call(_nx_fn(nx, b.w.graph.name)))
    expected = b.n_cols if b.same_layer else b.n_rows + b.n_cols
    if g.number_of_nodes() != expected:
        raise ValueError(
            f"ExplicitWiring: loaded graph has {g.number_of_nodes()} nodes, "
            f"expected {expected} for {b.w.source!r} -> {b.w.target!r}.")
    return _nx_graph_to_block(g, b)


def _pattern_block(b: _Block) -> np.ndarray:
    strategy_fn = BipartiteMappingStrategy.get(b.w.pattern, k_avg=b.P.M.variables.get('R_k_avg'))
    return strategy_fn(b.P, b.n_rows, b.n_cols, rng=b.rng).astype(np.uint8)


BLOCK_BUILDERS = {
    GNMWiring: _random_block,
    GNPWiring: _random_block,
    NetworkXWiring: _networkx_block,
    ExplicitWiring: _explicit_block,
    PatternWiring: _pattern_block,
}


def layer_sizes(P) -> dict:
    return {'B': P.M.I.bits, 'I': P.M.I.n_nodes, 'R': P.M.R.n_nodes, 'O': P.M.O.n_nodes}


def _select(chain, layers, order, sizes, rng) -> np.ndarray:
    """Selected positions in `order` (the block's layout), from a chain indexed over
    `layers` as written. Sorted and unique: a selection is a set of nodes."""
    start = dict(zip(order, np.cumsum([0] + [sizes[l] for l in order])))
    to_block = np.concatenate([start[l] + np.arange(sizes[l]) for l in layers]).astype(np.int64)
    if not chain:
        return np.sort(to_block)
    return np.unique(to_block[ChainedSelector(len(to_block), rng=rng).eval(chain)])


def build_patchwork(P) -> tuple[np.ndarray, np.ndarray]:
    """(w_bi, W): the B → I mapping and the block matrix over I, R, O (W[u, v] = 1: u feeds v)."""
    sizes = layer_sizes(P)
    quadrants = {}
    for wi, w in enumerate(P.M.wiring):
        src, tgt = w.source.split('+'), w.target.split('+')
        # Rows: layers that are also targets first, in the target's order, so a shared
        # layer sits on the block's diagonal (block[i, i] is node i → itself).
        order = [t for t in tgt if t in src] + [s for s in src if s not in tgt]
        for s in src:
            for t in tgt:
                if (s, t) in quadrants:
                    raise ValueError(f"Duplicate wiring for ({s}→{t})")

        rows = _select(w.source_selector, src, order, sizes, make_rng(P.M.R.seed, 'source_selector', wi))
        cols = _select(w.target_selector, tgt, tgt, sizes, make_rng(P.M.R.seed, 'target_selector', wi))
        same_layer = len(src) == 1 and src == tgt
        selected = bool(w.source_selector or w.target_selector)
        if selected and same_layer and type(w) in (NetworkXWiring, ExplicitWiring) and not np.array_equal(rows, cols):
            raise ValueError(f"{w.source}→{w.target}: a same-layer graph generator draws a graph ON a set of "
                             f"nodes, so source_selector and target_selector must select the same nodes.")
        if selected and getattr(w, 'self_loops', None) is not None:
            raise ValueError(f"{w.source}→{w.target}: self_loops with a selector is ambiguous (the drawn "
                             f"block's diagonal is not the nodes' own self-loops); drop one of them.")

        b = _Block(w=w, P=P, n_rows=len(rows), n_cols=len(cols), same_layer=same_layer,
                   rng=make_rng(P.M.R.seed, 'wiring', wi))
        block = np.zeros((sum(sizes[l] for l in order), sum(sizes[t] for t in tgt)), dtype=np.uint8)
        block[np.ix_(rows, cols)] = BLOCK_BUILDERS[type(w)](b)

        r0 = 0
        for s in order:
            c0 = 0
            for t in tgt:
                quadrants[(s, t)] = block[r0:r0 + sizes[s], c0:c0 + sizes[t]]
                c0 += sizes[t]
            r0 += sizes[s]

    def q(s, t):
        return quadrants.get((s, t), np.zeros((sizes[s], sizes[t]), dtype=np.uint8))
    W = np.block([[q(s, t) for t in GRAPH_LAYERS] for s in GRAPH_LAYERS])
    return q('B', 'I'), W
