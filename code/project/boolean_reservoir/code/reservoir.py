import numpy as np
from pathlib import Path
import networkx as nx
from project.boolean_reservoir.code.parameter import *
from project.boolean_reservoir.code.lut import Lut, lut_random
from project.boolean_reservoir.code.wiring import build_patchwork, wiring_from_matrix, wiring_from_graph, graph_from_wiring
from project.boolean_reservoir.code.utils.utils import make_rng
from project.boolean_reservoir.code.utils.reservoir_utils import InputPerturbationStrategy, InitializationStrategy, OutputActivationStrategy, BatchedTensorHistoryWriter, SaveAndLoadModel

_jl = None

def _get_jl():
    """Lazily import juliacall and load the reservoir engine on first use."""
    global _jl
    if _jl is None:
        from juliacall import Main as _jl_main
        _jl_main.include(str(Path(__file__).parent / 'res_engine.jl'))
        # Julia's own OpenBLAS defaults to several threads per process; grid-search
        # workers are one core each, so extra threads only oversubscribe the node.
        _jl_main.seval('LinearAlgebra.BLAS.set_num_threads(1)')
        _jl = _jl_main
    return _jl


def _parse_warmup_ticks(init_str: str) -> int | None:
    parts = init_str.split('-')
    if 'warmup' in parts:
        idx = parts.index('warmup')
        if idx + 1 < len(parts):
            return int(parts[idx + 1])
    return None


class BooleanReservoir:
    def __init__(self, params: Params = None, load_path=None, load_dict=None):
        load_dict = load_dict or {}

        if load_path is not None:
            load_path = Path(load_path)
            if load_path.suffix in ('.yaml', '.yml'):
                params = load_yaml_config(load_path)
            elif load_path.is_dir():
                load_dict = SaveAndLoadModel.load(checkpoint_path=load_path, parameter_override=params)
                params = None
            else:
                raise AssertionError(f'load_path must be a yaml file or checkpoint directory, got: {load_path}')

        self.P: Params = params if params else load_dict['parameters']
        self.M = self.P.M
        self.L = self.P.L
        self.I = self.P.M.I
        self.R = self.P.M.R
        self.O = self.P.M.O
        self.T = self.P.M.T

        self.training = True

        # INPUT LAYER
        # Every stochastic component below gets its OWN stream derived from the
        # owning layer's seed. They are independent, so a partial load_dict (say
        # 'graph' present but 'lut' absent) can no longer shift the others.
        N_I = self.I.n_nodes
        N_R = self.R.n_nodes
        N_O = self.O.n_nodes
        N_total = self.M.n_nodes
        self.N_total     = N_total
        self.input_slice = slice(0, N_I)
        self.res_slice   = slice(N_I, N_I + N_R)
        self.out_slice   = slice(N_I + N_R, N_total)

        node_indices = np.arange(N_total, dtype=np.int64)
        # Input nodes are the I layer. Which reservoir nodes they feed is the I → R wiring block.
        input_nodes = np.arange(N_I, dtype=np.int64)
        input_nodes_mask = np.zeros(N_total, dtype=np.bool_)
        input_nodes_mask[self.input_slice] = True
        ticks = np.array([int(c) for c in self.I.ticks_expanded], dtype=np.uint8)

        # RESERVOIR LAYER — the wiring table is the model (see wiring.py); the graph is drawn from it
        w_bi_gen = None
        self._graph = None
        if 'wiring' in load_dict:
            adj_list_np = np.asarray(load_dict['wiring']['adj_list'], dtype=np.int64)
            adj_list_mask_np = np.asarray(load_dict['wiring']['mask'], dtype=np.bool_)
        elif 'graph' in load_dict:  # checkpoints from before the table was saved
            adj_list_np, adj_list_mask_np = wiring_from_graph(load_dict['graph'])
            self._graph = load_dict['graph']
        else:
            w_bi_gen, W = build_patchwork(self.P)
            adj_list_np, adj_list_mask_np = wiring_from_matrix(W)
        w_bi = SaveAndLoadModel.load_or_generate('w_bi', load_dict, lambda: w_bi_gen)
        assert N_total == len(adj_list_np)
        clamped_nodes = BooleanReservoir.find_clamped_nodes(adj_list_mask_np, self.res_slice, input_nodes_mask)

        self.max_connectivity = adj_list_np.shape[-1]
        lut = Lut.coerce(SaveAndLoadModel.load_or_generate('lut', load_dict, lambda:
            self.draw_lut(self.P, adj_list_mask_np)
        ), adj_list_mask_np)

        powers_of_2 = self.precompute_powers_of_2(self.max_connectivity)

        initial_states_raw = SaveAndLoadModel.load_or_generate('init_state', load_dict, lambda:
            self.initialization_strategy(self.P)
        )
        initial_states = initial_states_raw if isinstance(initial_states_raw, np.ndarray) else np.array(initial_states_raw)
        states_parallel = np.tile(initial_states, (self.T.batch_size, 1))

        # OUTPUT LAYER (no randomness: zeros + ridge; O.seed gates nothing)
        # NOTE: this mask marks the RESERVOIR SLICE (the nodes the readout layer
        # reads from), not the output nodes themselves. The name is historical
        # and kept for backward compat with the Julia engine.
        reservoir_mask = np.zeros(N_total, dtype=np.bool_)
        reservoir_mask[self.res_slice] = True
        self._bipolar_readout  = (self.O.encoding == 'bipolar')
        self._timeseries_readout = (self.O.mode == 'time-series')
        # dataset.washout: the leading input steps that only set up a sample-dependent state; the
        # time-series readout skips them (the final-state readout never sees them anyway)
        self._washout = getattr(getattr(self.P, 'D', None), 'washout', 0) or 0
        self.output_activation = self.output_activation_strategy(self.P)

        # Readout weights — zeros until ridge solve pushes trained values
        readout_W = np.zeros((self.O.n_nodes, self.R.n_nodes), dtype=np.float32)
        readout_b = np.zeros(self.O.n_nodes, dtype=np.float32)
        if 'weights' in load_dict:
            w = load_dict['weights']
            readout_W = w['W']
            readout_b = w['b']

        # LOGGING
        self.init_logging()

        # NUMPY ATTRIBUTE STORAGE
        self.node_indices          = node_indices
        self.input_nodes           = input_nodes
        self.input_nodes_mask      = input_nodes_mask
        self.ticks                 = ticks
        self.w_bi                  = w_bi
        self.adj_list              = adj_list_np
        self.adj_list_mask         = adj_list_mask_np
        self.clamped_nodes = clamped_nodes
        self.lut                   = lut
        self.powers_of_2           = powers_of_2
        self.initial_states        = initial_states
        self.states_parallel       = states_parallel
        self.reservoir_mask         = reservoir_mask
        self.readout_W             = readout_W
        self.readout_b             = readout_b
        self.states_train_backup   = states_parallel.copy()
        self.states_eval_backup    = states_parallel.copy()

        self._julia_engine         = None
        self._julia_forward_engine = None

        if self._graph is not None:  # a loaded graph is kept as is (callers share it), just labelled
            self.add_graph_labels(self._graph)
        self.reset_reservoir(hard_reset=True)

    @property
    def graph(self) -> nx.DiGraph:
        """Picture of the wiring table, drawn on first use and redrawn after install_wiring.
        Predecessor order is the LUT bit order (first predecessor = MSB)."""
        if self._graph is None:
            self._graph = graph_from_wiring(self.adj_list, self.adj_list_mask)
            self.add_graph_labels(self._graph)
        return self._graph

    def init_logging(self):
        self.out_path = Path(self.L.out_path)
        self.timestamp_utc = SaveAndLoadModel.get_timestamp_utc()
        self.save_path = self.out_path / 'runs' / self.timestamp_utc
        self.L.save_path = self.save_path
        self.L.timestamp_utc = self.timestamp_utc
        self.L.history.save_path = self.save_path / 'history'
        self.record = self.L.history.record
        self.history = BatchedTensorHistoryWriter(save_path=self.L.history.save_path, buffer_size=self.L.history.buffer_size, persist_to_disk=self.L.history.persist_to_disk) if self.record else None

    def add_graph_labels(self, graph):
        labels_mapping = {node: node for node in graph.nodes()}
        nx.set_node_attributes(graph, labels_mapping, 'id')
        labels_mapping = {k: bool(v) for k, v in zip(graph.nodes(), self.input_nodes_mask)}
        nx.set_node_attributes(graph, labels_mapping, 'I')
        labels_mapping = {k: (self.res_slice.start <= k < self.res_slice.stop) for k in graph.nodes()}
        nx.set_node_attributes(graph, labels_mapping, 'R')
        labels_mapping = {k: (self.out_slice.start <= k < self.out_slice.stop) for k in graph.nodes()}
        nx.set_node_attributes(graph, labels_mapping, 'O')

        def _part(node):
            if self.input_nodes_mask[node]:
                return 'I'
            if self.out_slice.start <= node < self.out_slice.stop:
                return 'O'
            return 'R'

        edge_quadrants = {(u, v): _part(u) + _part(v) for u, v in graph.edges()}
        nx.set_edge_attributes(graph, edge_quadrants, 'quadrant')

    @staticmethod
    def build_wiring_and_lut(params: Params):
        """Cheap init: wiring table + LUT only. Skips input/output layers and state init."""
        _, W = build_patchwork(params)
        adj_list_np, adj_list_mask_np = wiring_from_matrix(W)
        result = object.__new__(BooleanReservoir)
        result.adj_list = adj_list_np
        result.adj_list_mask = adj_list_mask_np
        result.lut = BooleanReservoir.draw_lut(params, adj_list_mask_np)
        return result

    @staticmethod
    def input_perturbation_strategy(p: Params):
        return InputPerturbationStrategy.get(p.M.I.perturbation)

    # Per-layer draws. Each layer's rows come from their own stream (make_rng(R.seed, purpose,
    # layer)), so the reservoir's LUT and initial state do not depend on how many input
    # nodes precede it: the same R.seed gives the same reservoir whatever the input width
    # (e.g. with or without a query channel).
    LAYERS = ('I', 'R', 'O')

    @staticmethod
    def draw_lut(p: Params, adj_mask: np.ndarray) -> Lut:
        """Each node's table sized by its number of inputs (adj_mask), drawn per layer."""
        degrees = adj_mask.sum(axis=1)
        parts, start = [], 0
        for L in BooleanReservoir.LAYERS:
            n = getattr(p.M, L).n_nodes
            parts.append(lut_random(degrees[start:start + n], p=p.M.R.p, rng=make_rng(p.M.R.seed, 'lut', L)))
            start += n
        return Lut.concat(parts)

    @staticmethod
    def initialization_strategy(p: Params):
        """Initial state (1, N_total). Input nodes start at 0: they are latches that carry
        only input, so an input channel that stays silent (noop_input) feeds its reservoir
        targets a constant 0 — the lower half of their table, i.e. exactly the table of the
        same node without that channel — and changes nothing."""
        init = InitializationStrategy.get(p.M.R.init.split('-')[0])
        return np.concatenate([np.zeros((1, p.M.I.n_nodes), dtype=np.uint8)]
                              + [init(getattr(p.M, L).n_nodes, rng=make_rng(p.M.R.seed, 'init_state', L))
                                 for L in ('R', 'O')], axis=1)

    @staticmethod
    def output_activation_strategy(p: Params):
        return OutputActivationStrategy.get(p.M.O.activation)

    @staticmethod
    def get_timestamp_utc():
        return SaveAndLoadModel.get_timestamp_utc()

    def save(self, save_path=None):
        if save_path is None:
            save_path = self.save_path

        paths, self.L.last_checkpoint = SaveAndLoadModel.save_model({
            'P':              self.P,
            'w_bi':           self.w_bi,
            'graph':          self.graph,
            'wiring':         {'adj_list': self.adj_list, 'mask': self.adj_list_mask},
            'initial_states': self.initial_states,
            'lut':            self.lut,
            'weights':        {'W': self.readout_W, 'b': self.readout_b},
            'save_path':      save_path,
            'history':        self.L.history,
        })
        return paths

    def load(self, checkpoint_path: Path = None, paths: dict = None, parameter_override: Params = None):
        load_dict = SaveAndLoadModel.load(checkpoint_path=checkpoint_path, paths=paths, parameter_override=parameter_override)
        self.__init__(load_dict=load_dict)
        return load_dict

    def flush_history(self):
        if self.history:
            self.history.flush()

    @staticmethod
    def precompute_powers_of_2(bits):
        return (2.0 ** np.arange(bits, dtype=np.float32)[::-1]).copy()

    @staticmethod
    def find_clamped_nodes(adj_list_mask: np.ndarray, res_slice: slice, input_nodes_mask: np.ndarray | None = None) -> np.ndarray:
        """Nodes a tick leaves alone, holding the state set from outside: the nodes with
        no incoming edges (input nodes, which the perturbation writes; output nodes), except
        reservoir nodes. An isolated reservoir node is a k=0 node: it evaluates its 0-ary
        table lut[i, 0], a constant. This keeps a silent input edge neutral — a constant-0
        MSB selects that same entry — so a query arm's twin stays the same reservoir even
        where a query edge lands on a node the walk arm leaves isolated."""
        sourceless = ~adj_list_mask.any(axis=1)
        reservoir = np.zeros(len(sourceless), dtype=np.bool_)
        reservoir[res_slice] = True
        if input_nodes_mask is not None:
            reservoir &= ~input_nodes_mask
        return np.where(sourceless & ~reservoir)[0].astype(np.int64)

    def install_wiring(self, adj_list: np.ndarray, adj_mask: np.ndarray, lut: np.ndarray) -> None:
        """Replace the wiring table and LUT (e.g. with a CA rule, see ca.py). Row n of adj_list
        lists node n's inputs, valid columns first, the first one the LUT's MSB; row n of lut
        is node n's table (its first 2**k entries are used)."""
        max_k = adj_list.shape[1]
        self.adj_list         = adj_list.astype(np.int64)
        self.adj_list_mask    = adj_mask.astype(np.bool_)
        self.clamped_nodes    = self.find_clamped_nodes(self.adj_list_mask, self.res_slice, self.input_nodes_mask)
        self.lut              = Lut.from_rows(lut, self.adj_list_mask)
        self.max_connectivity = max_k
        self.powers_of_2      = self.precompute_powers_of_2(max_k)
        self._julia_engine         = None
        self._julia_forward_engine = None
        self._graph                = None  # redrawn from the new table on next access

    def bin2int(self, x):
        return np.dot(x.astype(np.float32), self.powers_of_2).astype(np.int64)

    def train(self, mode: bool = True):
        if self.training == mode or self.R.reset:
            self.training = mode
            return self
        current_backup = self.states_train_backup if self.training else self.states_eval_backup
        np.copyto(current_backup, self.states_parallel)
        target_backup = self.states_train_backup if mode else self.states_eval_backup
        self.reset_reservoir(target_state=target_backup)
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def reset_reservoir(self, samples=None, target_state=None, hard_reset=False):
        if samples is None:
            samples = self.states_parallel.shape[0]

        warmup = 'warmup' in self.R.init
        if target_state is None:
            target_state = np.broadcast_to(self.initial_states, (samples, self.initial_states.shape[1]))
            if warmup and not self.record and self.initial_states.shape[0] == 1:
                # Every row starts from the same state and rows evolve independently,
                # so warm up ONE row and copy it — identical result, 1/batch_size of
                # the ticks (reset=True re-warms on every batch). Recording keeps the
                # full-batch path below so the history is unchanged.
                self.states_parallel[:1] = self.initial_states
                self.warmup(ticks=_parse_warmup_ticks(self.R.init), m=1)
                target_state = np.broadcast_to(self.states_parallel[:1].copy(),
                                               (samples, self.initial_states.shape[1]))
                warmup = False

        self.states_parallel[:samples] = target_state[:samples]
        if warmup:
            self.warmup(ticks=_parse_warmup_ticks(self.R.init))

        if hard_reset and hasattr(self, 'states_train_backup'):
            self.states_train_backup[:samples] = self.states_parallel[:samples].copy()
            self.states_eval_backup[:samples]  = self.states_parallel[:samples].copy()

    def __call__(self, x, lengths=None):
        return self.forward(x, lengths=lengths)

    def forward(self, x, lengths=None):
        x_np = np.asarray(x)
        m = x_np.shape[0]

        if m > self.T.batch_size:
            outputs_list = []
            kb = 0
            for i in range(m // self.T.batch_size):
                ka = i * self.T.batch_size
                kb = ka + self.T.batch_size
                outputs_list.append(self.forward(x_np[ka:kb],
                                                 None if lengths is None else lengths[ka:kb]))
            if kb < m:
                outputs_list.append(self.forward(x_np[kb:],
                                                 None if lengths is None else lengths[kb:]))
            return np.concatenate(outputs_list, axis=0)

        self._check_lengths(lengths)

        if self.R.reset:
            self.reset_reservoir(samples=m)
        self.batch_record(m, phase='init', s=0, f=0)

        x_np = x_np.reshape(m, -1, self.I.chunks, self.I.chunk_size)
        s = x_np.shape[1]
        c = self.I.chunks
        k = self.I.bits // self.I.chunks
        lengths = None if lengths is None else self._validated_lengths(lengths, m, s)
        return self._forward_julia(x_np, m, s, c, k, lengths=lengths)

    def _check_lengths(self, lengths) -> None:
        """Shared lengths preconditions for per-sample lengths. Both readout modes accept
        per-sample lengths — the engine is mode-agnostic and each mode reduces the
        same per-step state buffer (see _reduce_states). What IS required is fresh
        state per sample: with reset=False, autonomous drift during padded ticks
        would contaminate the next sample's initial state."""
        if lengths is not None and not self.R.reset:
            raise ValueError("lengths-aware reads require R.reset=True: with state "
                             "continuity across batches, padded-tick drift would "
                             "contaminate the next sample (out-of-scope by design).")

    @staticmethod
    def _validated_lengths(lengths, m: int, s: int) -> np.ndarray:
        n = np.asarray(lengths)
        if n.shape != (m,):
            raise ValueError(f"lengths must have shape ({m},) — one true step count per "
                             f"sample — got {n.shape}")
        n = n.astype(np.int64, copy=False)
        if n.size and (n.min() < 1 or n.max() > s):
            raise ValueError(f"lengths must lie in [1, {s}] (found min={n.min()}, max={n.max()})")
        return n

    _PERT_CODE = {'xor': 0, 'override': 1, 'and': 2, 'or': 3}

    def _ensure_julia_engine(self):
        if self._julia_engine is not None:
            return
        jl = _get_jl()
        engine = jl.init_engine(
            self.adj_list,
            self.adj_list_mask,
            self.clamped_nodes,
            self.lut.flat,
            self.lut.offsets,
            self.w_bi,
            self.ticks,
            self.input_nodes,
            self.reservoir_mask,
            self.N_total,
            self.max_connectivity,
            self.T.batch_size,
            self.I.n_nodes,
            self.I.chunks,
            self.I.bits // self.I.chunks,
            self._PERT_CODE[self.I.perturbation],
            self.O.n_nodes,
        )
        # Push initial readout weights into the Julia engine
        jl.set_readout_b(engine, self.readout_W, self.readout_b)
        self._julia_engine = (jl.warmup_b, engine)

    def _ensure_julia_forward_engine(self):
        if self._julia_forward_engine is not None:
            return
        self._ensure_julia_engine()
        jl = _get_jl()
        _, engine = self._julia_engine
        self._julia_forward_engine = (jl.forward_sequence_b, engine)

    def set_readout_weights(self, W: np.ndarray, b: np.ndarray):
        """Update readout weights after ridge solve. Pushes to Julia engine if initialised."""
        self.readout_W = W.astype(np.float32)
        self.readout_b = b.astype(np.float32)
        if self._julia_engine is not None:
            jl = _get_jl()
            _, engine = self._julia_engine
            jl.set_readout_b(engine, self.readout_W, self.readout_b)

    def get_states(self, x, lengths=None, heads=None):
        """Run reservoir forward pass and return pre-readout float32 states for ridge training.

        Classification mode: returns (m, n_r) — with lengths, the state at each
        sample's own step n_i (exact by causality, see _reduce_states); without,
        the final state.
        Time-series mode: returns summed states over steps (m, n_r) with bias scaling applied.
        With lengths (multistep): sums only each sample's first n_i steps and
        returns the per-sample lengths as the bias scale (c enters the bipolar offset and the
        bias consistently). lengths=None keeps the exact pre-multistep code path.
        """
        x_np = np.asarray(x)
        m = x_np.shape[0]
        self._check_lengths(lengths)

        if self.R.reset:
            self.reset_reservoir(samples=m)

        x_np = x_np.reshape(m, -1, self.I.chunks, self.I.chunk_size)
        s = x_np.shape[1]
        lengths = None if lengths is None else self._validated_lengths(lengths, m, s)

        # The engine is mode-agnostic (init_engine takes no mode): handed a per-step
        # buffer it fills the (s, m, N_total) output-state view, and the readout mode
        # only selects the reduction (_reduce_states). Classification allocates the
        # buffer only when per-sample lengths make the per-step view necessary.
        # heads=[modes]: ONE pass serves every readout head -> {mode: (states, scale)}.
        if heads is not None or self._timeseries_readout or lengths is not None:
            step_buffer = np.empty((s, m, self.N_total), dtype=np.uint8, order='F')
        else:
            step_buffer = None

        x_forder = np.asfortranarray(x_np)
        self._ensure_julia_forward_engine()
        fwd_fn, engine = self._julia_forward_engine
        fwd_fn(engine, self.states_parallel, x_forder, None, step_buffer)

        # Convert F-order (Julia interop) to C-order before boolean-mask indexing
        step_c = None if step_buffer is None else np.ascontiguousarray(step_buffer)
        if heads is not None:
            return {h: self._reduce_states(step_c, m, s, lengths, mode=h) for h in heads}
        return self._reduce_states(step_c, m, s, lengths)

    def get_step_states(self, x) -> np.ndarray:
        """Reservoir states after every step (s, m, n_r) uint8, unshifted: the same engine pass as get_states, kept
        per step (for dynamics measurements, reservoir_features). m <= T.batch_size, as get_states."""
        x_np = np.asarray(x)
        m = x_np.shape[0]
        if self.R.reset:
            self.reset_reservoir(samples=m)
        x_np = x_np.reshape(m, -1, self.I.chunks, self.I.chunk_size)
        step_buffer = np.empty((x_np.shape[1], m, self.N_total), dtype=np.uint8, order='F')
        self._ensure_julia_forward_engine()
        fwd_fn, engine = self._julia_forward_engine
        fwd_fn(engine, self.states_parallel, np.asfortranarray(x_np), None, step_buffer)
        return np.ascontiguousarray(step_buffer)[:, :m, self.res_slice]

    def _reduce_states(self, step_c, m: int, s: int, lengths, mode: str | None = None):
        """Single home for the O.mode reductions over the per-step state buffer.

        Both readout modes consume the SAME per-step output states; they differ
        only in the reduction. The prologue is shared: extract the R-slice, then
        apply the bipolar shift — `_shift` below is the single bipolar site in
        the Python stack (res_engine.jl's bipolar_shift! is the Julia twin).
        Summation order and dtype are preserved verbatim from the pre-DRY code
        so existing datasets reproduce bit-exactly.

        time-series: masked sum over each sample's first n_i steps — EXACT by
        causality (RBN updates are causal, so states at steps 1..n_i equal a true
        n_i-step run; padded-tick states are never read). The
        consumed count c = n_i enters BOTH the bipolar offset (2*sum(o) - c) and
        the returned bias scale — hence lengths, not s. lengths=None sums all s
        steps and returns s as the bias scale (pre-multistep path). A washout w
        (dataset.washout) drops the first w steps from the sum and the count.
        classification: state at each sample's own step n_i (exact by the same
        causality), else the final state; bias scale 1.
        """
        def _shift(o: np.ndarray) -> np.ndarray:
            # THE bipolar site: {0,1} → {−1,+1} for the readout (O.encoding).
            return o * 2 - 1 if self._bipolar_readout else o

        timeseries = self._timeseries_readout if mode is None else (mode == 'time-series')
        if not timeseries and lengths is None:
            # the final state: the last step's buffer entry when one was filled (== the
            # engine's final states_parallel), else states_parallel itself
            final = self.states_parallel[:m] if step_c is None else step_c[s - 1, :m]
            return _shift(final[:, self.res_slice].astype(np.float32, order='C')), 1

        if timeseries:
            w = self._washout                                    # summed steps: [w, n_i)
            if lengths is None:
                o_sum = np.zeros((m, self.R.n_nodes), dtype=np.float32)
                for si in range(w, s):
                    o_sum += _shift(step_c[si, :m, self.res_slice].astype(np.float32))
                return o_sum, s - w   # caller uses the summed step count as bias column value
            mask = (np.arange(s)[None, :] < lengths[:, None]) & (np.arange(s)[None, :] >= w)  # (m, s) bool
            o_sum = np.zeros((m, self.R.n_nodes), dtype=np.float32)
            for si in range(s):
                col = mask[:, si]
                if not col.any():
                    continue
                o_sum += _shift(step_c[si, :m, self.res_slice].astype(np.float32)) * col[:, None]
            return o_sum, lengths - w if w else lengths

        # classification + lengths: state at each sample's own step n_i
        o = step_c[np.asarray(lengths) - 1, np.arange(m), self.res_slice].astype(np.float32)
        return _shift(o), np.ones(m, dtype=np.int64)

    def _forward_julia(self, x_np: np.ndarray, m: int, s: int, c: int, k: int, lengths=None):
        self._ensure_julia_forward_engine()
        fwd_fn, engine = self._julia_forward_engine
        jl = _get_jl()

        if self.record:
            n_entries = s * (c + int(self.ticks.sum()))
            history_buffer = np.zeros((n_entries, m, self.N_total), dtype=np.uint8)
        else:
            history_buffer = None

        if self._timeseries_readout or lengths is not None:
            # ts always needs the per-step buffer (summed readout); cls needs it
            # only with lengths (per-sample final states, reduction below).
            step_buffer = np.empty((s, m, self.N_total), dtype=np.uint8, order='F')
        else:
            step_buffer = None

        x_forder = np.asfortranarray(x_np)
        fwd_fn(engine, self.states_parallel, x_forder, history_buffer, step_buffer)

        if self.record:
            for i, meta in enumerate(self._build_history_metadata(s, c)):
                self.history.append_batch(history_buffer[i], meta)

        if self._timeseries_readout:
            if lengths is None and not self._washout:
                outputs = np.array(jl.apply_readout_timeseries_b(engine, step_buffer, m, s, self._bipolar_readout))
            else:
                # Masked readout (multistep): mirror apply_readout_timeseries_b in Python
                # with per-sample step masks. Algebraically
                #     sum_s (W @ o_s + b) = W @ sum_s(o_s) + s * b,
                # so this is the Julia computation with s := n_i per sample. NOT
                # bit-identical to the Julia accumulation (different float32 summation
                # order; allclose holds) — which is exactly why the Julia path is kept
                # for lengths=None: old datasets reproduce their numbers exactly.
                # The reduction itself lives in _reduce_states (single home); a washout
                # (skipped leading steps) takes this path too, with or without lengths.
                o_sum, scale = self._reduce_states(np.ascontiguousarray(step_buffer), m, s, lengths)
                outputs = (o_sum @ self.readout_W.T
                           + np.asarray(scale, dtype=np.float32).reshape(-1, 1) * self.readout_b[None, :])
                outputs = outputs.astype(np.float32, copy=False)
        elif lengths is not None:
            # Classification + lengths: reduce the per-step buffer to each
            # sample's own final state n_i (exact by causality), then apply the
            # readout in numpy with the cls bias convention (bias scale 1).
            # Mirrors get_states(x, lengths=n); the engine readout path
            # (apply_readout_b) is final-state-of-the-padded-walk and would be
            # silently wrong for mixed lengths.
            states, _ = self._reduce_states(np.ascontiguousarray(step_buffer), m, s, lengths)
            outputs = states @ self.readout_W.T + self.readout_b[None, :]
            outputs = outputs.astype(np.float32, copy=False)
        else:
            outputs = np.array(jl.apply_readout_b(engine, self.states_parallel, m, self._bipolar_readout))

        if self.output_activation:
            outputs = self.output_activation(outputs)

        self.states_parallel[:m, self.out_slice] = (outputs > 0.5).astype(np.uint8)
        return outputs

    def _build_history_metadata(self, s: int, c: int) -> list:
        metadata = []
        for si in range(s):
            for ci in range(c):
                metadata.append({'phase': 'input_layer', 's': si + 1, 'f': ci + 1})
                t = int(self.ticks[ci])
                is_last_chunk = (ci == c - 1)
                for ti in range(t):
                    is_last_tick = (ti == t - 1)
                    if is_last_chunk and is_last_tick and (self._timeseries_readout or si == s - 1):
                        metadata.append({'phase': 'output_layer', 's': si + 1, 'f': ci + 1})
                    else:
                        metadata.append({'phase': 'reservoir_layer', 's': si + 1, 'f': ci + 1, 't': ti + 1})
        return metadata

    def _reservoir_tick(self, m):
        """Single synchronous tick of the reservoir. Useful for tests and manual stepping."""
        self._ensure_julia_engine()
        jl = _get_jl()
        _, engine = self._julia_engine
        jl.warmup_b(engine, self.states_parallel, int(m), 1)

    def warmup(self, ticks=None, m=None):
        if m is None:
            m = self.T.batch_size
        if ticks is None:
            ticks = self.ticks[0]
        self._ensure_julia_engine()
        jl = _get_jl()
        _, engine = self._julia_engine
        if self.record:
            for t in range(int(ticks)):
                jl.warmup_b(engine, self.states_parallel, int(m), 1)
                self.batch_record(m, phase='init', s=0, f=0, t=t + 1)
        else:
            jl.warmup_b(engine, self.states_parallel, int(m), int(ticks))

    def batch_record(self, m, **meta_data):
        if self.record:
            self.history.append_batch(self.states_parallel[:m], meta_data)


if __name__ == '__main__':
    I = InputParams(
        perturbation='override',
        encoding='base2',
        features=2,
        chunk_size=4,
        bits=4,
        n_nodes=8,
        ticks='2',
        seed=0,
    )
    R = ReservoirParams(n_nodes=10, k_min=0, k_avg=3, k_max=7, p=0.5, self_loops=0.1, seed=0)
    O = OutputParams(n_nodes=2, seed=0)
    T = TrainingParams(batch_size=3, epochs=10, accuracy_threshold=0.05, learning_rate=0.001, seed=0)
    L = LoggingParams(out_path='/out/delete/', history=HistoryParams(record=True, buffer_size=10))

    model_params = ModelParams(input_layer=I, reservoir_layer=R, output_layer=O, training=T, wiring=[
        PatternWiring(source='B', target='I', pattern='identity'),
        PatternWiring(source='I', target='R', pattern='identity'),
        RandomWiring(source='R', target='R', k_avg=3, k_min=0, k_max='7', self_loops='0.1'),
    ])
    params = Params(model=model_params, logging=L)
    model = BooleanReservoir(params)

    s = 1
    x = make_rng(I.seed, 'demo_input').integers(
        0, 2, (T.batch_size, s, I.features, I.bits // I.features), dtype=np.uint8)
    print(model(x))
    model.flush_history()
    model.save()
