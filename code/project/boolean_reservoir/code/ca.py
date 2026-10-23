"""Cellular automata as BooleanReservoir wirings.

A CA rule is a wiring table + truth table (see wiring.py) written straight into a model
built from a small config (config/test/boolean_reservoir/ca/*.yaml): the config only
sizes the layers, the rule replaces whatever the patchwork drew (install_wiring).

Truth tables are MSB-first over a node's inputs in column order: a node reading
(left, self, right) looks up idx = left<<2 | self<<1 | right. lut_from_fn builds one
from a plain function of those bits, so a rule reads as its definition.

Rules (the tests in test/ca check each against a pure-NumPy reference):
  rule90            x_i ← x_{i-1} ⊕ x_{i+1}                       autonomous; Sierpinski
  rule150           x_i ← x_{i-1} ⊕ x_i ⊕ x_{i+1}, with serial input driving the ring
  gkl               Gacs-Kurdyumov-Levin density classification     autonomous
  second_order_rule90  x_i(t+1) ← x_{i-1} ⊕ x_{i+1} ⊕ x_i(t-1)     reversible; keyed time-lock
"""
from typing import Callable

import numpy as np

from project.boolean_reservoir.code.encoding import BooleanTransformer
from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.reservoir import BooleanReservoir


def lut_from_fn(f: Callable[..., int], k: int) -> np.ndarray:
    """Truth table of f over k inputs, first argument = most significant bit."""
    return np.array([f(*((idx >> (k - 1 - c)) & 1 for c in range(k))) for idx in range(2 ** k)],
                    dtype=np.uint8)


def _pad(lut: np.ndarray, width: int) -> np.ndarray:
    return np.concatenate([lut, np.zeros(width - len(lut), dtype=np.uint8)])


XOR2 = lut_from_fn(lambda l, r: l ^ r, 2)
XOR3 = lut_from_fn(lambda a, b, c: a ^ b ^ c, 3)
FIRST3 = lut_from_fn(lambda a, b, c: a, 3)  # copy the first input, ignore the rest
COPY1 = lut_from_fn(lambda a: a, 1)


def _gkl(l3, l1, s, r1, r3):
    """GKL: a 0 cell takes the majority of (itself, i-1, i-3); a 1 cell of (itself, i+1, i+3)."""
    return int(s + l1 + l3 >= 2) if s == 0 else int(s + r1 + r3 >= 2)


GKL_OFFSETS = (-3, -1, 0, +1, +3)
GKL = lut_from_fn(_gkl, 5)


# ── wiring helpers ────────────────────────────────────────────────────────────────────────

def ring(model: BooleanReservoir, offsets: tuple) -> tuple[np.ndarray, np.ndarray]:
    """(adj_list, mask) over all nodes: R node i reads R nodes (i + offset) % n_r, one
    column per offset in the given order; I and O rows read nothing."""
    n_r, start = model.R.n_nodes, model.res_slice.start
    adj = np.zeros((model.N_total, len(offsets)), dtype=np.int64)
    mask = np.zeros_like(adj, dtype=np.bool_)
    i = np.arange(n_r)
    adj[model.res_slice] = start + (i[:, None] + np.array(offsets)[None, :]) % n_r
    mask[model.res_slice] = True
    return adj, mask


def every_row(lut: np.ndarray, model: BooleanReservoir) -> np.ndarray:
    return np.tile(lut, (model.N_total, 1))


# ── rules ─────────────────────────────────────────────────────────────────────────────────

def install_rule90(model: BooleanReservoir) -> None:
    adj, mask = ring(model, (-1, +1))
    model.install_wiring(adj, mask, lut=every_row(XOR2, model))


def install_gkl(model: BooleanReservoir) -> None:
    adj, mask = ring(model, GKL_OFFSETS)
    model.install_wiring(adj, mask, lut=every_row(GKL, model))


def install_rule150(model: BooleanReservoir, injected: str = 'xor') -> None:
    """Rule 150 ring driven by serial input. I node j replaces the left neighbour of the
    ring cell j * (n_r // n_i), so the injection points spread evenly (one per I node).

    injected='xor'    the injected cell XORs the input into its update (transducer):
                      x_c ← u ⊕ x_c ⊕ x_{c+1}
    injected='input'  the injected cell IS the input, no mixing (linear transducer):
                      x_c ← u, and its neighbours see the previous input
    """
    n_i = model.I.n_nodes
    adj, mask = ring(model, (-1, 0, +1))
    lut = every_row(XOR3, model)
    cells = model.res_slice.start + np.arange(n_i) * (model.R.n_nodes // n_i)
    adj[cells, 0] = model.input_nodes
    if injected == 'input':
        lut[cells] = FIRST3
    elif injected != 'xor':
        raise ValueError(f"injected must be 'xor' or 'input', got {injected!r}")
    model.install_wiring(adj, mask, lut=lut)


def install_second_order_rule90(model: BooleanReservoir, direction: str = 'forward') -> None:
    """Second-order Rule 90 as a doubled state: R nodes 0..n-1 hold x(t) ('current'),
    n..2n-1 hold x(t-1) ('previous'), n = n_r // 2. The rule is its own inverse, so the
    backward wiring swaps the two layers' roles and undoes the forward steps exactly.

      forward   current[i]  ← current[i-1] ⊕ current[i+1] ⊕ previous[i]    previous[i] ← current[i]
      backward  previous[i] ← previous[i-1] ⊕ previous[i+1] ⊕ current[i]   current[i]  ← previous[i]

    The key bit (I node 0) replaces the left neighbour of the centre cell of the layer
    that computes: current in the forward direction, previous in the backward one.
    """
    if direction not in ('forward', 'backward'):
        raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")
    n = model.R.n_nodes // 2
    start = model.res_slice.start
    cur, prev = start + np.arange(n), start + n + np.arange(n)
    computing, copying = (cur, prev) if direction == 'forward' else (prev, cur)

    adj = np.zeros((model.N_total, 3), dtype=np.int64)
    mask = np.zeros_like(adj, dtype=np.bool_)
    lut = np.zeros((model.N_total, 8), dtype=np.uint8)
    adj[computing] = np.stack([np.roll(computing, 1), np.roll(computing, -1), copying], axis=1)
    mask[computing] = True
    lut[computing] = XOR3
    adj[copying, 0] = computing
    mask[copying, 0] = True
    lut[copying] = _pad(COPY1, 8)
    adj[computing[n // 2], 0] = model.input_nodes[0]
    model.install_wiring(adj, mask, lut=lut)


# ── building and driving a CA model ─────────────────────────────────────────────────────

def viz_mode(P) -> None:
    """One input bit per forward step, so each step is exactly one reservoir tick."""
    I = P.M.I
    I.features = I.bits = I.n_nodes = I.chunks = I.chunk_size = 1


def ca_model(config, install: Callable[[BooleanReservoir], None], viz: bool = False,
             edit: Callable | None = None) -> BooleanReservoir:
    """A model from a CA config with a rule installed. viz: one tick per step and the
    history persisted, for trace runs. edit(P) changes the config before the build.
    No global seeding: the CA configs pin seed 0 and every draw derives from it."""
    P = load_yaml_config(config)
    if viz:
        viz_mode(P)
        P.L.history.persist_to_disk = True
    if edit:
        edit(P)
    model = BooleanReservoir(P)
    install(model)
    return model


def set_state(model: BooleanReservoir, x: np.ndarray) -> None:
    """Start the reservoir from x (written from the first R node on; the rest keep theirs)."""
    start = model.res_slice.start
    model.initial_states[0, start:start + len(x)] = x
    model.reset_reservoir(hard_reset=True)


def ca_input(bits: np.ndarray, P) -> np.ndarray:
    """A 1-D bit stream as forward() input, shaped by the config's input encoding (so
    redundancy tiles each bit onto every I node). Without I nodes: (1, T, 1, 1) zeros."""
    if P.M.I.n_nodes == 0:
        return np.zeros((1, len(bits), 1, 1), dtype=np.uint8)
    return BooleanTransformer(P.M.I)(np.asarray(bits, dtype=np.uint8).reshape(1, -1, 1, 1))
