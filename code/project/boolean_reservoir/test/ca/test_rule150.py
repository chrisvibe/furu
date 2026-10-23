"""
Rule 150 driven by serial input: the end-to-end test of the input path.

Rule 150: S_i(t+1) = S_{i-1}(t) ⊕ S_i(t) ⊕ S_{i+1}(t)   (XOR of 3 neighbours)
Reference: Wolfram, A New Kind of Science (2002), p. 886 and Section 14.1, p. 952;
           Kutrib & Malcher (2012), "Transductions Computed by One-Dimensional
           Cellular Automata" (sequential input, boundary cell = source).

At every tick a new bit u_t enters the ring at cell 0, whose left neighbour is the
I node instead of cell N-1 (ca.install_rule150). Two ways for cell 0 to take it:

  injected='xor'    transducer:         x_0 ← u_t ⊕ x_0 ⊕ x_1
  injected='input'  linear transducer:  x_0 ← u_t (cell 0 IS the input; its
                    neighbours see u_{t-1}). Exercises a per-node LUT: cell 0's
                    table differs from the rest of the ring's.

Why this is the driven-input correctness test: Rule 90, GKL and the second-order
rule are autonomous or key-driven at one cell, while here the full pipeline runs
on every step —

  bit stream → override perturbation → I node → adjacency boundary → R

Rule 150 is additive over GF(2), so an error at any step (perturbation mode, I-node
read, boundary connection, per-node LUT) propagates into the final state: equality
after T=30 random bits implies every step was right.
"""
import numpy as np
import pytest
from pathlib import Path

from project.boolean_reservoir.code.ca import ca_model, install_rule150

CONFIGS = {
    'xor':   Path('config') / 'test' / 'boolean_reservoir' / 'ca' / 'rule150_transducer.yaml',
    'input': Path('config') / 'test' / 'boolean_reservoir' / 'ca' / 'rule150_linear_transducer.yaml',
}
INJECTED = list(CONFIGS)


def rule150_reference(n: int, input_bits: np.ndarray, injected: str) -> np.ndarray:
    """Pure-NumPy Rule 150 with serial input at cell 0. Returns (T+1, n) history."""
    s = np.zeros(n, dtype=np.uint8)
    history = [s.copy()]
    for bit in input_bits:
        nxt = np.empty_like(s)
        nxt[0]    = int(bit) if injected == 'input' else int(bit) ^ s[0] ^ s[1]
        nxt[1:-1] = s[:-2] ^ s[1:-1] ^ s[2:]
        nxt[-1]   = s[-2]  ^ s[-1]   ^ s[0]   # periodic on the right
        s = nxt
        history.append(s.copy())
    return np.stack(history)


def make_model(injected: str):
    return ca_model(CONFIGS[injected], lambda m: install_rule150(m, injected))


@pytest.mark.parametrize('injected', INJECTED)
def test_all_zeros_fixed_point(injected):
    """Zero state, zero input: 0 ⊕ 0 ⊕ 0 = 0 everywhere (R.init=zeros, reset=True)."""
    model = make_model(injected)
    model(np.zeros((1, 30, 1, 1), dtype=np.uint8))
    assert np.all(model.states_parallel[0, model.res_slice] == 0)


@pytest.mark.parametrize('injected', INJECTED)
def test_matches_reference(injected):
    """A seeded random stream (1, T, 1, 1): one bit to I node 0 before each tick."""
    T = 30
    model = make_model(injected)
    bits = np.random.default_rng(0).integers(0, 2, T, dtype=np.uint8)
    model(bits.reshape(1, T, 1, 1))
    expected = rule150_reference(model.R.n_nodes, bits, injected)[-1]
    assert np.array_equal(model.states_parallel[0, model.res_slice], expected), \
        f"Rule 150 ({injected}) mismatch at the final step"
