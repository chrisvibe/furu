"""
Gold-standard correctness test: Rule 90 Elementary Cellular Automaton.

Rule 90: state[i, t+1] = state[i-1, t] XOR state[i+1, t]

Seeding the centre node of an N-node ring with 1 produces a Sierpinski triangle. The
config has no input nodes, so forward() runs purely autonomous ticks. Any bug in the
adjacency parsing, LUT indexing, bit-shift ordering or double-buffer logic produces a
wrong pattern.

Input order: ca.install_rule90 puts the left neighbour in column 0 (MSB) and the right
one in column 1, so idx = left<<1 | right, the XOR lookup [0, 1, 1, 0].
"""
import numpy as np
from pathlib import Path

from project.boolean_reservoir.code.ca import ca_model, install_rule90, set_state

CONFIG = Path('config') / 'test' / 'boolean_reservoir' / 'ca' / 'rule90.yaml'
T = 30   # time steps to verify


def rule90_reference(n: int, steps: int) -> np.ndarray:
    """Pure-Python Rule 90 on an n-node ring from a centre seed; (steps+1, n) history."""
    state = np.zeros(n, dtype=np.uint8)
    state[n // 2] = 1
    history = [state.copy()]
    for _ in range(steps):
        state = np.array([state[(i - 1) % n] ^ state[(i + 1) % n] for i in range(n)], dtype=np.uint8)
        history.append(state.copy())
    return np.stack(history)


def test_rule90_eca_gold_standard():
    """The ring with the XOR LUT must reproduce Rule 90's Sierpinski pattern exactly
    at step T. reset=True restarts each forward() from the centre-seeded state.

    Covers: adjacency parsing, Julia 1-indexing, LUT endianness, bit-shift ordering
    and the double-buffered synchronous update."""
    model = ca_model(CONFIG, install_rule90)
    n = model.R.n_nodes
    seed = np.zeros(n, dtype=np.uint8); seed[n // 2] = 1
    set_state(model, seed)

    model(np.zeros((1, T)))

    assert np.array_equal(model.states_parallel[0, model.res_slice], rule90_reference(n, T)[-1]), \
        "Rule 90 mismatch at final step"
