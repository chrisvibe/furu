"""
Second-Order Rule 90 — Reversible Time-Lock Test.

Second-order (Toffoli-Margolus) Rule 90:
  x_i(t+1) = x_{i-1}(t) ⊕ x_{i+1}(t) ⊕ x_i(t-1)

This rule is always invertible for any N — the inverse is the same formula:
  x_i(t-1) = x_{i-1}(t) ⊕ x_{i+1}(t) ⊕ x_i(t+1)

Reference: Toffoli & Margolus (1987) "Cellular Automata Machines," MIT Press, Ch. 6.
           https://mitpress.mit.edu/9780262200608/cellular-automata-machines/

Doubled-state implementation (ca.install_second_order_rule90; 2N = 64 R-nodes, N = 32):
  Nodes   0..N-1   — current  layer x_i(t)
  Nodes   N..2N-1  — previous layer x_i(t-1)

Forward tick (lock direction):
  new_curr[i]  = left_curr ⊕ right_curr ⊕ self_prev   K=3, ca.XOR3
  new_prev[i]  = curr[i]                               K=1, ca.COPY1 (identity)

Backward tick (unlock direction):
  new_curr[i]  = prev[i]                               K=1, ca.COPY1
  new_prev[i]  = left_prev ⊕ right_prev ⊕ self_curr   K=3, ca.XOR3

A K=1 node reads entries 0..1 of its (zero-padded) LUT row: an identity lookup.

Time-lock structure:
  Lock:   T forward steps via BooleanReservoir with key stream  u_0 … u_{T-1}
  Unlock: T inverse  steps via pure-Python reference with reversed key u_{T-1} … u_0
  The round-trip assertion follows from mathematical invertibility alone — no
  independent reference is needed.  Any forward error (wrong LUT, wiring, I-node
  override, or double-buffer) propagates over all T steps and prevents recovery.

This is a stronger test than rule90_eca or rule150_transducer because:
  - It exercises heterogeneous K (K=3 current layer, K=1 previous layer) in one model.
  - Invertibility catches errors that a single final-state comparison might miss.
  - The key stream exercises the override perturbation path on every step.
"""

import numpy as np
from pathlib import Path

from project.boolean_reservoir.code.ca import ca_model, ca_input, install_second_order_rule90, set_state

CONFIG = Path('config') / 'test' / 'boolean_reservoir' / 'ca' / 'second_order_rule90.yaml'

N = 32   # visible ring size; the config's reservoir has 2N = 64 R nodes


# ── Reference (pure-NumPy) ─────────────────────────────────────────────────────

def _second_order_rule90_unlock(locked: np.ndarray, key_reversed: np.ndarray) -> tuple:
    """Inverse of T forward steps: recovers (x(0), x(-1)) from the locked state.

    locked:       1-D array of length 2n — [x(T) | x(T-1)]
    key_reversed: key stream in reverse order, shape (T,); the key enters at cell n // 2
    Returns: (x_rec, xp_rec) — should equal (x0, x_minus1)
    """
    n = len(locked) // 2
    x, xp = locked[:n].copy(), locked[n:].copy()
    for u in key_reversed:
        left         = np.roll(xp, 1)   # left[i] = xp[(i-1) % n]
        left[n // 2] = u
        x, xp        = xp, left ^ np.roll(xp, -1) ^ x
    return x, xp


def make_model():
    model = ca_model(CONFIG, install_second_order_rule90)
    assert model.R.n_nodes == 2 * N
    return model


# ── Pytest tests ───────────────────────────────────────────────────────────────

def test_second_order_rule90_zeros_fixedpoint():
    """All-zero doubled state with zero key must remain all-zero.

    x(t+1)[i] = 0⊕0⊕0 = 0; previous layer copies 0.  R.init=zeros + reset=True
    starts from all-zeros automatically — no manual state setup required.
    """
    model = make_model()
    x = ca_input(np.zeros(30, dtype=np.uint8), model.P)
    model(x)
    assert np.all(model.states_parallel[0, model.res_slice] == 0)


def test_second_order_rule90_timelock():
    """
    BooleanReservoir, forward-wired as second-order Rule 90, must recover the
    initial state after T forward steps (lock) + pure-Python inverse with reversed
    key (unlock).

    Time-lock structure:
      x0 = random initial state (current layer),  x_minus1 = zeros (previous layer)
      Lock:   T forward steps via BooleanReservoir with seeded key u_0…u_{T-1}
      Unlock: pure-Python inverse with reversed key u_{T-1}…u_0
      Assert: recovered current layer == x0  AND  recovered prev layer == zeros

    Invertibility of second-order Rule 90 is a mathematical guarantee for any N.
    Any error in the BooleanReservoir forward pass — wrong LUT, wiring, I-node
    override perturbation, or double-buffer synchrony — accumulates over T steps
    and prevents recovery.

    Covers: K=3 XOR3 LUT, K=1 COPY1 (identity), heterogeneous K in one model,
    I-node override at center cell for the current layer, ring-boundary wiring for
    the 2N doubled-state reservoir, and the Julia double-buffer synchronous update.
    """
    T   = 40
    rng = np.random.default_rng(42)
    x0       = rng.integers(0, 2, N, dtype=np.uint8)
    key_bits = rng.integers(0, 2, T, dtype=np.uint8)
    x_minus1 = np.zeros(N, dtype=np.uint8)

    model = make_model()
    set_state(model, x0)   # current layer ← x0; previous layer stays zero (R.init=zeros)

    model(ca_input(key_bits, model.P))

    locked = model.states_parallel[0, model.res_slice].copy()   # shape (2N,)

    x_rec, xp_rec = _second_order_rule90_unlock(locked, key_bits[::-1])

    assert np.array_equal(x_rec, x0), (
        "Second-order Rule 90 time-lock: current layer not recovered after round-trip"
    )
    assert np.array_equal(xp_rec, x_minus1), (
        "Second-order Rule 90 time-lock: previous layer (zeros) not recovered"
    )


def test_second_order_rule90_backward_wiring():
    """
    BooleanReservoir with backward wiring must recover x0 exactly.

    This test exercises the backward wiring directly — the one the trace runs
    (test/ca/trace_runs.py) use for the unlock half.
    It mirrors test_second_order_rule90_timelock but runs the backward pass through
    BooleanReservoir rather than the pure-Python reference, confirming that the
    trace runs' start and end states match.

    Sequence:
      1. Forward  T steps with BooleanReservoir (forward wiring) → locked
      2. Backward T steps with BooleanReservoir (backward wiring) → recovered
      3. Assert: recovered current layer == x0, recovered previous == zeros
    """
    T   = 40
    rng = np.random.default_rng(42)
    x0       = rng.integers(0, 2, N, dtype=np.uint8)
    key_bits = rng.integers(0, 2, T, dtype=np.uint8)

    # --- Forward (lock) ---
    model = make_model()
    set_state(model, x0)

    model(ca_input(key_bits, model.P))
    locked = model.states_parallel[0, model.res_slice].copy()

    # --- Backward (unlock) via BooleanReservoir ---
    install_second_order_rule90(model, 'backward')
    set_state(model, locked)

    model(ca_input(key_bits[::-1].copy(), model.P))

    recovered = model.states_parallel[0, model.res_slice]
    assert np.array_equal(recovered[:N], x0), (
        "Backward BooleanReservoir wiring: current layer not recovered"
    )
    assert np.array_equal(recovered[N:], np.zeros(N, dtype=np.uint8)), (
        "Backward BooleanReservoir wiring: previous layer (zeros) not recovered"
    )
