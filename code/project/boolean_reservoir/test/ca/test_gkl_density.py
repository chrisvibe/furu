"""
GKL (Gacs-Kurdyumov-Levin) Density Classification Rule.

Rule defined in: Gacs, Kurdyumov & Levin (1978).
Task context: https://en.wikipedia.org/wiki/Majority_problem_(cellular_automaton)

GKL transition for a 1-D ring of N cells with periodic boundaries:
  if S_i == 0: S_i(t+1) = Majority(S_i, S_{i-1}, S_{i-3})
  if S_i == 1: S_i(t+1) = Majority(S_i, S_{i+1}, S_{i+3})

Majority(a, b, c) = 1 iff a + b + c >= 2.

ca.install_gkl folds the state-dependent rule into one fixed 5-input LUT over
(i-3, i-1, i, i+1, i+3), MSB first: ca.GKL = lut_from_fn of the definition above.

Correctness proof
-----------------
test_gkl_reservoir_matches_reference is the formal correctness test: it runs
30 steps from a random initial state and checks every bit against a pure-NumPy
reference that implements the majority rule directly (not via the LUT).  Any wrong LUT entry, wrong column offset, or broken
endianness fails it immediately.  This is the GKL equivalent of Rule 90's
Sierpinski triangle test.

The trace runs (test/ca/trace_runs.py, gkl_density) demonstrate
GKL *behaviour* — convergence toward a uniform state driven by density — but
does NOT prove identity.  Unlike Rule 90's Sierpinski, convergence to all-0s/1s
is a property of many rules, not a fingerprint of GKL specifically.  The
reference-match test is what closes that gap.
"""

import numpy as np
from pathlib import Path

from project.boolean_reservoir.code.ca import ca_model, install_gkl, set_state

CONFIG = Path('config') / 'test' / 'boolean_reservoir' / 'ca' / 'gkl_density.yaml'

N = 149    # standard GKL ring size (the config's R.n_nodes)
T = 3 * N  # steps sufficient for convergence at extreme densities


def _gkl_reference(state: np.ndarray, steps: int) -> np.ndarray:
    """Pure-NumPy GKL from its definition, on a periodic ring. Returns (steps+1, N) history."""
    s = state.astype(np.uint8).copy()
    history = [s.copy()]
    for _ in range(steps):
        left = s + np.roll(s, 1) + np.roll(s, 3)     # self, i-1, i-3
        right = s + np.roll(s, -1) + np.roll(s, -3)  # self, i+1, i+3
        s = np.where(s == 0, left >= 2, right >= 2).astype(np.uint8)
        history.append(s.copy())
    return np.stack(history)


def make_model():
    return ca_model(CONFIG, install_gkl)


def test_gkl_uniform_fixed_points():
    """All-0s and all-1s must be fixed points in the BooleanReservoir with GKL wiring.

    n_nodes=0: forward(np.zeros((1, 1))) runs 1 GKL tick with no input.
    A true fixed point stays fixed for any tick count.
    """
    model = make_model()
    for val in (0, 1):
        set_state(model, np.full(model.R.n_nodes, val, dtype=np.uint8))
        model(np.zeros((1, 1)))
        state = model.states_parallel[0, model.res_slice]
        assert np.all(state == val), (
            f"Uniform-{val} not a fixed point; "
            f"changed at: {np.where(state != val)[0].tolist()}"
        )


def test_gkl_density_converges_above():
    """ρ≈0.8 (seed=0, N=149): reservoir with GKL wiring converges to all-1s within T=3N steps.

    n_nodes=0: forward(np.zeros((1, T))) runs T GKL ticks with no input.
    forward() resets to initial_states (the ρ=0.8 random state) at the start.
    """
    model = make_model()
    rng = np.random.default_rng(0)
    initial = (rng.random(model.R.n_nodes) < 0.8).astype(np.uint8)
    assert initial.sum() > model.R.n_nodes // 2, "Precondition failed: density not above 0.5"

    set_state(model, initial)
    model(np.zeros((1, T)))

    final = model.states_parallel[0, model.res_slice]
    assert np.all(final == 1), (
        f"Expected uniform-1 after {T} steps (ρ≈0.8); "
        f"{(final == 0).sum()} zeros remain."
    )


def test_gkl_density_converges_below():
    """ρ≈0.2 (seed=0, N=149): reservoir with GKL wiring converges to all-0s within T=3N steps.

    n_nodes=0: forward(np.zeros((1, T))) runs T GKL ticks with no input.
    forward() resets to initial_states (the ρ=0.2 random state) at the start.
    """
    model = make_model()
    rng = np.random.default_rng(0)
    initial = (rng.random(model.R.n_nodes) < 0.2).astype(np.uint8)
    assert initial.sum() < model.R.n_nodes // 2, "Precondition failed: density not below 0.5"

    set_state(model, initial)
    model(np.zeros((1, T)))

    final = model.states_parallel[0, model.res_slice]
    assert np.all(final == 0), (
        f"Expected uniform-0 after {T} steps (ρ≈0.2); "
        f"{(final == 1).sum()} ones remain."
    )


def test_gkl_reservoir_matches_reference():
    """
    BooleanReservoir wired as a GKL ring must reproduce the reference exactly.

    Covers: K=5 LUT endianness, the ring wiring (ca.ring),
    install_wiring bookkeeping, and Julia double-buffer synchronous update.
    """
    T_CHECK = 30
    model   = make_model()
    N_R     = model.R.n_nodes

    rng     = np.random.default_rng(42)
    initial = rng.integers(0, 2, N_R, dtype=np.uint8)

    set_state(model, initial)

    reservoir_history = [model.states_parallel[0, model.res_slice].copy()]
    for _ in range(T_CHECK):
        model._reservoir_tick(m=1)
        reservoir_history.append(model.states_parallel[0, model.res_slice].copy())

    actual   = np.stack(reservoir_history)       # (T_CHECK+1, N_R)
    expected = _gkl_reference(initial, T_CHECK)  # (T_CHECK+1, N_R)

    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected), (
        "GKL reservoir/reference mismatch at steps: "
        f"{np.where(~np.all(actual == expected, axis=1))[0].tolist()}"
    )
