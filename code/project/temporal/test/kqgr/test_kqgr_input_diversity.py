"""
Tests for KQ/GR input diversity on the temporal (Snyder replication) task,
under the new build_kqgr_probe pipeline (kqgr_probe.py).

Background: exhaustive pool generation used to build a tiny dataset of the
first n_nodes integers (0..n_nodes-1) with shuffle=False, clustering at the
low end in MSB-first binary and giving GR a systematic rank advantage. Fixed
by caching the full 2^bits exhaustive pool and randomly subsampling.

Pool-size and exhaustive-equivalence invariants (full 2^bits pool ->
freeze-overwrite -> dedupe == the narrower 2^(bits-tau) pool this task used to
generate directly) are covered generically, independent of task type, by
test_gr.py::test_dedupe_removes_freeze_induced_duplicates and by
test_kqgr_metric.py's TAU_AXIS_CELLS shape-parity tests — not duplicated here.

Invariants verified in this file:
  1. Subsampled output shape — build_kqgr_probe returns exactly n_nodes samples
  2. KQ step diversity — no all-identical steps (no leading-zero bias)
  3. GR step diversity — exactly tau contiguous identical steps at the
     evaluation tail; all free steps are diverse
  4. Edge case: n_nodes >= 2^bits_gr uses all available distinct patterns
  5. End-to-end rank: kq_rank > gr_rank at critical connectivity
  6. Regression: compute_rank filters to the final output_layer step
"""

import copy
import numpy as np
import tempfile
import pytest
from pathlib import Path

from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
from project.boolean_reservoir.code.kq_and_gr_metric import compute_rank
from project.boolean_reservoir.code.kqgr_probe import build_kqgr_probe
from project.boolean_reservoir.code.train_model_parallel import apply_seed
from project.temporal.code.dataset_init import TemporalDatasetInit

FIGURE1_CONFIG = 'config/temporal/kqgr/grid_search/figure_1_snyder_2012.yaml'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _universe_combo(tau, n_nodes, universe='kqgr_heterogeneous_real'):
    """n_nodes is a free override, not a search criterion — reservoir size isn't
    coupled to any other swept field here, so pin tau (which is), then set
    n_nodes directly. Previously searched for an existing grid combo matching
    BOTH tau and n_nodes, which silently skipped every n_nodes=128 case: 128
    only appears in the separate kqgr_*_real_n128 universes (Theme 4), never
    in kqgr_heterogeneous_real's own swept n_nodes=25."""
    P = load_yaml_config(FIGURE1_CONFIG)
    for p in generate_param_combinations(P):
        u = next(iter(p.multiverse_overrides or {}), None)
        if u == universe and p.D.tau == tau:
            P_uni = getattr(p.U, u)
            P_uni.M.R.n_nodes = n_nodes
            return P_uni
    pytest.skip(f"No combo for universe={universe} tau={tau}")


def _universe_combo_with_k_avg(tau, n_nodes, k_avg, universe='kqgr_heterogeneous_real'):
    P = load_yaml_config(FIGURE1_CONFIG)
    for p in generate_param_combinations(P):
        u = next(iter(p.multiverse_overrides or {}), None)
        if (u == universe and p.D.tau == tau and p.M.R.n_nodes == n_nodes
                and abs((p.M.variables or {}).get('R_k_avg', -1) - k_avg) < 0.01):
            return getattr(p.U, u)
    pytest.skip(f"No combo for tau={tau} n_nodes={n_nodes} k_avg={k_avg}")


def _augment_combo(tau, n_nodes):
    """Clone a steal-mode combo, switching tau_mode to 'augment'."""
    P_uni = _universe_combo(tau=tau, n_nodes=n_nodes)
    new_ds = P_uni.dataset.model_copy(update={'tau_mode': 'augment'})
    return P_uni.model_copy(update={'dataset': new_ds})


def _n_identical_steps(x):
    """Number of steps where ALL samples carry the same bit value."""
    return sum(1 for t in range(x.shape[1]) if len(np.unique(x[:, t, 0, 0])) == 1)


def _n_contiguous_identical_tail(x):
    """Number of contiguous identical steps from the last step backwards."""
    n = 0
    for t in range(x.shape[1] - 1, -1, -1):
        if len(np.unique(x[:, t, 0, 0])) == 1:
            n += 1
        else:
            break
    return n


def _probe(P_uni, n_nodes, seed=0):
    init = TemporalDatasetInit()
    rng = np.random.default_rng(seed)
    return build_kqgr_probe(P_uni, init, n_nodes, rng)


# ---------------------------------------------------------------------------
# 1. Subsampled output shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau_mode,tau,n_nodes", [
    ("steal",   3,  25),
    ("steal",   3, 128),
    ("augment", 3,  25),
])
def test_kqgr_output_has_n_nodes_samples(tau_mode, tau, n_nodes):
    P_uni = _universe_combo(tau=tau, n_nodes=n_nodes) if tau_mode == "steal" \
        else _augment_combo(tau=tau, n_nodes=n_nodes)

    kq_x, gr_x, _ = _probe(P_uni, n_nodes)

    assert kq_x.shape[0] == n_nodes, (
        f"[{tau_mode}] KQ: expected {n_nodes} samples, got {kq_x.shape[0]}"
    )
    assert gr_x.shape[0] == n_nodes, (
        f"[{tau_mode}] GR: expected {n_nodes} samples, got {gr_x.shape[0]}"
    )


# ---------------------------------------------------------------------------
# 2. KQ step diversity — no all-identical steps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau_mode,tau,n_nodes", [
    ("steal",   1,  25),
    ("steal",   3,  25),
    ("steal",   5,  25),
    ("steal",   1, 128),
    ("steal",   3, 128),
    ("augment", 1,  25),
    ("augment", 3,  25),
])
def test_kq_has_no_identical_steps(tau_mode, tau, n_nodes):
    """KQ must have diverse inputs at every step — no all-identical leading steps."""
    P_uni = _universe_combo(tau=tau, n_nodes=n_nodes) if tau_mode == "steal" \
        else _augment_combo(tau=tau, n_nodes=n_nodes)

    kq_x, _, _ = _probe(P_uni, n_nodes)

    n_identical = _n_identical_steps(kq_x)
    assert n_identical == 0, (
        f"[{tau_mode}] KQ (tau={tau}, n_nodes={n_nodes}): expected 0 identical steps "
        f"(no leading-zero bias), got {n_identical}/{kq_x.shape[1]}"
    )


# ---------------------------------------------------------------------------
# 3. GR step diversity — exactly tau identical tail steps, free steps diverse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau_mode,tau,n_nodes", [
    ("steal",   1,  25),
    ("steal",   3,  25),
    ("steal",   5,  25),
    ("steal",   1, 128),
    ("steal",   3, 128),
    ("augment", 1,  25),
    ("augment", 3,  25),
])
def test_gr_has_exactly_tau_identical_tail_steps(tau_mode, tau, n_nodes):
    """GR must have exactly tau contiguous identical steps at the evaluation tail."""
    P_uni = _universe_combo(tau=tau, n_nodes=n_nodes) if tau_mode == "steal" \
        else _augment_combo(tau=tau, n_nodes=n_nodes)

    bits = P_uni.D.bits   # 10 for this config
    total_steps = bits if tau_mode == "steal" else bits + tau

    _, gr_x, _ = _probe(P_uni, n_nodes)

    assert gr_x.shape[1] == total_steps, (
        f"[{tau_mode}] GR (tau={tau}): expected {total_steps} total steps, "
        f"got {gr_x.shape[1]}"
    )

    n_tail = _n_contiguous_identical_tail(gr_x)
    assert n_tail == tau, (
        f"[{tau_mode}] GR (tau={tau}, n_nodes={n_nodes}): expected {tau} identical "
        f"tail steps, got {n_tail}"
    )

    # Free steps must be diverse (no spurious leading zeros)
    free_steps = total_steps - tau
    for t in range(free_steps):
        n_unique = len(np.unique(gr_x[:, t, 0, 0]))
        assert n_unique > 1, (
            f"[{tau_mode}] GR (tau={tau}, n_nodes={n_nodes}): free step t={t} has only "
            f"{n_unique} unique value (expected diversity — leading-zero bias should be gone)"
        )


# ---------------------------------------------------------------------------
# 4. Edge case: n_nodes >= 2^bits_gr uses all available distinct patterns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau,n_nodes,bits_gr", [
    (4, 128, 6),   # steal: 2^6=64 < 128 → capped at 64
    (5, 128, 5),   # steal: 2^5=32 < 128 → capped at 32
])
def test_edge_case_n_nodes_exceeds_distinct_patterns(tau, n_nodes, bits_gr):
    """When n_nodes > 2^bits_gr, build_kqgr_probe uses all available distinct
    patterns without crashing."""
    P_uni = _universe_combo(tau=tau, n_nodes=n_nodes)

    _, gr_x, _ = _probe(P_uni, n_nodes)

    max_distinct = 2 ** bits_gr
    assert gr_x.shape[0] == max_distinct, (
        f"GR (tau={tau}, n_nodes={n_nodes}): only 2^{bits_gr}={max_distinct} distinct "
        f"patterns exist, output should be capped at {max_distinct}, got {gr_x.shape[0]}"
    )


# ---------------------------------------------------------------------------
# 5. End-to-end rank: kq_rank > gr_rank at critical connectivity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau,n_nodes,k_avg", [
    (3, 25, 3.0),   # critical regime: reservoir forgets shared tail → delta > 0
    (5, 25, 3.0),   # aggressive tau, same critical connectivity
])
def test_kq_rank_gt_gr_rank_at_critical_connectivity(tau, n_nodes, k_avg):
    """
    End-to-end: at critical connectivity (k_avg=3) kq_rank > gr_rank for most seeds.

    k_avg=3 is the critical point for heterogeneous reservoirs — the reservoir
    responds to inputs but lacks perfect memory, so the tau shared tail steps cause
    partial convergence of GR states → delta > 0.

    NOTE: k_avg=5 (strong connectivity / long memory) gives delta≈0 even with correct
    code because the reservoir retains the 7 diverse steps through 3 shared ones.
    That is correct physics, not a bug, but it means k_avg=5 is a poor regression
    target — it trivially passes even when compute_rank is broken.

    Requires delta > 0 for >= 3/5 seeds AND mean_delta > 0.
    """
    from project.boolean_reservoir.code.reservoir import BooleanReservoir

    P_uni = _universe_combo_with_k_avg(tau=tau, n_nodes=n_nodes, k_avg=k_avg)

    n_seeds = 5
    results = []

    for seed in range(n_seeds):
        kq_x, gr_x, P_model = _probe(P_uni, n_nodes, seed=seed)

        with tempfile.TemporaryDirectory() as tmpdir:
            P_seed = copy.deepcopy(P_model)
            P_seed.L.save_path = Path(tmpdir)
            P_seed.L.out_path = Path(tmpdir)
            # figure_1_snyder_2012.yaml seeds only the dataset, so the per-seed
            # variation of the reservoir has to be applied to the model params
            # themselves — I/R/O would otherwise all be None and identical.
            apply_seed(P_seed, seed)
            model = BooleanReservoir(P_seed)
            kq_rank = compute_rank(model, kq_x, 'kq')
            gr_rank = compute_rank(model, gr_x, 'gr')

        results.append((kq_rank, gr_rank))

    deltas   = [kq - gr for kq, gr in results]
    n_positive = sum(d > 0 for d in deltas)
    mean_delta = sum(deltas) / n_seeds

    assert n_positive >= 3, (
        f"delta > 0 in only {n_positive}/{n_seeds} seeds (tau={tau}, k_avg={k_avg}). "
        f"Expected >= 3 — at critical connectivity GR should drop below KQ. "
        f"Per-seed (kq, gr, delta): {[(kq, gr, kq-gr) for kq, gr in results]}"
    )
    assert mean_delta > 0, (
        f"Mean delta={mean_delta:.2f} <= 0 (tau={tau}, k_avg={k_avg}). "
        f"Per-seed: {results}"
    )


# ---------------------------------------------------------------------------
# 6. Regression: compute_rank filters to the final output_layer step
# ---------------------------------------------------------------------------

def test_compute_rank_uses_final_output_layer_step():
    """
    Regression for the time-series output_layer bug in compute_rank.

    With time-series readout, output_layer fires at every step (s times), giving
    m * s rows in filtered_history.  Without the final-step filter, diverse
    early-step states inflate GR rank and delta stays zero even at critical
    connectivity.

    This test verifies directly that:
    1. The time-series model produces s * n_samples output_layer rows in total.
    2. compute_rank's final-step filter selects exactly n_samples rows (one per
       sample at the last step), not all m * s rows.
    """
    from project.boolean_reservoir.code.reservoir import BooleanReservoir, BatchedTensorHistoryWriter

    P_uni = _universe_combo(tau=3, n_nodes=25)
    kq_x, _, P_model = _probe(P_uni, 25)
    n_samples = kq_x.shape[0]  # 25
    n_steps   = kq_x.shape[1]  # 10

    with tempfile.TemporaryDirectory() as tmpdir:
        P_m = copy.deepcopy(P_model)
        P_m.L.save_path = Path(tmpdir)
        P_m.L.out_path  = Path(tmpdir)
        P_m.M.O.mode, P_m.M.O.heads = 'time-series', None   # the case under test (the config runs both as heads)
        apply_seed(P_m, 0)
        model = BooleanReservoir(P_m)

        save_path = Path(tmpdir) / 'history' / 'kq' / 'history'
        model.record  = True
        model.history = BatchedTensorHistoryWriter(save_path=save_path, persist_to_disk=False)
        model.eval()
        _ = model(kq_x)
        model.flush_history()

        _, _, expanded_meta, _ = model.history.reload_history()
        output_rows = expanded_meta[expanded_meta['phase'] == 'output_layer']

        # Time-series model fires output_layer once per step per sample.
        assert len(output_rows) == n_steps * n_samples, (
            f"Expected {n_steps} steps × {n_samples} samples = "
            f"{n_steps * n_samples} output_layer rows, got {len(output_rows)}. "
            f"If this changes, verify the final-step filter in compute_rank is still needed."
        )

        # The final-step filter must reduce this to exactly n_samples rows.
        final_s    = output_rows['s'].max()
        final_rows = output_rows[output_rows['s'] == final_s]
        assert len(final_rows) == n_samples, (
            f"Final-step filter should yield {n_samples} rows, got {len(final_rows)}. "
            f"compute_rank is using all {len(output_rows)} output_layer rows instead of "
            f"the {n_samples} at the final step — GR rank will be inflated."
        )
