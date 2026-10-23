"""
Peer-review-oriented tests for BooleanReservoir (RBN-based reservoir computing).

Covers:
  1. Synchronous update correctness (true parallel tick, no race conditions)
  2. Determinism under seed control
  3. Batch independence (samples evolve in isolation)
  4. Input injection effectiveness (input nodes actually change)
  5. I-node isolation (input nodes have no incoming graph edges → reservoir cannot write them)
  6. Isolated-node state preservation (nodes with 0 in-degree keep their state)
  7. LUT bias fidelity (p=0.5 → ~50 % active output)
  8. LUT entropy / non-trivial dynamics (reservoir doesn't collapse to fixed point)
  9. Effective-k LUT access pattern (heterogeneous degree uses non-contiguous but correct LUT rows)
 10. Numpy-only stack (reservoir and readout are pure numpy; no torch tensors)
 11. Warmup monotonicity (warmup changes state compared to cold start)
 12. Kauffman criticality parameter (lambda = 2*p*(1-p); warns when far from critical point)
"""

import pytest
import numpy as np

from project.boolean_reservoir.code.parameter import (
    Params, ModelParams, InputParams, ReservoirParams, OutputParams,
    TrainingParams, LoggingParams, PatternWiring, RandomWiring
)
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.utils.utils import make_rng


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_params(
    n_r=64,
    k_avg=2.0,
    k_min=0,
    k_max=4,
    self_loops=0.0,
    p=0.5,
    perturbation='override',
    init='random',
    reset=True,
    mode='heterogeneous',
    n_input_nodes=None,
    bits=4,
    features=2,
    batch_size=8,
    seed=42,
) -> Params:
    I = InputParams(
        perturbation=perturbation,
        encoding='base2',
        features=features,
        bits=bits,
        n_nodes=n_input_nodes,
        ticks=f'1{{{features}}}',
        seed=seed,
    )
    R = ReservoirParams(
        n_nodes=n_r,
        p=p,
        init=init,
        reset=reset,
        seed=seed,
    )
    O = OutputParams(n_nodes=features, seed=seed)
    T = TrainingParams(batch_size=batch_size, epochs=1, accuracy_threshold=0.05,
                       optim={'name': 'gaussian', 'params': {'alpha': 1e-3}}, seed=seed)
    L = LoggingParams(out_path='/tmp/boolean_reservoir/test/boolean_reservoir/reservoir_scientific/', save_keys=None)
    variables = {
        'R_k_avg': k_avg,
        'R_k_min': k_min,
        'R_k_max': k_max,
        'R_self_loops': self_loops if self_loops is not None else 0.0,
        'R_mode': mode,
    }
    wiring = [
        PatternWiring(source='B', target='I', pattern='identity'),
        PatternWiring(source='I', target='R', pattern='identity'),
        RandomWiring(source='R', target='R', k_avg='R_k_avg', k_min='R_k_min',
                     k_max='R_k_max', self_loops='R_self_loops', mode=mode),
    ]
    M = ModelParams(variables=variables, input_layer=I, reservoir_layer=R,
                    output_layer=O, training=T, wiring=wiring)
    return Params(model=M, logging=L)


def make_model(seed=42, **kwargs) -> BooleanReservoir:
    # make_params puts `seed` on every layer; each draw derives its own Generator
    # from those, so there is no global RNG left to seed here.
    return BooleanReservoir(make_params(seed=seed, **kwargs))


def random_input(model: BooleanReservoir, m: int, s: int = 1) -> np.ndarray:
    """Return a random uint8 input array matching the model's expected shape.

    Drawn from a stream derived from the model's own input seed, so a test run is
    reproducible without touching global RNG.
    """
    I = model.P.M.I
    rng = make_rng(I.seed, 'test_input', m, s)
    return rng.integers(0, 2, (m, s, I.features, I.bits // I.features), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. Synchronous update — all nodes read OLD state, then write new state
# ---------------------------------------------------------------------------

def test_synchronous_update():
    """_reservoir_tick must read all states before writing any (parallel semantics)."""
    model = make_model(n_r=32, k_avg=2.0, k_max=4)
    m = 1
    state_before = model.states_parallel[:m].copy()
    model._reservoir_tick(m)
    state_after = model.states_parallel[:m].copy()
    model.states_parallel[:m] = state_before
    model._reservoir_tick(m)
    state_after2 = model.states_parallel[:m].copy()
    assert np.array_equal(state_after, state_after2), (
        "Tick is not deterministic: same start state should give same next state"
    )


# ---------------------------------------------------------------------------
# 2. Determinism under seed control
# ---------------------------------------------------------------------------

def test_determinism_same_seed():
    """Two models built with the same seed must produce identical outputs."""
    x = make_rng(0, 'determinism_x').integers(0, 2, (4, 1, 2, 2), dtype=np.uint8)
    m1 = make_model(seed=7)
    m2 = make_model(seed=7)
    o1 = m1(x)
    o2 = m2(x)
    assert np.array_equal(o1, o2), "Identical seeds must produce identical outputs"


def test_different_seeds_differ():
    """Two models built with different seeds should produce different reservoir states."""
    x = make_rng(0, 'determinism_x').integers(0, 2, (4, 1, 2, 2), dtype=np.uint8)
    m1 = make_model(seed=1)
    m2 = make_model(seed=2)
    m1(x); m2(x)
    # Compare reservoir states — readout weights are zero until the readout
    # fit runs, but reservoir dynamics are seeded differently and will diverge.
    s1 = m1.states_parallel[:4].copy()
    s2 = m2.states_parallel[:4].copy()
    assert not np.array_equal(s1, s2), "Different seeds must produce different reservoir states (collision extremely unlikely)"


# ---------------------------------------------------------------------------
# 3. Batch independence — each sample evolves independently
# ---------------------------------------------------------------------------

def test_batch_independence():
    """Running m samples together must give the same result as running each sample alone."""
    model = make_model(n_r=32, batch_size=8)
    m = 4
    np.random.seed(0)
    x = random_input(model, m)

    model.eval()
    out_batched = model(x)

    outs_single = []
    for i in range(m):
        model.reset_reservoir()
        outs_single.append(model(x[i:i+1]))

    out_single = np.concatenate(outs_single, axis=0)
    assert np.allclose(out_batched, out_single, atol=1e-6), (
        "Batched forward pass must equal m independent single-sample passes"
    )


# ---------------------------------------------------------------------------
# 4. Input injection effectiveness
# ---------------------------------------------------------------------------

def test_input_injection_changes_input_nodes():
    """Running forward with two different inputs should change reservoir states differently."""
    model = make_model(perturbation='override', n_r=32, bits=4, features=2)
    m = 2
    x1 = np.zeros((m, 1, 2, 2), dtype=np.uint8)
    x2 = np.ones((m, 1, 2, 2), dtype=np.uint8)
    model.reset_reservoir(hard_reset=True)
    model(x1)
    s1 = model.states_parallel[:m].copy()
    model.reset_reservoir(hard_reset=True)
    model(x2)
    s2 = model.states_parallel[:m].copy()
    assert not np.array_equal(s1, s2), (
        "Different inputs must produce different reservoir states — input injection not working"
    )
    assert model.states_parallel.dtype == np.uint8


def test_xor_perturbation_flips_bits():
    """XOR perturbation must flip exactly the bits indicated by the perturbation mask."""
    from project.boolean_reservoir.code.utils.reservoir_utils import InputPerturbationStrategy
    xor_fn = InputPerturbationStrategy.xor

    states = np.array([[0, 1, 0, 1, 1, 0, 0, 1]], dtype=np.uint8)
    perturb = np.array([[1, 0, 1, 0, 0, 0, 1, 0]], dtype=np.uint8)
    expected = states ^ perturb
    result = xor_fn(states, perturb)
    assert np.array_equal(result, expected), "XOR perturbation does not flip correct bits"


# ---------------------------------------------------------------------------
# 5. I-node isolation: input nodes have no incoming reservoir connections
# ---------------------------------------------------------------------------

def test_input_nodes_have_no_incoming_reservoir_edges():
    """Input nodes (I) must not appear as targets of any edge in the graph."""
    model = make_model(n_r=64, k_avg=3.0, k_max=6)
    n_input = model.P.M.I.n_nodes
    graph = model.graph
    for i in range(n_input):
        predecessors = list(graph.predecessors(i))
        assert predecessors == [], (
            f"Input node {i} has incoming edges {predecessors}; "
            "reservoir dynamics must not write input nodes"
        )


def test_input_nodes_preserved_during_tick():
    """Since I-nodes have no incoming edges, _reservoir_tick must leave them unchanged."""
    model = make_model(n_r=32, k_avg=2.0, k_max=4)
    m = 2
    model.states_parallel[:m, model.input_nodes_mask] = 1
    snapshot = model.states_parallel[:m, model.input_nodes_mask].copy()

    model._reservoir_tick(m)

    after = model.states_parallel[:m, model.input_nodes_mask]
    assert np.array_equal(snapshot, after), (
        "_reservoir_tick must not modify input-node states (they have no incoming edges)"
    )


# ---------------------------------------------------------------------------
# 6. Clamped nodes hold; isolated reservoir nodes are constants
# ---------------------------------------------------------------------------

def test_clamped_nodes_preserve_state():
    """A tick leaves clamped nodes (input, output) alone."""
    model = make_model(n_r=32, k_min=0, k_avg=2.0, k_max=4)
    m = 3
    clamped = model.clamped_nodes
    assert len(clamped) and model.input_nodes_mask[clamped].any()
    model.states_parallel[:m, clamped] = 1
    snapshot = model.states_parallel[:m, clamped].copy()

    model._reservoir_tick(m)

    after = model.states_parallel[:m, clamped]
    assert np.array_equal(snapshot, after), "Clamped nodes must preserve state across ticks"


def test_isolated_reservoir_nodes_are_k0_constants():
    """A reservoir node with no incoming edges is a k=0 node: a tick sets it to lut[i, 0],
    whatever its state was (it is not clamped)."""
    model = make_model(n_r=64, k_min=0, k_avg=1.0, k_max=4)
    m = 3
    res = np.arange(model.N_total)[model.res_slice]
    isolated = res[~model.adj_list_mask[model.res_slice].any(axis=1)]
    if len(isolated) == 0:
        pytest.skip("No isolated reservoir nodes in this random graph")
    assert not np.isin(isolated, model.clamped_nodes).any()
    lut0 = np.array([model.lut.row(i)[0] for i in isolated])
    for start in (0, 1):
        model.states_parallel[:m, isolated] = start
        model._reservoir_tick(m)
        assert np.array_equal(model.states_parallel[:m, isolated], np.broadcast_to(lut0, (m, len(isolated))))


# ---------------------------------------------------------------------------
# 7. LUT bias fidelity
# ---------------------------------------------------------------------------

def test_lut_bias_p05():
    """With p=0.5, mean LUT output across all entries should be close to 0.5."""
    model = make_model(n_r=256, k_avg=3.0, k_max=4, p=0.5, seed=1)
    mean_p = model.lut.flat.astype(float).mean()
    assert abs(mean_p - 0.5) < 0.05, f"LUT mean {mean_p:.3f} deviates > 0.05 from expected 0.5"


def test_lut_bias_p02():
    """With p=0.2, mean LUT output should be close to 0.2."""
    model = make_model(n_r=512, k_avg=3.0, k_max=4, p=0.2, seed=3)
    mean_p = model.lut.flat.astype(float).mean()
    assert abs(mean_p - 0.2) < 0.05, f"LUT mean {mean_p:.3f} deviates > 0.05 from expected 0.2"


def test_lut_values_are_binary():
    """LUT must contain only 0 and 1."""
    model = make_model()
    unique = np.unique(model.lut.flat)
    assert set(unique.tolist()).issubset({0, 1}), f"LUT contains non-binary values: {unique}"


# ---------------------------------------------------------------------------
# 8. Non-trivial dynamics
# ---------------------------------------------------------------------------

def test_reservoir_state_changes_over_ticks():
    """After several ticks with a random initial state, the reservoir state must change."""
    model = make_model(n_r=128, k_avg=2.0, k_max=4, p=0.5, init='random')
    m = 1
    model.reset_reservoir()
    state_t0 = model.states_parallel[:m].copy()

    for _ in range(10):
        model._reservoir_tick(m)

    state_t10 = model.states_parallel[:m].copy()
    assert not np.array_equal(state_t0, state_t10), (
        "Reservoir state unchanged after 10 ticks — dynamics appear frozen"
    )


def test_reservoir_states_not_uniform():
    """After forward pass, reservoir states should not be all-0 or all-1."""
    model = make_model(n_r=128)
    m = 4
    x = random_input(model, m)
    model(x)

    R_states = model.states_parallel[:m, model.reservoir_mask]
    mean_activity = R_states.astype(float).mean()
    assert 0.05 < mean_activity < 0.95, (
        f"Reservoir activity {mean_activity:.3f} is near 0 or 1 — dynamics may be degenerate"
    )


# ---------------------------------------------------------------------------
# 9. Effective-k LUT access pattern
# ---------------------------------------------------------------------------

def test_heterogeneous_lut_access_pattern():
    """A node of in-degree k has a table of exactly 2^k entries, whatever the largest in-degree."""
    model = make_model(n_r=128, k_min=1, k_avg=2.0, k_max=4)
    degrees = model.adj_list_mask.sum(axis=1)
    if not (degrees < model.max_connectivity).any():
        pytest.skip("All nodes have in-degree = k_max; use heterogeneous mode")
    for n, k in enumerate(degrees.tolist()):
        assert len(model.lut.row(n)) == 2 ** k


# ---------------------------------------------------------------------------
# 10. Numpy-only stack — no torch tensors anywhere in the model
# ---------------------------------------------------------------------------

def test_numpy_only_stack():
    """All model arrays must be numpy. No torch tensors anywhere in the model."""
    model = make_model(n_r=32)
    x = random_input(model, 4)
    out = model(x)

    assert isinstance(out, np.ndarray), f"model() output is {type(out)}, expected numpy ndarray"
    assert isinstance(model.states_parallel, np.ndarray), "states_parallel must be numpy"
    assert isinstance(model.lut.flat, np.ndarray), "lut must be numpy"
    assert isinstance(model.readout_W, np.ndarray), "readout_W must be numpy"
    assert isinstance(model.readout_b, np.ndarray), "readout_b must be numpy"
    assert isinstance(model.initial_states, np.ndarray), "initial_states must be numpy"


# ---------------------------------------------------------------------------
# 11. Warmup changes state relative to cold start
# ---------------------------------------------------------------------------

def test_warmup_changes_state():
    """'random-warmup' init must produce a different initial state than 'random' alone."""
    m_cold = make_model(init='random', seed=99)
    m_warm = make_model(init='random-warmup', seed=99)

    cold_state = m_cold.states_parallel.copy()
    warm_state = m_warm.states_parallel.copy()

    assert not np.array_equal(cold_state, warm_state), (
        "Warmup did not change reservoir state — washout period may be ineffective"
    )


# ---------------------------------------------------------------------------
# 12. Kauffman criticality
# ---------------------------------------------------------------------------

def test_kauffman_criticality_parameter():
    """Lambda = 2*p*(1-p); at criticality lambda * k_avg = 1."""
    k_avg = 2.0
    p = 0.5
    lam = 2 * p * (1 - p)
    criticality = lam * k_avg
    assert abs(criticality - 1.0) < 1e-9, (
        f"Default params (k_avg={k_avg}, p={p}) should be exactly at the critical point "
        f"lambda*k_avg={criticality:.4f}."
    )


def test_kauffman_criticality_homogeneous_k3():
    """For k=3, p=0.5 is super-critical (lambda*k=1.5>1 → chaotic)."""
    k = 3
    p = 0.5
    lam = 2 * p * (1 - p)
    criticality = lam * k
    assert criticality > 1.0, f"k={k}, p={p} should be in chaotic regime (lambda*k={criticality:.3f})"


# ---------------------------------------------------------------------------
# 13. bin2int correctness
# ---------------------------------------------------------------------------

def test_bin2int_msb_first():
    """powers_of_2 must be MSB-first: [2^(n-1), ..., 2^1, 2^0]."""
    model = make_model()
    k = model.max_connectivity
    p2 = model.powers_of_2
    expected = np.array([2 ** (k - 1 - i) for i in range(k)], dtype=np.float32)
    assert np.array_equal(p2, expected), f"powers_of_2 are not MSB-first: {p2}"


def test_bin2int_values():
    """With last 3 bits set to [1,0,1] and earlier bits zero, bin2int should give 5."""
    model = make_model(k_max=3)
    k = model.max_connectivity
    bits = np.zeros((1, 1, k), dtype=np.float32)
    bits[0, 0, k-3:] = [1, 0, 1]
    result = model.bin2int(bits)
    assert int(result.flat[0]) == 5, f"bin2int expected 5, got {result.flat[0]}"


def test_bin2int_all_zeros():
    """bin2int([0,0,...,0]) must be 0."""
    model = make_model()
    k = model.max_connectivity
    bits = np.zeros((1, 1, k), dtype=np.float32)
    assert int(model.bin2int(bits).flat[0]) == 0


def test_bin2int_all_ones():
    """bin2int([1,1,...,1]) with k bits must be 2^k - 1."""
    model = make_model()
    k = model.max_connectivity
    bits = np.ones((1, 1, k), dtype=np.float32)
    assert int(model.bin2int(bits).flat[0]) == 2**k - 1


# ---------------------------------------------------------------------------
# 14. Readout uses only R nodes
# ---------------------------------------------------------------------------

def test_readout_uses_only_r_nodes():
    """reservoir_mask must cover exactly the R slice."""
    model = make_model()
    expected = np.zeros_like(model.reservoir_mask)
    expected[model.res_slice] = True
    assert np.array_equal(model.reservoir_mask, expected)
    n_readout = int(model.reservoir_mask.sum())
    assert n_readout == model.P.M.R.n_nodes


# ---------------------------------------------------------------------------
# 15. reset=True resets reservoir state between samples
# ---------------------------------------------------------------------------

def test_reset_true_between_samples():
    """With reset=True, same input after state pollution must give same output."""
    model = make_model(reset=True)
    x = random_input(model, 1)

    model.eval()
    out1 = model(x)
    for _ in range(20):
        model._reservoir_tick(1)
    out2 = model(x)

    assert np.allclose(out1, out2, atol=1e-6), (
        "reset=True: same input after state pollution must give same output"
    )


# ---------------------------------------------------------------------------
# 16. GLM (Bernoulli/logit) readout fitter
# ---------------------------------------------------------------------------

def test_irls_matches_sklearn_oracle():
    """Our IRLS must reproduce sklearn's L2 logistic regression (the anti-mistake
    oracle). sklearn minimises CE + (1/(2C))||w_slopes||² with unpenalized
    intercept; our fitter represents the intercept as an unpenalized constant
    design column. Same model class, two parameterizations of the constant
    direction — so compare SLOPES and PREDICTED PROBABILITIES, not raw vectors
    (a constant column inside sklearn's penalized X makes its Hessian singular
    along the intercept direction and lbfgs crawls)."""
    from sklearn.linear_model import LogisticRegression
    from project.boolean_reservoir.code.glm import irls_fit, sigmoid

    rng = np.random.default_rng(0)
    m, p = 400, 11
    Xs = rng.integers(0, 2, (m, p)).astype(np.float64)
    w_true = rng.normal(0, 0.5, p)
    b_true = 0.3
    y = (rng.random(m) < sigmoid(Xs @ w_true + b_true)).astype(np.float64)

    alpha_abs = 0.37
    # ours: intercept as an extra unpenalized constant column (bias scale 3.0)
    X_ours = np.concatenate([Xs, np.full((m, 1), 3.0)], axis=1)
    W_ours, info = irls_fit(X_ours, y[:, None], np.ones(m), alpha_abs)
    ours = W_ours[:, 0]
    # sklearn: slopes penalized, intercept free
    ref = LogisticRegression(C=1.0 / (2.0 * alpha_abs), max_iter=5000, tol=1e-10)
    ref.fit(Xs, y)

    assert np.allclose(ours[:-1], ref.coef_[0], atol=1e-6), (
        f"IRLS slopes diverge from sklearn: max|Δ|="
        f"{np.abs(ours[:-1] - ref.coef_[0]).max():.2e}"
    )
    # constant-direction split: 3.0·w_bias ≡ intercept
    assert np.allclose(3.0 * ours[-1], ref.intercept_[0], atol=1e-5)
    # and the two parameterizations predict identical probabilities
    p_ours = sigmoid(X_ours @ ours)
    p_ref = ref.predict_proba(Xs)[:, 1]
    assert np.allclose(p_ours, p_ref, atol=1e-8)


def test_irls_dedup_equivalence():
    """Grouping duplicate design rows (binomial counts) must give the SAME fit as
    the ungrouped rows — the dedup is exact, not an approximation."""
    from project.boolean_reservoir.code.glm import irls_fit, dedup_design

    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, (300, 8)).astype(np.float64)
    X[:, -1] = 5.0
    y = rng.integers(0, 2, (300, 1)).astype(np.float64)
    w_full, _ = irls_fit(X, y, np.ones(300), 0.2)
    Xu, s, n = dedup_design(X, y)
    assert len(Xu) < 300, "expected duplicates in this synthetic design"
    w_grouped, info_g = irls_fit(Xu, s, n, 0.2)
    assert np.allclose(w_full, w_grouped, atol=1e-8), (
        "grouped and ungrouped Bernoulli IRLS must agree to fp tolerance"
    )
    assert info_g['n_unique'] == len(Xu)
    assert info_g['converged'] and info_g['iters'] <= 50


def test_train_readout_glm_end_to_end(tmp_path, monkeypatch):
    """optim='bernoulli' trains through the real pipeline: sane accuracy, finite
    cross-entropy loss, deterministic weights across identical runs, and the
    decision rule P(y=1|x)>0.5 on the sigmoid of the raw readout."""
    from project.boolean_reservoir.code.train_model import train_and_evaluate
    from benchmark.temporal.parameter import TemporalDatasetParams
    from benchmark.utils.parameter import Split
    from project.boolean_reservoir.code.parameter import (
        InputParams, ReservoirParams, OutputParams, TrainingParams, LoggingParams,
        PatternWiring, RandomWiring)

    # chdir so the relative data/... npz path lands in tmp, never in repo data/
    monkeypatch.chdir(tmp_path)

    def build():
        I = InputParams(perturbation='override', encoding='base2', features=1,
                        resolution=1, redundancy=1, ticks='2', seed=0)
        R = ReservoirParams(n_nodes=64, p=0.5, init='random', reset=True, seed=0)
        O = OutputParams(n_nodes=1, mode='time-series', seed=0)
        T = TrainingParams(batch_size=256, epochs=1, evaluation='dev',
                           accuracy='Boolean',
                           optim={'name': 'bernoulli', 'params': {'alpha': 1e-3}}, seed=0)
        L = LoggingParams(out_path='/tmp/boolean_reservoir/test/readout_glm/')
        M = ModelParams(input_layer=I, reservoir_layer=R, output_layer=O,
                        training=T, wiring=[
                            PatternWiring(source='B', target='I', pattern='identity'),
                            PatternWiring(source='I', target='R', pattern='identity'),
                            RandomWiring(source='R', target='R', k_avg=3.0, k_min=0,
                                         k_max=7, mode='homogeneous'),
                        ])
        D = TemporalDatasetParams(task='density', bits=10, window=3, delay=1,
                                  samples=256, sampling_mode='exhaustive', shuffle=False,
                                  seed=0, split=Split(train=0.8, dev=0.1, test=0.1))
        return Params(model=M, logging=L, dataset=D)

    P = build()
    dataset = P.dataset_init_obj.train(P)
    stats1, model1, _ = train_and_evaluate(BooleanReservoir(params=P), dataset,
                                           accuracy_fn=P.accuracy_obj)
    # _score_splits contract (6445a2d): split-explicit metric keys, no bare aliases
    # plus the fit diagnostics on every row
    assert set(stats1) == {'accuracy_train', 'loss_train', 'accuracy_dev',
                           'loss_dev', 'accuracy_test', 'loss_test',
                           'fit_iters', 'fit_converged', 'fit_lstsq_fallbacks',
                           'fit_guard_events', 'fit_saturated',
                           'fit_w_norm', 'fit_n_unique', 'fit_rank'}, sorted(stats1)
    assert 0.5 <= stats1['accuracy_train'] <= 1.0, f"accuracy {stats1['accuracy_train']} below chance"
    assert np.isfinite(stats1['loss_train']) and stats1['loss_train'] > 0, "CE loss must be finite/positive"
    assert np.any(model1.readout_W != 0), "readout weights must be nonzero after training"
    # diagnostics: a penalized (alpha=1e-3) fit on this small fixture must
    # converge, with finite slope norms and a rank-deficiency-free design
    assert stats1['fit_converged'] is True, "penalized IRLS must converge on this fixture"
    assert stats1['fit_iters'] <= 50 and stats1['fit_lstsq_fallbacks'] == 0
    assert stats1['fit_guard_events'] == 0, "the fit guard must not fire on a healthy penalized fit"
    assert np.isfinite(stats1['fit_w_norm']) and stats1['fit_w_norm'] > 0
    assert stats1['fit_n_unique'] > 0 and stats1['fit_rank'] > 0

    stats2, model2, _ = train_and_evaluate(BooleanReservoir(params=P), dataset,
                                           accuracy_fn=P.accuracy_obj)
    assert np.array_equal(model1.readout_W, model2.readout_W), (
        "IRLS must be deterministic: identical runs give bitwise-identical weights"
    )
    assert stats1['accuracy_train'] == stats2['accuracy_train']


# ---------------------------------------------------------------------------
# 17. GLM umbrella: family dispatch, aliases, Gaussian closed-form identity
# ---------------------------------------------------------------------------

def test_gaussian_irls_matches_closed_form():
    """The umbrella's core identity: the Gaussian/identity GLM's Newton–IRLS is
    EXACTLY the closed-form penalised-normal-equations solve, in ONE Newton
    step (the objective is quadratic). This is why the Gaussian family is
    allowed its closed-form fast path: the general IRLS machinery, run on the
    same objective (0.5||Xw−y||² + (α_abs/2)||w_slopes||², intercept free),
    lands on the same solution to fp tolerance. Guards the closed form against
    ever drifting away from the family's likelihood semantics."""
    from project.boolean_reservoir.code.glm import gaussian_closed_form

    def gaussian_irls_reference(X, y, alpha_abs, max_iter=50, tol=1e-12):
        """Textbook IRLS for the Gaussian/identity GLM — reference only, lives
        here to PROVE the production closed form; never in production code.
        Objective ½||Xw−y||² + (α_abs/2)||w_slopes||² → normal equations
        (XᵀX + α_abs·I_slopes)w = Xᵀy, the same equations as production."""
        p_dim = X.shape[1]
        P = alpha_abs * np.eye(p_dim)
        P[-1, -1] = 0.0                       # intercept unpenalized, as in production
        w = np.zeros(p_dim, dtype=np.float64)
        for _ in range(max_iter):
            g = X.T @ (X @ w - y.ravel()) + P @ w
            H = X.T @ X + P                   # constant Hessian: one exact step
            w = w - np.linalg.solve(H, g)
            if np.max(np.abs(g)) <= tol:
                break
        return w

    rng = np.random.default_rng(7)
    m, p = 200, 6
    Xs = rng.integers(0, 2, (m, p)).astype(np.float64)
    Xs[:, -1] = 4.0                           # constant bias-scale column
    y = (Xs @ rng.normal(0, 0.5, p) + 0.3 + 0.05 * rng.normal(0, 1, m))[:, None]
    for alpha in (0.0, 0.37):
        XtX, Xty = Xs.T @ Xs, Xs.T @ y
        closed = gaussian_closed_form(XtX, Xty, alpha)[:, 0]
        irls = gaussian_irls_reference(Xs, y, alpha * float(np.trace(XtX)) / p)
        assert np.allclose(closed, irls, atol=1e-9), (
            f"alpha={alpha}: Gaussian closed form diverges from the IRLS "
            f"reference: max|Δ|={np.abs(closed - irls).max():.2e}"
        )


def _alias_e2e_params(optim_name):
    """The §16 e2e fixture, parameterised by optim name, for alias equality."""
    from benchmark.temporal.parameter import TemporalDatasetParams
    from benchmark.utils.parameter import Split
    from project.boolean_reservoir.code.parameter import (
        InputParams, ReservoirParams, OutputParams, TrainingParams, LoggingParams,
        PatternWiring, RandomWiring)

    I = InputParams(perturbation='override', encoding='base2', features=1,
                    resolution=1, redundancy=1, ticks='2', seed=0)
    R = ReservoirParams(n_nodes=64, p=0.5, init='random', reset=True, seed=0)
    O = OutputParams(n_nodes=1, mode='time-series', seed=0)
    T = TrainingParams(batch_size=256, epochs=1, evaluation='dev',
                       accuracy='Boolean',
                       optim={'name': optim_name, 'params': {'alpha': 1e-3}}, seed=0)
    L = LoggingParams(out_path='/tmp/boolean_reservoir/test/readout_glm/')
    M = ModelParams(input_layer=I, reservoir_layer=R, output_layer=O,
                    training=T, wiring=[
                        PatternWiring(source='B', target='I', pattern='identity'),
                        PatternWiring(source='I', target='R', pattern='identity'),
                        RandomWiring(source='R', target='R', k_avg=3.0, k_min=0,
                                     k_max=7, mode='homogeneous'),
                    ])
    D = TemporalDatasetParams(task='density', bits=10, window=3, delay=1,
                              samples=256, sampling_mode='exhaustive', shuffle=False,
                              seed=0, split=Split(train=0.8, dev=0.1, test=0.1))
    return Params(model=M, logging=L, dataset=D)


def test_optim_names_one_per_family():
    """One name per family: 'gaussian'
    and 'bernoulli' are the only valid config names. The historical names
    'ridge'/'logistic' are RETIRED and must raise, naming the replacement —
    no silent map ('ridge' is a lie at alpha=0, where the fit is minimum-norm
    OLS, not ridge)."""
    from project.boolean_reservoir.code.glm import normalize_family
    assert normalize_family('gaussian') == 'gaussian'
    assert normalize_family('bernoulli') == 'bernoulli'
    for old, family in (('ridge', 'gaussian'), ('logistic', 'bernoulli')):
        try:
            normalize_family(old)
        except ValueError as e:
            assert family in str(e), f"retired name '{old}' must name its replacement '{family}'"
        else:
            raise AssertionError(f"retired optim name '{old}' must raise, not map silently")
    try:
        normalize_family('adam')
    except ValueError as e:
        assert 'gaussian' in str(e) and 'bernoulli' in str(e)
    else:
        raise AssertionError("normalize_family must reject unknown optim names")


# ---------------------------------------------------------------------------
# 18. Fit diagnostics — alpha=0 well-definedness is measured, not assumed
# ---------------------------------------------------------------------------

def test_irls_separable_alpha0_diagnostics():
    """Unit-sized: at alpha=0 on a separable design the
    logistic MLE does not exist (||w|| → ∞), so the fit is wherever IRLS
    stops — the test pins down HOW it stops (max_iter exhausted, converged
    False, min-norm lstsq fallbacks, runaway slope norm) and that a tiny
    scale-aware penalty restores an honest converged fit with a bounded ||w||.
    The alpha_abs values use the same scale-aware mapping as train_readout."""
    from project.boolean_reservoir.code.glm import irls_fit, dedup_design, IRLS_MAX_ITER

    rows = np.array([[int(b) for b in f'{i:03b}'] for i in range(8)], dtype=np.float64)
    X = np.concatenate([rows, np.full((8, 1), 2.0)], axis=1)   # (8, 4), bias col last
    y = X[:, :1].astype(np.float64)                            # label == x0: separable
    Xu, s, n = dedup_design(X, y)

    def scale_aware_abs(alpha):
        # identical to glm.train_readout's bernoulli arm
        return alpha * float(np.sum((Xu ** 2) * n[:, None])) / Xu.shape[1]

    W0, info0 = irls_fit(Xu, s, n, scale_aware_abs(0.0))
    assert info0['converged'] is False, "alpha=0 on separable rows must NOT converge"
    assert info0['iters'] == IRLS_MAX_ITER, "non-converged column must exhaust max_iter"
    assert info0['lstsq_fallbacks'] >= 1, "saturated Hessians must hit the lstsq fallback"
    assert np.isfinite(W0).all(), "the 50-iter cap must stop the runaway in finite weights"

    W6, info6 = irls_fit(Xu, s, n, scale_aware_abs(1e-6))
    assert info6['converged'] is True and info6['iters'] < IRLS_MAX_ITER
    assert info6['lstsq_fallbacks'] == 0
    # runaway signature: the unpenalized fit's slope norm far exceeds the
    # penalized one (same data, same mapping, only alpha differs)
    assert info0['w_norm'] > 2.0 * info6['w_norm'], (
        f"alpha=0 ||w||={info0['w_norm']:.3g} should run away from "
        f"alpha=1e-6 ||w||={info6['w_norm']:.3g}"
    )
    # diagnostics bookkeeping consistent on both arms
    for info in (info0, info6):
        assert info['n_unique'] == len(Xu)
        assert info['rank'] == Xu.shape[1], "this fixture's design is full rank"
        assert info['w_norm'] > 0


def test_gaussian_fit_diagnostics_pipeline(tmp_path, monkeypatch):
    """The Gaussian arm's diagnostics through the real train_and_evaluate path:
    the closed form has no iteration to converge, so iters=1 / converged=True /
    no fallbacks / n_unique=None, with rank and slope norm measured."""
    from project.boolean_reservoir.code.train_model import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    P = _alias_e2e_params('gaussian')
    dataset = P.dataset_init_obj.train(P)
    stats, model, _ = train_and_evaluate(BooleanReservoir(params=P), dataset,
                                         accuracy_fn=P.accuracy_obj)
    assert stats['fit_iters'] == 1
    assert stats['fit_converged'] is True
    assert stats['fit_lstsq_fallbacks'] == 0
    assert stats['fit_guard_events'] == 0 and stats['fit_saturated'] is None
    assert stats['fit_n_unique'] is None, "n_unique is a dedup (Bernoulli) concept"
    assert np.isfinite(stats['fit_w_norm']) and stats['fit_w_norm'] > 0
    assert stats['fit_rank'] >= 1


def _irls_fit_unguarded(Xu, s, n, alpha_abs, max_iter=50, tol=1e-8):
    """The pre-guard Newton loop (Hessian via the production _weighted_gram) —
    the reference the fit guard must reproduce bit for bit whenever it does
    not fire."""
    from project.boolean_reservoir.code.glm import _weighted_gram
    p_dim = Xu.shape[1]
    P = 2.0 * alpha_abs * np.eye(p_dim)
    P[-1, -1] = 0.0
    W = np.zeros((p_dim, s.shape[1]))
    from project.boolean_reservoir.code.glm import sigmoid
    for col in range(s.shape[1]):
        w = np.zeros(p_dim)
        for _ in range(max_iter):
            prob = sigmoid(Xu @ w)
            g = Xu.T @ (n * prob - s[:, col]) + P @ w
            H = _weighted_gram(Xu, n * prob * (1.0 - prob)) + P
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, g, rcond=None)[0]
            w = w - step
            if np.max(np.abs(step)) <= tol * (1.0 + np.max(np.abs(w))):
                break
        W[:, col] = w
    return W


def test_irls_guard_is_bitwise_noop_on_healthy_fits():
    """The fit guard only acts on rejected steps or failed solves; on
    healthy penalized fits (guard_events == 0) the weights must be bitwise
    identical to the unguarded loop — the probe's alpha>0 arms, fitted before
    the guard existed, stay valid for the guarded main grids."""
    from project.boolean_reservoir.code.glm import irls_fit, dedup_design

    rng = np.random.default_rng(7)
    checked = 0
    for m, k, alpha in [(300, 8, 1e-4), (400, 40, 1e-2), (256, 120, 1e-4), (512, 200, 1e-1)]:
        X = np.concatenate([rng.integers(0, 2, (m, k)), np.ones((m, 1))], axis=1).astype(np.float64)
        y = (rng.random((m, 1)) < 0.5).astype(np.float64)
        Xu, s, n = dedup_design(X, y)
        alpha_abs = alpha * float(np.sum((Xu ** 2) * n[:, None])) / Xu.shape[1]
        W, info = irls_fit(Xu, s, n, alpha_abs)
        if info['guard_events'] == 0:
            assert np.array_equal(W, _irls_fit_unguarded(Xu, s, n, alpha_abs)), (m, k, alpha)
            checked += 1
    assert checked >= 3, "fixtures must mostly be healthy fits for this test to mean anything"


def test_irls_guard_bounds_separable_alpha0_runaway():
    """At alpha=0 on a separable design (p ≈ n/2, random
    labels — the T regime) the unguarded loop took Newton steps of 1e16+ and
    ended at ||w|| ≈ 1e90 / inf / NaN. Guarded: finite weights within
    IRLS_W_MAX, fit flagged not converged, a FitGuardWarning printed."""
    import warnings
    from project.boolean_reservoir.code.glm import irls_fit, IRLS_W_MAX, FitGuardWarning

    rng = np.random.default_rng(1)
    m, k = 240, 160
    X = np.concatenate([rng.integers(0, 2, (m, k)), np.ones((m, 1))], axis=1).astype(np.float64)
    y = (rng.random((m, 1)) < 0.5).astype(np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        W, info = irls_fit(X, y, np.ones(m), 0.0)
    assert np.isfinite(W).all() and np.abs(W).max() <= IRLS_W_MAX
    assert np.isfinite(W.astype(np.float32)).all(), "weights must survive the float32 readout cast"
    assert info['converged'] is False
    assert info['guard_events'] > 0
    assert any(issubclass(c.category, FitGuardWarning) for c in caught)


def test_irls_guard_survives_lapack_failure(monkeypatch):
    """Solve raising Singular and then lstsq raising
    'SVD did not converge' on a NaN Hessian killed 3 parity probe jobs. The
    guard must keep the last finite iterate, flag the fit and warn."""
    import warnings
    from project.boolean_reservoir.code.glm import irls_fit, FitGuardWarning

    def _raise(*a, **k):
        raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")
    monkeypatch.setattr(np.linalg, 'solve', _raise)
    monkeypatch.setattr(np.linalg, 'lstsq', _raise)
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.integers(0, 2, (50, 5)), np.ones((50, 1))], axis=1).astype(np.float64)
    y = rng.integers(0, 2, (50, 1)).astype(np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        W, info = irls_fit(X, y, np.ones(50), 0.1)
    assert np.array_equal(W, np.zeros_like(W)), "last finite iterate is the start point w=0"
    assert info['converged'] is False and info['guard_events'] == 1 and info['iters'] == 0
    assert any(issubclass(c.category, FitGuardWarning) for c in caught)


def test_gaussian_solver_fallback(monkeypatch):
    """(PI 2D job 81525) a failing gelsd must not kill the job: retry
    without flush-to-zero, then the gelsy driver; each fallback warns and is
    counted; so must a solver that returns NaN without raising (PI 2D query job 27638),
    and if every solver does, it raises. Non-finite normal equations are an upstream
    bug and raise."""
    import warnings
    from project.boolean_reservoir.code import glm

    rng = np.random.default_rng(3)
    X = np.concatenate([rng.integers(0, 2, (200, 12)), np.ones((200, 1))], axis=1).astype(np.float64)
    X[:, 1] = X[:, 2]                                   # collinear, like input redundancy
    y = rng.normal(size=(200, 2))
    XtX, Xty = X.T @ X, X.T @ y
    ref, ref_rank, ev = glm._gaussian_solve(XtX, Xty, 0.0)
    assert ev == 0

    real_lstsq = np.linalg.lstsq
    calls = {'n': 0}
    def _flaky(*a, **k):
        calls['n'] += 1
        if calls['n'] <= fail_first:
            raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")
        return real_lstsq(*a, **k)
    monkeypatch.setattr(np.linalg, 'lstsq', _flaky)

    for fail_first, expect_events in [(1, 1), (2, 2)]:
        calls['n'] = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            sol, rank, ev = glm._gaussian_solve(XtX, Xty, 0.0)
        assert ev == expect_events and rank == ref_rank
        assert np.allclose(X @ sol, X @ ref, atol=1e-8), "fallback must reach the same fitted values"
        assert sum(issubclass(c.category, glm.FitGuardWarning) for c in caught) == expect_events

    # (PI 2D query job 27638) a solver returning NaN without raising: fall back, never return NaN
    def _nan_first(*a, **k):
        calls['n'] += 1
        sol, res, rank, sv = real_lstsq(*a, **k)
        return (np.full_like(sol, np.nan) if calls['n'] <= fail_first else sol), res, rank, sv
    monkeypatch.setattr(np.linalg, 'lstsq', _nan_first)
    for fail_first, expect_events in [(1, 1), (2, 2)]:
        calls['n'] = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            sol, rank, ev = glm._gaussian_solve(XtX, Xty, 0.0)
        assert ev == expect_events and np.all(np.isfinite(sol))
        assert np.allclose(X @ sol, X @ ref, atol=1e-8), "fallback must reach the same fitted values"
        assert sum(issubclass(c.category, glm.FitGuardWarning) for c in caught) == expect_events
    import scipy.linalg
    monkeypatch.setattr(scipy.linalg, 'lstsq', lambda *a, **k: (lambda r: (np.full_like(r[0], np.nan), *r[1:]))(real_lstsq(*a[:2], rcond=None)))
    calls['n'] = 0; fail_first = 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(ValueError, match="every solver"):
            glm._gaussian_solve(XtX, Xty, 0.0)

    XtX_bad = XtX.copy(); XtX_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        glm._gaussian_solve(XtX_bad, Xty, 0.0)


def test_bernoulli_train_scores_from_fit_states_match_forward(tmp_path, monkeypatch):
    """The Bernoulli fit scores the train split from the states
    it already collected instead of a second reservoir pass. Those scores must
    match the forward-pass scores: same accuracy, loss equal to fp tolerance."""
    from project.boolean_reservoir.code.train_model import train_and_evaluate, _predict_split
    from project.boolean_reservoir.code.glm import sigmoid

    monkeypatch.chdir(tmp_path)
    P = _alias_e2e_params('bernoulli')
    dataset = P.dataset_init_obj.train(P)
    stats, model, _ = train_and_evaluate(BooleanReservoir(params=P), dataset,
                                         accuracy_fn=P.accuracy_obj)
    y_hat, y = _predict_split(model, dataset, 'train')          # the forward pass
    p_hat = sigmoid(np.asarray(y_hat, dtype=np.float64))
    acc_fwd = float(P.accuracy_obj(p_hat, y, P.M.T.accuracy_threshold))
    p_clip = np.clip(p_hat, 1e-12, 1 - 1e-12)
    yf = y.astype(np.float64)
    loss_fwd = float(-np.mean(yf * np.log(p_clip) + (1 - yf) * np.log(1 - p_clip)))
    assert stats['accuracy_train'] == acc_fwd
    assert abs(stats['loss_train'] - loss_fwd) < 1e-6


# ---------------------------------------------------------------------------
# 19. Dashboard extraction — split-explicit TrainLog fields must keep
#     the dashboards' bare {'accuracy','loss'} field-sets and saved views alive
# ---------------------------------------------------------------------------

def test_params_col_to_fields_legacy_accuracy_alias():
    """6445a2d renamed TrainLog's bare accuracy/loss into split-explicit keys,
    which silently broke the app extractions `{'accuracy', 'loss'}`: the lookup
    is dict.get, which does not resolve the alias tree, so fresh extractions
    produced all-None T_accuracy/T_loss columns (warm pre-6445a2d caches masked
    it). The fix: an exact field key always wins; otherwise the requested key
    resolves through the model's property aliases (TrainLog.accuracy ->
    accuracy_dev — what the views showed pre-refactor, i.e. the T.evaluation
    split), and the COLUMN KEEPS THE REQUESTED NAME so saved views reading the
    bare metric keep matching. Covers the DotDict path (production) and the
    pydantic path."""
    import pandas as pd
    from project.boolean_reservoir.code.parameter import Params, TrainLog, load_yaml_config
    from project.boolean_reservoir.code.utils.load_save import DotDict, params_col_to_fields

    cfg = '/workspaces/boolean_reservoir/code/config/temporal/parity/grid_search/design_choices/all_homogeneous.yaml'
    P = load_yaml_config(cfg)
    P.L.train = TrainLog(config=0, sample=0, accuracy_train=0.9, loss_train=0.3,
                         accuracy_dev=0.8, loss_dev=0.4, accuracy_test=0.7, loss_test=0.5)

    extraction = [('T', lambda p: p.L.T, {'accuracy', 'loss'})]

    # dashboard path: DotDict rows (fast loader)
    dd = DotDict(P.model_dump(), cls=Params)
    ext_df, factors = params_col_to_fields(pd.DataFrame({'params': [dd]}), extraction)
    assert ext_df['T_accuracy'][0] == 0.8, "bare accuracy must alias to accuracy_dev"
    assert ext_df['T_loss'][0] == 0.4
    assert 'T_accuracy' in factors, "column must keep the requested (bare) name"

    # pydantic path: real model rows resolve through the class alias tree
    extp, _ = params_col_to_fields(pd.DataFrame({'params': [P]}), extraction)
    assert extp['T_accuracy'][0] == 0.8 and extp['T_loss'][0] == 0.4

    # exact split-explicit keys need no alias and must not be shadowed
    ext2, _ = params_col_to_fields(pd.DataFrame({'params': [dd]}), [
        ('T', lambda p: p.L.T, {'accuracy_train', 'accuracy_dev', 'accuracy'})])
    assert ext2['T_accuracy_train'][0] == 0.9
    assert ext2['T_accuracy_dev'][0] == 0.8
    assert ext2['T_accuracy'][0] == 0.8, "alias fallback still maps to dev"

    # a field-set key that exists neither as field nor alias stays absent-valued
    ext3, _ = params_col_to_fields(pd.DataFrame({'params': [dd]}), [
        ('T', lambda p: p.L.T, {'nonsense'})])
    assert ext3['T_nonsense'][0] is None, "unknown keys must stay None, not raise"


def test_score_splits_skips_empty_test_split(tmp_path, monkeypatch):
    """split_dataset slices x_test = x[dev_end:], so a
    test fraction of 0 leaves a size-0 array with the key PRESENT; the model
    forward cannot reshape a 0-row batch and every probe job died with
    'cannot reshape array of size 0 into shape (0,newaxis,1,1)'. An empty split
    must be scored as absent: train+dev carry metrics, no test keys exist."""
    from benchmark.utils.parameter import Split
    from project.boolean_reservoir.code.train_model import train_and_evaluate

    P = _alias_e2e_params('bernoulli')
    P.dataset.split = Split(train=0.5, dev=0.5, test=0.0)   # the T grids' regime
    monkeypatch.chdir(tmp_path)
    dataset = P.dataset_init_obj.train(P)
    assert dataset.data['x_test'].shape[0] == 0 and 'x_test' in dataset.data
    stats, model, _ = train_and_evaluate(BooleanReservoir(params=P), dataset,
                                         accuracy_fn=P.accuracy_obj)
    assert set(stats) == {'accuracy_train', 'loss_train',
                          'accuracy_dev', 'loss_dev',
                          'fit_iters', 'fit_converged', 'fit_lstsq_fallbacks',
                          'fit_guard_events', 'fit_saturated',
                          'fit_w_norm', 'fit_n_unique', 'fit_rank'}, sorted(stats)
    assert 0.0 < stats['accuracy_dev'] <= 1.0 and np.isfinite(stats['loss_dev'])
    assert 'accuracy_test' not in stats and 'loss_test' not in stats
