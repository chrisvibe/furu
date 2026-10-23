"""
Tests targeting the figure_1_snyder_2012.yaml grid search to detect bugs that could
cause homogeneous and heterogeneous reservoir modes to produce identical results.

Covers:
  1. Config expansion: R_mode is correctly set per universe (homogeneous vs heterogeneous)
  2. Universe tagging: expanded configs carry the correct multiverse_overrides key
  3. Direct mode wiring: homogeneous → uniform in-degree; heterogeneous → variable in-degree
  4. Variable-based mode resolution: mode="R_mode" wiring reference resolves from variables correctly
  5. GR divergence: homogeneous and heterogeneous models produce different GR ranks at k_avg=5
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from project.boolean_reservoir.code.parameter import (
    Params, ModelParams, InputParams, ReservoirParams, OutputParams,
    TrainingParams, LoggingParams, PatternWiring, RandomWiring,
    load_yaml_config,
)
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
from project.boolean_reservoir.code.utils.utils import make_rng
from project.boolean_reservoir.code.kq_and_gr_metric import compute_rank


YAML_PATH = 'config/temporal/kqgr/grid_search/figure_1_snyder_2012.yaml'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rr_in_degrees(model: BooleanReservoir) -> np.ndarray:
    """Return the in-degree of each R node counting only R→R edges."""
    import networkx as nx
    N_I = model.I.n_nodes
    N_R = model.R.n_nodes
    r_nodes = range(N_I, N_I + N_R)
    sub = model.graph.subgraph(r_nodes)
    return np.array([d for _, d in sub.in_degree()])


def _make_params_with_mode_variable(n_r, k_avg, mode, seed=42, out_path='/tmp/boolean_reservoir/test/temporal/kqgr/snyder_2012/') -> Params:
    """
    Build Params where the R→R wiring references R_mode as a variable string
    (same pattern as the YAML config), so we exercise the variable-resolution code path.
    """
    I = InputParams(perturbation='override', encoding='base2',
                    features=1, bits=10, ticks='1{10}', seed=seed)
    R = ReservoirParams(n_nodes=n_r, p=0.5, init='random', reset=True, seed=seed)
    O = OutputParams(n_nodes=1, seed=seed)
    T = TrainingParams(batch_size=32, epochs=1, accuracy_threshold=0.5,
                       optim={'name': 'gaussian', 'params': {'alpha': 1e-3}}, seed=seed)
    L = LoggingParams(out_path=out_path, save_keys=None)
    variables = {
        'R_k_avg': float(k_avg),
        'R_mode': mode,         # string variable — exercises mode="R_mode" resolution
    }
    wiring = [
        PatternWiring(source='B', target='I', pattern='identity'),
        RandomWiring(source='I', target='R',
                     k_avg=f'R_k_avg / (I_n_nodes + R_n_nodes)',
                     k_min=0, k_max='I_n_nodes', mode='heterogeneous'),
        RandomWiring(source='R', target='R',
                     k_avg='R_k_avg',
                     self_loops=f'R_k_avg / (n_nodes + 1)',
                     mode='R_mode'),   # <-- variable reference, not literal
    ]
    M = ModelParams(variables=variables, input_layer=I, reservoir_layer=R,
                    output_layer=O, training=T, wiring=wiring)
    return Params(model=M, logging=L)


# ---------------------------------------------------------------------------
# 1. Config expansion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("universe,expected_mode", [
    ('kqgr_heterogeneous',          'heterogeneous'),
    ('kqgr_heterogeneous_identity', 'heterogeneous'),
])
def test_expansion_r_mode_correct_per_universe(universe, expected_mode):
    """Every expanded config for a universe must have R_mode matching that universe."""
    P = load_yaml_config(YAML_PATH)
    combos = [c for c in generate_param_combinations(P)
              if next(iter(c.multiverse_overrides or {}), '') == universe]
    assert combos, f"No {universe} configs after expansion"
    for c in combos:
        r_mode = c.M.variables.get('R_mode')
        assert r_mode == expected_mode, (
            f"{universe}: R_mode={r_mode!r}, expected {expected_mode!r}"
        )


def test_expansion_r_k_avg_is_scalar_per_config():
    """Each expanded config must have R_k_avg as a scalar, not a list."""
    P = load_yaml_config(YAML_PATH)
    for c in generate_param_combinations(P):
        k_avg = c.M.variables.get('R_k_avg')
        assert not isinstance(k_avg, list), (
            f"R_k_avg is still a list in an expanded config — grid search expansion failed: {k_avg}"
        )


# ---------------------------------------------------------------------------
# 2. Universe tagging
# ---------------------------------------------------------------------------

def test_expansion_universe_names_are_tagged():
    """Expanded configs must be tagged with their universe key in multiverse_overrides."""
    P = load_yaml_config(YAML_PATH)
    combos = generate_param_combinations(P)

    universe_keys = {next(iter(c.multiverse_overrides or {}), None) for c in combos}
    expected_keys = {'kqgr_heterogeneous', 'kqgr_heterogeneous_identity',
                     'kqgr_homogeneous_real', 'kqgr_heterogeneous_real'}
    for key in expected_keys:
        assert key in universe_keys, f"Universe '{key}' missing from expanded configs"


# ---------------------------------------------------------------------------
# 3. Direct mode wiring: in-degree distributions
# ---------------------------------------------------------------------------

def test_homogeneous_direct_mode_uniform_indegree():
    """Direct mode='homogeneous' in wiring → every R node has exactly int(k_avg) in-edges."""
    k_avg = 5.0
    n_r = 100
    # Use the simpler direct-mode version from test_reservoir_scientific pattern
    I = InputParams(perturbation='override', encoding='base2',
                    features=1, bits=10, ticks='1{10}', seed=0)
    R = ReservoirParams(n_nodes=n_r, p=0.5, init='random', reset=True, seed=0)
    O = OutputParams(n_nodes=1, seed=0)
    T = TrainingParams(batch_size=32, epochs=1, accuracy_threshold=0.5,
                       optim={'name': 'gaussian', 'params': {'alpha': 1e-3}}, seed=0)
    L = LoggingParams(out_path='/tmp/boolean_reservoir/test/temporal/kqgr/snyder_2012/', save_keys=None)
    wiring = [
        PatternWiring(source='B', target='I', pattern='identity'),
        RandomWiring(source='I', target='R', k_avg=0.0, k_min=0, k_max='I_n_nodes',
                     mode='heterogeneous'),
        RandomWiring(source='R', target='R', k_avg=str(k_avg), k_min=0,
                     k_max=str(n_r), mode='homogeneous'),   # literal mode
    ]
    M = ModelParams(variables={}, input_layer=I, reservoir_layer=R,
                    output_layer=O, training=T, wiring=wiring)
    model = BooleanReservoir(Params(model=M, logging=L))

    degrees = _rr_in_degrees(model)
    expected_k = int(k_avg)
    assert (degrees == expected_k).all(), (
        f"Homogeneous mode: expected all R→R in-degrees == {expected_k}, "
        f"got min={degrees.min()} max={degrees.max()} unique={np.unique(degrees)}"
    )


def test_heterogeneous_direct_mode_variable_indegree():
    """Direct mode='heterogeneous' in wiring → R→R in-degrees vary (binomial-like)."""
    k_avg = 5.0
    n_r = 200
    I = InputParams(perturbation='override', encoding='base2',
                    features=1, bits=10, ticks='1{10}', seed=0)
    R = ReservoirParams(n_nodes=n_r, p=0.5, init='random', reset=True, seed=0)
    O = OutputParams(n_nodes=1, seed=0)
    T = TrainingParams(batch_size=32, epochs=1, accuracy_threshold=0.5,
                       optim={'name': 'gaussian', 'params': {'alpha': 1e-3}}, seed=0)
    L = LoggingParams(out_path='/tmp/boolean_reservoir/test/temporal/kqgr/snyder_2012/', save_keys=None)
    wiring = [
        PatternWiring(source='B', target='I', pattern='identity'),
        RandomWiring(source='I', target='R', k_avg=0.0, k_min=0, k_max='I_n_nodes',
                     mode='heterogeneous'),
        RandomWiring(source='R', target='R', k_avg=str(k_avg), k_min=0,
                     k_max=str(n_r), mode='heterogeneous'),  # literal mode
    ]
    M = ModelParams(variables={}, input_layer=I, reservoir_layer=R,
                    output_layer=O, training=T, wiring=wiring)
    model = BooleanReservoir(Params(model=M, logging=L))

    degrees = _rr_in_degrees(model)
    assert degrees.std() > 0.5, (
        f"Heterogeneous mode: expected variable in-degrees (std > 0.5), "
        f"got std={degrees.std():.3f} (all values: {np.unique(degrees)})"
    )


# ---------------------------------------------------------------------------
# 4. Variable-based mode resolution: mode="R_mode" in wiring
# ---------------------------------------------------------------------------

def test_variable_mode_homogeneous_gives_uniform_indegree():
    """When mode='R_mode' and variables['R_mode']='homogeneous', in-degrees must be uniform."""
    k_avg = 5.0
    n_r = 100
    with tempfile.TemporaryDirectory() as tmp:
        model = BooleanReservoir(_make_params_with_mode_variable(n_r, k_avg, 'homogeneous',
                                                                  seed=1, out_path=tmp + '/'))
    degrees = _rr_in_degrees(model)
    expected_k = int(k_avg)
    assert (degrees == expected_k).all(), (
        f"Variable-mode homogeneous: expected all in-degrees == {expected_k}, "
        f"got min={degrees.min()} max={degrees.max()} unique={np.unique(degrees)}"
    )


def test_variable_mode_heterogeneous_gives_variable_indegree():
    """When mode='R_mode' and variables['R_mode']='heterogeneous', in-degrees must vary."""
    k_avg = 5.0
    n_r = 200
    with tempfile.TemporaryDirectory() as tmp:
        model = BooleanReservoir(_make_params_with_mode_variable(n_r, k_avg, 'heterogeneous',
                                                                  seed=1, out_path=tmp + '/'))
    degrees = _rr_in_degrees(model)
    assert degrees.std() > 0.5, (
        f"Variable-mode heterogeneous: expected variable in-degrees, "
        f"got std={degrees.std():.3f}"
    )


# ---------------------------------------------------------------------------
# 5. GR divergence: modes produce different reservoir capacity
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_gr_differs_between_modes():
    """
    At k_avg=5 (critical region), homogeneous and heterogeneous reservoirs should
    produce different GR ranks.  Tested across multiple seeds for robustness.
    """
    k_avg = 5.0
    n_r = 25
    n_samples = 64
    bits = 10
    n_steps = bits  # one bit per step

    gr_hom_list = []
    gr_het_list = []

    for seed in range(5):
        with tempfile.TemporaryDirectory() as tmp_h, tempfile.TemporaryDirectory() as tmp_e:
            m_hom = BooleanReservoir(
                _make_params_with_mode_variable(n_r, k_avg, 'homogeneous',
                                                seed=seed, out_path=tmp_h + '/'))
            m_het = BooleanReservoir(
                _make_params_with_mode_variable(n_r, k_avg, 'heterogeneous',
                                                seed=seed, out_path=tmp_e + '/'))

            x = make_rng(seed, 'test_input').integers(
                0, 2, (n_samples, n_steps, 1, 1), dtype=np.uint8)
            gr_hom = compute_rank(m_hom, x, metric='gr')
            gr_het = compute_rank(m_het, x, metric='gr')

        gr_hom_list.append(gr_hom)
        gr_het_list.append(gr_het)

    mean_hom = np.mean(gr_hom_list)
    mean_het = np.mean(gr_het_list)

    assert mean_hom != mean_het, (
        f"GR should differ between modes (homogeneous mean={mean_hom:.1f}, "
        f"heterogeneous mean={mean_het:.1f})"
    )


# ---------------------------------------------------------------------------
# 6. Saved grid search results: statistical divergence between modes
#    (Snyder's shared-pool wiring is the one swept in both topology modes)
# ---------------------------------------------------------------------------

def _load_gr_df(n_nodes_filter=25):
    """Load GR values from saved grid search using the same extraction as the explore script."""
    from project.boolean_reservoir.code.utils.load_save import custom_load_grid_search_data, get_data_path
    if not get_data_path(YAML_PATH).exists():
        pytest.skip("No saved log — run the grid search first")
    extractions = [
        ('kqgr', lambda p: p.L.kqgr, {'kq', 'gr', 'delta'}),
        ('L', lambda p: p.L, {'universe'}),
        ('R', lambda p: p.M.R, {'n_nodes'}),
        ('R_k_avg', lambda p: p.M.variables.R_k_avg, None),
        ('R_mode', lambda p: 'heterogeneous' if 'heterogeneous' in (p.L.universe or '') else 'homogeneous', None),
    ]
    df, _ = custom_load_grid_search_data(
        config_paths=[YAML_PATH], extractions=extractions)
    df = df[(df['L_universe'].isin(['kqgr_homogeneous_real', 'kqgr_heterogeneous_real']))
            & (df['R_n_nodes'] == n_nodes_filter)
            & df['kqgr_gr'].notna()]
    if df.empty:
        pytest.skip("No matching rows in saved log — run the grid search first")
    return df


def test_saved_results_have_both_universe_modes():
    """Saved log must contain rows for both topology modes of Snyder's shared-pool wiring."""
    df = _load_gr_df()
    universes = set(df['L_universe'].unique())
    assert 'kqgr_homogeneous_real' in universes, "No kqgr_homogeneous_real rows in saved log"
    assert 'kqgr_heterogeneous_real' in universes, "No kqgr_heterogeneous_real rows in saved log"


def test_saved_results_delta_positive_at_critical_k_avg():
    """
    Regression for the compute_rank final-step fix.

    At k_avg=3 (critical regime), the shared tail steps cause partial GR convergence,
    so delta = kq - gr must be > 0 for a meaningful fraction of runs.

    If compute_rank uses ALL output_layer steps instead of the final step, early diverse
    states inflate GR rank to match KQ → delta=0 for every row, every k_avg.
    """
    df = _load_gr_df(n_nodes_filter=25)
    # Also need delta; reload to include it
    from project.boolean_reservoir.code.utils.load_save import custom_load_grid_search_data, get_data_path
    if not get_data_path(YAML_PATH).exists():
        pytest.skip("No saved log")
    extractions_with_delta = [
        ('kqgr', lambda p: p.L.kqgr, {'kq', 'gr', 'delta'}),
        ('L', lambda p: p.L, {'universe'}),
        ('R', lambda p: p.M.R, {'n_nodes'}),
        ('R_k_avg', lambda p: p.M.variables.R_k_avg, None),
    ]
    df_full, _ = custom_load_grid_search_data(config_paths=[YAML_PATH],
                                              extractions=extractions_with_delta)
    sub = df_full[
        (df_full['L_universe'].isin(['kqgr_homogeneous_real', 'kqgr_heterogeneous_real']))
        & (df_full['R_n_nodes'] == 25)
        & (df_full['R_k_avg'] == 3.0)
        & df_full['kqgr_delta'].notna()
    ]
    if sub.empty:
        pytest.skip("k_avg=3 data not present")

    frac_positive = (sub['kqgr_delta'] > 0).mean()
    assert frac_positive >= 0.3, (
        f"At k_avg=3 only {frac_positive:.1%} of rows have delta>0 (expected ≥30%). "
        f"Mean delta={sub['kqgr_delta'].mean():.2f}. "
        f"If delta is universally 0, compute_rank may be using all time steps not the final step."
    )


def test_saved_results_gr_distributions_are_not_identical():
    """
    Across all overlapping integer k_avg values, per-k_avg mean GR must differ between
    modes by more than 1 rank at some point.  If they were equal, the mode had no effect.
    """
    import pandas as pd
    df = _load_gr_df(n_nodes_filter=25)

    hom_means = df[df['L_universe'] == 'kqgr_homogeneous_real'].groupby('R_k_avg')['kqgr_gr'].mean()
    het_means = df[df['L_universe'] == 'kqgr_heterogeneous_real'].groupby('R_k_avg')['kqgr_gr'].mean()

    shared_k = hom_means.index.intersection(het_means.index)
    if len(shared_k) == 0:
        pytest.skip("No shared k_avg values between modes in saved log")

    max_diff = (hom_means[shared_k] - het_means[shared_k]).abs().max()
    assert max_diff > 1.0, (
        f"Max GR difference between modes is only {max_diff:.3f}. "
        f"Expected >1.0 if the connectivity distribution matters."
    )


if __name__ == '__main__':
    print("Run with: pytest project/temporal/test/test_snyder_2012_grid_search.py -v")
    print("Skip slow tests: pytest ... -m 'not slow'")
