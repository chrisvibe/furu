"""
Tests for the KQGR metric input construction and rank computation.

Verifies:
  1. Degenerate tau (too large) warns rather than crashes.
  2. State matrix rank vs distinct-input count for real reservoir runs.
  3. KQ and GR datasets have identical shapes (catches double-tau-counting bug).
  4. All KQ samples are distinct.
"""

import numpy as np
import pytest

from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.kqgr_probe import build_kqgr_probe

FIGURE1_CONFIG = 'config/temporal/kqgr/grid_search/figure_1_snyder_2012.yaml'
N_NODES = 25  # reservoir size in figure_1 config


# ---------------------------------------------------------------------------
# 1. Degenerate tau warning
# ---------------------------------------------------------------------------

def _load_kqgr_dataset_tau(tau, eval_mode='last'):
    from project.temporal.code.dataset_init import TemporalDatasetInit
    P = load_yaml_config(FIGURE1_CONFIG)
    P_uni = P.U.kqgr_homogeneous
    P_uni.D.tau = tau
    P_uni.D.evaluation = eval_mode
    P_uni.M.R.n_nodes = N_NODES
    rng = np.random.default_rng(0)
    _, gr_x, _ = build_kqgr_probe(P_uni, TemporalDatasetInit(), N_NODES, rng)
    return gr_x


def test_input_diversity_tau_degenerate_warns():
    """tau=7 with N=25, bits=10 → 2^3=8 < 25: should warn, not crash."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = _load_kqgr_dataset_tau(tau=7, eval_mode='last')
        assert any("repeat" in str(warning.message).lower() for warning in w),             "Expected a UserWarning about repeating inputs"
    assert x.shape[1] == 10


# ---------------------------------------------------------------------------
# 2. State matrix rank sanity
# ---------------------------------------------------------------------------

def _build_model_and_run_kqgr(tau, k_avg=5, eval_mode='last'):
    from project.boolean_reservoir.code.reservoir import BooleanReservoir, BatchedTensorHistoryWriter
    from project.temporal.code.dataset_init import TemporalDatasetInit
    from project.boolean_reservoir.code.parameter import (
        Params, ModelParams, InputParams, ReservoirParams, OutputParams, TrainingParams,
        PatternWiring,
    )
    from pathlib import Path
    import tempfile

    P = Params(
        model=ModelParams(
            input_layer=InputParams(
                n_nodes=1,
                perturbation='override', encoding='base2',
                features=1, resolution=10, chunk_size=1,
            ),
            reservoir_layer=ReservoirParams(
                n_nodes=N_NODES, k_avg=k_avg, k_min=k_avg, k_max=k_avg,
                mode='homogeneous', p=0.5, reset=True,
            ),
            output_layer=OutputParams(n_nodes=1),
            training=TrainingParams(),
            wiring=[
                PatternWiring(source='B', target='I', pattern='identity'),
                PatternWiring(source='I', target='R', pattern='identity'),
            ],
        ),
        multiverse_overrides={
            'kqgr': {'dataset': {'tau': tau, 'evaluation': eval_mode}}
        }
    )
    P_uni = P.U.kqgr

    rng = np.random.default_rng(0)
    _, x, P_model = build_kqgr_probe(P_uni, TemporalDatasetInit(), N_NODES, rng)
    n_distinct_inputs = len(np.unique(x.reshape(x.shape[0], -1), axis=0))

    with tempfile.TemporaryDirectory() as tmpdir:
        P_model.L.save_path = Path(tmpdir)
        P_model.L.out_path = Path(tmpdir)
        model = BooleanReservoir(P_model)
        model.eval()
        model.record = True
        model.history = BatchedTensorHistoryWriter(
            save_path=Path(tmpdir) / 'history',
            persist_to_disk=False
        )
        _ = model(x)
        model.flush_history()

        _, history, expanded_meta, _ = model.history.reload_history()
        df_filter = expanded_meta[expanded_meta['phase'] == 'output_layer']
        filtered = history[df_filter.index].astype(np.float32)
        state_matrix = filtered[:, ~model.input_nodes_mask]

    return state_matrix, n_distinct_inputs


@pytest.mark.parametrize("tau", [0, 1, 3])
def test_state_matrix_rank_sanity(tau):
    """Rank <= n_reservoir and rank <= distinct_inputs."""
    state_matrix, n_distinct_inputs = _build_model_and_run_kqgr(tau=tau, k_avg=5.0)
    rank = np.linalg.matrix_rank(state_matrix)
    n_reservoir = state_matrix.shape[1]
    assert rank <= n_reservoir
    assert rank <= n_distinct_inputs


# ---------------------------------------------------------------------------
# 3 & 4. KQ/GR shape invariant + KQ sample distinctness
# ---------------------------------------------------------------------------

# the path-integration cases of these tests: project/path_integration/test/test_kqgr_pi.py


def _get_kqgr_datasets(config_path, n_nodes=None):
    from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
    P = load_yaml_config(config_path)
    combos = generate_param_combinations(P)
    P0 = combos[0]
    universe_key = next(iter(P0.multiverse_overrides or {}), None)
    if universe_key is None:
        pytest.skip('No multiverse in config')
    P_universe = getattr(P0.U, universe_key)
    n = n_nodes if n_nodes is not None else P_universe.M.R.n_nodes
    init = P_universe.dataset_init_obj
    rng = np.random.default_rng(0)
    kq, gr, _ = build_kqgr_probe(P_universe, init, n, rng)
    return kq, gr, P_universe


@pytest.mark.parametrize('config_path', [FIGURE1_CONFIG])
def test_kq_gr_same_shape(config_path):
    """KQ and GR datasets must have identical shapes (catches double-tau-counting bug)."""
    kq_x, gr_x, _ = _get_kqgr_datasets(config_path)
    assert kq_x.shape == gr_x.shape, (
        'KQ shape ' + str(kq_x.shape) + ' != GR shape ' + str(gr_x.shape) +
        ' for ' + config_path
    )


@pytest.mark.parametrize('config_path', [FIGURE1_CONFIG])
def test_kq_all_samples_distinct(config_path):
    """KQ measures input separability — every sample must be distinct."""
    kq_x, _, _ = _get_kqgr_datasets(config_path)
    flat = kq_x.reshape(kq_x.shape[0], -1)
    n_distinct = len(np.unique(flat, axis=0))
    assert n_distinct == len(flat), (
        'KQ dataset has only ' + str(n_distinct) + '/' + str(len(flat)) + ' distinct samples.'
    )


# ---------------------------------------------------------------------------
# 4b. step_length_field(): the tau_axis='steps' dataset-field resolver
# ---------------------------------------------------------------------------

def test_step_length_field_temporal_is_bits():
    from benchmark.temporal.parameter import KQGRTemporalDatasetParams
    D = KQGRTemporalDatasetParams(name='temporal', tau=3, bits=10)
    assert D.step_length_field() == 'bits'


def test_step_length_field_unsupported_dataset_is_none():
    from benchmark.utils.parameter import KQGRDatasetParams
    D = KQGRDatasetParams(tau=0)
    assert D.step_length_field() is None


def _bare_steps_params(tau_mode):
    """A minimal duck-typed P with a KQGRDatasetParams dataset that has neither
    'bits' nor 'steps' — simulates an unsupported dataset type for tau_axis='steps'
    without needing a real fourth dataset class."""
    from types import SimpleNamespace
    from benchmark.utils.parameter import KQGRDatasetParams
    D = KQGRDatasetParams(tau=1, tau_axis='steps', tau_mode=tau_mode)
    return SimpleNamespace(dataset=D, D=D, M=SimpleNamespace(I=SimpleNamespace(resolution=10)))


def test_get_supply_width_raises_for_unsupported_steps_dataset():
    from project.boolean_reservoir.code.kqgr_probe import get_supply_width
    P = _bare_steps_params('steal')
    with pytest.raises(NotImplementedError):
        get_supply_width(P, 'steps')


def test_set_supply_width_raises_for_unsupported_steps_dataset():
    from project.boolean_reservoir.code.kqgr_probe import set_supply_width
    P = _bare_steps_params('augment')
    with pytest.raises(NotImplementedError):
        set_supply_width(P, 'steps', 5)


# ---------------------------------------------------------------------------
# 5. Rank must not be capped by batch_size
# ---------------------------------------------------------------------------

def test_compute_rank_not_capped_by_batch_size():
    """compute_rank must aggregate states across forward batches.

    forward() processes m > T.batch_size in batches and overwrites
    states_parallel per batch, so a single post-hoc read of states_parallel
    keeps only the last batch's states and silently caps the rank at
    batch_size. With m = 64 distinct random inputs injected in parallel
    (chunks=1, 40 driven nodes — PI-style) into a well-mixing reservoir
    (k=8), the final-state rank sits well above batch_size = 8 (15–22 across
    wiring draws), so a batch_size cap is unambiguous.
    """
    from pathlib import Path
    import tempfile
    from project.boolean_reservoir.code.reservoir import BooleanReservoir
    from project.boolean_reservoir.code.kq_and_gr_metric import compute_rank
    from project.boolean_reservoir.code.parameter import (
        Params, ModelParams, InputParams, ReservoirParams, OutputParams,
        TrainingParams, PatternWiring,
    )

    P = Params(
        model=ModelParams(
            input_layer=InputParams(
                perturbation='override', encoding='base2',
                features=1, resolution=20, redundancy=2, chunks=1,
            ),
            reservoir_layer=ReservoirParams(
                n_nodes=128, k_avg=8, k_min=8, k_max=8,
                mode='homogeneous', p=0.5, reset=True,
            ),
            output_layer=OutputParams(n_nodes=1),
            training=TrainingParams(batch_size=8, seed=0),
            wiring=[
                PatternWiring(source='B', target='I', pattern='identity'),
                PatternWiring(source='I', target='R', pattern='identity'),
            ],
        ),
    )
    m, s = 64, 3
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(m, s, 1, P.M.I.bits), dtype=np.uint8)
    assert len(np.unique(x.reshape(m, -1), axis=0)) == m

    with tempfile.TemporaryDirectory() as tmpdir:
        P.L.save_path = Path(tmpdir)
        P.L.out_path = Path(tmpdir)
        model = BooleanReservoir(P)
        rank = compute_rank(model, x)

    assert rank > P.M.T.batch_size, (
        'rank ' + str(rank) + ' capped at batch_size ' + str(P.M.T.batch_size) +
        ' — compute_rank is reading only the last forward batch.'
    )


# ---------------------------------------------------------------------------
# 6. compute_rank_pair: readout-representation rank vs final-state rank
# ---------------------------------------------------------------------------

def _pair_params(output_mode, k_avg, n_nodes=128, batch_size=32):
    from project.boolean_reservoir.code.parameter import (
        Params, ModelParams, InputParams, ReservoirParams, OutputParams,
        TrainingParams, PatternWiring,
    )
    return Params(
        model=ModelParams(
            input_layer=InputParams(
                perturbation='override', encoding='base2',
                features=1, resolution=20, redundancy=2, chunks=1,
            ),
            reservoir_layer=ReservoirParams(
                n_nodes=n_nodes, k_avg=k_avg, k_min=k_avg, k_max=k_avg,
                mode='homogeneous', p=0.5, reset=True,
            ),
            output_layer=OutputParams(n_nodes=1, mode=output_mode),
            training=TrainingParams(batch_size=batch_size, seed=0),
            wiring=[
                PatternWiring(source='B', target='I', pattern='identity'),
                PatternWiring(source='I', target='R', pattern='identity'),
            ],
        ),
    )


def _run_pair(output_mode, k_avg, x_fn):
    from pathlib import Path
    import tempfile
    from project.boolean_reservoir.code.reservoir import BooleanReservoir
    from project.boolean_reservoir.code.kq_and_gr_metric import (
        compute_rank, compute_rank_pair,
    )
    P = _pair_params(output_mode, k_avg)
    rng = np.random.default_rng(0)
    x = x_fn(rng, P.M.I.bits)
    with tempfile.TemporaryDirectory() as tmpdir:
        P.L.save_path = Path(tmpdir)
        P.L.out_path = Path(tmpdir)
        model = BooleanReservoir(P)
        r_readout, r_last = compute_rank_pair(model, x)
        r_single = compute_rank(model, x)
    return r_readout, r_last, r_single


def _random_streams(rng, bits, m=96, s=5):
    return rng.integers(0, 2, size=(m, s, 1, bits), dtype=np.uint8)


def _shared_tail_streams(rng, bits, m=96, s=5, tau=4):
    """GR-steps construction: streams share the last tau steps, so any rank
    above 1 comes from distinctions retained from older inputs."""
    x = _random_streams(rng, bits, m, s)
    tail = rng.integers(0, 2, size=(tau, 1, bits), dtype=np.uint8)
    x[:, s - tau:] = tail[None, :]
    return x


def test_compute_rank_pair_classification_readout_equals_last():
    """In classification mode get_states returns the final state (binary
    encoding), so both ranks of the pair and compute_rank must agree."""
    r_readout, r_last, r_single = _run_pair('classification', 8.0, _random_streams)
    assert r_readout == r_last == r_single


def test_compute_rank_pair_last_matches_compute_rank_timeseries():
    """The pair's last-state rank is Snyder's construction — it must equal
    compute_rank regardless of output mode."""
    _, r_last, r_single = _run_pair('time-series', 8.0, _random_streams)
    assert r_last == r_single


def test_compute_rank_pair_timeseries_sum_retains_memory():
    """In time-series mode the readout sees per-step states summed over
    steps. On shared-tail (GR-steps) probes, a subcritical reservoir's final
    state forgets old-input distinctions that the summed trace retains, so
    the readout rank must exceed the final-state rank."""
    r_readout, r_last, _ = _run_pair('time-series', 2.0, _shared_tail_streams)
    assert r_readout > r_last, (
        'readout rank ' + str(r_readout) + ' <= last-state rank ' + str(r_last) +
        ' — summed representation should retain old-input distinctions that '
        'the final state has forgotten.'
    )


# ---------------------------------------------------------------------------
# 7. gr_steps_probe: steps-axis retention GR built from a KQ probe set
# ---------------------------------------------------------------------------

def test_gr_steps_probe_shares_tail_keeps_head():
    from project.boolean_reservoir.code.kq_and_gr_metric import gr_steps_probe
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(64, 5, 1, 60), dtype=np.uint8)
    x_shared, tau = gr_steps_probe(x)
    # PI-like geometry: 60 bits/step >> log2(64)+2, so only step 0 need differ
    assert tau == 4
    assert (x_shared[:, 1:] == x_shared[:1, 1:]).all(), 'tails must be shared'
    assert (x_shared[:, 0] == x[:, 0]).all(), 'heads must be untouched'
    assert not (x[:, 1:] == x[:1, 1:]).all(), 'original x must not be mutated in place'


def test_gr_steps_probe_declines_narrow_streams():
    """1-bit-per-step temporal streams can't hold m distinct heads in a
    short prefix — the probe must decline (native GR covers steps there)."""
    from project.boolean_reservoir.code.kq_and_gr_metric import gr_steps_probe
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(512, 10, 1, 1), dtype=np.uint8)
    x_shared, tau = gr_steps_probe(x)
    assert x_shared is None and tau == 0


def test_gr_steps_rank_below_separation_rank():
    """Retention rank counts a subset of the distinctions separation rank
    counts, so on the same model gr_steps <= kq (small slack for rank
    noise). Uses a well-mixing time-series reservoir."""
    from pathlib import Path
    import tempfile
    from project.boolean_reservoir.code.reservoir import BooleanReservoir
    from project.boolean_reservoir.code.kq_and_gr_metric import (
        compute_rank_pair, gr_steps_probe,
    )
    P = _pair_params('time-series', 8.0)
    rng = np.random.default_rng(0)
    x = _random_streams(rng, P.M.I.bits)
    x_steps, tau = gr_steps_probe(x)
    assert tau >= 1
    with tempfile.TemporaryDirectory() as tmpdir:
        P.L.save_path = Path(tmpdir)
        P.L.out_path = Path(tmpdir)
        model = BooleanReservoir(P)
        kq, _ = compute_rank_pair(model, x)
        gr_steps, gr_steps_last = compute_rank_pair(model, x_steps)
    assert 1 <= gr_steps <= kq + 2
    assert 1 <= gr_steps_last


# ---------------------------------------------------------------------------
# 6. tau_mode x tau_axis: KQ and GR must be probed at the same width
# ---------------------------------------------------------------------------

TEMPORAL_DENSITY_CONFIG = 'config/temporal/density/grid_search/design_choices/all_homogeneous.yaml'

# Both live configs probe the steps axis, each with a universe whose grid varies
# tau_mode over both steal and augment. The resolution axis survives only in the
# retired design_choices/old/old_non_linear_boundary/ configs and is not used in
# the paper, so _probe_pair raises NotImplementedError rather than pretend to
# cover it. (This table previously labelled the PI cell 'resolution', which the
# config no longer is; the mislabel made both PI cells fail before they reached
# the shape assertions they exist to make.)
TAU_AXIS_CELLS = [
    (TEMPORAL_DENSITY_CONFIG, 'train', 'steps'),
]

PROBE_N_NODES = 32  # probe width is independent of reservoir size; keeps the build cheap


def _combo_for(P, universe, **pins):
    """Return the one Params for `universe` with the named grid axes pinned.

    Uses LazyParamCombinations' O(1) mixed-radix indexing (last axis varies
    fastest) rather than materialising the product — the PI grid alone is
    ~628k combinations. Axes not named in `pins` stay at their first value.
    """
    import math
    from project.boolean_reservoir.code.utils.param_utils import LazyParamCombinations
    combos = LazyParamCombinations(P)
    offset = 0
    for spec in combos._specs:
        if spec.key != universe:
            offset += len(spec)
            continue
        local = 0
        for i, ax in enumerate(spec.axes):
            want = pins.get(ax.path[-1])
            idx = ax.values.index(want) if want in ax.values else 0
            stride = math.prod(len(a.values) for a in spec.axes[i + 1:])
            local += idx * stride
        return combos[offset + local]
    raise KeyError('no universe %r in %r' % (universe, list(P.multiverse_overrides or {})))


def _probe_pair(config_path, universe, tau_mode, tau=3, **pins):
    """Build the KQ and GR probe tensors for one (universe, tau_mode) cell."""
    P = load_yaml_config(config_path)
    P_uni = getattr(_combo_for(P, universe, tau_mode=tau_mode, tau=tau, **pins).U, universe)
    if P_uni.D.tau_axis != 'steps':
        raise NotImplementedError(
            "tau_axis=%r is not supported here: the paper uses 'steps' only, and the "
            "resolution axis survives just in the retired old_non_linear_boundary "
            "configs. Re-add coverage deliberately if that changes."
            % (P_uni.D.tau_axis,))
    P_uni.M.R.n_nodes = PROBE_N_NODES
    init = P_uni.dataset_init_obj
    rng = np.random.default_rng(0)
    kq, gr, _ = build_kqgr_probe(P_uni, init, PROBE_N_NODES, rng)
    return kq, gr, P_uni


@pytest.mark.parametrize('tau_mode', ['steal', 'augment'])
@pytest.mark.parametrize('config_path,universe,tau_axis', TAU_AXIS_CELLS,
                         ids=['temporal'])
def test_kq_gr_same_probe_shape_per_tau_mode(config_path, universe, tau_axis, tau_mode):
    """KQ and GR must be probed at identical width in every (tau_axis, tau_mode) cell.

    Delta = KQ - GR only measures retention if both ranks come from probe sets of the
    same size. If GR runs over more steps (or more bits) than KQ it gets extra mixing
    for free, and Delta goes negative for reasons unrelated to retention.

    test_kq_gr_same_shape covers this too, but only for whichever tau_mode happens to
    sit first in the grid, so it never reached augment on the steps axis.
    """
    kq, gr, P_uni = _probe_pair(config_path, universe, tau_mode)
    assert P_uni.D.tau_axis == tau_axis, 'cell mislabelled: expected tau_axis=%s' % tau_axis
    if P_uni.D.tau_mode != tau_mode:
        pytest.skip('universe %r pins tau_mode=%r' % (universe, P_uni.D.tau_mode))
    assert kq.shape == gr.shape, (
        'tau_axis=%s tau_mode=%s: KQ probe %s != GR probe %s'
        % (tau_axis, tau_mode, kq.shape, gr.shape))


@pytest.mark.parametrize('config_path,universe,tau_axis', TAU_AXIS_CELLS,
                         ids=['temporal'])
def test_augment_widens_the_kq_probe_too(config_path, universe, tau_axis):
    """'augment' must add tau to the KQ probe as well, not only to GR.

    steal carves the tau shared positions out of the existing budget, so the probe
    keeps the task's own width. augment adds them on top, so both KQ and GR must grow
    relative to steal. If only GR grows, GR is measured over a longer stream than KQ
    and the pair is not comparable — which is exactly what a same-shape assertion
    alone would still permit if a fix shrank GR instead of widening KQ.
    """
    kq_steal, _, _ = _probe_pair(config_path, universe, 'steal')
    kq_aug, _, P_aug = _probe_pair(config_path, universe, 'augment')
    if P_aug.D.tau_mode != 'augment':
        pytest.skip('universe %r pins tau_mode=%r' % (universe, P_aug.D.tau_mode))
    axis = 1 if tau_axis == 'steps' else 3  # probe tensor is (m, s, f, b)
    assert kq_aug.shape[axis] > kq_steal.shape[axis], (
        'tau_axis=%s: augment KQ probe %s is no wider than steal KQ probe %s — '
        'augment did not augment the KQ side' % (tau_axis, kq_aug.shape, kq_steal.shape))


# ---------------------------------------------------------------------------
# 8. independent vs. dependent (paired) probe construction
# ---------------------------------------------------------------------------

def _temporal_probe_universe(tau, tau_mode, independent):
    P = load_yaml_config(TEMPORAL_DENSITY_CONFIG)
    P_uni = getattr(_combo_for(P, 'train', tau=tau, tau_mode=tau_mode).U, 'train')
    P_uni.D.tau, P_uni.D.tau_mode, P_uni.D.independent = tau, tau_mode, independent
    return P_uni


@pytest.mark.parametrize('tau_mode', ['steal', 'augment'])
@pytest.mark.parametrize('tau', [1, 2])
def test_dependent_mode_guarantees_shape_parity(tau, tau_mode):
    """Dependent (paired) mode: KQ and GR must match in sample count exactly, by construction, at small and
    production (n_nodes=2048) scale. The path-integration case: test_kqgr_pi.py."""
    P_uni = _temporal_probe_universe(tau, tau_mode, independent=False)
    init = P_uni.dataset_init_obj
    rng = np.random.default_rng(0)
    for n_nodes in (32, 2048):
        kq, gr, _ = build_kqgr_probe(P_uni, init, n_nodes, rng)
        assert kq.shape == gr.shape, (
            'dependent mode: KQ %s != GR %s at n_nodes=%d (tau=%d, tau_mode=%s)'
            % (kq.shape, gr.shape, n_nodes, tau, tau_mode)
        )


def test_independent_mode_runs_without_crashing():
    """Independent mode: the two-separate-draws path runs, and the widths (steps/features/bits) match."""
    P_uni = _temporal_probe_universe(1, 'steal', independent=True)
    init = P_uni.dataset_init_obj
    rng = np.random.default_rng(0)
    kq, gr, _ = build_kqgr_probe(P_uni, init, 32, rng)
    assert kq.shape[1:] == gr.shape[1:], 'widths (steps/features/bits) must still match'


def test_snyder_config_stays_independent_by_default():
    """The Snyder replication must keep its established (independent)
    methodology under the new dependent-by-default global setting."""
    P = load_yaml_config(FIGURE1_CONFIG)
    universe_key = next(iter(P.multiverse_overrides))
    P_uni = getattr(P.U, universe_key)
    assert P_uni.D.independent is True
