"""dataset.sweep: several KQ/GR settings measured on ONE reservoir per job.

Pins what makes the sweep a measurement rather than an axis:
  * the sweep never changes the grid's combinations (so never a job's index or seed);
  * a sweep entry for the job's own setting equals that job's plain kqgr value, so a
    swept value means the same thing as a grid-axis value;
  * the probe cache (kqgr_probe.PROBE_CACHE_SIZE) changes nothing but speed, rng included.
"""
import numpy as np
import pytest

from project.boolean_reservoir.code import kqgr_probe
from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.utils.param_utils import LazyParamCombinations
from project.boolean_reservoir.code.train_model_parallel import BooleanReservoirJobFactory
from project.boolean_reservoir.code.kqgr_probe import build_kqgr_probe

TEMPORAL_CFG = 'config/temporal/density/grid_search/design_choices/all_heterogeneous.yaml'
SWEEP = {'tau': [3, 5], 'tau_mode': ['steal', 'augment'], 'evaluation': ['first', 'last', 'random']}   # = the config's axes
FIELDS = {'kq', 'gr', 'delta', 'kq_last', 'gr_last', 'delta_last'}


def _config(sweep=None):
    P0 = load_yaml_config(TEMPORAL_CFG)
    P0.dataset.bits = 10                      # small 2**width pools for a unit test
    if sweep is not None:
        for u in (P0.multiverse_overrides or {}).values():
            u.setdefault('dataset', {})['sweep'] = sweep
    return P0


def _job(P0, idx):
    combos = LazyParamCombinations(P0)
    i = round(idx * (len(combos) - 1))
    job = BooleanReservoirJobFactory(P0, combos)(i, 0, len(combos), 1, {})
    Pu = getattr(job.P.U, next(iter(job.P.multiverse_overrides)))
    return job, Pu


def _logs(job, field):
    L = job.P.L
    if L.heads:
        return {h: getattr(hl, field) for h, hl in L.heads.items()}
    return {None: getattr(L, field)}


def test_sweep_is_not_an_axis():
    assert len(LazyParamCombinations(_config())) == len(LazyParamCombinations(_config(SWEEP)))


def test_sweep_settings_and_names():
    D = _config(SWEEP).U.__getattr__(next(iter(_config(SWEEP).multiverse_overrides))).D
    settings = D.sweep_settings()
    assert len(settings) == 2 * 2 * 3
    assert settings[0] == {'tau': 3, 'tau_mode': 'steal', 'evaluation': 'first'}
    assert D.setting_name(5, 'steal', 'last') == 'last_steal_t5'
    assert D.setting_name(5, 'steal', 'last', True) == 'last_steal_t5_ind'


def test_sweep_rejects_unknown_fields():
    with pytest.raises(Exception, match='unknown KQ/GR fields'):
        P0 = _config({'tau': [2], 'washout': [3]})
        getattr(P0.U, next(iter(P0.multiverse_overrides))).D


@pytest.mark.parametrize('idx', [0.0, 0.37, 0.71, 1.0])
def test_sweep_entry_equals_plain_kqgr(idx):
    job, Pu = _job(_config(SWEEP), idx)
    if getattr(Pu.D, 'tau_axis', '') != 'steps':
        pytest.skip('not a steps-axis universe')
    job._run_kqgr(Pu)
    D = Pu.D
    own = D.setting_name(D.tau, D.tau_mode, D.evaluation, D.independent)
    for h, sweep in _logs(job, 'kqgr_sweep').items():
        assert len(sweep) == 12
        plain = _logs(job, 'kqgr')[h]
        assert own in sweep
        assert sweep[own].model_dump(include=FIELDS) == plain.model_dump(include=FIELDS)
        assert (sweep[own].tau, sweep[own].tau_mode, sweep[own].evaluation) == (D.tau, D.tau_mode, D.evaluation)


@pytest.mark.parametrize('idx', [0.0, 0.5, 1.0])
def test_probe_cache_is_byte_identical(idx, monkeypatch):
    _, Pu = _job(_config(), idx)
    if getattr(Pu.D, 'tau_axis', '') != 'steps':
        pytest.skip('not a steps-axis universe')
    n, seed = Pu.M.R.n_nodes, Pu.D.seed

    def probe():
        rng = np.random.default_rng(seed)
        kq, gr, _ = build_kqgr_probe(Pu, Pu.dataset_init_obj, n, rng)
        return kq, gr, rng.random()            # the rng's end state, via its next draw

    monkeypatch.setattr(kqgr_probe, 'PROBE_CACHE_SIZE', 0)
    kqgr_probe._PROBE_CACHE.clear()
    ref = probe()
    monkeypatch.setattr(kqgr_probe, 'PROBE_CACHE_SIZE', 4)
    first, hit = probe(), probe()             # a miss that stores, then a hit
    assert len(kqgr_probe._PROBE_CACHE) == 1
    for got in (first, hit):
        assert np.array_equal(got[0], ref[0]) and np.array_equal(got[1], ref[1]) and got[2] == ref[2]
    kqgr_probe._PROBE_CACHE.clear()
