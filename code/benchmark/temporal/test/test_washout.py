"""dataset.washout: random lead-in bits before the scored stream, so every sample reaches its windows from its own
state instead of the shared reset state (primacy, furu probe_delay 2026-10-12).

  1. The prefix changes nothing that is scored: the same scored streams, labels, heads and split, row for row, and
     exhaustive sampling stays exhaustive (dev/test streams never seen in training).
  2. The time-series readout skips the washout steps: the sum of the per-step states from step w on, with the summed
     step count as bias scale, in get_states and in forward alike.
  3. KQ/GR freezes within the scored steps.
"""
import numpy as np
import pytest

from benchmark.temporal.parameter import TemporalDatasetParams
from benchmark.temporal.temporal_density_parity_dataset import TemporalDataset
from project.boolean_reservoir.code.kqgr_probe import freeze
from project.boolean_reservoir.code.parameter import Params, load_yaml_config
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.train_model_parallel import apply_seed
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
from pathlib import Path

W = 10
_PROBE = Path(__file__).resolve().parents[3] / 'config/test/temporal/density_w5_d5.yaml'


def _ds(tmp_path, washout, **kw):
    D = TemporalDatasetParams(task='parity', bits=8, window=3, delay=2, samples=256, sampling_mode='exhaustive',
                              shuffle=True, seed=3, washout=washout, generate_data=True, **kw)
    D.path = tmp_path / f'wo{washout}' / 'dataset.npz'
    return TemporalDataset(D)


def test_prefix_changes_nothing_scored(tmp_path):
    plain, led = _ds(tmp_path, 0), _ds(tmp_path, W)
    assert led.x.shape[-1] == plain.x.shape[-1] + W
    assert np.array_equal(led.x[..., W:], plain.x) and np.array_equal(led.y, plain.y)
    assert len({r.tobytes() for r in led.x[..., W:]}) == len(led.x), 'still exhaustive over the scored bits'
    assert 0.3 < led.x[..., :W].mean() < 0.7 and len({r.tobytes() for r in led.x[..., :W]}) > 100, 'random lead-in'
    plain.split_dataset(), led.split_dataset()
    for k in ('', '_dev', '_test'):
        assert np.array_equal(led.data['x' + k][..., W:], plain.data['x' + k]), k


def _model(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    p = generate_param_combinations(load_yaml_config(str(_PROBE)))[2]
    apply_seed(p, 1)
    d = p.model_dump()
    d['dataset'].update(samples=256, washout=W)
    d['model']['output_layer'].update(heads=None, mode=mode)
    p = Params.model_validate(d)
    return p, p.dataset_init_obj.train(p)


def test_timeseries_readout_skips_the_washout(tmp_path, monkeypatch):
    p, ds = _model(tmp_path, monkeypatch, 'time-series')
    model = BooleanReservoir(params=p)
    x = ds.x[:p.M.T.batch_size]
    s = x.shape[1]
    states, scale = model.get_states(x)
    per_step = [np.asarray(model.get_states(x, lengths=np.full(len(x), n), heads=['classification'])['classification'][0])
                for n in range(W + 1, s + 1)]
    assert scale == s - W and np.array_equal(states, np.sum(per_step, axis=0))
    rng = np.random.default_rng(0)
    model.set_readout_weights(rng.normal(size=model.readout_W.shape), rng.normal(size=model.readout_b.shape))
    assert np.allclose(model(x), states @ model.readout_W.T + scale * model.readout_b, atol=1e-4), 'forward agrees'


def test_freeze_stays_in_the_scored_steps():
    x = np.arange(4 * 20).reshape(4, 20, 1, 1)
    rng = np.random.default_rng(0)
    first = freeze(x, 3, 'steps', 'first', rng, ref_idx=0, start=W)
    assert np.array_equal(np.flatnonzero((first != x).any((0, 2, 3))), np.arange(W, W + 3))
    for seed in range(20):
        changed = np.flatnonzero((freeze(x, 3, 'steps', 'random', np.random.default_rng(seed), ref_idx=0, start=W) != x).any((0, 2, 3)))
        assert changed.min() >= W
