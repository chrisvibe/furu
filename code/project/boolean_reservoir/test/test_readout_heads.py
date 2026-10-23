"""Readout heads (output_layer.heads): several readout modes fitted on ONE reservoir run.

  1. get_states(heads=...) == separate single-mode models, bit for bit (seeded twins share
     the reservoir: the readout mode never enters the graph/LUT draw).
  2. glm.train_readout_heads: every head's fit IS the single-mode fit (identical fit
     diagnostics, hence weights); scores agree to float32 rounding (state-based predictions
     vs the engine's float32 forward).
  3. Config: heads exclusive with a mode list; mode defaults into the heads.
  4. The grid job logs one row per model with per-head train + kqgr, and the loader
     expands it into one single-mode-shaped row per head.
"""
import copy
import dataclasses

import numpy as np
import orjson
import pytest

from project.boolean_reservoir.code import glm
from project.boolean_reservoir.code.parameter import HeadLog, OutputParams, Params, load_yaml_config
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.train_model import _score_splits, criterion_fn
from project.boolean_reservoir.code.train_model_parallel import BooleanReservoirJob, apply_seed
from project.boolean_reservoir.code.utils.load_save import explode_heads
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
from pathlib import Path

HEADS = ['time-series', 'classification']
_CODE = Path(__file__).resolve().parents[3]
_PROBE = _CODE / 'config/test/temporal/density_w5_d5.yaml'


def _params(tmp_path, monkeypatch, samples=600, **output):
    monkeypatch.chdir(tmp_path)
    p = generate_param_combinations(load_yaml_config(str(_PROBE)))[0]
    apply_seed(p, 7)
    d = p.model_dump()
    d['dataset'].update(samples=samples)
    d['model']['output_layer'].update(output)
    return Params.model_validate(d)


def _with_mode(p, mode):
    q = copy.deepcopy(p)
    q.M.O.mode, q.M.O.heads = mode, None
    return q


def test_head_states_equal_single_mode_models(tmp_path, monkeypatch):
    p = _params(tmp_path, monkeypatch)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, (40, 5, p.M.I.chunks, p.M.I.chunk_size)).astype(np.uint8)
    n = rng.integers(1, 6, 40)
    single = {h: BooleanReservoir(params=_with_mode(p, h)) for h in HEADS}
    assert single[HEADS[0]].lut == single[HEADS[1]].lut, "seeded twins share the reservoir"
    for host in HEADS:
        for lengths in (None, n):
            both = single[host].get_states(x, lengths=lengths, heads=HEADS)
            for h, (states, scale) in both.items():
                ref_states, ref_scale = single[h].get_states(x, lengths=lengths)
                assert np.array_equal(states, ref_states) and np.array_equal(np.asarray(scale), np.asarray(ref_scale))


def test_each_head_is_the_single_mode_fit(tmp_path, monkeypatch):
    p = _params(tmp_path, monkeypatch)
    ds = p.dataset_init_obj.train(p)
    acc = p.accuracy_obj
    single = {h: glm.train_readout(BooleanReservoir(params=_with_mode(p, h)), ds, 'gaussian', 0.0,
                                   acc, criterion_fn, _score_splits)[0] for h in HEADS}
    multi = glm.train_readout_heads(BooleanReservoir(params=_with_mode(p, 'classification')), ds, HEADS,
                                    'gaussian', 0.0, acc, criterion_fn, _score_splits)['train']
    for h in HEADS:
        assert set(multi[h]) == set(single[h])
        for k, v in single[h].items():
            if k.startswith('fit_'):
                assert multi[h][k] == v, (h, k)                     # the SAME estimator and weights
            else:
                assert multi[h][k] == pytest.approx(v, abs=1e-5), (h, k)


def _with_targets(ds, targets_of, idx_of):
    """A copy of ds with step-readout keys (PI's row keys, any targets): 'positions' (m, K, 1) =
    targets_of(y) per split, 'q' = idx_of(m), and y set to the indexed target (the contract)."""
    out = copy.copy(ds)
    for key in ('', '_dev', '_test'):
        t, i = targets_of(ds.data['y' + key]), idx_of(len(ds.data['y' + key]))
        setattr(out, 'positions' + key, t)
        setattr(out, 'q' + key, i)
        setattr(out, 'y' + key, t[np.arange(len(i)), i - 1])
    return out


def test_step_readouts_are_one_readout_per_target(tmp_path, monkeypatch):
    """Step readouts: readout t fitted on every row to target t, row i scored by readout idx[i].
    Targets y and -y under a random idx: one shared readout cannot tell the sign, the step
    readouts score as the plain y fit exactly (the -y readout is that fit negated). And
    the shared readout is the plain fit of the indexed y, with step readouts or not."""
    p = _params(tmp_path, monkeypatch)
    base = p.dataset_init_obj.train(p)
    acc = p.accuracy_obj
    fit = lambda d, **kw: glm.train_readout_heads(BooleanReservoir(params=_with_mode(p, 'classification')), d, HEADS,
                                                  'gaussian', 0.0, acc, criterion_fn, _score_splits, **kw)
    rng = np.random.default_rng(1)
    ds = _with_targets(base, lambda y: np.stack([y, -y], axis=1), lambda m: rng.integers(1, 3, m))
    plain, stepped, alone = fit(base), fit(ds, groups=(glm.ReadoutGroup('step_readouts', 'positions', index='q'),)), fit(ds)
    assert set(stepped) == {'train', 'step_readouts'} and set(alone) == {'train'}
    for h in HEADS:
        for k, v in alone['train'][h].items():
            assert stepped['train'][h][k] == pytest.approx(v, rel=1e-6, abs=1e-6), ('shared', h, k)
        for s in ('train', 'dev', 'test'):
            assert stepped['step_readouts'][h][f'loss_{s}'] == pytest.approx(plain['train'][h][f'loss_{s}'], rel=1e-4), (h, s)
            assert stepped['step_readouts'][h][f'loss_{s}'] < stepped['train'][h][f'loss_{s}'], 'the shared readout cannot tell the sign'
        assert stepped['step_readouts'][h]['fit_rank'] == plain['train'][h]['fit_rank']


def test_heads_config():
    o = OutputParams(heads=['time-series', 'classification'])
    assert o.mode == 'classification'
    assert OutputParams(heads=['time-series']).mode == 'time-series', "mode defaults into the heads"
    for bad in (dict(heads=['time-series'], mode=['classification', 'time-series']),
                dict(heads=['time-series', 'time-series']), dict(heads=[])):
        with pytest.raises(ValueError):
            OutputParams(**bad)


def test_job_logs_heads_and_loader_expands_them(tmp_path, monkeypatch):
    p = _params(tmp_path, monkeypatch, samples=300, heads=HEADS)
    p.L.out_path = tmp_path / 'run'
    P = BooleanReservoirJob(i=3, j=1, total_configs=4, total_samples=2, locks={}, P=p)._run()['history']
    assert P.L.train is None and P.L.kqgr is None and set(P.L.heads) == set(HEADS)
    for h in HEADS:
        assert P.L.heads[h].train.accuracy_dev is not None and P.L.heads[h].train.config == 3
        if 'kqgr' in P.L.grid_search.run:
            assert P.L.heads[h].kqgr.kq is not None
    rows = explode_heads(orjson.loads(P.model_dump_json()))
    assert [r['model']['output_layer']['mode'] for r in rows] == HEADS
    for r, h in zip(rows, HEADS):
        assert r['logging']['heads'] is None and r['model']['output_layer']['heads'] is None
        assert r['logging']['train']['accuracy_dev'] == P.L.heads[h].train.accuracy_dev
        assert set(HeadLog.model_fields) <= set(r['logging']), 'every measurement group carried over'
        Params.model_validate(r)                                  # a valid single-mode row


def test_loader_expands_heads_rows(tmp_path):
    import pandas as pd
    from project.boolean_reservoir.code.utils.load_save import _hydrate_params
    base = {'model': {'output_layer': {'mode': 'classification', 'heads': HEADS}},
            'logging': {'heads': {h: {'train': {'accuracy_dev': float(i)}} for i, h in enumerate(HEADS)}}}
    plain = {'model': {'output_layer': {'mode': 'time-series', 'heads': None}}, 'logging': {'train': {'accuracy_dev': 9.0}}}
    df = pd.DataFrame({'i': [0, 1], 'j': [0, 0], 'params_json': [orjson.dumps(base).decode(), orjson.dumps(plain).decode()]})
    out = _hydrate_params(df, fast=True, model_class=Params, keep_params_json=True)
    assert out['i'].tolist() == [0, 0, 1], 'one row per head; single-mode rows untouched'
    modes = [orjson.loads(js)['model']['output_layer']['mode'] for js in out['params_json']]
    accs = [orjson.loads(js)['logging']['train']['accuracy_dev'] for js in out['params_json']]
    assert modes == HEADS + ['time-series'] and accs == [0.0, 1.0, 9.0], 'kept JSON describes its own row'


# ── un-indexed readout groups: temporal dataset.heads ────────────────────────────────────────────
_TEMPORAL_HEADS = [{'task': 'density', 'window': 5, 'delay': 5}, {'task': 'parity', 'window': 3, 'delay': 0}]


def _temporal(tmp_path, monkeypatch, samples=1024, family='bernoulli'):
    monkeypatch.chdir(tmp_path)
    p = generate_param_combinations(load_yaml_config(str(_PROBE)))[0]
    apply_seed(p, 7)
    d = p.model_dump()
    d['dataset'].update(samples=samples, heads=_TEMPORAL_HEADS)
    d['model']['training']['optim'] = {'name': family, 'params': {'alpha': 1e-6 if family == 'bernoulli' else 0.0}}
    return Params.model_validate(d)


def test_temporal_head_targets_are_those_tasks_labels(tmp_path, monkeypatch):
    """Each head's targets are the y of a dataset of that task, window and delay on the same streams,
    so the head equal to the dataset's own task reproduces y on every split."""
    p = _temporal(tmp_path, monkeypatch)
    ds = p.dataset_init_obj.train(p)
    q = p.model_copy(deep=True)
    q.D.task, q.D.window, q.D.delay, q.D.heads = 'parity', 3, 0, None
    other = q.dataset_init_obj.train(q)
    for key in ('', '_dev', '_test'):
        t = ds.data['targets' + key]
        assert t.shape == (len(ds.data['y' + key]), 2, 1)
        assert np.array_equal(ds.data['x' + key], other.data['x' + key]), 'same streams'
        assert np.array_equal(t[:, 0], ds.data['y' + key]) and np.array_equal(t[:, 1], other.data['y' + key])


@pytest.mark.parametrize('family', ['gaussian', 'bernoulli'])
def test_unindexed_group_is_one_readout_per_target(tmp_path, monkeypatch, family):
    """Readout t is fitted and scored on every row against target t: the head that is the dataset's own
    task scores as the y readout (gaussian: the same solve; bernoulli: its own IRLS on the same column)."""
    p = _temporal(tmp_path, monkeypatch, family=family)
    ds = p.dataset_init_obj.train(p)
    groups = p.dataset_init_obj.readout_groups(p)
    out = glm.train_readout_heads(BooleanReservoir(params=_with_mode(p, 'classification')), ds, HEADS, family,
                                  p.M.T.optim.params.alpha, p.accuracy_obj, criterion_fn, _score_splits, groups=groups)
    assert set(out) == {'train', 'targets'}
    for h in HEADS:
        assert set(out['targets'][h]) == {'density_w5_d5', 'parity_w3_d0'}
        own = out['targets'][h]['density_w5_d5']
        for s in ('train', 'dev', 'test'):
            assert own[f'accuracy_{s}'] == pytest.approx(out['train'][h][f'accuracy_{s}'], abs=1e-3), (h, s)
            assert own[f'loss_{s}'] == pytest.approx(out['train'][h][f'loss_{s}'], rel=1e-3), (h, s)


def test_job_logs_named_targets(tmp_path, monkeypatch):
    p = _temporal(tmp_path, monkeypatch, samples=512)
    p.L.out_path = tmp_path / 'run'
    P = BooleanReservoirJob(i=0, j=1, total_configs=1, total_samples=2, locks={}, P=p)._run()['history']
    for h in HEADS:
        assert set(P.L.heads[h].targets) == {'density_w5_d5', 'parity_w3_d0'}
        assert P.L.heads[h].targets['parity_w3_d0'].accuracy_dev is not None
    for r, h in zip(explode_heads(orjson.loads(P.model_dump_json())), HEADS):
        assert r['logging']['targets']['parity_w3_d0']['accuracy_dev'] == P.L.heads[h].targets['parity_w3_d0'].accuracy_dev
        Params.model_validate(r)


def _channels(tmp_path, monkeypatch, c=2, samples=512):
    p = _temporal(tmp_path, monkeypatch, samples=samples)
    d = p.model_dump()
    d['dataset']['dimensions'] = c
    d['model']['input_layer']['features'] = c
    return Params.model_validate(d)


def test_channels_step_t_carries_every_channels_bit_t(tmp_path, monkeypatch):
    """c streams in parallel: model step t holds (channel 0 bit t, ..., channel c-1 bit t), washout included."""
    from benchmark.temporal.temporal_density_parity_dataset import TemporalDataset
    p = _channels(tmp_path, monkeypatch, c=3)
    p.D.washout = 4
    raw = p.dataset_init_obj._create_raw_dataset(p)
    q = p.model_copy(deep=True)
    q.D.update_path()
    streams = TemporalDataset(q.D).x                                  # (m, 1, c, washout + b)
    assert raw.x.shape == (len(streams), streams.shape[-1], 3, 1)
    assert np.array_equal(raw.x[..., 0], streams[:, 0].swapaxes(1, 2))


def test_channels_are_scored_apart(tmp_path, monkeypatch):
    """With c > 1 every head is scored per channel (<head>_c<j>); the channel losses average to the pooled loss."""
    p = _channels(tmp_path, monkeypatch, c=2)
    ds = p.dataset_init_obj.train(p)
    (g,) = p.dataset_init_obj.readout_groups(p)
    fit = lambda grp: glm.train_readout_heads(BooleanReservoir(params=_with_mode(p, 'classification')), ds, HEADS, 'bernoulli',
                                              p.M.T.optim.params.alpha, p.accuracy_obj, criterion_fn, _score_splits, groups=(grp,))
    per, pooled = fit(g), fit(dataclasses.replace(g, per_output=False))
    for h in HEADS:
        assert set(per['targets'][h]) == {f'{n}_c{j}' for n in ('density_w5_d5', 'parity_w3_d0') for j in range(2)}
        for n in ('density_w5_d5', 'parity_w3_d0'):
            for s in ('train', 'dev', 'test'):
                mean = np.mean([per['targets'][h][f'{n}_c{j}'][f'loss_{s}'] for j in range(2)])
                assert mean == pytest.approx(pooled['targets'][h][n][f'loss_{s}'], rel=1e-5), (h, n, s)


def test_capacity_sums_clipped_channel_information():
    from project.temporal.code.capacity import H, capacity, head_of
    assert head_of('density_w3_d5_c12') == 'density_w3_d5' and head_of('parity_w3_d0') == 'parity_w3_d0'
    t = {'density_w3_d5_c0': {'loss_dev': 0.0}, 'density_w3_d5_c1': {'loss_dev': H / 2},
         'density_w3_d5_c2': {'loss_dev': 1.3 * H}, 'parity_w3_d0': {'loss_dev': H}}
    assert capacity(t) == pytest.approx({'density_w3_d5': 1.5, 'parity_w3_d0': 0.0}), 'perfect 1, half 0.5, worse than chance 0'
