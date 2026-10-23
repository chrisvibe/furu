"""provenance.py: content seeds, reservoir_id, and the RESULTS_VERSION tripwire.

The tripwire runs small grid jobs (a temporal job with two readout heads, the same with KQ/GR at furu's probe setting,
and a path-integration job with train + KQ/GR where its config is published, so not in furu's tree) and compares what
they build and measure with results_version.json. When it fails, results changed: bump
provenance.RESULTS_VERSION (old rows then no longer merge with new ones) and re-pin with
    PIN_RESULTS=1 python -m pytest project/boolean_reservoir/test/test_provenance.py
Re-pinning without a bump is refused.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from project.boolean_reservoir.code import provenance
from project.boolean_reservoir.code.parameter import Params, load_yaml_config
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.train_model_parallel import BooleanReservoirJobFactory
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
from project.boolean_reservoir.code.utils.utils import generate_unique_seed

_CODE = Path(__file__).resolve().parents[3]
PIN = Path(__file__).parent / 'results_version.json'


def _kqgr(d):
    d['logging']['grid_search']['run'] = ['train', 'kqgr']
    d['multiverse_overrides'] = {'train': {'dataset': {'tau': 5, 'tau_mode': 'steal', 'evaluation': 'last',
                                                       'tau_axis': 'steps'}}}


JOBS = {name: job for name, job in {   # name: (config, samples, edit of the loaded config)
    'pi_kqgr': ('config/test/path_integration/1D/grid_search/kqgr_steps_axis.yaml', 300, None),
    'temporal_heads': ('config/test/temporal/density_w5_d5.yaml', 1024, None),
    'temporal_kqgr': ('config/test/temporal/density_w5_d5.yaml', 1024, _kqgr),
}.items() if (_CODE / job[0]).exists()}


def _combos(name, samples=None):
    config, n, edit = JOBS[name]
    P = load_yaml_config(str(_CODE / config))
    if edit:
        P = _variant(P, edit)
    combos = list(generate_param_combinations(P))
    d = combos[0].model_dump()
    d['dataset']['samples'] = samples or n
    return P, [Params.model_validate(d), *combos[1:]]


def _variant(p, edit):
    d = p.model_dump()
    edit(d)
    return Params.model_validate(d)


def test_content_seed_follows_the_reservoir_only():
    P, combos = _combos('temporal_heads')
    G = P.L.grid_search
    p = combos[0]
    same = [_variant(p, lambda d: d['model']['training']['optim']['params'].update(alpha=0.5)),
            _variant(p, lambda d: d['model']['output_layer'].update(heads=['classification'], mode='classification')),
            _variant(p, lambda d: d['dataset'].update(delay=3))]
    for q in same:
        assert provenance.content_seed(G.seed, q, 0) == provenance.content_seed(G.seed, p, 0)
    seeds = {provenance.content_seed(G.seed, q, j) for q in combos for j in range(2)}
    assert len(seeds) == 2 * len(combos), 'every reservoir content x sample gets its own seed'


def test_factory_seeding_and_stamp():
    P, combos = _combos('temporal_heads')
    P.L.grid_search.seed = 3
    f = BooleanReservoirJobFactory(P, combos)
    by_index = f._override_parameter(P, 5, 2)
    assert by_index.M.R.seed == generate_unique_seed(3, 5, 2), "'index' (the default) keeps the old seeds"
    assert by_index.L.reservoir_id == provenance.reservoir_id(by_index)
    assert by_index.L.results_version == provenance.RESULTS_VERSION
    P.L.grid_search.seeding = 'content'
    by_content = f._override_parameter(P, 5, 2)
    assert by_content.M.R.seed == by_content.M.I.seed == provenance.content_seed(3, combos[5], 2)
    reordered = BooleanReservoirJobFactory(P, combos[::-1])._override_parameter(P, len(combos) - 6, 2)
    assert reordered.L.reservoir_id == by_content.L.reservoir_id, 'the same reservoir wherever it sits in the grid'


def _h(*arrays) -> str:
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(f'{a.dtype}{a.shape}'.encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


def _print(name, tmp_path, monkeypatch) -> dict:
    """What one job builds (construction hashes, reservoir_id) and measures (rounded accuracies, KQ/GR ranks)."""
    monkeypatch.chdir(tmp_path)
    P, combos = _combos(name)
    P.L.grid_search.seed = 11
    job = BooleanReservoirJobFactory(P, combos)(0, 0, len(combos), 1, {})
    job.P.L.out_path = tmp_path / name
    model = BooleanReservoir(job.P)
    out = {'reservoir_id': job.P.L.reservoir_id,
           'build': _h(model.adj_list, model.adj_list_mask, model.lut.flat, model.lut.offsets, model.initial_states,
                       model.w_bi, model.input_nodes)}
    L = job._run()['history'].L
    groups = {h: v for h, v in (L.heads or {}).items()} or {'own': L}
    for h, g in groups.items():
        for m in ('accuracy', 'loss'):
            out[f'{h}.{m}'] = [round(getattr(g.train, f'{m}_{s}'), 4) for s in ('train', 'dev', 'test')]
        if g.kqgr is not None:
            out[f'{h}.kqgr'] = [g.kqgr.kq, g.kqgr.gr]
    return out


def test_results_unchanged_or_version_bumped(tmp_path, monkeypatch):
    now = {name: _print(name, tmp_path, monkeypatch) for name in JOBS}
    pinned = json.loads(PIN.read_text()) if PIN.exists() else None
    if os.environ.get('PIN_RESULTS'):
        assert pinned is None or provenance.RESULTS_VERSION > pinned['results_version'], \
            'bump provenance.RESULTS_VERSION before re-pinning'
        PIN.write_text(json.dumps({'results_version': provenance.RESULTS_VERSION, 'jobs': now}, indent=1) + '\n')
        return
    assert pinned is not None, f'no pin: run with PIN_RESULTS=1 to write {PIN.name}'
    if pinned['results_version'] != provenance.RESULTS_VERSION:
        pytest.fail(f'RESULTS_VERSION is {provenance.RESULTS_VERSION}, pinned {pinned["results_version"]}: '
                    're-pin with PIN_RESULTS=1')
    changed = {f'{n}.{k}': (pinned['jobs'][n].get(k), v) for n in now for k, v in now[n].items()
               if pinned['jobs'].get(n, {}).get(k) != v}
    assert not changed, (f'results changed (pinned, now): {changed}. If intended, bump provenance.RESULTS_VERSION and '
                         're-pin with PIN_RESULTS=1; old rows then no longer merge with new ones.')
