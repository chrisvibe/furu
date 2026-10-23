"""
Verify that recorded history is consistent with the reservoir's LUT rules.

For every `reservoir_layer` step in the history:
  state[t, node] == LUT[node][ msb_first_index(predecessor_states at t-1) ]
  (a reservoir node with no predecessors reads LUT[node][0], a constant)

where msb_first_index runs over the node's own k predecessors, the first in
G.predecessors() order the most-significant bit (lut_index).

This is the fundamental RBN update rule. If it holds, the Julia engine is
correctly implementing the Boolean logic defined by the LUT.
"""
import pytest
import numpy as np
from pathlib import Path

CONFIG = Path(__file__).parent / 'config' / 'sample_model.yaml'


@pytest.fixture(scope='module')
def loaded():
    from project.boolean_reservoir.code.parameter import load_yaml_config
    from project.boolean_reservoir.code.reservoir import BooleanReservoir, BatchedTensorHistoryWriter

    params = load_yaml_config(CONFIG)
    dataset = params.dataset_init_obj.train(params)
    x = dataset.data['x'][:4]  # 4 samples is enough to exercise every LUT entry path

    model = BooleanReservoir(params=params)
    model.record = True
    model.history = BatchedTensorHistoryWriter(
        save_path=Path('/tmp/boolean_reservoir/test/lut_consistency'),
        persist_to_disk=False,
    )
    model.eval()
    model(x)
    model.flush_history()

    _, history, expanded_meta, _ = model.history.reload_history()
    return model, history, expanded_meta


def test_node_count(loaded):
    """History columns == N_total nodes in the model."""
    model, history, _ = loaded
    assert history.shape[1] == model.N_total, (
        f'History has {history.shape[1]} columns but model.N_total={model.N_total}')


def test_phases_present(loaded):
    """Expected phases appear in the history."""
    _, _, meta = loaded
    phases = set(meta['phase'].unique())
    assert 'init'            in phases, f'Missing init phase. Got: {phases}'
    assert 'reservoir_layer' in phases, f'Missing reservoir_layer phase. Got: {phases}'


def test_init_state_zeros(loaded):
    """With `init: zeros`, the recorded init step must be all-zero for every node."""
    model, history, expanded_meta = loaded

    agg    = expanded_meta[expanded_meta['sample_id'] == 0]
    states = history[agg.index.values]
    phases = agg['phase'].values

    init_rows = [i for i, ph in enumerate(phases) if ph == 'init']
    assert init_rows, 'No init phase found for sample_id=0'

    for t in init_rows:
        assert states[t].sum() == 0, (
            f'Init state (step {t}) is not all-zero: '
            f'{np.nonzero(states[t])[0].tolist()} are set')


def test_reservoir_layer_lut_consistency(loaded):
    """
    For every `reservoir_layer` step t, each reservoir node n must satisfy:

        state[t, n]  ==  LUT[n][ sum(state[t-1, pred_j] * 2^j) ]

    where pred_j are node n's predecessors in predecessor order (MSB = first pred).
    This is the fundamental RBN update rule.
    """
    model, history, expanded_meta = loaded

    G         = model.graph
    lut       = model.lut
    res_nodes = list(range(model.input_slice.stop, model.res_slice.stop))

    agg    = expanded_meta[expanded_meta['sample_id'] == 0]
    states = history[agg.index.values]
    phases = agg['phase'].values

    res_steps = [i for i, ph in enumerate(phases) if ph == 'reservoir_layer']
    assert res_steps, 'No reservoir_layer steps found for sample_id=0'

    from project.boolean_reservoir.code.lut import lut_index

    def _lut_idx(n: int, prev) -> int:
        preds = list(G.predecessors(n))
        return lut_index([int(prev[p]) for p in preds])

    failures = []
    for t in res_steps:
        assert t > 0, f'reservoir_layer at t=0 has no previous state'
        for n in res_nodes:
            # an in-degree-0 reservoir node reads its 0-ary table: idx 0, a constant
            # (BooleanReservoir.find_clamped_nodes)
            idx      = _lut_idx(n, states[t - 1])
            expected = int(lut.row(n)[idx])
            actual   = int(states[t, n])
            if actual != expected:
                preds = list(G.predecessors(n))
                failures.append(
                    f'  node={n} t={t} (prev={phases[t-1]}): '
                    f'LUT[{n}][{idx}]={expected} != actual={actual} | '
                    f'preds={preds} states_t-1={[int(states[t-1,p]) for p in preds]}'
                )

    assert not failures, (
        f'{len(failures)} LUT violations across '
        f'{len(res_steps)} reservoir_layer steps:\n' + '\n'.join(failures[:20])
    )
    print(f'\n  Checked {len(res_nodes) * len(res_steps):,} '
          f'(node × step) pairs — all consistent.')



def test_intermediate_states_are_distinct(loaded):
    """History phases must not all be identical rows.

    Regression guard for the history_buffer bug: if _forward_julia records
    states_parallel (final state) instead of history_buffer[i], every phase
    entry gets the same state and this test fails with a clear message.
    """
    model, history, expanded_meta = loaded

    agg = expanded_meta[expanded_meta['sample_id'] == 0]
    states = history[agg.index.values]
    phases = agg['phase'].values

    init_idx = next((i for i, ph in enumerate(phases) if ph == 'init'), None)
    res_idx  = next((i for i, ph in enumerate(phases) if ph == 'reservoir_layer'), None)

    assert init_idx is not None and res_idx is not None, \
        "Need both init and reservoir_layer phases to compare"

    assert not np.array_equal(states[init_idx], states[res_idx]), (
        "Init state equals a reservoir_layer state — all history rows are identical. "
        "history_buffer regression: _forward_julia is recording states_parallel "
        "(final state) instead of history_buffer[i] (intermediate state)."
    )
