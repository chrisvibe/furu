"""Equivalence tests for the encode-after-trim fast path in build_kqgr_probe.

The fast path runs freeze/dedupe/trim on the RAW pool and encodes only the surviving
rows, instead of materialising the whole pool at full redundancy. It is only taken when
tau_axis='steps' (freeze touches the step axis, which encoding leaves alone) AND the raw
input is already binary (so the encoder is a per-row bijection and dedupe agrees raw vs
encoded). These tests pin that the output is byte-identical to the original ordering.
"""
import numpy as np
import pytest

from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.utils.param_utils import LazyParamCombinations
from project.boolean_reservoir.code.kqgr_probe import build_kqgr_probe

TEMPORAL_CFG = 'config/temporal/density/grid_search/design_choices/all_heterogeneous.yaml'
# continuous (path-integration) input: project/path_integration/test/test_kqgr_pi.py


def _compare(Pu, independent=None):
    if independent is not None:
        Pu.D.independent = independent
    n_nodes, seed = Pu.M.R.n_nodes, Pu.D.seed
    kq_f, gr_f, _ = build_kqgr_probe(Pu, Pu.dataset_init_obj, n_nodes,
                                     np.random.default_rng(seed), fast=True)
    kq_s, gr_s, _ = build_kqgr_probe(Pu, Pu.dataset_init_obj, n_nodes,
                                     np.random.default_rng(seed), fast=False)
    assert kq_f.shape == kq_s.shape and kq_f.dtype == kq_s.dtype
    assert gr_f.shape == gr_s.shape and gr_f.dtype == gr_s.dtype
    assert np.array_equal(kq_f, kq_s), 'KQ differs between fast and original path'
    assert np.array_equal(gr_f, gr_s), 'GR differs between fast and original path'


def _universe(combos, idx):
    p = combos[int(idx)]
    name = next(iter(p.multiverse_overrides or {}), None)
    return getattr(p.U, name) if name else None


@pytest.mark.parametrize('frac', [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0])
@pytest.mark.parametrize('independent', [False, True])
def test_temporal_fast_path_equivalence(frac, independent):
    """Binary input + tau_axis='steps' takes the fast path; must be byte-identical."""
    P0 = load_yaml_config(TEMPORAL_CFG)
    P0.dataset.bits = 10  # keep 2**width pools small enough for a unit test
    combos = LazyParamCombinations(P0)
    Pu = _universe(combos, round(frac * (len(combos) - 1)))
    if Pu is None or getattr(Pu.D, 'tau_axis', '') != 'steps':
        pytest.skip('not a steps-axis universe')
    _compare(Pu, independent)

