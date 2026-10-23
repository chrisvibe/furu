"""Results must depend on seeds alone — not on cache state, call order, or on how
much global RNG something else consumed first.

Regression tests for the bug found on 2026-08-14 (negative kqgr_delta investigation):

    set_seed(D.seed)                       # global RNG := S0
    if data_path.exists() and not generate_data:
        load_data()                        # consumes NO rng      -> still S0
    else:
        generate_data(...)                 # np.random.shuffle    -> now S1
        save_data()
    if shuffle:
        shuffle_data()                     # np.random.permutation <- S0 or S1!

Seeding fixed the RNG's starting state but not how much was consumed before the
draw that mattered, so the row order — and therefore which rows became the KQ/GR
probe — depended on whether the .npz cache happened to exist. The same grid-search
job gave kq=306/gr=97 on a cold disk and kq=314/gr=435 on a warm one.
"""
from pathlib import Path

import numpy as np
import pytest

from benchmark.temporal.temporal_density_parity_dataset import (
    TemporalDatasetParams,
    TemporalDataset,
)


def _params(tmp_path, **overrides):
    kwargs = dict(
        task='density',
        bits=10,
        window=5,
        delay=2,
        dimensions=1,
        samples=64,
        sampling_mode='exhaustive',
        shuffle=True,
        seed=0,
    )
    kwargs.update(overrides)
    D = TemporalDatasetParams(**kwargs)
    # `path` is derived by a model_validator from the dataset fields, so passing it
    # to the constructor has no effect — override afterwards to keep the .npz out
    # of the repo's shared data/ cache.
    D.path = Path(tmp_path) / 'ds.npz'
    return D


def test_shuffle_identical_across_generate_and_load_branches(tmp_path):
    """The .npz cache existing must not change the row order.

    First construction generates and saves; the second finds the cache and loads.
    Before the fix the generate branch consumed global RNG that the load branch did
    not, so shuffle_data drew a different permutation on each path.
    """
    generated = TemporalDataset(_params(tmp_path, generate_data=True))
    assert (tmp_path / 'ds.npz').exists(), 'expected the first build to write the cache'
    loaded = TemporalDataset(_params(tmp_path, generate_data=False))

    assert np.array_equal(generated.x, loaded.x), 'row order differs between generate and load'
    assert np.array_equal(generated.y, loaded.y)


def test_dataset_unaffected_by_prior_global_rng_consumption(tmp_path):
    """Whatever ran earlier in the process must not shift the dataset.

    Emulates a reused grid-search worker: jobs run back to back and previously
    shared one global stream.
    """
    first = TemporalDataset(_params(tmp_path / 'a', generate_data=True))
    np.random.rand(12345)          # something else consumes the global RNG
    second = TemporalDataset(_params(tmp_path / 'b', generate_data=True))

    assert np.array_equal(first.x, second.x)
    assert np.array_equal(first.y, second.y)


def test_different_seeds_still_differ(tmp_path):
    """Guard against 'fixed' meaning 'constant' — seed must still control the draw."""
    a = TemporalDataset(_params(tmp_path / 'a', generate_data=True, seed=0))
    b = TemporalDataset(_params(tmp_path / 'b', generate_data=True, seed=1))
    assert not np.array_equal(a.x, b.x), 'different seeds produced an identical dataset'


# ---------------------------------------------------------------------------
# model construction: per-purpose streams must be independent
# ---------------------------------------------------------------------------

CONFIG = 'config/temporal/density/grid_search/design_choices/snyder.yaml'


def _single_model_params():
    from project.boolean_reservoir.code.parameter import load_yaml_config
    from project.boolean_reservoir.code.utils.param_utils import LazyParamCombinations
    P = LazyParamCombinations(load_yaml_config(CONFIG))[0]
    P.M.I.seed = P.M.R.seed = P.M.O.seed = 7
    return P


def test_partial_load_dict_does_not_shift_lut_or_init_state():
    """Supplying only the graph must not change the LUT or the initial state.

    load_or_generate skips a generator when its key is present, so with one shared
    global stream a partial load left the stream at a different position and the LUT
    silently changed. Separate per-purpose streams make it structurally impossible.
    """
    from project.boolean_reservoir.code.reservoir import BooleanReservoir
    pytest.importorskip('networkx')

    full = BooleanReservoir(_single_model_params())
    partial = BooleanReservoir(_single_model_params(), load_dict={'graph': full.graph})

    assert full.lut == partial.lut, (
        'LUT shifted when only the graph was loaded'
    )
    assert np.array_equal(np.asarray(full.initial_states), np.asarray(partial.initial_states)), (
        'initial state shifted when only the graph was loaded'
    )


def test_model_unaffected_by_prior_global_rng_consumption():
    """Two identically-seeded models must match regardless of ambient RNG state."""
    from project.boolean_reservoir.code.reservoir import BooleanReservoir
    pytest.importorskip('networkx')

    a = BooleanReservoir(_single_model_params())
    np.random.rand(9999)
    b = BooleanReservoir(_single_model_params())

    assert a.lut == b.lut
    assert np.array_equal(np.asarray(a.initial_states), np.asarray(b.initial_states))
    assert np.array_equal(a.adj_list, b.adj_list)
