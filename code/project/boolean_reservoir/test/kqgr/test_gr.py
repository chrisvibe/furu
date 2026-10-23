import pytest
import numpy as np
from project.boolean_reservoir.code.kqgr_probe import freeze, dedupe, trim_to_n


def create_free_data(m=10, s=1, f=2, b=10):
    return np.random.randint(0, 2, (m, s, f, b), dtype=np.uint8)


@pytest.mark.parametrize("evaluation", ['first', 'last', 'random'])
def test_freeze_shape_unchanged(evaluation):
    """freeze() overwrites in place — never grows the array."""
    tau, b = 3, 10
    rng = np.random.default_rng(0)
    x = create_free_data(b=b)
    out = freeze(x, tau, axis='resolution', evaluation=evaluation, rng=rng)
    assert out.shape == x.shape


@pytest.mark.parametrize("evaluation", ['first', 'last', 'random'])
def test_freeze_identical_bits_across_samples(evaluation):
    """The tau frozen positions must be identical for all m samples."""
    tau, f, b = 3, 2, 10
    rng = np.random.default_rng(0)
    x = create_free_data(f=f, b=b)
    out = freeze(x, tau, axis='resolution', evaluation=evaluation, rng=rng)

    identical_mask = (out == out[0:1]).all(axis=0)  # (s, f, b)
    n_identical = identical_mask.sum().item()
    assert n_identical >= tau * f


def test_freeze_last_positions_identical():
    """'last' mode: last tau bits of each feature identical across samples."""
    tau, f, b = 3, 2, 10
    rng = np.random.default_rng(0)
    x = create_free_data(f=f, b=b)
    out = freeze(x, tau, axis='resolution', evaluation='last', rng=rng)

    for feat in range(f):
        tail = out[:, :, feat, -tau:]
        assert (tail == tail[0:1]).all(), f"Feature {feat}: last {tau} bits must be identical"


def test_freeze_first_positions_identical():
    """'first' mode: first tau bits of each feature identical across samples."""
    tau, f, b = 3, 2, 10
    rng = np.random.default_rng(0)
    x = create_free_data(f=f, b=b)
    out = freeze(x, tau, axis='resolution', evaluation='first', rng=rng)

    for feat in range(f):
        head = out[:, :, feat, :tau]
        assert (head == head[0:1]).all(), f"Feature {feat}: first {tau} bits must be identical"


def test_freeze_steps_axis_last_positions_identical():
    """'steps' axis: last tau steps identical across samples, earlier steps untouched."""
    tau, s, b = 2, 5, 10
    rng = np.random.default_rng(0)
    x = create_free_data(s=s, b=b)
    out = freeze(x, tau, axis='steps', evaluation='last', rng=rng)

    assert np.array_equal(out[:, :s - tau], x[:, :s - tau]), "earlier steps must be untouched"
    tail = out[:, s - tau:]
    assert (tail == tail[0:1]).all(), f"last {tau} steps must be identical across samples"


def test_freeze_zero_is_noop():
    """tau=0 must return x unchanged."""
    rng = np.random.default_rng(0)
    x = create_free_data(b=10)
    out = freeze(x, tau=0, axis='resolution', evaluation='last', rng=rng)
    assert (out == x).all(), "tau=0 should not modify any bits"


def test_freeze_explicit_ref_idx():
    """An explicit ref_idx picks a deterministic, fixed reference row."""
    tau, m, b = 3, 10, 10
    rng = np.random.default_rng(0)
    x = create_free_data(m=m, b=b)
    out = freeze(x, tau, axis='resolution', evaluation='last', rng=rng, ref_idx=0)
    assert (out[:, :, :, -tau:] == x[0:1, :, :, -tau:]).all()


def test_dedupe_removes_freeze_induced_duplicates():
    """Exhaustive bits=10 pool, tau=3: freeze+dedupe collapses to exactly 2**(10-3)=128
    distinct rows — the equivalence this whole design relies on (see kqgr_probe.py
    module docstring)."""
    b, tau = 10, 3
    mask = (1 << np.arange(b - 1, -1, -1))
    ints = np.arange(2 ** b)
    x = ((ints[:, None] & mask[None, :]) != 0).reshape(2 ** b, 1, 1, b).astype(np.uint8)
    rng = np.random.default_rng(0)

    out = freeze(x, tau, axis='resolution', evaluation='last', rng=rng)
    unique, index = dedupe(out)
    assert unique.shape[0] == 2 ** (b - tau)
    assert index.shape[0] == 2 ** (b - tau)


def test_trim_to_n_warns_when_short():
    x = create_free_data(m=10)
    index = np.arange(10)
    rng = np.random.default_rng(0)
    with pytest.warns(UserWarning, match="Only 10 distinct patterns"):
        trimmed, trimmed_idx = trim_to_n(x, index, n=20, rng=rng)
    assert trimmed.shape[0] == 10


def test_trim_to_n_subsamples_when_over():
    x = create_free_data(m=50)
    index = np.arange(50)
    rng = np.random.default_rng(0)
    trimmed, trimmed_idx = trim_to_n(x, index, n=10, rng=rng)
    assert trimmed.shape[0] == 10
    assert trimmed_idx.shape[0] == 10
    # index must stay aligned with the trimmed rows
    assert (x[trimmed_idx] == trimmed).all()
