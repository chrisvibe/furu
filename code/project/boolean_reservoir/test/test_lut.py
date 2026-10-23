"""The compact LUT (lut.Lut, lut.lut_random): drawn without the unused entries, yet entry for
entry the tables of the column-major layout it defines, so a node's table never depends
on any other node's degree."""
import numpy as np
import pytest

from project.boolean_reservoir.code.lut import Lut, lut_random, lut_sensitivity
from project.boolean_reservoir.code.utils.utils import make_rng


def _column_major(n, max_k, p, rng):
    """The layout lut_random's docstring defines, drawn whole: entry (i, c) = double c*n + i."""
    return np.ascontiguousarray(rng.random((2 ** max_k, n)).T < p, dtype=np.uint8)


@pytest.mark.parametrize('trial', range(40))
def test_draws_exactly_the_layout(trial):
    r = np.random.default_rng(trial)
    n, K = int(r.integers(1, 40)), int(r.integers(0, 9))
    deg = r.integers(0, K + 1, n)
    deg[r.integers(0, n)] = K
    p = float(r.random())
    dense = _column_major(n, K, p, make_rng(trial, 'lut'))
    lut = lut_random(deg, p=p, rng=make_rng(trial, 'lut'), chunk_doubles=int(r.integers(1, 200)))
    for i in range(n):
        assert np.array_equal(lut.row(i), dense[i, :2 ** deg[i]])


def test_wider_table_keeps_the_narrower_as_its_first_half():
    """One more input (e.g. a query edge, the new MSB) leaves the node's old table and
    every other node's table as they were."""
    deg = np.array([2, 0, 3, 1, 2])
    more = deg.copy(); more[2] += 2
    a, b = (lut_random(d, rng=make_rng(0, 'lut')) for d in (deg, more))
    for i in range(len(deg)):
        assert np.array_equal(b.row(i)[:len(a.row(i))], a.row(i))


def test_from_rows_and_coerce():
    rows = np.array([[1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 1, 1]], dtype=np.uint8)
    mask = np.array([[1, 1], [0, 0], [1, 0]], dtype=bool)
    lut = Lut.from_rows(rows, mask)
    assert [r.tolist() for r in map(lut.row, range(3))] == [[1, 0, 1, 1], [0], [1, 1]]
    assert Lut.coerce({'flat': lut.flat, 'offsets': lut.offsets}, mask) == lut == Lut.coerce(rows, mask)


def test_legacy_padded_checkpoint_loads(tmp_path):
    """A checkpoint saved before the compact LUT holds lut.npy, one padded row per node."""
    from project.boolean_reservoir.code.parameter import load_yaml_config
    from project.boolean_reservoir.code.reservoir import BooleanReservoir
    P = load_yaml_config('config/test/boolean_reservoir/networkx_wiring.yaml')
    P.L.out_path = str(tmp_path)
    P.L.history.record = False
    P.M.R.seed = 0
    model = BooleanReservoir(P)
    model.save()
    ckpt = model.L.last_checkpoint
    width = 2 ** model.adj_list.shape[1]
    padded = np.zeros((len(model.lut), width), dtype=np.uint8)
    for n in range(len(model.lut)):
        padded[n, :len(model.lut.row(n))] = model.lut.row(n)
    (ckpt / 'lut.npz').unlink()
    np.save(ckpt / 'lut.npy', padded)
    assert BooleanReservoir(load_path=ckpt).lut == model.lut


def test_sensitivity_of_known_functions():
    """Constant, copy of the first input (MSB), XOR and AND: per-position flip rates, MSB-first."""
    rows = np.array([
        [0, 0, 0, 0],   # constant: insensitive to both inputs
        [0, 0, 1, 1],   # copy of the MSB (first input): 1 on it, 0 on the other
        [0, 1, 1, 0],   # XOR: parity, flips on every input
        [0, 0, 0, 1],   # AND: each input matters half the time
    ], dtype=np.uint8)
    s = lut_sensitivity(Lut.from_rows(rows, np.ones((4, 2), dtype=bool)))
    assert np.allclose(s, [[0, 0], [1, 0], [1, 1], [0.5, 0.5]])


def test_sensitivity_pads_past_a_nodes_degree():
    lut = Lut.from_rows(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.uint8),
                        np.array([[True, False], [True, True]]))   # degrees 1 and 2
    s = lut_sensitivity(lut)
    assert s[0, 0] == 1 and np.isnan(s[0, 1]) and np.allclose(s[1], [0.5, 0.5])
