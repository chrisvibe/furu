import numpy as np
import pandas as pd
import pytest

from project.boolean_reservoir.code.stats import rho_within_designs, rho_within_groups, seed_filter, spearman_by


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 4 * 3 * 8                                   # 4 designs x 3 heads x 8 seeds
    df = pd.DataFrame({'head': np.repeat(['a', 'b', 'c'], n // 3), 'design': np.tile(np.repeat(range(4), 8), 3),
                       'x': rng.integers(0, 5, n).astype(float), 'acc': rng.random(n)})
    df['acc_test'] = df['acc'] + rng.normal(0, 0.01, n)
    df.loc[(df['head'] == 'c') & (df['design'] == 0), 'x'] = 1.0   # one constant design: no rho
    return df


def test_spearman_by_matches_pandas(frame):
    s = spearman_by(frame, ['head', 'design'], 'x', 'acc')
    ref = frame.groupby(['head', 'design'])[['x', 'acc']].apply(lambda g: g['x'].corr(g['acc'], method='spearman'))
    assert (s['n'] == 8).all()
    np.testing.assert_allclose(s['rho'], ref, equal_nan=True)
    assert s['rho'].isna().sum() == 1


def test_rho_within_groups_weights_by_rows(frame):
    out = rho_within_groups(frame, ['head'], ['design'], ['x'], 'acc', min_n=8).set_index('head')
    s = spearman_by(frame, ['head', 'design'], 'x', 'acc').dropna()
    assert out.loc['c', 'groups'] == 3 and out.loc['c', 'n'] == 24          # the constant design drops out
    assert out.loc['a', 'rho'] == pytest.approx(s.loc['a', 'rho'].mean())   # equal group sizes: plain mean
    assert rho_within_groups(frame, ['head'], ['design'], ['x'], 'acc', min_n=9).empty


def test_rho_within_designs_and_seed_filter(frame):
    gains = seed_filter(frame, ['head', 'design'], 'x', 'acc_test')
    g = frame[(frame['head'] == 'a') & (frame['design'] == 1)]
    picked = g.loc[g['x'] == g['x'].max(), 'acc_test'].mean()
    assert gains.loc[('a', 1), 'gain_x'] == pytest.approx(picked - g['acc_test'].mean())
    assert gains.loc[('a', 1), 'gain_oracle'] == pytest.approx(g['acc_test'].max() - g['acc_test'].mean())
    median = seed_filter(frame, ['head', 'design'], 'x', 'acc_test', base='median').loc[('a', 1)]
    assert median['gain_oracle'] == pytest.approx(g['acc_test'].max() - g['acc_test'].median())

    out = rho_within_designs(frame, ['head'], ['design'], ['x'], 'acc', min_n=6, gains=gains).set_index('head')
    rho = spearman_by(frame, ['head', 'design'], 'x', 'acc')['rho']
    assert out.loc['c', 'designs'] == 4                                     # counted, though one has no rho
    assert out.loc['c', 'rho'] == pytest.approx(rho.loc['c'].mean())
    assert out.loc['c', 'rho_se'] == pytest.approx(rho.loc['c'].std() / np.sqrt(3))
    assert out.loc['a', 'gain_x'] == pytest.approx(gains.loc['a', 'gain_x'].mean())
