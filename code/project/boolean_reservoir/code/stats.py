"""
Does a reservoir measure (KQ, GR, delta) rank reservoirs by their score? The within-design tests, shared by the papers.
The caller picks the rows (e.g. working designs only) and the groups;
never pool across k, which sets both KQ and GR.

  spearman_by          Spearman rho of x against y within each group, from average ranks; the building block
  rho_within_groups    test B: rho across the reservoirs of each group (design factors fixed except the ones the caller
                       leaves out), groups with >= min_n rows, mean per `by` weighted by group size
  rho_within_designs   test C: rho across the seeds of each design (one setting, no centring), designs with >= min_n
                       seeds, mean and standard error per `by`
  seed_filter          per design: the score gain of picking the seed with the highest `pick` (ties averaged) over the
                       design's typical seed (its mean or median), and of the best seed (the oracle)
"""
import numpy as np
import pandas as pd


def spearman_by(df: pd.DataFrame, keys: list[str], x: str, y: str) -> pd.DataFrame:
    """Spearman rho of x against y within each group of keys (average ranks), and the group size. NaN where either is
    constant."""
    by = [df[k] for k in keys]
    a, b = df.groupby(by, observed=True)[x].rank(), df.groupby(by, observed=True)[y].rank()
    s = pd.DataFrame({'a': a, 'b': b, 'ab': a * b, 'aa': a * a, 'bb': b * b}).groupby(by, observed=True).agg(
        n=('a', 'size'), a=('a', 'sum'), b=('b', 'sum'), ab=('ab', 'sum'), aa=('aa', 'sum'), bb=('bb', 'sum'))
    va, vb = s['aa'] - s['a'] ** 2 / s['n'], s['bb'] - s['b'] ** 2 / s['n']
    rho = (s['ab'] - s['a'] * s['b'] / s['n']) / np.sqrt(va * vb)
    return pd.DataFrame({'n': s['n'], 'rho': rho.where((va > 1e-9) & (vb > 1e-9))})


def rho_within_groups(df: pd.DataFrame, by: list[str], groups: list[str], metrics: list[str], score: str,
                      min_n: int) -> pd.DataFrame:
    """Test B. One row per `by` x metric: groups (with a defined rho and >= min_n rows), n (their rows), rho (weighted
    by n)."""
    rows = []
    for m in metrics:
        s = spearman_by(df, [*by, *groups], m, score).dropna()
        s = s[s['n'] >= min_n].reset_index()
        rows += [{**dict(zip(by, key)), 'metric': m, 'groups': len(g), 'n': g['n'].sum(),
                  'rho': np.average(g['rho'], weights=g['n'])} for key, g in s.groupby(by)]
    return pd.DataFrame(rows)


def seed_filter(df: pd.DataFrame, design: list[str], pick: str, score: str, base: str = 'mean') -> pd.DataFrame:
    """Per design: gain_<pick>, the mean score of the seeds with the highest pick minus the design's `base` ('mean' or
    'median') score; gain_oracle, the best seed's score minus the same."""
    by = [df[k] for k in design]
    s = df.groupby(by, observed=True)[score]
    typical = s.agg(base)
    top = df[df[pick] == df.groupby(by, observed=True)[pick].transform('max')]
    return pd.DataFrame({f'gain_{pick}': top.groupby([top[k] for k in design], observed=True)[score].mean() - typical,
                         'gain_oracle': s.max() - typical})


def rho_within_designs(df: pd.DataFrame, by: list[str], design: list[str], metrics: list[str], score: str,
                       min_n: int, gains: pd.DataFrame | None = None) -> pd.DataFrame:
    """Test C. One row per `by` x metric: designs (with >= min_n seeds; a constant side gives no rho but still counts),
    rho (mean over the designs), rho_se (its standard error), and the mean of each column of `gains` (seed_filter's on
    by + design) over the same designs."""
    rows = []
    for m in metrics:
        s = spearman_by(df, [*by, *design], m, score)
        s = s[s['n'] >= min_n]
        if gains is not None:
            s = s.join(gains)
        s = s.reset_index()
        extra = [] if gains is None else list(gains.columns)
        rows += [{**dict(zip(by, key)), 'metric': m, 'designs': len(g), 'rho': g['rho'].mean(),
                  'rho_se': g['rho'].std() / np.sqrt(g['rho'].count()), **g[extra].mean().to_dict()}
                 for key, g in s.groupby(by)]
    return pd.DataFrame(rows)
