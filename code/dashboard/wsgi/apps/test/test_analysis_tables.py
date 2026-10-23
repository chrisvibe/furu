"""pytest tests for `load_analysis_tables`: analysis CSVs as one scatter frame."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # code/dashboard, as the other tests

from wsgi.base import load_analysis_tables  # noqa: E402


def _write(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)                       # with the unnamed index column, as the analysis writes it


def test_tables_variants_and_tags(tmp_path):
    gaps = pd.DataFrame({'stage': ['integration', 'selection'], 'R_k_avg': [1.0, 2.0], 'ratio': [1.01, 1.35]})
    _write(tmp_path / '1D/split_gaps.csv', gaps)
    _write(tmp_path / '1D/cell_stat=good/split_gaps.csv', gaps.assign(ratio=[1.0, 1.3]))
    _write(tmp_path / '2D/split_gaps.csv', gaps)
    _write(tmp_path / '2D/ladder.csv', pd.DataFrame({'R_k_avg': [1.0], 'nmse': [0.5]}))
    roots = [({'dims': d}, tmp_path / d) for d in ('1D', '2D', '3D')]   # 3D: missing, skipped
    df, factors = load_analysis_tables(roots, ['split_gaps', 'ladder'])
    assert not any(c.startswith('Unnamed') for c in df.columns)
    assert df.groupby(['dims', 'variant', 'table']).size().to_dict() == {
        ('1D', 'cell_stat=good', 'split_gaps'): 2, ('1D', 'main', 'split_gaps'): 2,
        ('2D', 'main', 'ladder'): 1, ('2D', 'main', 'split_gaps'): 2}
    assert {'stage', 'R_k_avg', 'table', 'variant', 'dims'} <= set(factors)
    assert df.loc[(df['variant'] == 'cell_stat=good'), 'ratio'].tolist() == [1.0, 1.3]


def test_measures_are_not_factors(tmp_path):
    _write(tmp_path / 'a/t.csv', pd.DataFrame({'k': [1.0, 2.0] * 50, 'nmse': [i / 100 for i in range(100)]}))
    _, factors = load_analysis_tables([({}, tmp_path / 'a')], ['t'])
    assert 'k' in factors and 'nmse' not in factors


def test_nothing_found(tmp_path):
    df, factors = load_analysis_tables([({}, tmp_path / 'none')], ['t'])
    assert df.empty and factors == []
