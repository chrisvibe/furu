"""pytest tests: a saved view's axes survive the browser load path.

Dash 4's dcc.Dropdown drops a value that is not among its options. A view whose y is a computed field
(`fields-expr`) therefore opened with no y in the browser ("Select X and Y axes"), while its export,
which never uses the dropdowns, rendered fine. View loading now sets the options with the values.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # code/dashboard, as the other tests

from dashboards.scatter import ScatterDashboard  # noqa: E402

CODE = Path(__file__).resolve().parents[4]                     # the code/ workspace (out/ is linked there)
# (site, its views dir): the analysis app is one kind deployed per paper (DASHBOARD_SITE);
# the site's tables/roots live in wsgi.apps.analysis._SITES.
ANALYSIS_SITES = [('hegg', CODE / 'out/dashboard/hegg/analysis'),
                  ('furu', CODE / 'out/dashboard/furu/analysis')]


def _values(opts):
    return {o['value'] for o in opts if not o.get('disabled')}


def test_computed_y_is_offered_with_the_view():
    df = pd.DataFrame({'k': [1.0, 2.0, 3.0] * 4, 'stage': list('ab' * 6), 'r': range(12)})
    d = ScatterDashboard(df, ['k', 'stage'], renderer='svg')
    assert _values(d._view_axis_options(None)) == {'k', 'r', 'stage'}   # numeric columns, then text factors
    opts = d._view_axis_options("r2 = r * 2\nhalf = np.where(stage == 'a', r, np.nan)")
    assert {'r2', 'half'} <= _values(opts)


@pytest.mark.parametrize('site, views_dir', ANALYSIS_SITES, ids=[s for s, _ in ANALYSIS_SITES])
def test_every_analysis_view_opens_with_its_axes(site, views_dir):
    """Each saved analysis view: x and every y are among the options the load sets (the browser path),
    its filters name values the tables have, and it reconstructs (the export path)."""
    import importlib
    from wsgi.base import load_analysis_tables
    from dashboards.utils import categorize_object_columns
    from dashboards.theme import register_templates
    if not views_dir.is_dir():
        pytest.skip('no views in this checkout')
    # The module reads DASHBOARD_SITE at import and exposes both papers' configs in _SITES;
    # the env only has to make the import possible, the test drives _SITES directly.
    os.environ.setdefault('DASHBOARD_SITE', site)
    m = importlib.import_module('wsgi.apps.analysis')
    site_cfg = m._SITES[site]
    df, factors = load_analysis_tables([(t, CODE / r) for t, r in m.site_roots(site_cfg)], site_cfg['tables'])
    if df.empty:
        pytest.skip('no analysis tables in this checkout')
    categorize_object_columns(df)
    register_templates()
    d = ScatterDashboard(df, factors, renderer='svg', initial_views_dir=str(views_dir))
    views = sorted(views_dir.glob('*.json'))
    assert views
    for path in views:
        v = json.loads(path.read_text())
        opts = _values(d._view_axis_options(v.get('fields-expr')))
        assert v['x-axis'] in opts, path.name
        assert set([v['y-axis']] if isinstance(v['y-axis'], str) else v['y-axis']) <= opts, path.name
        fig = d._reconstruct_figure(v, theme_override=['light'])   # raises ViewDataMismatch on stale filters
        assert len(fig.data), path.name
