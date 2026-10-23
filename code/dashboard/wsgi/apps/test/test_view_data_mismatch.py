"""pytest tests for `ViewDataMismatch`: a saved view whose static filters name values the app's data
does not have must fail loudly (409 from the export route), never render a plausible wrong figure.

The case behind it (2026-10-12): path-integration booted before the 2D main grids finished, so the
pi2d views exported a black "No data" plot while the UI, whose dropdowns drop unknown values, showed
the 1D data under the 2D view's name.
"""
import json
import sys
from pathlib import Path

import dash
import pandas as pd
import pytest
from dash import html

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # code/dashboard, as the other tests

from dashboards import utils  # noqa: E402
from dashboards.view_filters import apply_view_filters, missing_filter_values  # noqa: E402

DF = pd.DataFrame({'grid': pd.Categorical(['1D', '1D', 'pilot']), 'n': [2048, 2048, 512],
                   'loops': [0.0, 0.1, 0.0]})


def _view(**filters):
    return {'_extras': {'factor_filters': filters}}


def test_all_present_filters_rows():
    f, _ = apply_view_filters(DF, _view(grid=['1D'], n=['2048'], loops=['0.0'], unused=[]))
    assert len(f) == 1


def test_missing_value_and_missing_column_are_reported():
    assert missing_filter_values(DF, {'grid': ['1D', '2D'], 'n': ['2048'], 'absent': ['x']}) == \
        {'grid': ['2D'], 'absent': ['x']}


def test_apply_view_filters_raises_on_missing_value():
    with pytest.raises(utils.ViewDataMismatch) as e:
        apply_view_filters(DF, _view(grid=['2D']))
    assert e.value.missing == {'grid': ['2D']}


def test_export_route_answers_409(tmp_path):
    (tmp_path / 'v.json').write_text(json.dumps(_view(grid=['2D'])))

    def reconstruct(view_data, width, height, theme_override):
        return apply_view_filters(DF, view_data)

    app = dash.Dash('mismatch-test')
    app.layout = html.Div('x')
    utils.register_export_route(app, reconstruct, safe_roots=[str(tmp_path)], default_search_dirs=[str(tmp_path)])
    r = app.server.test_client().get('/?v&format=png')
    assert r.status_code == 409
    assert "'grid': ['2D']" in r.get_data(as_text=True)
