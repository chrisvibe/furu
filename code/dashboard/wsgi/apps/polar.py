"""
Polar views over the design grids: one app kind, deployed once per paper. The container's DASHBOARD_SITE picks the
paper (hegg | furu; unset = hegg, as the hegg deploy predates it), i.e. which grids load and which views folder is home.

furu's design grid is read as the furu `design` app reads it: one row per job x readout x head (8
delayed heads). Filter `readout` and `target` first; the per-head metric is accuracy_dev. The second furu entry is the
design means (cells table: acc = mean dev accuracy over a design's seeds; floor = the head's floor, a reference).
"""
import os
import sys
from pathlib import Path

from dash import Dash, html

from wsgi.base import _filter_existing
from dashboards.polar import create_polar_dashboard, _config_label
from dashboards.utils import VIEWS_ROOT, FURU_VIEWS_ROOT

SITE = os.environ.get('DASHBOARD_SITE', 'hegg')
if SITE not in ('hegg', 'furu'):
    raise SystemExit(f'wsgi.apps.polar: DASHBOARD_SITE must be hegg | furu, not {SITE!r}')


def _hegg_registry():
    from wsgi.apps.path_integration    import paths as _pi_paths,  extractions as _pi_extractions
    from wsgi.apps.figure1_snyder_2012 import paths as _f1_paths,  extractions as _f1_extractions
    entries = []
    for app_paths, app_extractions in [
        (_pi_paths, _pi_extractions),
        (_f1_paths, _f1_extractions),
    ]:
        for p in app_paths:
            entries.append({'path': p, 'extractions': app_extractions,
                            'label': _config_label(p)})
    return entries


def _furu_registry():
    # the design app's own load() -> (df, factors), so polar and scatter read the same frame; and analyze_design_choices'
    # tables with the head's floor: cells (one row per readout x head x design x k̄, so a max over the other factors is
    # the best design's accuracy on this run, in-sample) and polar (the best design per head x perturbation x r chosen
    # on the pilot and scored on this run, rule 6: fig 2)
    from project.boolean_reservoir.test.furu.analyze_design_choices import GRID, OUT, with_floor
    from wsgi.apps.design import load
    from wsgi.base import frame_factors

    def table(name, label):
        path = str(OUT / f'{name}.csv')

        def load_table():
            import pandas as pd
            df = with_floor(pd.read_csv(path))
            df['I_ticks'] = df['I_ticks'].astype(str)
            return df, frame_factors(df)

        return {'path': path, 'load': load_table, 'label': label, 'exists': Path(path).exists}

    return [{'path': GRID, 'load': load, 'label': _config_label(GRID)},
            table('cells', 'design means (cells)'), table('polar', 'pilot-chosen best designs (polar)')]


_SITES = {
    'hegg': dict(views_root=VIEWS_ROOT,      registry=_hegg_registry),
    'furu': dict(views_root=FURU_VIEWS_ROOT, registry=_furu_registry),
}

_site     = _SITES[SITE]
_REGISTRY = _site['registry']()


def __getattr__(name):
    if name in ('app', 'server'):
        url_prefix = os.environ.get('URL_PREFIX', '/')
        available  = [e for e in _REGISTRY if (e['exists']() if 'exists' in e else _filter_existing([e['path']]))]
        if not available:
            _app = Dash(__name__, url_base_pathname=url_prefix)
            _app.layout = html.Div('Data not yet available.',
                                   style={'padding': '40px', 'fontFamily': 'monospace'})
        else:
            _app = create_polar_dashboard(available, url_prefix=url_prefix,
                                          views_dir=_site['views_root'] / 'polar')
        mod = sys.modules[__name__]
        mod.app, mod.server = _app, _app.server
        return getattr(mod, name)
    raise AttributeError(name)
