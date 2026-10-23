"""
furu probe 2, channel capacity, raw: one row per job x readout x head over the capacity grids
(config/temporal/multihead/capacity/c<c>.yaml), from project.boolean_reservoir.test.furu.analyze_capacity. cap is in channels
(Dambre et al. 2012), cap_frac = cap / c; c x I_redundancy = drive. Grids load once they have a merged data.parquet:
restart the app after a pull.
"""
from wsgi.base import load_frame_dashboard, make_lazy_app, frame_factors
from dashboards.utils import FURU_VIEWS_ROOT


def load():
    from project.boolean_reservoir.test.furu.analyze_capacity import load as load_capacity
    df = load_capacity()
    return df, (frame_factors(df) if not df.empty else [])


__getattr__ = make_lazy_app(__name__, lambda: load_frame_dashboard(
    load, dashboard_kwargs={'initial_views_dir': str(FURU_VIEWS_ROOT / 'capacity')}))
