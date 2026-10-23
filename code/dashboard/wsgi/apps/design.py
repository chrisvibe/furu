"""
furu's design grid, raw: one row per job x readout x head (8 delayed heads), from the analysis script's
own loader, so the dashboard and the furu tables read the same frame (the run analyze_design_choices.RUN names),
c = 1, washout 10; kqgr_* is the paper's KQ/GR setting (analyze_design_choices.SETTING), re-measured on every
reservoir; floor is the head's floor, design_acc its design's accuracy (the median over the seeds). Filter `readout` and `target` (or task / window / delay) first; R_k_avg is k̄.
"""
from wsgi.base import load_frame_dashboard, make_lazy_app, frame_factors
from dashboards.utils import FURU_VIEWS_ROOT


def load():
    from project.boolean_reservoir.test.furu.analyze_design_choices import load as load_design, with_floor, with_design_acc
    from project.temporal.code.capacity import head_parts
    df = with_design_acc(with_floor(load_design()))
    df = df.join(head_parts(df['target']))
    return df, frame_factors(df)


__getattr__ = make_lazy_app(__name__, lambda: load_frame_dashboard(
    load, dashboard_kwargs={'initial_views_dir': str(FURU_VIEWS_ROOT / 'design')}))
