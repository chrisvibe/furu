"""
The analysis tables behind the papers' numbers, as a scatter source. A figure
drawn from the table the text cites shows exactly the cited statistic; the grid apps plot raw rows,
for exploring. One app kind, deployed once per paper: the container's DASHBOARD_SITE picks the
paper (hegg | furu), i.e. which tables load and which views folder is home.

One frame over every table: filter `table` first, then the assumption column (`dims` and `variant`
for hegg, `grid` and `variant` for furu). Tables are read at start: restart the app after
re-running the analysis.
"""
import os

from wsgi.base import load_table_dashboard, make_lazy_app
from dashboards.utils import VIEWS_ROOT, FURU_VIEWS_ROOT

try:
    SITE = os.environ['DASHBOARD_SITE']
except KeyError:
    raise SystemExit('wsgi.apps.analysis needs DASHBOARD_SITE (hegg | furu) — the deploy sets it')

_SITES = {
    'hegg': dict(
        views_root=VIEWS_ROOT,
        roots=[({'dims': f'{d}D'}, f'out/path_integration/{d}D/analysis') for d in (1, 2)]  # analyze_pi_main_grids
             + [({'dims': f'{d}D'}, f'out/path_integration/{d}D/demo') for d in (1, 2)]     # demo_tables (walks)
             + [({}, 'out/boolean_reservoir/topology')],                                    # demo_tables (in-degrees)
        tables=[
            'split_gaps',      # Fig. split_gaps (R2b): dev / train and dev / test NMSE ratio per stage x k
            'master_shifts',   # the collapse's fitted shift and cost per design level (R1)
            'optimum',         # each design's best k (R1)
            'ladder',          # share solved and NMSE per stage x k (R2)
            'k0_controls',     # the k = 0 controls (R2)
            # the paper's remaining data figures (collapse_figures.write_tables; demo_tables.write_tables)
            'collapse_curves', # Fig. collapse_illustration: loss curves as designed and slid by the fitted shift
            'shift_check',     # Fig. collapse_check: fitted against measured shift per subset and design level
            'optimum_shift',   # Fig. optimum_shift: measured shift (bootstrap CI) and fitted, per task and design level
            'walk_demo',       # Fig. path-integration-demo: example walks and their velocity input
            'in_degree',       # Fig. indegree_distributions: in-degree histograms per wiring, size and k
        ],
    ),
    'furu': dict(
        views_root=FURU_VIEWS_ROOT,
        roots=lambda: [({'grid': 'design'}, str(_furu_out()))],   # analyze_design_choices.OUT (its RUN), read lazily
        tables=[
            # analyze_design_choices (its docstring says which table backs which claim)
            'cells',              # design accuracy per readout x head x design x k
            'floors', 'drive', 'variance', 'best_k', 'k_by_drive', 'perturbation', 'readout',
            'polar',              # Fig 2: the best design per head x injection x r, chosen on the separate run
            'twins', 'twins_summary', 'draws',                # C3 at the operating point; the draw's spread
            'self_loops_summary',                             # C3 for self-loops, k retuned
            'settings', 'settings_summary', 'delta_zero',     # rule 4: the KQ/GR settings, paired (test B)
            'peaks', 'peaks_s',                               # C6b: does delta's peak find the best k (per s)
            'design_op', 'design_rho_summary',                # the probe across the operating point's designs
            'seeds', 'seeds_recommended',                     # C6c: test C and the seed filter
            'guard', 'dev_test',  # fit health; dev against test
            'wiring_k', 'wiring_repro',                       # final run: best k under both wirings; reproducibility
        ],
    ),
}



def _furu_out():
    from project.boolean_reservoir.test.furu.analyze_design_choices import OUT
    return OUT


def site_roots(site: dict) -> list:
    """A site's (tags, folder) roots; furu's is a function, so hegg never imports the furu analysis."""
    return site['roots']() if callable(site['roots']) else site['roots']


_site = _SITES[SITE]
roots = site_roots(_site)
tables = _site['tables']

__getattr__ = make_lazy_app(__name__, lambda: load_table_dashboard(
    roots, tables, dashboard_kwargs={'initial_views_dir': str(_site['views_root'] / 'analysis')}))
