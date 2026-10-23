from wsgi.base import load_scatter_dashboard, make_lazy_loader, RESERVOIR_METRICS_EXTRACTION
from dashboards.utils import FURU_VIEWS_ROOT

paths = [
    'config/temporal/kqgr/grid_search/figure_1_snyder_2012.yaml',
]

_UNIVERSE_LABELS = {   # paper-facing: these are the figure's legend entries
    'kqgr_heterogeneous':          'separate draw',
    'kqgr_homogeneous':            'separate draw',
    'kqgr_heterogeneous_identity': 'dedicated',
    'kqgr_homogeneous_identity':   'dedicated',
    'kqgr_heterogeneous_real':     'shared pool',
    'kqgr_homogeneous_real':       'shared pool',
}

def _universe_label(p):
    """The wiring's label; the dedicated mapping also carries its drive r (I.redundancy),
    since the replication sweeps it there."""
    label = _UNIVERSE_LABELS.get(p.L.universe, p.L.universe)
    return f'{label}, r = {p.M.I.redundancy}' if label == 'dedicated' else label


extractions = [
    ('kqgr', lambda p: p.L.kqgr, {'kq', 'gr', 'delta'}),
    RESERVOIR_METRICS_EXTRACTION,
    ('L', lambda p: p.L, {'universe'}),
    ('L_universe_label', _universe_label, None),
    ('D', lambda p: p.D, {'tau', 'evaluation'}),
    ('R', lambda p: p.M.R, {'n_nodes', 'init'}),
    ('I', lambda p: p.M.I, {'redundancy'}),
    ('R_k_avg', lambda p: p.M.variables.R_k_avg, None),
    ('R_mode', lambda p: p.M.variables.get('R_mode') or (
        'heterogeneous' if 'heterogeneous' in (p.L.universe or '') else 'homogeneous'
    ), None),
    ('O',          lambda p: p.M.O,       {'mode'}),
]

__getattr__ = make_lazy_loader(__name__, paths, extractions,
                               dashboard_kwargs={'initial_views_dir': str(FURU_VIEWS_ROOT / 'figure1')})

if __name__ == '__main__':
    app.run(port=8053, debug=False, dev_tools_hot_reload=False)
