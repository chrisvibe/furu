from wsgi.base import load_scatter_dashboard, make_lazy_loader, GRID_SEARCH_EXTRACTIONS_COMMON
from dashboards.utils import FURU_VIEWS_ROOT

paths = [
    'config/temporal/density/grid_search/design_choices/all_heterogeneous.yaml',
    'config/temporal/density/grid_search/design_choices/all_homogeneous.yaml',
    'config/temporal/parity/grid_search/design_choices/all_heterogeneous.yaml',
    'config/temporal/parity/grid_search/design_choices/all_homogeneous.yaml',
    'config/temporal/density/grid_search/design_choices/snyder.yaml',
    'config/temporal/parity/grid_search/design_choices/snyder.yaml',
]

extractions = [
    ('T', lambda p: p.L.T,       {'accuracy', 'loss'}),
    ('D', lambda p: p.D,         {'task', 'window', 'delay'}),
] + GRID_SEARCH_EXTRACTIONS_COMMON

__getattr__ = make_lazy_loader(__name__, paths, extractions,
                               dashboard_kwargs={'initial_views_dir': str(FURU_VIEWS_ROOT / 'temporal')})
