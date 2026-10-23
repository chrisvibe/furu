import os
from dashboards.trace import create_trace_plot_dashboard
from dashboards.utils import VIEWS_ROOT

app = create_trace_plot_dashboard(
    config_base_path=os.environ.get('TRACE_PLOT_CONFIG_BASE', '/code/config'),
    out_base=os.environ.get('TRACE_PLOT_OUT_BASE', '/out'),
    initial_views_dir=os.environ.get('TRACE_PLOT_VIEWS_DIR', str(VIEWS_ROOT / 'trace')),
    url_prefix=os.environ.get('URL_PREFIX', '/'),
)
server = app.server

if __name__ == '__main__':
    app.run(port=8055, debug=False, dev_tools_hot_reload=False)
