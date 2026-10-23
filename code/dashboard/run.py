#!/usr/bin/env python3
"""
Dashboard launcher — works for both local development and production deployment.

Local dev:
    python code/dashboard/run.py temporal
    python code/dashboard/run.py path_integration --port 8051
    python code/dashboard/run.py trace --debug

Production (gunicorn, used by deploy/docker-compose.yaml):
    python code/dashboard/run.py temporal --port 8052 --production
"""
import sys
import os
from pathlib import Path

# Set up paths so imports work the same in dev and production
_repo_root = Path(__file__).resolve().parent.parent  # boolean_reservoir/code/
_dashboard  = Path(__file__).resolve().parent        # boolean_reservoir/code/dashboard/
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_dashboard))
os.chdir(_repo_root)  # grid search configs resolved relative to /code

APPS = {
    'temporal':         'wsgi.apps.temporal',
    'path_integration': 'wsgi.apps.path_integration',
    'analysis':         'wsgi.apps.analysis',   # one kind per paper: DASHBOARD_SITE=hegg|furu picks the tables
    'design':           'wsgi.apps.design',
    'capacity':         'wsgi.apps.capacity',
    'figure1':          'wsgi.apps.figure1_snyder_2012',
    'polar':            'wsgi.apps.polar',
    'trace':            'wsgi.apps.trace',
}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run a boolean_reservoir dashboard')
    parser.add_argument('app', choices=list(APPS), help='Which dashboard to launch')
    parser.add_argument('--port', type=int, default=8050, help='Port (default: 8050)')
    parser.add_argument('--debug', action='store_true', help='Enable Dash debug mode (dev only)')
    parser.add_argument('--production', action='store_true',
                        help='Use gunicorn WSGI server instead of Flask dev server')
    parser.add_argument('--workers', type=int, default=1, help='Gunicorn worker count (production only)')
    parser.add_argument('--timeout', type=int, default=120, help='Gunicorn worker timeout (production only)')
    args = parser.parse_args()

    import importlib
    mod = importlib.import_module(APPS[args.app])

    if args.production:
        from gunicorn.app.base import BaseApplication

        class _StandaloneApp(BaseApplication):
            def __init__(self, application, options=None):
                self.options = options or {}
                self.application = application
                super().__init__()
            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key.lower(), value)
            def load(self):
                return self.application

        print(f'\nStarting {args.app} dashboard (gunicorn) at http://0.0.0.0:{args.port}\n')
        _StandaloneApp(mod.app.server, {
            'bind':    f'0.0.0.0:{args.port}',
            'workers': args.workers,
            'timeout': args.timeout,
        }).run()
    else:
        print(f'\nStarting {args.app} dashboard at http://localhost:{args.port}\n')
        mod.app.run(host='0.0.0.0', port=args.port, debug=args.debug)
