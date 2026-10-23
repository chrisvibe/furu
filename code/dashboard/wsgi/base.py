"""
Shared loader for scatter dashboard wsgi apps.

Filters config paths to those with existing data files, so apps start
with whatever data is available rather than crashing when runs are missing.
"""
import os
import sys
from pathlib import Path

import pandas as pd

from dashboards.scatter import create_scatter_dashboard
from dashboards.utils import _DEFAULT_CACHE_DIR, categorize_object_columns, is_dataset_hidden
from project.boolean_reservoir.code.utils.load_save import (
    custom_load_grid_search_data, get_data_path, is_cache_warm, load_yaml_config,
)
from project.boolean_reservoir.code.utils.metrics import get_reservoir_metrics, reservoir_key
from project.boolean_reservoir.code.utils.single_run import single_run_base

RESERVOIR_METRICS_EXTRACTION = ('M', lambda p: get_reservoir_metrics(p), {'lut_p'}, reservoir_key)

# Extractions shared across all grid-search scatter apps.
# Each app prepends its task-specific T and D entries, then appends this list.
GRID_SEARCH_EXTRACTIONS_COMMON = [
    ('kqgr',       lambda p: p.L.kqgr,   {'kq', 'gr', 'delta'}),
    # rank_query is left out: it is capped at |queries| - 1 = 4 and sits there in every row
    # getattr: grids from before the walk x query metrics have no walk_query log at all (None = skipped)
    ('walk_query', lambda p: getattr(p.L, 'walk_query', None), {'kq', 'rank_walk', 'rank_interaction', 'interaction_energy', 'interaction_share'}),
    RESERVOIR_METRICS_EXTRACTION,
    ('L',          lambda p: p.L,         {'universe', 'out_path'}),
    ('L_out_name', lambda p: Path(p.L.out_path).name if p.L.out_path else None, None),
    # the KQGR settings of the universe the row's kqgr stage ran in
    ('kqgr',       lambda p: getattr(p.U, p.L.universe or 'kqgr').D, {'tau', 'tau_mode', 'tau_axis', 'evaluation'}),
    ('O',          lambda p: p.M.O,       {'mode'}),
    ('I',          lambda p: p.M.I,       {'perturbation', 'encoding', 'redundancy', 'ticks'}),
    ('R',          lambda p: {
        **dict(p.M.R or {}),
        **{k[2:]: v for k, v in (p.M.variables or {}).items() if k.startswith('R_')},
    }, {'mode', 'k_avg', 'self_loops', 'n_nodes', 'init'}),
]


def make_lazy_app(module_name: str, build):
    """Return a module-level __getattr__ that defers build() -> (app, server) until app/server is accessed."""
    def __getattr__(name):
        if name in ('app', 'server'):
            mod = sys.modules[module_name]
            mod.app, mod.server = build()
            return getattr(mod, name)
        raise AttributeError(name)
    return __getattr__


def make_lazy_loader(module_name: str, paths: list, extractions: list,
                     dashboard_kwargs: dict = None):
    """make_lazy_app for a grid-search scatter app (load_scatter_dashboard)."""
    return make_lazy_app(module_name, lambda: load_scatter_dashboard(paths, extractions, dashboard_kwargs=dashboard_kwargs))


_DEFAULT_VIEWS_DIR = Path('/out/dashboard')


def views_dir(path_str, default: Path = _DEFAULT_VIEWS_DIR) -> Path:
    """Resolve a user-supplied views directory string, falling back to default."""
    return Path(path_str.strip()) if path_str and path_str.strip() else default


def list_view_options(path_str=None, default: Path = _DEFAULT_VIEWS_DIR) -> list:
    """Return [{label, value}] for all saved views in the given directory."""
    vdir = views_dir(path_str, default)
    if not vdir.exists():
        return []
    return [{'label': p.stem, 'value': p.stem} for p in sorted(vdir.glob('*.json'))]


# Re-exported so wsgi.base stays the stable public surface for this predicate;
# the definition itself lives in dashboards/theme.py, the single source of truth.
from dashboards.theme import is_light_theme  # noqa: F401


def register_config_file_route(app, url_prefix, allowed_root=Path('/code/config')):
    """Register a GET route at {url_prefix}config-file that serves YAML files.

    Accepts ?path=<path-relative-to-/code> and returns the file as plain text.
    Requests outside allowed_root are rejected with 403.
    """
    from flask import request as _req, Response as _Resp

    @app.server.route(f'{url_prefix}config-file')
    def _serve_config_file():
        rel  = _req.args.get('path', '')
        full = Path('/code') / rel
        try:
            if not full.resolve().is_relative_to(allowed_root.resolve()):
                return _Resp('Forbidden', status=403, content_type='text/plain')
        except Exception:
            return _Resp('Forbidden', status=403, content_type='text/plain')
        if not full.exists():
            return _Resp('Not found', status=404, content_type='text/plain')
        return _Resp(full.read_text(), content_type='text/plain; charset=utf-8')


def register_view_refresh(app, open_trigger_id, list_options_fn, body_state_id=None,
                          view_picker_id='view-picker', views_dir_id='views-dir'):
    """Register callbacks to keep a view-picker dropdown in sync with the filesystem.

    Registers two callbacks:
    - Refreshes on views-dir input change
    - Refreshes when the panel opens (open_trigger_id click)
      body_state_id: if given, only refresh when that element's style.display == 'none'
                     (panel is currently closed and about to open)

    view_picker_id / views_dir_id: override for dashboards with prefixed component IDs
    (e.g. trace uses 'tp-view-select' and 'tp-views-dir').
    """
    from dash import Output, Input, State, no_update

    @app.callback(
        Output(view_picker_id, 'options', allow_duplicate=True),
        Input(views_dir_id, 'value'),
        prevent_initial_call=True,
    )
    def _on_dir_change(views_dir_str):
        return list_options_fn(views_dir_str)

    if body_state_id:
        @app.callback(
            Output(view_picker_id, 'options', allow_duplicate=True),
            Input(open_trigger_id, 'n_clicks'),
            State(body_state_id, 'style'),
            State(views_dir_id, 'value'),
            prevent_initial_call=True,
        )
        def _on_panel_open(_, body_style, views_dir_str):
            if (body_style or {}).get('display') == 'none':
                return list_options_fn(views_dir_str)
            return no_update
    else:
        @app.callback(
            Output(view_picker_id, 'options', allow_duplicate=True),
            Input(open_trigger_id, 'n_clicks'),
            State(views_dir_id, 'value'),
            prevent_initial_call=True,
        )
        def _on_panel_open(_, views_dir_str):
            return list_options_fn(views_dir_str)


def _filter_existing(config_paths: list) -> list:
    existing = []
    for p in config_paths:
        if is_dataset_hidden(p):
            print(f'INFO: {p} hidden via DASHBOARD_HIDDEN_DATASETS, skipping', file=sys.stderr, flush=True)
            continue
        try:
            data_path = get_data_path(p)
            if data_path.exists():
                existing.append(p)
            else:
                print(f'INFO: no data yet for {p}, skipping', file=sys.stderr, flush=True)
        except Exception as e:
            print(f'WARNING: could not resolve {p}: {e}', file=sys.stderr, flush=True)
    return existing


def warm_cache(config_paths: list, extractions: list, **loader_kwargs):
    """Populate the extraction cache for config_paths without loading data into memory."""
    existing = _filter_existing(config_paths)
    if not existing:
        return
    loader_kwargs.setdefault('cache_dir', _DEFAULT_CACHE_DIR)
    custom_load_grid_search_data(config_paths=existing, extractions=extractions,
                                 warmup=True, **loader_kwargs)


def _join_weights_summary(df, config_paths):
    """§6: left-join the single-run weight summary manifest, when it exists.

    The manifest (`weights_summary.parquet`, written by wsgi.precompute_weights)
    lives under each config's single-run base: the config's L.out_path with the
    'grid_search' -> 'single_run/weights' replace convention. This is a sidecar,
    NOT an extraction: weights are per-(i, j) so a key_fn dedups nothing, a
    getter would train inside the loader's cache-miss path (the OOM path), and
    the cache key hashes the getter — not the npz state — so new runs would
    leave cached NaN columns until a cold re-derive.

    JOIN KEY is (L_out_path, i, j): i restarts at 0 per config and an app loads
    several configs, so an (i, j)-only join would silently attach another
    config's run's weights. No new arg needed: L_out_path is already an
    extracted column.
    """
    manifests = []
    for cp in config_paths:
        try:
            config_out = load_yaml_config(cp).L.out_path
        except Exception as e:
            print(f'WARNING: could not resolve {cp} for weights summary: {e}',
                  file=sys.stderr, flush=True)
            continue
        manifest_path = single_run_base(config_out) / 'weights_summary.parquet'
        if not manifest_path.exists():
            continue
        try:
            m = pd.read_parquet(manifest_path)
        except Exception as e:
            print(f'WARNING: could not read {manifest_path}: {e}', file=sys.stderr, flush=True)
            continue
        if m.empty:
            continue
        manifests.append(m.assign(_lout=m['L_out_path'].astype(str)))
    if not manifests:
        return df
    manifest = (pd.concat(manifests, ignore_index=True)
                  .drop_duplicates(subset=['_lout', 'i', 'j'], keep='last'))
    return (df.assign(_lout=df['L_out_path'].astype(str))
              .merge(manifest.drop(columns=['L_out_path']),
                     on=['_lout', 'i', 'j'], how='left')
              .drop(columns=['_lout']))


def _placeholder_app(url_prefix: str):
    print('INFO: no data available, starting placeholder app', file=sys.stderr, flush=True)
    from dash import Dash, html
    app = Dash(__name__, url_base_pathname=url_prefix)
    app.layout = html.Div('Data not yet available.', style={'padding': '40px', 'fontFamily': 'monospace'})
    return app, app.server


# A numeric table column with at most this many values is a factor (k̄, redundancy, n), not a measure.
TABLE_FACTOR_MAX = 32


def load_analysis_tables(roots: list, tables: list) -> tuple[pd.DataFrame, list[str]]:
    """Analysis CSVs as one long frame, for plotting the very statistic a paper cites.

    roots: [(tags, directory)], e.g. [({'dims': '1D'}, 'out/path_integration/1D/analysis')]; every row of a root
    carries its tags as columns. tables: CSV stems, read from each root and from each of its subfolders (the
    analysis writes one per assumption variant, e.g. 'cell_stat=good'). Each row is tagged with `table` and
    `variant` (the subfolder; 'main' for the root itself). Tables share columns by name, so one frame holds them
    all; filter on `table`. Factors: the text columns, and the numeric ones with at most TABLE_FACTOR_MAX values.
    """
    frames = []
    for tags, root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        folders = [root] + sorted(d for d in root.iterdir() if d.is_dir())
        for folder in folders:
            for table in tables:
                f = folder / f'{table}.csv'
                if f.exists():
                    t = pd.read_csv(f)
                    t = t.drop(columns=[c for c in t.columns if c.startswith('Unnamed: ')])
                    frames.append(t.assign(table=table, variant='main' if folder == root else folder.name, **tags))
    if not frames:
        return pd.DataFrame(), []
    df = pd.concat(frames, ignore_index=True)
    factors = frame_factors(df)
    # a table without a column fills it with NaN, which turns an integer column float ('16.0'): give the
    # integer-valued factors their integer spelling back, so legends and filters read '16'
    for c in factors:
        col = df[c]
        if pd.api.types.is_float_dtype(col) and col.notna().any() and (col.dropna() % 1 == 0).all():
            df[c] = col.astype('Int64')
    return df, factors


def frame_factors(df: pd.DataFrame) -> list[str]:
    """A frame's factor columns: the text and bool columns, and the numeric ones with at most TABLE_FACTOR_MAX values."""
    return [c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
            or df[c].nunique() <= TABLE_FACTOR_MAX]


def load_frame_dashboard(load, dashboard_kwargs: dict = None):
    """A scatter dashboard over any frame: load() -> (df, factors), e.g. an analysis script's own long-format loader
    (furu's one row per job x readout x head). Returns (app, server); a placeholder app while load() is empty.
    Read once at start: restart the app after new data, as for a grid."""
    url_prefix = os.environ.get('URL_PREFIX', '/')
    df, factors = load()
    if df.empty:
        return _placeholder_app(url_prefix)
    categorize_object_columns(df)
    app = create_scatter_dashboard(df, factors, url_prefix=url_prefix, **(dashboard_kwargs or {}))
    return app, app.server


def load_table_dashboard(roots: list, tables: list, dashboard_kwargs: dict = None):
    """A scatter dashboard over analysis tables (load_analysis_tables). Returns (app, server)."""
    return load_frame_dashboard(lambda: load_analysis_tables(roots, tables), dashboard_kwargs)


def load_scatter_dashboard(config_paths: list, extractions: list,
                   dashboard_kwargs: dict = None, **loader_kwargs):
    """Load data and create a scatter dashboard, skipping missing data files.

    Returns (app, server). If no data exists yet, returns a placeholder app.
    """
    existing = _filter_existing(config_paths)
    url_prefix = os.environ.get('URL_PREFIX', '/')

    if not existing:
        return _placeholder_app(url_prefix)

    loader_kwargs.setdefault('cache_dir', _DEFAULT_CACHE_DIR)
    # params_json is the boot-memory whale (~3.7 GB resident on path-integration
    # alone): load without it. The dashboard lazy-fetches per-row params on
    # demand via config_by_lout (scatter._fetch_params_row); keep_params_json=True
    # remains available to tools that need the whole column in one pass.
    loader_kwargs.setdefault('keep_params_json', False)
    df, factors = custom_load_grid_search_data(config_paths=existing, extractions=extractions, **loader_kwargs)
    df = _join_weights_summary(df, existing)
    # Low-cardinality string columns → category (str-first, see the helper's
    # docstring: the astype(str) value contract is what factor filters and
    # saved views match on, and a naive category cast zeroed them).
    categorize_object_columns(df)
    # Map each config's L.out_path to its config path, so scatter can
    # lazy-fetch a row's params without holding the params_json column.
    config_by_lout = {}
    for cp in existing:
        try:
            config_by_lout[str(load_yaml_config(cp).L.out_path)] = cp
        except Exception:
            continue
    app = create_scatter_dashboard(df, factors, url_prefix=url_prefix,
                                   config_by_lout=config_by_lout,
                                   **(dashboard_kwargs or {}))
    return app, app.server
