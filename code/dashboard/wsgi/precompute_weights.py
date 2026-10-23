"""Precompute single-run weights for a scatter view's or config's rows (PLAN_graph_viz.md §2).

Pass -n for a dry run (counts only); without it, missing rows are trained:

    python -m wsgi.precompute_weights -n --view <name>                 # counts only
    python -m wsgi.precompute_weights --view <name> --limit 20         # train at most 20
    python -m wsgi.precompute_weights --view <name>                    # train all missing
    python -m wsgi.precompute_weights -n --config <path/to/config.yaml>
    python -m wsgi.precompute_weights --config <path/to/config.yaml> --limit 50

--view computes the view's filtered rows; --config computes ALL of that
config's train-universe rows (no view filtering) and infers the app from
whichever mappable app's `paths` contains the config, reusing that app's
extractions verbatim so the warm cache keys stay valid. Either way rows train
from their OWN serialised job params into the (i, j)-addressable checkpoint of
utils/single_run.py, write the seeded layout.npy, and rebuild the
weights_summary.parquet sidecar (§6) after each trained row.

Two hard guards, non-negotiable:

- CACHE: the app's own `paths`/`extractions` are imported verbatim (exactly as
  wsgi/warmup.py does) and the load uses wsgi.base's cache dir. Extraction cache
  keys are content-hashed (load_save.py): a new extraction = new hash = cold
  re-derive of the whole campaign — which peaked at 13.3 GB and was OOM-killed.
  The view's directory maps to the app; no view file records its app and none
  should have to. `df_filter_mask` is NOT used for view filters: it is hashed
  into the cache key and would fork the warm cache per view.
- SCALE: training is sequential, one model at a time (2048 nodes x 30000
  samples x 5 steps is the memory risk and a pool multiplies it), and rebuilds
  the summary manifest after each trained row so progress is durable. Check
  the counts with -n first: an unfiltered view or whole config can be the
  entire train universe (580,608 rows for the 2D continuous campaign).

The default load is two-pass: pass 1 without params_json (same extractions ->
same cache key -> warm), then `fetch_params_json` scans the raw parquet for the
surviving (i, j) set only. `--single-pass` opts into `keep_params_json=True`
(cache-key-neutral, but roughly 1.9 GB of RAM for the 2D continuous config alone).

A compose service can come later, modelled on `warmup` (same mem_limit/
memswap_limit). Needs /out mounted rw — the compose services mount ..:/code:ro,
which is fine because this script writes only under /out.
"""
import argparse
import importlib
import sys
from pathlib import Path

from wsgi.base import _DEFAULT_CACHE_DIR
from dashboards.utils import VIEWS_ROOT, FURU_VIEWS_ROOT, load_view_json
from dashboards.view_filters import apply_view_filters
from project.boolean_reservoir.code.utils.load_save import custom_load_grid_search_data
from project.boolean_reservoir.code.utils.single_run import (
    SingleRunError,
    checkpoint_dir,
    fetch_params_json,
    layout_path,
    load_or_train_weights,
    rehydrate_params,
    rebuild_weights_summary,
    save_layout,
    single_run_base,
    weights_path,
)

# Views-dir basename -> app module providing module-level paths/extractions
# (mirrors run.py's APPS naming). polar reuses the other apps' paths/extractions
# via its registry, trace is not grid-search based and analysis reads tables, so
# none of the three is mappable here.
# The views tree is flat per site (deploy/sites.yaml): /out/dashboard/{site}/{app}/
# — figure1 and temporal are furu's, path_integration hegg's; basenames follow the URLs.
APPS_BY_VIEWS_DIR = {
    'path-integration': 'wsgi.apps.path_integration',
    'temporal':         'wsgi.apps.temporal',
    'figure1':          'wsgi.apps.figure1_snyder_2012',
    'design':           'wsgi.apps.design',
    'capacity':         'wsgi.apps.capacity',
}

# Both paper sites' views trees; the basename of the containing dir keys the mapping.
_SCATTER_ROOTS = (VIEWS_ROOT, FURU_VIEWS_ROOT)


def resolve_view_dir(view_name: str, views_dir: str = None):
    """Return (views_dir, app_module_name) for a view name.

    An explicit --views-dir is keyed on its basename. Without one, search
    _SCATTER_ROOTS/*/ for the view JSON: one hit disambiguates, more is an error.
    """
    if views_dir:
        vdir = Path(views_dir)
    else:
        hits = sorted(hit for root in _SCATTER_ROOTS for hit in root.glob(f'*/{view_name}.json'))
        if len(hits) == 1:
            vdir = hits[0].parent
        elif not hits:
            raise SystemExit(f'view "{view_name}" not found under '
                             f'{[str(r) for r in _SCATTER_ROOTS]}/*/ — '
                             'pass --views-dir explicitly')
        else:
            raise SystemExit(f'view "{view_name}" is ambiguous under '
                             f'{[str(r) for r in _SCATTER_ROOTS]}: '
                             f'{[str(h) for h in hits]} — pass --views-dir explicitly')
    app_name = vdir.name
    if app_name not in APPS_BY_VIEWS_DIR:
        raise SystemExit(f'no app mapping for views-dir "{app_name}" — supported: '
                         f'{sorted(APPS_BY_VIEWS_DIR)} (analysis reads tables and '
                         f'polar/trace are not grid-search based — no grid to precompute)')
    return vdir, APPS_BY_VIEWS_DIR[app_name]


def resolve_config_app(config_path: str):
    """Return (module_name, [config_path]) for a single grid-search config.

    The config must be one of a mappable app's paths: that app's extraction
    set is reused verbatim so the warm cache keys stay valid. A config no app
    loads would need a NEW extraction set -> new cache keys -> cold re-derive
    of that campaign, so it is refused instead. Relative, absolute and ~-
    prefixed paths all resolve against the app's entries.
    """
    target = Path(config_path).expanduser().resolve()
    for app_name, module_name in sorted(APPS_BY_VIEWS_DIR.items()):
        app_mod = importlib.import_module(module_name)
        if config_path in app_mod.paths or target in {Path(p).resolve() for p in app_mod.paths}:
            return module_name, [config_path]
    raise SystemExit(
        f'config "{config_path}" is not in any mappable app\'s paths '
        f'({sorted(APPS_BY_VIEWS_DIR.values())}) — refusing: a new extraction set '
        'would fork the warm extraction cache (cold re-derive of the campaign).')


def load_rows(view_data, config_paths, extractions, single_pass: bool):
    """Pass 1 (+ pass 2) load, then (optional) view filters + the train-universe filter.

    view_data=None (config mode) skips view filtering entirely — every row of
    the given configs is a candidate. Returns the surviving rows with a
    `params_json` column attached. Zero rows is a hard error: the expected
    cause is a kqgr-universe view/config, which must never train. The zero-row
    check runs BEFORE the pass-2 params fetch/merge: kqgr-universe sources
    then refuse loudly without needing L_out_path, which some apps' extraction
    sets do not carry (e.g. snyder_2012 extracts L.{universe} only). Apps that
    DO have train-universe rows must include out_path in their L extraction —
    the (L_out_path, i, j) join needs it.
    """
    loader_kwargs = dict(config_paths=config_paths, extractions=extractions,
                         cache_dir=_DEFAULT_CACHE_DIR)
    if single_pass:
        df, _ = custom_load_grid_search_data(keep_params_json=True, **loader_kwargs)
    else:
        df, _ = custom_load_grid_search_data(**loader_kwargs)

    if view_data is not None:
        filtered, _ff = apply_view_filters(df, view_data)
    else:
        filtered = df

    # §0: the kqgr universe is a different model (n_nodes: 64) — never train it.
    # Filtering here also shrinks the pass-2 fetch to the rows that can train.
    filtered = filtered[filtered['L_universe'] == 'train']
    if filtered.empty:
        raise SystemExit(
            "zero rows survived (view/config filters + L_universe == 'train') — refusing "
            'to train. Expected cause: a kqgr-universe view or config, which must never '
            'train ~48k useless 64-node models into the same base.')

    if not single_pass:
        pjson = fetch_params_json(config_paths, zip(filtered['i'], filtered['j']))
        if not pjson.empty:
            pjson = pjson.assign(_lout=pjson['L_out_path'].astype(str),
                                 i=pjson['i'].astype(int), j=pjson['j'].astype(int))
            filtered = (filtered.assign(_lout=filtered['L_out_path'].astype(str))
                        .merge(pjson[['_lout', 'i', 'j', 'params_json']],
                               on=['_lout', 'i', 'j'], how='left')
                        .drop(columns=['_lout']))

    if 'params_json' not in filtered.columns:
        filtered['params_json'] = None
    return filtered


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--view', default=None,
                        help="saved view name (JSON stem): compute the view's filtered rows")
    source.add_argument('--config', default=None, metavar='PATH',
                        help="grid-search config yaml: compute ALL of that config's "
                             'train-universe rows (no view filtering)')
    parser.add_argument('--views-dir', default=None,
                        help=f'views directory (default: search {[str(r) for r in _SCATTER_ROOTS]}/*/)')
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='count present/missing checkpoints and exit without training')
    parser.add_argument('--limit', type=int, default=None,
                        help='train at most N missing rows (default: no limit)')
    parser.add_argument('--force', action='store_true',
                        help='retrain even if the checkpoint exists')
    parser.add_argument('--layouts', action='store_true',
                        help='backfill layout.npy for checkpointed rows, then exit')
    parser.add_argument('--single-pass', action='store_true',
                        help='load keep_params_json=True in one pass (opt-in; default is two-pass)')
    args = parser.parse_args()

    if args.view:
        vdir, app_module_name = resolve_view_dir(args.view, args.views_dir)
        view_data = load_view_json(args.view, vdir)
        if view_data is None:
            raise SystemExit(f'view "{args.view}" not found in {vdir}')
        app_mod = importlib.import_module(app_module_name)
        config_paths, extractions = app_mod.paths, app_mod.extractions
        print(f'view "{args.view}" from {vdir} -> {app_module_name} '
              f'({len(config_paths)} configs)', file=sys.stderr, flush=True)
    else:
        app_module_name, config_paths = resolve_config_app(args.config)
        extractions = importlib.import_module(app_module_name).extractions
        print(f'config "{args.config}" -> {app_module_name}',
              file=sys.stderr, flush=True)
        view_data = None

    filtered = load_rows(view_data, config_paths, extractions, args.single_pass)
    print(f'{len(filtered)} rows survive the train-universe filter'
          + (' and view filters' if view_data is not None else ''),
          file=sys.stderr, flush=True)

    # Per-row addressing: (L_out_path, i, j) -> base / config out_path / row
    # params. i restarts at 0 per config and an app loads several configs, so
    # an (i, j)-only key silently COLLAPSES rows across configs (observed in
    # the dry-run campaign: a temporal view lost half its rows this way).
    row_params, row_base, row_config_out = {}, {}, {}
    for r in filtered.itertuples():
        rk = (str(r.L_out_path), int(r.i), int(r.j))
        row_config_out[rk] = str(r.L_out_path)
        row_base[rk] = single_run_base(r.L_out_path)
        row_params[rk] = (rehydrate_params(r.params_json)
                          if getattr(r, 'params_json', None) else None)

    if args.layouts:
        return backfill_layouts(row_params, row_base)

    present = [rk for rk in row_base if weights_path(row_base[rk], rk[1], rk[2]).exists()]
    present_set = set(present)
    missing = [rk for rk in row_base if rk not in present_set]
    no_params = [rk for rk in row_base if row_params[rk] is None]
    print(f'checkpoints: {len(present)} present, {len(missing)} missing, '
          f'{len(no_params)} without params_json', file=sys.stderr, flush=True)
    for rk in sorted(present)[:10]:
        print(f'  present: {checkpoint_dir(row_base[rk], rk[1], rk[2])}', file=sys.stderr)

    if args.dry_run:
        print('dry run (-n): nothing trained.', file=sys.stderr)
        return 0

    todo = missing[:max(args.limit, 0)] if args.limit is not None else missing
    print(f'training {len(todo)} of {len(missing)} missing rows'
          + ('' if args.limit is not None else ' (no --limit)'),
          file=sys.stderr, flush=True)
    return train_missing(todo, row_params, row_base, row_config_out,
                         filtered, force=args.force)


def backfill_layouts(row_params, row_base):
    """Write seeded layout.npy for checkpointed rows that lack one (§2 step 6)."""
    from project.boolean_reservoir.code.reservoir import BooleanReservoir

    done = skipped = 0
    for rk in sorted(row_base):
        base, params = row_base[rk], row_params[rk]
        cp = checkpoint_dir(base, rk[1], rk[2])
        if not weights_path(base, rk[1], rk[2]).exists() or layout_path(base, rk[1], rk[2]).exists():
            continue
        if params is None:
            skipped += 1
            continue
        model = BooleanReservoir(load_path=cp)
        # §0 sanity assert: the checkpoint's own params ARE the row's params.
        assert model.P.M.R.seed == params.M.R.seed, \
            f'(i={rk[1]}, j={rk[2]}): checkpoint seed {model.P.M.R.seed} != row seed {params.M.R.seed}'
        save_layout(base, rk[1], rk[2], model.graph)
        done += 1
        print(f'  layout: {cp} -> {layout_path(base, rk[1], rk[2])}', file=sys.stderr, flush=True)
    print(f'layouts backfilled: {done} (skipped {skipped} without params_json)', file=sys.stderr)
    return 0


def train_missing(todo, row_params, row_base, row_config_out, filtered, force: bool):
    """Sequentially train the given (L_out_path, i, j) rows.

    One bad row must not kill a batch: SingleRunError (and any training failure)
    is logged and the batch continues. The summary manifest is rebuilt after
    each trained row (the "batch") so progress is durable.
    """
    acc_by_rk = {(str(r.L_out_path), int(r.i), int(r.j)): getattr(r, 'T_accuracy', None)
                 for r in filtered.itertuples()}
    trained, failed = 0, []
    for rk in todo:
        _lout, i, j = rk
        base, params = row_base[rk], row_params[rk]
        if params is None:
            failed.append((rk, 'no params_json'))
            continue
        try:
            res = load_or_train_weights(params, i, j, base, force=force)
        except SingleRunError as e:
            failed.append((rk, str(e)))
            print(f'FAIL (i={i}, j={j}): {e}', file=sys.stderr, flush=True)
            continue
        except Exception as e:  # training failures logged, batch continues
            failed.append((rk, repr(e)))
            print(f'FAIL (i={i}, j={j}): {e!r}', file=sys.stderr, flush=True)
            continue
        if res['trained']:
            save_layout(base, i, j, res['model'].graph)
            grid_acc = acc_by_rk.get(rk)
            new_acc = res['model'].P.L.train.accuracy_dev if res['model'].P.L.train else None
            if grid_acc is not None and new_acc is not None and abs(new_acc - float(grid_acc)) > 1e-6:
                print(f'WARN (i={i}, j={j}): recomputed dev accuracy {new_acc:.6f} '
                      f'!= grid row {float(grid_acc):.6f}', file=sys.stderr, flush=True)
            rebuild_weights_summary(base, row_config_out[rk])
            trained += 1
            print(f'trained (i={i}, j={j}) -> {weights_path(base, i, j)} '
                  f'[{trained}/{len(todo)}]', file=sys.stderr, flush=True)
    print(f'done: trained {trained}/{len(todo)}, failed {len(failed)}', file=sys.stderr)
    for rk, err in failed:
        print(f'  failed (i={rk[1]}, j={rk[2]}, L_out={rk[0]}): {err}', file=sys.stderr)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
