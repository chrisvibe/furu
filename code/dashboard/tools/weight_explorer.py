#!/usr/bin/env python3
"""
Lightweight caller for the 3D graph weight explorer.

Wraps the two existing tools so you don't have to remember their module-paths
or PYTHONPATH plumbing:

  1. wsgi.precompute_weights                -- train/mop (i, j) weight checkpoints
  2. tools.weights_dash                   -- serve the 3D weight view

Run from anywhere; the repo & dashboard dirs are located from this file's path.

Two commands, one job
---------------------
explore     -- make sure a SPECIFIC (config, i, j) sample is trained, then open
              its 3D view. `--i` is required, `--j` defaults to 0. When the
              sample is already warmed it is instant (a cheap path-existence
              check + spawn the view). When cold, it precomputes ONLY that one
              sample via parquet predicate-pushdown on `i` -- it never loads
              the whole campaign (explore precomputes what it alone needs).
              No separate precompute step -- you cannot see the view without
              the checkpoint, so explore IS the combined step.

precompute  -- bulk backfill a scope so the scatter dashboard renders warm.
              Exactly one source, chosen explicitly:
                  --view NAME     the saved view's rows AFTER filtering
                  --config PATH   one grid-search campaign's train-universe
                  --config PATH --i X   only row X's j-samples in that config
              `--limit N` always means "at most N missing rows in the scope".

Why explore's precompute is separate from precompute's scope: for ONE sample
the full load path (load_rows -> fetch_params_json over the whole universe,
~1.9 GB in RAM for the big campaign) is wasteful, so explore uses a scoped
single-row fetch instead. Use `precompute` when you want the whole row / view /
campaign warmed up front.

Why no bare-(i,j) scoping across a view: `i` restarts at 0 PER config and an
app's view loads several configs, so an (i,j)-only key silently collapses rows
across configs (the tool's own docstring warns about this). `--i` only
uniquely identifies a row within one config, hence --i is config-scoped only;
view scoping is whole-filter, no --i.

Commands
--------
  # open row 374120 sample 0 on the default config, training it first if needed
  python weight_explorer.py explore --i 374120

  # same, sample 3
  python weight_explorer.py explore --i 374120 --j 3

  # just report whether (374120, 0) is warmed (no training)
  python weight_explorer.py explore --i 374120 -n

  # dry-run: how many rows a saved view's filter covers
  python weight_explorer.py precompute --view pi2d_cont_disp_vs_vel_cls -n

  # backfill exactly what that view shows (no limit)
  python weight_explorer.py precompute --view pi2d_cont_disp_vs_vel_cls

  # backfill one config's whole train-universe, at most 20 missing rows
  python weight_explorer.py precompute --limit 20

  # backfill ONLY row 374120's j-samples on the default config
  python weight_explorer.py precompute --i 374120
"""
import argparse
import os
import sys
from pathlib import Path

# This file lives at <repo>/dashboard/tools/, so parent.parent.parent is the repo root.
_REPO = Path(__file__).resolve().parent.parent.parent   # the repo root
_DASH = _REPO / 'dashboard'
for _p in (_REPO, _DASH):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# The underlying tools (wsgi.precompute_weights, the explorer, load_yaml_config)
# resolve relative config paths against the process cwd and expect repo-root
# cwd (dashboard/run.py does os.chdir(_repo_root) for the same reason). Anchor
# to the repo root so this script works no matter where it is launched from.
os.chdir(_REPO)

# The explorer's default grid-search campaign (a mappable app's config, so the
# warm extraction cache is reused). Mirrors weights_dash.DEFAULT_CONFIG.
DEFAULT_CONFIG = 'config/path_integration/2D/grid_search/design_choices/continuous.yaml'


def _abs_config(config: str) -> str:
    """Resolve a config path so relative paths are repo-root-anchored, not cwd-
    dependent. (The tools load configs via relative-path.resolve()/open(); even
    though we os.chdir(_REPO) above, making the path explicit removes ambiguity.)"""
    p = Path(config).expanduser()
    if p.is_absolute() or str(config).startswith('~'):
        return str(p.resolve())
    return str((_REPO / p).resolve())


def _load_precompute_module():
    """Import wsgi.precompute_weights lazily -- it pulls in project.
    boolean_reservoir -> juliacall, which must not run for `--help`."""
    from wsgi import precompute_weights
    return precompute_weights


def _precompute_scope(precompute_weights, *, view: str | None, config: str,
                      i: int | None, limit: int | None, dry_run: bool,
                      force: bool, layouts: bool, single_pass: bool) -> int:
    """Resolve the source (view | config), filter, and train missing rows.

    Mirrors precompute_weights.main() but lets us inject the `--i` row filter
    between load_rows() and the train step. Reuses the app's verbatim
    extractions/cache keys, so the warm extraction cache is never forked.
    """
    import importlib

    if view is not None:
        # View-driven source spans that app's configs (multi-config), post-filter.
        vdir, app_module_name = precompute_weights.resolve_view_dir(view)
        view_data = precompute_weights.load_view_json(view, vdir)
        if view_data is None:
            raise SystemExit(f'view "{view}" not found in {vdir}')
        if i is not None:
            raise SystemExit('--i is ambiguous inside a view (it spans several '
                             'configs; i restarts per config). Precompute the whole '
                             'view, or scope --i under a single --config.')
        app_mod = importlib.import_module(app_module_name)
        config_paths, extractions = app_mod.paths, app_mod.extractions
        print(f'view "{view}" -> {app_module_name} ({len(config_paths)} configs)',
              file=sys.stderr, flush=True)
    else:
        app_module_name, config_paths = precompute_weights.resolve_config_app(config)
        extractions = importlib.import_module(app_module_name).extractions
        view_data = None
        print(f'config "{config}" -> {app_module_name}', file=sys.stderr, flush=True)

    filtered = precompute_weights.load_rows(view_data, config_paths, extractions,
                                            single_pass)
    if i is not None:
        before = len(filtered)
        filtered = filtered[filtered['i'] == i]
        print(f'row scope --i {i}: {len(filtered)}/{before} rows', file=sys.stderr, flush=True)
    else:
        print(f'scope = {len(filtered)} train-universe rows'
              + (' (view-filtered)' if view_data is not None else ''),
              file=sys.stderr, flush=True)
    if filtered.empty:
        raise SystemExit('scope produced zero rows. Nothing to train.')

    # Per-row addressing is (L_out_path, i, j), NOT (i, j) -- i restarts per config.
    row_params, row_base, row_config_out = {}, {}, {}
    for r in filtered.itertuples():
        rk = (str(r.L_out_path), int(r.i), int(r.j))
        row_config_out[rk] = str(r.L_out_path)
        row_base[rk] = precompute_weights.single_run_base(r.L_out_path)
        row_params[rk] = (precompute_weights.rehydrate_params(r.params_json)
                          if getattr(r, 'params_json', None) else None)

    if layouts:
        return precompute_weights.backfill_layouts(row_params, row_base)

    present = [rk for rk in row_base if precompute_weights.weights_path(row_base[rk], rk[1], rk[2]).exists()]
    present_set = set(present)
    missing = [rk for rk in row_base if rk not in present_set]
    no_params = [rk for rk in row_base if row_params[rk] is None]
    print(f'checkpoints: {len(present)} present, {len(missing)} missing, '
          f'{len(no_params)} without params_json', file=sys.stderr, flush=True)
    for rk in sorted(present)[:10]:
        print(f'  present: {precompute_weights.checkpoint_dir(row_base[rk], rk[1], rk[2])}',
              file=sys.stderr)

    if dry_run:
        print('dry run (-n): nothing trained.', file=sys.stderr)
        return 0

    todo = missing[:max(limit, 0)] if limit is not None else missing
    print(f'training {len(todo)} of {len(missing)} missing rows'
          + (f' (--limit {limit})' if limit is not None else ' (no --limit)'),
          file=sys.stderr, flush=True)
    return precompute_weights.train_missing(todo, row_params, row_base, row_config_out,
                                            filtered, force=force)


def _precompute_one(config: str, i: int, j: int, force: bool) -> int:
    """Train JUST the one (config, i, j) sample -- no full-universe load.

    The explore path uses this for a cold row. The bulk scope path
    (_precompute_scope) goes through load_rows(), whose pass-2 fetch pulls
    params_json for the whole train-universe (~1.9 GB RAM for the 2D continuous
    campaign) -- unnecessary when all we want is one row. This uses
    fetch_params_json's parquet predicate-pushdown on `i` to read only the
    target row's params, then trains that single sample.
    """
    from wsgi import precompute_weights
    from project.boolean_reservoir.code.utils.single_run import (
        fetch_params_json, load_or_train_weights, rehydrate_params,
        save_layout, single_run_base,
    )
    from project.boolean_reservoir.code.parameter import load_yaml_config

    _app_name, config_paths = precompute_weights.resolve_config_app(config)
    # Scope pass-2 BEFORE loading anything: predicate-pushdown reads only the
    # (i, j) we need, so this stays cheap even for a huge campaign.
    pjson = fetch_params_json(config_paths, [(i, j)])
    if pjson.empty:
        print(f'no params row for (--i {i}, --j {j}) in this config; nothing to train.',
              file=sys.stderr)
        return 2
    row = pjson.iloc[0]
    params = rehydrate_params(row['params_json'])

    base = single_run_base(load_yaml_config(config).L.out_path)
    try:
        res = load_or_train_weights(params, i, j, base, force=force)
    except Exception as e:  # noqa: BLE001 -- mirror train_missing's batch-safe boundary
        print(f'FAIL (i={i}, j={j}): {e!r}', file=sys.stderr)
        return 1
    if res['trained']:
        save_layout(base, i, j, res['model'].graph)
        print(f'trained (i={i}, j={j}) -> {precompute_weights.checkpoint_dir(base, i, j)}',
              file=sys.stderr)
    else:
        print(f'checkpoint already present (i={i}, j={j}).', file=sys.stderr)
    return 0


def _record_states_for(config: str, i: int, j: int, max_samples: int | None = None) -> int:
    """States probe: train one (config, i, j) row with states recording.

    Mirrors `_precompute_one`'s predicate-pushdown fetch (single row, no universe
    load) but ends in `single_run.record_states_probe`, which does a full
    train-and-record-forward and writes the recorded states. Prints where the
    states landed; callers hand that dir to
    `BatchedTensorHistoryWriter(...).reload_history()`.
    """
    from wsgi import precompute_weights
    from project.boolean_reservoir.code.utils.single_run import (
        fetch_params_json, rehydrate_params, record_states_probe, single_run_base,
    )
    from project.boolean_reservoir.code.parameter import load_yaml_config

    _app_name, config_paths = precompute_weights.resolve_config_app(config)
    pjson = fetch_params_json(config_paths, [(i, j)])
    if pjson.empty:
        print(f'no params row for (--i {i}, --j {j}) in this config; nothing to record.',
              file=sys.stderr)
        return 2
    row = pjson.iloc[0]
    params = rehydrate_params(row['params_json'])
    base = single_run_base(load_yaml_config(config).L.out_path)
    try:
        sp = record_states_probe(params, i, j, base, max_samples=max_samples)
    except Exception as e:  # noqa: BLE001 -- batch-safe boundary, like train_missing
        print(f'FAIL (i={i}, j={j}): {e!r}', file=sys.stderr)
        return 1
    print(f'states recorded (i={i}, j={j}) -> {sp}\n'
          f'  load with: BatchedTensorHistoryWriter("{sp}").reload_history()',
          file=sys.stderr)
    return 0


def _weights_exist(config: str, i: int, j: int) -> tuple[bool, Path]:
    """True iff the exact (config, i, j) checkpoint exists. Cheap path check --
    no campaign load. Import lazy (project -> juliacall)."""
    from project.boolean_reservoir.code.utils.single_run import (
        checkpoint_dir, single_run_base, weights_path,
    )
    from project.boolean_reservoir.code.parameter import load_yaml_config
    base = single_run_base(load_yaml_config(config).L.out_path)
    has = weights_path(base, i, j).exists()
    return has, checkpoint_dir(base, i, j)


def _explore(config: str, i: int, j: int | None, dry_run: bool, force: bool,
             record_states: bool = False) -> int:
    jj = j if j is not None else 0
    present, cp = _weights_exist(config, i, jj)

    if dry_run:
        if present:
            print(f'checkpoint exists for (--i {i}, --j {jj}):\n  {cp}')
        else:
            print(f'no checkpoint for (--i {i}, --j {jj}):\n  {cp}\n'
                  f'A real explore would precompute it. Nothing trained (-n).',
                  file=sys.stderr)
        return 0

    # Pure states probe: record the dynamics for this
    # one row and return -- do NOT open the graph view. States are never present
    # implicitly (grid-search checkpoints record no states), so this is always a
    # full train+record; record_states_probe does the predicate-pushdown param
    # fetch itself, so it stays single-row-cheap even on a huge campaign.
    if record_states:
        print(f'states probe (--i {i}, --j {jj}): recording reservoir dynamics...',
              file=sys.stderr, flush=True)
        return _record_states_for(_abs_config(config), i, jj)

    if not present:
        print(f'no checkpoint for (--i {i}, --j {jj}):\n  {cp}\n'
              f'Precomputing this one sample first...', file=sys.stderr, flush=True)
        rc = _precompute_one(config, i, jj, force)
        if rc != 0:
            print(f'precompute for (--i {i}, --j {jj}) finished with rc={rc}; '
                  'not opening the view.', file=sys.stderr)
            return rc
        # Re-check -- the sample may still lack a checkpoint if training failed.
        present, cp = _weights_exist(config, i, jj)
        if not present:
            print(f'row (--i {i}, --j {jj}) still has no checkpoint after training:\n  {cp}',
                  file=sys.stderr)
            return 2

    # Delegate to the actual explorer; it opens its own Dash dev server.
    import subprocess
    env = os.environ.copy()
    old = env.get('PYTHONPATH')
    env['PYTHONPATH'] = os.pathsep.join([str(_REPO), str(_DASH)]
                                        + ([old] if old else []))
    cmd = [sys.executable, '-m', 'tools.weights_dash',
           '--config', config, '--i', str(i), '--j', str(jj)]
    print(f'\n[weight-explorer] $ {subprocess.list2cmdline(cmd)}\n', flush=True)
    return subprocess.run(cmd, cwd=_REPO, env=env).returncode


def _cmd_precompute(args) -> int:
    cfg = _abs_config(args.config or DEFAULT_CONFIG)  # None -> default config
    return _precompute_scope(_load_precompute_module(), view=args.view,
                             config=cfg, i=args.i, limit=args.limit,
                             dry_run=args.dry_run, force=args.force,
                             layouts=args.layouts, single_pass=args.single_pass)


def _cmd_explore(args) -> int:
    return _explore(_abs_config(args.config), args.i, args.j, args.dry_run,
                    args.force, record_states=args.states)


def _add_precompute_flags(p, with_i: bool):
    if with_i:
        p.add_argument('--i', type=int, default=None,
                       help='scope to ONLY this row of the --config (its j-samples). '
                            'Meaningless/forbidden with --view (i restarts per config).')
    p.add_argument('--limit', type=int, default=None,
                   help='train at most N missing rows IN THE CURRENT SCOPE '
                        '(default: no limit)')
    p.add_argument('-n', '--dry-run', action='store_true',
                   help='count present/missing and exit without training')
    p.add_argument('--force', action='store_true',
                   help='retrain even if a checkpoint exists')
    p.add_argument('--layouts', action='store_true',
                   help='backfill layout.npy for checkpointed rows, then exit')
    p.add_argument('--single-pass', action='store_true',
                   help='load keep_params_json=True in one pass (default two-pass)')
    return p


def main():
    parser = argparse.ArgumentParser(
        description='Weight explorer: eager-scoped precompute + 3D view.')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('precompute',
                       help='bulk-backfill a scope: --view NAME, or --config '
                            '[--i X], so the dashboard renders warm')
    src = p.add_mutually_exclusive_group()
    src.add_argument('--view', metavar='NAME', default=None,
                     help='saved scatter view name; precompute its filtered rows '
                          '(a full filter scope spanning that app\'s configs)')
    src.add_argument('--config', metavar='PATH', default=None,
                     help=f'config yaml scope (default when neither given: '
                          f'{DEFAULT_CONFIG})')
    _add_precompute_flags(p, with_i=True)
    p.set_defaults(func=_cmd_precompute)

    p = sub.add_parser('explore',
                       help='ensure ONE row of the --config is trained, then open '
                            'its 3D view (requires --i; --j defaults to 0); with '
                            '--states, record the row\'s reservoir dynamics instead')
    p.add_argument('--config', metavar='PATH', default=DEFAULT_CONFIG,
                   help=f'grid-search config yaml (default: {DEFAULT_CONFIG})')
    p.add_argument('--i', type=int, required=True,
                   help='grid-search combination index (config-scoped)')
    p.add_argument('--j', type=int, default=None, help='sample index (default 0)')
    p.add_argument('-n', '--dry-run', action='store_true',
                   help='just report whether the checkpoint exists; train/explore nothing')
    p.add_argument('--force', action='store_true',
                   help='retrain this sample even if a checkpoint exists')
    p.add_argument('--states', action='store_true',
                   help='force a states-persisting retrain '
                        'of this row and print where the recorded dynamics landed; '
                        'do not open the 3D view')
    p.set_defaults(func=_cmd_explore)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()