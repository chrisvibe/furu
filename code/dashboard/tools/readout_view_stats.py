"""Batch readout stats over a scatter view's checkpointed rows.

For every checkpointed row of a saved view, load weights.npz -> W [N_O, N_R]
and compute per-output readout geometry with tools.utils.readout_analysis'
per_output_row_diagnostics — the SAME tested function the graph tool uses (DRY;
no reimplementation) — plus global_concentration (Gini) per matrix.

Sceptical framing (the question this feeds): single-node-ish readouts are
evidence the reservoir is acting as an input hash; broadly-distributed readouts
are consistent with task representation. These are NECESSARY-not-sufficient
signals: a randn "hash" readout can look MORE distributed than an honest solve,
so geometry alone cannot separate compute from hash — treat this
as description, not verdict. The feature-rank probe (saved states) is the
real measurement.

Usage (from the repo's code/ dir — stdlib `code` shadowing):

    BOOLEAN_RESERVOIR_CACHE_DIR=/out/cache python dashboard/tools/readout_view_stats.py
    BOOLEAN_RESERVOIR_CACHE_DIR=/out/cache python dashboard/tools/readout_view_stats.py \
        --view pi2d_cont_disp_vs_vel_ts_acc90 \
        --views-dir config/test/views/path_integration

WARM-CACHE NOTE: the cache dir comes from BOOLEAN_RESERVOIR_CACHE_DIR. The
devcontainer containerEnv points it at /out/cache (the warm shared cache) since
2026-09-08; before that fix it pointed at /tmp (wiped on every rebuild), which
made any batch run SILENTLY cold-re-derive the whole campaign (the 13.3 GB
OOM-risk path precompute_weights' docstring warns about). If
`echo $BOOLEAN_RESERVOIR_CACHE_DIR` does not say /out/cache, prefix the
override shown above.

Read-only over out/: no models are trained or loaded here; rows without a
weights.npz are counted and skipped. Train them with wsgi.precompute_weights.
"""
import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# parents[2] == <repo>/code (this file lives in code/dashboard/tools/).
_CODE = Path(__file__).resolve().parents[2]
for _p in (str(_CODE), str(_CODE / 'dashboard')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wsgi.precompute_weights import load_rows, resolve_view_dir  # noqa: E402
from dashboards.utils import load_view_json  # noqa: E402
from tools.utils.readout_analysis import (  # noqa: E402
    global_concentration,
    per_output_row_diagnostics,
)

DEFAULT_VIEW = 'pi2d_cont_disp_vs_vel_ts_acc90'
DEFAULT_VIEWS_DIR = _CODE / 'config' / 'test' / 'views' / 'path_integration'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--view', default=DEFAULT_VIEW,
                    help=f'saved view name (default: {DEFAULT_VIEW})')
    ap.add_argument('--views-dir', default=str(DEFAULT_VIEWS_DIR),
                    help=f'directory holding the view JSON '
                         f'(default: {DEFAULT_VIEWS_DIR})')
    ap.add_argument('--acc-min', type=float, default=0.9,
                    help='safety-net accuracy scope applied on top of the '
                         "view's own filter-expr (default: 0.9)")
    ap.add_argument('--csv', default='/tmp/readout_view_stats.csv',
                    help='output CSV path (default: /tmp/readout_view_stats.csv)')
    args = ap.parse_args()

    vdir, app_module_name = resolve_view_dir(args.view, args.views_dir)
    view_data = load_view_json(args.view, vdir)
    assert view_data is not None, f'view {args.view!r} not found in {vdir}'
    app_mod = importlib.import_module(app_module_name)
    print(f'view {args.view!r} -> {app_module_name} ({len(app_mod.paths)} configs)',
          flush=True)

    filtered = load_rows(view_data, app_mod.paths, app_mod.extractions,
                         single_pass=False)
    print(f'{len(filtered)} rows survive view + train-universe filters', flush=True)

    hi = filtered[filtered['T_accuracy'] > args.acc_min]
    print(f'{len(hi)} rows with T_accuracy > {args.acc_min}', flush=True)

    from project.boolean_reservoir.code.utils.single_run import (
        single_run_base, weights_path)

    records, missing = [], 0
    for r in hi.itertuples():
        base = single_run_base(r.L_out_path)
        wp = weights_path(base, int(r.i), int(r.j))
        if not wp.exists():
            missing += 1
            continue
        z = np.load(wp)
        W = z['W']
        gini = global_concentration(W)
        k_avg = getattr(r, 'R_k_avg', None)
        for k, d in enumerate(per_output_row_diagnostics(W)):
            records.append(dict(
                i=int(r.i), j=int(r.j), L_out_path=str(r.L_out_path),
                T_accuracy=float(r.T_accuracy),
                R_k_avg=float(k_avg) if k_avg is not None else np.nan,
                out_k=k, pr=d['pr'], support=d['support'],
                top_share=d['top_share'], top_node=d['top_node'],
                single_node=d['single_node'], gini=gini,
            ))
    n_rows = len({(r['L_out_path'], r['i'], r['j']) for r in records}) if records else 0
    print(f'checkpoints: {n_rows} rows analysed '
          f'({missing} rows missing weights.npz — skipped, nothing trained)',
          flush=True)
    if not records:
        print('nothing to summarise — train the rows with wsgi.precompute_weights '
              '(see its docstring; check counts with -n first)', flush=True)
        return 1

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.csv, index=False)
    print(f'wrote {args.csv} ({len(df)} output rows)', flush=True)

    # ---- summary (per output dim) ----
    for k, g in df.groupby('out_k'):
        print(f'\n=== output dim {k} (n={len(g)}) ===', flush=True)
        print(g[['pr', 'support', 'top_share']].describe()
              .loc[['mean', '50%', 'min', 'max']].round(3).to_string(), flush=True)
        print(f'  single_node outputs: {int(g["single_node"].sum())}/{len(g)}',
              flush=True)
        print(f'  support <= 5 (few-node read): '
              f'{int((g["support"] <= 5).sum())}/{len(g)}', flush=True)
        print(f'  support >= 100 (broad read):  '
              f'{int((g["support"] >= 100).sum())}/{len(g)}', flush=True)

    # ---- crude associations (description, not verdict) ----
    print('\n=== correlations (pearson, per output dim) ===', flush=True)
    for k, g in df.groupby('out_k'):
        if len(g) < 3:
            continue
        for col in ('T_accuracy', 'R_k_avg'):
            x = g[col].astype(float)
            if x.notna().all() and x.std() > 0:
                c = np.corrcoef(x, g['pr'])[0, 1]
                print(f'  out {k}: corr(pr, {col}) = {c:+.3f}', flush=True)

    print('\nREMINDER: PR/Gini/support are necessary-not-sufficient for '
          'compute-vs-hash (randn-hash counterexample); the '
          'feature-rank probe (saved states) is the real measurement.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
