"""
The furu analysis: every number in the paper's results comes from one table written here. Every analysis choice is a
named parameter below, so the analysis can be re-run under other choices.

Runs (RUNS): the pilot (grid_search/design_choices.yaml), on which every choice was made, and the final run
(final/design.yaml + final/wiring.yaml), on fresh reservoirs, from which every reported number comes. A design is one
level of every DESIGN factor and k; its reservoirs are its seeds (12 in the pilot, 20 in the final run). A head is
readout (final-state = classification, summed = time-series) x target (task_w<window>_d<delay>). r is the input
fan-out (I_redundancy), s the self-loop fraction. Accuracy is dev; test only confirms it (dev_test). load() is the long
frame the dashboards read: one row per reservoir x readout x target, with KQ/GR at SETTING, from RUN.

Rules
  - Design accuracy is the AGG over a design's seeds: a typical reservoir. Beside it, every best-of table gives the
    top single reservoir (*_top columns), which is luck.
  - Best designs are chosen on the pilot (acc_choice) and scored on the final run, so luck in the choice does not lift
    the score.
  - Fan-out and injection are compared over the whole grid (the best design per level); k on the mean curve over the
    other factors (a design at chance adds a flat offset, so no design is dropped).
  - Every other factor, the seeds and KQ/GR are compared at the head's operating point: the OP levels of its best
    design, with ticks, init and self-loops free. A self-loop takes one of its node's k inputs, so s is held equal in
    every comparison there, as k is (WITHIN); the cost of s itself is measured with each design at its own best k
    (retuned).
  - KQ/GR is never correlated across k or s: k sets both KQ and GR, and s spends k's edge budget.

Tables (<run>/analysis/<name>.csv), by section of the results
  data          cells: per head x design, design accuracy (and acc_mean, acc_sd, n) and test accuracy; floors: per
                head, the memoryless Bayes accuracy (final-state sees the current bit, summed the count of ones)
  best k        best_k: per head x r, the mean curve's peak k (k_best), the k within BAND of the peak (k_lo..k_hi),
                the peak, its loss at k = K_FIXED (loss_fixed) and at delta's peak k (loss_delta), the peak k of the
                mean KQ, GR and delta curves, and the top reservoir's k; k_loss: per readout x task x r, best_k's
                loss_fixed and loss_delta, mean and max over the task's heads; k_by_drive: the best-k figure (r >= K_DRIVE_MIN), per readout x series x r, a series
                a task (its heads pooled) or delta (task-free); k_by_drive_low: the same below K_DRIVE_MIN, where the
                curves are near flat (k_lo..k_hi says how flat); k_boot: k_by_drive's best k with its BOOT_CI interval
                over BOOT bootstraps of the seeds (the designs fixed: the grid is exhaustive); k_task: the paired
                bootstrap difference of best_k's k between heads on the same reservoirs (parity - density; d10 - d5),
                with each head's best-design accuracy (acc_a, acc_b) and whether either best k is undetermined
                (K_CI_MAX); k_task_summary: per comparison, the counts without the undetermined pairs;
                k_shift: k_by_drive's best k split by each other factor's level (k_other = k_best - s, the best k in
                inputs from other nodes), and k_shift_range its spread over the levels; k_design: per readout x task
                x r, the median and quartiles of each design's own best k (curves spanning MIN_SPAN), beside the mean
                curve's; variance: per head, each
                factor's share of the variance of the design means over the grid (where designs work, not what a
                factor does there); wiring_k, wiring_best, wiring_curve: homogeneous against heterogeneous wiring
                (final/wiring.yaml; wiring_best's best designs chosen on half the seeds and scored on the other half,
                with a bootstrap interval, wiring_best_summary per readout x task x r), and wiring_repro: its
                heterogeneous reservoirs rebuild design.yaml's exactly
  fan-out       drive, perturbation, polar: the best design per head x r, x injection, x injection x r (polar: the
                fan-out figure's points); selection_drop: per readout x r, drive's acc_choice - acc in % (mean, max); fanout: per head x injection, polar's peak r and accuracy and the accuracy
                at the largest r; readout: per head, the best design above the floor
  other factors twins: per head x factor x level at the operating point, design accuracy minus the twin at the REF
                level (all else equal), in %: pairs, median, quartiles, share better; self_loops: the same for s
                against s = 0, each design at its own best k at the operating point's r and injection (dk: the
                twins' k difference); *_summary: the same per readout x task; beta_op: per readout x task, a Beta
                regression (logit link) of single-reservoir accuracy on the task variant and the factors, with SEs
                clustered by design (seeds are nested in designs), and each level's average marginal effect in %
                with its BOOT_CI interval over BETA_SIMS coefficient draws: ticks and init at the operating point (s
                a control), s on the retuned designs (ticks and init controls); draws: the seeds of each head's best
                design; draw_share: the variance of single-reservoir accuracy split into within- and between-design;
                perfect: the reservoirs at dev accuracy 1, their test accuracy and their designs' accuracy
  probe         peaks: per readout x task, does delta's peak find the best k (beside a fixed K_FIXED)?; peaks_s: the
                same per s; settings: per head x setting x metric, Spearman across the reservoirs at the operating
                point, within each s (settings_summary: per readout x task; delta_zero: the share of delta = 0);
                design_op: per head x design at the operating point, design accuracy and mean KQ, GR, delta;
                design_rho: their Spearman across designs within each s, averaged over the s levels, with a
                permutation p (design_rho_summary: per readout x task); design_rho_k_step: design_rho at the
                operating point and with its k one grid step lower or higher (step -1, 0, +1); seeds: per head x setting x metric,
                Spearman across the seeds of a design, averaged over the operating point's designs, and the gain of
                the highest-delta seed and of the best seed (oracle) over the design's AGG; seeds_recommended: the
                same at SETTING on the RECOMMENDED designs
  checks        guard: fit health per head; dev_test: mean |dev - test| against the noise the split sizes alone give

Run: python -m project.boolean_reservoir.test.furu.analyze_design_choices [final|pilot]   (default final)
Tables: `... analyze_design_choices tables` writes every run's (acc, kq), as load_tables returns them, to
TABLES/<run>/{acc,kq}.parquet (about 50 MB in all). Where a run's grid data is missing, load_tables reads these
instead, so the analysis and the dashboards run without the grids.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.othermod.betareg import BetaModel

from benchmark.utils.parameter import KQGRDatasetParams
from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.stats import rho_within_designs, rho_within_groups, seed_filter
from project.boolean_reservoir.code.utils.load_save import custom_load_grid_search_data

_CFG = 'config/temporal/multihead'
RUNS = {
    'pilot': dict(config=f'{_CFG}/grid_search/design_choices.yaml',
                  data=Path('out/furu/grid_search/design_choices_kqgr_sweep/data/data.parquet'),
                  out=Path('out/furu/grid_search/design_choices_kqgr_sweep/analysis')),
    'final': dict(config=f'{_CFG}/final/design.yaml', data=Path('out/furu/grid_search/final/design/data/data.parquet'),
                  out=Path('out/furu/grid_search/final/design/analysis')),
    'wiring': dict(config=f'{_CFG}/final/wiring.yaml', data=Path('out/furu/grid_search/final/wiring/data/data.parquet'),
                   out=Path('out/furu/grid_search/final/design/analysis')),
}
for _run, _r in RUNS.items():
    _r['tables'] = Path('out/furu/tables') / _run   # export_tables: the published stand-in for the grid data
RUN = 'final'                    # what load() and the dashboards read: 'pilot' or 'final'
GRID, DATA, OUT, TABLES = (RUNS[RUN][k] for k in ('config', 'data', 'out', 'tables'))
DESIGN = ['I_perturbation', 'I_redundancy', 'I_ticks', 'R_init', 'R_self_loops']
K = 'R_k_avg'
MODE = 'R_mode'                  # wiring mode: heterogeneous everywhere but final/wiring.yaml
HEAD = ['readout', 'target']
DROP_DELAYS = (0,)               # "we focus on recall after a delay"
AGG = 'median'                   # design accuracy over the seeds: a typical reservoir; one lucky draw can't move it
OP = ['I_redundancy', 'I_perturbation', K]   # the operating point: frozen at the head's best design, the rest free
WITHIN = [*OP, 'R_self_loops']   # a self-loop takes one of its node's k inputs, so s is held equal in every comparison
                                 # at the operating point, as k is: correlations within each s, twins at equal s
REF = {'I_ticks': '1', 'R_init': 'random'}   # the twins' reference levels at the operating point
SELF_REF = {'R_self_loops': 0.0}   # the reference of s, compared on the retuned designs
RECOMMENDED = {'I_ticks': '1', 'R_self_loops': 0.0}   # the advice (init free): seeds_recommended's designs
WIRING_REF = {MODE: 'heterogeneous'}
SETTINGS = [KQGRDatasetParams.setting_name(tau, mode, ev)
            for ev in ('first', 'last', 'random') for mode in ('steal', 'augment') for tau in (3, 5)]
SETTING = 'last_steal_t5'        # the paper's KQ/GR setting, chosen on the pilot's settings table
KQGR = ['kq', 'gr', 'delta']
FIT = ['accuracy_dev', 'accuracy_test', 'fit_converged', 'fit_guard_events', 'fit_lstsq_fallbacks', 'fit_saturated']
BITS = 15                        # scored bits per stream (all 2^15 streams)
MIN_B = 24                       # reservoirs per WITHIN group in settings
MIN_C = 6                        # seeds per design in seeds
MIN_D = 6                        # designs per WITHIN group in design_rho: ticks x init
PERM = 2000                      # design_rho: permutations for its p
BAND = 0.01                      # best_k: the k whose mean accuracy is within this of the mean curve's peak
MIN_SPAN = 0.02                  # k_design: a design's curve over k counts only if its accuracy spans at least this
                                 # (a curve flat at the floor has no best k)
NEAR = 0.5                       # peaks: a peak is "near" the best k within this many k
K_FIXED = 2.0                    # the critical k of an undriven RBN, beside delta's peak and the best k
K_DRIVE_MIN = 4                  # best-k statistics from this r up: at r = 1 accuracy is at the floor for every k, so
                                 # the mean curve is flat and has no best k to find
BOOT = 200                       # k_boot, k_task: bootstrap draws of the seeds
BOOT_CI = 0.95                   # k_boot, k_task, beta_op: interval coverage
K_CI_MAX = 1.0                   # k_task: a head's best k is undetermined where its own BOOT_CI interval is wider than
                                 # this (a curve at chance peaks anywhere), and k_task_summary leaves its pairs out
BOOT_SEED = 0
BETA_SIMS = 2000                 # beta_op: coefficient draws for the marginal effects' intervals
PERFECT = 0.9995                 # a reservoir at dev accuracy 1 (the dashboards' perfect-reservoir ring)


def targets(config: str = GRID) -> list[str]:
    names = [f'{h.task}_w{h.window}_d{h.delay}' for h in load_yaml_config(config).D.heads]
    return [n for n in names if int(n.rsplit('_d', 1)[1]) not in DROP_DELAYS]


def task(t: pd.Series) -> pd.Series:
    return t.str.split('_').str[0]


def load_tables(config: str = GRID, data: Path = DATA, tables: Path = TABLES, **_) -> tuple[pd.DataFrame, pd.DataFrame]:
    """acc: one row per reservoir x readout x target; kq: one row per reservoir x readout x setting. Without the grid
    data, the published tables (export_tables)."""
    if not data.exists() and (tables / 'acc.parquet').exists():
        return pd.read_parquet(tables / 'acc.parquet'), pd.read_parquet(tables / 'kq.parquet')
    names = targets(config)
    extractions = [
        ('tg', lambda p: {f'{n}__{m}': r[m] for n, r in p.L.targets.items() if n in names for m in FIT},
         {f'{n}__{m}' for n in names for m in FIT}),
        ('sw', lambda p: {f'{s}__{m}': v[m] for s, v in p.L.kqgr_sweep.items() for m in KQGR},
         {f'{s}__{m}' for s in SETTINGS for m in KQGR}),
        ('O', lambda p: p.M.O, {'mode'}),
        ('I', lambda p: p.M.I, {'perturbation', 'redundancy', 'ticks'}),
        ('R', lambda p: {**dict(p.M.R or {}),
                         **{k[2:]: v for k, v in (p.M.variables or {}).items() if k.startswith('R_')}},
         {'k_avg', 'self_loops', 'init', 'mode'}),
        ('L', lambda p: p.L, {'reservoir_id'}),
    ]
    wide, _ = custom_load_grid_search_data(data_paths=[data], extractions=extractions)
    wide = wide.rename(columns={'O_mode': 'readout', 'L_reservoir_id': 'reservoir_id'})
    wide[MODE] = wide[MODE].fillna(WIRING_REF[MODE]) if MODE in wide else WIRING_REF[MODE]   # the pilot has no R_mode
    for c in ['readout', 'I_perturbation', 'I_ticks', 'R_init', MODE]:
        wide[c] = wide[c].astype(str)
    keys = ['i', 'j', 'reservoir_id', 'readout', *DESIGN, K, MODE]
    acc = pd.concat([wide[keys].assign(target=n, **{m: wide[f'tg_{n}__{m}'].to_numpy() for m in FIT}) for n in names],
                    ignore_index=True)
    acc = acc.assign(fit_converged=acc['fit_converged'].astype(bool), fit_saturated=acc['fit_saturated'].astype(bool))
    kq = pd.concat([wide[['i', 'j', 'readout']].assign(setting=s, **{m: wide[f'sw_{s}__{m}'].to_numpy() for m in KQGR})
                    for s in SETTINGS], ignore_index=True)
    return acc, kq


def export_tables() -> None:
    for run, r in RUNS.items():
        r['tables'].mkdir(parents=True, exist_ok=True)
        for name, t in zip(('acc', 'kq'), load_tables(**r)):
            t.to_parquet(r['tables'] / f'{name}.parquet', compression='zstd', index=False)
        print(f'{run}: {r["tables"]}')


def load(config: str = GRID, data: Path = DATA, tables: Path = TABLES) -> pd.DataFrame:
    acc, kq = load_tables(config, data, tables)
    one = kq[kq['setting'] == SETTING].drop(columns='setting').rename(columns={m: f'kqgr_{m}' for m in KQGR})
    return acc.merge(one, on=['i', 'j', 'readout'])


def cells(acc: pd.DataFrame, design: tuple[str, ...] = (*DESIGN, K)) -> pd.DataFrame:
    g = acc.groupby([*HEAD, *design], observed=True)
    return g['accuracy_dev'].agg(n='size', acc=AGG, acc_mean='mean', acc_sd='std').join(
        g['accuracy_test'].agg(AGG).rename('acc_test')).reset_index()


def floors(names: list[str]) -> pd.DataFrame:
    bits = np.array(list(itertools.product([0, 1], repeat=BITS)))
    rows = []
    for n in names:
        t, w, d = n.split('_')
        w, d = int(w[1:]), int(d[1:])
        win = bits[:, BITS - d - w:BITS - d].sum(1)
        y = 2 * win > w if t == 'density' else win % 2 == 1
        for readout, x in (('classification', bits[:, -1]), ('time-series', bits.sum(1))):
            p = pd.Series(y).groupby(x).agg(['mean', 'size'])
            rows.append({'readout': readout, 'target': n,
                         'floor': (np.maximum(p['mean'], 1 - p['mean']) * p['size']).sum() / len(y)})
    return pd.DataFrame(rows)


def top(acc: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """The outlier prong: the top single reservoir (dev) per group, its test accuracy and k."""
    t = acc.loc[acc.groupby(by, observed=True)['accuracy_dev'].idxmax(), [*by, 'accuracy_dev', 'accuracy_test', K]]
    return t.rename(columns={'accuracy_dev': 'acc_top', 'accuracy_test': 'acc_top_test', K: 'k_top'}
                    ).reset_index(drop=True)


def best(c: pd.DataFrame, acc: pd.DataFrame, by: list[str], choose: pd.DataFrame | None = None) -> pd.DataFrame:
    """The best design per group, chosen on `choose` (the pilot's cells in the final run; default c itself), with its
    accuracy and test accuracy in c (the average prong; acc_choice = its accuracy where it was chosen), and acc's top
    reservoir (the outlier prong)."""
    keys = [*HEAD, *DESIGN, K]
    pick = c if choose is None else choose
    chosen = pick.loc[pick.groupby(by, observed=True)['acc'].idxmax(), [*keys, 'acc']].rename(
        columns={'acc': 'acc_choice'})
    cols = list(dict.fromkeys([*by, 'acc', 'acc_test', 'acc_choice', K, *DESIGN]))   # a group column is a design column
    return chosen.merge(c, on=keys)[cols].merge(top(acc, by), on=by)


def with_floor(df: pd.DataFrame) -> pd.DataFrame:
    """df with each head's floor (a reference for the dashboards)."""
    return df.merge(floors(targets()), on=HEAD)


def at(df: pd.DataFrame, op: pd.DataFrame, keys: list[str] = OP) -> pd.DataFrame:
    """df's rows at each head's operating point (op: one row per head, its OP levels), or at its `keys` only."""
    return df.merge(op[[*HEAD, *keys]], on=[*HEAD, *keys])


def variance(c: pd.DataFrame) -> pd.DataFrame:
    """Long: one row per head x factor (the factor's name in `factor`, so the table's columns never shadow the
    design's own columns in a frame of all tables) with its share of the variance of the design means over the whole
    grid. Most designs sit at chance, so a share says where in the grid designs work, not what a factor does there
    (twins)."""
    rows = []
    for (ro, t), g in c.groupby(HEAD):
        mu = g['acc_mean'].mean()
        tot = ((g['acc_mean'] - mu) ** 2).sum()
        share = lambda f: ((g.groupby(f, observed=True)['acc_mean'].transform('mean') - mu) ** 2).sum() / tot
        shares = {**{f: share(f) for f in [*DESIGN, K]}, 'r_x_k': share(['I_redundancy', K]),
                  'r_x_k_x_perturbation': share(['I_redundancy', K, 'I_perturbation'])}
        rows += [{'readout': ro, 'target': t, 'factor': f, 'share': v} for f, v in shares.items()]
    return pd.DataFrame(rows)


def draw_share(c: pd.DataFrame) -> pd.DataFrame:
    """Per head: the variance of single-reservoir accuracy over the grid, split into within-design (the draw: wiring,
    LUTs and initial state, all from one seed) and between-design (every factor varied); share = within / total."""
    within = (c['acc_sd'] ** 2 * (c['n'] - 1) / c['n']).groupby([c[h] for h in HEAD]).mean()
    between = c.groupby(HEAD)['acc_mean'].var(ddof=0)
    return pd.DataFrame({'within': within, 'between': between, 'share': within / (within + between)}).reset_index()


def with_design_id(acc: pd.DataFrame) -> pd.DataFrame:
    """acc with `design`, an id per design (in the order of its first reservoir), and `seed`, its reservoirs numbered
    from 0. The final run numbers designs i and their seeds j, the pilot each reservoir i, so neither is the id."""
    acc = acc.sort_values(['i', 'j'], kind='stable')
    design = acc.groupby([*DESIGN, K, MODE], observed=True, sort=False).ngroup()   # MODE: one level but in wiring
    return acc.assign(design=design, seed=acc.groupby([*HEAD, design], observed=True).cumcount())


def with_design_acc(df: pd.DataFrame) -> pd.DataFrame:
    """df with each row's design accuracy (AGG over its design's seeds; a reference for the dashboards)."""
    return df.assign(design_acc=df.groupby([*HEAD, *DESIGN, K], observed=True)['accuracy_dev'].transform(AGG))


def perfect(acc: pd.DataFrame) -> pd.DataFrame:
    """Per readout x task: the reservoirs at dev accuracy 1, their lowest test accuracy and their designs' accuracy."""
    p = with_design_acc(acc)
    p = p[p['accuracy_dev'] >= PERFECT]
    return p.groupby([p['readout'], task(p['target']).rename('task')]).agg(
        reservoirs=('accuracy_dev', 'size'), variants=('target', 'nunique'), test_min=('accuracy_test', 'min'),
        design_acc_median=('design_acc', 'median'),
        design_acc_min=('design_acc', 'min'), design_acc_max=('design_acc', 'max')).reset_index()


def twins(m: pd.DataFrame, ref: dict, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Twin differences: for each factor in ref, every design in m (cells) at another level against its twin at the
    reference level (keys other than the factor equal), on design accuracy, in %; where k is not a key, also the
    twins' k difference (dk). Returns per head x factor x level and per readout x task x factor x level: pairs,
    median, quartiles, share better (and the median dk)."""
    pairs = []
    for f, level in ref.items():
        other = [x for x in keys if x != f]
        cols = ['acc', *([K] if K not in other else [])]
        p = m[m[f] != level].set_index(other)[[f, *cols]].join(
            m[m[f] == level].set_index(other)[cols], rsuffix='_ref', how='inner').reset_index()
        pairs.append(pd.DataFrame({'readout': p['readout'], 'target': p['target'], 'factor': f,
                                   'level': p[f].astype(str), 'diff': 100 * (p['acc'] - p['acc_ref']),
                                   **({'dk': p[K] - p[f'{K}_ref']} if K in cols else {})}))
    pairs = pd.concat(pairs, ignore_index=True).assign(task=lambda d: task(d['target']))
    agg = dict(pairs=('diff', 'size'), median=('diff', 'median'), q25=('diff', lambda d: d.quantile(0.25)),
               q75=('diff', lambda d: d.quantile(0.75)), better=('diff', lambda d: (d > 0).mean()),
               **({'dk': ('dk', 'median')} if 'dk' in pairs else {}))
    summarise = lambda by: pairs.groupby(by, observed=True).agg(**agg).reset_index()
    return summarise([*HEAD, 'factor', 'level']), summarise(['readout', 'task', 'factor', 'level'])


def retuned(c: pd.DataFrame, op: pd.DataFrame, choose: pd.DataFrame | None = None) -> pd.DataFrame:
    """The designs at each head's r and injection, each at its own best k, chosen on `choose` (the pilot's cells;
    default c), as rows of c: s spends k's edge budget, so its cost is compared with k retuned."""
    keys = [*HEAD, *DESIGN]
    pick = at(c if choose is None else choose, op, OP[:2])
    k = pick.loc[pick.groupby(keys, observed=True)['acc'].idxmax(), [*keys, K]]
    return k.merge(c, on=[*keys, K])


def draws(acc: pd.DataFrame, head: pd.DataFrame) -> pd.DataFrame:
    """The draw's spread: per head, the seeds of its best design (head: best's one row per head)."""
    keys = [*HEAD, *DESIGN, K]
    out = acc.merge(head[keys], on=keys).groupby(HEAD)['accuracy_dev'].agg(
        sd='std', q25=lambda a: a.quantile(0.25), median='median', q75=lambda a: a.quantile(0.75), top='max')
    return out.assign(top_gain=out['top'] - out['median']).reset_index()


def design_kqgr(acc: pd.DataFrame, kq: pd.DataFrame, setting: str = SETTING) -> pd.DataFrame:
    """Per readout x design: KQ, GR and delta under one setting, the MEAN over the seeds (not AGG: most summed
    reservoirs have delta 0, so a median is 0 on whole curves and their peak falls on the first k)."""
    res = acc.drop_duplicates(['i', 'j', 'readout'])[['i', 'j', 'readout', *DESIGN, K]]
    d = res.merge(kq[kq['setting'] == setting], on=['i', 'j', 'readout'])
    return d.groupby(['readout', *DESIGN, K], observed=True)[KQGR].mean().reset_index()


def mean_curve(c: pd.DataFrame, by: list[str], cols: tuple[str, ...] = ('acc',)) -> pd.DataFrame:
    """One curve per `by` group: cols averaged over the designs at each k (the other factors free)."""
    return c.groupby([*by, K], observed=True)[list(cols)].mean().reset_index()


def peak_k(m: pd.DataFrame, by: list[str], col: str) -> pd.Series:
    """Per `by` group of mean_curve's m: the k at which col peaks."""
    return m.loc[m.groupby(by, observed=True)[col].idxmax()].set_index(by)[K]


def best_k(c: pd.DataFrame, kqgr: pd.DataFrame, acc: pd.DataFrame, by: tuple[str, ...] = (*HEAD, 'I_redundancy')
           ) -> pd.DataFrame:
    by = list(by)
    m = mean_curve(c.merge(kqgr, on=['readout', *DESIGN, K]), by, ('acc', *KQGR))
    near = m[m['acc'] >= m.groupby(by, observed=True)['acc'].transform('max') - BAND]
    peak = m.groupby(by, observed=True)['acc'].max()
    at_k = lambda k: m.merge(k.rename(K).reset_index(), on=[*by, K]).set_index(by)['acc']   # k: one k per by group
    out = pd.concat([peak_k(m, by, 'acc').rename('k_best'),
                     near.groupby(by, observed=True)[K].agg(k_lo='min', k_hi='max'),
                     peak.rename('peak'),
                     (peak - m[m[K] == K_FIXED].set_index(by)['acc']).rename('loss_fixed'),
                     (peak - at_k(peak_k(m, by, 'delta'))).rename('loss_delta'),
                     *[peak_k(m, by, x).rename(f'k_{x}') for x in KQGR]], axis=1).reset_index()
    return out.merge(top(acc, by), on=by)


def k_by_drive(c: pd.DataFrame, kqgr: pd.DataFrame, acc: pd.DataFrame) -> pd.DataFrame:
    by = ['readout', 'task', 'I_redundancy']
    bk = best_k(c.assign(task=task(c['target'])), kqgr, acc.assign(task=task(acc['target'])), by)
    d = bk.drop_duplicates(['readout', 'I_redundancy'])[['readout', 'I_redundancy', 'k_delta']]   # task-free
    tasks = bk[[*by, 'k_best', 'k_lo', 'k_hi', 'loss_fixed']].rename(columns={'task': 'series', 'k_best': 'k'})
    return pd.concat([tasks,
                      d.rename(columns={'k_delta': 'k'}).assign(series='delta')], ignore_index=True)


def k_loss(bk: pd.DataFrame) -> pd.DataFrame:
    """Per readout x task x r: best_k's loss_fixed and loss_delta (the accuracy lost at K_FIXED and at delta's peak),
    mean and max over the task's heads. Not k_by_drive's loss_fixed, which is the pooled curve's."""
    d = bk.assign(task=task(bk['target']))
    return d.groupby(['readout', 'task', 'I_redundancy'])[['loss_fixed', 'loss_delta']].agg(['mean', 'max']).pipe(
        lambda t: t.set_axis([f'{c}_{s}' for c, s in t.columns], axis=1)).reset_index()


def k_design(c: pd.DataFrame, kbd: pd.DataFrame) -> pd.DataFrame:
    """Per readout x task x r: each design's own best k (the peak of its accuracy curve over k, one curve per head x
    design), over the curves that span at least MIN_SPAN: n_used of n_all, the median best k and its quartiles, beside
    the mean curve's best k (k_curve, from k_by_drive; none below K_DRIVE_MIN). A check that the mean curve's peak is
    where single designs peak, not an artefact of averaging."""
    keys = [*HEAD, *DESIGN]
    g = c.groupby(keys, observed=True)['acc']
    peaks_ = c.loc[g.idxmax(), [*keys, K]].assign(span=(g.max() - g.min()).to_numpy())
    peaks_ = peaks_.assign(task=task(peaks_['target']), used=peaks_['span'] >= MIN_SPAN)
    by = ['readout', 'task', 'I_redundancy']
    used = peaks_[peaks_['used']].groupby(by)[K]
    out = pd.concat([peaks_.groupby(by)['used'].agg(n_all='size', n_used='sum'),
                     used.median().rename('k_median'), used.quantile(0.25).rename('k_q25'),
                     used.quantile(0.75).rename('k_q75')], axis=1).reset_index()
    curve = kbd.rename(columns={'series': 'task', 'k': 'k_curve'})[[*by, 'k_curve']]
    return out.merge(curve, on=by, how='left')


def selection_drop(drive: pd.DataFrame) -> pd.DataFrame:
    """Per readout x r: how much the best design per head (drive: chosen on the pilot) loses from the pilot to the
    final run, acc_choice - acc in %, mean and max over the heads (the winner's curse of choosing on the pilot)."""
    d = drive.assign(drop=100 * (drive['acc_choice'] - drive['acc']))
    return d.groupby(['readout', 'I_redundancy'])['drop'].agg(heads='size', drop_mean='mean', drop_max='max'
                                                                ).reset_index()


def k_shift(c: pd.DataFrame, factors: tuple[str, ...] = ('I_perturbation', 'I_ticks', 'R_self_loops', 'R_init')
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per readout x task x r x factor x level: the best k of the mean curve pooled over the task's heads, as
    k_by_drive's, split by one more factor: does it move the best k the way r does? k_shift_range: per readout x task
    x r x factor, the spread of that best k over the factor's levels (k_max - k_min)."""
    c = c.assign(task=task(c['target']))
    out = []
    for f in factors:
        by = ['readout', 'task', 'I_redundancy', f]
        k = peak_k(mean_curve(c, by), by, 'acc').rename('k_best').reset_index()
        k = k.rename(columns={f: 'level'}).assign(factor=f)
        if f == 'R_self_loops':   # a self-loop takes one of its node's k inputs: the best k in inputs from other nodes
            k['k_other'] = k['k_best'] - k['level']
        out.append(k.assign(level=lambda d: d['level'].astype(str)))
    out = pd.concat(out, ignore_index=True)
    spread = out.groupby(['readout', 'task', 'I_redundancy', 'factor'])['k_best'].agg(
        levels='size', k_min='min', k_max='max').assign(range=lambda d: d['k_max'] - d['k_min']).reset_index()
    return out, spread


def peaks(bk: pd.DataFrame, by: tuple[str, ...] = ('readout', 'task')) -> pd.DataFrame:
    d = bk.assign(miss_delta=(bk['k_delta'] - bk['k_best']).abs(), miss_fixed=(K_FIXED - bk['k_best']).abs())
    d = d if 'task' in d else d.assign(task=task(d['target']))
    return d.groupby(list(by)).apply(lambda g: pd.Series({
        'points': len(g), 'rho': g['k_delta'].corr(g['k_best'], method='spearman'),
        'near_delta': (g['miss_delta'] <= NEAR).mean(), 'near_fixed': (g['miss_fixed'] <= NEAR).mean(),
        'miss_delta': g['miss_delta'].mean(), 'miss_fixed': g['miss_fixed'].mean()}), include_groups=False
    ).reset_index()


def at_op(acc: pd.DataFrame, kq: pd.DataFrame, setting: str, op: pd.DataFrame) -> pd.DataFrame:
    """The reservoirs at the operating point with their KQ/GR under one setting."""
    w = at(acc, op)[['i', 'j', *HEAD, *DESIGN, K, 'accuracy_dev', 'accuracy_test']]
    return w.merge(kq[kq['setting'] == setting], on=['i', 'j', 'readout'])


def settings(acc: pd.DataFrame, kq: pd.DataFrame, op: pd.DataFrame
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = pd.concat([rho_within_groups(at_op(acc, kq, s, op), HEAD, WITHIN, KQGR, 'accuracy_dev', MIN_B
                                       ).assign(setting=s) for s in SETTINGS], ignore_index=True)
    out = out[[*HEAD, 'setting', 'metric', 'groups', 'n', 'rho']]
    summary = out.assign(task=task(out['target'])).groupby(['readout', 'task', 'setting', 'metric']).apply(
        lambda g: np.average(g['rho'], weights=g['n']), include_groups=False).rename('rho').reset_index()
    zero = (kq['delta'] == 0).groupby([kq['readout'], kq['setting']]).mean().rename('delta_zero').reset_index()
    return out, summary, zero


def design_op(acc: pd.DataFrame, kq: pd.DataFrame, op: pd.DataFrame, setting: str = SETTING) -> pd.DataFrame:
    """Per head x design at the operating point: design accuracy (AGG) and each metric's mean over the seeds."""
    return at_op(acc, kq, setting, op).groupby([*HEAD, *DESIGN, K], observed=True).agg(
        acc=('accuracy_dev', AGG), **{m: (m, 'mean') for m in KQGR}).reset_index()


def perm_p(d: pd.DataFrame, groups: list[str], metric: str, score: str, n: int, rng: np.random.Generator) -> float:
    """Two-sided permutation p of the n-weighted mean within-group Spearman rho (rho_within_groups' groups of >= MIN_D
    rows with both sides varying), score shuffled within each group."""
    ranks = [(g[metric].rank().to_numpy(), g[score].rank().to_numpy()) for _, g in d.groupby(groups, observed=True)
             if len(g) >= MIN_D and g[metric].nunique() > 1 and g[score].nunique() > 1]
    if not ranks:
        return np.nan
    corr = lambda x, y: ((x - x.mean()) * (y - y.mean(-1, keepdims=True))).sum(-1) / np.sqrt(
        ((x - x.mean()) ** 2).sum() * ((y - y.mean(-1, keepdims=True)) ** 2).sum(-1))
    w = np.array([len(x) for x, _ in ranks])
    obs = np.average([corr(x, y) for x, y in ranks], weights=w)
    null = sum(wi * corr(x, rng.permuted(np.tile(y, (n, 1)), axis=1)) for wi, (x, y) in zip(w, ranks)) / w.sum()
    return (1 + (np.abs(null) >= abs(obs) - 1e-12).sum()) / (n + 1)


def design_rho(d: pd.DataFrame, n: int = PERM, seed: int = BOOT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Does the probe rank the designs left free at the operating point? Per head, the Spearman
    rho of each metric's design mean over the seeds with design accuracy, across design_op's designs within each s
    (WITHIN), averaged over the s levels, and its permutation p. Returns per head, and
    the mean per readout x task."""
    out = rho_within_groups(d, HEAD, WITHIN, KQGR, 'acc', MIN_D)
    rng = np.random.default_rng(seed)
    head = lambda r: d[(d['readout'] == r['readout']) & (d['target'] == r['target'])]
    out['p'] = [perm_p(head(r), WITHIN, r['metric'], 'acc', n, rng) for _, r in out.iterrows()]
    summary = out.assign(task=task(out['target'])).groupby(['readout', 'task', 'metric'])['rho'].mean().reset_index()
    return out, summary


def design_rho_k_step(acc: pd.DataFrame, kq: pd.DataFrame, op: pd.DataFrame, steps: tuple[int, ...] = (-1, 0, 1)
                      ) -> pd.DataFrame:
    """How much does design_rho depend on the operating point? Per head, design_rho at the operating point (step 0)
    and at the same r and injection with k one grid step lower (-1) or higher (+1), none past the grid's ends."""
    ks = np.sort(acc[K].unique())
    out = []
    for step in steps:
        i = np.searchsorted(ks, op[K].to_numpy()) + step
        ok = (i >= 0) & (i < len(ks))
        o = op[ok].assign(**{K: ks[i[ok]]})
        out.append(design_rho(design_op(acc, kq, o))[0].merge(o[[*HEAD, *OP]], on=HEAD).assign(step=step))
    return pd.concat(out, ignore_index=True)[[*HEAD, *OP, 'step', 'metric', 'groups', 'n', 'rho', 'p']]


def seeds(acc: pd.DataFrame, kq: pd.DataFrame, op: pd.DataFrame, settings_: list[str] = SETTINGS,
          levels: dict | None = None) -> pd.DataFrame:
    """Does the probe rank a design's seeds? Per head, the Spearman across the seeds of each design at the operating
    point (or only those at `levels`, e.g. RECOMMENDED), and the seed filter's gains."""
    design = [*DESIGN, K]
    out = []
    for s in settings_:
        w = at_op(acc, kq, s, op)
        for f, v in (levels or {}).items():
            w = w[w[f] == v]
        gains = seed_filter(w, [*HEAD, *design], 'delta', 'accuracy_dev', AGG)
        out.append(rho_within_designs(w, HEAD, design, KQGR, 'accuracy_dev', MIN_C, gains).assign(setting=s))
    return pd.concat(out, ignore_index=True)[
        [*HEAD, 'setting', 'metric', 'designs', 'rho', 'rho_se', 'gain_delta', 'gain_oracle']]


def peaks_s(c: pd.DataFrame, kqgr: pd.DataFrame, acc: pd.DataFrame) -> pd.DataFrame:
    """peaks per s level: does delta's peak find the best k at every s, as s moves it? best_k per readout x task x r x
    s (pass r >= K_DRIVE_MIN), pooled per s over readout x task x r."""
    by = ['readout', 'task', 'I_redundancy', 'R_self_loops']
    bk = best_k(c.assign(task=task(c['target'])), kqgr, acc.assign(task=task(acc['target'])), by)
    return peaks(bk, ('R_self_loops',))


def guard(acc: pd.DataFrame) -> pd.DataFrame:
    return acc.groupby(HEAD).apply(lambda t: pd.Series({
        'n': len(t), 'not_converged': (~t['fit_converged']).mean(), 'guard': (t['fit_guard_events'] > 0).mean(),
        'lstsq_fallback': (t['fit_lstsq_fallbacks'] > 0).mean(), 'saturated': t['fit_saturated'].mean()}),
        include_groups=False).reset_index()


def dev_test(acc: pd.DataFrame, config: str = GRID) -> pd.DataFrame:
    D = load_yaml_config(config).D
    noise = 0.5 * np.sqrt(1 / (D.split.dev * D.samples) + 1 / (D.split.test * D.samples)) * np.sqrt(2 / np.pi)
    gap = acc.assign(gap=(acc['accuracy_dev'] - acc['accuracy_test']).abs())
    return gap.groupby(HEAD)['gap'].mean().rename('gap').to_frame().join(
        gap.groupby([*HEAD, K])['gap'].mean().groupby(HEAD).max().rename('gap_k_max')).assign(noise=noise).reset_index()


def wiring_k(c_wiring: pd.DataFrame) -> pd.DataFrame:
    by = [*HEAD, 'I_redundancy', MODE]
    k = peak_k(mean_curve(c_wiring, by), by, 'acc').unstack(MODE)
    shift = k['homogeneous'] - k['heterogeneous']
    out = shift.groupby(HEAD).agg(same=lambda d: (d == 0).mean(), within_1=lambda d: (d.abs() <= 1).mean(), bias='mean')
    k = k.reset_index()
    for mode in ('heterogeneous', 'homogeneous'):
        out[f'rho_{mode}'] = k.groupby(HEAD).apply(lambda g: g[mode].corr(g['I_redundancy'], method='spearman'),
                                                   include_groups=False)
    return out.reset_index()


def wiring_best(acc_wiring: pd.DataFrame, n: int = BOOT, ci: float = BOOT_CI, seed: int = BOOT_SEED
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Is either wiring's best design more accurate? Per head x r x wiring, the best design (injection x k, wiring.yaml's
    levels for both wirings) chosen on one half of the seeds and scored on the other, both ways round and averaged
    (acc_<mode>), so no accuracy is scored on the reservoirs it was chosen on. diff = homogeneous - heterogeneous;
    diff_in: the same chosen and scored on all seeds (in sample); diff_lo..diff_hi: diff's BOOT_CI interval over n
    bootstraps of the seeds (each half resampled within itself, per design, the same draw for every head).
    Summary per readout x task x r, and per r over every head (readout, task = 'all'): n heads, diff's median, quartiles
    and mean, the mean's interval, the share of heads with diff > 0 (pos) and of draws with mean > 0 (pos_boot)."""
    a = with_design_id(acc_wiring)
    x = a.pivot_table(index=[*HEAD, 'I_redundancy', MODE, 'I_perturbation', K], columns='seed', values='accuracy_dev')
    hd = x.index.to_frame(index=False)[HEAD].drop_duplicates().reset_index(drop=True)
    rs, modes = (x.index.unique(lv) for lv in ('I_redundancy', MODE))
    X = x.to_numpy().reshape(len(hd), len(rs), len(modes), -1, x.shape[1])   # head x r x wiring x design x seed
    assert list(modes) == ['heterogeneous', 'homogeneous'] and not np.isnan(X).any()
    agg, half = getattr(np, AGG), X.shape[-1] // 2

    def score(y: np.ndarray) -> np.ndarray:   # head x r x wiring: the best design's accuracy, cross-fitted
        lo, hi = agg(y[..., :half], -1), agg(y[..., half:], -1)
        pick = lambda choose, ev: np.take_along_axis(ev, choose.argmax(-1)[..., None], -1)[..., 0]
        return (pick(lo, hi) + pick(hi, lo)) / 2

    rng = np.random.default_rng(seed)
    draw = lambda y: np.take_along_axis(y, rng.integers(0, y.shape[-1], y.shape[1:])[None], -1)   # per design
    s0, ins = score(X), agg(X, -1).max(-1)
    b = np.stack([score(np.concatenate([draw(X[..., :half]), draw(X[..., half:])], -1)) for _ in range(n)])
    d0, db = s0[..., 1] - s0[..., 0], b[..., 1] - b[..., 0]   # head x r; draw x head x r
    q = (1 - ci) / 2
    wb = hd.loc[hd.index.repeat(len(rs))].reset_index(drop=True).assign(
        I_redundancy=np.tile(rs, len(hd)), acc_homogeneous=s0[..., 1].ravel(), acc_heterogeneous=s0[..., 0].ravel(),
        diff=d0.ravel(), diff_lo=np.quantile(db, q, 0).ravel(), diff_hi=np.quantile(db, 1 - q, 0).ravel(),
        diff_in=(ins[..., 1] - ins[..., 0]).ravel())
    hd = hd.assign(task=task(hd['target']))
    rows = []
    for (ro, t), g in [*hd.groupby(['readout', 'task']), (('all', 'all'), hd)]:
        for j, r in enumerate(rs):
            d, m = d0[g.index, j], db[:, g.index, j].mean(1)
            rows.append({'readout': ro, 'task': t, 'I_redundancy': r, 'n': len(d), 'median': np.median(d),
                         'q25': np.quantile(d, 0.25), 'q75': np.quantile(d, 0.75), 'mean': d.mean(),
                         'mean_lo': np.quantile(m, q), 'mean_hi': np.quantile(m, 1 - q), 'pos': (d > 0).mean(),
                         'pos_boot': (m > 0).mean(), 'median_in': np.median(ins[g.index, j, 1] - ins[g.index, j, 0])})
    return wb, pd.DataFrame(rows)


def wiring(acc_wiring: pd.DataFrame, acc_design: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The wiring check: the best k under both wirings (wiring_k), the best designs' accuracy (wiring_best), the mean
    curves' difference per k (wiring_curve), and wiring.yaml's heterogeneous reservoirs against the same reservoirs in
    design.yaml (merged on reservoir_id x head: the absolute accuracy differences, all 0
    when the run reproduces)."""
    het = acc_wiring[acc_wiring[MODE] == WIRING_REF[MODE]]
    m = het.merge(acc_design, on=['reservoir_id', *HEAD], suffixes=('', '_design'))
    repro = pd.DataFrame([{'reservoirs': het['reservoir_id'].nunique(), 'matched': m['reservoir_id'].nunique(),
                           **{f'max_abs_{a}': (m[a] - m[f'{a}_design']).abs().max()
                              for a in ('accuracy_dev', 'accuracy_test')}}])
    c_wiring = cells(acc_wiring, (*DESIGN, K, MODE))
    by = [*HEAD, 'I_redundancy']
    m = mean_curve(c_wiring, [*by, MODE]).pivot_table(index=[*by, K], columns=MODE, values='acc')
    wb, wb_summary = wiring_best(acc_wiring)
    return {'wiring_k': wiring_k(c_wiring), 'wiring_repro': repro, 'wiring_best': wb, 'wiring_best_summary': wb_summary,
            'wiring_curve': (m['homogeneous'] - m['heterogeneous']).rename('diff').reset_index()}


def boot_peaks(acc: pd.DataFrame, by: list[str], n: int = BOOT, seed: int = BOOT_SEED
               ) -> tuple[pd.Series, pd.DataFrame]:
    """The best k per `by` x r (by: ['readout', 'task'] pools a task's heads, as k_by_drive; ['readout', 'target'] is
    one head, as best_k) and its n bootstraps of the reservoir draw: each design's seeds resampled with replacement,
    the same draw for every head, the designs fixed (the grid is exhaustive). The same seed gives the same draws for
    every `by`, so two heads' bootstraps are paired."""
    acc = with_design_id(acc)
    heads = acc[HEAD].drop_duplicates().sort_values(HEAD).reset_index(drop=True)
    heads = heads.assign(task=task(heads['target']))
    designs = acc[['design', 'I_redundancy', K]].drop_duplicates().sort_values('design').reset_index(drop=True)
    seeds_ = acc['seed'].nunique()
    x = acc.pivot_table(index=[*HEAD, 'design'], columns='seed', values='accuracy_dev').to_numpy().reshape(
        len(heads), len(designs), seeds_)
    cell = pd.MultiIndex.from_frame(designs[['I_redundancy', K]])

    def peaks_(y: np.ndarray) -> pd.Series:   # y: heads x designs design accuracy -> per by x r: the best k
        out = []
        for key, g in heads.groupby(by):
            m = pd.Series(y[g.index].mean(0), index=cell).groupby(level=[0, 1]).mean()
            out.append(m.groupby(level=0).idxmax().map(lambda rk: rk[1]).rename_axis('I_redundancy').to_frame('k')
                       .assign(**dict(zip(by, key))))
        return pd.concat(out).reset_index().set_index([*by, 'I_redundancy'])['k']

    agg = getattr(np, AGG)
    rng = np.random.default_rng(seed)
    b = pd.concat([peaks_(agg(np.take_along_axis(x, rng.integers(0, seeds_, (len(designs), seeds_))[None], 2), 2))
                   for _ in range(n)], axis=1)
    return peaks_(agg(x, 2)), b


def k_boot(acc: pd.DataFrame, point: pd.DataFrame, n: int = BOOT, ci: float = BOOT_CI, seed: int = BOOT_SEED
           ) -> pd.DataFrame:
    """How sure is the best k (k_by_drive) of the reservoir draw? acc at r >= K_DRIVE_MIN; point: k_by_drive."""
    k0, b = boot_peaks(acc, ['readout', 'task'], n, seed)
    ref = point[point['series'] != 'delta'].set_index(['readout', 'series', 'I_redundancy'])['k']
    assert (k0.sort_index().to_numpy() == ref.sort_index().to_numpy()).all(), 'k_boot must reproduce k_by_drive'
    a = (1 - ci) / 2
    return pd.DataFrame({'k': k0, 'k_lo': b.quantile(a, axis=1), 'k_hi': b.quantile(1 - a, axis=1),
                         'same': b.eq(k0, axis=0).mean(axis=1)}).reset_index()


def k_task(acc: pd.DataFrame, bk: pd.DataFrame, drive: pd.DataFrame, n: int = BOOT, ci: float = BOOT_CI,
           seed: int = BOOT_SEED) -> pd.DataFrame:
    """Does the task move the best k beyond its precision? Per head x r (r >= K_DRIVE_MIN; bk: best_k, which it must
    reproduce), the best k and its paired bootstraps (boot_peaks); then per pair of heads on the same reservoirs,
    b minus a: parity - density at the same readout, window, delay and r, and delay 10 - delay 5 at the same readout,
    task, window and r. diff, its BOOT_CI interval, and the share of draws above, at and below 0; acc_a, acc_b: each
    head's best-design accuracy at that r (drive's acc); undetermined: either head's own best-k interval is wider
    than K_CI_MAX."""
    k0, b = boot_peaks(acc, ['readout', 'target'], n, seed)
    ref = bk[bk['I_redundancy'] >= K_DRIVE_MIN].set_index(['readout', 'target', 'I_redundancy'])['k_best']
    assert (k0.sort_index() == ref.sort_index()).all(), 'k_task must reproduce best_k'
    a_ = (1 - ci) / 2
    wide = (b.quantile(1 - a_, axis=1) - b.quantile(a_, axis=1) > K_CI_MAX).to_numpy()
    hacc = drive.set_index(['readout', 'target', 'I_redundancy'])['acc'].reindex(k0.index).to_numpy()
    parts = k0.index.get_level_values('target').str.split('_')
    key = pd.DataFrame({'readout': k0.index.get_level_values('readout'), 'task': parts.str[0], 'w': parts.str[1],
                        'd': parts.str[2], 'I_redundancy': k0.index.get_level_values('I_redundancy')})
    rows = []
    pairs = (('parity - density', 'task', ('density', 'parity')), ('d10 - d5', 'd', ('d5', 'd10')))
    for comparison, vary, (lo, hi) in pairs:
        fixed = [c for c in ('readout', 'task', 'w', 'd', 'I_redundancy') if c != vary]
        for keys, g in key.groupby(fixed):
            ia, ib = g.index[g[vary] == lo][0], g.index[g[vary] == hi][0]
            d = b.iloc[ib].to_numpy() - b.iloc[ia].to_numpy()
            rows.append({'comparison': comparison, **dict(zip(fixed, keys)), 'k_a': k0.iloc[ia], 'k_b': k0.iloc[ib],
                         'diff': k0.iloc[ib] - k0.iloc[ia],
                         'diff_lo': np.quantile(d, a_), 'diff_hi': np.quantile(d, 1 - a_),
                         'up': (d > 0).mean(), 'same': (d == 0).mean(), 'down': (d < 0).mean(),
                         'acc_a': hacc[ia], 'acc_b': hacc[ib], 'undetermined': wide[ia] | wide[ib]})
    return pd.DataFrame(rows)


def k_task_summary(kt: pd.DataFrame) -> pd.DataFrame:
    """Per comparison of k_task, its pairs without the undetermined ones (excluded: their count): how many have the
    same best k, a higher (up) or lower (down) one, and an interval that excludes 0 (beyond); beyond_acc_max: the
    highest min(acc_a, acc_b) among the beyond pairs, i.e. how accurate the weaker variant gets where the task or
    delay moves the best k beyond its precision."""
    def one(g: pd.DataFrame) -> pd.Series:
        u = g[~g['undetermined']]
        out = u[(u['diff_lo'] > 0) | (u['diff_hi'] < 0)]
        return pd.Series({'pairs': len(u), 'excluded': len(g) - len(u), 'same': (u['diff'] == 0).sum(),
                          'up': (u['diff'] > 0).sum(), 'down': (u['diff'] < 0).sum(), 'beyond': len(out),
                          'beyond_acc_max': out[['acc_a', 'acc_b']].min(axis=1).max()})
    return kt.groupby('comparison', sort=False).apply(one, include_groups=False).reset_index()


def fanout(polar: pd.DataFrame, fl: pd.DataFrame) -> pd.DataFrame:
    """The fan-out figure's lines: per head x perturbation, over polar's best design per r, the r of the peak, the
    peak, the accuracy at the largest r and the head's floor."""
    rows = []
    for (ro, t, pert), g in polar.groupby([*HEAD, 'I_perturbation']):
        g = g.sort_values('I_redundancy')
        top = g.loc[g['acc'].idxmax()]
        rows.append({'readout': ro, 'target': t, 'I_perturbation': pert, 'r_peak': top['I_redundancy'],
                     'acc_peak': top['acc'], 'r_max': g['I_redundancy'].iloc[-1], 'acc_r_max': g['acc'].iloc[-1]})
    return pd.DataFrame(rows).merge(fl, on=HEAD)


def beta(m: pd.DataFrame, ref: dict, controls: tuple[str, ...] = (), ci: float = BOOT_CI, sims: int = BETA_SIMS,
         seed: int = BOOT_SEED) -> pd.DataFrame:
    """Per readout x task, a Beta regression of the single-reservoir accuracy of m (acc rows) on the task variant, the
    factors in ref (their reference levels as the baseline) and the controls; per level of a ref factor, the average
    marginal effect in % (ame) and its BOOT_CI interval over `sims` coefficient draws."""
    m = m.assign(task=lambda d: task(d['target']), I_ticks=lambda d: d['I_ticks'].astype(str),
                 R_self_loops=lambda d: d['R_self_loops'].astype(float))
    terms = {f: f"C({f}, Treatment({level!r}))" for f, level in ref.items()}
    formula = f"y ~ {' + '.join(['C(target)', *terms.values(), *(f'C({x})' for x in controls)])}"
    rng, a, rows = np.random.default_rng(seed), (1 - ci) / 2, []
    for (ro, t), g in m.groupby(['readout', 'task']):
        n = len(g)
        g = g.assign(y=(g['accuracy_dev'] * (n - 1) + 0.5) / n)   # Smithson-Verkuilen squeeze off {0, 1}
        design = g.groupby([*DESIGN, K], observed=True).ngroup().to_numpy()   # seeds are nested in designs
        mod = BetaModel.from_formula(formula, g)
        r = mod.fit(disp=0, cov_type='cluster', cov_kwds={'groups': design})
        x, names, k = mod.exog, mod.exog_names, mod.exog.shape[1]
        draws_ = rng.multivariate_normal(r.params.to_numpy(), r.cov_params().to_numpy(), sims, method='cholesky')[:, :k]
        for f, term in terms.items():
            cols = [j for j, nm in enumerate(names[:k]) if nm.startswith(term)]
            x0 = x.copy()
            x0[:, cols] = 0                                                  # every row at the reference level
            for j in cols:
                x1 = x0.copy()
                x1[:, j] = 1
                ame = lambda beta_: 100 * (1 / (1 + np.exp(-x1 @ beta_)) - 1 / (1 + np.exp(-x0 @ beta_))).mean()
                d = np.array([ame(bb) for bb in draws_])
                rows.append({'readout': ro, 'task': t, 'factor': f, 'level': names[j].split('[T.')[1].rstrip(']'),
                             'coef': r.params.iloc[j], 'se': r.bse.iloc[j], 'p': r.pvalues.iloc[j],
                             'ame': ame(r.params.to_numpy()[:k]), 'ame_lo': np.quantile(d, a),
                             'ame_hi': np.quantile(d, 1 - a), 'reservoirs': n, 'designs': design.max() + 1,
                             'converged': r.mle_retvals.get('converged')})
    return pd.DataFrame(rows)


def analyze(acc: pd.DataFrame, kq: pd.DataFrame, pilot: tuple[pd.DataFrame, pd.DataFrame] | None = None
            ) -> dict[str, pd.DataFrame]:
    """Every table of one run. pilot = the pilot's (acc, kq): the best designs, and so the operating points, are then
    chosen on it; without it, on the run itself."""
    c = cells(acc)
    c_pilot = None if pilot is None else cells(pilot[0])
    fl = floors(targets())
    head = best(c, acc, HEAD, c_pilot)
    op = head[[*HEAD, *OP]]
    rt = retuned(c, op, c_pilot)
    kqgr = design_kqgr(acc, kq)
    bk = best_k(c, kqgr, acc)
    driven = lambda *ds: [d[d['I_redundancy'] >= K_DRIVE_MIN] for d in ds]
    undriven = lambda *ds: [d[d['I_redundancy'] < K_DRIVE_MIN] for d in ds]
    kbd = k_by_drive(*driven(c, kqgr, acc))
    ks, ks_range = k_shift(c)
    drive = best(c, acc, [*HEAD, 'I_redundancy'], c_pilot)
    kt = k_task(driven(acc)[0], bk, drive)
    tw, tw_summary = twins(at(c, op), REF, [*HEAD, *DESIGN, K])
    sl, sl_summary = twins(rt, SELF_REF, [*HEAD, *DESIGN])
    polar = best(c, acc, [*HEAD, 'I_perturbation', 'I_redundancy'], c_pilot)
    readout = head.merge(fl, on=HEAD)
    st, st_summary, zero = settings(acc, kq, op)
    dop = design_op(acc, kq, op)
    dr, dr_summary = design_rho(dop)
    return {
        # data
        'cells': c, 'floors': fl,
        # best k
        'best_k': bk, 'k_by_drive': kbd, 'k_by_drive_low': k_by_drive(*undriven(c, kqgr, acc)), 'k_boot': k_boot(driven(acc)[0], kbd), 'k_task': kt,
        'k_task_summary': k_task_summary(kt),
        'k_loss': k_loss(bk), 'k_shift': ks, 'k_shift_range': ks_range, 'k_design': k_design(c, kbd), 'variance': variance(c),
        # fan-out, injection and readout
        'drive': drive, 'selection_drop': selection_drop(drive),
        'perturbation': best(c, acc, [*HEAD, 'I_perturbation'], c_pilot), 'polar': polar, 'fanout': fanout(polar, fl),
        'readout': readout.assign(above_floor=readout['acc'] - readout['floor'],
                                  above_floor_top=readout['acc_top'] - readout['floor']),
        # the other factors and the draw
        'twins': tw, 'twins_summary': tw_summary, 'self_loops': sl, 'self_loops_summary': sl_summary,
        'beta_op': pd.concat([beta(at(acc, op), REF, ('R_self_loops',)).assign(k='operating point'),
                              beta(acc.merge(rt[[*HEAD, *DESIGN, K]]), SELF_REF, tuple(REF)).assign(k='retuned')],
                             ignore_index=True),
        'draws': draws(acc, head), 'draw_share': draw_share(c), 'perfect': perfect(acc),
        # the probe
        'peaks': peaks(driven(bk)[0]), 'peaks_s': peaks_s(*driven(c, kqgr, acc)),
        'settings': st, 'settings_summary': st_summary, 'delta_zero': zero,
        'design_op': dop, 'design_rho': dr, 'design_rho_summary': dr_summary,
        'design_rho_k_step': design_rho_k_step(acc, kq, op),
        'seeds': seeds(acc, kq, op), 'seeds_recommended': seeds(acc, kq, op, [SETTING], RECOMMENDED),
        # checks
        'guard': guard(acc), 'dev_test': dev_test(acc)}


def main(run: str = 'final') -> None:
    acc, kq = load_tables(**RUNS[run])
    pilot = None if run == 'pilot' else load_tables(**RUNS['pilot'])
    tables = analyze(acc, kq, pilot)
    if run == 'final' and any(p.exists() for p in (RUNS['wiring']['data'], RUNS['wiring']['tables'] / 'acc.parquet')):
        tables.update(wiring(load_tables(**RUNS['wiring'])[0], acc))
    out = RUNS[run]['out']
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out / f'{name}.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 400)
    pd.set_option('display.max_columns', 40)
    sm = tables['settings_summary']
    sm = sm[sm['metric'] == 'delta'].pivot(index='setting', columns=['readout', 'task'], values='rho')
    sd = tables['seeds']
    print(f'=== settings (delta rho)\n{sm.round(2).to_string()}')
    print(f'\n=== seeds at {SETTING}\n{sd[sd["setting"] == SETTING].round(3).to_string(index=False)}')
    for name in ('design_rho_summary', 'seeds_recommended', 'twins_summary', 'self_loops_summary', 'draws', 'peaks',
                 'peaks_s', 'readout', 'dev_test', 'k_boot', 'beta_op', 'wiring_k', 'wiring_repro'):
        if name in tables:
            print(f'\n=== {name}\n{tables[name].round(3).to_string(index=False)}')
    designs = acc.groupby([*DESIGN, K], observed=True).ngroups
    print(f'\n{run}: {designs} designs, {len(acc)} reservoir x readout x target rows; CSV: {out}')


if __name__ == '__main__':
    export_tables() if sys.argv[1:] == ['tables'] else main(*sys.argv[1:])
