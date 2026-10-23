import json
import numpy as np
import pandas as pd
from pathlib import Path
from project.boolean_reservoir.code.utils.utils import generate_unique_seed
from project.boolean_reservoir.code.provenance import content_seed, stamp
from project.boolean_reservoir.code.utils.load_save import save_grid_search_results
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.train_model import train_and_evaluate_heads, DatasetCache
from project.boolean_reservoir.code.parameter import *
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
from project.parallel_grid_search.code.train_model_parallel import generic_parallel_grid_search
from project.parallel_grid_search.code.parallel_utils import JobInterface, DATASET_LOCK_KEY, RUN
from project.boolean_reservoir.code.kq_and_gr_metric import compute_walk_query_split, compute_rank_pair, gr_steps_probe
from project.boolean_reservoir.code.kqgr_probe import build_kqgr_probe
from project.boolean_reservoir.code.graph import calc_spectral_radius
from project.boolean_reservoir.code.lut import calc_lut_p
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)


def measure_kqgr(P_universe: Params, heads=None, config: int = None, sample: int = None) -> tuple:
    """KQ/GR of the reservoir P_universe's seeds build, under P_universe.D's one setting.
    Returns ({head or None: KQGRMetrics}, kq_x, model). The grid job and the re-measure
    passes on finished grids both call this, so a re-measured value is the grid's value."""
    universe_dataset_init = P_universe.dataset_init_obj
    n_nodes = P_universe.M.R.n_nodes
    seed = P_universe.D.seed
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    kq_x, gr_x, P_model = build_kqgr_probe(P_universe, universe_dataset_init, n_nodes, rng)
    model = BooleanReservoir(P_model)

    def ranks(x):   # {head or None: (readout_rank, last_rank)} from one pass
        r = compute_rank_pair(model, x, heads=heads)
        return r if heads is not None else {None: r}

    kq, gr = ranks(kq_x), ranks(gr_x)
    # Steps-axis retention GR: only where the native KQGR runs on the
    # resolution axis (PI) — temporal's stock GR already probes steps.
    steps, tau_steps = {k: (None, None) for k in kq}, None
    if getattr(P_universe.D, 'tau_axis', '') == 'resolution':
        x_steps, tau_steps_found = gr_steps_probe(kq_x)
        if x_steps is not None:
            steps, tau_steps = ranks(x_steps), tau_steps_found

    metrics = {}
    for k in kq:
        (kq_rank, kq_last), (gr_rank, gr_last), (gr_steps, gr_steps_last) = kq[k], gr[k], steps[k]
        metrics[k] = KQGRMetrics(
            config=config, sample=sample, kq=kq_rank, gr=gr_rank,
            delta=kq_rank - gr_rank,
            kq_last=kq_last, gr_last=gr_last, delta_last=kq_last - gr_last,
            gr_steps=gr_steps, gr_steps_last=gr_steps_last,
            delta_steps=None if gr_steps is None else kq_rank - gr_steps,
            tau_steps=tau_steps if gr_steps is not None else None,
        )
    return metrics, kq_x, model


def _native_setting(D) -> dict:
    return {f: getattr(D, f) for f in D.SWEEP_FIELDS}


def with_kqgr_setting(P: Params, setting: dict) -> Params:
    """P with its dataset's KQ/GR setting replaced (re-validated, so derived fields follow)."""
    D = type(P.D).model_validate({**P.D.model_dump(), **setting, 'sweep': None})
    return P.model_copy(update={'dataset': D})


class BooleanReservoirJob(JobInterface):
    """Grid search job - handles pinning/warmup to bypass OS swap."""

    def __init__(self, i: int, j: int, total_configs: int, total_samples: int,
                 locks: dict, P: Params):
        super().__init__(i, j, total_configs, total_samples, locks)
        self.P: Params = P
        self.dataset_init = P.dataset_init_obj
        self.accuracy = P.accuracy_obj

    def _init_dataset(self, init_fn, *args, **kwargs):
        # The mutex is now applied deeper, in DatasetCache._cache_dataset, so it guards
        # only the raw .npz build rather than the whole encode/copy/probe path.
        # Holding it here serialised every worker on the node: 48/48 workers were
        # queued on /tmp/<node>/dataset.lock.db with load 2.0 on 128 cores, giving
        # ~0.6 j/s cluster-wide on temporal/density/all_heterogeneous (2026-08-12).
        DatasetCache.build_lock = self.locks.get(DATASET_LOCK_KEY)
        return init_fn(*args, **kwargs)

    def _run_kqgr(self, P_universe: Params, name: str = 'kqgr'):
        heads = self.P.M.O.heads                 # None -> the model's own readout mode
        metrics, kq_x, model = self._init_dataset(measure_kqgr, P_universe, heads, self.i, self.j)
        self._log('kqgr', metrics)

        # dataset.sweep: more KQ/GR settings on this same reservoir (measurements, not axes)
        sweep = {}
        for setting in getattr(P_universe.D, 'sweep_settings', lambda: [])():
            m, _, _ = self._init_dataset(measure_kqgr, with_kqgr_setting(P_universe, setting), heads, self.i, self.j)
            name_s = P_universe.D.setting_name(**{**_native_setting(P_universe.D), **setting})
            for h, v in m.items():
                sweep.setdefault(h, {})[name_s] = v.model_copy(update={**_native_setting(P_universe.D), **setting})
        if sweep:
            self._log('kqgr_sweep', sweep)

        # The walk x query interaction, where the task crosses a query with the walks
        crossed = P_universe.dataset_init_obj.walk_query_probe(kq_x, P_universe)
        if crossed is not None:
            split = compute_walk_query_split(model, crossed[1], crossed[0], heads=heads)
            self._log('walk_query', {k: WalkQueryMetrics(**v) for k, v in split.items()})

    def _run_training(self, model: BooleanReservoir):
        dataset = self._init_dataset(self.dataset_init.train, self.P)
        # A single-mode run is the one-head case of the heads trainer (same estimator),
        # logged under None. Tasks with readout groups (PI's step readouts, temporal heads) also log them.
        heads = self.P.M.O.heads
        groups = self.dataset_init.readout_groups(self.P)
        readouts = train_and_evaluate_heads(model, dataset, heads or [self.P.M.O.mode], accuracy_fn=self.accuracy,
                                            groups=groups)
        if hasattr(model, 'save') and callable(model.save):
            model.save()
        # best_stats carries the per-split accuracy_<split>/loss_<split> fields
        # from _score_splits — passed through whole, so the parquet keeps the
        # train-vs-dev gap (generalization) per row.
        log = lambda st: TrainLog(config=self.i, sample=self.j, **st)
        named = {g.name for g in groups if g.index is None}     # {target name: stats} per head
        for field, per_head in readouts.items():
            self._log(field, {(h if heads else None): {n: log(s) for n, s in st.items()} if field in named else log(st)
                              for h, st in per_head.items()})

    def _log(self, field: str, per_head: dict):
        """Single-mode runs log L.<field>; heads runs log L.heads[head].<field> — one row
        per model either way, measurements as fields (like train + kqgr)."""
        if None in per_head:
            setattr(self.P.L, field, per_head[None])
            return
        heads = self.P.L.heads if self.P.L.heads is not None else {}
        for h, value in per_head.items():
            heads.setdefault(h, HeadLog())
            setattr(heads[h], field, value)
        self.P.L.heads = heads

    def _run(self, device=None):
        model = BooleanReservoir(self.P)
        # Spectral radius is not meaningful for discrete graphs; skipped.
        # If re-enabled, cache it in Params at combo-expansion time — eigvals on a
        # large adjacency matrix costs ~0.24s/job and would otherwise run 471k times.
        self.P.L.reservoir_metrics = ReservoirMetrics(
            spectral_radius=None,
            lut_p=calc_lut_p(model.lut),
        )
        universe_name = next(iter(self.P.multiverse_overrides or {}), None)
        self.P.L.universe = universe_name

        run = self.P.L.grid_search.run if self.P.L.grid_search else ['train']

        if 'kqgr' in run and universe_name:
            P_universe = getattr(self.P.U, universe_name)
            self._run_kqgr( P_universe, name=universe_name)

        if 'train' in run:
            self._run_training(model)

        return {'status': 'completed', 'history': self.P}

    
def apply_seed(p: Params, seed: int):
    """Apply seed to all relevant param sections"""
    p.M.I.seed = p.M.R.seed = p.M.O.seed = seed
    if p.M.T:
        p.M.T.seed = seed

class BooleanReservoirJobFactory:
    def __init__(self, P: Params, param_combinations: list):
        self.P = P
        self.param_combinations = param_combinations

    def __call__(self, i, j, total_configs, total_samples, locks):
        p = self._override_parameter(self.P, i, j)
        return BooleanReservoirJob(
            i=i, j=j,
            total_configs=total_configs,
            total_samples=total_samples,
            locks=locks,
            P=p,
        )
    
    def _override_parameter(self, p: Params, i: int, j: int): # TODO what is more elegant. makimg this static or accesing self.P?
        """Override parameters for given config/sample index: the job's seeds (grid_search.seeding) and provenance"""
        p_new = deepcopy(self.param_combinations[i])
        G = self.P.L.grid_search
        seed = content_seed(G.seed, p_new, j) if G.seeding == 'content' else generate_unique_seed(G.seed, i, j)
        apply_seed(p_new, seed)
        stamp(p_new)
        return p_new

def boolean_reservoir_grid_search(
    yaml_path: str,
    param_combinations: list = None,
    cpu_memory_per_job_gb: float = 1,
    cpu_cores_per_job: int = 1,
    cpu_max_workers: int = None,
    exploration_rate: float = 0.1,
    history_write_thresh: int = 1000,
    compact: bool = True,
):
    """Boolean Reservoir specific grid search using the generic function"""
    yaml_path = Path(yaml_path)
    P: Params = load_yaml_config(yaml_path)

    if (Path(P.L.out_path) / RUN.completed_flag).exists():
        logger.info(f"{RUN.completed_flag} exists — grid search already complete, exiting")
        return P

    # No global seeding here. P.L.grid_search.seed reaches results only through
    # _override_parameter -> generate_unique_seed(seed, i, j) (or provenance.content_seed) -> apply_seed, which
    # gives each job its own explicit seeds; every draw downstream derives a local
    # Generator from those. Nothing between here and the workers reads global RNG:
    # generate_param_combinations is deterministic (mixed-radix indexing), and the
    # scheduler's backoff jitter is deliberately unseeded so workers do not retry in
    # lockstep — seeding it here was an unintended side effect of set_seed.

    if param_combinations is None:
        param_combinations = generate_param_combinations(P)

    # Create job factory
    factory = BooleanReservoirJobFactory(P, param_combinations)

    # Define callbacks
    def save_config(output_path: Path):
        save_yaml_config(P, output_path, copy_from_original_file_path=yaml_path)

    def process_results(entries, marks, batch_file: Path):
        """Write one batch of results to a uniquely named Parquet file."""
        if entries:
            df = pd.DataFrame({
                'params': entries,
                'i': [m[0] for m in marks],
                'j': [m[1] for m in marks],
            })
            save_grid_search_results(df, batch_file)

    # TEMP: null out spectral_radius NaN values written by older runs.
    # Remove once all grid-search outputs have been recompacted.
    def _patch_spectral_radius(df: pd.DataFrame) -> pd.DataFrame:
        def _fix(s):
            d = json.loads(s)  # stdlib json: accepts bare NaN
            try:
                d['logging']['reservoir_metrics']['spectral_radius'] = None
            except (KeyError, TypeError):
                pass
            return json.dumps(d)
        df = df.copy()
        df['params_json'] = df['params_json'].apply(_fix)
        return df

    # Run generic grid search
    generic_parallel_grid_search(
        job_factory=factory,
        total_configs=len(param_combinations),
        samples_per_config=P.L.grid_search.n_samples,
        output_path=P.L.out_path,
        cpu_memory_per_job_gb=cpu_memory_per_job_gb,
        cpu_cores_per_job=cpu_cores_per_job,
        cpu_max_workers=cpu_max_workers,
        exploration_rate=exploration_rate,
        save_config=save_config,
        process_results=process_results,
        history_write_thresh=history_write_thresh,
        compact=compact,
        compact_transform=_patch_spectral_radius,
    )
    return P
