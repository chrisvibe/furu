from abc import ABC, abstractmethod
import copy
import numpy as np
import psutil
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code import glm
from project.boolean_reservoir.code.parameter import *
from project.boolean_reservoir.code.visualization import *


# ---------------------------------------------------------------------------
# Accuracy functions
# ---------------------------------------------------------------------------

class AccuracyFunction(ABC):
    @abstractmethod
    def accuracy(self, y_hat, y, threshold, normalize=True):
        pass


class EuclideanDistanceAccuracy(AccuracyFunction):
    def accuracy(self, y_hat, y, threshold, normalize=True):
        distances = np.sqrt(np.sum((y_hat - y) ** 2, axis=1))
        correct = (distances < threshold).sum()
        return correct / len(y) if normalize else int(correct)


class BooleanAccuracy(AccuracyFunction):
    def accuracy(self, y_hat, y, threshold, normalize=True):
        correct = ((y_hat > threshold) == y.astype(bool)).sum()
        return correct / len(y) if normalize else int(correct)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _mse_loss(y_hat, y):
    return float(np.mean((y_hat - y) ** 2))


def _bce_loss(y_hat, y):
    eps = 1e-7
    yh = np.clip(y_hat, eps, 1 - eps)
    return float(-np.mean(y * np.log(yh) + (1 - y) * np.log(1 - yh)))


def criterion_fn(strategy: str):
    return {'MSE': _mse_loss, 'BCE': _bce_loss}[strategy]


# ---------------------------------------------------------------------------
# Dataset cache (for KQ/GR)
# ---------------------------------------------------------------------------

class DatasetCache:
    """Memory-aware dataset cache with OS-style aging (exponential fading LRU+LFU)."""
    _kq_cache = {}
    # Cross-process mutex guarding ONLY the raw dataset build (the .npz read /
    # generate / save).  Set per worker from the job's locks; None = no locking.
    build_lock = None
    _kq_cache_current_bytes = 0
    _kq_cache_max_bytes = 1 * 1024 * 1024 * 1024  # 1 GB per worker
    _ram_threshold_percent = 90.0
    _AGE_MSB = 1 << 15

    def _get_dataset_size_bytes(self, dataset):
        total = 0
        if hasattr(dataset, 'data') and isinstance(dataset.data, dict):
            for v in dataset.data.values():
                if isinstance(v, np.ndarray):
                    total += v.nbytes
        return total

    def _age_cache(self):
        for entry in self._kq_cache.values():
            entry[2] >>= 1

    def _evict_smart(self, needed_bytes):
        if psutil.virtual_memory().percent > self._ram_threshold_percent:
            self._kq_cache.clear()
            self._kq_cache_current_bytes = 0
            return
        while self._kq_cache_current_bytes + needed_bytes > self._kq_cache_max_bytes and self._kq_cache:
            target_key = min(self._kq_cache, key=lambda k: self._kq_cache[k][2])
            _, size, _ = self._kq_cache.pop(target_key)
            self._kq_cache_current_bytes -= size

    def _cache_dataset(self, cache_key: str, create_fn):
        if cache_key not in self._kq_cache:
            if psutil.virtual_memory().percent > 80.0:
                self._kq_cache.clear()
                self._kq_cache_current_bytes = 0
            # Only create_fn touches the shared .npz on disk, so it is the only part
            # needing the cross-process mutex.  Deep-copying, encoding, splitting and
            # the KQGR probe maths stay OUTSIDE the lock so workers actually
            # parallelise (previously the whole path was serialised node-wide).
            if self.build_lock is not None:
                with self.build_lock:
                    dataset = create_fn()
            else:
                dataset = create_fn()
            ds_size = self._get_dataset_size_bytes(dataset)
            self._age_cache()
            self._evict_smart(ds_size)
            self._kq_cache[cache_key] = [dataset, ds_size, self._AGE_MSB]
            self._kq_cache_current_bytes += ds_size
        else:
            self._kq_cache[cache_key][2] |= self._AGE_MSB
        return copy.deepcopy(self._kq_cache[cache_key][0])


class DatasetInit(DatasetCache, ABC):
    @abstractmethod
    def _create_raw_dataset(self, P: Params): pass

    @abstractmethod
    def _process_dataset(self, dataset, P: Params): pass

    def train(self, P: Params):
        dataset = self._create_raw_dataset(P)
        if hasattr(P.dataset, 'tau') and P.dataset.tau != 0:
            P = copy.deepcopy(P)
            P.dataset.tau = 0
        return self._process_dataset(dataset, P)

    def readout_groups(self, P: Params) -> tuple:
        """The task's extra readouts on the same states as y (glm.ReadoutGroup); () for none."""
        return ()

    def walk_query_probe(self, kq_x: np.ndarray, P: Params) -> tuple | None:
        """A crossed KQGR probe for tasks with a second, task-framing input (PI's
        retroactive query): (levels, x) with x level-major — every probe input under every
        level — for kq_and_gr_metric.compute_walk_query_split. None: the task has none."""
        return None


# ---------------------------------------------------------------------------
# Readout training — the two historical entry points, now thin wrappers over
# the unified GLM module (project.boolean_reservoir.code.glm). The estimator
# lives there: families, links, solvers, ONE scale-aware alpha. This
# file keeps orchestration + scoring glue only.
# ---------------------------------------------------------------------------


def train_readout_ridge(model: BooleanReservoir, dataset, alpha: float, accuracy: AccuracyFunction) -> tuple:
    """Backward-compatible name: fit the Gaussian/identity GLM readout.

    ('ridge' was the historical config name, retired — configs
    use 'gaussian'.) At alpha=0 the fit IS minimum-norm OLS (Jaeger's canonical
    pseudoinverse). The estimator
    (streamed sufficient statistics + scale-aware penalised normal equations,
    intercept unpenalized) lives in glm.collect_sufficient_statistics and
    glm.gaussian_closed_form; see glm.train_readout for the family dispatch.
    """
    return glm.train_readout(model, dataset, family='gaussian', alpha=alpha,
                             accuracy=accuracy, criterion_fn=criterion_fn,
                             score_splits=_score_splits)


def _predict_split(model: BooleanReservoir, dataset, split: str):
    """Model predictions on one split through the trained readout — the single
    split-iteration site (lengths-aware when the dataset carries 'n').
    The TRAIN arrays are stored unprefixed ('x'/'y'/'n', as consumed by the
    matrix collectors); evaluation splits carry the '_<split>' suffix.
    Returns (y_hat, y) or None when the split is absent OR EMPTY — split_dataset
    slices `x_test = x[dev_end:]`, so a test fraction of 0 yields a size-0 array
    with the key PRESENT (the temporal grids run test: 0); the
    model forward cannot reshape a 0-row batch, and an empty split has no
    metrics to report."""
    key = '' if split == 'train' else '_' + split
    x = dataset.data.get('x' + key)
    if x is None or len(x) == 0:
        return None
    y = dataset.data['y' + key]
    n = dataset.data.get('n' + key)
    y_hat = model(x, lengths=n) if n is not None else model(x)
    return y_hat, y


def _score_splits(model: BooleanReservoir, dataset, T, accuracy, score_pair,
                  precomputed: dict | None = None) -> dict:
    """Score every split present (train / evaluation / test) — one call site for
    all optimizer paths, so the train-vs-eval gap (generalization) lands in the
    parquet as accuracy_<split>/loss_<split>. No bare keys: every metric is
    split-explicit, and T.evaluation (= 'dev' in the campaigns) has no alias.
    precomputed: {split: (y_hat, y)} for splits whose predictions the fitter
    already has (the Bernoulli fit holds the train states) — skips a second
    reservoir pass over that split."""
    scored = {}
    for split in dict.fromkeys(('train', T.evaluation, 'test')):
        ps = (precomputed or {}).get(split) or _predict_split(model, dataset, split)
        if ps is None:
            continue
        acc, loss = score_pair(*ps)
        scored[f'accuracy_{split}'] = acc
        scored[f'loss_{split}'] = loss
    return scored


# ---------------------------------------------------------------------------
# GLM (Bernoulli/logit) readout training (moved to glm.py)
# ---------------------------------------------------------------------------


def train_readout_glm(model: BooleanReservoir, dataset, alpha: float,
                      accuracy: AccuracyFunction) -> tuple:
    """Backward-compatible name: fit the Bernoulli/logit GLM readout.

    Deduplicated grouped Newton–IRLS on the state alphabet (exact — see
    glm.dedup_design), decision rule P(y=1|x) > 0.5 ⟺ logit > 0 unchanged,
    reported loss = the fit's own cross-entropy. See glm.train_readout.
    """
    return glm.train_readout(model, dataset, family='bernoulli', alpha=alpha,
                             accuracy=accuracy, criterion_fn=criterion_fn,
                             score_splits=_score_splits)


def train_and_evaluate(model: BooleanReservoir, dataset, record_stats=False, verbose=False,
                       accuracy_fn: AccuracyFunction = None):
    if accuracy_fn is None:
        accuracy_fn = EuclideanDistanceAccuracy().accuracy
    family, alpha = _readout_family(model.P.M.T)
    return glm.train_readout(model, dataset, family=family, alpha=alpha,
                             accuracy=accuracy_fn, criterion_fn=criterion_fn,
                             score_splits=_score_splits)


def train_and_evaluate_heads(model: BooleanReservoir, dataset, heads: list,
                             accuracy_fn: AccuracyFunction = None, groups: tuple = ()) -> dict:
    """train_and_evaluate for every readout head on ONE reservoir run: {'train': {head:
    best_stats}}, plus {group.name: ...} per readout group (see glm.train_readout_heads —
    each head is the single-mode estimator)."""
    if accuracy_fn is None:
        accuracy_fn = EuclideanDistanceAccuracy().accuracy
    family, alpha = _readout_family(model.P.M.T)
    return glm.train_readout_heads(model, dataset, heads, family=family, alpha=alpha,
                                   accuracy=accuracy_fn, criterion_fn=criterion_fn,
                                   score_splits=_score_splits, groups=groups)


def _readout_family(T) -> tuple:
    # ONE family dispatch: the family names are the only valid config
    # names — historical 'ridge'/'logistic' raise with the replacement named.
    return glm.normalize_family(T.optim.name), getattr(T.optim.params, 'alpha', 1e-3)


def train_single_model(yaml_or_checkpoint_path='', parameter_override: Params = None,
                       model=None, save_model=True):
    if model is None:
        load_path = yaml_or_checkpoint_path if yaml_or_checkpoint_path else None
        model = BooleanReservoir(params=parameter_override, load_path=load_path)
    P = model.P
    dataset = P.dataset_init_obj.train(P)
    _, model, train_history = train_and_evaluate(
        model, dataset, record_stats=True, verbose=True, accuracy_fn=P.accuracy_obj
    )
    if save_model:
        model.save()
    return P, model, dataset, train_history
