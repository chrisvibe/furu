"""Unified GLM readout: one estimator family, two response distributions.

The readout layer is a single linear model over the (fixed) reservoir features,
fitted by maximum likelihood under a generalized linear model per target type —
the paper's umbrella. A family here is (response distribution, link, solver):

    gaussian   N(w·h, σ²)   identity   closed form (lstsq on the scale-aware
                                       penalized normal equations; the α→0 limit
                                       is minimum-norm OLS, Jaeger's pseudoinverse)
    bernoulli  Bern(sigmoid(w·h))      logit     Newton–IRLS on the deduplicated
                                                 state alphabet (exact grouped
                                                 binomial form)

One name per family: the family names
ARE the config names — the response distribution is the branch axis, so the
name names it.

    'gaussian'  N(w·h, σ²)   identity   closed form (lstsq on the scale-aware
                                        penalized normal equations; the α→0 limit
                                        is minimum-norm OLS, Jaeger's pseudoinverse)
    'bernoulli' Bern(sigmoid(w·h))      logit     Newton–IRLS on the deduplicated
                                                  state alphabet (exact grouped
                                                  binomial form)

The historical names 'ridge'/'logistic' are RETIRED: normalize_family rejects
them with a pointed error naming the replacement (no silent map — 'ridge' in
particular is a lie at alpha=0, where the fit is minimum-norm OLS, not ridge).
Old saved params/checkpoints fail loudly at fit time instead of half-working.

ONE alpha everywhere: alpha is scale-aware, penalty = alpha·tr(XᵀX)/p on
the slope weights with the INTERCEPT/bias column unpenalized, identical
definition in both families. The legacy absolute alpha=1e-3 was numerically
inert (cond(XᵀX) ≈ 1e19–1e178) — i.e. effectively OLS — reproducible in the
alpha → 0 limit.

Why two collectors (structural, do NOT unify):
  - the Gaussian closed form streams sufficient statistics (XtX, Xty) per
    batch, because PI campaign designs make full X too large to hold;
  - Newton–IRLS needs X itself (XᵀWX per step), which is affordable on the
    classification grids (m ≈ 1e3–1e4).
Scoring glue (split iteration, loss/accuracy reporting) stays in train_model;
this module is estimation only and imports nothing from it.
"""
import warnings
from dataclasses import dataclass
import numpy as np
import scipy.linalg
import scipy.linalg.blas
from project.boolean_reservoir.code.parameter import TrainLog
from project.boolean_reservoir.code.utils.utils import flush_subnormals

# ---------------------------------------------------------------------------
# Disclosed fixed constants — deterministic, no epochs, no stochasticity
# (mirrors the closed-form virtue of the Gaussian family)
# ---------------------------------------------------------------------------

IRLS_MAX_ITER = 50
IRLS_TOL = 1e-8

# Fit guard: the Newton loop must always end in FINITE weights and a
# flag, never a NaN readout or an uncaught LAPACK error. Measured need: at
# alpha=0 on separable T designs, solve() on the near-singular Hessian returned
# steps that ran ||w|| to 1e90/inf — NaN rows scored as chance, float64 weights
# past float32 max went NaN in set_readout_weights, and lstsq on a NaN Hessian
# killed jobs ("SVD did not converge"). A guard that never fires leaves the
# arithmetic bitwise unchanged; it fires only on rejected steps or failed solves.
IRLS_MAX_HALVINGS = 30      # step-halvings tried before the column stops
IRLS_W_MAX = 1e30           # |w| bound: far above any healthy fit, safely below float32 max
IRLS_F_SLACK = 1e-10        # relative objective slack: fp noise near the optimum is not an increase
SATURATION_EPS = 1e-12      # p(1-p) below this on every row = the logit has saturated


class FitGuardWarning(RuntimeWarning):
    """A readout-fit guard activated. The fit still returns finite
    weights; per-row counts land in TrainLog.fit_guard_events."""


def _guard_warn(msg: str) -> None:
    # stable messages: Python's default filter shows each once per process and
    # call site, so a grid search logs the reason without flooding; the per-row
    # count is in the parquet (fit_guard_events)
    warnings.warn(msg, FitGuardWarning, stacklevel=2)

# Retired config names → family. NOT a validation surface — every real config
# uses the family name; this exists only so old saved params/checkpoints fail
# with the replacement named instead of a bare "unsupported".
RETIRED_NAMES = {
    'ridge': 'gaussian',
    'logistic': 'bernoulli',
}


def normalize_family(name: str) -> str:
    """Canonicalise an optim name to its GLM family; raise on anything else.

    One name per family: 'gaussian' and 'bernoulli' are the
    only valid config names. Historical names raise, naming the replacement —
    no silent map."""
    if name in ('gaussian', 'bernoulli'):
        return name
    if name in RETIRED_NAMES:
        raise ValueError(
            f"optim name '{name}' was retired — use '{RETIRED_NAMES[name]}' (the GLM family name)."
        )
    raise ValueError(
        f"Unsupported readout family '{name}' — supported: "
        f"'gaussian' (Gaussian/identity GLM, closed form) and "
        f"'bernoulli' (Bernoulli/logit GLM, Newton–IRLS). "
        "Online training was intentionally removed; see torch-removal branch notes."
    )


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function (no overflow for |z| large)."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# ---------------------------------------------------------------------------
# Gaussian family (identity link): closed-form MLE
# ---------------------------------------------------------------------------

def _bias_column(bias_scale, m: int) -> np.ndarray:
    bias_scale = np.asarray(bias_scale, dtype=np.float64)
    return (np.full((m, 1), float(bias_scale), dtype=np.float64)
            if bias_scale.ndim == 0 else bias_scale.reshape(m, 1))


def iter_state_batches(model, x_all, n_all, batch_size: int, heads=None):
    """The one reservoir pass every collector uses: yields (start, end, {key: (states,
    bias_col)}) per batch. heads=None -> the model's own readout mode under key None;
    heads=[modes] -> every head from the SAME forward (model.get_states(heads=...)).

    time-series: summed states with bias column = step count s (or the per-sample
    true length n_i for multistep datasets, which carry 'n'), matching forward():
    sum_s(W @ o_s + b) = W @ X_eff + s*b. classification: final state, bias 1.
    """
    model.eval()
    for start in range(0, len(x_all), batch_size):
        end = min(start + batch_size, len(x_all))
        n_batch = None if n_all is None else n_all[start:end]
        out = (model.get_states(x_all[start:end], lengths=n_batch, heads=heads) if heads is not None
               else {None: model.get_states(x_all[start:end], lengths=n_batch)})
        yield start, end, {k: (st, _bias_column(sc, len(st))) for k, (st, sc) in out.items()}


def collect_sufficient_statistics(model, dataset, batch_size: int, heads=None, y=None):
    """Accumulate X^T X and X^T y in one forward pass (see iter_state_batches for the
    design and bias column). Returns (XtX, Xty), or {head: (XtX, Xty)} with heads.
    y: the (m, n_targets) targets, default dataset y — several target sets side by side
    share one XtX.

    Streaming sufficient statistics rather than raw rows is a MEMORY decision,
    not a mathematical one: PI campaign designs are too large to hold X, and the
    Gaussian/identity MLE needs only XtX and Xty. Do NOT rewire the Bernoulli
    collector to this — IRLS needs X itself for XᵀWX per Newton step.
    """
    n_r = model.R.n_nodes
    y_all = dataset.data['y'] if y is None else y
    acc = {}
    for start, end, per_key in iter_state_batches(model, dataset.data['x'], dataset.data.get('n'),
                                                  batch_size, heads):
        y_batch = y_all[start:end].astype(np.float64)
        for k, (states, bias_col) in per_key.items():
            XtX, Xty = acc.setdefault(k, (np.zeros((n_r + 1, n_r + 1)), np.zeros((n_r + 1, y_all.shape[1]))))
            X_aug = np.concatenate([states.astype(np.float64), bias_col], axis=1)   # (m, n_r+1)
            XtX += X_aug.T @ X_aug
            Xty += X_aug.T @ y_batch
    return acc if heads is not None else acc[None]


def _matrix_rank(M: np.ndarray) -> int:
    with flush_subnormals():
        return int(np.linalg.matrix_rank(M))


def gaussian_closed_form(XtX: np.ndarray, Xty: np.ndarray, alpha: float) -> np.ndarray:
    """Solve the scale-aware penalised normal equations of the Gaussian GLM.

    The penalty added to the normal equations is alpha·tr(XtX)/p — relative to
    the mean column energy of the design (one meaning everywhere), so it
    binds regardless of the (astronomically collinear) state scale. The
    intercept (last row/column) is UNPENALIZED, matching the Bernoulli fit and
    making the readout invariant to affine recodings of y.

    lstsq rather than solve: identical on the (nonsingular) regularized arm,
    and for alpha → 0 it degrades to the minimum-norm OLS solution (the
    pseudoinverse Jaeger-style fit) instead of raising on the redundant
    collinear design (exact duplicate columns from input redundancy).

    Returns sol (p, n_out): one column per output, LAST ROW = intercept.
    """
    return _gaussian_solve(XtX, Xty, alpha)[0]


def _gaussian_solve(XtX: np.ndarray, Xty: np.ndarray, alpha: float) -> tuple:
    """gaussian_closed_form plus the numerical rank lstsq computed on the way
    (of XtX + reg — equal to rank(XtX) only at alpha == 0) and the number of
    fit-guard events (solver fallbacks).

    Guard: a solve that fails — LAPACK gelsd raising (PI 2D job 81525, "SVD
    did not converge", once in 629k fits) or returning a non-finite solution without
    raising (PI 2D query job 27638 on one node: NaN weights from finite normal
    equations at alpha 0, rank 254 of 2049; not reproducible on another machine) —
    is retried without flush-to-zero, then with the gelsy driver (complete orthogonal
    factorization — a different algorithm, the same minimum-norm solution: on 27638's
    rank-deficient equations it matches gelsd to 3e-9 and the logged losses to 4 decimals;
    gelss is no substitute, its rank cutoff differs). If every attempt fails, it raises: a NaN
    readout is never logged. A non-finite XtX/Xty is an upstream bug no solver can
    fix, so it raises too."""
    if not (np.all(np.isfinite(XtX)) and np.all(np.isfinite(Xty))):
        raise ValueError("gaussian readout: non-finite XtX/Xty — upstream states or targets contain NaN/inf")
    p = XtX.shape[0]                                 # n_r + 1 (bias column last)
    alpha_abs = alpha * float(np.trace(XtX)) / p
    reg = alpha_abs * np.eye(p, dtype=np.float64)
    reg[-1, -1] = 0.0                                # intercept unpenalized
    A = XtX + reg

    def gelsd_ftz():
        with flush_subnormals():
            return np.linalg.lstsq(A, Xty, rcond=None)

    attempts = [('lstsq (gelsd)', gelsd_ftz),
                ('lstsq (gelsd) without flush-to-zero', lambda: np.linalg.lstsq(A, Xty, rcond=None)),
                ('the gelsy driver', lambda: scipy.linalg.lstsq(A, Xty, lapack_driver='gelsy'))]
    for events, (name, solve) in enumerate(attempts):
        try:
            sol, _, rank, _ = solve()
        except np.linalg.LinAlgError:
            _guard_warn(f"gaussian readout: {name} failed; retrying with the next solver")
            continue
        if np.all(np.isfinite(sol)):
            return sol, int(rank), events
        _guard_warn(f"gaussian readout: {name} returned a non-finite solution; retrying with the next solver")
    raise ValueError("gaussian readout: every solver failed or returned a non-finite solution")


# ---------------------------------------------------------------------------
# Bernoulli family (logit link): deterministic Newton–IRLS
# ---------------------------------------------------------------------------

def collect_design(model, dataset, batch_size: int, heads=None, y=None):
    """Collect RAW design rows for the IRLS fitter: X states (m, n_r), the bias-scale
    column, and y (m, n_out) in one batched pass — the same pass as
    collect_sufficient_statistics (iter_state_batches). Returns the tuple, or
    {head: tuple} with heads.

    Unlike the Gaussian collector, the Newton iteration needs X itself (XᵀWX per
    step). Classification grids are m≈1e3–1e4, so holding X is fine here.
    y: as collect_sufficient_statistics.
    """
    y_all = dataset.data['y'] if y is None else y
    parts = {}
    for start, end, per_key in iter_state_batches(model, dataset.data['x'], dataset.data.get('n'),
                                                  batch_size, heads):
        for k, (states, bias_col) in per_key.items():
            xs, bcs, ys = parts.setdefault(k, ([], [], []))
            xs.append(states.astype(np.float64))
            bcs.append(bias_col)
            ys.append(y_all[start:end].astype(np.float64))
    out = {k: (np.concatenate(xs), np.concatenate(bcs), np.concatenate(ys)) for k, (xs, bcs, ys) in parts.items()}
    return out if heads is not None else out[None]


def dedup_design(X: np.ndarray, y: np.ndarray) -> tuple:
    """Group identical design rows: exact regrouping, not an approximation.

    The Bernoulli log-likelihood over duplicate rows equals the grouped
    binomial(s_i of n_i) form, and weighted least squares gives identical
    normal equations — so fitting on unique rows with count weights IS the same
    estimator, computed on the (much smaller, ~1% unique in near-ordered
    regimes) state alphabet. Returns (Xu (n_u, p), s (n_u, n_out) successes,
    n (n_u,) group sizes).
    """
    Xu, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    s = np.zeros((len(Xu), y.shape[1]), dtype=np.float64)
    np.add.at(s, inverse, y)
    return Xu, s, counts.astype(np.float64)


def _weighted_gram(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    """XᵀCX for diagonal C = diag(c ≥ 0) via BLAS dsyrk on √c·X: the symmetric
    rank-k update does half the flops of the general product (the IRLS Hessian
    is the per-step bottleneck at n_train ≈ 16k). (√c·X)ᵀ is Fortran-ordered,
    so no copy; dsyrk fills the upper triangle, mirrored below."""
    A_t = (X * np.sqrt(c)[:, None]).T
    G = scipy.linalg.blas.dsyrk(1.0, A_t)
    return np.triu(G) + np.triu(G, 1).T


def _bernoulli_objective(Xu: np.ndarray, s_col: np.ndarray, n: np.ndarray,
                         P: np.ndarray, w: np.ndarray) -> float:
    """Penalised grouped-binomial negative log-likelihood — the function whose
    gradient irls_fit uses: Σ n·log(1+e^η) − s·η + ½wᵀPw (P = 2α, so the last
    term is α||w_slopes||²). logaddexp keeps it finite for any finite η."""
    eta = Xu @ w
    return float(np.sum(n * np.logaddexp(0.0, eta) - s_col * eta) + 0.5 * (w @ (P @ w)))


def irls_fit(Xu: np.ndarray, s: np.ndarray, n: np.ndarray, alpha_abs: float,
             max_iter: int = IRLS_MAX_ITER, tol: float = IRLS_TOL) -> tuple:
    """Deterministic Newton–IRLS for the grouped Bernoulli GLM with logit link.

    Xu: (n_u, p) unique design rows (last column = bias scale); s: (n_u, n_out)
    successes; n: (n_u,) group sizes; alpha_abs: absolute L2 penalty on the
    weights with the INTERCEPT UNPENALIZED (textbook practice; makes the fit
    invariant to affine recodings of y). Returns (W, info) — W (p, n_out), one
    column per output, last row = intercept. Objective is strictly concave for
    alpha_abs > 0, so the optimum is unique and the iteration deterministic.

    Fit guard — the loop always ends in finite weights:
      - a Newton step is accepted only if it is finite, keeps |w| ≤ IRLS_W_MAX
        and does not increase the penalised objective (up to IRLS_F_SLACK fp
        noise); otherwise it is halved, up to IRLS_MAX_HALVINGS times (R's
        glm.fit does the same on divergence);
      - if no acceptable step exists, or even the lstsq fallback fails, the
        column stops at its last finite iterate (not converged);
      - convergence is only declared on a full (unhalved) step.
    A guard that never fires leaves the arithmetic bitwise identical to the
    unguarded loop. The guard makes the fit FINITE, not well-posed: at alpha=0
    on separable data the objective keeps falling as ||w|| → ∞, so the runaway
    is bounded by IRLS_W_MAX and max_iter, and reported via `saturated`.

    info carries the FIT DIAGNOSTICS:
      iters            Newton steps used, max over output columns
      converged        False iff some column stopped without the step-norm
                       tolerance firing on a full step — or, at alpha_abs=0,
                       ended saturated (the tolerance then fires because the
                       curvature vanished, not because an optimum was found)
      lstsq_fallbacks  singular-Hessian Newton steps taken by lstsq
      guard_events     step-halvings + guard stops, summed over columns
      saturated        some column ended with p(1-p) < SATURATION_EPS on every
                       row: the fit separates the training set
      w_norm           max over columns of ||w_slopes||_2 (intercept excluded)
      n_unique         len(Xu) — unique design rows fed to the fit
      rank             numerical rank of Xu (matrix_rank)
    """
    p_dim = Xu.shape[1]
    P = 2.0 * alpha_abs * np.eye(p_dim)              # ∇(α||w||²) = 2αw — the
    P[-1, -1] = 0.0                                  # intercept is unpenalized
    W = np.zeros((p_dim, s.shape[1]), dtype=np.float64)
    iters_used, converged_cols, saturated_cols = [], [], []
    lstsq_fallbacks = 0
    guard_events = 0
    for col in range(s.shape[1]):
        s_col = s[:, col]
        w = np.zeros(p_dim, dtype=np.float64)
        f = _bernoulli_objective(Xu, s_col, n, P, w)
        converged = False
        steps = 0
        for _ in range(max_iter):
            prob = sigmoid(Xu @ w)
            g = Xu.T @ (n * prob - s_col) + P @ w       # ∇(CE + α||w_slopes||²)
            H = _weighted_gram(Xu, n * prob * (1.0 - prob)) + P
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                # Saturated (near-separable) groups can zero the working
                # weights; P keeps H positive semi-definite, so the min-norm
                # Newton step is well-defined.
                try:
                    step = np.linalg.lstsq(H, g, rcond=None)[0]
                except np.linalg.LinAlgError:
                    _guard_warn("bernoulli readout: lstsq failed on the Newton system; "
                                "keeping the last finite iterate (fit not converged)")
                    guard_events += 1
                    break
                lstsq_fallbacks += 1
            if not np.all(np.isfinite(step)):
                _guard_warn("bernoulli readout: non-finite Newton step; "
                            "keeping the last finite iterate (fit not converged)")
                guard_events += 1
                break
            t = 1.0
            accepted = False
            for _h in range(IRLS_MAX_HALVINGS + 1):
                w_new = w - t * step
                if np.max(np.abs(w_new)) <= IRLS_W_MAX:
                    f_new = _bernoulli_objective(Xu, s_col, n, P, w_new)
                    if np.isfinite(f_new) and f_new <= f + IRLS_F_SLACK * (1.0 + abs(f)):
                        accepted = True
                        break
                t *= 0.5
            if not accepted:
                _guard_warn("bernoulli readout: no finite, non-increasing Newton step after "
                            "step-halving; keeping the last finite iterate (fit not converged)")
                guard_events += 1
                break
            if t < 1.0:
                _guard_warn("bernoulli readout: Newton step rejected (objective increase or "
                            "|w| > IRLS_W_MAX); step-halving engaged")
                guard_events += 1
            w, f = w_new, f_new
            steps += 1
            if t == 1.0 and np.max(np.abs(step)) <= tol * (1.0 + np.max(np.abs(w))):
                converged = True
                break
        prob = sigmoid(Xu @ w)
        saturated = bool(np.all(prob * (1.0 - prob) < SATURATION_EPS))
        if saturated and alpha_abs == 0:
            converged = False
        W[:, col] = w
        iters_used.append(steps)
        converged_cols.append(converged)
        saturated_cols.append(saturated)
    slopes = W[:-1]
    info = {
        'iters': max(iters_used) if iters_used else 0,
        'converged': all(converged_cols),
        'lstsq_fallbacks': lstsq_fallbacks,
        'guard_events': guard_events,
        'saturated': any(saturated_cols),
        'w_norm': float(np.max(np.linalg.norm(slopes, axis=0))) if slopes.size else 0.0,
        'n_unique': int(len(Xu)),
        'rank': _matrix_rank(Xu),
    }
    return W, info


# ---------------------------------------------------------------------------
# Family dispatch — the single fit entry point
# ---------------------------------------------------------------------------

def fit_gaussian(XtX: np.ndarray, Xty: np.ndarray, alpha: float) -> tuple:
    """Closed-form Gaussian readout from sufficient statistics -> (W, b, fit_diag).

    Fit diagnostics: the closed form has no iteration to converge — one lstsq
    solve IS the optimum — so iters=1, converged is always True and there are no IRLS
    fallbacks. rank(XtX) answers the α→0 well-definedness question for this family
    (rank deficiency of the normal equations, incl. the bias column); n_unique does not
    apply."""
    sol, solve_rank, solver_fallbacks = _gaussian_solve(XtX, Xty, alpha)   # (n_r+1, n_out)
    fit_diag = dict(
        fit_iters=1,
        fit_converged=True,
        fit_lstsq_fallbacks=0,
        fit_guard_events=solver_fallbacks,
        fit_saturated=None,
        fit_w_norm=float(np.max(np.linalg.norm(sol[:-1], axis=0))),
        fit_n_unique=None,
        # lstsq already ranked XtX (same default tolerance) when unregularized
        fit_rank=solve_rank if alpha == 0 else _matrix_rank(XtX),
    )
    return sol[:-1].T.astype(np.float32), sol[-1].astype(np.float32), fit_diag


def fit_bernoulli(X_states: np.ndarray, bias_col: np.ndarray, y: np.ndarray, alpha: float) -> tuple:
    """Newton–IRLS Bernoulli readout on the deduplicated design -> (W, b, fit_diag).

    Fit diagnostics: measured, not assumed — at alpha=0 on a separable design
    the MLE does not exist and the reported fit is wherever IRLS stopped
    (fit_converged=False, runaway fit_w_norm)."""
    X = np.concatenate([X_states, bias_col], axis=1)
    Xu, s_, n = dedup_design(X, y)
    alpha_abs = alpha * float(np.sum((Xu ** 2) * n[:, None])) / Xu.shape[1]
    Wc, info = irls_fit(Xu, s_, n, alpha_abs)
    fit_diag = dict(
        fit_iters=info['iters'],
        fit_converged=info['converged'],
        fit_lstsq_fallbacks=info['lstsq_fallbacks'],
        fit_guard_events=info['guard_events'],
        fit_saturated=info['saturated'],
        fit_w_norm=info['w_norm'],
        fit_n_unique=info['n_unique'],
        fit_rank=info['rank'],
    )
    return Wc[:-1].T.astype(np.float32), Wc[-1].astype(np.float32), fit_diag


def predict_from_states(X_states: np.ndarray, bias_col: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The readout on states the caller already holds: W·x_eff + scale·b — what
    forward() computes on the same states (same float32 weights; float64 accumulation,
    so equal to forward() up to float32 summation order)."""
    return X_states @ W.T.astype(np.float64) + bias_col * b.astype(np.float64)


def score_pair_fn(family: str, T, accuracy, criterion_fn):
    """(y_hat, y) -> (accuracy, loss). The REPORTED LOSS is family semantics — the
    objective the fit minimises: the MSE-criterion for gaussian, the fit's own
    cross-entropy for bernoulli (whose y_hat are raw logits)."""
    if family == 'gaussian':
        T_criterion = criterion_fn(T.criterion)

        def _score_pair(y_hat, y):
            return (float(accuracy(y_hat, y, T.accuracy_threshold)),
                    float(T_criterion(y_hat, y)))
        return _score_pair

    def _score_pair(y_hat_raw, y):
        p_hat = sigmoid(np.asarray(y_hat_raw, dtype=np.float64))
        eps = 1e-12
        p_clip = np.clip(p_hat, eps, 1.0 - eps)
        loss = float(-np.mean(y.astype(np.float64) * np.log(p_clip)
                              + (1.0 - y.astype(np.float64)) * np.log(1.0 - p_clip)))
        acc = float(accuracy(p_hat, y, T.accuracy_threshold))
        return acc, loss
    return _score_pair


def train_readout(model, dataset, family: str, alpha: float, accuracy,
                  criterion_fn, score_splits) -> tuple:
    """Fit the linear readout as a GLM under `family`, push weights to the engine.

    family: canonical GLM family ('gaussian' | 'bernoulli') — the only valid
    config names; historical names raise.
    alpha: THE ONE scale-aware regularisation strength (same definition
    in both families — see the module docstring).
    criterion_fn / score_splits: the scoring glue owned by train_model
    (split iteration and reported loss); the REPORTED LOSS is family semantics
    — the objective the fit actually minimises (score_pair_fn).

    Returns (best_stats, model, []) exactly like the historical per-family
    trainers, so callers are untouched. best_stats carries the split metrics
    PLUS the fit diagnostics (fit_iters/fit_converged/fit_lstsq_fallbacks/
    fit_guard_events/fit_saturated/fit_w_norm/fit_n_unique/fit_rank) — TrainLog fields, so they reach
    the parquet on every grid-search row.
    """
    family = normalize_family(family)
    T = model.P.M.T
    if family == 'gaussian':
        W, b, fit_diag = fit_gaussian(*collect_sufficient_statistics(model, dataset, T.batch_size), alpha)
        precomputed = None                                   # streaming fit keeps no states
    else:
        X_states, bias_col, y = collect_design(model, dataset, T.batch_size)
        W, b, fit_diag = fit_bernoulli(X_states, bias_col, y, alpha)
        # Train predictions from the states the fit already holds: skips a second
        # reservoir pass over the train split (~20% of a T job at 32768 samples).
        precomputed = {'train': (predict_from_states(X_states, bias_col, W, b), y)}
    model.set_readout_weights(W, b)
    model.eval()
    scored = score_splits(model, dataset, T, accuracy, score_pair_fn(family, T, accuracy, criterion_fn),
                          precomputed=precomputed)
    # fit diagnostics ride in best_stats so every TrainLog constructor (here and
    # train_model_parallel._run_training) picks them up — they land in the
    # parquet as L.train.fit_* per row.
    best_stats = {**scored, **fit_diag}
    model.P.L.train = TrainLog(**best_stats)
    return best_stats, model, []


@dataclass(frozen=True)
class ReadoutGroup:
    """Extra readouts a task fits on the same states as y: targets (m, K, n_out), K targets per row.

    index set (PI's step readouts: the position after every step, and q): row i is scored by readout
      index[i] (1-based) against y, the contract being y = targets[i, index[i] - 1] -> ONE stats dict.
      train vs the group is then the cost of serving every index with one readout, at the same data per fit.
    index None (temporal heads: density/parity x window x delay of one stream): readout t is scored on every
      row against targets[:, t] -> {names[t]: stats}.
    per_output (index None only): with n_out > 1 each output column is scored apart, as
      {f'{names[t]}_c{j}': stats} (furu's channels: a pooled score hides a dead channel). n_out 1 keeps names[t].
    name is the log field (LoggingParams / HeadLog); targets, index are dataset keys.
    """
    name: str
    targets: str
    index: str | None = None
    names: tuple = ()
    per_output: bool = False


def _w_norm(W, rows) -> float:
    return float(np.max(np.linalg.norm(W[rows].astype(np.float64), axis=1)))


def train_readout_heads(model, dataset, heads: list, family: str, alpha: float, accuracy,
                        criterion_fn, score_splits, groups: tuple = ()) -> dict:
    """Fit one readout per head on ONE reservoir: {'train': {head: best_stats}}.

    Every head is fitted by exactly the estimator train_readout uses for that readout
    mode (fit_gaussian / fit_bernoulli on that head's states) — only the reservoir
    passes are shared: one pass collects every head's fit statistics, and one pass per
    evaluation split serves every head's scores (predict_from_states, handed to
    score_splits as precomputed predictions). The model's own readout keeps the head
    equal to its O.mode, so a saved model stays consistent with its mode.

    groups: the task's ReadoutGroups -> also {group.name: {head: best_stats}}, or {head: {target name:
    best_stats}} for an un-indexed group. Same passes and design as the y fit: every target is solved
    together as extra columns of y (column-wise the same fit as alone); Bernoulli fits each block (y, then
    each group) apart, since IRLS pools its diagnostics over the columns it is handed.
    """
    family = normalize_family(family)
    T = model.P.M.T
    y = dataset.data['y']
    n_out = y.shape[1]
    # the W/b rows of each block: y's columns, then each group's K * n_out (target-major)
    blocks, end = {'train': slice(0, n_out)}, n_out
    for g in groups:
        blocks[g.name] = slice(end, end + dataset.data[g.targets].shape[1] * n_out)
        end = blocks[g.name].stop
    wide = np.concatenate([y, *(dataset.data[g.targets].reshape(len(y), -1) for g in groups)], axis=1) if groups else y
    # fits[h] = (W, b, {block: fit diagnostics})
    if family == 'gaussian':
        fits = {}
        for h, (XtX, Xty) in collect_sufficient_statistics(model, dataset, T.batch_size, heads, y=wide).items():
            W, b, diag = fit_gaussian(XtX, Xty, alpha)       # one solve: the diagnostics are the
            fits[h] = (W, b, {r: {**diag, 'fit_w_norm': _w_norm(W, rows)} for r, rows in blocks.items()}
                       if groups else {'train': diag})       # design's, but for w_norm per block
        precomputed = {h: {} for h in heads}
        splits = ('train', T.evaluation, 'test')
    else:
        designs = collect_design(model, dataset, T.batch_size, heads, y=wide)
        fits = {}
        for h, (X, bias, Y) in designs.items():
            parts = {r: fit_bernoulli(X, bias, Y[:, rows], alpha) for r, rows in blocks.items()}
            fits[h] = (np.concatenate([p[0] for p in parts.values()]), np.concatenate([p[1] for p in parts.values()]),
                       {r: p[2] for r, p in parts.items()})
        precomputed = {h: {'train': predict_from_states(designs[h][0], designs[h][1], *fits[h][:2])} for h in heads}
        splits = (T.evaluation, 'test')
    for split in dict.fromkeys(splits):
        key = '' if split == 'train' else '_' + split
        x = dataset.data.get('x' + key)
        if x is None or len(x) == 0:
            continue
        preds = {h: [] for h in heads}
        for _, _, per_head in iter_state_batches(model, x, dataset.data.get('n' + key), T.batch_size, heads):
            for h, (states, bias_col) in per_head.items():
                preds[h].append(predict_from_states(states.astype(np.float64), bias_col, *fits[h][:2]))
        for h in heads:
            precomputed[h][split] = np.concatenate(preds[h])
    if model.O.mode in fits:
        model.set_readout_weights(fits[model.O.mode][0][:n_out], fits[model.O.mode][1][:n_out])
    model.eval()
    pair = score_pair_fn(family, T, accuracy, criterion_fn)

    def scores(h, diag, pick):
        """score_splits on the (prediction, target) pairs pick(y_hat_wide, split key) of head h, plus diag."""
        pre = {}
        for split, y_hat in precomputed[h].items():
            y_hat, target = pick(y_hat, '' if split == 'train' else '_' + split)
            if model.output_activation:
                y_hat = model.output_activation(y_hat)
            pre[split] = (y_hat, target)
        return {**score_splits(model, dataset, T, accuracy, pair, precomputed=pre), **diag}

    def block(y_hat, g):    # (m, K, n_out): the group's readouts
        return y_hat[:, blocks[g.name]].reshape(len(y_hat), -1, n_out)

    def own(g):             # row i -> readout index[i], against y
        def pick(y_hat, key):
            idx = dataset.data[g.index + key] - 1
            return block(y_hat, g)[np.arange(len(y_hat)), idx], dataset.data['y' + key]
        return pick

    def each(g, t, j=slice(None)):   # readout t on every row, against target t (output columns j)
        return lambda y_hat, key: (block(y_hat, g)[:, t, j], dataset.data[g.targets + key][:, t, j])

    out = {'train': {h: scores(h, fits[h][2]['train'], lambda y_hat, key: (y_hat[:, :n_out], dataset.data['y' + key]))
                     for h in heads}}
    for g in groups:
        if g.index is not None:
            out[g.name] = {h: scores(h, fits[h][2][g.name], own(g)) for h in heads}
        else:
            rows = lambda t: slice(blocks[g.name].start + t * n_out, blocks[g.name].start + (t + 1) * n_out)
            if g.per_output and n_out > 1:   # one score per output column: (t, j) -> f'{name}_c{j}'
                cols = [(f'{n}_c{j}', t, slice(j, j + 1)) for t, n in enumerate(g.names) for j in range(n_out)]
            else:
                cols = [(n, t, slice(None)) for t, n in enumerate(g.names)]
            out[g.name] = {h: {n: scores(h, {**fits[h][2][g.name], 'fit_w_norm': _w_norm(fits[h][0], rows(t))}, each(g, t, j))
                               for n, t, j in cols} for h in heads}
    return out
