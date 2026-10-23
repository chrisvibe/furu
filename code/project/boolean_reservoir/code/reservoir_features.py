"""
Per-reservoir features for "what separates winners from losers" probes (PI and furu): a
reservoir's structure from its tables, and its dynamics from a forward pass on the task's own inputs. Task-agnostic:
the caller hands in the model, the inputs and (for recall) the targets, so PI and temporal probes share the code.

static(model, n_inputs), no simulation (R nodes; the input edge is a driven node's first input):
  sensitivity    mean LUT sensitivity (inputs whose flip flips the output; 1 = the random-Boolean-network critical value)
  constant       fraction of R nodes with a constant table
  input_cover    fraction of driven R nodes whose table depends on the input bit
  input_sens     their mean sensitivity to it
  input_fanout   mean R->R out-degree of the driven R nodes
  reach          fraction of R nodes within HOPS R->R hops of a driven node
  outdeg_sd      spread of the R->R out-degree

dynamic(model, x, start), from the states after every step of x (steps >= start; start = a washout):
  flip_rate      mean fraction of R nodes that change per step
  frozen         fraction of R nodes that never change, over every sample and step
  flip_driven    flip rate of the driven nodes / flip_undriven of the others
  damage_<l>     Derrida-style spread: flip the whole input at step `start`, the mean Hamming distance between the two
                 runs' states l steps later, as a fraction of R (l in DAMAGE_LAGS)
  diversity      mean Hamming distance between the final states of different samples (a cheap KQ analogue)

recall(final, targets), the information final states hold about each binary target, by a ridge readout fitted on half
the samples and scored on the other half: {name: accuracy} (e.g. one input bit per lag: a memory profile).
"""
import numpy as np
from scipy.sparse import csr_matrix

from project.boolean_reservoir.code.lut import lut_sensitivity
from project.boolean_reservoir.code.parameter import Params, load_yaml_config
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.train_model_parallel import BooleanReservoirJobFactory
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations

HOPS = 4
STATIC = ['sensitivity', 'constant', 'input_cover', 'input_sens', 'input_fanout', 'reach', 'outdeg_sd']
DAMAGE_LAGS = (1, 2, 5, 10)


def job_params(config, i: int, j: int = 0, mode: str | None = None, combos: list | None = None) -> Params:
    """Grid job (i, j) of a config, as the grid ran it (its seeds), as a single-readout model for `mode` if given.
    Pass combos (generate_param_combinations of the config) when rebuilding many jobs of one grid."""
    P = load_yaml_config(str(config))
    combos = combos if combos is not None else generate_param_combinations(P)
    p = BooleanReservoirJobFactory(P, combos)(i, j, len(combos), 1, {}).P
    if mode is None:
        return p
    d = p.model_dump()
    d['model']['output_layer'].update(heads=None, mode=mode)
    return type(p).model_validate(d)


def _driven(model: BooleanReservoir, n_inputs: int) -> np.ndarray:
    res = np.arange(model.N_total)[model.res_slice]
    return (model.adj_list_mask[res] & (model.adj_list[res] < n_inputs)).any(axis=1)


def static(model: BooleanReservoir, n_inputs: int) -> dict:
    res = np.arange(model.N_total)[model.res_slice]
    adj, mask = model.adj_list[res], model.adj_list_mask[res]
    sens = lut_sensitivity(model.lut)[res]
    has_input = _driven(model, n_inputs)
    rr = mask & (adj >= n_inputs) & np.isin(adj, res)
    rows, cols = np.nonzero(rr)                           # edge src = adj[row, col] -> dst = res[row]
    A = csr_matrix((np.ones(len(rows)), (adj[rows, cols] - res[0], rows)), shape=(len(res), len(res)))
    outdeg = np.asarray(A.sum(axis=1)).ravel()
    seen = frontier = has_input.copy()
    for _ in range(HOPS):
        frontier = (A.T @ frontier.astype(float) > 0) & ~seen
        seen = seen | frontier
    return dict(sensitivity=float(np.nansum(sens, axis=1).mean()),
                constant=float((np.nansum(sens, axis=1) == 0).mean()),
                input_cover=float((sens[has_input, 0] > 0).mean()),   # the input edge is the first input
                input_sens=float(sens[has_input, 0].mean()),
                input_fanout=float(outdeg[has_input].mean()),
                reach=float(seen.mean()),
                outdeg_sd=float(outdeg.std()))


def step_states(model: BooleanReservoir, x: np.ndarray) -> np.ndarray:
    """(s, m, n_r) uint8 states after every step, in batches of the model's batch size."""
    b = model.T.batch_size
    return np.concatenate([model.get_step_states(x[k:k + b]) for k in range(0, len(x), b)], axis=1)


def dynamic(model: BooleanReservoir, x: np.ndarray, n_inputs: int, start: int = 0, S: np.ndarray | None = None) -> dict:
    """S: step_states(model, x), if the caller already has them."""
    S = (step_states(model, x) if S is None else S)[start:]          # (s', m, n)
    flips = S[1:] != S[:-1]                                           # (s'-1, m, n)
    driven = _driven(model, n_inputs)
    x2 = x.copy()
    x2[:, start] = 1 - x2[:, start]                                   # flip the whole input at step `start`
    S2 = step_states(model, x2)[start:]
    damage = (S != S2).mean(axis=(1, 2))                              # per step since the flip
    final = S[-1].astype(np.int16)
    m = min(len(final), 256)
    diversity = (final[:m, None, :] != final[None, :m, :]).mean() * m / (m - 1)
    out = dict(flip_rate=float(flips.mean()), frozen=float((~flips.any(axis=(0, 1))).mean()),
               flip_driven=float(flips[..., driven].mean()) if driven.any() else np.nan,
               flip_undriven=float(flips[..., ~driven].mean()) if (~driven).any() else np.nan,
               diversity=float(diversity))
    out.update({f'damage_{l}': float(damage[l]) if l < len(damage) else np.nan for l in DAMAGE_LAGS})
    return out


def recall(final: np.ndarray, targets: dict, alpha: float = 1e-3) -> dict:
    """{name: accuracy} of a ridge readout on final states (m, n) for each binary target (m,), fitted on the first
    half of the samples and scored on the second."""
    X = np.hstack([final.astype(np.float64) * 2 - 1, np.ones((len(final), 1))])
    h = len(X) // 2
    G = X[:h].T @ X[:h] + alpha * np.eye(X.shape[1])
    out = {}
    for name, y in targets.items():
        t = np.asarray(y, dtype=np.float64) * 2 - 1
        w = np.linalg.solve(G, X[:h].T @ t[:h])
        out[name] = float(((X[h:] @ w > 0) == (t[h:] > 0)).mean())
    return out
