"""Lazy data provisioning for dashboard tools.

Each `ensure_*` function checks whether a required artifact exists for the
given (config, i, j) and, if missing, computes it on the fly.  Callers get
back the artifact path and never worry about the provisioning step.

Rigid single-purpose tools (`weight_explorer._precompute_one`,
`_record_states_for`) do the actual work; this module is the thin shared
wrapper that decides whether to call them.
"""

from __future__ import annotations

from pathlib import Path

from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.utils.single_run import (
    checkpoint_dir,
    single_run_base,
    states_path,
    weights_path,
)


def ensure_weights(config: str, i: int, j: int = 0) -> Path:
    """Return the checkpoint dir for (config, i, j), training if absent.

    Prints progress to stderr.  Returns the checkpoint *directory* (not the
    weights.npz file) so callers can hand it straight to ``BooleanReservoir``.
    """
    base = single_run_base(load_yaml_config(config).L.out_path)
    wp = weights_path(base, i, j)
    cp = checkpoint_dir(base, i, j)

    if wp.exists():
        return cp

    print(f'Weights missing for (i={i}, j={j}) — precomputing...',
          file=__import__('sys').stderr, flush=True)

    from tools.weight_explorer import _precompute_one

    rc = _precompute_one(config, i, j, force=False)
    if rc != 0:
        raise RuntimeError(
            f'Failed to precompute weights for (i={i}, j={j}) — '
            f'_precompute_one returned {rc}')
    if not wp.exists():
        raise RuntimeError(
            f'Weights still missing after precompute for (i={i}, j={j})')
    print(f'  done -> {cp}', file=__import__('sys').stderr, flush=True)
    return cp


def ensure_states(config: str, i: int, j: int = 0) -> Path:
    """Return the states path for (config, i, j), recording if absent.

    Implicitly ensures weights exist first (recording needs a trained model).
    Returns the history directory that ``BatchedTensorHistoryWriter.reload_history``
    expects.
    """
    # Weights must exist before we can record states.
    ensure_weights(config, i, j)

    base = single_run_base(load_yaml_config(config).L.out_path)
    sp = states_path(base, i, j)

    # states_path resolves through a symlink; check the underlying history files.
    history_files = list(sp.glob('tensor_*.npy')) if sp.exists() else []
    if history_files:
        return sp

    print(f'States missing for (i={i}, j={j}) — recording probe...',
          file=__import__('sys').stderr, flush=True)

    from tools.weight_explorer import _record_states_for

    rc = _record_states_for(config, i, j)
    if rc != 0:
        raise RuntimeError(
            f'Failed to record states for (i={i}, j={j}) — '
            f'_record_states_for returned {rc}')
    if not list(sp.glob('tensor_*.npy')):
        raise RuntimeError(
            f'States still missing after recording for (i={i}, j={j})')
    print(f'  done -> {sp}', file=__import__('sys').stderr, flush=True)
    return sp