"""
Tests for history recording configuration.

Verifies that the persist_to_disk param is honoured by the BatchedTensorHistoryWriter
created in BooleanReservoir.init_logging(). Regression for the bug where the param
was ignored and history always wrote to disk regardless of config.
"""
import pytest
import numpy as np
from pathlib import Path
from project.boolean_reservoir.code.parameter import load_yaml_config, Params
from project.boolean_reservoir.code.reservoir import BooleanReservoir

CONFIG = Path(__file__).parent / 'config' / 'sample_model.yaml'


def _make_params(persist_to_disk: bool, out_path: Path) -> Params:
    P = load_yaml_config(CONFIG)
    P.logging.out_path = str(out_path)
    P.logging.history.persist_to_disk = persist_to_disk
    return P


def _run_forward(P: Params):
    dataset = P.dataset_init_obj.train(P)
    x = dataset.data['x'][:4]
    model = BooleanReservoir(params=P)
    model.eval()
    model(x)
    model.flush_history()
    return model


def test_persist_to_disk_false_writes_no_files(tmp_path):
    """persist_to_disk=False must not write any .npy history files."""
    P = _make_params(persist_to_disk=False, out_path=tmp_path)
    model = _run_forward(P)

    history_dir = tmp_path / 'runs'
    npy_files = list(history_dir.rglob('*.npy')) if history_dir.exists() else []
    assert not npy_files, (
        f"persist_to_disk=False but .npy files were written: {npy_files}"
    )


def test_persist_to_disk_true_writes_files(tmp_path):
    """persist_to_disk=True must write history .npy files to disk."""
    P = _make_params(persist_to_disk=True, out_path=tmp_path)
    model = _run_forward(P)

    history_dir = tmp_path / 'runs'
    npy_files = list(history_dir.rglob('tensor_*.npy')) if history_dir.exists() else []
    assert npy_files, (
        "persist_to_disk=True but no tensor_*.npy files were written under runs/"
    )


def test_persist_to_disk_false_history_still_readable(tmp_path):
    """With persist_to_disk=False, history must still be reloadable from memory."""
    P = _make_params(persist_to_disk=False, out_path=tmp_path)
    model = _run_forward(P)

    _, history, expanded_meta, _ = model.history.reload_history()
    assert history.shape[0] > 0, "No history entries returned from in-memory reload"
    assert 'phase' in expanded_meta.columns
    assert 'init' in expanded_meta['phase'].values
