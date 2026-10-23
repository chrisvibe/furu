"""
Channel capacity of a temporal readout (furu): per task head, the information the readouts recover, summed
over the c channels, after Dambre et al. (2012), "Information processing capacity of dynamical systems":

    capacity = sum_c max(0, 1 - BCE_c / H),   H = ln 2 (every head is balanced over exhaustive streams)

A perfectly read channel adds 1, a channel at chance adds 0, so capacity <= c ("bits carried per task"). The clip keeps
over-fitted channels (BCE above chance on dev) from subtracting. Computed at load from the logged per-channel scores
(logging.targets: '<head>' at c = 1, '<head>_c<j>' at c > 1), so the definition can change without a re-run.
"""
import re

import numpy as np

H = np.log(2)
_CHANNEL = re.compile(r'^(?P<head>.+)_c(?P<j>\d+)$')


def head_of(target: str) -> str:
    """'density_w3_d5_c7' -> 'density_w3_d5'; a c = 1 name is its own head."""
    m = _CHANNEL.match(target)
    return m['head'] if m else target


def head_parts(targets) -> 'pd.DataFrame':
    """task / window / delay columns from head names ('density_w3_d5'), for a pandas Series of them."""
    parts = targets.str.extract(r'^(?P<task>[a-z]+)_w(?P<window>\d+)_d(?P<delay>\d+)$')
    return parts.astype({'window': int, 'delay': int})


def capacity(targets: dict, split: str = 'dev') -> dict:
    """{head: capacity} from logging.targets ({target: {'loss_<split>': BCE, ...}} or objects with that attribute)."""
    out = {}
    for name, rec in targets.items():
        loss = rec[f'loss_{split}'] if isinstance(rec, dict) else getattr(rec, f'loss_{split}')
        out[head_of(name)] = out.get(head_of(name), 0.0) + max(0.0, 1.0 - loss / H)
    return out


def extraction(heads: list[str], split: str = 'dev') -> tuple:
    """A custom_load_grid_search_data extraction: one 'cap_<head>' column per task head."""
    return ('cap', lambda p: capacity(p.L.targets, split) if p.L.targets else None, set(heads))
