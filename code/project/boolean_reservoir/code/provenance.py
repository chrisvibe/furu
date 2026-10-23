"""
Provenance of a grid row, so a later grid on the same reservoirs is a plain merge instead of a re-measure.

  RESULTS_VERSION  manual: bump it whenever the same params and seeds would build, run or measure something different.
                   test_results_version.py is the tripwire: it fails when pinned reservoirs' results change while
                   RESULTS_VERSION stays the same.
  reservoir_id     what builds the reservoir: its content (reservoir_content) and its seeds. Rows with the same
                   reservoir_id and results_version come from the same reservoir, whichever grid ran them.
  content_seed     grid_search.seeding = 'content': a job's seed comes from the grid seed, the reservoir content and j,
                   not the config index i. A grid that adds axes, KQ/GR settings or readouts then rebuilds the same
                   reservoirs, and settings that don't change the reservoir share it (paired, like dataset.sweep).
  git_commit       the commit the grid ran on, '+dirty' with uncommitted changes, None outside a git checkout.

stamp() writes reservoir_id, results_version and git_commit into P.logging; rows merge on (reservoir_id,
results_version). git_commit is only for the record.
"""
import functools
import hashlib
import json
import subprocess
from pathlib import Path

from project.boolean_reservoir.code.utils.utils import generate_unique_seed

RESULTS_VERSION = 1


def reservoir_content(P) -> dict:
    """The fields that build the reservoir, seeds excluded: model variables, input and reservoir layers, wiring and the
    output node count. Defaults are left out, so a new field whose default keeps the old behaviour moves no id."""
    d = P.M.model_dump(mode='json', exclude_defaults=True,
                       include={'variables', 'input_layer', 'reservoir_layer', 'wiring'})
    for layer in ('input_layer', 'reservoir_layer'):
        d.get(layer, {}).pop('seed', None)
    return {**d, 'output_nodes': P.M.O.n_nodes}


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def reservoir_id(P) -> str:
    return _digest({'content': reservoir_content(P), 'seeds': [P.M.I.seed, P.M.R.seed]})[:16]


def content_seed(grid_seed, P, j: int) -> int:
    return generate_unique_seed(grid_seed, 'content', _digest(reservoir_content(P)), j)


@functools.cache
def git_commit() -> str | None:
    def git(*args):
        return subprocess.run(['git', *args], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=30,
                              check=True).stdout.strip()
    try:
        return git('rev-parse', 'HEAD') + ('+dirty' if git('status', '--porcelain') else '')
    except (OSError, subprocess.SubprocessError):
        return None


def stamp(P) -> None:
    P.L.reservoir_id, P.L.results_version, P.L.git_commit = reservoir_id(P), RESULTS_VERSION, git_commit()
