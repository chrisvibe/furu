import sys
import importlib
from pathlib import Path

# Ensure the code root is on sys.path so 'project.*' imports resolve to the
# project/ package directory and not to any same-named file elsewhere
# (e.g. /opt/conda/envs/boolean_reservoir/bin/project.py).
_CODE_ROOT = str(Path(__file__).parent)
sys.path.insert(0, _CODE_ROOT)
importlib.invalidate_caches()  # force re-scan after sys.path mutation

# code/data and code/out link to ../data and ../out, which a fresh clone lacks (docker/setup.sh makes them); every
# test that writes a dataset or a result would fail on the dangling link, so create them here too.
for _dir in ('data', 'out'):
    (Path(__file__).parent / _dir).resolve().mkdir(parents=True, exist_ok=True)
