import os as _os
# Belt-and-suspenders: ensure JULIA_DEPOT_PATH is set before juliacall starts Julia,
# in case this subpackage is imported independently of the parent __init__.py.
_os.environ.setdefault(
    'JULIA_DEPOT_PATH',
    '/tmp/julia_depot:' + _os.environ.get('JULIA_DEPOT_PATH', _os.path.expanduser('~/.julia'))
)
# Cap Julia's GC heap so it returns pages to the OS rather than holding them indefinitely.
# Julia ≥ 1.9; silently ignored on older versions. Override with a larger value for
# workloads that genuinely need more heap (e.g. large path-integration datasets).
_os.environ.setdefault('JULIA_MAX_HEAP_SIZE', '2G')
del _os
try:
    from juliacall import Main as jl  # must precede any torch import
except ImportError:
    pass
