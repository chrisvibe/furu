import os as _os
# Redirect Julia's volatile depot files (manifest_usage.toml.pid, compiled cache, etc.)
# to node-local /tmp. Without this, many workers concurrently starting Julia all write
# pid-lock files to ~/.julia/logs/ on NFS, which triggers ESTALE (-116) errors that
# cause PythonCall.jl to fail to start. /tmp is always node-local — no NFS, no stale
# handles. The original depot is kept as fallback so installed packages are still found.
_os.environ.setdefault(
    'JULIA_DEPOT_PATH',
    '/tmp/julia_depot:' + _os.environ.get('JULIA_DEPOT_PATH', _os.path.expanduser('~/.julia'))
)
# Must be set before juliacall starts Julia — code/__init__.py also sets this but runs
# after juliacall is already imported here, so the cap would arrive too late without this.
_os.environ.setdefault('JULIA_MAX_HEAP_SIZE', '2G')
del _os
try:
    from juliacall import Main as jl  # must precede any torch import
except ImportError:
    pass
