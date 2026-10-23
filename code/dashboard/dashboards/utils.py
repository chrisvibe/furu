"""
Shared utilities for Dash dashboard apps (scatter, polar, trace).
"""
import contextlib
import hmac
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def categorize_object_columns(df, max_uniques: int = 100_000) -> None:
    """Cast low-cardinality string columns to category, in place.

    Boot-memory lever: pandas string/object columns dominate the combined
    grid-search frame (~1.55 GB as object strings across the stack); category
    dtype stores each value once. Columns that are numeric, datetime, or
    high-cardinality are left untouched, as are columns whose values cannot be
    hashed (lists, arrays — nunique raises).

    The cast is STR-FIRST on purpose: values are stringified BEFORE the
    category conversion, so ``astype(str)`` on the result yields exactly the
    strings the original column yielded. A naive ``astype('category')`` can
    change that contract (e.g. int+NaN object columns coerce to float
    categories, so a saved view matching ``'3'`` matches nothing) and zeroed
    kqgr saved-view filters on deploy. Anything that matches on
    ``astype(str)`` — view_filters factor filters, UI dropdowns — keeps
    working, which is the invariant this helper exists to protect.
    """
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        try:
            if s.nunique(dropna=True) <= max_uniques:
                df[col] = s.astype(str).astype('category')
        except (TypeError, ValueError):
            continue

# Shared extraction-cache location — the one definition, used by wsgi.base and
# dashboards.polar. /out/cache is the persistent, content-hash-keyed extraction
# cache: the same host directory the compose services see as /cache
# (deploy/docker-compose.yaml). The env var remains an escape hatch (e.g. a
# scratch cache); the old /tmp/boolean_reservoir_cache default was
# per-container and therefore always cold after a restart.
_DEFAULT_CACHE_DIR = Path(os.environ.get('BOOLEAN_RESERVOIR_CACHE_DIR', '/out/cache'))

# ── Views directory write protection ────────────────────────────────────────
# Two independent rules guard every save/delete of a view file:
#
#   1. PATH BOUNDARY — always enforced, for everyone, admin or not. A write must
#      land inside one of the app's safe roots: the project's own out/dashboard
#      tree (persistent, what the paper cites) or /tmp/dashboard (ephemeral
#      scratch). Nothing outside the project is writable by anyone, whatever
#      password they hold. The admin password never widens this boundary.
#
#   2. DEFAULT-DIR LOCK — the directory an app launches with holds the curated
#      views the paper cites, so on a deploy that sets DASHBOARD_ADMIN_TOKEN it
#      is read-only unless the request carries that token, supplied once as a
#      URL fragment #admin_token=… and then cached in browser localStorage.
#      Visitors playing with a public deploy can still save/delete freely under
#      any other allowed directory, e.g. /tmp/dashboard/scratch, so the sandbox
#      is unchanged.
#
# If DASHBOARD_ADMIN_TOKEN is unset there is no lock and the default dir is
# writable. That is the local-checkout case: whoever ran the generate scripts
# to produce their own out/ owns every view in it and has no token to present.
#
# The consequence is that a PUBLIC deploy MUST set DASHBOARD_ADMIN_TOKEN —
# omitting it leaves the curated views open to anyone. deploy/docker-compose.yaml
# passes it to every dashboard service; that is where to look if this regresses.

_TMP_SUGGESTION = '/tmp/dashboard/scratch'

_ADMIN_ENV = 'DASHBOARD_ADMIN_TOKEN'

# The URL flag that carries the token: #admin_token=<secret>
#
# It rides in the FRAGMENT, not the query string, and that is a security
# property rather than a style choice: fragments are never sent to the server,
# so the secret cannot reach nginx's access log ($request) nor the Referer
# header that every subsequent request on the page carries. The token reaches
# the server only inside the _dash-update-component POST body, which is not
# logged. Moving this back to ?admin_token= would silently reintroduce both
# leaks.
ADMIN_URL_PARAM = 'admin_token'

# The id of the hidden dcc.Store holding the token, shared by EVERY dashboard.
#
# This is a constant rather than a per-app literal for a specific reason:
# dcc.Store(storage_type='local') keys browser localStorage by the bare
# component id, and localStorage is per-origin, so all apps served from this
# host share one cached token — a single #admin_token=… visit authenticates
# every dashboard. Trace once spelled this 'tp-admin-token' to match its own
# id prefix, which silently isolated it; naming it in one place is what stops
# that drift recurring. Do not re-introduce per-app spellings.
ADMIN_STORE_ID = 'admin-token'

# Persistent (project) root + ephemeral scratch root. Used as the default write
# boundary by every dashboard; see rule 1 above.
DEFAULT_SAFE_ROOTS = ['/out/dashboard', '/tmp/dashboard']

# The default views folders: one site root per paper (deploy/sites.yaml), one flat
# folder per app inside it — {site}/{app}/ mirrors the public URL /{site}/{app}/.
# Archived views live in /out/dashboard/old/, non-paper ones in /out/dashboard/other/;
# open those by setting the views dir.
VIEWS_ROOT = Path(os.environ.get('DASHBOARD_VIEWS_ROOT', '/out/dashboard/hegg'))
# furu's views, same layout: /out/dashboard/furu/{design,capacity,analysis,temporal,figure1}/
FURU_VIEWS_ROOT = Path(os.environ.get('DASHBOARD_FURU_VIEWS_ROOT', '/out/dashboard/furu'))


def admin_token_configured() -> bool:
    """True if the deploy set an admin password at all."""
    return bool(os.environ.get(_ADMIN_ENV, '').strip())


def _token_spellings(token):
    """The equivalent spellings of one supplied token.

    A base64 token (openssl rand -base64 32) contains '+' and '=', and reaches
    us through a URL a human typed. Two manglings are near-inevitable and both
    look identical to the eye:

      * surrounding quotes, copied straight out of the env-file line
      * '+' arriving as ' ', because some layer form-decoded the query string

    Neither a quote nor a space can occur in base64, so accepting these costs
    nothing: they are alternate renderings of the same secret, not extra
    secrets. A wrong token is still wrong in every spelling.
    """
    raw = (token or '').strip()
    out = []
    for c in (raw, raw.strip('"').strip("'")):
        for v in (c, c.replace(' ', '+')):
            if v and v not in out:
                out.append(v)
    return out


def _token_from_fragment(hash_):
    """Pull the admin token out of a URL fragment, or None if it is not there.

    Returns None (meaning "no flag present — leave the cache alone") rather than
    '' so that a genuinely empty flag, `#admin_token=`, stays distinguishable and
    still revokes.

    Parts are split as raw strings rather than parsed with URLSearchParams-style
    helpers: a bare `#view_name` segment must survive untouched.
    """
    import urllib.parse as _up
    prefix = f'{ADMIN_URL_PARAM}='
    for part in (hash_ or '').lstrip('#').split('&'):
        if part.startswith(prefix):
            return _up.unquote(part[len(prefix):])
    return None


def match_admin_token(token):
    """Return the canonical token if `token` is a valid spelling of it, else None.

    Callers that only need a yes/no use check_admin_token. The capture callback
    uses this one so that what lands in localStorage is the clean secret rather
    than whatever quoting or '+'-mangling the URL happened to carry.

    Returns None when the env var is unset. That means "this deploy has no admin
    token", so no token can ever validate against it; the default-dir lock is
    skipped entirely in that case rather than being satisfied — see
    check_write_allowed and the module header.
    """
    expected = os.environ.get(_ADMIN_ENV, '').strip()
    if not expected:
        return None
    # compare_digest on every candidate; never short-circuit on a match.
    matched = False
    for candidate in _token_spellings(token):
        if hmac.compare_digest(candidate, expected):
            matched = True
    return expected if matched else None


def check_admin_token(token) -> bool:
    """True iff `token` matches DASHBOARD_ADMIN_TOKEN (in any valid spelling)."""
    return match_admin_token(token) is not None


# Deliberately says nothing about admin mode, and reads identically whether or
# not DASHBOARD_ADMIN_TOKEN is configured — a visitor must not be able to learn
# that a privileged path exists, nor probe whether this deploy has one. The URL
# flag is a private backdoor; discovering it is not part of the UI.
PROTECTED_DIR_MSG = (
    f'⚠ Default directory is read-only — save/delete under a different path '
    f'instead (e.g. {_TMP_SUGGESTION}). Note: /tmp is cleared on server restart.'
)


def is_default_views_dir(vdir: Path, default_vdir: Path) -> bool:
    try:
        return Path(vdir).resolve() == Path(default_vdir).resolve()
    except Exception:
        return False


def check_write_allowed(vdir, default_vdir, safe_roots, admin_token=None):
    """Return an error message if writing to `vdir` is forbidden, else None.

    The safe-roots boundary is checked FIRST and is never lifted by the admin
    password; the password only unlocks the curated default directory. A deploy
    that configures no token has no lock to lift — see the module header.
    """
    if not _is_safe_path(Path(vdir), list(safe_roots)):
        return f'⚠ Directory must be under {[str(r) for r in safe_roots]}'
    if (is_default_views_dir(vdir, default_vdir) and admin_token_configured()
            and not check_admin_token(admin_token)):
        return PROTECTED_DIR_MSG
    return None


def guard_view_write(vdir, default_vdir, safe_roots, admin_token, n_outputs: int):
    """Single guard every save/delete callback returns through.

    Returns None when the write is permitted, otherwise a ready-to-return tuple
    `(error_message, no_update, ...)` padded to the callback's output count.

    Callers only supply their output arity — the policy itself lives in exactly
    one place (check_write_allowed). Previously each dashboard hand-rolled this
    sequence, and two of the six call sites had quietly lost the safe-roots half
    of the check, letting a crafted directory write outside the project.
    """
    err = check_write_allowed(vdir, default_vdir, safe_roots, admin_token)
    if err is None:
        return None
    from dash import no_update
    return (err, *([no_update] * (n_outputs - 1)))


def admin_token_state():
    """The State every save/delete callback appends to read the cached token.

    Exists so no call site spells the store id itself; see ADMIN_STORE_ID.
    """
    from dash import State
    return State(ADMIN_STORE_ID, 'data')


def admin_token_store(store_id: str = ADMIN_STORE_ID):
    """Hidden browser-local cache of the admin token. Renders nothing.

    There is no password box anywhere in the UI. The token arrives once as a URL
    fragment (#admin_token=…), is verified server-side, then cached here in
    localStorage. Every save/delete replays it as a State and the server verifies
    it again — the cache holds the SECRET, never a "logged in" flag, so editing
    localStorage by hand grants nothing.

    Returns a Div holding two stores: the token cache and a dummy sink for the
    clientside URL scrub (see register_admin_token_capture).
    """
    from dash import dcc, html
    return html.Div([
        dcc.Store(id=store_id, storage_type='local'),
        dcc.Store(id=f'{store_id}-url-clean'),
    ], style={'display': 'none'})


def register_admin_token_capture(app, store_id: str = ADMIN_STORE_ID, url_id: str = 'url'):
    """Wire #admin_token=… → verified → localStorage, then scrub it from the URL.

    A WRONG token clears the cache, so `#admin_token=x` is also how you log out.
    """
    from dash import Input, Output, no_update

    @app.callback(
        Output(store_id, 'data'),
        Input(url_id, 'hash'),
        prevent_initial_call=False,
    )
    def _capture_admin_token(hash_):
        supplied = _token_from_fragment(hash_)
        if supplied is None:
            return no_update            # no flag in URL — keep whatever is cached
        # Verify before caching: a bad token revokes admin instead of caching junk.
        # Cache the canonical spelling, so a quoted or '+'-mangled URL still
        # leaves a clean secret in localStorage.
        return match_admin_token(supplied)

    # Scrub the secret from the address bar so it can't be shoulder-surfed or
    # screenshotted into a figure URL, and so it does not linger in the tab's
    # session history. replaceState does not update dcc.Location's `hash`, so the
    # capture callback above still sees the token, and this cannot loop.
    #
    # Only the fragment is rewritten: pathname and search are put back verbatim
    # because these dashboards address views with a bare `?view_name` segment
    # that must survive byte-for-byte.
    app.clientside_callback(
        r"""
        function(hash) {
            if (!hash || hash.indexOf('%s=') === -1)
                return window.dash_clientside.no_update;
            const kept = hash.replace(/^#/, '').split('&')
                             .filter(p => p && !p.startsWith('%s='));
            const frag = kept.length ? '#' + kept.join('&') : '';
            window.history.replaceState(
                null, '', window.location.pathname + window.location.search + frag);
            return window.dash_clientside.no_update;
        }
        """ % (ADMIN_URL_PARAM, ADMIN_URL_PARAM),
        Output(f'{store_id}-url-clean', 'data'),
        Input(url_id, 'hash'),
        prevent_initial_call=False,
    )

# ── Dataset visibility ────────────────────────────────────────────────────────
# DASHBOARD_HIDDEN_DATASETS: comma-separated substrings; any config path containing
# one is excluded from every app. Lets a deploy hide unpublished datasets without
# touching each app's `paths` list.

def is_dataset_hidden(path) -> bool:
    raw = os.environ.get('DASHBOARD_HIDDEN_DATASETS', '')
    terms = [t.strip() for t in raw.split(',') if t.strip()]
    return any(t in str(path) for t in terms)

# ── Aggregation ───────────────────────────────────────────────────────────────

_AGG_OPTIONS = [
    {'label': 'Mean',   'value': 'mean'},
    {'label': 'Median', 'value': 'median'},
    {'label': 'Min',    'value': 'min'},
    {'label': 'Max',    'value': 'max'},
    {'label': 'Std',    'value': 'std'},
]
_AGG_OPTIONS_WITH_RAW = [{'label': 'None (raw)', 'value': 'raw'}] + _AGG_OPTIONS
_DEFAULT_AGG = 'mean'


def aggregate_df(plot_df, x_col, y_col, group_by_cols, y_agg, x_is_binned=False, extra_agg_cols=None):
    if y_agg == 'raw' or (not group_by_cols and not x_is_binned):
        return plot_df
    group_cols = list(dict.fromkeys(group_by_cols + [x_col]))
    group_cols = [c for c in group_cols if c in plot_df.columns]
    if y_col in group_cols:
        return plot_df
    cols_to_agg = [c for c in ([y_col] + (extra_agg_cols or []))
                   if c in plot_df.columns and c not in group_cols]
    named_agg = {col: (col, y_agg) for col in cols_to_agg}
    named_agg['_agg_count'] = (cols_to_agg[0], 'count')
    agged = plot_df.groupby(group_cols, dropna=False).agg(**named_agg).reset_index()
    if 'design' not in agged.columns:
        first, *rest = group_cols
        s = agged[first].astype(str)
        if rest:
            s = s.str.cat([agged[c].astype(str) for c in rest], sep=' | ')
        agged['design'] = s
    return agged

# ── View file I/O ─────────────────────────────────────────────────────────────

def _views_dir(path_str):
    return Path(path_str.strip()) if path_str and path_str.strip() else Path('/tmp/dashboard')

def _list_views(views_dir):
    if not views_dir.exists():
        return []
    return sorted(p.stem for p in views_dir.glob('*.json'))

def _view_dropdown_options(views_dir):
    return [{'label': name, 'value': name} for name in _list_views(views_dir)]

def view_file_path(name: str, vdir: Path) -> Path:
    return vdir / f'{name}.json'

def save_view_json(name: str, data: dict, vdir: Path) -> Path:
    vdir.mkdir(parents=True, exist_ok=True)
    p = view_file_path(name, vdir)
    p.write_text(json.dumps(data, indent=2, default=str))
    return p

def load_view_json(name: str, vdir: Path) -> 'dict | None':
    p = view_file_path(name, vdir)
    return json.loads(p.read_text()) if p.exists() else None

def delete_view_json(name: str, vdir: Path) -> bool:
    p = view_file_path(name, vdir)
    if p.exists():
        p.unlink()
        return True
    return False

def _is_safe_path(p: Path, safe_roots: list) -> bool:
    try:
        resolved = Path(p).resolve()
        roots = [Path(r).resolve() for r in safe_roots]
    except Exception:
        return False
    return any(resolved == r or r in resolved.parents for r in roots)

def _fmt(n: int) -> str:
    if n >= 1_000_000: return f'{n/1_000_000:.1f}M'
    if n >= 10_000:    return f'{n/1_000:.0f}k'
    if n >= 1_000:     return f'{n/1_000:.1f}k'
    return str(n)

def _make_eval_ns(df):
    import numpy as np
    ns = {col: df[col] for col in df.columns}
    ns['np'] = np
    ns.update({'int': int, 'float': float, 'str': str, 'bool': bool,
               'True': True, 'False': False, 'None': None, 'abs': abs,
               'round': round, 'min': min, 'max': max, 'len': len, 'list': list})
    return ns

def parse_and_apply_fields(working_df, fields_text):
    """Parse 'name = expr' lines and eval onto df. Returns (df, info, error)."""
    if not fields_text or not fields_text.strip():
        return working_df, '', ''
    errors = []
    added = []
    eval_ns = None
    for i, line in enumerate(fields_text.strip().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            errors.append(f'L{i}: missing "=" in "{line}"')
            continue
        name, expr = line.split('=', 1)
        name = name.strip()
        expr = expr.strip()
        if not name or not expr:
            errors.append(f'L{i}: empty name or expression')
            continue
        try:
            working_df[name] = working_df.eval(expr)
            added.append(name)
        except Exception:
            try:
                if eval_ns is None:
                    eval_ns = _make_eval_ns(working_df)
                result = eval(expr, {"__builtins__": {}}, eval_ns)  # noqa: S307
                working_df[name] = result
                eval_ns[name] = working_df[name]
                added.append(name)
            except Exception as e:
                errors.append(f'L{i} ({name}): {e}')
    return working_df, (f'Added: {", ".join(added)}' if added else ''), ('\n'.join(errors) if errors else '')

def make_combo_column(df, factors, keep=None, exclude=None, return_as_str=True):
    if keep is not None:
        factors_subset = [f for f in factors if f in keep]
    elif exclude is not None:
        factors_subset = [f for f in factors if f not in exclude]
    else:
        factors_subset = factors

    if return_as_str:
        first, *rest = factors_subset if factors_subset else [factors_subset]
        s = df[first].astype(str)
        combo = s.str.cat([df[c].astype(str) for c in rest], sep='_') if rest else s
    else:
        combo = df[factors_subset].apply(tuple, axis=1)

    return combo, factors_subset


FIGURE_PRESETS_FILE = 'figure_presets.yaml'
_presets_cache = {}


def load_figure_presets(views_dir) -> dict:
    """A site's shared figure presets: the nearest figure_presets.yaml at or above views_dir, up to
    /out/dashboard (one per paper site, e.g. /out/dashboard/hegg/). {} when there is none.
      series: {name: {color, label?, dash?}}  one colour per series across every figure of the site. color is
                     a palette index (or an offset dict, e.g. {h: 3}). A trace takes its series' colour
                     when its group is the series name, or when the view's color_offsets name it
                     ({"series": name}); explicit offsets win. label is the legend name it exports as; dash its
                     line style (plotly's names, e.g. a head's delay), unless the view sets one.
      axis_titles: {column: title}  export axis titles, for views with an _extras.figure block
      figure: {...}  defaults under every view's _extras.figure (same keys as apply_figure_options)
      factors: {column: {levels, scale?, range?, label?, tick?}}  an ordered colour factor: a view coloured by
                     the column takes the scale sampled evenly over the listed levels (so a level keeps
                     its colour whatever subset a view shows), and exports level v as label.format(v);
                     tick (a format or a {level: text} map) is the level's compact spelling, which a polar's
                     direction labels join when every factor of the direction has one ('O64')
      text: {title?, axis_title?, ticks?, legend?}  export text sizes under ?font=, as ratios of the
                     font; a missing key keeps the code default (scatter's title ratios; ticks and
                     legend at the furniture scale)
    Re-read when the file changes, so editing it needs no app restart."""
    d = Path(views_dir).resolve() if views_dir else None
    while d is not None:
        f = d / FIGURE_PRESETS_FILE
        if f.is_file():
            mtime = f.stat().st_mtime
            hit = _presets_cache.get(f)
            if hit is None or hit[0] != mtime:
                import yaml
                hit = (mtime, yaml.safe_load(f.read_text()) or {})
                _presets_cache[f] = hit
            return hit[1]
        if d.parent == d or d == Path('/out/dashboard') or d.name == 'dashboard':
            break
        d = d.parent
    return {}


def preset_series(presets: dict, name) -> 'str | None':
    """The series a trace or group name belongs to: the name itself, or its group part
    ('<group> [<y>]'), or one ' | '-joined part of a combined group. None when no series matches."""
    series = (presets or {}).get('series') or {}
    if not series or not isinstance(name, str):
        return None
    if name in series:
        return name
    group = name.rsplit(' [', 1)[0] if name.endswith(']') else name
    for part in [group, *group.split(' | ')]:
        if part in series:
            return part
    return None


def preset_color_offset(presets: dict, offset):
    """Resolve a {"series": name} colour offset to that series' palette offset; others pass through."""
    if isinstance(offset, dict) and 'series' in offset:
        color = (((presets or {}).get('series') or {}).get(offset['series']) or {}).get('color')
        if color is None:
            return None
        return color if isinstance(color, dict) else {'d': int(color)}
    return offset


def group_key(v):
    """Numeric groups ('1', '16', '128') in numeric order, others as text after them."""
    try:
        return (0, float(v), '')
    except (TypeError, ValueError):
        return (1, 0.0, str(v))


def level_name(v) -> str:
    """A factor level as the presets spell it: an integer-valued number without its '.0' (a grid's k̄ 1.0 and an
    analysis table's 1 are both '1'), anything else as str."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f.is_integer() else str(v)


def preset_series_dash(presets: dict, name) -> 'str | None':
    """The line dash the site presets give the series a trace or group name belongs to; None when they give none."""
    series = preset_series(presets, name)
    return ((((presets or {}).get('series') or {}).get(series) or {}).get('dash')) if series else None


def preset_series_symbol(presets: dict, name) -> 'str | None':
    """The marker symbol the site presets give any part of a trace or group name ('density_w5_d5 | xor': xor's),
    so a second colour factor can set the marker while the first sets colour and dash; None when none does."""
    series = (presets or {}).get('series') or {}
    if not series or not isinstance(name, str):
        return None
    group = name.rsplit(' [', 1)[0] if name.endswith(']') else name
    for part in [group, *group.split(' | ')]:
        if (series.get(part) or {}).get('symbol'):
            return series[part]['symbol']
    return None


def preset_factor_symbol(presets: dict, columns, name) -> 'str | None':
    """The marker symbol the site presets give a level of a colour factor (factors.<column>.symbols), for a group
    name whose ' | ' parts are the levels of `columns` in order ('0.1 | 2' with [R_self_loops, I_ticks]: ticks 2's);
    None when none does."""
    factors = (presets or {}).get('factors') or {}
    if not isinstance(name, str):
        return None
    parts = (name.rsplit(' [', 1)[0] if name.endswith(']') else name).split(' | ')
    for column, part in zip([columns] if isinstance(columns, str) else (columns or []), parts):
        symbols = (factors.get(column) or {}).get('symbols') or {}
        found = next((v for k, v in symbols.items() if level_name(k) == level_name(part)), None)
        if found:
            return found
    return None


def preset_level_label(presets: dict, column, v) -> str:
    """A level as the site presets spell it in a legend: the column's factor label ('r = {}') filled with the level,
    else the level's own name."""
    spec = ((presets or {}).get('factors') or {}).get(column) or {}
    return spec['label'].format(v) if spec.get('label') else level_name(v)


def preset_level_tick(presets: dict, column, v):
    """A level's compact spelling for a figure's angle or axis tick: the column's factor tick, either a {level:
    text} map or a format ('{}'), filled with the level; None when the presets give the column none."""
    tick = (((presets or {}).get('factors') or {}).get(column) or {}).get('tick')
    if isinstance(tick, dict):
        return tick.get(level_name(v))
    return tick.format(v) if tick else None


def preset_axis_title(presets: dict, key, default=None):
    """The export title the site presets give a column (or any key, e.g. 'acc (max)'), else default."""
    return ((presets or {}).get('axis_titles') or {}).get(key, default)


def preset_series_labels(presets: dict) -> dict:
    """{series: label} for the series the site presets give an export label."""
    series = (presets or {}).get('series') or {}
    return {k: v['label'] for k, v in series.items() if isinstance(v, dict) and v.get('label')}


def preset_factor_colors(presets: dict, column) -> dict:
    """{level_name(level): colour} for an ordered colour factor of the presets; {} when the column is not one."""
    spec = ((presets or {}).get('factors') or {}).get(column) or {}
    levels = spec.get('levels') or []
    if not levels:
        return {}
    from plotly.colors import sample_colorscale
    lo, hi = spec.get('range') or (0.0, 1.0)
    at = [lo + (hi - lo) * i / max(1, len(levels) - 1) for i in range(len(levels))]
    return dict(zip(map(level_name, levels), sample_colorscale(spec.get('scale') or 'Viridis', at)))


def resolve_figure_options(options, presets: dict, x_col=None, y_cols=(), color_col=None):
    """A view's _extras['figure'] with the site presets filled in: presets['figure'] as defaults,
    axis titles from presets['axis_titles'] (x column; first y column), legend names for the levels
    of an ordered colour factor (presets['factors'], when the view colours by one; the view's own
    names win), and names that refer to a series exported as that series' label. Views without a
    figure block are left alone (opt-in)."""
    if not options:
        return options
    presets = presets or {}
    out = {**(presets.get('figure') or {}), **options}
    factor = ((presets.get('factors') or {}).get(color_col) or {})
    if factor.get('label') and factor.get('levels'):
        out['names'] = {**{str(v): preset_level_label(presets, color_col, v) for v in factor['levels']},
                        **(out.get('names') or {})}
    titles = presets.get('axis_titles') or {}
    x_key, y_key = x_col, (y_cols[0] if y_cols else None)
    if out.get('swap_axes'):    # titles name the axes as drawn
        x_key, y_key = y_key, x_key
    if 'x_title' not in out and x_key in titles:
        out['x_title'] = titles[x_key]
    if 'y_title' not in out and y_key in titles:
        out['y_title'] = titles[y_key]
    if 'color_title' not in out and color_col in titles:
        out['color_title'] = titles[color_col]
        if out.get('bin2d'):     # the colour is a per-bin statistic of the column, so the title names it
            out['color_title'] = f"{out['bin2d'].get('agg', 'median')} {out['color_title']}"
    series = presets.get('series') or {}
    labels = preset_series_labels(presets)
    if labels and 'series_labels' not in out:
        out['series_labels'] = labels
    if 'category_labels' not in out and (labels or titles):
        out['category_labels'] = {**titles, **labels}   # a category on an axis: a column (its title) or a series
    if out.get('names'):
        out['names'] = {k: ((series.get(v) or {}).get('label', v) if isinstance(v, str) else v)
                        for k, v in out['names'].items()}
    return out


def _translucent(color: str, alpha: float) -> str:
    """A CSS colour ('#rrggbb', 'rgb(r,g,b)' or a name plotly knows) at the given opacity, as rgba()."""
    from plotly.colors import unlabel_rgb, hex_to_rgb
    named = {'white': (255, 255, 255), 'black': (0, 0, 0)}
    if color in named:
        rgb = named[color]
    elif color.startswith('#'):
        rgb = hex_to_rgb(color)
    else:
        rgb = unlabel_rgb(color)
    return 'rgba({}, {}, {}, {})'.format(*(int(c) for c in rgb), alpha)


_REF_LINE = 'rgba(128,128,128,0.8)'


def facet_cells(fig) -> list:
    """The panels of a faceted figure (scatter's facet option), as [{row, col, rv, cv, x, y}] with x/y the panel's
    axis references ('x2', 'y2'); [] for a figure of one plot."""
    meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    return list((meta.get('facet') or {}).get('cells') or [])


def _axis_key(ref: str) -> str:
    """'x2' -> 'xaxis2' (the layout key of a trace's axis reference)."""
    return f'{ref[0]}axis{ref[1:]}'


def _panels(fig) -> list:
    """(xref, yref) of every plot panel: each facet, or the one plot."""
    return [(c['x'], c['y']) for c in facet_cells(fig)] or [('x', 'y')]


def _in_panel(t, x, y) -> bool:
    return (t.xaxis or 'x') == x and (t.yaxis or 'y') == y


def _numbers(values) -> list:
    return [v for v in (values if values is not None else []) if isinstance(v, (int, float)) and v == v]


def _trace_rows(t, rows):
    """The rows behind a trace's points: scatter puts each point's row index first in its customdata (-1 when
    aggregated). None when the trace has no such rows."""
    if rows is None or t.customdata is None or len(t.customdata) == 0:
        return None
    try:
        idx = [int(c[0]) for c in t.customdata]
        return None if min(idx) < 0 else rows.loc[idx]
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _trace_y(t) -> 'str | None':
    """The y column a trace plots in a multi-y figure (its generated name '<group> [<y>]'), else None."""
    name = t.name or ''
    return name.rsplit(' [', 1)[1][:-1] if name.endswith(']') and ' [' in name else None


def _for(spec: dict, t) -> bool:
    """Whether a per-trace option applies to t: every markers trace, or with spec['for'] = y only those of that y."""
    if (getattr(t, 'mode', None) or 'markers') == 'lines':
        return False
    return 'for' not in spec or _trace_y(t) == spec['for'] or (_trace_y(t) is None and t.name == spec['for'])


def _trace_color(t):
    for c in (getattr(t.marker, 'color', None), getattr(t.line, 'color', None)):
        if isinstance(c, str):
            return c
    return _REF_LINE


def _remember(t, **generated):
    """Keep a trace's generated name / mode in meta.name0 / meta.mode0 before the figure options change them:
    saved views key colours and visibility by them (save_view and the page's legend listeners read them back)."""
    meta = t.meta if isinstance(t.meta, dict) else {}
    t.meta = {**generated, **meta}


def _rename(t, name):
    _remember(t, name0=t.name)
    t.name = name


def _swap_axes(fig):
    """Transpose a figure: x and y of every trace (error bars too) and the content of every axis pair."""
    for t in fig.data:
        x, y = t.x, t.y
        t.x, t.y = y, x
        if hasattr(t, 'error_x'):
            ex, ey = t.error_x.to_plotly_json(), t.error_y.to_plotly_json()
            t.error_x, t.error_y = ey, ex
    keys = ('title', 'range', 'type', 'autorange', 'tickmode', 'tickvals', 'ticktext', 'dtick', 'tick0', 'tickformat')
    for xref, yref in _panels(fig):
        xa, ya = fig.layout[_axis_key(xref)], fig.layout[_axis_key(yref)]
        xs, ys = ({k: v for k, v in a.to_plotly_json().items() if k in keys} for a in (xa, ya))
        for a, new in ((xa, ys), (ya, xs)):
            a.update({k: None for k in keys})
            a.update(new)


def _add_error_bars(fig, specs, rows):
    for spec in specs:
        for t in fig.data:
            r = _trace_rows(t, rows) if _for(spec, t) else None
            if r is None:
                continue
            y = np.asarray(t.y, dtype=float)
            t.error_y = dict(type='data', symmetric=False, array=r[spec['hi']].to_numpy(float) - y,
                             arrayminus=y - r[spec['lo']].to_numpy(float), thickness=spec.get('width', 2.5),
                             width=spec.get('cap', 0))


def _bin2d(fig, spec, rows):
    """Per panel, the shown points' rows binned on spec x and y (bin widths dx, dy) and drawn as a heatmap of spec z's
    agg (default median) on the figure's colour axis, a bin with fewer than min_n rows left blank; the points are
    hidden. For a cloud too dense to show its colour (e.g. 90k reservoirs), and a vector file of a few hundred cells."""
    import plotly.graph_objects as go
    x, y, z = spec['x'], spec['y'], spec['z']
    dx, dy, agg, min_n = spec.get('dx', 1), spec.get('dy', 1), spec.get('agg', 'median'), spec.get('min_n', 1)
    panels = {}
    for t in fig.data:
        r = _trace_rows(t, rows) if t.visible not in (False, 'legendonly') else None
        if r is not None and {x, y, z} <= set(r.columns):
            panels.setdefault((t.xaxis or 'x', t.yaxis or 'y'), []).append((t, r))
    for (xref, yref), members in panels.items():
        r = pd.concat([m[1] for m in members])
        for t, _ in members:
            t.visible = False
        bx, by = (r[x] // dx) * dx + dx / 2, (r[y] // dy) * dy + dy / 2
        g = r[z].groupby([by, bx])
        cell = g.agg(agg).where(g.size() >= min_n).unstack()
        fig.add_trace(go.Heatmap(x=cell.columns, y=cell.index, z=cell.to_numpy(), coloraxis='coloraxis',
                                 xaxis=xref, yaxis=yref, hoverongaps=False, showlegend=False,
                                 hovertemplate=f'{x} %{{x}}<br>{y} %{{y}}<br>{agg} {z} %{{z:.3f}}<extra></extra>'))


def _dodge_x(fig, step):
    """x as evenly spaced categories (its levels in order, spelled as ticks), each panel's shown traces side by side
    around their level, step apart (in category widths), so overlapping points and error bars separate."""
    levels = sorted({v for t in fig.data if getattr(t, 'x', None) is not None for v in t.x}, key=group_key)
    pos = {v: i for i, v in enumerate(levels)}
    by_axis = {}
    for t in fig.data:
        if getattr(t, 'x', None) is not None and t.visible not in (False, 'legendonly'):
            by_axis.setdefault(t.xaxis or 'x', []).append(t)
    for traces in by_axis.values():
        for i, t in enumerate(traces):
            off = (i - (len(traces) - 1) / 2) * step
            t.x = [pos[v] + off for v in t.x]
    fig.update_xaxes(type='linear', tickmode='array', tickvals=list(range(len(levels))),
                     ticktext=[level_name(v) for v in levels], range=[-0.5, len(levels) - 0.5])


def _add_rings(fig, specs, rows):
    """An open circle around each shown point whose row matches spec['where'] (a pandas expression, e.g. a reservoir
    at accuracy 1), with spec['label'] as one legend entry."""
    import plotly.graph_objects as go
    for spec in specs:
        shown = False
        for t in list(fig.data):
            r = _trace_rows(t, rows) if _for(spec, t) and t.visible not in (False, 'legendonly') else None
            if r is None:
                continue
            hit = r.eval(spec['where']).to_numpy(bool)
            if not hit.any():
                continue
            # the ring takes its points' trace type: an svg scatter sits under every WebGL trace, so a plain ring
            # around scattergl dots is hidden by them
            ring = go.Scattergl if t.type == 'scattergl' else go.Scatter
            fig.add_trace(ring(x=np.asarray(t.x)[hit], y=np.asarray(t.y)[hit], mode='markers', xaxis=t.xaxis,
                               yaxis=t.yaxis, name=spec.get('label', spec['where']), legendgroup=f"ring:{spec['where']}",
                               showlegend=not shown, hoverinfo='skip',
                               marker=dict(symbol='circle-open', size=spec.get('size', 12),
                                           color=spec.get('color', _REF_LINE), line=dict(width=2))))
            shown = True


def _add_arrows(fig, spec, rows):
    """An arrow per point, from (x0, y0) by (dx, dy): columns of the point's row, or numbers."""
    for t in fig.data:
        r = _trace_rows(t, rows) if _for(spec, t) else None
        if r is None:
            continue
        x0, y0, dx, dy = (r[spec[k]].to_numpy(float) if isinstance(spec[k], str) else np.full(len(r), float(spec[k]))
                          for k in ('x0', 'y0', 'dx', 'dy'))
        xref, yref = t.xaxis or 'x', t.yaxis or 'y'
        for a, b, c, d in zip(x0, y0, dx, dy):
            if np.isnan([a, b, c, d]).any() or (c == 0 and d == 0):
                continue
            fig.add_annotation(x=a + c, y=b + d, ax=a, ay=b, xref=xref, yref=yref, axref=xref, ayref=yref, text='',
                               showarrow=True, arrowhead=spec.get('head', 2), arrowsize=spec.get('size', 1),
                               arrowwidth=spec.get('width', 1.5), arrowcolor=spec.get('color') or _trace_color(t))


def _share_legend_entries(fig):
    """Traces with the same name and kind (a fitted line, or points) in the same legend share one entry (the first
    that shows one) and toggle together: a series drawn in several panels, or one name given to several y columns.
    A series' points and its fitted line keep an entry each unless legend: per-series joins them."""
    def key_of(t):
        return t.legend or 'legend', t.name, getattr(t, 'mode', None) == 'lines' and 'mode0' not in (t.meta or {})

    first = {}
    for t in fig.data:
        if not t.name or t.showlegend is False or t.visible is False:
            continue
        key = key_of(t)
        if key in first:
            t.showlegend = False
            t.legendgroup = first[key].legendgroup
        else:
            t.legendgroup = t.legendgroup or f'{key[0]}:{t.name}'
            first[key] = t
    for t in fig.data:   # the hidden twins (per-series markers, a null-named y) follow their entry's group
        key = key_of(t)
        if key in first and t is not first[key] and t.showlegend is False:
            t.legendgroup = first[key].legendgroup


def _place_axis_titles(fig, options):
    """x/y titles: on every axis of a plot; on a faceted figure the x title on the bottom row (or per row, x_titles:
    {row value: title}) and the y title on the first column. A title of '' clears the column name."""
    cells = facet_cells(fig)
    x_title, y_title, by_row = options.get('x_title'), options.get('y_title'), options.get('x_titles') or {}
    if not cells:
        if x_title is not None:
            fig.update_xaxes(title_text=x_title)
        if y_title is not None:
            fig.update_yaxes(title_text=y_title)
        return
    last = max(c['row'] for c in cells)
    for c in cells:
        if x_title is not None or by_row:
            axis = fig.layout[_axis_key(c['x'])]
            axis.title.text = by_row.get(str(c['rv']), '') if by_row else x_title if c['row'] == last else ''
            if axis.title.text:   # a row with its own x title shows its own tick labels
                axis.showticklabels = True
        if y_title is not None:
            fig.layout[_axis_key(c['y'])].title.text = y_title if c['col'] == 0 else ''
    if options.get('y_title_span'):   # one y title for the grid, centred on its height (fit_furniture places it)
        first = [c for c in cells if c['col'] == 0]
        axes = [fig.layout[_axis_key(c['y'])] for c in first]
        text = y_title if y_title is not None else axes[0].title.text
        for a in axes:
            a.title.text = ''
        lo, hi = min(a.domain[0] for a in axes), max(a.domain[1] for a in axes)
        fig.update_layout(meta={**(fig.layout.meta or {}), 'grid_y_axis': _axis_key(first[0]['y'])})
        fig.add_annotation(text=text, name='grid_y_title', textangle=-90, showarrow=False,
                           xref='paper', x=fig.layout[_axis_key(first[0]['x'])].domain[0], xanchor='right',
                           yref='paper', y=(lo + hi) / 2, yanchor='middle', font_size=round(1.2 * _FONT_BASE_PX))


def _facet_legends(fig, where: str):
    """One legend per facet column, inside that column's bottom panel at `where` ('bottom right', 'top left', ...),
    or with 'below' under the column, at the foot of the figure (the bottom margin grows to hold it)."""
    cells = facet_cells(fig)
    last = max(c['row'] for c in cells)
    bg = _translucent(fig.layout.paper_bgcolor or fig.layout.template.layout.paper_bgcolor or 'white', 0.8)
    for c in (c for c in cells if c['row'] == last):
        legend = 'legend' if c['col'] == 0 else f'legend{c["col"] + 1}'
        for t in fig.data:
            if any(_in_panel(t, o['x'], o['y']) for o in cells if o['col'] == c['col']):
                t.legend = legend
        xd, yd = fig.layout[_axis_key(c['x'])].domain, fig.layout[_axis_key(c['y'])].domain
        if where == 'below':
            fig.update_layout({legend: dict(x=xd[0], xanchor='left', yref='container', y=0, yanchor='bottom')})
            continue
        v, h = where.split()
        fig.update_layout({legend: dict(x=xd[1] - 0.005 if h == 'right' else xd[0] + 0.005,
                                        y=yd[0] + 0.01 if v == 'bottom' else yd[1] - 0.01,
                                        xanchor=h, yanchor=v, bgcolor=bg)})


def _add_reference_marks(fig, options, legendrank):
    """hlines / dlines as legend-labelled line traces in every panel; vlines and hbands as shapes."""
    shown = set()

    def line(x, y, xref, yref, label, dash):
        fig.add_scatter(x=x, y=y, mode='lines', name=label, showlegend=bool(label) and label not in shown,
                        legendgroup=f'ref:{label}' if label else None, xaxis=xref, yaxis=yref, hoverinfo='skip', legendrank=legendrank,
                        line=dict(color=_REF_LINE, width=1, dash=dash))
        shown.add(label)

    for xref, yref in _panels(fig):
        data = [t for t in fig.data if _in_panel(t, xref, yref)]
        xs = [x for t in data for x in _numbers(t.x)]
        ys = [y for t in data for y in _numbers(t.y)]
        # a line trace over the data's x span, labelled in the legend: it scales as legend furniture
        # (an annotation scales as data text, ~7x at paper export) and never overprints the data
        span = [min(xs), max(xs)] if xs else [0, 1]
        for h in options.get('hlines') or []:
            line(span, [h['y']] * 2, xref, yref, h.get('label'), h.get('dash', 'dash'))
        for d in options.get('dlines') or []:   # y = slope x + intercept: over the x range if set, else the data
            lo, hi = min(xs + ys, default=0), max(xs + ys, default=1)
            xx = list(fig.layout[_axis_key(xref)].range or [lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)])
            line(xx, [d.get('slope', 1) * x + d.get('intercept', 0) for x in xx], xref, yref,
                 d.get('label'), d.get('dash', 'solid'))
        for v in options.get('vlines') or []:
            fig.add_shape(type='line', x0=v['x'], x1=v['x'], y0=0, y1=1, xref=xref, yref=f'{yref} domain',
                          line=dict(color=_REF_LINE, width=1, dash=v.get('dash', 'solid')), layer='below')
        for b in options.get('hbands') or []:
            fig.add_shape(type='rect', x0=0, x1=1, y0=b['y0'], y1=b['y1'], xref=f'{xref} domain', yref=yref,
                          fillcolor=b.get('color', 'rgba(128,128,128,0.15)'), line_width=0, layer='below')


def apply_figure_options(fig, options, rows=None):
    """Opt-in, per-view figure options for publication exports, from a view's
    _extras['figure'] (a go.Figure, in place; resolve_figure_options fills in the site presets first):
      legend: 'per-series'  one legend entry per series: a series' fitted line joins its
                            markers' entry (same legendgroup), so the pair toggles together
      legend_swatch: 'line' with legend 'per-series': the fitted line carries the entry instead
                            of the markers, so the swatch shows the series' colour and dash
      title: False          no figure title (the axis titles carry it)
      color_title           a continuous colour scale's title in place of the column name (from the presets'
                            axis titles of the colour column, prefixed with bin2d's agg when binned)
      x_title / y_title     paper-facing axis titles in place of the column names ('' for none); on a
                            faceted figure the x title goes on the bottom row, the y title on the first column
      x_titles: {row value: title}  on a faceted figure, an x title per facet row
      y_title_span: true    on a faceted figure, one y title for the whole grid (centred on its height) in place of
                            one per row, e.g. where each row is shorter than the title
      names: {trace: label} paper-facing legend names in place of the generated ones
                            ('<group> [<y>]', or just '<group>', which then names it for every y), listed in
                            legend order; a null label drops that series from the legend. Unlisted series keep theirs.
      names_from: column    each trace's legend name is that column's value in its rows (e.g. a label the
                            analysis wrote with the fitted numbers in it); names still apply after it
      hlines: [{y, label?, dash?}]  labelled horizontal reference lines (e.g. a theoretical value)
      dlines: [{slope?, intercept?, label?, dash?}]  lines y = slope x + intercept (default: the identity)
      vlines: [{x, dash?}]  vertical lines over the full plot height
      hbands: [{y0, y1, color?}]  shaded horizontal bands over the full plot width (e.g. 'not solved')
      legend_pos: {x, y, xanchor, yanchor, orientation?}  the legend's place in paper coordinates, on a
                            semi-opaque panel of the figure's background: inside the plot (0-1) it keeps the
                            plot's full width; outside (e.g. x 1.02, xanchor left) the plot narrows
      facet_legend: 'bottom right'  (faceted figures) one legend per facet column, inside its bottom panel ('below':
                    under the column, at the foot of the figure)
      legend_extra: [{name, symbol?, dash?}]  legend-only entries in grey, e.g. what a marker shape means
      series_labels: {series: label}  (from the presets) a trace not in names whose group is a series
                            exports with that series' label in its place
      trace_mode: 'lines+markers' | 'lines'  connect each series' points in table order
      error_bars: [{lo, hi, for?, width?, cap?}]  y error bars from the columns lo and hi of each point's row
                            (width: the bar's thickness, cap: its end caps' width, in px)
      bin2d: {x, y, z, dx?, dy?, agg?, min_n?}  per panel the points binned on x and y, a heatmap of z's agg (median)
                            in place of the dots (fewer than min_n rows: blank); rings still mark single points.
                            Set by the scatter's Heatmap control (ScatterDashboard._heatmap_spec), not in a
                            view's figure block
      x_dodge: step         x as evenly spaced categories, each panel's shown traces side by side around a level,
                            step apart in category widths (e.g. 0.08), so their points and error bars don't overlap
      rings: [{where, label?, size?, color?, for?}]  an open circle around each point whose row matches where (a pandas
                            expression, e.g. 'accuracy_dev >= 0.9995'), one legend entry per ring spec
      arrows: {x0, y0, dx, dy, for?, width?}  an arrow per point from (x0, y0) by (dx, dy), each a column or
                            a number (e.g. a walk's velocity input at each step)
      category_labels: {category: label}  (from the presets: axis titles and series labels) the tick labels of
                            categories on x, e.g. a head's or a factor's export spelling
      swap_axes: true       transpose the figure (e.g. categories on y: plot them on x and swap)
      colorbar: {...}       plotly colorbar properties of the continuous colour scale (e.g. a short horizontal bar in
                            a panel's empty corner: orientation 'h', x, y, xanchor, yanchor, len, thickness)
      axes: {x: {...}, y: {...}}  plotly axis properties for every x / y axis (type 'log', range, tickvals and
                            ticktext, scaleanchor for equal scaling, ...)
    Options that read the data (names_from, error_bars, rings, arrows) take `rows`: the frame the figure was drawn from,
    which each point's customdata indexes. Traces are matched by their generated names, so the renames come last.
    """
    if not options:
        return
    if options.get('trace_mode'):
        for t in fig.data:
            if (getattr(t, 'mode', None) or 'markers') == 'markers':
                _remember(t, mode0='markers')
                t.mode = options['trace_mode']
    _add_error_bars(fig, options.get('error_bars') or [], rows)
    if options.get('x_dodge'):
        _dodge_x(fig, options['x_dodge'])
    _add_rings(fig, options.get('rings') or [], rows)
    if options.get('bin2d'):   # after the rings, which mark single points of the hidden dots
        _bin2d(fig, options['bin2d'], rows)
    if options.get('arrows'):
        _add_arrows(fig, options['arrows'], rows)
    spelled = options.get('category_labels') or {}
    xs = [getattr(t, 'x', None) for t in fig.data]
    cats = list(dict.fromkeys(v for x in xs if x is not None for v in x if isinstance(v, str)))
    if any(c in spelled for c in cats):
        fig.update_xaxes(tickvals=cats, ticktext=[spelled.get(c, c) for c in cats])
    if options.get('swap_axes'):
        _swap_axes(fig)
    if options.get('legend') == 'per-series':
        markers = {t.name: t for t in fig.data if (getattr(t, 'mode', None) or 'markers') != 'lines'}
        line_swatch = options.get('legend_swatch') == 'line'
        for t in fig.data:
            if getattr(t, 'mode', None) == 'lines' and t.name in markers:
                m = markers[t.name]
                t.legendgroup = m.legendgroup
                # dots off or hidden: the markers carry no entry, so the line keeps its own
                marker_entry = m.showlegend is not False and m.visible is not False
                t.showlegend = line_swatch or not marker_entry
                if line_swatch:
                    m.showlegend = False
    if options.get('title') is False:
        fig.update_layout(title=None, meta={**(fig.layout.meta or {}), 'title': False})
    _place_axis_titles(fig, options)
    if options.get('color_title') is not None and fig.layout.coloraxis.colorbar is not None:
        fig.layout.coloraxis.colorbar.title.text = options['color_title']
    if options.get('colorbar') and fig.layout.coloraxis.colorbar is not None:
        fig.layout.coloraxis.colorbar.update(options['colorbar'])
    for axis, props in (options.get('axes') or {}).items():
        (fig.update_xaxes if axis == 'x' else fig.update_yaxes)(**props)
    if options.get('names_from'):
        for t in fig.data:
            r = _trace_rows(t, rows)
            if r is not None and options['names_from'] in r.columns and len(r):
                _rename(t, str(r[options['names_from']].iloc[0]))
    names = options.get('names') or {}
    rank = {name: i for i, name in enumerate(names)}
    series_labels = options.get('series_labels') or {}
    for t in fig.data:
        key = t.name if t.name in names else (t.name or '').rsplit(' [', 1)[0] if _trace_y(t) else None
        if key not in names:
            s = preset_series({'series': series_labels}, t.name)
            if s is not None:
                _rename(t, series_labels[s] if t.name == s else t.name.replace(s, series_labels[s], 1))
            continue
        t.legendrank = rank[key]
        if names[key] is None:
            t.showlegend = False
        else:
            _rename(t, names[key])
    _share_legend_entries(fig)
    pos = options.get('legend_pos')
    if pos:
        bg = fig.layout.paper_bgcolor or fig.layout.template.layout.paper_bgcolor or 'white'
        fig.update_layout(legend=dict(**pos, bgcolor=_translucent(bg, 0.8)))
    if options.get('facet_legend') and facet_cells(fig):
        _facet_legends(fig, options['facet_legend'])
    _add_reference_marks(fig, options, legendrank=len(names) + 1)
    for e in options.get('legend_extra') or []:
        fig.add_scatter(x=[None], y=[None], mode='lines+markers' if e.get('dash') else 'markers', name=e['name'],
                        marker=dict(symbol=e.get('symbol', 'circle'), color='gray', size=8),
                        line=dict(color='gray', dash=e.get('dash', 'solid')), legendrank=len(names) + 2)
    fitted = fit_furniture(fig.layout.to_plotly_json())
    fig.update_layout(margin=fitted['margin'], annotations=fitted['annotations'])


def fit_furniture(layout: dict, data=(), height: int = None) -> dict:
    """The room around the plot that plotly's automargin misjudges, at the figure's font sizes. Changes the layout dict
    in place and returns it (data: the traces, as dicts). Facet titles are lifted clear of the top y tick label (half
    of it stands above the panel); a grid's shared y title (y_title_span) goes left of the first column's tick
    labels; with no title (figure option title: false) the top margin fits only the top row's facet titles, or else
    that half tick label. An export (height: the image's, px) also gets a bottom margin under the x tick labels and
    title (automargin leaves their descenders outside it), and its legends at the figure's foot (yref container,
    y = 0; each column's legend under its column) hang from one top line, the tallest's. Run by apply_figure_options
    and again by scale_figure_fonts."""
    def size(font, default):
        return (font if isinstance(font, dict) else {}).get('size') or default

    base = size(layout.get('font'), _FONT_BASE_PX)
    xtick = size((layout.get('xaxis') or {}).get('tickfont'), base)
    ytick = size((layout.get('yaxis') or {}).get('tickfont'), base)
    margin = layout['margin'] = {**(layout.get('margin') or {})}
    anns = layout['annotations'] = [{**a} for a in layout.get('annotations') or []]
    facet_titles = [a for a in anns if a.get('name') == 'facet_title']
    lift = round(0.5 * ytick)
    for a in facet_titles:
        a['yshift'] = lift
    for a in anns:
        if a.get('name') == 'grid_y_title':
            axis = layout.get((layout.get('meta') or {}).get('grid_y_axis', 'yaxis')) or {}
            title = size(a.get('font'), 1.2 * base)
            tick = size(axis.get('tickfont'), base)
            labels = _tick_label_chars(axis) * 0.56 * tick + 0.3 * tick   # Liberation Sans digits: 0.556 em
            a['xshift'] = -round(labels + 0.25 * title)
            margin['l'] = max(margin.get('l', 80), round(labels + 1.5 * title))
    if (layout.get('meta') or {}).get('title') is False:
        margin['t'] = round(1.4 * max(size(a.get('font'), base) for a in facet_titles) + lift) if facet_titles \
            else round(0.7 * ytick)
    if height:
        foot = [k for k in layout if (k == 'legend' or k[:6] == 'legend' and k[6:].isdigit())
                and (layout[k] or {}).get('yref') == 'container' and (layout[k] or {}).get('y') == 0]
        tall = {k: _legend_height(layout, k, data) for k in foot if layout[k].get('orientation') != 'h'}
        if len(tall) > 1:
            top = max(tall.values())
            for k in tall:
                layout[k] = {**layout[k], 'y': min(1.0, top / height), 'yanchor': 'top'}
        if not foot:
            titled = [layout[k] for k in layout if k.startswith('xaxis') and _axis_title(layout[k]).get('text')]
            if titled:
                title = max(size(_axis_title(a).get('font'), 1.2 * base) for a in titled)
                margin['b'] = max(margin.get('b', 80), round(1.25 * xtick + 1.6 * title))
    return layout


def _axis_title(axis: dict) -> dict:
    t = (axis or {}).get('title')
    return {'text': t} if isinstance(t, str) else t or {}


def _tick_label_chars(axis: dict) -> int:
    """An axis's longest tick label, in characters, from what the figure fixes (ticktext, tickvals, or range and
    dtick on a linear axis); 3 where plotly picks the ticks."""
    import re
    if axis.get('ticktext'):
        return max(len(re.sub('<[^>]+>', '', str(t))) for t in axis['ticktext'])
    vals = axis.get('tickvals')
    if vals is None and axis.get('range') and axis.get('dtick') and axis.get('type') in (None, '-', 'linear'):
        lo, hi = sorted(axis['range'])
        d = float(axis['dtick'])
        vals = np.arange(np.ceil(lo / d) * d, hi + d * 1e-9, d)
    if vals is None or not len(vals):
        return 3
    return max(len(f'{round(float(v), 6) + 0:g}') for v in vals)


def _legend_height(layout: dict, key: str, data) -> float:
    """A vertical legend's height (px), estimated from its entries' line counts and font, on the high side: plotly
    cuts a legend that outgrows its room to a scrolling box."""
    lines = [(t.get('name') or '').count('<br>') + 1 for t in data
             if (t.get('legend') or 'legend') == key and t.get('showlegend') is not False
             and t.get('visible') is not False and t.get('name')]
    fs = (layout[key].get('font') or {}).get('size') or (layout.get('font') or {}).get('size') or _FONT_BASE_PX
    return 1.1 * (sum(1.3 * fs * n + 5 for n in lines) + 10)


def apply_trace_vis(fig, trace_visibility):
    """Apply a {name|mode: visible} dict to figure traces in-place.

    Accepts either a go.Figure or a plain dict (from fig.to_dict()).
    """
    if not trace_visibility:
        return
    if hasattr(fig, 'data'):
        for trace in fig.data:
            key = f"{trace.name or ''}|{getattr(trace, 'mode', None) or 'markers'}"
            if key in trace_visibility:
                trace.visible = trace_visibility[key]
    else:
        for t in fig.get('data', []):
            key = f"{t.get('name') or ''}|{t.get('mode') or 'markers'}"
            if key in trace_visibility:
                t['visible'] = trace_visibility[key]

_MIME = {'png': 'image/png', 'svg': 'image/svg+xml', 'pdf': 'application/pdf',
         'html': 'text/html'}


class ViewDataMismatch(ValueError):
    """A saved view's static filters name values that the app's data does not have.

    Rendering anyway draws a plausible wrong figure: the export an empty "No data" plot, the UI the
    unfiltered data (a dropdown drops values that are not among its options). The usual cause is an
    app that booted before a grid finished — scatter apps read their grids once, at boot (2026-10-12:
    path-integration lacked the 2D main grids, so pi2d views exported black). The export route turns
    this into a 409.
    """
    def __init__(self, missing: dict):
        self.missing = missing
        super().__init__(f'view filters name values absent from this app\'s data: {missing}. '
                         'The app reads its grids at boot: restart it if they finished later, '
                         'otherwise the view is stale.')


def _strip_interactive_svg(svg: str) -> str:
    """Fix Plotly SVG for static rendering.

    Drag hitbox elements (angulardrag, radialdrag, subplot drag rects) use
    'fill: transparent' as a CSS colour value.  Static SVG renderers treat
    it as opaque black, producing thick artefacts.  'fill: none' is the
    correct SVG idiom for no fill and is safe to substitute globally because
    Plotly only uses 'fill: transparent' on these invisible hitboxes.
    """
    return svg.replace('fill: transparent', 'fill: none')


_MERGE_SVG_JS = """
    () => {
        const svgs = Array.from(document.querySelectorAll('.main-svg'));
        if (svgs.length === 1) return svgs[0].outerHTML;
        const root = svgs[0].cloneNode(true);
        for (const svg of svgs.slice(1))
            for (const child of svg.children)
                root.appendChild(child.cloneNode(true));
        return root.outerHTML;
    }
"""

def _render_html_to_image(html: str, width: int, height: int, fmt: str = 'png') -> bytes:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu'])
        page = browser.new_page(viewport={'width': width, 'height': height})
        page.set_content(html, wait_until='networkidle')
        if fmt in ('svg', 'pdf'):
            svg = _strip_interactive_svg(page.evaluate(_MERGE_SVG_JS))
            browser.close()
            if fmt == 'svg':
                return svg.encode()
            import cairosvg
            return cairosvg.svg2pdf(bytestring=svg.encode(), output_width=width, output_height=height)
        img_bytes = page.screenshot(full_page=False, type='png')
        browser.close()
        return img_bytes


@contextlib.contextmanager
def _no_display_env():
    """Hide X/wayland display vars from kaleido's chrome for one render.

    Restored afterwards, so the rest of the process never notices. Why this
    exists: a headless render process that inherits DISPLAY (e.g.
    dsh sessions in the devcontainer get the VS Code socket :4) breaks WebGL —
    chrome 149 initialises GL through ANGLE-on-Vulkan, which requires a live,
    AUTHENTICATED X connection first and then refuses its SwiftShader fallback,
    so every scattergl export drew axes but no data. Production containers set
    no DISPLAY at all and were always fine; popping the vars makes the chrome
    spawn identical to prod's.
    """
    keys = [k for k in ('DISPLAY', 'WAYLAND_DISPLAY') if k in os.environ]
    saved = {k: os.environ.pop(k) for k in keys}
    try:
        yield
    finally:
        os.environ.update(saved)


def _vectorize_scattergl(fig):
    """Recast scattergl traces as plain scatter, for the vector formats.

    plotly rasterises a scattergl trace at static-export time on every host:
    the svg carries the WebGL canvas as an embedded ``<image data:image…>``
    (94 KB against 17 KB / 75 <path> for the same trace as scatter) and
    pdf inherits it. For paper figures that means a fixed-resolution bitmap
    where zoomable vector paths belong. Only the vector formats are cast — png
    from a working-GL host renders scattergl fine and must stay byte-stable for
    the existing pipelines. Trace properties these dashboards use are shared
    between the two types, so the cast is a plain dict copy.
    """
    fig_dict = fig.to_plotly_json()
    traces = fig_dict.get('data') or []
    if not any(t.get('type') == 'scattergl' for t in traces):
        return fig
    fig_dict['data'] = [
        dict(t, type='scatter') if t.get('type') == 'scattergl' else t
        for t in traces
    ]
    import plotly.graph_objects as go
    return go.Figure(fig_dict)


def render_figure_image(fig, width: int, height: int, fmt: str = 'png') -> bytes:
    import plotly.graph_objects as go
    import plotly.io as pio

    if fmt in ('svg', 'pdf'):
        fig = _vectorize_scattergl(fig)
    if any(getattr(t, 'visible', None) == 'legendonly' for t in fig.data):
        # a print can't be clicked: a trace hidden behind its legend entry leaves no greyed entry either
        fig = go.Figure(fig)
        for t in fig.data:
            if t.visible == 'legendonly':
                t.showlegend = False
    with _no_display_env():
        return pio.to_image(fig, format=fmt, width=width, height=height)


def render_figure_png(fig, width: int, height: int) -> bytes:
    return render_figure_image(fig, width, height, fmt='png')


def render_figure_html(fig, width: int, height: int, *, plotlyjs: str = 'inline') -> bytes:
    """Render a figure to a self-contained interactive HTML document.

    plotlyjs='inline' embeds the full plotly.js bundle (~4.6 MB) in the page,
    so the file works offline — GitHub Pages, a file: double-click, email.
    'cdn' emits a <script src> tag instead (~2 kB page, needs the network).
    Figures with no explicit layout width/height get autosize=True so the plot
    fills its container; figures whose fitter baked an explicit size (the
    polar table fits to the frame) keep it exactly.
    """
    import plotly.graph_objects as _go
    import plotly.io as _pio

    fig2 = _go.Figure(fig)
    if fig2.layout.width is None or fig2.layout.height is None:
        fig2.layout.autosize = True
    return _pio.to_html(
        fig2,
        config={'responsive': True, 'displaylogo': False},
        include_plotlyjs=(True if plotlyjs == 'inline' else 'cdn'),
        full_html=True,
        default_width=width,
        default_height=height,
    ).encode('utf-8')


# ── Font override for paper exports (?font=X) ────────────────────────────────
#
# A paper includes every dashboard figure scaled down to a
# 108mm text block, which shrinks the fonts with it (a 10px tick label typesets
# at ~3pt).  `?font=X` on an export URL asks for a figure whose BASE font is X
# px, so that after LaTeX's downscaling the text lands at the intended physical
# size.  Key formula:
#     typeset_pt = font_px * tex_target_pt / export_width_px
# with the per-figure X values.

_FONT_BASE_PX = 12   # plotly's built-in default layout font size — the implicit
                     # base every existing figure renders at (verified in exports)

_FURNITURE_EXP = 0.6  # reference furniture (tick labels, legend) grows as
                      # k**0.6 while data text grows as k — see scale_figure_fonts


def _scale_fonts_in(node, k: float) -> None:
    """Recursively multiply every explicit font size under `node` by `k`.

    A font dict is any dict value stored under a key named 'font' or ending in
    'font' (font, tickfont, textfont, titlefont, header.font, cells.font, ...),
    at any depth of layout or a trace.  Non-font sizes (markers, lines, margins)
    are never touched.  'template' is skipped: templates are theme-level objects
    and this repo's set no sizes (and must stay pristine — see theme.py).
    """
    if isinstance(node, dict):
        for key, val in node.items():
            if key == 'template':
                continue
            if key == 'font' or (isinstance(key, str) and key.endswith('font')):
                if isinstance(val, dict):
                    size = val.get('size')
                    if isinstance(size, (int, float)) and not isinstance(size, bool):
                        val['size'] = size * k
                    _scale_fonts_in(val, k)   # nested containers, if any
                continue
            _scale_fonts_in(val, k)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _scale_fonts_in(item, k)


def scale_figure_fonts(fig, base_px: int, *, title_scale: float = None,
                       axis_title_scale: float = None,
                       title_px: float = None, axislabel_px: float = None,
                       tick_scale: float = None, legend_scale: float = None, height: int = None):
    """Return a figure like `fig` but with its base font size set to `base_px`.

    Explicit font sizes are multiplied by base_px/_FONT_BASE_PX and the base
    layout.font.size is SET to base_px, so fonts that inherit plotly's default
    land at base_px too.  Data text (titles, annotations, trace labels) moves
    by the full factor; reference furniture (tick labels, legend) moves by
    k**_FURNITURE_EXP so it cannot crowd out the plot at paper sizes.  The
    override also enables title/xaxis/yaxis automargin, so enlarged text pushes
    the margins out instead of being clipped at whatever margins the figure
    carries.

    Title pinning (opt-in): plotly.js renders an INHERITED figure title at
    ~1.4x the base font and inherited axis titles at ~1.2x (measured via SVG
    font-size attributes across base 12..80, plotly 6.8.0), while an EXPLICIT
    title font size is honored exactly.  Passing `title_scale` /
    `axis_title_scale` pins explicit sizes at those multiples of `base_px`
    (applied after the furniture pass, which would otherwise rescale them),
    turning the title print size into a deliberate choice instead of a side
    effect.  `title_px` / `axislabel_px` are absolute px values that beat
    everything else.  Callers that pass none of the four (polar, trace) get
    today's behavior unchanged.  `tick_scale` / `legend_scale` likewise pin the
    tick labels (every x/y axis) / legend at those multiples of `base_px` in
    place of the furniture scale (a site's presets text: ticks / legend).
    `height` (the image's, px) lets fit_furniture lay out the margins and foot legends at the scaled sizes.

    plotly's to_plotly_json() returns a copy, so the figure is rebuilt from the
    scaled props; the input figure is left untouched.
    """
    import plotly.graph_objects as _go

    k = base_px / _FONT_BASE_PX
    props = fig.to_plotly_json() if hasattr(fig, 'to_plotly_json') else fig
    _scale_fonts_in(props.get('layout'), k)
    for trace in props.get('data') or []:
        _scale_fonts_in(trace, k)
    layout = props.get('layout')
    if layout is None:
        layout = {}
        props['layout'] = layout
    font = layout.get('font')
    if not isinstance(font, dict):
        font = {}
        layout['font'] = font
    font['size'] = base_px
    # With the override active, let plotly grow the margins so the enlarged
    # title, tick labels and axis titles fit instead of being clipped.  This
    # line is reached only through the ?font= override, so exports without it
    # and interactive pages are unaffected.  (plotly.js in this version has no
    # layout.margin.automargin; the per-object attributes are the supported
    # levers.  Legends have no automargin at all — their automatic placement
    # already re-anchors per size, as a geometry probe showed.)
    title = layout.get('title')
    if isinstance(title, str):
        title = {'text': title}
    layout['title'] = {**(title or {}), 'automargin': True}
    # every x/y axis and legend: a faceted figure has xaxis2, ..., and a legend per facet column (legend2, ...)
    _xy_axes = ['xaxis', 'yaxis'] + sorted(a for a in layout if a.startswith(('xaxis', 'yaxis')) and a[5:].isdigit())
    _legends = ['legend'] + sorted(a for a in layout if a.startswith('legend') and a[6:].isdigit())
    for _axis in _xy_axes:
        layout[_axis] = {**(layout.get(_axis) or {}), 'automargin': True}
    # Furniture cap: tick labels and the legend are reference marks, not data
    # text.  At paper sizes they were crowding out the plot outright (4-entry
    # legends wider than the plot, 48 pinned angular tickvals overprinting).
    # Let them grow more slowly than the data text (k**0.6), which also
    # de-collides dense tick rings without touching the pinned tickvals.  Axis
    # titles ride along inside their axis subtree; trace textfonts, figure
    # titles and annotations stay at full k.  Only reached through the ?font=
    # override.
    k_furn = max(1.0, min(k, k ** _FURNITURE_EXP))
    if k_furn < k:
        _rescale = k_furn / k

        def _furnish(sub, kind):
            # Explicit furniture fonts: bring them from k down to k_furn.
            # Inherited ones (no size of their own — they render at the scaled
            # base, which is what made scatter legends wider than the plot):
            # pin them at the furniture scale so they actually shrink too.
            # A missing subtree is created, since inheriting is the case that
            # needs the pin most.
            _key = 'font' if kind == 'legend' else 'tickfont'
            if not isinstance(sub, dict):
                sub = {}
            _scale_fonts_in(sub, _rescale)
            _probe = sub.get(_key)
            if not (isinstance(_probe, dict) and _probe.get('size') is not None):
                sub.setdefault(_key, {})['size'] = _FONT_BASE_PX * k_furn
            return sub

        for _axes in _xy_axes:
            layout[_axes] = _furnish(layout.get(_axes), 'axes')
        for _axes in ('angularaxis', 'radialaxis'):
            if isinstance(layout.get(_axes), dict):
                layout[_axes] = _furnish(layout[_axes], 'axes')
        for _legend in _legends:
            layout[_legend] = _furnish(layout.get(_legend), 'legend')
        _polar = layout.get('polar')
        if isinstance(_polar, dict):
            for _axes in ('angularaxis', 'radialaxis'):
                if isinstance(_polar.get(_axes), dict):
                    _polar[_axes] = _furnish(_polar[_axes], 'axes')

    # Furniture pin: a site that wants ticks or legend at a set ratio of the font (presets text).
    # Every x/y axis, so facetted figures (xaxis2, ...) follow too.
    if tick_scale is not None:
        for _axis in [a for a in layout if a.startswith(('xaxis', 'yaxis'))]:
            sub = {**(layout.get(_axis) or {})}
            sub['tickfont'] = {**(sub.get('tickfont') or {}), 'size': tick_scale * base_px}
            layout[_axis] = sub
    if legend_scale is not None:
        for _legend in _legends:
            legend = {**(layout.get(_legend) or {})}
            legend['font'] = {**(legend.get('font') or {}), 'size': legend_scale * base_px}
            layout[_legend] = legend

    # Title pin — after the furniture pass, which rescales every font inside
    # the axis subtrees (including xaxis.title.font) and so must not see the
    # pinned sizes.  Only reached through the ?font= override AND only when
    # the caller asked for a title ratio / absolute px; without it the
    # inherited plotly inflation stands and exports stay as they were.
    if title_px is not None or title_scale is not None \
            or axislabel_px is not None or axis_title_scale is not None:
        def _pin(font_dict, size):
            font_dict = dict(font_dict) if isinstance(font_dict, dict) else {}
            font_dict['size'] = size
            return font_dict

        if title_px is not None or title_scale is not None:
            size = title_px if title_px is not None else title_scale * base_px
            title = layout.get('title')
            if isinstance(title, str):
                title = {'text': title}
            title = {**(title or {})}
            title['font'] = _pin(title.get('font'), size)
            layout['title'] = title
        if axislabel_px is not None or axis_title_scale is not None:
            size = axislabel_px if axislabel_px is not None \
                else axis_title_scale * base_px
            for _axis in _xy_axes:
                sub = {**(layout.get(_axis) or {})}
                _title = {**(sub.get('title') or {})}
                _title['font'] = _pin(_title.get('font'), size)
                sub['title'] = _title
                layout[_axis] = sub
            # a facet panel's title and a grid's shared y title are axis-level labels: the axis titles' size
            layout['annotations'] = [{**a, 'font': _pin(a.get('font'), size)}
                                     if a.get('name') in ('facet_title', 'grid_y_title')
                                     else a for a in layout.get('annotations') or []]

    fit_furniture(layout, props.get('data') or [], height)

    return _go.Figure(data=props.get('data'), layout=layout,
                      frames=props.get('frames'))


def view_name_from_query(query: str, skip=()) -> str | None:
    """The view a URL query names: its first bare key (?<name>&format=...) not in skip. The one form, live and in
    exports; ?view=<name> names no view."""
    import urllib.parse as _urlparse
    parts = [_urlparse.unquote(q).strip() for q in (query or '').lstrip('?').split('&')]
    return next((q for q in parts if q and '=' not in q and q not in skip), None)


def register_export_route(app, reconstruct_fn, safe_roots: list,
                       default_search_dirs: list = None,
                       title_scale: float = None,
                       axis_title_scale: float = None):
    """
    Register a ?<view_name>&format=<fmt> GET handler on app.server (Flask).

    Supported formats: png, svg, pdf, html, json

    html returns a self-contained interactive page (plotly.js inlined by
    default, so it works offline; &plotlyjs=cdn swaps in a CDN <script> tag).
    json returns the figure's plotly JSON — the format consumed by
    dashboard/tools/export_static_html.py to bundle many views into one
    tabbed offline page.

    reconstruct_fn may be:
      - a callable(view_data, width, height, theme_override) -> Figure
      - a dict[str, callable] — dispatched via ?target=<key>; defaults to the
        first key when ?target is omitted.

    Usage
    -----
        # single-target (callable)
        curl 'http://host:port/?my_view&format=png&width=1200&height=800' -o out.png

        # multi-target (dict)
        curl 'http://host:port/?my_view&format=png&target=polar&width=1200&height=800' -o polar.png
        curl 'http://host:port/?my_view&format=png&target=table&width=600&height=400'  -o table.png

    &font=<px> (optional) overrides the figure's base font size for this export
    only, scaling every explicit font by <px>/12 and setting the base to <px> —
    used by the paper pipeline so figure text survives LaTeX downscaling.
    Valid range 6–120.  Absent ⇒ the export is byte-identical to before.

    &legend=off (optional) drops the legend for this export only, e.g. the panel of a paper figure that sits beside
    the one carrying it; the browser view keeps its legend (and its clicks).

    title_scale / axis_title_scale (optional, constructor-level) pin the figure
    title / x+y axis titles at those multiples of the base font instead of
    plotly's inherited ~1.4x / ~1.2x inflation — the scatter family passes
    caption-size ratios (see scatter.py).  Per-request ?title=<px> and
    ?axislabel=<px> (6–120) override the pinned sizes as absolute values.

    width/height are the size of the rendered image, not a viewport onto a larger
    one: a reconstruct_fn is handed both and is expected to fit its whole content
    inside them (scaling it if need be) rather than let the canvas crop it.
    """
    import urllib.parse as _urlparse
    import orjson
    from flask import request as _req, Response as _Resp

    if default_search_dirs is None:
        default_search_dirs = [Path('/out/dashboard'), Path('/tmp/dashboard')]

    _safe     = [Path(r) for r in safe_roots]
    _defaults = [Path(d) for d in default_search_dirs]
    _multi    = isinstance(reconstruct_fn, dict)

    # Reconstruct fns may declare a `font_px` parameter and apply the override
    # themselves (the polar table fitter must, since it fits the table to the
    # frame).  Those are handed font_px and NOT post-scaled; all others get the
    # central post-scale in scale_figure_fonts().
    _fns       = list(reconstruct_fn.values()) if _multi else [reconstruct_fn]
    _takes_font = {id(fn): 'font_px' in inspect.signature(fn).parameters
                   for fn in _fns}

    def _serves_font_px(fn):
        return _takes_font[id(fn)]

    @app.server.before_request
    def _maybe_serve_image():
        args = _req.args
        fmt  = args.get('format', '').lower()
        if fmt not in _MIME and fmt != 'json':
            return None
        # plotly.js embedding for format=html only: 'inline' bundles the
        # ~4.6 MB library into the page (works offline); 'cdn' emits a
        # <script> tag.  Anything else is a 400, not a silent inline.
        plotlyjs = (args.get('plotlyjs') or 'inline').lower()
        if fmt == 'html' and plotlyjs not in ('inline', 'cdn'):
            return _Resp('plotlyjs must be "inline" or "cdn"',
                         status=400, content_type='text/plain; charset=utf-8')

        font_px = None
        if 'font' in args:
            raw = args.get('font', '').strip()
            if not raw.isdigit():
                return _Resp('font must be a positive integer (px), e.g. &font=29',
                             status=400, content_type='text/plain; charset=utf-8')
            font_px = int(raw)
            if not 6 <= font_px <= 120:
                return _Resp('font must be in [6, 120] px',
                             status=400, content_type='text/plain; charset=utf-8')

        # Per-figure absolute-px overrides for the pinned title / axis-title
        # sizes.  Only meaningful together with
        # ?font=; ignored by fitters that serve font_px themselves (polar
        # table), which fit their own text to the frame.
        _px_overrides = {}
        for _name in ('title', 'axislabel'):
            if _name in args:
                raw = args.get(_name, '').strip()
                if not raw.isdigit():
                    return _Resp(f'{_name} must be a positive integer (px), '
                                 f'e.g. &{_name}=52',
                                 status=400, content_type='text/plain; charset=utf-8')
                _val = int(raw)
                if not 6 <= _val <= 120:
                    return _Resp(f'{_name} must be in [6, 120] px',
                                 status=400, content_type='text/plain; charset=utf-8')
                _px_overrides[_name] = _val

        view_name = view_name_from_query(_req.query_string.decode('utf-8'))
        if not view_name:
            return _Resp(f'Missing view name — use ?<view_name>&format={fmt} (a bare key, not view=)',
                         status=400, content_type='text/plain; charset=utf-8')

        width  = int(args.get('width',  1200))
        height = int(args.get('height', 800))
        theme_param    = args.get('theme')
        theme_override = (['light'] if theme_param == 'light'
                          else (['dark'] if theme_param == 'dark' else None))

        if _multi:
            target = args.get('target', next(iter(reconstruct_fn)))
            fn = reconstruct_fn.get(target)
            if fn is None:
                return _Resp(
                    f'Unknown target "{target}". Valid targets: {list(reconstruct_fn)}',
                    status=400, content_type='text/plain; charset=utf-8')
        else:
            fn = reconstruct_fn

        if args.get('dir'):
            explicit = _views_dir(args.get('dir'))
            if not _is_safe_path(explicit, _safe):
                return _Resp(f'dir must be under {[str(r) for r in _safe]}',
                             status=400, content_type='text/plain; charset=utf-8')
            search_dirs = [explicit]
        else:
            search_dirs = _defaults

        found = None
        for d in search_dirs:
            c = d / f'{view_name}.json'
            if c.exists():
                found = c
                break

        if found is None:
            searched  = [str(d / f'{view_name}.json') for d in search_dirs]
            available = [p.stem for d in search_dirs if d.exists()
                         for p in sorted(d.glob('*.json'))]
            lines = [
                f'View "{view_name}" not found.',
                f'Searched: {searched}',
                f'Available: {available}' if available else 'No views saved yet.',
                'Save the view first via the Views panel in the dashboard.',
            ]
            return _Resp('\n'.join(lines), status=404,
                         content_type='text/plain; charset=utf-8')

        try:
            view_data = orjson.loads(found.read_bytes())
            if _serves_font_px(fn):
                # The fitter applies the override itself (e.g. the polar table
                # fits fonts to the frame pre-fit); skip the central post-scale.
                fig = fn(view_data, width, height, theme_override,
                         font_px=font_px)
            else:
                fig = fn(view_data, width, height, theme_override)
                if font_px:
                    # the view's site may set its own text ratios (presets text)
                    text = load_figure_presets(found.parent).get('text') or {}
                    fig = scale_figure_fonts(
                        fig, font_px,
                        title_scale=text.get('title', title_scale),
                        axis_title_scale=text.get('axis_title', axis_title_scale),
                        title_px=_px_overrides.get('title'),
                        axislabel_px=_px_overrides.get('axislabel'),
                        tick_scale=text.get('ticks'),
                        legend_scale=text.get('legend'),
                        height=height)
            if args.get('legend') == 'off':
                fig.update_layout(showlegend=False)
        except ViewDataMismatch as e:
            import sys
            print(f'[export] "{view_name}": {e}', file=sys.stderr, flush=True)
            return _Resp(f'View "{view_name}": {e}', status=409, content_type='text/plain; charset=utf-8')
        except Exception as e:
            import sys, traceback
            print(f'[export] reconstruct failed for "{view_name}": {e}\n'
                  + traceback.format_exc(), file=sys.stderr, flush=True)
            return _Resp(f'Failed to reconstruct view "{view_name}": {e}',
                         status=500, content_type='text/plain; charset=utf-8')

        try:
            if fmt == 'html':
                payload = render_figure_html(fig, width, height, plotlyjs=plotlyjs)
                mimetype = _MIME['html']
            elif fmt == 'json':
                import plotly.io as _pio
                payload = _pio.to_json(fig).encode('utf-8')
                mimetype = 'application/json'
            else:
                payload = render_figure_image(fig, width, height, fmt=fmt)
                mimetype = _MIME[fmt]
        except Exception as e:
            import sys, traceback
            print(f'[export] render failed for "{view_name}" ({fmt}): {e}\n'
                  + traceback.format_exc(), file=sys.stderr, flush=True)
            return _Resp(f'{fmt.upper()} rendering failed: {e}',
                         status=500, content_type='text/plain; charset=utf-8')

        return _Resp(payload, mimetype=mimetype,
                     headers={'Content-Disposition':
                               f'inline; filename="{view_name}.{fmt}"'})
