#!/usr/bin/env python3
r"""Bundle saved dashboard views into ONE self-contained interactive HTML page.

The export route (dashboards/utils.py::register_export_route) serves one
figure per request.  This script walks a set of RUNNING dashboard services,
fetches each saved view as plotly JSON (?format=json — reconstructed by the
service exactly as its ?format=png export would be, so per-app presets,
themes and font-aware fitters all apply), and writes a single html file:
every figure behind its own tab, plotly.js embedded ONCE (~4.6 MB), figures
drawn lazily when their tab is first opened.  The output is fully offline —
drop it in docs/ for GitHub Pages, attach it, or just double-click it.

Why fetch from the running services instead of reconstructing here: loading
grids in this process is a known cold-cache OOM risk, while
the services already hold theirs warm.

Usage (from code/, services up):

    python3 dashboard/tools/export_static_html.py \
        --pair http://host.docker.internal:8050 out/dashboard/hegg/scatter \
        --pair http://host.docker.internal:8054 out/dashboard/hegg/trace \
        --view speed_main --out out/dashboard/static/hegg.html

Each --pair binds a service base URL to the repo-side views directory whose
*.json files name the views that service serves (the app's own
default_search_dirs must cover the container-side mount of that directory —
the normal setup).  --view restricts and globally reorders; without it every
view of every pair is bundled.  --target / --font / --theme are passed to
every fetch exactly like their ?format=png counterparts on the export route.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import urlopen

# …/boolean_reservoir (this file: code/dashboard/tools/export_static_html.py)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / 'out' / 'dashboard' / 'static' / 'index.html'

FETCH_TIMEOUT = 120   # s — a cold polar/target fit can take tens of seconds

# Same interactive knobs as the format=html export branch in
# dashboards/utils.py (register_export_route → render_figure_html).
PANE_CONFIG = {'responsive': True, 'displaylogo': False}


class FetchError(Exception):
    """A view could not be fetched; str(e) carries the reason."""


def fetch_figure_json(base_url, view, params=None, timeout=FETCH_TIMEOUT):
    """Fetch one view's figure as plotly JSON from a running export route.

    Returns the parsed figure dict ({'data': […], 'layout': {…}}).  Raises
    FetchError with the server's message body on any HTTP error (the export
    route's 404 body lists the views that service can serve) or on a non-JSON
    body — the latter usually means the URL is a dashboard without an export
    route (format=json only exists where register_export_route was called).
    """
    qs = urlencode({'format': 'json', **(params or {})})
    url = f"{base_url.rstrip('/')}/?{quote(view, safe='')}&{qs}"
    try:
        with urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
    except HTTPError as e:
        body = e.read().decode('utf-8', 'replace').strip()
        raise FetchError(f'HTTP {e.code} for {url}:\n    {body[:600]}') from e
    except OSError as e:                      # URLError, timeouts, refused …
        raise FetchError(f'cannot reach {url}: {e} (is that dashboard up?)') from e
    try:
        return json.loads(raw)
    except ValueError:
        raise FetchError(f'{url} returned a non-JSON body (no export route?): '
                         f'{raw[:200]!r}') from None


def _js_json_str(obj) -> str:
    """JSON for embedding inside a <script> block.

    '</' can appear inside any text label ('a</b>' in a hover tooltip) and
    must never reach the HTML parser near a </script>; '<\\/' inside a JSON
    string decodes to exactly the same characters, so this is lossless.
    Only used on figure JSON we embed ourselves — see build_html for why the
    plotly.js bundle is NOT escaped.
    """
    return json.dumps(obj).replace('</', '<\\/')


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  body    { margin: 0; background: #fff; color: #222;
            font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; }
  nav     { position: sticky; top: 0; z-index: 5; background: #fff;
            padding: 6px 10px; border-bottom: 1px solid #ddd; }
  nav button { border: 1px solid #ccc; background: #f7f7f7; border-radius: 4px;
               margin: 2px 4px 2px 0; padding: 4px 10px; font-size: 13px;
               cursor: pointer; }
  nav button.active { background: #2a6fdb; border-color: #2a6fdb; color: #fff; }
  .pane   { width: 100%; height: 85vh; }
  footer  { padding: 8px 14px; font-size: 12px; color: #666;
            border-top: 1px solid #eee; }
</style>
</head>
<body>
<nav>__TABS__</nav>
__PANES__
<footer>__FOOTER__</footer>
<script>__PLOTLYJS__</script>
<script>
const FIGURES = [
__FIGURES__
];
const CONFIG = __CONFIG__;
const rendered = {};
function activate(i) {
  document.querySelectorAll('nav button').forEach(
      b => b.classList.toggle('active', Number(b.dataset.i) === i));
  document.querySelectorAll('.pane').forEach(p => { p.style.display = 'none'; });
  const pane = document.getElementById('pane' + i);
  pane.style.display = 'block';
  if (!rendered[i]) {
    const fig = FIGURES[i];
    rendered[i] = Plotly.newPlot(pane, fig.data, fig.layout, CONFIG);
  } else {
    Plotly.Plots.resize(pane);
  }
}
activate(0);
</script>
</body>
</html>
"""


def build_html(items, page_title, plotlyjs_src):
    """Return the full standalone page: one lazily-rendered tab per item.

    items: [{'label': str, 'fig': plotly-json dict, 'source': str}, …] in tab
    order.  plotlyjs_src is the plotly.js library, embedded verbatim: the
    library itself is inlined unescaped by plotly.io.to_html's
    include_plotlyjs=True path, so its <script> safety is plotly's own
    invariant — escaping a 4.6 MB minified bundle could only corrupt it.
    """
    import html as _h

    tabs, panes, figs = [], [], []
    for i, it in enumerate(items):
        tabs.append(f'<button class="tab" data-i="{i}" '
                    f'onclick="activate({i})">{_h.escape(it["label"])}</button>')
        panes.append(f'  <div class="pane" id="pane{i}" style="display:none"></div>')
        figs.append('  ' + _js_json_str(it['fig']))

    seen, sources = set(), []
    for it in items:
        if it['source'] not in seen:
            seen.add(it['source'])
            sources.append(_h.escape(it['source']))
    footer = (f'Generated {datetime.now():%Y-%m-%d %H:%M} · '
              f'{len(items)} figure{"s" if len(items) != 1 else ""} · '
              f'sources: {", ".join(sources)}')

    page = _PAGE_TEMPLATE
    page = page.replace('__FIGURES__', ',\n'.join(figs))
    page = page.replace('__CONFIG__', _js_json_str(PANE_CONFIG))
    page = page.replace('__TABS__', ''.join(tabs))
    page = page.replace('__PANES__', '\n'.join(panes))
    page = page.replace('__FOOTER__', footer)
    page = page.replace('__TITLE__', _h.escape(page_title))
    # LAST: placeholder replacement must never fire on library content
    page = page.replace('__PLOTLYJS__', plotlyjs_src)
    return page


def load_plotlyjs() -> str:
    """The plotly.js bundle this repo's plotly ships, inlined into the page."""
    try:
        from plotly.offline import get_plotlyjs
        src = get_plotlyjs()
        return src.decode('utf-8') if isinstance(src, bytes) else src
    except ImportError as e:
        sys.exit(f'error: embedding plotly.js needs the repo env\'s plotly: {e}')


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Bundle dashboard views into one offline tabbed HTML page.')
    p.add_argument('--pair', action='append', nargs=2, default=[],
                   metavar=('URL', 'VIEWS_DIR'),
                   help='service base URL + its repo-side views directory '
                        '(repeatable)')
    p.add_argument('--view', action='append', default=[], metavar='NAME',
                   help='only bundle these views, in the order given '
                        '(repeatable; default: every view of every pair)')
    p.add_argument('--target', metavar='KEY',
                   help='dispatch ?target=KEY on multi-target routes (e.g. '
                        'polar app: polar vs table); tab labels get " [KEY]"')
    p.add_argument('--font', type=int, metavar='PX',
                   help='?font=PX — same base-font override as the png export')
    p.add_argument('--theme', choices=('light', 'dark'),
                   help='?theme= — rebuild views with that theme')
    p.add_argument('--title', default='Dashboards', help='page <title>')
    p.add_argument('--out', default=str(DEFAULT_OUT),
                   help=f'output html path, relative to the current directory '
                        f'(default: {DEFAULT_OUT})')
    args = p.parse_args(argv)
    if not args.pair:
        p.error('at least one --pair URL VIEWS_DIR is required')
    if args.font is not None and not 6 <= args.font <= 120:
        p.error('--font must be in [6, 120] px (matches the export route)')
    return args


def main(argv=None):
    args = parse_args(argv)
    pairs = [(url, d, Path(d)) for url, d in args.pair]

    names = {}
    for url, lbl, vd in pairs:
        if not vd.is_dir():
            print(f'warning: views dir not found — skipping pair {url} · {vd}',
                  file=sys.stderr)
            names[(url, lbl, vd)] = set()
            continue
        names[(url, lbl, vd)] = {q.stem for q in sorted(vd.glob('*.json'))}
        if not names[(url, lbl, vd)]:
            print(f'warning: no *.json views in {vd}', file=sys.stderr)

    jobs = []                                   # (url, dir_label, view_name)
    if args.view:
        for name in args.view:
            src = next(((u, l, d) for u, l, d in pairs
                        if name in names[(u, l, d)]), None)
            if src is None:
                print(f'warning: no pair serves a view named "{name}"',
                      file=sys.stderr)
            else:
                jobs.append((src[0], src[1], name))
    else:
        for u, l, d in pairs:
            for n in sorted(names[(u, l, d)]):
                if (u, l, n) not in jobs:
                    jobs.append((u, l, n))

    params = {}
    if args.target:
        params['target'] = args.target
    if args.font is not None:
        params['font'] = args.font
    if args.theme:
        params['theme'] = args.theme

    items, warnings = [], []
    for url, lbl, view in jobs:
        try:
            fig = fetch_figure_json(url, view, params)
        except FetchError as e:
            warnings.append(f'{view}: {e}')
            continue
        label = f'{view} [{args.target}]' if args.target else view
        items.append({'label': label, 'fig': fig, 'source': f'{url} · {lbl}'})

    if not items:
        for w in warnings:
            print(f'warning: {w}', file=sys.stderr)
        print('error: nothing bundled (no views matched, or every fetch '
              'failed) — nothing written', file=sys.stderr)
        return 1

    # Cross-pair label collisions (two services bundling the same view name):
    # disambiguate with the views-dir basename, then a counter.
    seen = set()
    for it in items:
        if it['label'] in seen:
            base = Path(it['source'].rsplit(' · ', 1)[1]).name
            cand, n = f"{it['label']} ({base})", 2
            while cand in seen:
                cand, n = f"{it['label']} ({n})", n + 1
            it['label'] = cand
        seen.add(it['label'])

    page = build_html(items, args.title, load_plotlyjs())
    out = Path(args.out)                      # relative paths → cwd
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(page.encode('utf-8'))

    for w in warnings:
        print(f'warning: {w}', file=sys.stderr)
    print(f'wrote {out}: {len(items)} figure(s), '
          f'{len(page.encode("utf-8")) / 1e6:.1f} MB, '
          f'{len(warnings)} fetch(es) skipped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
