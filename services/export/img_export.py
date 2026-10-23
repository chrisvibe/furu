"""Image export proxy: renders a dashboard view headless (kaleido/Chrome inside the
app container) and returns the bytes.

Addressed exactly like the public site, one segment pair per app:

    GET /{site}/{app}?<view>&format=png|svg|pdf&width=...&height=...   (the view a bare key, never view=)

Backends come from deploy/sites.yaml (the router table — bind-mounted, see
docker-compose.yaml), the same file that renders nginx.conf. A legacy flat
/?app=<key> shim maps the pre-2026-10-14 keys so old scripts keep working
until both papers' get_images.sh are repointed; then delete it.
"""
import os

import httpx
import yaml
from fastapi import FastAPI, Request, Response

SITES_FILE = os.environ.get('IMG_EXPORT_SITES', '/etc/img-export/sites.yaml')


def _load_sites() -> dict[str, dict[str, str]]:
    with open(SITES_FILE) as f:
        table = yaml.safe_load(f) or {}
    sites = table.get('sites') or {}
    return {site: {app: f"http://{cfg['service']}:{cfg['port']}" for app, cfg in (spec.get('apps') or {}).items()}
            for site, spec in sites.items()}


SITES = _load_sites()

# The old flat app= keys → (site, app). The papers' URL paths used these as the
# last segment; hegg's analysis vs furu's needed the furu- prefix in that scheme.
_LEGACY_KEYS = {
    'path-integration': ('hegg', 'path-integration'),
    'analysis':         ('hegg', 'analysis'),
    'polar':            ('hegg', 'polar'),
    'trace-plot':       ('hegg', 'trace'),
    'furu-design':      ('furu', 'design'),
    'furu-capacity':    ('furu', 'capacity'),
    'furu-analysis':    ('furu', 'analysis'),
    'figure1':          ('furu', 'figure1'),
    'temporal':         ('furu', 'temporal'),
}

app = FastAPI()


def _available() -> str:
    return ', '.join(f'/{s}/{a}' for s, apps in SITES.items() for a in apps)


async def _render(backend: str, query: str) -> Response:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f'{backend}/?{query}', timeout=120.0)
        except Exception as e:
            return Response(f'Connection error: {e}', status_code=502,
                            media_type='text/plain')
    media = r.headers.get('content-type', 'application/octet-stream')
    return Response(content=r.content, status_code=r.status_code, media_type=media)


@app.get('/{site}/{app_name}')
async def export_img(site: str, app_name: str, request: Request):
    backend = SITES.get(site, {}).get(app_name)
    if backend is None:
        return Response(f'Unknown app "{site}/{app_name}". Available: {_available()}',
                        status_code=400, media_type='text/plain')
    # The app's export route carries everything as query params; the path named the app.
    return await _render(backend, request.url.query)


@app.get('/')
async def export_img_legacy(request: Request):
    key = request.query_params.get('app')
    if not key:
        return Response(f'Require the app path: /{{site}}/{{app}}?<view>&format=... '
                        f'(or legacy &app=<name>). Available: {_available()}',
                        status_code=400, media_type='text/plain')
    mapped = _LEGACY_KEYS.get(key)
    if mapped is None:
        return Response(f'Unknown app "{key}". Available: {_available()}',
                        status_code=400, media_type='text/plain')
    site, app_name = mapped
    # Strip &app= — the path-based route doesn't know it.
    qs = '&'.join(p for p in request.url.query.split('&') if not p.startswith('app='))
    return await _render(SITES[site][app_name], qs)
