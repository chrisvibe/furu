"""Single source of truth for dashboard theming (chrome + figures).

This is the *only* module under ``dashboards/`` that may contain a colour
literal.  It deliberately imports nothing from this project — ``polar.py``
already imports ``wsgi.base``, which imports ``dashboards.scatter``, so a latent
import cycle exists and must not be deepened.  plotly + dash + stdlib only.

Two background tokens exist on purpose and must not be collapsed:

``surface``
    the *page* colour.  ``trace`` deliberately paints its figure canvas the page
    colour, so its ``paper_bgcolor``/``plot_bgcolor`` map here, never to ``paper``.
``paper`` / ``plot``
    the *figure canvas* — the values ``plotly_dark`` / ``plotly_white`` use.

``fig_title_font``, ``overlay_text``, ``table_font`` and the table fills are
separate tokens rather than aliases of ``fig_font`` because their current values
differ by a hair; folding them in would silently change exported figures.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc


# ── Categorical palettes ──────────────────────────────────────────────────────
# Designed with the dataviz method and checked with its validator
# (scripts/validate_palette.js) against the *figure canvas* of each theme —
# white for light, rgb(17,17,17) for dark:
#
#   node validate_palette.js "<light hexes>" --mode light --surface "#ffffff"
#   node validate_palette.js "<dark hexes>"  --mode dark  --surface "#111111"
#
# Slots 1-10 are a validated ordered core: hues are assigned in this fixed order
# and NEVER cycled.  Both modes pass every check on that core — lightness band,
# chroma floor, adjacent CVD separation (worst ΔE 9.2 light / 9.1 dark, floor 8)
# and normal-vision separation (24.1 / 15.9, floor 15).  Three light slots sit
# below 3:1 against white, so the dataviz relief rule applies there: ship visible
# direct labels (the legend) rather than relying on the swatch alone.
#
# Slots 11-61 are a documented extended ramp — a golden-angle (137.508°) OKLCH
# hue sweep at the mode's mid-band lightness with a small alternating
# lightness/chroma jitter, each entry walked to the in-band lightness that clears
# the chroma floor.  The whole 61 still passes lightness band, chroma floor,
# normal-vision separation and contrast, but groups past the core are NOT
# CVD-separation-guaranteed; that is the accepted cost of supporting more groups
# than any palette can keep distinguishable.
#
# Both palettes MUST stay the same length: `bump_color_offset` stores palette
# offsets in saved-view JSON, and equal lengths keep those offsets meaningful
# across a theme flip.

PALETTE_LIGHT = (
    # ── Validated ordered core (10 hues, fixed order, never cycled) ───────────
    '#2a78d6',     # 1 blue
    '#eb6834',     # 2 orange
    '#1baf7a',     # 3 aqua
    '#985704',     # 4 brown
    '#e87ba4',     # 5 magenta
    '#008300',     # 6 green
    '#4a3aa7',     # 7 violet
    '#e34948',     # 8 red
    '#0a87aa',     # 9 teal
    '#eda100',     # 10 yellow
    # ── Extended ramp (slots 11-61) ──────────────────────────────────────────
    '#019164', '#8667b9', '#b88001', '#018ea6', '#c04883', '#729b44',
    '#5072db', '#b55f35', '#02a292', '#9b60a8', '#8e7a01', '#2597cd',
    '#c8465f', '#388e51', '#8978e9', '#ac6810', '#0098a2', '#be6ca4',
    '#748400', '#437cc3', '#de5e46', '#019070', '#955ac4', '#af8517',
    '#0188a8', '#b45777', '#55a233', '#6672c3', '#bb5c01', '#03a19b',
    '#ac50ab', '#857e01', '#0094e1', '#b9585b', '#01925a', '#927bcf',
    '#a66d01', '#0194a6', '#d15c9e', '#698729', '#4375db', '#cb6f4e',
    '#028e7a', '#9661ac', '#a88900', '#0086b2', '#c74668', '#56a05b',
    '#6f68d7', '#af661b', '#03a0a4',
)

PALETTE_DARK = (
    # ── Validated ordered core (10 hues, fixed order, never cycled) ───────────
    '#3987e5',     # 1 blue
    '#d95926',     # 2 orange
    '#199e70',     # 3 aqua
    '#b66c18',     # 4 brown
    '#d55181',     # 5 magenta
    '#008300',     # 6 green
    '#9085e9',     # 7 violet
    '#e66767',     # 8 red
    '#0994ba',     # 9 teal
    '#c98500',     # 10 yellow
    # ── Extended ramp (slots 11-61) ──────────────────────────────────────────
    '#029265', '#8869ba', '#b67e00', '#018ea6', '#c14985', '#719942',
    '#5173dd', '#b76137', '#03a090', '#9c61a9', '#907b01', '#2395cb',
    '#ca4860', '#3a9052', '#8776e8', '#ad6a13', '#0098a2', '#bc6aa2',
    '#768501', '#457dc4', '#dc5d44', '#009171', '#975bc6', '#ad8414',
    '#0189aa', '#b65978', '#53a131', '#6774c5', '#be5d00', '#009f9a',
    '#ae52ad', '#877f01', '#0292de', '#ba5a5c', '#02935b', '#917ace',
    '#a86e01', '#0194a6', '#cf5a9c', '#6a882a', '#4576dc', '#c96e4d',
    '#02907c', '#9863ae', '#a68700', '#0087b4', '#c84869', '#559e5a',
    '#716ad9', '#b0671d', '#009ea2',
)

# Hue stops for the `<digit> + shift-click` legend recolour gesture.
HUE_STOPS = (0, 137, 275, 52, 190, 327, 105, 242, 20, 158)

# Figure text. Phase 1 kept three near-miss variants of this (trace's annotations,
# polar's corner title, polar's export-table font) so the refactor could be proven
# pixel-neutral; phase 2 collapses them onto one value per theme.
_FIG_FONT_DARK  = '#f2f5fa'
_FIG_FONT_LIGHT = '#2a3f5f'


# ── The theme record ──────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Theme:
    name: str                      # 'dark' | 'light' — also the data-theme value

    # page chrome → emitted as CSS custom properties
    surface: str                   # page background
    panel: str                     # inputs / raised panels
    control: str                   # buttons
    border: str
    text: str
    text_dim: str
    faint: str
    accent: str                    # "active"/"selected" background
    accent_text: str
    danger: str
    ok: str
    delete: str
    dd_bg: str
    dd_bg_hover: str
    dd_bg_sel: str
    dd_shadow: str
    grid_hit_bg: str               # AG Grid constrained-cell highlight
    grid_hit_fg: str
    row_hit_bg: str                # AG Grid clicked-direction row highlight
    row_hit_fg: str
    on_solid: str
    color_scheme: str              # CSS color-scheme
    grid_class: str                # ag-theme-alpine[-dark]

    # plotly
    template: str                  # 'br_dark' | 'br_light'
    paper: str                     # figure canvas
    plot: str
    fig_font: str
    fig_title_font: str            # polar's corner title — own token, see docstring
    legend_bg: str
    legend_border: str
    palette: tuple
    other: str                     # the "Other" catch-all group colour
    uniform: str                   # single-series scatter marker colour
    hsv_sv: tuple                  # (s, v) for the hue-stop gesture

    # figure overlays (trace) + export table (polar)
    sel_stroke: str
    sel_fill: str
    hl_row: str
    hl_row_fill: str
    hl_col_fill: str
    hl_parent: str
    hl_parent_fill: str
    hl_child: str
    hl_child_fill: str
    binary_lo: str
    binary_hi: str
    greys: str                     # named continuous scale for non-binary traces
    polar_mark: str                # polar angular-sector highlight
    overlay_text: str              # trace's paper-space annotations
    table_font: str                # polar export table
    table_header_fill: str
    table_line: str               # go.Table cell/header gridlines
    table_cell_fill: str

    @property
    def is_light(self) -> bool:
        return self.name == 'light'

    def css_vars(self) -> dict[str, str]:
        """The one Python→CSS bridge.  Keys are custom-property names."""
        return {
            '--surface':      self.surface,
            '--panel':        self.panel,
            '--control':      self.control,
            '--border':       self.border,
            '--text':         self.text,
            '--text-dim':     self.text_dim,
            '--faint':        self.faint,
            '--accent':       self.accent,
            '--accent-text':  self.accent_text,
            '--danger':       self.danger,
            '--ok':           self.ok,
            '--delete':       self.delete,
            '--on-solid':     self.on_solid,
            '--dd-bg':        self.dd_bg,
            '--dd-bg-hover':  self.dd_bg_hover,
            '--dd-bg-sel':    self.dd_bg_sel,
            '--dd-shadow':    self.dd_shadow,
            '--grid-hit-bg':  self.grid_hit_bg,
            '--grid-hit-fg':  self.grid_hit_fg,
            '--row-hit-bg':   self.row_hit_bg,
            '--row-hit-fg':   self.row_hit_fg,
            '--color-scheme': self.color_scheme,
        }


# ── The two themes ────────────────────────────────────────────────────────────
# Every figure-affecting token below is seeded with the value the code used
# before the refactor, so phase 1 reproduces today's exports exactly.

DARK = Theme(
    name='dark',

    surface='#1e1e1e',
    panel='#2a2a2a',
    control='#333333',
    border='#444444',
    text='#dddddd',
    text_dim='#888888',
    faint='#555555',
    accent='#1a3a5c',
    accent_text='#ffffff',
    danger='#ff6b6b',
    ok='#4CAF50',          # solid affirmative-action button
    delete='#f44336',      # solid destructive-action button
    dd_bg='#2a2a2a',
    dd_bg_hover='#3a3a3a',
    dd_bg_sel='#1a3a5c',
    dd_shadow='rgba(0,0,0,0.6)',
    grid_hit_bg='#0d2035',
    grid_hit_fg='#66aaff',
    row_hit_bg='#1a3a2a',
    row_hit_fg='#aaffaa',
    on_solid='#ffffff',    # text on a solid ok/delete button
    color_scheme='dark',
    grid_class='ag-theme-alpine-dark',

    template='br_dark',
    paper='rgb(17,17,17)',
    plot='rgb(17,17,17)',
    fig_font=_FIG_FONT_DARK,
    fig_title_font=_FIG_FONT_DARK,
    legend_bg='rgb(17,17,17)',
    legend_border='#444',
    palette=PALETTE_DARK,
    other='#CCCCCC',
    uniform='#3987e5',      # the core palette's blue
    hsv_sv=(1.0, 1.0),

    sel_stroke='rgba(255,200,50,0.55)',
    sel_fill='rgba(255,200,50,0.07)',
    hl_row='LimeGreen',
    hl_row_fill='rgba(50,205,50,0.10)',
    hl_col_fill='rgba(50,205,50,0.08)',
    hl_parent='#5dade2',
    hl_parent_fill='rgba(93,173,226,0.12)',
    hl_child='#ec7063',
    hl_child_fill='rgba(236,112,99,0.12)',
    binary_lo='#111111',
    binary_hi='#eeeeee',
    greys='Greys_r',
    polar_mark='rgba(255, 200, 50, 0.18)',
    overlay_text=_FIG_FONT_DARK,
    table_font=_FIG_FONT_DARK,
    table_header_fill='#222222',
    table_line='#2a2a2a',
    table_cell_fill='rgb(17,17,17)',   # == paper
)

LIGHT = replace(
    DARK,
    name='light',

    surface='#ffffff',
    panel='#ffffff',
    control='#ececec',
    border='#cccccc',
    text='#1a1a1a',
    text_dim='#666666',
    faint='#999999',
    accent='#cfe4f7',
    accent_text='#0d2035',
    danger='#c62828',
    ok='#2e7d32',
    delete='#c62828',
    dd_bg='#ffffff',
    dd_bg_hover='#eef2f6',
    dd_bg_sel='#cfe4f7',
    dd_shadow='rgba(0,0,0,0.18)',
    grid_hit_bg='#dceafa',
    grid_hit_fg='#14508c',
    row_hit_bg='#dcf0e0',
    row_hit_fg='#1b5e20',
    color_scheme='light',
    grid_class='ag-theme-alpine',

    template='br_light',
    paper='white',
    plot='white',
    fig_font=_FIG_FONT_LIGHT,
    fig_title_font=_FIG_FONT_LIGHT,
    legend_bg='white',
    legend_border='#444',
    palette=PALETTE_LIGHT,
    other='#888888',
    uniform='#2a78d6',      # the core palette's blue
    hsv_sv=(0.85, 0.75),

    binary_lo='#eeeeee',
    binary_hi='#111111',
    greys='Greys',
    overlay_text=_FIG_FONT_LIGHT,
    table_font=_FIG_FONT_LIGHT,
    table_line='white',
    table_header_fill='#e8e8e8',
    table_cell_fill='white',           # == paper
)

_BY_NAME = {'dark': DARK, 'light': LIGHT}


# ── CSS custom-property references, for inline style dicts ───────────────────
# An inline style may reference a custom property, so a style dict written in
# Python needs no server-side knowledge of the active theme at all — the
# :root / [data-theme="light"] blocks decide at computed-value time.
SURFACE     = 'var(--surface)'
PANEL       = 'var(--panel)'
CONTROL     = 'var(--control)'
BORDER      = 'var(--border)'
TEXT        = 'var(--text)'
TEXT_DIM    = 'var(--text-dim)'
FAINT       = 'var(--faint)'
ACCENT      = 'var(--accent)'
ACCENT_TEXT = 'var(--accent-text)'
DANGER      = 'var(--danger)'
OK          = 'var(--ok)'
DELETE      = 'var(--delete)'
ON_SOLID    = 'var(--on-solid)'
GRID_HIT_BG = 'var(--grid-hit-bg)'
GRID_HIT_FG = 'var(--grid-hit-fg)'
ROW_HIT_BG  = 'var(--row-hit-bg)'
ROW_HIT_FG  = 'var(--row-hit-fg)'


# ── Resolver ──────────────────────────────────────────────────────────────────

def resolve(theme) -> Theme:
    """The ONE theme predicate.

    Accepts a Theme, a name, or whatever a ``dcc.Checklist`` produced.  The only
    contract is membership of the string ``'light'`` — which is why trace can
    keep its ``'light'`` entry buried inside ``tp-plot-toggles``.
    """
    if isinstance(theme, Theme):
        return theme
    if theme is None:
        return DARK
    if isinstance(theme, str):
        return _BY_NAME.get(theme, LIGHT if theme == 'light' else DARK)
    try:
        return LIGHT if 'light' in theme else DARK
    except TypeError:
        return DARK


def is_light_theme(theme) -> bool:
    """Back-compat shim; re-exported from ``wsgi.base`` for external callers."""
    return resolve(theme).is_light


# ── Plotly templates ──────────────────────────────────────────────────────────

def _overlay(t: Theme) -> go.layout.Template:
    """The per-theme layer merged over a built-in plotly template.

    The explicit ``legend`` block is the actual fix for the theme-toggle bug:
    neither ``plotly_dark`` nor ``plotly_white`` states legend styling at all, so
    legends previously depended on plotly.js's own fallbacks.  Every value here
    reproduces those computed defaults exactly (bgcolor ← paper, font ← layout
    font, bordercolor ``#444``, borderwidth ``0``).
    """
    return go.layout.Template(layout=dict(
        paper_bgcolor=t.paper,
        plot_bgcolor=t.plot,
        font=dict(color=t.fig_font),
        colorway=list(t.palette),
        title=dict(font=dict(color=t.fig_font)),
        legend=dict(
            font=dict(color=t.fig_font),
            bgcolor=t.legend_bg,
            bordercolor=t.legend_border,
            borderwidth=0,
        ),
    ))


_registered = False


def register_templates() -> None:
    """Register ``br_dark`` / ``br_light``.  Idempotent.

    MUST use the two-argument ``merge_templates``: ``pio.templates['plotly_dark']``
    with a single name returns the *cached built-in object itself*, so mutating
    it would corrupt ``plotly_dark`` for the whole process.  The two-argument
    form validates — and therefore copies — both inputs.
    """
    global _registered
    if _registered:
        return
    for t, base in ((DARK, 'plotly_dark'), (LIGHT, 'plotly_white')):
        pio.templates[t.template] = pio.templates.merge_templates(
            pio.templates[base], _overlay(t))
    # Read once per go.Figure() construction from a never-mutated module global,
    # so it carries no multi-worker hazard.  It only affects error/empty figures
    # that set no template, turning a plotly-white flash into correct dark.
    pio.templates.default = DARK.template
    _registered = True


# ── Shared UI pieces ──────────────────────────────────────────────────────────

THEME_PREF_STORE_ID = 'theme-pref'


def make_theme_toggle(toggle_id: str = 'theme-toggle', **checklist_kwargs):
    """The Light-mode checklist.  scatter and polar both build this."""
    kwargs = dict(
        options=[{'label': ' Light mode', 'value': 'light'}],
        value=[],
        style={'display': 'inline-block', 'fontSize': '11px'},
        inputStyle={'marginRight': '3px'},
        labelStyle={'color': 'var(--text-dim)'},
    )
    kwargs.update(checklist_kwargs)
    return dcc.Checklist(id=toggle_id, **kwargs)


def theme_pref_store():
    """Holds *only* the user preference.  Never written by view save/load, so no
    saved-view JSON changes shape."""
    return dcc.Store(id=THEME_PREF_STORE_ID, storage_type='local')


def _css_block(selector: str, t: Theme) -> str:
    body = '\n'.join(f'    {k}: {v};' for k, v in t.css_vars().items())
    return f'{selector} {{\n{body}\n}}'


def theme_css() -> str:
    """The generated custom-property blocks — the only Python→CSS bridge.

    Structure and selectors live in ``assets/theme.css``, which contains no
    colour literals at all.
    """
    return (_css_block(':root', DARK) + '\n'
            + _css_block('[data-theme="light"]', LIGHT) + '\n')


def index_string(base: str) -> str:
    """Inject the generated theme variables into a Dash index template."""
    return (base
            .replace('<body>', '<body class="br-body">')
            .replace('</head>', f'<style>\n{theme_css()}</style>\n</head>'))


# Flip <html data-theme=…> the instant the checklist changes: every CSS-driven
# piece of chrome follows with no server round-trip.  The same callback persists
# the choice into the local-storage Store.
_SYNC_JS = """
function(value) {
    var light = Array.isArray(value) ? value.indexOf('light') >= 0 : value === 'light';
    document.documentElement.dataset.theme = light ? 'light' : 'dark';
    return light ? 'light' : 'dark';
}
"""

# First load follows the stored preference, or the OS when there is none.
# Returning a *merged* list (rather than a replacement) is what lets trace keep
# 'light' buried inside its existing tp-plot-toggles value.
# Args: [0] store id (constant trigger), [1] stored pref, [2] current toggle value.
_INIT_JS = """
function(_id, pref, current) {
    var light = (pref === 'light' || pref === 'dark')
        ? pref === 'light'
        : !!(window.matchMedia
             && window.matchMedia('(prefers-color-scheme: light)').matches);
    document.documentElement.dataset.theme = light ? 'light' : 'dark';
    var list = Array.isArray(current) ? current : (current ? [current] : []);
    var has  = list.indexOf('light') >= 0;
    if (light === has) return window.dash_clientside.no_update;
    return light ? list.concat(['light'])
                 : list.filter(function (v) { return v !== 'light'; });
}
"""


def register_theme_attribute(app, toggle_id: str = 'theme-toggle') -> None:
    """Mirror the checklist onto ``<html data-theme>`` and seed it from the OS.

    The two callbacks deliberately do not form a cycle: the init one is triggered
    by the Store's constant ``id`` and only *reads* its data as State.
    """
    from dash import Input, Output, State

    app.clientside_callback(
        _SYNC_JS,
        Output(THEME_PREF_STORE_ID, 'data'),
        Input(toggle_id, 'value'),
    )
    app.clientside_callback(
        _INIT_JS,
        Output(toggle_id, 'value'),
        Input(THEME_PREF_STORE_ID, 'id'),
        State(THEME_PREF_STORE_ID, 'data'),
        State(toggle_id, 'value'),
    )
