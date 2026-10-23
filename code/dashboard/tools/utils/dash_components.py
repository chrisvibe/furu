"""Shared dashboard chrome for the single-(i,j) tool cluster (issues #70/#71).

`weights_dash.py` grew a set of generic interactive controls
(pandas-query filter, computed fields, colourbar ticks, top-N clamp, parts
checklist); `states_dash.py`'s serve mode needs the same controls over a
different points-frame. This module is the ONE home for them so both
dashboards consume a single implementation.

What lives here (all pure pandas/numpy -- no Julia, no server boot, no figure
assembly; each dashboard keeps its own trace/layout code because a TIME
trajectory, a 3D graph with edges and a 2D node spread are genuinely different
concerns):

  - `ComputedFields`   -- the "name = pandas-expression" derived-column registry:
    parse from the text box, eval into the points df, colour-normalize the new
    columns. The normalization routes through the shared
    `readout_analysis.normalize` (the graph tool previously re-implemented the
    min-max guard inline -- exactly the duplication the #71 DRY review noted).
  - `colorbar_ticks`   -- normalized [0,1] back to raw-value (or categorical
    category-name) tick labels for a plotly colourbar.
  - `clamp_top_n`, `fmt_val`, `COLORSCALE_OPTIONS`/`DEFAULT_COLORSCALE`.
  - `part_labels`      -- the per-node I/O/R labelling from the model's OWN
    masks (the `_part()` semantics: NOT the multi-label 'O' node attribute,
    which coincides with 'R' and cannot distinguish output from reservoir).
  - `query_help_components` + `register_query_help_toggle` -- the collapsible
    pandas-query cheat sheet (dash is imported lazily so the pure helpers stay
    importable without it).

Consumers: `weights_dash` (node points + edges) and
`states_dash` (TIME/SPACE state points).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

DEFAULT_COLORSCALE = 'Viridis'

# Common perceptually-uniform colorscales for colour-by-attribute traces.
COLORSCALE_OPTIONS = [
    'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
    'Turbo', 'Rainbow', 'Jet', 'Blues', 'Reds', 'Greens',
    'RdYlBu', 'RdBu', 'Spectral', 'Portland', 'Picnic',
]


def clamp_top_n(top_n, n_max):
    """Clamp a Top-N box value to `[0, n_max]` (0/None = no cap).

    Avoids an illegal slice on small graphs/states where the default N exceeds
    the population.
    """
    if top_n is None or top_n <= 0:
        return 0
    return min(int(top_n), n_max)


def fmt_val(v):
    """Compact value formatting for hover text (4 significant digits for floats)."""
    return f'{v:.4g}' if isinstance(v, (float, np.floating)) else str(v)


def colorbar_ticks(df, attr):
    """Return `(tickvals, ticktext)` mapping a normalized [0,1] colourbar back
    to interpretable labels for `df[attr]`.

    Numeric attributes: tick labels show raw data values across [min, max].
    Categorical attributes: tick labels show the actual category names at their
    factorized-code positions (0 = first category). A constant column gets a
    single tick at its own value (mirrors `readout_analysis.normalize`'s
    constant -> 0.0 guard).
    """
    raw = df[attr]
    if pd.api.types.is_numeric_dtype(raw):
        numeric = raw.astype(float)
        lo, hi = float(numeric.min()), float(numeric.max())
        if hi - lo <= 0:
            return [0.0], [f'{lo:.4g}']
        tickvals = [0.0, 0.25, 0.5, 0.75, 1.0]
        ticktext = [f'{lo + v * (hi - lo):.4g}' for v in tickvals]
        return tickvals, ticktext
    codes, uniques = pd.factorize(raw, sort=False)
    if len(uniques) <= 1:
        return [0.0], [str(uniques[0]) if len(uniques) else '?']
    hi = float(len(uniques) - 1)
    tickvals = [float(c) / hi for c in range(len(uniques))]
    ticktext = [str(u) for u in uniques]
    return tickvals, ticktext


class ComputedFields:
    """Registry of derived columns parsed from `name = expression` text boxes.

    One home for the graph tool's and the state tool's identical computed-field
    flow: the dashboard callback calls `parse()` with the raw box value, then
    the renderer calls `apply(df)` + `normalize_into(df, df_normalized)` before
    colouring, exactly as the graph tool always did.
    """

    def __init__(self):
        self.fields: dict[str, str] = {}  # name -> expression

    def parse(self, text) -> str:
        """Register/clear from the raw input-box value; returns a UI status line.

        Semantics preserved from the graph tool verbatim: a `name = expr`
        string registers (empty name or expr -> invalid); a non-empty string
        without '=' is invalid; an EMPTY box clears every computed field.
        """
        text = text or ''
        if '=' in text:
            name, _, expr = text.partition('=')
            name, expr = name.strip(), expr.strip()
            if name and expr:
                self.fields[name] = expr
                return f'Computed field: {name} = {expr}'
            return 'Invalid: use "name = expression"'
        if text.strip():
            return 'Invalid: use "name = expression" (e.g. my = abs(w0))'
        self.fields.clear()
        return ''

    def names(self):
        return sorted(self.fields)

    def options(self, base_attrs):
        """Dropdown options: base attribute columns plus computed fields."""
        return sorted(set(base_attrs) | set(self.fields))

    def apply(self, df):
        """Eval each expression into `df` (in place). Returns the applied
        `(name, expr)` pairs; failures are reported to stderr and skipped (the
        field stays registered but never crashes the render)."""
        applied = []
        for name, expr in list(self.fields.items()):
            try:
                df[name] = df.eval(expr, engine='python')
                applied.append((name, expr))
            except Exception as exc:  # noqa: BLE001 -- surfaced, not hidden
                print(f'Computed field {name!r} ({expr!r}) failed: {exc}',
                      file=sys.stderr, flush=True)
        return applied

    def normalize_into(self, source_df, target_norm):
        """Add `<name>_normalized_` colour columns for newly-applied computed
        fields into `target_norm` (the pre-normalized frame), skipping columns
        that already have one. Uses the shared constant-column-guarded
        `readout_analysis.normalize` instead of a private re-implementation."""
        from .readout_analysis import normalize

        missing = [n for n in self.fields
                   if n in source_df.columns and n + '_normalized_' not in target_norm.columns]
        if not missing:
            return
        sub = normalize(source_df[missing])
        for col in sub.columns:
            target_norm[col] = sub[col]


def part_labels(model):
    """Per-node I/O/R labels (0..N_total-1) from the model's own masks.

    'I' = input_nodes_mask, 'O' = inside out_slice, else 'R'. This is the ONLY
    way to distinguish reservoir (R) from output (O) nodes: the graph's 'O'
    attribute is multi-label ("read by the output layer") and currently
    coincides with 'R'.
    """
    node_ids = np.arange(model.N_total)
    out = (node_ids >= model.out_slice.start) & (node_ids < model.out_slice.stop)
    return np.where(model.input_nodes_mask, 'I', np.where(out, 'O', 'R'))


# ── Query help (dash bits imported lazily so the pure helpers above stay
# ── importable without dash installed) ───────────────────────────────────────

def query_help_components(help_text, toggle_id='query_help_toggle',
                          div_id='query_help_examples'):
    """The `?` toggle + collapsible cheat-sheet block for the pandas-query box.

    `help_text` is dashboard-specific (each tool documents its own columns);
    the toggle/collapse machinery is shared.
    """
    from dash import html

    toggle = html.Span(' ?', id=toggle_id,
                       style={'cursor': 'pointer', 'color': '#007bff',
                              'fontWeight': 'bold', 'fontSize': 14, 'marginLeft': 6})
    div = html.Div(id=div_id, children=[
        html.Pre(help_text,
                 style={'fontSize': 11, 'lineHeight': 1.4, 'background': '#f5f5f5',
                        'padding': '8px', 'borderRadius': '4px',
                        'border': '1px solid #ddd'}),
    ], style={'display': 'none', 'marginTop': '4px'})
    return toggle, div


def register_query_help_toggle(app, toggle_id='query_help_toggle',
                               div_id='query_help_examples'):
    """Register the odd-click-open / even-click-close callback for the help box."""
    from dash import Input, Output

    @app.callback(
        Output(div_id, 'style'),
        Input(toggle_id, 'n_clicks'),
        prevent_initial_call=True,
    )
    def _toggle(n_clicks):
        if n_clicks and n_clicks % 2 == 1:
            return {'display': 'block', 'marginTop': '4px'}
        return {'display': 'none', 'marginTop': '4px'}
