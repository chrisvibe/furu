"""View JSON -> filtered df, in the one canonical order (PLAN_graph_viz.md §3).

The ordering matters and is why this lives in exactly ONE place:

    1. static factor filters   (`.astype(str).isin(values)`)
    2. computed fields         (`parse_and_apply_fields`)
    3. derived factor filters  (after fields, so computed columns exist)
    4. `filter-expr`           (`df.query`)

`ScatterDashboard._reconstruct_figure` implemented this ordering for the
server-side PNG export; `wsgi.precompute_weights` needs the SAME rows the
dashboard would render. If the CLI re-implemented the ordering, the two could
drift and the CLI would train the wrong rows — silently, since a filter
selecting different rows still "works".

Consumers: `_reconstruct_figure` (this package) and the precompute CLI; the UI's view
loader reuses `missing_filter_values` to warn about the same mismatch.
Aggregation/binning is deliberately NOT part of this: the CLI needs raw rows
carrying `i`/`j` (and in raw mode `aggregate_df` returns the df unchanged
anyway — it destroys `i`/`j` only when a real aggregation is active).
"""
import numpy as np
import pandas as pd

from .utils import ViewDataMismatch, parse_and_apply_fields


def missing_filter_values(df, factor_filters):
    """`{factor: [saved values]}` for the static filter values `df` does not have (empty when all match).

    Values compare as strings, the `.astype(str)` contract of step 1; a filtered factor that is not a
    column at all counts as missing in full.
    """
    missing = {}
    for factor, values in factor_filters.items():
        if values:
            present = {str(v) for v in df[factor].unique()} if factor in df.columns else set()
            if absent := [v for v in values if v not in present]:
                missing[factor] = absent
    return missing


def apply_view_filters(df, view_data):
    """Apply a saved view's filters to `df`. Returns `(filtered_df, factor_filters)`.

    `factor_filters` is the resolved static filter dict (the view's
    `_factor_filters` or `_extras.factor_filters`), returned because callers
    that continue into binning/aggregation still need it (color grouping).

    A static filter value that the data does not have raises `ViewDataMismatch` (see there): the
    rows would otherwise be a silent subset, or none. A failing `filter-expr` RAISES — the caller
    owns the policy (the dashboard reports the error and continues unfiltered by the query; the CLI
    wants fail-loud). Derived filters (step 3) skip a missing column silently.
    """
    _extras = view_data.get('_extras') or {}

    # 1. Static factor filters
    _factor_filters = view_data.get('_factor_filters') or _extras.get('factor_filters') or {}
    missing = missing_filter_values(df, _factor_filters)
    if missing:
        raise ViewDataMismatch(missing)
    mask = np.ones(len(df), dtype=bool)
    for factor, values in _factor_filters.items():
        if values and factor in df.columns:
            mask &= df[factor].astype(str).isin(values)
    filtered_df = df[mask].copy()

    # 2. Computed fields
    fields_expr = view_data.get('fields-expr') or ''
    if fields_expr.strip():
        filtered_df, _, _ = parse_and_apply_fields(filtered_df, fields_expr)

    # 3. Derived factor filters (after fields so computed columns exist)
    _derived_filters = view_data.get('_derived_filter_filters') or _extras.get('derived_filter_filters') or {}
    for factor, values in _derived_filters.items():
        if values and factor in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[factor].astype(str).isin(values)]

    # 4. filter-expr
    filter_expr = view_data.get('filter-expr') or ''
    if filter_expr.strip():
        filtered_df = filtered_df.query(filter_expr)

    return filtered_df, _factor_filters
