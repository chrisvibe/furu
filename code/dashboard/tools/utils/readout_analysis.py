"""Pure readout-strategy analysis for one BooleanReservoir checkpoint.

Separated from the Dash server (weights_dash.py) so the diagnostics
are unit-testable without booting the app. None of these functions touches Julia or reservoir states; they read
only the stored `readout_W`, the graph and the input mask.

Scope: the question is whether the
readout is evidence the reservoir COMPUTED the task or just passed the input
through. Readout geometry alone cannot PROVE either (a ridge-solved readout is a
spread linear map either way), so these are honest *necessary* proxies, not
sufficient proof -- the UI must label them as hypotheses, not verdicts:

  - per-output row concentration: does one output read few reservoir nodes?
  - cascade depth from inputs: structural "must recurrence happen before the
    readout sees this node" proxy.

The upgraded `per_output_diagnostic` FUSES the two: it reports, for each output,
the depth of the node it reads hardest, so a concentrated readout on a SHALLOW
node is flagged as "input-shadow / hash-consistent" while the same concentration
on a DEEP node is flagged "consistent with reservoir computation."

The true discriminator (effective rank of the reservoir feature matrix vs the
input) needs states, which single_run does NOT persist (SAVE_KEYS=['weights',
'parameters']) -- that is a separate probe.
"""
import numpy as np
import pandas as pd
import networkx as nx


def apply_pandas_query(df, query):
    """Apply a pandas `query` string, returning (filtered_df, error_message).

    Never silently drops to the full frame on a bad query: on error it returns
    the input df unchanged plus a message the caller surfaces in the UI (the old
    dash tool swallowed the exception and quietly rendered everything, which
    made a typo look like a meaningful cluster).
    """
    if not query or not query.strip():
        return df, ''
    try:
        return df.query(query), ''
    except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not hidden
        return df, f'Pandas query failed: {exc}'


def normalize(df, suffix='_normalized_'):
    """Rescale every column of `df` to [0, 1], adding columns named `<col><suffix>`.

    ONLY for pandas colour-rescaling consumers (the graph tool / any DataFrame
    legend). NOT for the numpy state-reducer (`state_analysis.reduce*`): that
    path feeds (n_rows, n_feat) arrays to PCA/UMAP/tSNE, and rescaling columns to
    [0,1] would distort the metric; it also has no div-by-zero to guard (it does
    not zscore). Do NOT route the reducer through this.

    - Non-numeric (categorical/string) columns are factorized (codes mapped to
      [0,1]) via pd.factorize, which handles pandas' `str` extension dtype (the
      old `dtype == object` check silently missed it).
    - A constant column (min == max, including an all-equal categorical) is
      given 0.0, NOT NaN -- the old code divided by zero and broke the colour
      scale once a filter left a single node.
    """
    df2 = pd.DataFrame(index=df.index)
    for col in df:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric = series.astype(float)
        else:
            codes, _ = pd.factorize(series, sort=False)
            numeric = codes.astype(float)
        lo, hi = numeric.min(), numeric.max()
        if hi - lo > 0:
            df2[col + suffix] = (numeric - lo) / (hi - lo)
        else:
            df2[col + suffix] = 0.0
    return df2


def per_output_row_diagnostics(readout_W):
    """Per-output readout geometry (rows of `readout_W`), no structure yet.

    Each output dim k reports how it reads the reservoir:
      pr           participation ratio  (1 = single node, ~n_r = uniform)
      support      count of reservoir nodes with |W| above 1% of the row max
      top_share    fraction of the row's total |W| held by its single largest node
      top_node     RESERVOIR-RELATIVE index of the node carrying max |W| (None if unused)
      single_node  bool: does this output read essentially one node?

    `top_node` is reservoir-relative (0..N_R-1). The Dash tool maps it to an
    absolute graph node id via (res_slice.start + top_node) before querying depth.
    """
    W = np.asarray(readout_W, dtype=float)
    if W.ndim == 1:
        W = W[None, :]  # single output
    rows = []
    for k in range(W.shape[0]):
        a = np.abs(W[k])
        s = a.sum()
        if s == 0:  # unused output dim (all-zero row)
            rows.append(dict(pr=0.0, support=0, top_share=0.0,
                             top_node=None, single_node=False))
            continue
        argmax = int(np.argmax(a))
        pr = s * s / np.sum(a * a)
        support = int(np.sum(a > 0.01 * a.max()))
        top_share = a.max() / s
        rows.append(dict(pr=float(pr), support=support, top_share=float(top_share),
                         top_node=argmax, single_node=(support == 1)))
    return rows


def fuse_with_depth(geoms, depth_by_abs_node, res_slice_start):
    """Merge each output's geometry with the cascade depth of the node it reads
    hardest, and produce a single hypothesis label + numeric flag.

    Params:
      geoms             from per_output_row_diagnostics
      depth_by_abs_node dict abs_node_id -> depth (as built by
                        cascade_depth_from_inputs; None == unreachable)
      res_slice_start   model.res_slice.start (readout column 0 == this abs node)

    Returns a NEW list (does not mutate geoms), one dict per output, adding:
      top_depth    cascade depth of the top_node (None if unused/unreachable)
      depth_lbl    'I'  (the top node is actually an input node)
                   'shallow' (depth 1 -> directly input-shadowing)
                   'deep'    (depth >= 2 -> needs recurrence)
      verdict      fused hypothesis label (see below)
      verdict_int  coarse numeric: +1 = computed-like, 0 = uncertain/partial,
                   -1 = hash-like (signed for easy colouring/filtering)
    """
    out = []
    for g in geoms:
        entry = dict(g)  # copy so caller's geoms stays intact
        if g['top_node'] is None:
            entry['top_depth'] = None
            entry['depth_lbl'] = 'unused'
            entry['verdict'] = 'unused output (all-zero readout)'
            entry['verdict_int'] = 0
            out.append(entry)
            continue

        abs_node = res_slice_start + g['top_node']
        dep = depth_by_abs_node.get(abs_node)
        if dep is None:
            entry['top_depth'] = None
            entry['depth_lbl'] = 'unreachable'
        elif dep == 0:
            entry['top_depth'] = 0
            entry['depth_lbl'] = 'I'          # top node IS an input node
        elif dep == 1:
            entry['top_depth'] = 1
            entry['depth_lbl'] = 'shallow'
        else:
            entry['top_depth'] = dep
            entry['depth_lbl'] = 'deep'

        # Fused hypothesis. Necessary-not-sufficient throughout.
        concentrated = g['single_node'] or (g['top_share'] >= 0.5)
        spread = (g['support'] is not None and g['support'] is not False
                  and g['support'] > 1 and g['pr'] >= 2)
        if concentrated and entry['depth_lbl'] in ('I', 'shallow'):
            verdict = ('reads a SHALLOW node hard - suspect input hash / '
                       'read-through, not reservoir computation')
            verdict_int = -1
        elif spread:
            verdict = ('spread over many nodes - readout itself is doing the '
                       'work - suspect input hash / overfit')
            verdict_int = -1
        elif concentrated and entry['depth_lbl'] == 'deep':
            verdict = ('reads a DEEP node hard - consistent with genuine '
                       'reservoir computation (still necessary, not sufficient)')
            verdict_int = +1
        elif concentrated:
            verdict = ('concentrated but depth unknown - geometry hints at a '
                       'pointer readout; depth would disambiguate')
            verdict_int = 0
        else:
            verdict = ('partial support on a shallow-ish node - weak/degenerate '
                       'readout; interpret with care')
            verdict_int = 0
        entry['verdict'] = verdict
        entry['verdict_int'] = verdict_int
        out.append(entry)
    return out


def global_concentration(readout_W):
    """Single-scalar Gini of all |W|: 0 = uniform, 1 = maximally spiky."""
    a = np.abs(np.asarray(readout_W, dtype=float)).ravel()
    if a.size == 0 or a.sum() == 0:
        return 0.0
    s = np.sort(a)
    n = a.size
    return float((2 * np.dot(np.arange(1, n + 1), s) - (n + 1) * s.sum())
                 / (n * s.sum()))


def cascade_depth_from_inputs(graph, input_nodes_mask):
    """Directed BFS distance of each node from the input layer (multi-source).

    Let directed edge u -> v mean u influences v downstream (the state of v at
    time t is a function of u's state at t-1; u is a predecessor of v).
    The cascade depth of a node is the shortest directed path from ANY input
    node to it. An input node is depth 0.

    Structural interpretability: a reservoir node at depth d cannot be influenced
    by an input in fewer than d reservoir steps, so d=1 nodes are directly
    input-shadowing -- a large readout there is consistent with an input hash,
    whereas a large readout on a deep node (d > 1) requires real recurrence.
    This is NECESSARY evidence for 'reservoir did the work', not sufficient: a
    deep subnetwork can still be a wired-through hash.

    Uses only the stored graph + input mask: no reservoir states, no Julia.

    Returns dict node -> int depth (unreachable nodes get None).
    """
    out_map = {n: [v for _, v in graph.out_edges(n)] for n in graph.nodes()}
    depth = {n: None for n in graph.nodes()}
    queue = []
    for n in graph.nodes():
        if input_nodes_mask[n]:
            depth[n] = 0
            queue.append(n)
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        d = depth[cur]
        for nxt in out_map[cur]:
            if depth[nxt] is None:
                depth[nxt] = d + 1
                queue.append(nxt)
    return depth