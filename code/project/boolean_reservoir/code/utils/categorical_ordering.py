import pandas as pd
import numpy as np
import itertools
import warnings

def grayish_sort(df: pd.DataFrame, factors: list, ascending: list = None):
    """
    Stable mixed-radix Gray-code–like sort over factors.
    Works on a copy of df: converts columns to ordered categoricals internally if needed.
    Returns the original df sorted by Grayish order.

    ascending: per-factor bool list (default all True). False reverses that factor's
               category order, flipping the direction of the polar axis for that factor.
    """
    if df.empty or not factors:
        return df

    asc_flags = ascending if isinstance(ascending, list) else [True] * len(factors)

    # Work on a copy
    df_copy = df[factors].copy()

    # Convert each factor to ordered categorical (internally)
    schema = {}
    for i, f in enumerate(factors):
        if pd.api.types.is_categorical_dtype(df_copy[f]) and df_copy[f].cat.ordered:
            cats = list(df_copy[f].cat.categories)
        else:
            cats = [c for c in pd.unique(df_copy[f]) if pd.notna(c)]
            # Sort appropriately based on dtype
            try:
                cats = sorted(cats, key=lambda x: float(x))
            except (ValueError, TypeError):
                cats = sorted(cats, key=lambda x: str(x))
        if not asc_flags[i]:
            cats = cats[::-1]
        
        df_copy[f] = pd.Categorical(df_copy[f], categories=cats, ordered=True)
        schema[f] = cats
    
    # Build categories and radix sizes
    categories = [schema[f] for f in factors]
    radices = [len(c) for c in categories]

    # Guard: if the Cartesian product is huge, skip gray ordering and use plain sort.
    product = 1
    for r in radices:
        product *= r
        if product > 10_000:
            warnings.warn(
                f'grayish_sort: Cartesian product ({product:,}) exceeds 10,000 — '
                f'falling back to plain sort. Reduce the number of design factors.',
                stacklevel=2,
            )
            return df.sort_values(factors, ascending=True).reset_index(drop=True)

    # Generate Gray order in index space
    gray_indices = list(mixed_radix_gray_gen(radices))
    if not gray_indices:
        return df.sort_values(factors, ascending=asc_flags).reset_index(drop=True)

    # Convert index tuples → label tuples
    gray_labels = [
        tuple(categories[i][idx[i]] for i in range(len(factors)))
        for idx in gray_indices
    ]
    
    # Vectorised lookup via mixed-radix integer coding.
    # Combine per-factor Categorical codes into one integer per row using strides,
    # then map to gray order with numpy fancy indexing — no Python-level row loop.
    strides = np.ones(len(factors), dtype=np.int64)
    for i in range(len(factors) - 2, -1, -1):
        strides[i] = strides[i + 1] * radices[i + 1]

    cat_codes   = np.column_stack([df_copy[f].cat.codes.values for f in factors])
    row_codes   = cat_codes.dot(strides)                    # (n_rows,)

    gray_arr    = np.array(gray_indices, dtype=np.int64).reshape(-1, len(factors))  # (n_combos, n_factors)
    combo_codes = gray_arr.dot(strides)                     # (n_combos,)

    code_to_gray = np.empty(int(combo_codes.max()) + 1, dtype=np.int64)
    code_to_gray[combo_codes] = np.arange(len(gray_labels), dtype=np.int64)

    df_copy['_gray_order'] = code_to_gray[row_codes]
    
    # Sort original df using the Gray order from the copy
    sorted_df = df.iloc[df_copy['_gray_order'].argsort(kind='stable')]
    
    return sorted_df

def mixed_radix_gray_gen(levels):
    if not levels:
        yield []
        return
    first, rest = levels[0], levels[1:]
    tail = list(mixed_radix_gray_gen(rest))
    for i in range(first):
        seq = tail if i % 2 == 0 else reversed(tail)
        for code in seq:
            yield [i] + code


# ---------------- Example ----------------
if __name__ == "__main__":
    A = ['low', 'medium', 'high']
    B = ['blue', 'red']
    C = [0, 1]
    D = ['small', 'medium', 'large']

    FACTORS = ['A', 'B', 'C', 'D']

    data = list(itertools.product(A, B, C, D))
    df = pd.DataFrame(data + data[:3], columns=FACTORS)

    # Intentionally leave all columns as object/string or int
    df_sorted = grayish_sort(df, FACTORS)
    print(df_sorted)
