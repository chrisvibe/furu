from dataclasses import dataclass

import numpy as np
from project.boolean_reservoir.code.utils.utils import make_rng


@dataclass(frozen=True, eq=False)
class Lut:
    """Every node's truth table, back to back: node n's is flat[offsets[n]:offsets[n+1]],
    2**k_n entries for its k_n inputs, indexed MSB-first over them (lut_index). This is
    the form the engine runs on; there is no padded per-node row."""
    flat: np.ndarray     # uint8
    offsets: np.ndarray  # int64, n_nodes + 1

    def row(self, n: int) -> np.ndarray:
        return self.flat[self.offsets[n]:self.offsets[n + 1]]

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __eq__(self, other) -> bool:
        return (isinstance(other, Lut) and np.array_equal(self.flat, other.flat)
                and np.array_equal(self.offsets, other.offsets))

    @staticmethod
    def from_rows(rows: np.ndarray, mask: np.ndarray) -> 'Lut':
        """From per-node rows (a CA rule's tables, or an older checkpoint's padded
        lut.npy): node n keeps rows[n, :2**k_n], k_n its number of inputs in mask."""
        sizes = np.int64(1) << np.asarray(mask, dtype=np.bool_).sum(axis=1)
        offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(sizes)
        rows = np.asarray(rows, dtype=np.uint8)
        flat = np.concatenate([rows[n, :s] for n, s in enumerate(sizes.tolist())]) if len(sizes) else np.zeros(0, np.uint8)
        return Lut(flat, offsets)

    @staticmethod
    def concat(parts: list['Lut']) -> 'Lut':
        flat = np.concatenate([q.flat for q in parts])
        sizes = np.concatenate([np.diff(q.offsets) for q in parts])
        offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(sizes)
        return Lut(flat, offsets)

    @staticmethod
    def coerce(obj, mask: np.ndarray) -> 'Lut':
        """A LUT as stored or handed over: a Lut, {'flat', 'offsets'} (lut.npz), or padded rows."""
        if isinstance(obj, Lut):
            return obj
        if isinstance(obj, dict):
            return Lut(np.asarray(obj['flat'], dtype=np.uint8), np.asarray(obj['offsets'], dtype=np.int64))
        return Lut.from_rows(obj, mask)


def calc_lut_p(lut: Lut) -> float:
    """Realized fraction of 1-entries over the entries the nodes read."""
    return float(lut.flat.mean())


def lut_sensitivity(lut: Lut) -> np.ndarray:
    """Per node and input position (MSB-first, as lut_index): the fraction of the node's
    input patterns for which flipping that one input flips the output. (n_nodes, max_k),
    NaN past a node's own degree. A row's sum is the node's (average) sensitivity, the
    quantity whose network mean is 1 at the order/chaos boundary of random Boolean
    networks (Shmulevich & Kauffman 2004): 0 for a constant, 1 for a copy, k for parity."""
    degrees = np.round(np.log2(np.diff(lut.offsets))).astype(np.int64)
    max_k = int(degrees.max()) if len(degrees) else 0
    out = np.full((len(lut), max_k), np.nan)
    for k in np.unique(degrees[degrees > 0]):
        nodes = np.where(degrees == k)[0]
        T = lut.flat[lut.offsets[nodes][:, None] + np.arange(2 ** k)]       # (nodes, 2**k)
        x = np.arange(2 ** k)
        for pos in range(k):                                                 # MSB-first position
            out[nodes, pos] = (T != T[:, x ^ (1 << (k - 1 - pos))]).mean(axis=1)
    return out


def lut_random(degrees: np.ndarray, p=0.5, rng=None, chunk_doubles=1 << 20) -> Lut:
    """One random truth table per node, node i's of size 2**degrees[i].

    Stream layout: entry (i, c) — node i, input pattern c — is the double at position
    c * n + i of rng's stream (a column-major n x 2**max_k table). So an entry never
    depends on any node's degree: a wider table (one more input, e.g. a query edge as
    the new MSB) keeps the narrower one as its first half, and the same seed gives the
    same tables whatever else the graph holds. Callers draw each layer from its own
    stream (BooleanReservoir.draw_lut), so the tables do not depend on the input width
    either. numpy RNG, so identical across CPU and GPU.

    Only the entries read are drawn. Column c is read by the nodes with 2**k > c;
    walking the columns in order, the stream is drawn over the span of those nodes and
    skipped elsewhere (bit_generator.advance: one step per double), so the cost is the
    entries read, not n * 2**max_k."""
    assert 0 <= p <= 1
    if rng is None:
        rng = make_rng()
    degrees = np.asarray(degrees, dtype=np.int64)
    n = len(degrees)
    sizes = np.int64(1) << degrees
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    flat = np.empty(int(offsets[-1]), dtype=np.uint8)
    pos = 0  # stream position drawn up to

    def skip_to(target):
        nonlocal pos
        if target > pos:
            rng.bit_generator.advance(target - pos)
            pos = target

    # Columns in bands [lo, hi) between consecutive table sizes: within a band the
    # set of nodes reading each column is the same.
    edges = np.unique(np.concatenate([[0], sizes]))
    for lo, hi in zip(edges[:-1].tolist(), edges[1:].tolist()):
        readers = np.nonzero(sizes >= hi)[0]
        a0, a1 = int(readers[0]), int(readers[-1]) + 1
        if (a1 - a0) * 4 >= n:   # most of each column is read: draw whole columns
            step = max(1, chunk_doubles // n)
            for c0 in range(lo, hi, step):
                c1 = min(hi, c0 + step)
                skip_to(c0 * n)
                block = rng.random((c1 - c0, n))
                pos = c1 * n
                flat[offsets[readers][None, :] + np.arange(c0, c1)[:, None]] = block[:, readers] < p
        else:                    # a few wide nodes: draw just their span of each column
            span, at, sel = a1 - a0, offsets[readers], readers - a0
            advance, draw = rng.bit_generator.advance, rng.random
            step = max(1, chunk_doubles // span)
            buf = np.empty((min(step, hi - lo), span))
            for c0 in range(lo, hi, step):
                c1 = min(hi, c0 + step)
                for j in range(c1 - c0):  # the hot loop: one skip + one draw per column
                    skip_to((c0 + j) * n + a0) if j == 0 else advance(n - span)
                    draw(out=buf[j])
                pos = (c1 - 1) * n + a1
                flat[at[None, :] + np.arange(c0, c1)[:, None]] = buf[:c1 - c0, sel] < p
    return Lut(flat, offsets)


def lut_index(pred_states: list) -> int:
    """MSB-first binary index from predecessor states: the first predecessor is the
    most-significant bit (2^(k-1)), matching the Julia engine's per-node shifts."""
    k = len(pred_states)
    return sum(int(pred_states[i]) << (k - 1 - i) for i in range(k))


def lut_lookup(lut: Lut, node: int, pred_states: list) -> int:
    """Return the LUT output for a node given its predecessor states."""
    return int(lut.row(node)[lut_index(pred_states)])


# TODO add alternative with exactly pxN entries like the graphs atm which generate E edges then shuffle them, not like flipping a coin for each edge


# TODO proposed new LUT? tune approximability by Taylor expansion around p=0.5, then sample from the distribution of LUTs with that approximability (ties into ideas approximable LUT in biological systems: https://www.nature.com/articles/s41540-024-00392-y)

# def lut_taylor_approximability(
#     lut: np.ndarray,
#     max_incoming_edges: int,
#     order: int = 1,
# ) -> np.ndarray:
#     """
#     Measure how well each Boolean LUT is approximated by a
#     Taylor expansion of the continuous extension around p=0.5.

#     Parameters
#     ----------
#     lut:
#         Shape (n_luts, 2**k), or (2**k,) for one LUT.
#         MSB-first indexing, matching lut_index().

#     max_incoming_edges:
#         k, the number of inputs.

#     order:
#         Taylor approximation order.

#     Returns
#     -------
#     error:
#         Mean absolute error of the truncated Taylor approximation
#         evaluated on all Boolean input states.

#         Lower error = more approximable.
#     """
#     lut = np.asarray(lut, dtype=np.float64)

#     if lut.ndim == 1:
#         lut = lut[None, :]

#     n_luts, n_states = lut.shape
#     k = max_incoming_edges

#     assert n_states == 2 ** k
#     assert 0 <= order <= k

#     # All Boolean input states, MSB -> LSB.
#     states = (
#         (np.arange(n_states)[:, None]
#          >> np.arange(k - 1, -1, -1))
#         & 1
#     )

#     # Coordinates relative to Taylor expansion point p=0.5.
#     x = states - 0.5

#     # Build the centered polynomial basis.
#     #
#     # Each column corresponds to one subset of variables:
#     #
#     #   1
#     #   (x1)
#     #   (x2)
#     #   (x1*x2)
#     #   ...
#     #
#     # The full basis exactly represents every Boolean function.
#     subsets = []

#     for mask in range(2 ** k):
#         variables = [
#             i for i in range(k)
#             if (mask >> (k - 1 - i)) & 1
#         ]

#         if len(variables) <= order:
#             subsets.append(variables)

#     basis = np.empty((n_states, len(subsets)), dtype=np.float64)

#     for j, variables in enumerate(subsets):
#         if variables:
#             basis[:, j] = np.prod(x[:, variables], axis=1)
#         else:
#             basis[:, j] = 1.0

#     # We need the coefficients of the polynomial.
#     #
#     # Since the full centered polynomial basis is exact, obtain
#     # all coefficients using the complete basis first.
#     full_subsets = []

#     for mask in range(2 ** k):
#         variables = [
#             i for i in range(k)
#             if (mask >> (k - 1 - i)) & 1
#         ]
#         full_subsets.append(variables)

#     full_basis = np.empty(
#         (n_states, 2 ** k),
#         dtype=np.float64,
#     )

#     for j, variables in enumerate(full_subsets):
#         if variables:
#             full_basis[:, j] = np.prod(
#                 x[:, variables],
#                 axis=1,
#             )
#         else:
#             full_basis[:, j] = 1.0

#     # Because this matrix depends only on k, invert it once.
#     transform = np.linalg.inv(full_basis)

#     # coefficients:
#     #
#     #       full_basis @ coefficients = LUT
#     #
#     # shape: (2**k, n_luts)
#     coefficients = transform @ lut.T

#     # Keep only terms <= requested order.
#     keep = np.array([
#         len(variables) <= order
#         for variables in full_subsets
#     ])

#     coefficients[~keep, :] = 0.0

#     # Reconstruct approximate outputs on the Boolean states.
#     approx = full_basis @ coefficients

#     # Mean absolute error for each LUT.
#     error = np.mean(
#         np.abs(approx - lut.T),
#         axis=0,
#     )

#     return error


# def make_lut_library(
#     n_luts: int,
#     max_incoming_edges: int,
#     approximability: float,
#     order: int = 1,
#     rng=None,
# ):
#     """
#     Generate LUTs with controlled Taylor approximability.

#     approximability:
#         0.0 -> least approximable part of random LUT distribution
#         1.0 -> most approximable part of random LUT distribution

#     The LUTs are sampled from a random p=0.5 population and then
#     selected according to their measured Taylor approximation error.

#     Returns
#     -------
#     lut:
#         Shape (n_luts, 2**k), dtype uint8.
#     """
#     assert 0.0 <= approximability <= 1.0
#     assert n_luts > 0

#     if rng is None:
#         rng = np.random.default_rng()

#     k = max_incoming_edges
#     n_states = 2 ** k

#     # Generate a candidate population substantially larger than needed.
#     # This gives us a reasonably fine approximationability distribution.
#     n_candidates = max(
#         10 * n_luts,
#         10_000,
#     )

#     candidates = (
#         rng.random((n_candidates, n_states)) < 0.5
#     ).astype(np.uint8)

#     # Measure Taylor approximation error.
#     errors = lut_taylor_approximability(
#         candidates,
#         k,
#         order=order,
#     )

#     # Low error = highly approximable.
#     order_indices = np.argsort(errors)

#     # Map approximability [0,1] onto the sorted population.
#     #
#     # 0 -> least approximable
#     # 1 -> most approximable
#     center = int(
#         approximability * (n_candidates - 1)
#     )

#     # Rather than selecting exactly one LUT, sample from a local
#     # window around the requested quantile.
#     window = max(
#         n_luts * 2,
#         100,
#     )

#     lo = max(0, center - window // 2)
#     hi = min(n_candidates, center + window // 2)

#     pool = order_indices[lo:hi]

#     selected = rng.choice(
#         pool,
#         size=n_luts,
#         replace=len(pool) < n_luts,
#     )

#     return candidates[selected]