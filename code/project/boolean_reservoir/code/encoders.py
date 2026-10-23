"""Value → bit-pattern encoders: the single source of truth for how a scalar
value in [0,1] becomes a pattern of bits across `bits` ticks.

This module is deliberately free of the config layer (parameter.py): every
kernel is a pure function of (values, bits[, rng]), so encoders can be tested,
benchmarked and reused without building a Params tree. encoding.py layers the
pipeline on top — BooleanEncoder dispatches a config's `encoding` name through
ENCODERS here, BooleanTransformer (there) applies redundancy / interleaving /
the binary_embedding key-XOR afterwards.

Adding an encoder:

    @register('my_code')
    def dec2my_code(x, bits):
        '''One-line summary (shows up in the dispatch error message).'''
        ...
    # then add 'my_code' to EncodingName in parameter.py — the static Literal
    # is pinned to this registry by test_encoders.py, so the two cannot drift.

Contract for every kernel:
    values : float ndarray, values in [0,1]  (validated by validate_analog)
    bits   : int resolution — number of bit ticks emitted per value
    returns: bool ndarray, shape values.shape + (bits,)
Deterministic kernels are pure; stochastic kernels take `rng` (a numpy
Generator) and must map a given (values, rng state) to the same pattern every
time — downstream code depends on that reproducibility (see dec2rate).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Callable, Dict, Mapping

import numpy as np

from project.boolean_reservoir.code.utils.primes import PRIMES
from project.boolean_reservoir.code.utils.utils import make_rng

__all__ = [
    'ENCODERS', 'ENCODING_NAMES', 'EncoderSpec', 'BinaryEmbedding', 'bin2dec',
    'dec2base2_delta_sigma', 'dec2bin', 'dec2base2_lsb', 'dec2delta_sigma',
    'dec2delta_sigma_free', 'dec2gray', 'dec2hamming_weight', 'dec2phase',
    'dec2primes', 'dec2rate', 'dec2tally', 'get_encoder',
    'hamming_weight2dec', 'load_primes', 'quantize', 'unpack_bits',
    'validate_analog',
]


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------

def validate_analog(values):
    """The encoder input contract: floats in [0,1] (normalized values).

    Boolean/integer inputs bypass this layer entirely — feed them to
    BooleanTransformer directly.
    """
    assert np.issubdtype(values.dtype, np.floating), (
        f"BooleanEncoder expects float input (normalized [0,1]). "
        f"Got dtype={values.dtype}. For boolean/integer inputs use BooleanTransformer directly.")
    assert values.max() <= 1, f"values must be normalized to [0,1], got max={values.max()}"
    assert values.min() >= 0, f"values must be normalized to [0,1], got min={values.min()}"
    return values


# ---------------------------------------------------------------------------
# Shared bit arithmetic (the DRY core of the positional family)
# ---------------------------------------------------------------------------

def quantize(values, bits):
    """Map [0,1] → integer codes 0 .. 2**bits - 1 (floor; endpoints included)."""
    return (values * (2**bits - 1)).astype(np.int64)


def _bit_shifts(bits, lsb_first):
    shift = np.arange(bits, dtype=np.int64)
    return shift if lsb_first else shift[::-1]


def unpack_bits(codes, bits, lsb_first=False):
    """Integer codes → bool bit plane (..., bits); bit 0 is the MSB unless lsb_first."""
    return (codes[..., None] & (1 << _bit_shifts(bits, lsb_first))) != 0


def _sigma_tick(integrator, values):
    """One first-order delta-sigma step: accumulate, threshold, subtract the emit."""
    integrator = integrator + values
    bit = integrator >= 0.5
    return bit, integrator - bit.astype(np.float32)


# ---------------------------------------------------------------------------
# Dense codebooks — a code space has only 2**bits distinct words, so lookup
# codes gather precomputed rows instead of re-deriving the pattern per value.
# ---------------------------------------------------------------------------

_DENSE_MAX_BITS = 16   # dense table = 2**bits x bits bytes; 16 -> 1 MiB
_DENSE_CODEBOOKS: Dict[tuple, np.ndarray] = {}


@lru_cache(maxsize=None)
def _revdoor_cycle(n, k):
    """All C(n,k) k-subsets of {0..n-1} as bitmasks, in revolving-door order.

    A constant-weight Gray code: consecutive masks differ by exactly one
    element swap (one bit in, one out -> distance 2), and the last is adjacent
    to the first the same way, so the listing is a Hamiltonian cycle on the
    Johnson graph J(n,k) and can be cut open at any of its words. Classical
    construction (Nijenhuis-Wilf's revolving door): R(n,k) = R(n-1,k) followed
    by R(n-1,k-1) reversed, each word opened by element n-1 — both seams and
    the wrap edge are single swaps (first word is always {0..k-1}, last is
    {0..k-2, n-1}). Exists for every 1 <= k <= n-1, so this is construction,
    not search.
    """
    if k == 0:
        return (0,)
    if k == n:
        return ((1 << n) - 1,)
    high = 1 << (n - 1)
    return _revdoor_cycle(n - 1, k) + tuple(
        m | high for m in reversed(_revdoor_cycle(n - 1, k - 1)))


def _weight_gray_order(bits):
    """The hamming_weight enumeration as a list of 2**bits bitmasks.

    Weight-primary, unchanged: the count-k bucket occupies rows
    sum(C(bits, j) for j < k) .. that plus C(bits,k). Within each bucket the
    revolving-door cycle is cut open at the bucket's start word, so consecutive
    rows differ by one element swap (distance 2 — same weight can never give 1),
    and each next bucket's first word is the previous bucket's last word plus
    its lowest absent element (boundary distance 1). The whole enumeration is
    therefore a permutation of the 2**bits words whose adjacent-value profile
    never exceeds distance 2, with exactly `bits` distance-1 steps.
    """
    full = (1 << bits) - 1
    order = []
    start = 0
    for k in range(bits + 1):
        cycle = _revdoor_cycle(bits, k)
        i = cycle.index(start)                        # cut point = bucket's open
        order.extend(cycle[i:] + cycle[:i])
        free = ~order[-1] & full
        start = order[-1] | (free & -free)            # open the next bucket
    return order


def _dense_codebook(bits, order):
    """All 2**bits words as a dense (2**bits, bits) bool table, cached read-only.

    order='identity'    → row q is the binary expansion of q (base2's table)
    order='weight_gray' → rows ordered by (Hamming weight, constant-weight
                          Gray within each count bucket) — the hamming_weight
                          codebook: adjacent rows never differ by more than
                          2 bits, bucket boundaries by exactly 1.
    Dense memory is 2**bits x bits bytes, hence the cap: base2 falls back to
    per-call unpacking above it, while hamming_weight's code IS the ordering
    and refuses instead.
    """
    if not 1 <= bits <= _DENSE_MAX_BITS:
        raise ValueError(
            f"dense codebook needs 1 <= bits <= {_DENSE_MAX_BITS} "
            f"(2**bits x bits table), got {bits}")
    if order not in ('identity', 'weight_gray'):
        raise ValueError(f"unknown codebook order {order!r}")
    key = (bits, order)
    table = _DENSE_CODEBOOKS.get(key)
    if table is None:
        if order == 'identity':
            table = unpack_bits(np.arange(1 << bits, dtype=np.int64), bits)
        else:
            masks = np.array(_weight_gray_order(bits), dtype=np.int64)
            table = unpack_bits(masks, bits)
        table.flags.writeable = False
        _DENSE_CODEBOOKS[key] = table
    return table


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncoderSpec:
    """A registered kernel plus what the dispatcher needs to know about it."""
    fn: Callable[..., np.ndarray]   # (values, bits[, rng]) -> bool (..., bits)
    stochastic: bool = False        # kernel takes `rng`; seeded via make_rng(I.seed, 'rate')


_ENCODERS: Dict[str, EncoderSpec] = {}


def register(name: str, *, stochastic: bool = False):
    """Attach a kernel to the encoding registry under its config name."""
    def wrap(fn):
        if name in _ENCODERS:
            raise ValueError(f"encoder {name!r} registered twice")
        _ENCODERS[name] = EncoderSpec(fn=fn, stochastic=stochastic)
        return fn
    return wrap


def get_encoder(name: str) -> EncoderSpec:
    """Look up a registered encoder by its config name."""
    try:
        return _ENCODERS[name]
    except KeyError:
        raise ValueError(f"encoding {name!r} is not an option! Valid: {list(_ENCODERS)}") from None


# ---------------------------------------------------------------------------
# Kernels — registration order matches the historical _ENCODERS dict so the
# dispatch error message lists encodings in the same order as before.
# ---------------------------------------------------------------------------

@register('base2')
def dec2bin(values, bits):
    """Positional binary (base-2), MSB first: bit k carries weight 2^(bits-1-k).

    The code space has only 2**bits distinct words, so below the dense-table
    cap the kernel gathers precomputed rows instead of re-deriving the bit
    pattern per value (~4x faster, bit-identical); above it, the arithmetic
    path applies. Codes reduce to their low `bits` (legacy semantics for
    direct out-of-range calls); the pipeline validates [0,1] first.
    """
    codes = quantize(values, bits) & ((1 << bits) - 1)
    if bits <= _DENSE_MAX_BITS:
        return np.take(_dense_codebook(bits, 'identity'), codes, axis=0)
    return unpack_bits(codes, bits)


@register('base2_lsb')
def dec2base2_lsb(values, bits):
    # Same values as dec2bin, presented LSB-first: bit index 0 is the LEAST
    # significant bit, so as ticks it is digested FIRST and the MSB arrives
    # LAST (freshest in the reservoir's fading memory). Pure presentation-order
    # probe of the MSB-recency question — positional weights are unchanged.
    return unpack_bits(quantize(values, bits), bits, lsb_first=True)


@register('primes')
def dec2primes(values, bits):
    """Mixed-radix greedy code over descending primes closed with 1.

    Weights [p_{bits-1}, ..., p_1, 1] sum-cover every integer, and small value
    increments touch the LOW weights — the ticks digested last — making this
    the mixed-radix dual of base2_lsb's recency argument.
    """
    weights, max_val = _prime_table(bits)
    values = (values * max_val).round().astype(np.int64)
    result = np.zeros(values.shape + (bits,), dtype=bool)
    for i in range(bits):
        result[..., i] = values >= weights[i]
        values = values - result[..., i].astype(np.int64) * weights[i]
    return result


@lru_cache(maxsize=None)
def _prime_table(bits):
    """(weights, max_val) per bits — the table is read-only, safe to share."""
    weights = np.array(load_primes(bits - 1)[::-1] + [1], dtype=np.int64)
    return weights, int(weights.sum())


def load_primes(n):
    if n > 1000:
        raise ValueError(f"Cannot load more than 1000 primes, got {n}")
    return PRIMES[:n]


@register('tally')
def dec2tally(values, bits):
    """Thermometer / unary code: exactly round(bits * value) leading ticks are 1."""
    d = (bits * values).round()[..., None]
    bit_range = np.arange(bits, dtype=np.float32)
    return bit_range < d


# 'binary_embedding' consumes base2 bits — the key-XOR expansion happens in
# BooleanTransformer via BinaryEmbedding below, so the registered kernel is
# the plain positional code under a second name (same function object).
# Registered HERE, fifth, because the historical _ENCODERS dict — and the
# EncodingName Literal pinned to it — list it between tally and gray.
register('binary_embedding')(dec2bin)


@register('gray')
def dec2gray(values, bits):
    """Reflected binary (Gray) code, MSB first: adjacent values flip exactly one tick."""
    codes = quantize(values, bits)
    return unpack_bits(codes ^ (codes >> 1), bits)


# dec2rate is stochastic by construction: each bit is a Bernoulli draw with
# p = the encoded value. It is driven by an explicit generator supplied per
# call from I.seed (BooleanEncoder._encode), so a given input encodes to the
# SAME pattern every time rather than a fresh one per call. That removes the
# irreducible readout noise this note used to warn about: with n_samples=1 and
# single-shot ridge, inconsistent representations of identical inputs were pure
# loss. Rate coding still represents a value only in expectation over `bits`,
# so it remains a coarse code — but it is reproducible.
@register('rate', stochastic=True)
def dec2rate(values, bits, rng=None):
    """Bernoulli rate code: bit k is 1 with probability values (expectation code)."""
    if rng is None:
        rng = make_rng()
    return rng.random(values.shape + (bits,)) < values[..., None]


@register('delta_sigma')
def dec2delta_sigma(values, bits):
    """First-order delta-sigma (sum code): the 1s-count tracks the value; the
    integrator resets per call, so each call encodes its own values in isolation."""
    result = np.zeros(values.shape + (bits,), dtype=bool)
    integrator = np.zeros_like(values, dtype=np.float32)
    for i in range(bits):
        bit, integrator = _sigma_tick(integrator, values)
        result[..., i] = bit
    return result


@register('delta_sigma_free')
def dec2delta_sigma_free(values, bits):
    # Free-running (never-resetting) first-order delta-sigma: unlike
    # dec2delta_sigma the integrator is NOT reset per step, so the 1s-count
    # over the whole stream tracks the running sum of x — the code itself
    # performs the integration the readout would otherwise have to learn.
    # x is (m, s, f) zero-padded beyond each sample's true length; padded
    # steps contribute nothing to the integrator (x=0 drains it, emitting no
    # new ones once below 0.5 — wait, it emits ones while >= 0.5, draining
    # leftovers; those padded ticks are masked downstream, so harmless).
    assert values.ndim == 3, f"dec2delta_sigma_free expects (m, s, f), got {values.shape}"
    m, s, f = values.shape
    result = np.zeros((m, s, f, bits), dtype=bool)
    integ = np.zeros((m, f), dtype=np.float32)
    for si in range(s):
        for k in range(bits):
            bit, integ = _sigma_tick(integ, values[:, si])
            result[:, si, :, k] = bit
    return result


@register('base2_delta_sigma')
def dec2base2_delta_sigma(values, bits, head=4):
    # Hybrid: top `head` bits positional (base2, MSB-first) + delta_sigma on
    # the 2^-head residual. Positional precision where it is cheap, sum-code
    # decodability for the fine part. Encoder error ~ (1/2^head)/2 + 1/(2*(bits-head))
    # scaled by 2^-head residual: ~0.031 + 0.031*0.0625 ~= 0.033 at head=4, bits=10,
    # vs pure delta_sigma ~0.05 and base2 ~0.0005.
    # NOTE the coarse grid is floor(x * 2**head) over 2**head levels — NOT
    # quantize(x, head), which spreads over 2**head - 1 codes; kept distinct
    # for bit-parity with the historical encoder.
    q = 1 << head
    coarse = np.floor(values * q).astype(np.int64).clip(0, q - 1)
    resid = (values * q).astype(np.float32) - coarse
    hi = unpack_bits(coarse, head)
    lo = dec2delta_sigma(resid.astype(np.float32), bits - head)
    return np.concatenate([hi, lo], axis=-1)


@register('phase')
def dec2phase(values, bits):
    # Timing/phase code (theta phase-precession analog): constant 50% duty
    # square wave, the value is carried by the PHASE SHIFT of the pattern
    # across ticks, not by the count of ones. Information in onset timing.
    k = np.arange(bits, dtype=np.float32)
    return (((k / bits) + values[..., None]) % 1.0) < 0.5


# ---------------------------------------------------------------------------
# Hamming-weight-ordered codebook code — the full 2**bits state space with the
# ones-count monotone in the value: a linear readout reads the coarse value
# off the count, the fine value off within-bucket position. The within-bucket
# order is a constant-weight Gray (revolving door), capping adjacent-value
# transitions at 2 flips. New encoders are appended to the registry after the
# historical ones.
# ---------------------------------------------------------------------------

_HAMMING_RANKS: Dict[int, np.ndarray] = {}


def _hamming_tables(bits):
    """(codebook, rank) for the hamming_weight code.

    codebook comes from the shared dense cache in (Hamming weight,
    constant-weight Gray) order; rank[v] is the rank of the word with integer
    value v — the exact inverse permutation, derived from the codebook itself,
    so it follows whatever within-bucket order the codebook carries. Read-only.
    """
    codebook = _dense_codebook(bits, 'weight_gray')
    rank = _HAMMING_RANKS.get(bits)
    if rank is None:
        ints = (codebook * (1 << _bit_shifts(bits, lsb_first=False))).sum(axis=-1)
        rank = np.empty(1 << bits, dtype=np.int64)
        rank[ints] = np.arange(1 << bits)
        rank.flags.writeable = False
        _HAMMING_RANKS[bits] = rank
    return codebook, rank


@register('hamming_weight')
def dec2hamming_weight(values, bits):
    """All 2**bits words ordered by (Hamming weight, constant-weight Gray).

    encode = table lookup at rank quantize(x, bits) — the cheapest kernel in
    the registry. The ones-count is monotone in x, so the count channel
    carries a coarse value while the count-k bucket (C(bits,k) words) refines
    it: full 2**bits expressiveness with the count alignment of a sum code.
    Within a bucket the order is revolving-door (2-change): consecutive words
    differ by one element swap, and each bucket opens by extending the
    previous bucket's last word — so adjacent values never move more than
    2 ticks, and exactly `bits` steps move one. The price is binomial bucket
    widths — the count window is ~0.25 wide near x = 1/2 and ~0.001 at the
    extremes (bits=10), the mirror image of a uniform quantizer — and the
    within-bucket position pattern no longer tracks the residual magnitude
    (the Gray trade, confined per bucket).
    """
    codebook, _ = _hamming_tables(bits)
    return np.take(codebook, quantize(values, bits), axis=0)


def hamming_weight2dec(words, bits):
    """Exact decode: word → its rank in the Hamming-weight order → [0,1]."""
    _, rank = _hamming_tables(bits)
    ints = (words * (1 << _bit_shifts(bits, lsb_first=False))).sum(axis=-1)
    return rank[ints] / (2**bits - 1)


# ---------------------------------------------------------------------------
# Public registry views — created after all registrations above.
# ---------------------------------------------------------------------------

ENCODERS: Mapping[str, EncoderSpec] = MappingProxyType(_ENCODERS)
ENCODING_NAMES: tuple = tuple(_ENCODERS)


# ---------------------------------------------------------------------------
# Decode side — the inverse of the positional codes (normalized to [0,1]).
# ---------------------------------------------------------------------------

def bin2dec(x, bits, small_endian=False):
    """Bit plane (..., bits) → values in [0,1]; MSB first unless small_endian."""
    weights = 1 << _bit_shifts(bits, small_endian)
    return (weights * x).sum(axis=-1) / (2**bits - 1)


# ---------------------------------------------------------------------------
# Key-XOR encoder — expands base2 bits through random boolean keys.
# ---------------------------------------------------------------------------

class BinaryEmbedding:
    """Random-key XOR encoder: each of n key banks XORs the b base2 bits.

    Deterministic given the rng: identical inputs encode to identical patterns
    (the keys are drawn once at construction), which is what makes it usable
    with single-shot ridge readouts.
    """

    def __init__(self, b, n, rng=None):
        self.b = None
        self.n = None
        self.rng = rng if rng is not None else make_rng()
        self.random_boolean_keys = None
        self.set_random_boolean_keys(b, n)

    def _encode_bits(self, bits: np.ndarray):
        batch_size, seq_length, features, b = bits.shape
        assert b == self.b, "Input bit resolution does not match encoder configuration."
        bits_expanded = bits[:, :, :, np.newaxis, :]          # (B, T, F, 1, b)
        encoded = bits_expanded ^ self.random_boolean_keys     # broadcast to (B, T, F, n, b)
        return encoded.reshape(batch_size, seq_length, features, -1)

    def encode_float(self, data: np.ndarray):
        bits = dec2bin(data, self.b).astype(np.uint8)
        return self._encode_bits(bits)

    def encode_boolean(self, data: np.ndarray):
        return self._encode_bits(data)

    def set_random_boolean_keys(self, new_b=None, new_n=None):
        self.n = new_n if new_n is not None else self.n
        self.b = new_b if new_b is not None else self.b
        self.random_boolean_keys = self.rng.integers(
            0, 2, (1, 1, 1, self.n, self.b), dtype=np.uint8
        )
