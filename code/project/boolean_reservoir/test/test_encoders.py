"""encoders.py — the value→bits layer: registry, kernels, contract, decode.

The registry↔Literal pin is the load-bearing test: parameter.py's EncodingName
is a STATIC Literal while encoders.ENCODERS is the RUNTIME source of truth;
this file is what keeps the two from drifting apart when an encoder is added.
"""
import pytest
import numpy as np
from math import comb
from typing import get_args

from project.boolean_reservoir.code.encoders import (
    ENCODERS, ENCODING_NAMES, BinaryEmbedding, bin2dec, dec2base2_delta_sigma, dec2bin,
    dec2base2_lsb, dec2delta_sigma, dec2delta_sigma_free, dec2gray, dec2hamming_weight,
    dec2phase, dec2primes, dec2rate, dec2tally, get_encoder, hamming_weight2dec,
    load_primes, quantize, unpack_bits, validate_analog, _hamming_tables,
    _revdoor_cycle,
)
from project.boolean_reservoir.code.parameter import EncodingName
from project.boolean_reservoir.code.utils.utils import make_rng


# ---------------------------------------------------------------------------
# Registry ↔ config-schema pin (the DRY guard)
# ---------------------------------------------------------------------------

def test_registry_matches_encoding_name_literal():
    """Adding an encoder without updating EncodingName (or vice versa) fails here."""
    assert list(ENCODERS) == list(get_args(EncodingName)), (
        f"registry/schema drift: registry={list(ENCODERS)}, "
        f"Literal={list(get_args(EncodingName))}")
    assert ENCODING_NAMES == tuple(get_args(EncodingName))


def test_stochastic_flag_matches_kernel_signature():
    """Every kernel flagged stochastic must accept rng; deterministic ones must not."""
    x = np.linspace(0, 1, 6).reshape(1, 2, 3)  # 3-D: delta_sigma_free requires (m, s, f)
    for name, spec in ENCODERS.items():
        if spec.stochastic:
            out = spec.fn(x, 4, rng=make_rng(1, 'rate'))
        else:
            out = spec.fn(x, 4)
        assert out.dtype == np.bool_ and out.shape == x.shape + (4,), name


def test_get_encoder_error_message():
    with pytest.raises(ValueError, match=r"encoding 'nope' is not an option! Valid: \[") as ex:
        get_encoder('nope')
    assert list(ENCODERS)[0] in str(ex.value)  # valid names listed in registry order


def test_register_rejects_duplicate():
    from project.boolean_reservoir.code.encoders import register
    with pytest.raises(ValueError, match="registered twice"):
        register('base2')(lambda values, bits: None)


# ---------------------------------------------------------------------------
# Kernel contract: shape, dtype, [0,1] input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('bits', [1, 4, 10])
def test_all_encoders_shape_dtype_range(bits):
    x = np.linspace(0, 1, 12).reshape(3, 4)
    x3 = np.linspace(0, 1, 24).reshape(2, 3, 4)
    for name, spec in ENCODERS.items():
        fn = (lambda v, b: spec.fn(v, b, rng=make_rng(1, 'rate'))) if spec.stochastic else spec.fn
        if name == 'delta_sigma_free':
            # contract: strictly (m, s, f) — pinned, not skipped
            with pytest.raises(AssertionError, match='expects \(m, s, f\)'):
                fn(x, bits)
            out = fn(x3, bits)
            assert out.shape == (2, 3, 4, bits), name
        elif name == 'base2_delta_sigma' and bits < 4:
            # head=4 default requires bits >= head — legacy raises, not clamps
            with pytest.raises(ValueError, match='negative dimensions'):
                fn(x, bits)
        else:
            out = fn(x, bits)
            assert out.shape == (3, 4, bits), name
            assert out.dtype == np.bool_, name


def test_validate_analog_rejects():
    x = np.linspace(0, 1, 6)
    with pytest.raises(AssertionError, match='expects float input'):
        validate_analog((x > 0.5))
    with pytest.raises(AssertionError, match='must be normalized'):
        validate_analog(x + 1.0)
    with pytest.raises(AssertionError, match='must be normalized'):
        validate_analog(x - 0.5)


# ---------------------------------------------------------------------------
# Per-kernel semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('bits', [4, 10])
def test_base2_roundtrip(bits):
    x = np.linspace(0, 1, 35).reshape(5, 7)
    decoded = bin2dec(dec2bin(x, bits), bits)
    assert np.abs(decoded - x).max() < 1 / (2**bits - 1) + 1e-9


def test_base2_endpoints_exact():
    out = dec2bin(np.array([0.0, 1.0], dtype=np.float32), 4)
    assert (out[0] == False).all() and (out[1] == True).all()  # noqa: E712


def test_lsb_is_mirrored_msb():
    x = np.linspace(0, 1, 12).reshape(3, 4)
    assert np.array_equal(dec2base2_lsb(x, 6), dec2bin(x, 6)[..., ::-1])


def test_gray_adjacent_values_flip_one_bit():
    bits = 5
    codes = np.arange(2**bits)
    gray = unpack_bits(codes ^ (codes >> 1), bits)
    flips = (gray[1:] != gray[:-1]).sum(axis=-1)
    assert (flips == 1).all()


def test_tally_count_matches_round():
    bits = 10
    x = np.linspace(0, 1, 21)
    assert (dec2tally(x, bits).sum(axis=-1) == (bits * x).round()).all()


def test_rate_is_expectation_code():
    bits, m = 200, 64
    x = np.linspace(0.05, 0.95, m)
    out = dec2rate(x, bits, rng=make_rng(3, 'rate'))
    # binomial noise: pointwise 4-sigma bound on |mean - p| (deterministic given seed)
    sigma = np.sqrt(x * (1 - x) / bits)
    assert np.abs(out.mean(axis=-1) - x).max() < (4 * sigma).max()


def test_rate_reproducible_for_same_seed():
    x = np.linspace(0, 1, 12).reshape(3, 4)
    a = dec2rate(x, 8, rng=make_rng(42, 'rate'))
    b = dec2rate(x, 8, rng=make_rng(42, 'rate'))
    assert np.array_equal(a, b)


def test_delta_sigma_count_tracks_value():
    bits = 16
    x = np.linspace(0, 1, 33)
    count = dec2delta_sigma(x, bits).sum(axis=-1)
    assert np.abs(count - bits * x).max() <= 1.0 + 1e-6


def test_delta_sigma_free_integrates_across_steps():
    """Free-running integrator: total ones over the stream track bits * sum(x),
    NOT per-step quantization — the documented difference vs dec2delta_sigma."""
    bits, steps = 8, 5
    x = np.zeros((2, steps, 1), dtype=np.float32)
    x[0, :, 0] = 0.2    # per-step 1.6 → reset code emits 2/step; free integrates 8 total
    x[1, :, 0] = 0.0
    out = dec2delta_sigma_free(x, bits)
    total = out[0].sum()
    assert abs(total - bits * x[0].sum()) <= 1.0          # the stream is integrated
    assert out[1].sum() == 0                              # zero stream drains, emits nothing
    reset_total = sum(dec2delta_sigma(x[0, si], bits).sum() for si in range(steps))
    assert total < reset_total                            # carry across steps, not per-step


def test_base2_delta_sigma_head_is_positional_tail_is_sigma():
    bits, head = 10, 4
    x = np.linspace(0, 1, 33)
    out = dec2base2_delta_sigma(x, bits, head=head)
    assert out.shape == (33, bits)
    coarse = np.floor(x * (1 << head)).astype(np.int64).clip(0, (1 << head) - 1)
    assert np.array_equal(out[..., :head], unpack_bits(coarse, head))
    assert out[..., head:].sum(axis=-1).max() <= bits - head


def test_phase_constant_duty():
    bits = 10
    x = np.linspace(0, 1, 21)
    duty = dec2phase(x, bits).sum(axis=-1)
    assert np.abs(duty - bits / 2).max() <= 1  # 50% duty up to one tick


def test_primes_covers_all_integers_and_caps():
    bits = 6
    weights, max_val = None, None
    from project.boolean_reservoir.code.encoders import _prime_table
    weights, max_val = _prime_table(bits)
    # greedy over descending primes + 1 must represent every code 0..max_val
    for code in range(max_val + 1):
        c = code
        for w in weights:
            take = c >= w
            c = c - int(take) * w
        assert c == 0, code
    with pytest.raises(ValueError, match='Cannot load more than 1000 primes'):
        load_primes(1001)


# ---------------------------------------------------------------------------
# BinaryEmbedding — deterministic key-XOR expansion
# ---------------------------------------------------------------------------

def test_binary_embedding_shape_and_determinism():
    enc = BinaryEmbedding(b=4, n=3, rng=make_rng(42, 'embedding'))
    bits = dec2bin(np.linspace(0, 1, 12).reshape(3, 4), 4)          # (3, 4, 4)
    a = enc.encode_boolean(bits[None])                              # contract: (B, T, F, b)
    b = enc.encode_boolean(bits[None])
    # bool ^ uint8 keys → uint8 output in {0,1} (legacy dtype behavior)
    assert a.shape == (1, 3, 4, 12) and a.dtype == np.uint8 and set(np.unique(a)) <= {0, 1}
    assert np.array_equal(a, b)  # keys fixed at construction → reproducible
    # each key bank preserves the XOR relation: key_j ⊕ input = output_j
    keys = enc.random_boolean_keys[0, 0, 0]                          # (n, b)
    assert np.array_equal(a[0, ..., :4], bits ^ keys[0])


# ---------------------------------------------------------------------------
# hamming_weight — full 2**bits code space ordered by (weight, const-weight Gray)
# ---------------------------------------------------------------------------

def test_hamming_weight_codebook_is_permutation():
    bits = 6
    codebook, rank = _hamming_tables(bits)
    ints = (codebook * (1 << np.arange(bits - 1, -1, -1, dtype=np.int64))).sum(-1)
    assert sorted(ints.tolist()) == list(range(2**bits))
    assert np.array_equal(rank[ints], np.arange(2**bits))       # rank inverts codebook


def test_hamming_weight_ordering():
    """Primary: weight ascending (unchanged). Secondary: revolving-door —
    within a count-k bucket consecutive words keep the weight and differ by
    exactly one element swap (the constant-weight Gray order), NOT MSB-value
    ascending (the colex secondary let adjacent values flip up to
    10 bits)."""
    bits = 5
    codebook, _ = _hamming_tables(bits)
    weights = codebook.sum(-1)
    assert (np.diff(weights) >= 0).all()                        # primary: weight ascending
    for w in np.unique(weights):                                # secondary: 2-change within bucket
        bucket = codebook[weights == w]
        assert ((bucket[1:] != bucket[:-1]).sum(-1) == 2).all()


def test_hamming_weight_adjacent_transitions_cap_two():
    """The property the paper claims: across consecutive rows,
    max ΔH <= 2 — asserted, not eyeballed. ΔH=1 can only happen at the `bits`
    bucket boundaries (same-weight words differ by an even distance), so the
    profile is exactly: `bits` steps of 1 flip, everything else 2 flips."""
    for bits in (1, 2, 4, 10, 16):
        codebook, _ = _hamming_tables(bits)
        flips = (codebook[1:] != codebook[:-1]).sum(-1)
        assert flips.max() <= 2, bits
        assert int((flips == 1).sum()) == bits, bits            # one per bucket pair
        assert int((flips == 2).sum()) == 2**bits - 1 - bits, bits


def test_hamming_weight_boundary_interlock_and_reference_order():
    """The b=4 enumeration, spelled out: each bucket
    reads as its revolving-door cycle cut open at the bucket's first word, and
    every next bucket opens by extending the previous bucket's last word with
    one element (that is what makes the boundary steps 1-flip). Pinning the
    exact masks: archived grid results are reproducible only if this order
    never churns silently."""
    codebook, _ = _hamming_tables(4)
    ints = (codebook * (1 << np.arange(3, -1, -1, dtype=np.int64))).sum(-1)
    expected = [
        0b0000,
        0b0001, 0b0010, 0b0100, 0b1000,          # {0} {1} {2} {3}
        0b1001, 0b0011, 0b0110, 0b0101, 0b1100, 0b1010,
        # {0,3} {0,1} {1,2} {0,2} {2,3} {1,3} — cycle cut at {0,3}
        0b1011, 0b0111, 0b1101, 0b1110,          # {0,1,3} {0,1,2} {0,2,3} {1,2,3}
        0b1111,
    ]
    assert ints.tolist() == expected
    # boundaries: {3}->{0,3}, {1,3}->{0,1,3}, {1,2,3}->{0,1,2,3}, plus {}->{0}
    flips = (codebook[1:] != codebook[:-1]).sum(-1)
    assert [int(i) for i in np.flatnonzero(flips == 1)] == [0, 4, 10, 14]


def test_revolving_door_cycle():
    """The bucket primitive, pinned independently of the bucket cuts: a cyclic
    constant-weight Gray code (Hamiltonian cycle on J(n,k)) — consecutive AND
    last->first differ by exactly one element swap; every k-subset exactly once."""
    for n, k in ((5, 2), (6, 3), (7, 3), (8, 4)):
        cycle = _revdoor_cycle(n, k)
        assert len(cycle) == comb(n, k)
        assert len(set(cycle)) == len(cycle)                    # no repeats
        words = unpack_bits(np.array(cycle, dtype=np.int64), n)
        assert (words.sum(-1) == k).all()                       # constant weight
        assert ((words[1:] != words[:-1]).sum(-1) == 2).all()   # consecutive: one swap
        assert ((words[:1] != words[-1:]).sum(-1) == 2).all()   # cyclic closure


def test_hamming_weight_buckets_are_binomial():
    bits = 6
    codebook, _ = _hamming_tables(bits)
    weights = codebook.sum(-1)
    sizes = [int((weights == k).sum()) for k in range(bits + 1)]
    assert sizes == [comb(bits, k) for k in range(bits + 1)]


def test_hamming_weight_roundtrip_and_monotone_count():
    bits = 10
    x = np.linspace(0, 1, 101)
    out = dec2hamming_weight(x, bits)
    q = quantize(x, bits)
    assert np.array_equal(hamming_weight2dec(out, bits), q / (2**bits - 1))
    assert (np.diff(out.sum(-1)) >= 0).all()                    # count channel monotone


def test_hamming_weight_rejects_oversized_codebook():
    with pytest.raises(ValueError, match='1 <= bits <= 16'):
        dec2hamming_weight(np.linspace(0, 1, 3), 17)


def test_hamming_weight_count_window_is_binomial_not_uniform():
    """The design trade, pinned: the count-k x-window is ~C(bits,k)/(2**bits-1) wide."""
    bits = 10
    codebook, _ = _hamming_tables(bits)
    weights = codebook.sum(-1)
    # rank of the first count-5 word vs rank of the first count-6 word → window edges
    s5 = int(np.argmax(weights == 5))
    s6 = int(np.argmax(weights == 6))
    width = (s6 - s5) / (2**bits - 1)
    assert 0.24 < width < 0.25                                  # ~252/1023, vs delta_sigma's 0.1
    w0 = int(np.argmax(weights == 1))
    assert w0 / (2**bits - 1) < 0.002                           # extremes are ~1000x finer


def test_base2_table_path_matches_arithmetic_reference():
    """The gather table is a cache of the arithmetic code — bit-identical, incl.
    legacy low-bits wrap for direct out-of-range calls and the >16 fallback."""
    for bits in (1, 4, 10, 16):
        x = np.linspace(0, 1, 33)
        assert np.array_equal(dec2bin(x, bits), unpack_bits(quantize(x, bits), bits))
    weird = np.array([1.2, -0.3, 2.0])
    assert np.array_equal(dec2bin(weird, 4), unpack_bits(quantize(weird, 4), 4))
    x20 = np.linspace(0, 1, 5)
    assert np.array_equal(dec2bin(x20, 20), unpack_bits(quantize(x20, 20), 20))
