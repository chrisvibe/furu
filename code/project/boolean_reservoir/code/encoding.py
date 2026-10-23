"""Encoding pipeline: normalize → encode → transform for reservoir injection.

The pure value→bits kernels and their registry live in encoders.py (importable
without the config layer). This module wires them to Params and applies the
bits-level transforms — redundancy/permute expansion, feature interleaving,
the binary_embedding key-XOR — plus the dataset normalizers. Every encoder
symbol is re-exported here so `from ...encoding import X` keeps working for
all existing consumers (temporal / path_integration / neurogym dataset_init,
path_integration bin2dec tests, ca).
"""
import numpy as np

from project.boolean_reservoir.code.encoders import (
    ENCODERS, ENCODING_NAMES, BinaryEmbedding, bin2dec, dec2base2_delta_sigma, dec2bin,
    dec2base2_lsb, dec2delta_sigma, dec2delta_sigma_free, dec2gray, dec2phase,
    dec2primes, dec2rate, dec2tally, get_encoder, load_primes, quantize,
    unpack_bits, validate_analog,
)
from project.boolean_reservoir.code.parameter import InputParams
from project.boolean_reservoir.code.utils.utils import make_rng

__all__ = [
    'BooleanTransformer', 'BooleanEncoder', 'interleave_features', 'channel_spec', 'noop_input',
    'MinMaxNormalization', 'StandardScaler', 'BinaryEmbedding',
    'ENCODERS', 'ENCODING_NAMES', 'get_encoder', 'validate_analog',
    'quantize', 'unpack_bits', 'bin2dec', 'load_primes',
    'dec2bin', 'dec2base2_lsb', 'dec2tally', 'dec2gray', 'dec2rate',
    'dec2delta_sigma', 'dec2delta_sigma_free', 'dec2base2_delta_sigma',
    'dec2phase', 'dec2primes',
]


class BooleanEncoder:
    """Config-facing front door: normalized floats → bit patterns → BooleanTransformer.

    Takes an input-layer spec (InputParams), not the whole Params: every channel the
    reservoir reads — the task input, and side channels such as the query channel
    (channel_spec) — goes through this one encoder."""

    def __init__(self, I: InputParams):
        self.I = I
        apply_redundancy = (self.I.encoding != 'binary_embedding')
        self.transformer = BooleanTransformer(I, apply_redundancy=apply_redundancy)

    def __call__(self, x):
        bin_values = self._encode(x)
        return self.transformer(bin_values)

    def _encode(self, values):
        x = validate_analog(values)
        spec = get_encoder(self.I.encoding)
        if spec.stochastic:
            # Fresh stream per call, derived from I.seed: the same value must encode
            # to the same pattern every time, not drift as the encoder is reused.
            return spec.fn(x, self.I.resolution, rng=make_rng(self.I.seed, 'rate'))
        return spec.fn(x, self.I.resolution)


class BooleanTransformer:
    def __init__(self, I: InputParams, apply_redundancy=True):
        self.I = I
        self.redundancy = self.I.redundancy
        self.interleaving = self.I.interleaving
        self.apply_redundancy = apply_redundancy

        if self.I.encoding == 'binary_embedding':
            self.binary_encoder = BinaryEmbedding(b=self.I.resolution, n=self.I.redundancy,
                                                  rng=make_rng(self.I.seed, 'embedding'))
        else:
            self.binary_encoder = None

    def __call__(self, bin_values):
        # numpy bool_ is a 1-byte 0/1, so viewing it as uint8 is free and makes the
        # redundancy expansion below allocate uint8 directly rather than bool followed
        # by a full-size astype copy.  Byte-for-byte identical output.
        if bin_values.dtype == np.bool_:
            bin_values = bin_values.view(np.uint8)
        if self.apply_redundancy and self.redundancy > 1:
            perm_str = self.I.permute
            if perm_str and not isinstance(perm_str, list):
                m, s, f, b = bin_values.shape
                K = self.redundancy
                perm = [int(c) for c in perm_str]
                expanded = np.broadcast_to(bin_values[:, :, :, np.newaxis, :], (m, s, f, K, b)).copy()
                full_perm = [0, 1] + [2 + p for p in perm]
                bin_values = np.transpose(expanded, full_perm).reshape(m, s, f, -1)
            else:
                bin_values = np.repeat(bin_values, self.redundancy, axis=-1)
        elif self.binary_encoder is not None:
            bin_values = self.binary_encoder.encode_boolean(bin_values)

        if self.interleaving:
            bin_values = interleave_features(bin_values, group_size=self.interleaving)

        return bin_values.astype(np.uint8, copy=False)


def channel_spec(I: InputParams, encoding: str, redundancy: int) -> InputParams:
    """The spec for one more single-feature input channel next to the input layer's own:
    same resolution and seed, its own encoding and redundancy, no permute/interleave.
    Feed it to BooleanEncoder — side channels reuse the one encoding path."""
    return I.model_copy(update=dict(encoding=encoding, redundancy=redundancy, features=1,
                                    permute=None, interleaving=0))


def noop_input(perturbation: str, previous: np.ndarray) -> np.ndarray:
    """Input bits that leave the input nodes untouched: the perturbation's identity
    element (override: rewrite the current value = `previous`; xor/or: 0; and: 1).
    Input nodes have no incoming edges and hold their state across ticks, so a step of
    no-op input is exactly a free reservoir tick in every perturbation mode."""
    if perturbation == 'override':
        return previous.copy()
    fill = 1 if perturbation == 'and' else 0
    return np.full_like(previous, fill)


def interleave_features(x, group_size=1):
    shape = x.shape
    n, m, d, b = shape
    gs = group_size
    num_full = b // gs
    rem = b % gs
    parts = []
    for i in range(num_full):
        for j in range(d):
            parts.append(x[:, :, j:j+1, i*gs:(i+1)*gs])
    if rem > 0:
        for j in range(d):
            parts.append(x[:, :, j:j+1, -rem:])
    return np.concatenate(parts, axis=-1).reshape(shape)


class MinMaxNormalization:
    """Per-feature [0, 1] scaling. Calling it fits and applies; transform() applies an
    already-fitted scaling (e.g. the training set's, to data from another distribution)."""
    def __call__(self, data):
        return self.fit(data).transform(data)

    def fit(self, data):
        data = data.astype(np.float32)
        self.min_ = data.min(axis=(0, 1), keepdims=True)
        self.max_ = data.max(axis=(0, 1), keepdims=True)
        return self

    def transform(self, data):
        denom = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return (data.astype(np.float32) - self.min_) / denom

    def inverse(self, data):
        return data * (self.max_ - self.min_) + self.min_


class StandardScaler:
    """Per-feature standardization; fit/transform as MinMaxNormalization."""
    def __call__(self, data):
        return self.fit(data).transform(data)

    def fit(self, data):
        data = data.astype(np.float32)
        self.mean_ = data.mean(axis=(0, 1), keepdims=True)
        self.std_ = np.maximum(data.std(axis=(0, 1), keepdims=True), 1e-8)
        return self

    def transform(self, data):
        return (data.astype(np.float32) - self.mean_) / self.std_

    def inverse(self, data):
        return data * self.std_ + self.mean_
