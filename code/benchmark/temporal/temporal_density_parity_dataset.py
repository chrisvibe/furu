import numpy as np
from benchmark.temporal.parameter import TemporalDatasetParams
from project.boolean_reservoir.code.utils.utils import make_rng
from benchmark.utils.base_dataset import BaseDataset


class TemporalDatasetBase(BaseDataset):
    def __init__(self, D: TemporalDatasetParams, task):
        # bits i.e. 12: 101010111010
        # window (in the bit stream) i.e. 4: 10101011[1010]
        # delay (for the window) i.e. 1: 1010101[1101]0
        super().__init__(D)
        self.task = task

        if self.data_path.exists() and not D.generate_data:
            self.load_data()
        else:
            if D.bits < D.window + D.delay:
                print('Warning: bits < delay + window, overriding bits...')
                D.bits = D.window + D.delay
            raw_data = self.generate_data(D.samples, D.bits, D.delay, D.window)
            self.set_data(raw_data)
            self.save_data()
        if D.shuffle:
            self.shuffle_data()

    @staticmethod
    def gen_integer_samples(samples, stream_length, sampling_mode, shuffle=False, rng=None):
        """Generate integer samples based on sampling mode."""
        if rng is None:
            rng = make_rng()
        if sampling_mode == 'exhaustive':
            max_possible = 2**stream_length
            if stream_length > 20:
                raise ValueError(
                    f"Exhaustive mode with stream_length={stream_length} would generate "
                    f"{max_possible} samples, which is too large. Consider random mode or smaller stream_length."
                )
            int_samples = np.arange(max_possible)
            if shuffle:
                rng.shuffle(int_samples)
            if samples > len(int_samples):
                int_samples = np.tile(int_samples, samples // len(int_samples) + 1)
            int_samples = int_samples[:samples]
        else:  # 'random'
            if stream_length > 63:
                raise ValueError(
                    f"stream_length={stream_length} is too large for random integer generation. "
                    f"Maximum is 63 bits."
                )
            int_samples = rng.integers(0, 2**stream_length, size=samples)
        return int_samples

    def generate_data(self, samples, stream_length, delay, window):
        """Generate dataset from integer samples."""
        total_streams = samples * self.D.dimensions
        # Explicit generation stream, independent of the 'shuffle' stream used by
        # shuffle_data — so neither can shift the other by consuming first.
        rng = make_rng(self.D.seed, 'generate')
        if self.D.sampling_mode == 'exhaustive' and self.D.dimensions > 1:
            # one exhaustive list per channel, each its own permutation: tiling ONE list over samples x c streams
            # repeats every sample 2**b / c times (sample i == sample i + 2**b / c), so train and dev shared samples
            int_samples = np.stack([self.gen_integer_samples(samples, stream_length, 'exhaustive', shuffle=True, rng=rng)
                                    for _ in range(self.D.dimensions)], axis=1).ravel()
        else:
            int_samples = self.gen_integer_samples(total_streams, stream_length, self.D.sampling_mode,
                                                   shuffle=self.D.shuffle, rng=rng)

        # Vectorised bit unpacking: (total_streams, stream_length), dtype bool
        mask = (1 << np.arange(stream_length - 1, -1, -1)).astype(np.int64)
        arrays = (int_samples[:, None] & mask[None, :]) != 0

        x = arrays.reshape(samples, self.D.dimensions, stream_length)
        y = self.task(x, window, delay).astype(np.float32)
        if self.D.washout:   # unscored random lead-in; windows count from the end, so labels are unchanged
            prefix = make_rng(self.D.seed, 'washout').integers(0, 2, (samples, self.D.dimensions, self.D.washout)) != 0
            x = np.concatenate([prefix, x], axis=-1)
        x = x[:, np.newaxis]  # m x 1 x d x (washout + b)
        return {'x': x, 'y': y}

class TemporalDensityDataset(TemporalDatasetBase):
    def __init__(self, p: TemporalDatasetParams):
        TemporalDatasetBase.__init__(self, p, self.density_task)

    @staticmethod
    def density_task(bits, window, delay):
        """Over the last axis: one stream (b,) or many (..., b)."""
        b = bits.shape[-1]
        if delay + window > b:
            raise ValueError("delay + window must be less than or equal to the bits in the input stream")
        count_ones = bits[..., b - window - delay:b - delay].sum(axis=-1)
        return 2 * count_ones > window


class TemporalParityDataset(TemporalDatasetBase):
    def __init__(self, p: TemporalDatasetParams):
        TemporalDatasetBase.__init__(self, p, self.parity_task)

    @staticmethod
    def parity_task(bits, window, delay):
        """Over the last axis: one stream (b,) or many (..., b)."""
        b = bits.shape[-1]
        if delay + window > b:
            raise ValueError("delay + window must be less than or equal to the bits in the input stream")
        count_ones = bits[..., b - window - delay:b - delay].sum(axis=-1)
        return count_ones % 2 != 0


TASKS = {'density': TemporalDensityDataset.density_task, 'parity': TemporalParityDataset.parity_task}


def head_targets(x, heads) -> np.ndarray:
    """Each head's labels of the streams x (m, ..., d, b): (m, len(heads), d) float32, the y of a dataset
    of that task, window and delay on the same streams."""
    streams = x.reshape(len(x), -1, x.shape[-1])
    return np.stack([TASKS[h.task](streams, h.window, h.delay) for h in heads], axis=1).astype(np.float32)


class TemporalDataset:
    def __new__(cls, D: TemporalDatasetParams):
        assert isinstance(D, TemporalDatasetParams)
        if D.task == 'density':
            return TemporalDensityDataset(D)
        elif D.task == 'parity':
            return TemporalParityDataset(D)
        else:
            raise ValueError(f"Unknown task: {D.task}")
