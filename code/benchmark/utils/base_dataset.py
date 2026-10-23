import os
import numpy as np
from benchmark.utils.parameter import DatasetParameters
from project.boolean_reservoir.code.utils.utils import make_rng
from math import floor

# Per-sample side arrays, row-locked to x/y through shuffle and split:
# 'n' = true step counts (multistep PI), 'q' = recall query steps (query PI),
# 'positions' = the position after every step (query PI; y-valued, y = positions[q - 1]),
# 'targets' = the extra heads' labels (temporal dataset.heads; (m, H, d)).
# Suffix variants ride in the same dict so save/load/split stay symmetric with x/y.
_ROW_KEYS = ['n', 'q', 'positions', 'targets']
_DATA_KEYS = [f'{k}{sfx}' for sfx in ('', '_dev', '_test') for k in ['x', 'y', *_ROW_KEYS]]


class BaseDataset:
    def __init__(self, D: DatasetParameters):
        self.D = D
        for k in _DATA_KEYS:
            setattr(self, k, None)
        self.normalizer_x = None
        self.normalizer_y = None
        self.encoder_x = None

    @property
    def data_path(self):
        return self.D.path.with_suffix('.npz')

    @property
    def data(self):
        return {k: getattr(self, k, None) for k in _DATA_KEYS}

    def set_data(self, data_dict):
        for key, array in data_dict.items():
            setattr(self, key, array)

    def split_dataset(self, split=[0.8, 0.1, 0.1]):
        split_train = split[0] if self.D.split is None else self.D.split.train
        split_dev   = split[1] if self.D.split is None else self.D.split.dev
        split_test  = split[2] if self.D.split is None else self.D.split.test
        assert float(sum((split_train, split_dev, split_test))) == 1.0, "Split ratios must sum to 1."

        x, y = self.x, self.y
        n = x.shape[0]
        train_end = floor(split_train * n)
        dev_end   = floor((split_train + split_dev) * n)

        self.x      = x[:train_end]
        self.y      = y[:train_end]
        self.x_dev  = x[train_end:dev_end]
        self.y_dev  = y[train_end:dev_end]
        self.x_test = x[dev_end:]
        self.y_test = y[dev_end:]

        # side arrays are row-locked to x/y; slice each from its ORIGINAL array
        for k in _ROW_KEYS:
            a = getattr(self, k)
            if a is not None:
                setattr(self, k, a[:train_end])
                setattr(self, f'{k}_dev', a[train_end:dev_end])
                setattr(self, f'{k}_test', a[dev_end:])

    def save_data(self):
        # Atomic (temp file + rename): concurrent processes sharing a data dir
        # (cluster jobs on NFS) must see either no file or a complete one — a
        # direct write let a reader hit a truncated npz (EOFError). Generation is
        # deterministic, so racing writers just replace identical content.
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.data_path.with_name(f'{self.data_path.name}.{os.getpid()}.tmp')
        with open(tmp, 'wb') as fh:   # file handle: savez would append .npz to a path
            np.savez_compressed(fh, **{k: v for k, v in self.data.items() if v is not None})
        os.replace(tmp, self.data_path)

    def load_data(self):
        loaded = np.load(self.data_path)
        self.set_data(dict(loaded))

    def set_normalizer_x(self, normalizer_x):
        self.normalizer_x = normalizer_x

    def set_normalizer_y(self, normalizer_y):
        self.normalizer_y = normalizer_y

    def set_encoder_x(self, encoder_x):
        self.encoder_x = encoder_x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def normalize(self):
        self.x = self.normalizer_x(self.x)
        self.y = self.normalizer_y(self.y)

    def inverse_normalize_x(self, x):
        return self.normalizer_x.inverse(x)

    def inverse_normalize_y(self, y):
        return self.normalizer_y.inverse(y)

    def shuffle_data(self):
        # Own seeded stream, NOT the global RNG: shuffle_data runs after the
        # load-vs-generate branch, and those two branches consume different
        # amounts of global RNG (load consumes none, generate shuffles). Drawing
        # globally therefore made the permutation depend on whether the .npz
        # cache existed. This also gives NeuroGymDataset a seeded shuffle, which
        # it never had — it did no seeding of its own at all.
        rng = make_rng(self.D.seed, 'shuffle')
        perm = rng.permutation(self.x.shape[0])
        self.x = self.x[perm]
        self.y = self.y[perm]
        # side arrays ride the same permutation so they stay row-locked to x/y
        for k in _ROW_KEYS:
            a = getattr(self, k)
            if a is not None:
                setattr(self, k, a[perm])

    def encode_x(self):
        # copy=False: BooleanTransformer already returns uint8, so this is a no-op
        # cast instead of a full duplicate of a multi-GB array.
        self.x = self.encoder_x(self.x).astype(np.uint8, copy=False)
