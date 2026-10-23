import copy
from benchmark.temporal.temporal_density_parity_dataset import TemporalDataset, head_targets
from benchmark.temporal.parameter import TemporalDatasetParams
from project.boolean_reservoir.code.encoding import BooleanTransformer
from project.boolean_reservoir.code.glm import ReadoutGroup
from project.boolean_reservoir.code.parameter import Params
from project.boolean_reservoir.code.train_model import DatasetInit


class TemporalDatasetInit(DatasetInit):

    def _create_raw_dataset(self, P: Params):
        cache_key = str(('raw', self.__class__.__name__, P.D.model_dump()))
        def _build():
            D = P.D
            D.update_path()
            assert isinstance(D, TemporalDatasetParams), f"Unsupported dataset params type {type(D)}"
            return TemporalDataset(D)  # flat x — reshape applied post-cache
        dataset = self._cache_dataset(cache_key, _build)  # always a deep copy
        I = P.M.I
        if P.D.heads:   # from the raw streams (m, 1, d, b), before the model's reshape
            dataset.set_data({'targets': head_targets(dataset.x, P.D.heads)})
        # (m, 1, d, b) -> (m, 1, b, d) first: step t then carries every channel's bit t (else the reshape strings
        # one channel's consecutive bits into a step). A no-op in memory order at d = 1.
        x = dataset.x.swapaxes(-1, -2).reshape(dataset.x.shape[0], -1, I.features, I.resolution)  # (m, s, f, b)
        dataset.set_data({'x': x, 'y': dataset.y})
        return dataset

    def readout_groups(self, P: Params) -> tuple:
        """dataset.heads: one readout per head on the same streams, each scored on every row; with several
        channels (dataset.dimensions > 1) each channel is scored apart, as <head>_c<j>."""
        return (ReadoutGroup('targets', 'targets', names=tuple(h.name for h in P.D.heads), per_output=True),) if P.D.heads else ()

    def _process_dataset(self, dataset, P: Params):
        dataset = copy.deepcopy(dataset)
        encoder = BooleanTransformer(P.M.I)
        dataset.set_encoder_x(encoder)
        dataset.encode_x()
        dataset.split_dataset()
        return dataset


if __name__ == '__main__':
    pass
    from project.boolean_reservoir.code.parameter import load_yaml_config
    from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
    P = load_yaml_config('config/temporal/kq_and_gr/grid_search/test.yaml')
    P.D.samples = 10
    P.D.sampling_mode = 'random'
    P.M.I.redundancy = 2
    P.M.I.features = 2
    p = generate_param_combinations(P)[0]
    dataset = TemporalDatasetInit().train(p)
