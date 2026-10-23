import pandas as pd
import numpy as np
from project.boolean_reservoir.code.parameter import *
from project.boolean_reservoir.code.utils.utils import print_pretty_binary_matrix, override_symlink, make_rng
from project.boolean_reservoir.code.utils.param_utils import ExpressionEvaluator
from project.boolean_reservoir.code.graph import random_constrained_stub_matching, constrain_degree_of_bipartite_mapping
from datetime import datetime, timezone
import pickle
import re


class BatchedTensorHistoryWriter:
    def __init__(self, save_path='history', buffer_size=64, persist_to_disk=True):
        self.save_path = Path(save_path)
        self.file_index = 0
        self.time = 0
        self.buffer_size = buffer_size
        self.persist_to_disk = persist_to_disk
        self.buffer = []
        self.meta_buffer = []

    def append_batch(self, batch_tensor, meta_data):
        self.buffer.append(batch_tensor.copy())
        self.meta_buffer.append(meta_data)
        meta_data['file_idx'] = self.file_index
        meta_data['batch_number'] = len(self.meta_buffer)
        meta_data['samples'] = len(batch_tensor)
        meta_data['time'] = self.time
        self.time += 1
        if self.persist_to_disk and len(self.buffer) >= self.buffer_size:
            self._write_buffer()

    def _write_buffer(self):
        if not self.buffer or not self.persist_to_disk:
            return
        self.save_path.mkdir(parents=True, exist_ok=True)
        data = np.concatenate(self.buffer, axis=0)
        tensor_path = self.save_path / f'tensor_{self.file_index}.npy'
        np.save(tensor_path, data)
        meta_path = self.save_path / f'meta_{self.file_index}.csv'
        pd.DataFrame(self.meta_buffer).to_csv(meta_path, index=False)
        self.buffer = []
        self.meta_buffer = []
        self.file_index += 1

    def flush(self):
        if self.persist_to_disk:
            self._write_buffer()

    def reload_history(self, history_path=None, checkpoint_path=None, include={}, exclude={}):
        if self.persist_to_disk:
            history_path = Path(history_path) if history_path else self.save_path
            all_data = []
            all_meta_data = []
            idx = 0
            assert any(history_path.glob('*.npy')), f"No files found at {history_path}. Try Recording the data? Maybe the path is wrong"
            for _ in history_path.glob('*.npy'):
                tensor_path = history_path / f'tensor_{idx}.npy'
                tensor_data = np.load(tensor_path)
                meta_path = history_path / f'meta_{idx}.csv'
                meta_data = pd.read_csv(meta_path)
                all_data.append(tensor_data)
                all_meta_data.append(meta_data)
                idx += 1
            combined_data = np.concatenate(all_data, axis=0)
            combined_meta_data = pd.concat(all_meta_data, ignore_index=True, axis=0)
        else:
            combined_data = np.concatenate(self.buffer, axis=0)
            combined_meta_data = pd.DataFrame(self.meta_buffer)

        df = combined_meta_data
        expanded_meta_data = df.loc[df.index.repeat(df['samples'])].reset_index(drop=True)
        expanded_meta_data['sample_id'] = expanded_meta_data.groupby(['phase', 's', 'f']).cumcount()
        expanded_meta_data.drop(columns=['samples'], inplace=True)

        checkpoint_path = checkpoint_path if checkpoint_path else self.save_path / 'checkpoint'
        load_dict = dict()
        if self.persist_to_disk and checkpoint_path.exists():
            load_dict = SaveAndLoadModel.load_from_path_dict_or_checkpoint_folder(
                checkpoint_path=checkpoint_path,
                load_key_include_set=include,
                load_key_exclude_set=exclude,
            )

        return load_dict, combined_data, expanded_meta_data, combined_meta_data


class SaveAndLoadModel:
    # Absent from checkpoints saved before the wiring table was stored; the model then
    # rebuilds the table from the graph, so a missing file is expected, not a warning.
    OPTIONAL_KEYS = {'wiring'}
    # Saved before the LUT was stored compact: lut.npy, one padded row per node.
    LEGACY_SUFFIX = {'lut': '.npy'}

    @staticmethod
    def get_timestamp_utc():
        return datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S_%f")

    @staticmethod
    def load_or_generate(load_key: str, load_dict: dict, generator: callable):
        if load_key in load_dict:
            return load_dict[load_key]
        else:
            return generator()

    @staticmethod
    def save_model(config):
        P = config['P']
        if P.L.save_keys is None:
            return dict(), None
        save_path = config['save_path']
        checkpoint_path = save_path / 'checkpoints' / SaveAndLoadModel.get_timestamp_utc()
        checkpoint_path.mkdir(parents=True, exist_ok=False)
        override_symlink(save_path.name, save_path.parent / 'last_run')

        all_paths = SaveAndLoadModel.make_load_path_dict(checkpoint_path)
        paths = {k: all_paths[k] for k in P.L.save_keys if k in all_paths}

        save_map = {
            'parameters': lambda path: save_yaml_config(P, path.parent),
            'w_bi':        lambda path: np.save(path, config['w_bi']),
            'graph':       lambda path: SaveAndLoadModel.save_graph(path, config['graph']),
            'wiring':      lambda path: np.savez(path,
                                                  adj_list=config['wiring']['adj_list'],
                                                  mask=config['wiring']['mask']),
            'init_state':  lambda path: np.save(path, config['initial_states']),
            'lut':         lambda path: np.savez(path, flat=config['lut'].flat, offsets=config['lut'].offsets),
            'weights':     lambda path: np.savez(path,
                                                  W=config['weights']['W'],
                                                  b=config['weights']['b']),
        }

        for key in P.L.save_keys:
            if key not in save_map:
                raise KeyError(f"Unsupported key in save_keys: '{key}'")
            save_map[key](paths[key])

        override_symlink(checkpoint_path.name, checkpoint_path.parent / 'last_checkpoint')

        history = config.get('history')
        if history and history.record:
            history.save_path.mkdir(parents=True, exist_ok=True)
            override_symlink(Path('../checkpoints') / checkpoint_path.name, history.save_path / 'checkpoint')

        return paths, checkpoint_path

    @staticmethod
    def save_graph(path, graph):
        with open(path, 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_path_dict_or_checkpoint_folder(
        path_dict: dict = None,
        checkpoint_path = None,
        load_key_include_set: set = None,
        load_key_exclude_set: set = None,
    ):
        if path_dict is None:
            if checkpoint_path is None:
                raise ValueError("Either path_dict or checkpoint_path must be provided.")
            path_dict = SaveAndLoadModel.make_load_path_dict(checkpoint_path)

        if load_key_include_set is not None:
            path_dict = {k: path_dict[k] for k in load_key_include_set}

        if load_key_exclude_set is not None:
            path_dict = {k: v for k, v in path_dict.items() if k not in load_key_exclude_set}

        load_map = {
            'parameters': lambda path: load_yaml_config(path),
            'w_bi':        lambda path: np.load(path),
            'graph':       SaveAndLoadModel.load_graph,
            'wiring':      lambda path: dict(np.load(path)),
            'init_state':  lambda path: np.load(path),
            'lut':         lambda path: dict(np.load(path)),
            'weights':     lambda path: dict(np.load(path)),
        }

        d = dict()
        for key, path in path_dict.items():
            if key not in load_map:
                raise KeyError(f"Unsupported key in path_dict: '{key}'")
            legacy = path.with_suffix(SaveAndLoadModel.LEGACY_SUFFIX.get(key, path.suffix))
            if path.exists():
                d[key] = load_map[key](path)
            elif legacy.exists():  # padded per-node rows; the model keeps each node's first 2**k
                d[key] = np.load(legacy)
            elif key in SaveAndLoadModel.OPTIONAL_KEYS:
                pass
            else:
                print(f"Warning: Model object key '{key}' does not exist at '{path}' (A replacement may be generated)")
        return d

    def load(checkpoint_path: Path = None, paths: dict = None, parameter_override: Params = None):
        load_dict = SaveAndLoadModel.load_from_path_dict_or_checkpoint_folder(
            path_dict=paths, checkpoint_path=checkpoint_path
        )
        if parameter_override:
            load_dict['parameters'] = parameter_override
        return load_dict

    @staticmethod
    def load_graph(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def make_load_path_dict(folder_path):
        folder_path = Path(folder_path)
        files = [
            ('parameters', 'yaml'),
            ('w_bi',       'npy'),
            ('graph',      'pkl'),
            ('wiring',     'npz'),
            ('init_state', 'npy'),
            ('lut',        'npz'),
            ('weights',    'npz'),
        ]
        return {name: folder_path / f'{name}.{ext}' for name, ext in files}


class InputPerturbationStrategy:
    @staticmethod
    def xor(states, perturbations):
        return states ^ perturbations

    @staticmethod
    def and_(states, perturbations):
        return states & perturbations

    @staticmethod
    def or_(states, perturbations):
        return states | perturbations

    @staticmethod
    def override(states, perturbations):
        return perturbations

    @staticmethod
    def get(strategy: str):
        strategies = {
            'xor':      InputPerturbationStrategy.xor,
            'and':      InputPerturbationStrategy.and_,
            'or':       InputPerturbationStrategy.or_,
            'override': InputPerturbationStrategy.override,
        }
        if strategy not in strategies:
            raise ValueError(f'Unknown perturbation strategy: {strategy}')
        return strategies[strategy]


class InitializationStrategy:
    # Every strategy takes rng so `get(...)` returns a uniform callable; only
    # `random` uses it. rng=None -> unseeded entropy.
    @staticmethod
    def random(n_nodes, rng=None):
        if rng is None:
            rng = make_rng()
        return rng.integers(0, 2, (1, n_nodes), dtype=np.uint8)

    @staticmethod
    def zeros(n_nodes, rng=None):
        return np.zeros((1, n_nodes), dtype=np.uint8)

    @staticmethod
    def ones(n_nodes, rng=None):
        return np.ones((1, n_nodes), dtype=np.uint8)

    @staticmethod
    def every_other(n_nodes, rng=None):
        states = np.zeros((1, n_nodes), dtype=np.uint8)
        states[0, ::2] = 1
        return states

    @staticmethod
    def get(strategy: str):
        strategies = {
            'random':      InitializationStrategy.random,
            'zeros':       InitializationStrategy.zeros,
            'ones':        InitializationStrategy.ones,
            'every_other': InitializationStrategy.every_other,
        }
        if strategy not in strategies:
            raise ValueError(f'Unknown initialization strategy: {strategy}')
        return strategies[strategy]


class OutputActivationStrategy:
    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def get(strategy: str):
        strategies = {
            None:      OutputActivationStrategy.identity,
            'sigmoid': OutputActivationStrategy.sigmoid,
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown output activation strategy: {strategy}")
        return strategies[strategy]


class ChainedSelector:
    def __init__(self, max_val: int, min_val: int = 0, var: str = 'i', parameters: dict | None = None,
                 rng=None):
        # rng drives the 'R' (random subset) step. pandas' .sample() otherwise draws
        # from the GLOBAL numpy RNG — a hidden draw with no np.random.* to grep for.
        self.rng = rng if rng is not None else make_rng()
        self.max_val = max_val
        self.min_val = min_val
        self.var = var
        self.parameters = parameters or {}
        if var in self.parameters:
            raise ValueError(f"Variable '{var}' cannot be in parameters dict")
        self.parameters['min'] = min_val
        self.parameters['max'] = max_val

    def eval(self, chain: str) -> np.ndarray:
        s = pd.Series(range(self.min_val, self.max_val), name=self.var)
        chain = chain.strip()
        if not chain:
            return s.values.astype(np.int64)

        def substitute(expr: str) -> str:
            for key, val in self.parameters.items():
                expr = expr.replace(key, str(val))
            return expr

        for link in chain.split('->'):
            link = link.strip()
            if not link:
                continue
            match = re.match(r'^([A-Za-z]+)\s+(.*)$', link)
            if not match:
                raise ValueError(f"Invalid step syntax: {link}")
            tag, expr = match.groups()
            expr = substitute(expr.strip())
            tag = tag.upper()
            if tag == 'F':
                s = s[s.eval(expr)]
            elif tag == 'S':
                parts = [int(p) if p else None for p in expr.split(':')]
                slc = slice(*parts)
                s = s.iloc[slc]
            elif tag == 'R':
                n = int(expr) if expr else len(s)
                s = s.sample(n=min(n, len(s)), random_state=self.rng)
            else:
                raise ValueError(f"Unknown step tag: {tag}")
        return s.values.astype(np.int64)


class BipartiteMappingStrategy:
    @staticmethod
    def get(strategy_str: str, k_avg=None):
        if '-' in strategy_str:
            strategy_name, parameters_str = strategy_str.split('-', 1)
        else:
            strategy_name = strategy_str
            parameters_str = None

        strategy_map = {
            'identity': BipartiteMappingStrategy._identity,
            'zeroes':   BipartiteMappingStrategy._zeroes,
            'stub':     BipartiteMappingStrategy._stub,
            'in':       BipartiteMappingStrategy._in_degree,
            'out':      BipartiteMappingStrategy._out_degree,
        }

        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown bipartite mapping strategy: {strategy_name}")

        strategy_fn = strategy_map[strategy_name]

        def wrapped(p: Params, a: int, b: int, rng=None):
            return strategy_fn(p, a, b, parameters_str, k_avg=k_avg, rng=rng)

        return wrapped

    @staticmethod
    def _identity(p: Params, a: int, b: int, parameters_str=None, k_avg=None, rng=None):
        w = np.zeros((a, b), dtype=np.uint8)
        if b > 0:
            w[np.arange(a), np.arange(a) % b] = 1
        return w

    @staticmethod
    def _zeroes(p: Params, a: int, b: int, parameters_str=None, k_avg=None, rng=None):
        return np.zeros((a, b), dtype=np.uint8)

    @staticmethod
    def _stub(p: Params, a: int, b: int, parameters_str: str, k_avg=None, rng=None):
        expression_evaluator = ExpressionEvaluator({'a': a, 'b': b, 'I_n_nodes': a, 'R_n_nodes': b, 'R_k_avg': k_avg})
        params = parameters_str.split(':')
        assert len(params) == 5, "Stub strategy must have format 'a_min:a_max:b_min:b_max:p'"
        a_min_expr, a_max_expr, b_min_expr, b_max_expr, p_expr = params
        a_min = int(expression_evaluator.eval(a_min_expr))
        a_max = int(expression_evaluator.eval(a_max_expr))
        b_min = int(expression_evaluator.eval(b_min_expr))
        b_max = int(expression_evaluator.eval(b_max_expr))
        p = expression_evaluator.eval(p_expr)
        assert 0 <= p <= 1
        assert 0 <= b_min <= b_max <= a
        assert 0 <= a_min <= a_max <= b
        w = random_constrained_stub_matching(a, b, a_min, a_max, b_min, b_max, p, rng=rng)
        return np.array(w, dtype=np.uint8)

    @staticmethod
    def _constrain_degree(p: Params, a: int, b: int, parameters_str: str, in_degree: bool, k_avg=None, rng=None):
        if k_avg is None and parameters_str and 'R_k_avg' in parameters_str:
            raise ValueError("Strategy expression uses 'R_k_avg' but k_avg was not provided")
        expression_evaluator = ExpressionEvaluator({'a': a, 'b': b, 'I_n_nodes': a, 'R_n_nodes': b, 'R_k_avg': k_avg})
        params = parameters_str.split(':')
        assert len(params) == 3, "Degree strategies must have format 'min_degree:max_degree:p'"
        min_degree_expr, max_degree_expr, p_expr = params
        min_degree = int(expression_evaluator.eval(min_degree_expr))
        max_degree = int(expression_evaluator.eval(max_degree_expr))
        p = expression_evaluator.eval(p_expr)
        assert 0 <= p <= 1
        if in_degree:
            assert 0 <= min_degree <= max_degree <= a
        else:
            assert 0 <= min_degree <= max_degree <= b
        w = constrain_degree_of_bipartite_mapping(a, b, min_degree, max_degree, p, in_degree=in_degree, rng=rng)
        return np.array(w, dtype=np.uint8)

    @staticmethod
    def _in_degree(p: Params, a: int, b: int, parameters_str: str, k_avg=None, rng=None):
        return BipartiteMappingStrategy._constrain_degree(p, a, b, parameters_str, in_degree=True, k_avg=k_avg, rng=rng)

    @staticmethod
    def _out_degree(p: Params, a: int, b: int, parameters_str: str, k_avg=None, rng=None):
        return BipartiteMappingStrategy._constrain_degree(p, a, b, parameters_str, in_degree=False, k_avg=k_avg, rng=rng)
