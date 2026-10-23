from pydantic import BaseModel, model_validator, ConfigDict, Field
import yaml
import copy, itertools, math
from dataclasses import dataclass, field as dc_field
from typing import Callable, ClassVar
from pathlib import Path, PosixPath, WindowsPath
import sympy
from inspect import signature, Parameter
import re

def pydantic_init():
    BaseModel.__str__ = lambda self: yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False).strip()
    BaseModel.__repr__ = BaseModel.__str__ 
    def represent_pathlib_path(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

    yaml.add_representer(Path, represent_pathlib_path)
    yaml.add_representer(PosixPath, represent_pathlib_path)
    yaml.add_representer(WindowsPath, represent_pathlib_path)

def calculate_w_broadcasting(operator: Callable[[float, float], float], a, b):
    # list-list, list-value, value-list, value-value
    if isinstance(a, list) and isinstance(b, list):
        return [operator(a[i], b[i]) for i in range(len(a))]
    elif isinstance(a, list):
        return [operator(x, b) for x in a]
    elif isinstance(b, list):
        return [operator(a, y) for y in b]
    else:
        return operator(a, b)


class ExpressionEvaluator:
    def __init__(self, symbols: dict = None):
        self.symbols = symbols or {}
        self._sympy_symbols = {k: sympy.Symbol(k) for k in self.symbols.keys()}
        # Allow int()/floor() in expressions — both map to sympy.floor so they work with Symbols
        self._sympy_locals = {**self._sympy_symbols, 'int': sympy.floor, 'floor': sympy.floor}

    def eval(self, expr):
        """Convert a string expression to a float using sympy."""
        if isinstance(expr, list):
            return expr
        elif not isinstance(expr, str):
            return expr

        try:
            parsed_expr = sympy.sympify(expr, locals=self._sympy_locals)
            if self.symbols:
                symbol_values = {
                    self._sympy_symbols[k]: v for k, v in self.symbols.items()
                }
                result = parsed_expr.subs(symbol_values)
            else:
                result = parsed_expr
            return float(result.evalf())
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expr}': {e}")

class CallParams(BaseModel):
    """Base class for call parameters - extend this for specific use cases"""
    model_config = ConfigDict(extra='allow')  # Allow any field to be added

class DynamicParams(BaseModel):
    name: str
    params: CallParams = Field(default_factory=CallParams)

    def call(self, func, evaluator=None, **overrides):
        """
        Call a function using parameters from this DynamicParams instance, optionally
        evaluating string expressions.
        Args:
            func (callable): The target function to call.
            evaluator (ExpressionEvaluator, optional): If provided, string values in 
                self.params will be evaluated using this evaluator.
            **overrides: Optional parameter overrides for this call.
        Returns:
            The result of func(**final_params), where final_params is a merge of:
            - defaults from func signature
            - values from self.params (evaluated if evaluator provided)
            - explicit overrides in **overrides (unevaluated)
        Notes:
            - self.params remains unmodified; evaluation occurs only during the call.
            - Only parameters that exist in func's signature are passed.
            - Only self.params values are evaluated; overrides are passed as-is.
            - Priority order: defaults < self.params < overrides.
        """
        sig = signature(func)
        # Evaluate self.params that are in func signature
        evaluated = {k: evaluator.eval(v) if evaluator else v
                     for k, v in self.params if k in sig.parameters}
        # Overrides go in unevaluated, highest priority
        valid_params = {**evaluated, **overrides}
        # Get defaults from function signature
        defaults = {n: p.default for n, p in sig.parameters.items()
                    if p.default is not Parameter.empty}
        # Merge defaults with valid params
        final_params = {**defaults, **valid_params}
        return func(**final_params)

    _evaluator: ClassVar[ExpressionEvaluator] = ExpressionEvaluator()

    @model_validator(mode='after')
    def evaluate_expressions(self):
        """Evaluate any expressions in params"""
        def evaluate_value(x):
            if isinstance(x, str):
                try:
                    return self._evaluator.eval(x)
                except:
                    return x
            elif isinstance(x, list):
                return [evaluate_value(item) for item in x]
            return x
        
        params_dict = self.params.model_dump()
        self.params = CallParams(**{key: evaluate_value(value) for key, value in params_dict.items()})
        return self 

@dataclass
class AxisSpec:
    path: tuple   # e.g. ('model', 'input_layer', 'n_nodes')
    values: list


@dataclass
class UniverseSpec:
    key: str | None   # universe name; None = Mother-only
    base_dict: dict   # merged dict used for Params reconstruction
    axes: list = dc_field(default_factory=list)

    def __len__(self):
        return math.prod(len(ax.values) for ax in self.axes) if self.axes else 1


def _extract_axes_dict(d: dict, path: tuple):
    """Recurse into a plain dict field (e.g. model.variables), yielding axes for list values."""
    for k, v in d.items():
        full_path = path + (k,)
        if isinstance(v, list) and v:
            yield AxisSpec(path=full_path, values=[
                x.model_dump() if isinstance(x, BaseModel) else x for x in v
            ] if any(isinstance(x, (BaseModel, dict)) for x in v) else v)
        elif isinstance(v, BaseModel):
            yield from _extract_axes(v, full_path)
        elif isinstance(v, dict):
            yield from _extract_axes_dict(v, full_path)


def _extract_axes(obj, path: tuple = ()):
    """Walk a Pydantic object, yielding one AxisSpec per expandable field."""
    if not isinstance(obj, BaseModel):
        return
    for fname, finfo in obj.model_fields.items():
        if finfo.json_schema_extra and not finfo.json_schema_extra.get('expand', True):
            continue
        if fname == 'multiverse_overrides':
            continue
        value = getattr(obj, fname)
        full_path = path + (fname,)
        if isinstance(value, list) and value:
            if not any(isinstance(x, (BaseModel, dict)) for x in value):
                yield AxisSpec(path=full_path, values=value)
            else:
                yield AxisSpec(path=full_path, values=[
                    x.model_dump() if isinstance(x, BaseModel) else copy.deepcopy(x)
                    for x in value
                ])
        elif isinstance(value, BaseModel):
            yield from _extract_axes(value, full_path)
        elif isinstance(value, dict):
            yield from _extract_axes_dict(value, full_path)


def _build_universe_specs(P) -> list:
    """Return one UniverseSpec per universe (or one for the Mother if no multiverse)."""
    if P.multiverse_overrides and any(P.multiverse_overrides.values()):
        specs = []
        for key in P.multiverse_overrides:
            P_merged = getattr(P.U, key)
            raw_override = P.multiverse_overrides[key] or {}
            raw_gs = (raw_override.get('logging') or {}).get('grid_search') or {}
            uni_run = raw_gs.get('run') if isinstance(raw_gs, dict) else None
            if uni_run is None:
                uni_run = P_merged.L.grid_search.run if P_merged.L.grid_search else ['kqgr']
            if P_merged.L.grid_search is not None:
                new_gs = P_merged.L.grid_search.model_copy(update={'run': uni_run})
                P_merged = P_merged.model_copy(update={
                    'logging': P_merged.L.model_copy(update={'grid_search': new_gs})
                })
            base_dict = P_merged.model_dump()
            base_dict['multiverse_overrides'] = {key: {}}
            specs.append(UniverseSpec(key=key, base_dict=base_dict, axes=list(_extract_axes(P_merged))))
        return specs
    else:
        base_dict = P.model_dump()
        return [UniverseSpec(key=None, base_dict=base_dict, axes=list(_extract_axes(P)))]


def _set_at_path(d: dict, path: tuple, val):
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = val


class LazyParamCombinations:
    """Lazy, len()-aware replacement for generate_param_combinations.

    Construction is O(fields) — no Params objects created until iteration or indexing.
    Supports len(), iteration, and integer/slice indexing.  Integer indexing is O(1) via
    direct mixed-radix decomposition — safe for mid-run resumed grid searches where the
    first claimed index can be far from 0.
    """

    def __init__(self, P):
        self._specs = _build_universe_specs(P)
        self._len   = sum(len(s) for s in self._specs)

    def __len__(self):
        return self._len

    def __iter__(self):
        yield from self._generate()

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._len)
            return [self[i] for i in range(start, stop, step or 1)]
        if idx < 0:
            idx += self._len
        if idx < 0 or idx >= self._len:
            raise IndexError(f'index {idx} out of range')
        offset = idx
        for spec in self._specs:
            spec_len = len(spec)
            if offset < spec_len:
                return self._item_at(spec, offset)
            offset -= spec_len
        raise IndexError(f'index {idx} out of range')

    def _item_at(self, spec, offset):
        """Compute the Params at linear offset within spec without iterating predecessors."""
        from project.boolean_reservoir.code.parameter import Params
        if not spec.axes:
            return Params.model_validate(spec.base_dict)
        values = [ax.values for ax in spec.axes]
        combo = []
        remaining = offset
        for ax_values in reversed(values):  # last axis varies fastest (itertools.product order)
            combo.append(ax_values[remaining % len(ax_values)])
            remaining //= len(ax_values)
        combo.reverse()
        d = copy.deepcopy(spec.base_dict)
        for path, val in zip([ax.path for ax in spec.axes], combo):
            _set_at_path(d, path, val)
        return Params.model_validate(d)

    def _generate(self):
        from project.boolean_reservoir.code.parameter import Params
        for spec in self._specs:
            if not spec.axes:
                yield Params.model_validate(spec.base_dict)
                continue
            paths  = [ax.path for ax in spec.axes]
            values = [ax.values for ax in spec.axes]
            for combo in itertools.product(*values):
                d = copy.deepcopy(spec.base_dict)
                for path, val in zip(paths, combo):
                    _set_at_path(d, path, val)
                yield Params.model_validate(d)


def generate_param_combinations(params):
    return LazyParamCombinations(params)

def expand_ticks(s):
    # Expand pattern repetition: '(123){3}' → '123123123'
    s = re.sub(r'\(([^)]+)\)\{(\d+)\}', lambda m: m.group(1) * int(m.group(2)), s)
    # Expand single-char RLE: '1{3}' → '111'
    s = re.sub(r'(.)\{(\d+)\}', lambda m: m.group(1) * int(m.group(2)), s)
    return s

def get_wiring_by_target(wiring: list, target: str):
    if not wiring:
        return None
    return next((w for w in wiring if w.target == target), None)
