from typing import Union, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pathlib import Path
import math
from benchmark.utils.parameter import DatasetParameters, KQGRDatasetParams
from benchmark.path_integration.constrained_foraging_path import (
    LevyFlightStrategy, SimpleRandomWalkStrategy, PhysicsWalkStrategy,
    PolygonBoundary, IntervalBoundary, NoBoundary,
    SoftBoundary, LinearSoftBoundary, QuadraticSoftBoundary, CubicSoftBoundary,
    generate_polygon_points, stretch_polygon
)
from project.boolean_reservoir.code.utils.param_utils import DynamicParams, ExpressionEvaluator
from project.boolean_reservoir.code.encoders import ENCODING_NAMES

def strategy_factory(p: DynamicParams):
    """Factory function to create strategy objects"""
    strategy_map = {
        'LevyFlightStrategy': LevyFlightStrategy,
        'SimpleRandomWalkStrategy': SimpleRandomWalkStrategy,
        'PhysicsWalkStrategy': PhysicsWalkStrategy,
    }

    if p.name not in strategy_map:
        raise ValueError(f"Unsupported strategy type: {p.name}. Available: {list(strategy_map.keys())}")

    return p.call(strategy_map[p.name])
    
def boundary_factory(p: DynamicParams):
    """Factory function to create boundary objects"""
    evaluator = ExpressionEvaluator(symbols={'pi': math.pi})
    
    if p.name == 'PolygonBoundary':
        points = p.call(generate_polygon_points, evaluator=evaluator)
        boundary = PolygonBoundary(points=points)
        return p.call(stretch_polygon, evaluator=evaluator, boundary=boundary)
    
    elif p.name == 'IntervalBoundary':
        return p.call(IntervalBoundary)
    
    elif p.name == 'NoBoundary':
        return p.call(NoBoundary)

    elif p.name == 'LinearSoftBoundary':
        return p.call(LinearSoftBoundary)

    elif p.name == 'QuadraticSoftBoundary':
        return p.call(QuadraticSoftBoundary)

    elif p.name == 'CubicSoftBoundary':
        return p.call(CubicSoftBoundary)

    else:
        raise ValueError(f"Unsupported boundary type: {p.name}. Available: ['PolygonBoundary', 'IntervalBoundary', 'NoBoundary', 'LinearSoftBoundary', 'QuadraticSoftBoundary', 'CubicSoftBoundary']")


class QueryParams(BaseModel):
    """Retroactive query: after the walk, 'where were you after step q?' — target
    positions[q]. The query rides a second input channel (its own input nodes, silent during
    the walk) encoded by the same encoder as the walk; the fields mirror steps_mix and
    the input layer's vocabulary."""
    model_config = ConfigDict(extra='forbid')
    mix: Optional[Tuple[int, ...]] = Field(
        None,
        json_schema_extra={'expand': False},
        description=(
            "Query steps q, stratified over these values (each in 1..steps). A tuple, like "
            "steps_mix, so it never grid-expands. None = the query channel exists but no query "
            "is asked (the KQGR pool: same model width, silent query channel)."))
    encoding: Union[str, List[str]] = Field(
        'tally', description="Query encoding — any registered encoder, as input_layer.encoding "
                             "(tally: well-separated queries; base2 is the weak control).")
    redundancy: Union[int, List[int]] = Field(
        7, description="Query copies per encoded bit, as input_layer.redundancy: the query's footprint "
                       "is resolution x redundancy input nodes (each drives one reservoir node).")
    hold: Union[int, List[int]] = Field(
        0, description="Input steps after the query step during which every input is held (written "
                       "as the perturbation's identity input), so the reservoir runs on before the "
                       "readout; each step ends with input_layer.ticks LUT updates like any other, so hold "
                       "adds hold x ticks LUT updates (exactly free ticks for the final state; the "
                       "summed-states readout also adds each held step's state). The engine runs one "
                       "tick count per step, so query-only ticks are expressed as held steps. "
                       "0 = the readout follows the query step directly.")

    @field_validator('encoding')
    @classmethod
    def _known_encoding(cls, v):
        bad = [e for e in (v if isinstance(v, list) else [v]) if e not in ENCODING_NAMES]
        if bad:
            raise ValueError(f"unknown query encoding {bad}; valid: {list(ENCODING_NAMES)}")
        return v

    @field_validator('redundancy')
    @classmethod
    def _positive(cls, v):
        if any(x < 1 for x in (v if isinstance(v, list) else [v])):
            raise ValueError("query redundancy must be >= 1")
        return v

    @field_validator('hold')
    @classmethod
    def _non_negative(cls, v):
        if any(x < 0 for x in (v if isinstance(v, list) else [v])):
            raise ValueError("query hold must be >= 0")
        return v


class PathIntegrationDatasetParams(DatasetParameters):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    name: Literal["path_integration"] = "path_integration"
    init: str = Field('PathIntegration', json_schema_extra={'expand': False})
    coordinate: Union[str, List[str]] = Field('cartesian', description='Coordinate system for inputs: cartesian or polar')
    output_coordinate: Union[str, List[str]] = Field('cartesian', description='Coordinate system for target output: cartesian or polar')
    mode: Union[str, List[str]] = Field('displacement', description="Input representation from optimistic trajectory: 'displacement' (sum=p_final), 'velocity', or 'acceleration' (a_ext+a_bnd_opt)")
    dimensions: Union[int, List[int]] = Field(2, description="Number of dimensions")
    steps: int = Field(
        10,
        description=(
            "Single fixed step count per dataset. Multi-length datasets use steps_mix "
            "(the only multi-length form — list values in YAML remain grid axes everywhere, "
            "so the per-dataset length spec is deliberately a separate field; steps axes for "
            "grid search go through the multiverse."))
    steps_mix: Optional[Tuple[int, ...]] = Field(
        None,
        json_schema_extra={'expand': False},
        description=(
            "If set: per-sample step counts drawn with EXACT per-value counts (stratified) "
            "from these values; x is zero-padded to max(steps_mix) and each sample's true "
            "length rides in the dataset 'n' array. Mutually exclusive with steps — when "
            "steps_mix is set, steps is dropped and ignored. Deliberately a TUPLE, not a "
            "list: pydantic coerces YAML lists to tuples, while param_utils._extract_axes "
            "grid-expands every list field and has_list_in_a_field() blocks path generation "
            "(the list form of steps stays a grid axis; the tuple typing dodges both mechanisms)."
        ))
    query: Optional[QueryParams] = Field(
        None, description="Retroactive-query task (see QueryParams); exclusive with steps_mix.")

    strategy: Union[DynamicParams, List[DynamicParams]] = Field(
        default=DynamicParams(
            name='SimpleRandomWalkStrategy',
        ),
        description="Strategy configuration"
    )

    boundary: Union[DynamicParams, List[DynamicParams]] = Field(
        default=DynamicParams(
            name='PolygonBoundary',
            params={'n_sides': 4, 'radius': 1.0, 'rotation': math.pi/4, 'stretch_x': 1.0, 'stretch_y': 1.0}
        ),
        description="Boundary configuration"
    )

    @model_validator(mode='before')
    def _drop_steps_when_steps_mix_set(cls, values):
        # steps_mix wins over steps (deliberate drop, not a raise,
        # because a strict mutual-exclusion validator would false-positive on every
        # generate_param_combinations revalidation: _item_at re-validates from
        # P.model_dump(), which ALWAYS carries both keys (steps at its default).
        if isinstance(values, dict) and values.get('steps_mix') is not None:
            values.pop('steps', None)
        return values

    @model_validator(mode='after')
    def _check_query(self):
        if self.query is None:
            return self
        if self.steps_mix is not None:
            raise ValueError("query and steps_mix are mutually exclusive")
        bad = [q for q in (self.query.mix or ()) if not 1 <= q <= self.steps]
        if bad:
            raise ValueError(f"query.mix values must lie in 1..steps={self.steps}, got {bad}")
        return self

    @property
    def query_mix(self):
        """The query steps actually asked (None: no query, or a silent query channel)."""
        return self.query.mix if self.query else None

    @model_validator(mode='after')
    def update_path_after_init(self):
        self.path = self._generate_path()
        return self

    @property
    def strategy_obj(self):
        """Property that constructs the strategy object using factory"""
        return strategy_factory(self.strategy)
    
    @property
    def boundary_obj(self):
        """Property that constructs the boundary object using factory"""
        return boundary_factory(self.boundary)

    @property
    def init_obj(self):
        from project.path_integration.code.dataset_init import PathIntegrationDatasetInit
        return PathIntegrationDatasetInit()
        
    def _generate_path(self) -> Path:
        """Generate path based on parameters"""
        if self.has_list_in_a_field(): # not yet expanded (doesnt check recursively)
            return

        strategy_str = self.strategy.name[:3].upper()  # First 3 chars of name
        strategy_hash = self._hash_params(self.strategy.params)
        resolution_bits = self.strategy.params.model_dump().get('resolution_bits', None)
        res_str = f'res-{resolution_bits}' if resolution_bits is not None else 'res-cont'

        boundary_str = self.boundary.name[:3].upper()  # First 3 chars of name
        boundary_hash = self._hash_params(self.boundary.params)

        # Full hash of all data-generating fields — catches any param not explicit in the path above
        full_hash_dict = {
            'coordinate': self.coordinate,
            'output_coordinate': self.output_coordinate,
            'dimensions': self.dimensions,
            'steps': self.steps,
            'mode': self.mode,
            'strategy': self.strategy.model_dump(),
            'boundary': self.boundary.model_dump(),
            'samples': self.samples,
            'seed': self.seed,
            'reset': self.reset,
        }
        # Per-sample mixes enter the hash ONLY when set (an unconditional None entry would
        # move every cached npz). Of the query only the mix shapes the raw data; its
        # encoding / redundancy / hold are processing, so their arms share one npz.
        for mix in ('steps_mix', 'query_mix'):
            if getattr(self, mix):
                full_hash_dict[mix] = sorted(getattr(self, mix))
        full_hash = self._hash_dict(full_hash_dict, n_chars=8)

        # s-mix segment lists the sorted mix; 'steps' only when no mix (mutually exclusive)
        s_segment = (f's-mix-{"-".join(map(str, sorted(self.steps_mix)))}'
                     if self.steps_mix else f's-{self.steps}')
        if self.query_mix:
            s_segment += f'_q-{"-".join(map(str, sorted(self.query_mix)))}'
        return (
            Path('data/path_integration')
            / f'c-{self.coordinate}'
            / f'oc-{self.output_coordinate}'
            / f'd-{self.dimensions}'
            / s_segment
            / f'm-{self.mode}'
            / strategy_str
            / strategy_hash
            / res_str
            / boundary_str
            / boundary_hash
            / self._base_data_path()
            / full_hash
            / 'dataset.npz'
        )

    def update_path(self):
        """Update the path based on current parameters"""
        self.path = self._generate_path()


class KQGRPathIntegrationDatasetParams(KQGRDatasetParams, PathIntegrationDatasetParams):
    """PathIntegration dataset augmented with KQGR fields (tau, evaluation).

    steps_mix / query are ALLOWED here: the train universe carries them
    (training) alongside tau/tau_mode (kqgr freeze). The kqgr probe never
    consumes that framing — build_kqgr_probe starts from kqgr_pool(), so the
    probe only ever sees plain fixed-length walks (with a silent query channel,
    so the model keeps its width).
    """

    def kqgr_pool(self) -> 'KQGRPathIntegrationDatasetParams':
        # Defined HERE, not on PathIntegrationDatasetParams: KQGRDatasetParams
        # precedes it in the MRO and its default would shadow a base override.
        D = self.model_copy(deep=True)
        if D.steps_mix:
            D.steps = max(D.steps_mix)   # the widest training window
            D.steps_mix = None
        if D.query:                      # the query is task framing, not input statistics;
            D.query = D.query.model_copy(update={'mix': None})   # the channel stays (model width)
        D.update_path()
        return D

if __name__ == '__main__':
    import yaml
    
    yaml_content = """
    dimensions: [2, 3]
    steps: 10
    strategy:
      name: LevyFlightStrategy
      params:
        alpha: [2.0, 3.0]
    boundary:
      name: PolygonBoundary
      params:
        n_sides: [4, 6]
        radius: 0.2
        rotation: pi/4
    samples: 64
    """
    config = yaml.safe_load(yaml_content)
    p = PathIntegrationDatasetParams(**config)
    
    print(f"Strategy: {p.strategy}")
    print(f"Boundary: {p.boundary}")
    
    # Test with generate_param_combinations
    try:
        from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
        p_list = generate_param_combinations(p)
        
        print(f"\nGenerated {len(p_list)} combinations")
        for i, combo in enumerate(p_list[:2]):
            print(f"\n--- Combination {i+1} ---")
            print(f"  dimensions: {combo.dimensions}")
            print(f"  strategy.params: {combo.strategy.params}")
            print(f"  boundary.params: {combo.boundary.params}")
            print(f"  strategy_obj: {combo.strategy_obj}")
            print(f"  boundary_obj: {combo.boundary_obj}")
    except ImportError as e:
        print(f"\nCould not test with generate_param_combinations: {e}")
