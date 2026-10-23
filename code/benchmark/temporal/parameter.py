from benchmark.utils.parameter import DatasetParameters, KQGRDatasetParams
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import List, Literal, Optional, Tuple, Union
from pathlib import Path

class TemporalHead(BaseModel):
    """One extra target on the same stream: the task over a window, delay steps back."""
    model_config = ConfigDict(extra='forbid')
    task: Literal['density', 'parity']
    window: int
    delay: int

    @property
    def name(self) -> str:
        return f'{self.task}_w{self.window}_d{self.delay}'


class TemporalDatasetParams(DatasetParameters):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    name: Literal["temporal"] = "temporal"
    init: str = Field('Temporal', json_schema_extra={'expand': False})
    dimensions: Union[int, List[int]] = Field(1, description="Number of independent bit streams")
    task: str = Field('density', description="two options: density or parity")
    bits: Union[int, List[int]] = Field(10, description="length of the bit stream")
    window: Union[int, List[int]] = Field(5, description="size of the window for temporal data")
    delay: Union[int, List[int]] = Field(0, description="delay; shifts window from right to left; higher delay is easier as bits are processed right to left")
    sampling_mode: Union[str, List[str]] = Field( 'random',
    description="'random': random bit patterns with repetition allowed. "
                "'exhaustive': enumerate patterns 0 to 2^bits-1, taking first 'samples' patterns (or cycling if samples > 2^bits)"
    )
    washout: int = Field(0, ge=0, description=(
        "Random bits fed before the `bits` scored ones, so every sample reaches its windows from its own "
        "state instead of the shared reset state (at 0, windows near the stream start are read off the "
        "reset transient: primacy). Drawn from their own seeded stream; never scored, and skipped by the "
        "time-series readout. Exhaustive sampling stays exhaustive over the scored bits."))
    heads: Optional[Tuple[TemporalHead, ...]] = Field(
        None, json_schema_extra={'expand': False}, description=(
            "Extra targets read from the SAME streams and reservoir run, one readout each, logged per "
            "head in logging.targets (glm.ReadoutGroup, un-indexed); y stays task/window/delay. Computed "
            "from x, so the stored data does not change. A tuple, like path_integration's steps_mix, so "
            "it is neither a grid axis nor a block on path generation."))

    @model_validator(mode='after')
    def _heads_fit_the_stream(self):
        if self.heads and not self.has_list_in_a_field():
            need = max(h.window + h.delay for h in self.heads)
            if self.bits < need:
                raise ValueError(f'dataset.heads need bits >= window + delay = {need}, got bits={self.bits}')
        return self

    @model_validator(mode='after')
    def update_path_after_init(self):
        self.path = self._generate_path()
        return self

    def _generate_path(self):
        if self.has_list_in_a_field(): # not yet expanded (doesnt check recursively)
            return

        return (Path('data/temporal')
            / self.task
            / f'{self.dimensions}D'
            / f's-{self.sampling_mode}'
            / f'b-{self.bits}'
            / f'w-{self.window}'
            / f'd-{self.delay}'
            / (f'wo-{self.washout}' if self.washout else '')
            / self._base_data_path()
            / 'dataset.npz'
        )
    
    def update_path(self):
        self.path = self._generate_path()

    @property
    def init_obj(self):
        from project.temporal.code.dataset_init import TemporalDatasetInit
        return TemporalDatasetInit()


class KQGRTemporalDatasetParams(KQGRDatasetParams, TemporalDatasetParams):
    """
    Temporal dataset augmented with KQGR fields (tau, evaluation).

    Hijacks TemporalDensityDataset for data generation but only uses x values (input
    bitstreams) - y values (labels) are ignored since KQ/GR metrics evaluate reservoir
    kernel quality and generalization capability, not task performance.

    tau/evaluation_mode control GR metric: during encoding, tau bits per feature are set
    identical across all samples to test generalization under reduced input diversity.
    """