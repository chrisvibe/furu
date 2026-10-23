from typing import Union, List, Literal, Optional
from pydantic import Field, model_validator, ConfigDict
from pathlib import Path
from benchmark.utils.parameter import DatasetParameters, KQGRDatasetParams


class NeuroGymDatasetParams(DatasetParameters):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    name: Literal["neurogym"] = "neurogym"
    init: str = Field('NeuroGym', json_schema_extra={'expand': False})
    task: Union[str, List[str]] = Field('GoNogo', description="NeuroGym task name (without -v0 suffix)")
    neurogym_seed: Union[int, List[int]] = Field(0, description="Seed passed to the NeuroGym API for trial generation")

    @model_validator(mode='after')
    def update_path_after_init(self):
        self.path = self._generate_path()
        return self

    @property
    def init_obj(self):
        from project.neurogym.code.dataset_init import NeuroGymDatasetInit
        return NeuroGymDatasetInit()

    def _generate_path(self) -> Optional[Path]:
        if self.has_list_in_a_field():
            return None
        return (
            Path('data/neurogym')
            / self.task
            / f'seed{self.neurogym_seed}_n{self.samples}.pt'
        )

    def update_path(self):
        self.path = self._generate_path()


class KQGRNeuroGymDatasetParams(KQGRDatasetParams, NeuroGymDatasetParams):
    """NeuroGym dataset augmented with KQGR capacity metric fields."""
