import re
from pydantic import BaseModel, Field, model_validator, field_validator, PrivateAttr, ConfigDict
import yaml
from typing import Annotated, Dict, Any, List, Union, Optional, Literal
from pathlib import Path
from benchmark.path_integration.parameter import PathIntegrationDatasetParams, KQGRPathIntegrationDatasetParams
from benchmark.temporal.parameter import TemporalDatasetParams, KQGRTemporalDatasetParams
from benchmark.neurogym.parameter import NeuroGymDatasetParams, KQGRNeuroGymDatasetParams
from project.boolean_reservoir.code.utils.param_utils import pydantic_init, calculate_w_broadcasting, DynamicParams, CallParams, ExpressionEvaluator, expand_ticks
import copy
from shutil import copy as shutil_copy
pydantic_init()


def deep_merge(base: dict, override: dict) -> dict:
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)

    # Type-Swap Protector: if dataset type changes, discard base entirely
    if 'name' in base and 'name' in override and base['name'] != override['name']:
        return copy.deepcopy(override)

    merged = copy.deepcopy(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


EncodingName = Literal['base2', 'base2_lsb', 'primes', 'tally', 'binary_embedding', 'gray', 'rate',
                       'delta_sigma', 'delta_sigma_free', 'base2_delta_sigma', 'phase', 'hamming_weight']


class InputParams(BaseModel):
    seed: Optional[int] = Field(
        None, description="Random seed, None disables seed")
    perturbation: Union[Literal['xor', 'and', 'or', 'override'], List[Literal['xor', 'and', 'or', 'override']]] = Field(
        'xor', description=(
            "How the incoming input bit is combined with the current I-node state. "
            "'override': I[i] ← input[i] (pure replacement). "
            "'xor': I[i] ← I[i] ⊕ input[i]. "
            "'and': I[i] ← I[i] & input[i]. "
            "'or':  I[i] ← I[i] | input[i]. "
            "Only I-nodes are written; w_bi defines which input bits map to which I-nodes."
        ))
    encoding: Union[EncodingName, List[EncodingName]] = Field(
        'base2', description="Binary encoding type")
    interleaving: Union[int, List[int]] = Field(
        0, description="Multidimensionsional weaving of inputs, int dictates group size. n=1: abc, def -> ad, be, cf -> adb, ecf | n=2: abcd, efgh -> ab, ef, cd, gh -> abef, cdgh")
    n_nodes: Optional[Union[int, List[int]]] = Field(
        None, description="Number of input nodes; I")
    features: Union[int, List[int]] = Field(
        None, description=(
            "Number of input features before binary encoding. "
            "Defaults chunks=features so each feature occupies one sequential perturbation sub-step."
        ))
    bits: Optional[Union[int, List[int]]] = Field(
        None, description="Total bits after encoding (= features × resolution × redundancy + query_bits = chunks × chunk_size).")
    query_bits: int = Field(
        0, description=(
            "Bits of a query channel appended after the encoded features (its own input nodes). "
            "Derived by Params from dataset.query (resolution × query.redundancy) — not set by hand. "
            "A query channel shares the single perturbation sub-step, so it requires chunks = 1."))
    resolution: Optional[Union[int, List[int]]] = Field(
        None, description="Bits per dimension before redundancy")
    redundancy: Union[int, List[int]] = Field(
        1, description="Redundancy factor of resolution")
    chunks: Optional[Union[int, List[int]]] = Field(
        None, description=(
            "Number of sequential perturbation sub-steps per forward step (c in x.shape=(m,s,c,b)). "
            "Defaults to features. Each chunk maps to a contiguous band of w_bi rows and is "
            "followed by ticks[ci] reservoir ticks."
        ))
    chunk_size: Optional[Union[int, List[int]]] = Field(
        None, description=(
            "Bits processed simultaneously per chunk (b in x.shape=(m,s,c,b); = bits // chunks). "
            "Defaults to resolution × redundancy per feature."
        ))
    permute: Optional[Union[str, List[str]]] = Field(
        None,
        description="Permutation of (features, redundancy, resolution) dims applied during "
                    "redundancy expansion. E.g. '021' interleaves redundancy copies so each "
                    "chunk contains K identical bits (parallel redundancy). None = block repeat.")
    ticks: Optional[Union[str, List[str]]] = Field(
        None, description=(
            "Reservoir ticks to run after each chunk's perturbation. "
            "Scalar '1' gives one tick per chunk; '1{n}' expands to n ones; "
            "'2{3}1{2}' gives three 2-tick chunks then two 1-tick chunks. "
            "Total ticks per forward call = s × sum(ticks_expanded). "
            "With features=1, bits=1, ticks='1': one tick per forward step."
        ))

    @model_validator(mode='after')
    def calculate_bits(self):
        """Calculate bits from other parameters if not set"""
        # Calculate from resolution and features
        if (self.bits is None and
            not isinstance(self.features, list) and
            not isinstance(self.resolution, list) and
                not isinstance(self.redundancy, list)):

            if self.resolution is not None and self.features is not None:
                self.bits = self.features * self.resolution * self.redundancy + self.query_bits

        # Also handle the reverse: if bits is set, calculate resolution
        elif (self.bits is not None and
              self.resolution is None and
              not isinstance(self.bits, list) and
              not isinstance(self.features, list) and
              not isinstance(self.redundancy, list)):

            if self.features is not None:
                self.resolution = self.bits // (self.features *
                                                self.redundancy)

        return self

    # Both set - calculate bits if not set
    @model_validator(mode='after')
    def handle_chunking(self):
        """Handle chunks and chunk_size bidirectionally"""
        if self.query_bits and not isinstance(self.bits, list):
            if self.chunks not in (None, 1):
                raise ValueError(f"a query channel needs chunks = 1, got {self.chunks}")
            self.chunks, self.chunk_size = 1, self.bits
            return self
        if (not isinstance(self.chunks, list) and
                not isinstance(self.chunk_size, list)):

            if self.chunks is not None and self.chunk_size is not None:
                if self.bits is None:
                    self.bits = self.chunks * self.chunk_size
            elif self.bits is not None and not isinstance(self.bits, list):
                if self.chunks is not None:
                    self.chunk_size = self.bits // self.chunks
                elif self.chunk_size is not None:
                    self.chunks = self.bits // self.chunk_size
                elif self.features is not None and not isinstance(self.features, list):
                    if (self.permute is not None and not isinstance(self.permute, list)
                            and not isinstance(self.redundancy, list)):
                        self.chunk_size = self.redundancy
                        self.chunks = self.bits // self.chunk_size
                    else:
                        self.chunks = self.features
                        if self.resolution is not None and not isinstance(self.resolution, list):
                            self.chunk_size = self.resolution * self.redundancy
        return self

    @model_validator(mode='after')
    def default_n_nodes(self):
        if self.n_nodes is None:
            self.n_nodes = calculate_w_broadcasting(
                lambda x, y: x, self.bits, None)
        return self

    @model_validator(mode='after')
    def handle_no_input(self):
        """When n_nodes=0 (autonomous CA, no input), fill in features/chunks/chunk_size/bits
        so the Julia engine initialises correctly (c=1, k_bits=1, N_I=0) and saved
        parameters round-trip through YAML without validation errors.
        The perturbation loop runs for n in 1:0, so no actual input is injected."""
        if self.n_nodes == 0 and self.chunks is None:
            self.features   = 1
            self.chunks     = 1
            self.chunk_size = 1
            self.bits       = 1
        return self

    @model_validator(mode='after')
    def set_default_ticks(self):
        if self.chunks is None or isinstance(self.chunks, list):
            return self
        if self.ticks is None:
            self.ticks = f'1{{{self.chunks}}}'
        return self

    @property
    def ticks_expanded(self):
        if self.ticks is None:
            return '1' * self.chunks if self.chunks else None

        expanded = expand_ticks(self.ticks)

        # Repeat/truncate to match chunks
        if self.chunks and len(expanded) != self.chunks:
            if len(expanded) < self.chunks:
                # Repeat pattern to fill
                expanded = (expanded * (self.chunks //
                            len(expanded) + 1))[:self.chunks]
            else:
                # Truncate
                expanded = expanded[:self.chunks]

        return expanded

    @property # TODO remove (was misspelled, should be perturbation)
    def pertubation(self):
        return self.perturbation


_INIT_PATTERN = re.compile(
    r'^(random|zeros|ones|every_other)(-warmup(-\d+)?)?$'
)


class ReservoirParams(BaseModel):
    seed: Optional[int] = Field(
        None, description="Random seed, None disables seed")
    n_nodes: Optional[Union[int, List[int]]] = Field(
        None, description="Number of reservoir nodes (R) excluding input nodes (I)")
    p: Union[float, List[float]] = Field(
        0.5, description="Probability for 1 in LUT (look up table)")
    reset: Optional[Union[bool, List[bool]]] = Field(
        True, description=(
            "If True, reset states_parallel to initial_states at the start of each forward() call. "
            "Set False for tasks where reservoir state must persist across calls "
            "(e.g. path integration with continuous trajectories)."
        ))
    init: Union[str, List[str]] = Field(
        'random',
        description=(
            "Init strategy: base or base-warmup or base-warmup-N. "
            "Base: random | zeros | ones | every_other. "
            "'-warmup' warms up for ticks[0] ticks; '-warmup-N' warms up for exactly N ticks."
        )
    )

    @field_validator('init', mode='before')
    @classmethod
    def _validate_init(cls, v):
        for item in (v if isinstance(v, list) else [v]):
            if not _INIT_PATTERN.match(str(item)):
                raise ValueError(
                    f"Invalid init strategy '{item}'. "
                    f"Expected base[-warmup[-N]] where base ∈ "
                    f"{{random, zeros, ones, every_other}}"
                )
        return v


# TODO add w_out and distribution like in input_layer. atm we assume full readout of R
class OutputParams(BaseModel):
    seed: Optional[int] = Field(
        None, description="Random seed, None disables seed")
    n_nodes: Union[int, List[int]] = Field(
        1, description="Dimension of output data")
    activation: Optional[Union[str, List[str]]] = Field(
        None, description=(
            "Activation applied to the linear readout output. "
            "Also controls what forward() returns and how O-node states are set: "
            "None → forward() returns raw linear readout, O-nodes set to (raw > 0.5); "
            "'sigmoid' → forward() returns sigmoid output, O-nodes set to (sigmoid > 0.5). "
            "Default None (linear) is correct for regression (the 'gaussian' GLM), for "
            "MSE reporting on classification, and for BCEWithLogitsLoss (sigmoid is applied "
            "inside the loss). "
            "Use 'sigmoid' only when the criterion is BCELoss, which expects inputs in (0,1). "
            "With optim='bernoulli' (Bernoulli/logit GLM fit) None is "
            "correct: fit and decision rule (P(y=1|x)>0.5 ⟺ logit>0) operate on the raw logit, "
            "and the fitter applies the sigmoid itself for the reported CE loss. "
            "WARNING: combining 'sigmoid' with optim='gaussian' is incorrect — "
            "the Gaussian/identity GLM minimises MSE on the pre-sigmoid output so the training "
            "objective and evaluation are mismatched; use None (linear readout) whenever the "
            "Gaussian family is in the optimizer sweep."
        ))
    encoding: Optional[Union[Literal['binary', 'bipolar'], List[Literal['binary', 'bipolar']]]] = Field(
        "binary", description="Encoding of reservoir states for readout: 'binary'={0,1}, 'bipolar'={-1,1}")
    mode: Optional[Union[Literal['classification', 'time-series'], List[Literal['classification', 'time-series']]]] = Field(
        "classification", description="'classification': single readout from final state. 'time-series': sum readouts from all steps, activation applied once on the sum.")
    heads: Optional[List[Literal['classification', 'time-series']]] = Field(
        None, json_schema_extra={'expand': False}, description=(
            "Readout heads fitted on ONE reservoir run (shared forward passes and seeds: a "
            "paired comparison, ~40% cheaper than a mode grid axis). Each head is fitted "
            "exactly as a single-mode run of that mode; results land per head in "
            "logging.heads, and loaders expand them into one row per head. Exclusive with a "
            "mode LIST (the grid axis = separate runs, independent seeds); mode is then the "
            "model's own readout and defaults to the first head."))

    @model_validator(mode='after')
    def _heads(self):
        if self.heads is None:
            return self
        if isinstance(self.mode, list):
            raise ValueError("output_layer: use either heads (one shared run) or a mode list "
                             "(a grid axis of separate runs), not both")
        if not self.heads or len(set(self.heads)) != len(self.heads):
            raise ValueError(f"output_layer.heads must be distinct modes, got {self.heads}")
        if self.mode not in self.heads:
            self.mode = self.heads[0]
        return self


class TrainingParams(BaseModel):
    seed: Optional[int] = Field(
        None, description="Random seed, None disables seed")
    batch_size: Union[int, List[int]] = Field(
        128, description="Number of samples per forward pass")
    criterion: Optional[Union[str, List[str]]] = Field(
        'MSE', description="ML criterion, fex MSE, BCE")
    epochs: Union[int, List[int]] = Field(100, description="Number of epochs")
    accuracy_threshold: Union[float, List[float]] = Field(
        0.5, description="Threshold for generic accuracy metric")
    evaluation: Optional[str] = Field(
        'test', description="test, dev, train etc")
    shuffle: bool = Field(True, description="Shuffle dataset")
    drop_last: bool = Field(True, description="Drop last")
    optim: Union[DynamicParams, List[DynamicParams]] = Field(
        # NOTE: the 'adam' default is a legacy stub that now RAISES at fit time
        # (normalize_family rejects it) — every real config sets optim explicitly.
        default=DynamicParams(name='adam', params={
                              'lr': 1e-3, 'weight_decay': 1e-3}),
        description="Readout fit: GLM family name — 'gaussian' "
                    "(Gaussian/identity GLM, closed form) or 'bernoulli' "
                    "(Bernoulli/logit GLM, Newton–IRLS). One name per family "
                    ": the historical 'ridge'/'logistic' are "
                    "retired and raise at fit time. "
                    "params.alpha = the ONE scale-aware strength; 0 = "
                    "minimum-norm OLS"
    )
    accuracy: Optional[str] = Field(
        None,
        description="Accuracy function: 'Euclidean' or 'Boolean'. None infers from dataset type.",
        json_schema_extra={'expand': False}
    )

    @property
    def accuracy_obj(self):
        from project.boolean_reservoir.code.train_model import EuclideanDistanceAccuracy, BooleanAccuracy
        mapping = {'Euclidean': EuclideanDistanceAccuracy, 'Boolean': BooleanAccuracy}
        if self.accuracy not in mapping:
            raise ValueError(f"Unknown accuracy '{self.accuracy}'. Available: {list(mapping)}")
        return mapping[self.accuracy]().accuracy


class _WiringBase(BaseModel):
    """One block of the patchwork (see wiring.py): source layer(s) → target layer(s).

    Layers: B (input bits), I, R, O, joined with '+' (e.g. source 'I+R'). B only ever
    feeds I, as its own block 'B' → 'I'. Selectors (ChainedSelector chains) restrict the
    block to some source rows / target columns, indexed over the layers as written; the
    block is drawn on the selected nodes only."""
    model_config = ConfigDict(extra='forbid')
    source: str
    target: str
    source_selector: Optional[str] = None
    target_selector: Optional[str] = None

    @field_validator('source', 'target')
    @classmethod
    def _layers(cls, v: str) -> str:
        parts = v.split('+')
        if not all(p in ('B', 'I', 'R', 'O') for p in parts) or len(set(parts)) != len(parts):
            raise ValueError(f"{v!r}: expected distinct layers from B, I, R, O joined with '+' (e.g. 'I+R')")
        return v

    @model_validator(mode='after')
    def _b_only_feeds_i(self):
        if 'B' in self.target.split('+') or ('B' in self.source.split('+') and (self.source, self.target) != ('B', 'I')):
            raise ValueError(f"{self.source}→{self.target}: the input bits B only feed I, as the block 'B' → 'I'")
        return self


class _RandomWiringBase(_WiringBase):
    k_avg: Union[float, str]
    k_min: Union[int, str] = 0
    k_max: Union[int, str] = 'n_nodes'
    self_loops: Optional[Union[float, str]] = None
    mode: Union[str, List[str]] = 'heterogeneous'  # literal, variable reference (e.g. "R_mode"), or grid-search list
    degree: Literal['in', 'out'] = 'in'


class GNMWiring(_RandomWiringBase):
    """G(n,m) random wiring: exactly k_avg*n_nodes edges distributed via pigeonhole."""
    type: Literal['gnm'] = 'gnm'


class GNPWiring(_RandomWiringBase):
    """G(n,p) random wiring: each node's degree drawn from Binomial(k_max-k_min, p)."""
    type: Literal['gnp'] = 'gnp'


# Keep alias for any code that still imports RandomWiring directly
RandomWiring = GNMWiring


class NetworkXWiring(_WiringBase):
    """Wraps any NetworkX graph generator as reservoir wiring.

    Same-layer (source == target): uses nx.<graph.name>(n=tgt_size, **graph.params).
    Cross-layer (source != target): uses nx.bipartite.<graph.name>(n=src_size, m=tgt_size, **graph.params)
    and extracts the biadjacency matrix.

    graph.params values can be numeric literals or structural-variable expressions
    (I_n_nodes, R_n_nodes, src_size, tgt_size, n_nodes, and non-string model.variables).
    """
    type: Literal['networkx'] = 'networkx'
    graph: DynamicParams
    directed: bool = True  # only applies to same-layer; bipartite extraction is always directed
    k_max: Optional[int] = None  # clip in-degree to this value; required for scale-free graphs where hub nodes would overflow the LUT


class ExplicitWiring(_WiringBase):
    """Load a pre-existing graph via any NetworkX reader or converter.

    graph.name: a top-level networkx function, e.g. 'read_edgelist', 'read_adjlist',
                'read_graphml', 'read_gml'. graph.params: its kwargs (e.g. path).
    The loaded graph must have exactly tgt_size nodes (for same-layer wiring)
    or src_size + tgt_size nodes (cross-layer, bipartite extraction).
    graph.params values support structural-variable expressions.
    """
    type: Literal['explicit'] = 'explicit'
    graph: DynamicParams
    directed: bool = True


class PatternWiring(_WiringBase):
    """A fixed mapping (BipartiteMappingStrategy): 'identity', 'zeroes', 'stub-…', 'in-…', 'out-…'.
    Degree-based patterns take their k_avg from model.variables['R_k_avg'], not from the block."""
    type: Literal['pattern'] = 'pattern'
    pattern: str  # 'identity' | 'zeroes'


WiringConfig = Annotated[
    Union[GNMWiring, GNPWiring, NetworkXWiring, ExplicitWiring, PatternWiring],
    Field(discriminator='type')
]


class ModelParams(BaseModel):
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Named scalars used in wiring expressions (e.g. R_k_avg, R_k_min). "
                    "List values trigger grid-search expansion."
    )
    input_layer: InputParams
    reservoir_layer: ReservoirParams
    output_layer: OutputParams
    training: TrainingParams
    wiring: List[WiringConfig] = Field(
        default_factory=list, json_schema_extra={'expand': False}
    )

    @property
    def I(self):
        return self.input_layer

    @property
    def R(self):
        return self.reservoir_layer

    @property
    def O(self):
        return self.output_layer

    @property
    def T(self):
        return self.training

    @property
    def n_nodes(self):
        return self.I.n_nodes + self.R.n_nodes + self.O.n_nodes


class GridSearchParams(BaseModel):
    seed: Optional[int] = Field(
        None, description="Random seed, None disables seed")
    n_samples: Optional[int] = Field(
        1, ge=1, description="Number of samples per configuration in grid search")
    seeding: Literal['index', 'content'] = Field(
        'index', description=(
            "How a job's seed is derived (provenance.py). 'index': from the grid seed, the config index i and the "
            "sample j, so seeds move when the grid's lists change. 'content': from the grid seed, the reservoir's "
            "content and j, so the same reservoir comes back in any grid that contains it."))
    run: List[str] = Field(
        default=['train'],
        description=(
            "Per-config run control: 'train' runs training, 'kqgr' runs KQGR metrics. "
            "Defaults to ['train']. Override per-universe via logging.grid_search.run."
        ),
        json_schema_extra={'expand': False}
    )


class HistoryParams(BaseModel):
    record: Optional[bool] = Field(
        False, description="Reservoir dynamics state recording")
    buffer_size: Optional[int] = Field(
        64, description="Number of batched snapshots per output file")
    save_path: Optional[Path] = Field(
        Path('out/history'), description="Where model is saved when recording history")
    persist_to_disk: Optional[bool] = Field(False, description="Save state history in files not RAM")

class ReservoirMetrics(BaseModel):
    spectral_radius: float | None = None
    lut_p: float | None = None

class TrainLog(BaseModel):
    config: int | None = None
    sample: int | None = None
    # Per-split metrics from _score_splits (train / evaluation / test), one key
    # per metric per split — split-explicit, no bare aliases. T.evaluation
    # ('dev' in the campaigns) is accuracy_dev/loss_dev; the train-vs-dev gap is
    # the generalization signal, test is logged but never used for selection.
    # No 'epoch': the optimizers are closed-form single-shot fits; epoch was a
    # vestige of the removed SGD trainer and had zero readers.
    accuracy_train: float | None = None
    loss_train: float | None = None
    accuracy_dev: float | None = None
    loss_dev: float | None = None
    accuracy_test: float | None = None
    loss_test: float | None = None
    # Fit diagnostics: was the readout fit well-defined? Filled by
    # glm.train_readout; None on rows fitted before these existed (and for the
    # Gaussian family where the field is meaningless). The α=0 Bernoulli
    # well-definedness decision reads fit_converged and
    # fit_lstsq_fallbacks; the collinearity picture is fit_rank and fit_n_unique.
    fit_iters: int | None = None            # Newton steps used, max over output columns; 1 for Gaussian
    fit_converged: bool | None = None       # False if any column stopped without meeting tol on a full step (or ended saturated at alpha=0); always True for Gaussian
    fit_lstsq_fallbacks: int | None = None  # singular-Hessian lstsq Newton steps (IRLS)
    fit_guard_events: int | None = None     # fit-guard activations: IRLS step-halvings + guard stops; Gaussian solver fallbacks
    fit_saturated: bool | None = None       # Bernoulli: the logit saturated on every training row (the fit separates the train set); None for Gaussian
    fit_w_norm: float | None = None         # ||w_slopes||_2, max over output columns — runaway at alpha=0 on separable designs
    fit_n_unique: int | None = None         # unique design rows after dedup (Bernoulli); None for Gaussian
    fit_rank: int | None = None             # numerical rank: matrix_rank(XtX) for Gaussian, matrix_rank(Xu) for Bernoulli

    @property
    def accuracy(self):
        # Pre-6445a2d view-compat alias: the dashboards' bare metric_col
        # 'T_accuracy' historically held the T.evaluation-split metric (dev in
        # the campaigns). The split-explicit keys above are the real fields;
        # this alias keeps the {'accuracy', 'loss'} extraction field-sets and
        # every saved view meaningful. NOT a field: model_dump ignores it.
        return self.accuracy_dev

    @property
    def loss(self):
        return self.loss_dev

class KQGRMetrics(BaseModel):
    config: int | None = None
    sample: int | None = None
    # readout-representation ranks: what the ridge readout actually sees
    # (summed per-step states in time-series mode, final state otherwise)
    kq: int | None = None
    gr: int | None = None
    delta: int | None = None
    # final-state ranks: Snyder's original construction, kept for
    # comparability with Snyder 2012/2013 (identical to kq/gr/delta in
    # classification output mode)
    kq_last: int | None = None
    gr_last: int | None = None
    delta_last: int | None = None
    # steps-axis retention ranks (gr_steps_probe): only computed when the
    # universe's native KQGR runs on the resolution axis (PI), where the
    # stock GR never probes multi-step retention. delta_steps = kq - gr_steps.
    gr_steps: int | None = None
    gr_steps_last: int | None = None
    delta_steps: int | None = None
    tau_steps: int | None = None
    # the setting measured: filled in kqgr_sweep entries only (a plain kqgr row's
    # setting is its universe's dataset fields)
    tau: int | None = None
    tau_mode: str | None = None
    evaluation: str | None = None
    independent: bool | None = None

class WalkQueryMetrics(BaseModel):
    """The walk x query split of the readout states (tasks with a walk_query_probe: PI's
    retroactive query): every KQ walk under every q, split two-way into walk, query and
    interaction parts (kq_and_gr_metric.walk_query_split). Its own procedure, so its own group."""
    kq: int | None = None                 # rank of every walk under every query
    rank_walk: int | None = None
    rank_query: int | None = None
    rank_interaction: int | None = None
    interaction_energy: float | None = None
    interaction_share: float | None = None

class HeadLog(BaseModel):
    """One readout head's measurements of a shared reservoir run (output_layer.heads).
    One group per measurement procedure; a procedure that does not apply leaves its
    group None. Same groups as LoggingParams' single-mode fields."""
    train: TrainLog | None = None                 # the task readout: one fit for every row
    step_readouts: TrainLog | None = None         # one readout per step, each row scored by its q's (query tasks)
    targets: dict[str, TrainLog] | None = None    # one readout per named target, scored on every row (temporal heads)
    kqgr: KQGRMetrics | None = None               # walk KQ / GR ranks
    kqgr_sweep: dict[str, KQGRMetrics] | None = None   # the same, per dataset.sweep setting (same reservoir)
    walk_query: WalkQueryMetrics | None = None    # walk x query split of the states (query tasks)


class LoggingParams(BaseModel):
    timestamp_utc: Optional[str] = Field(None, description="timestamp utc")
    out_path: Path = Field(
        Path('out'), description="Where to save all logs for this config")
    save_path: Optional[Path] = Field(
        Path('out'), description="Where last run was saved")
    last_checkpoint: Optional[Path] = Field(
        None, description="Where last checkpoint was saved")
    grid_search: Optional[GridSearchParams] = Field(None)
    history: HistoryParams = Field(
        default_factory=HistoryParams, description="Parameters pertaining to recoding of reservoir dynamics")
    universe: str | None = None
    reservoir_id: str | None = Field(None, description="The reservoir's content + seeds (provenance.py)")
    results_version: int | None = Field(None, description="provenance.RESULTS_VERSION the row was made under")
    git_commit: str | None = Field(None, description="The commit the grid ran on, '+dirty' with uncommitted changes")
    train: TrainLog | None = None
    step_readouts: TrainLog | None = None
    targets: dict[str, TrainLog] | None = None
    kqgr: KQGRMetrics | None = None
    kqgr_sweep: dict[str, KQGRMetrics] | None = None
    walk_query: WalkQueryMetrics | None = None
    heads: dict[str, HeadLog] | None = Field(
        None, description="Per-head measurement groups (HeadLog) of an output_layer.heads run (else the groups above)")
    reservoir_metrics: ReservoirMetrics | None = None
    save_keys: Optional[List[str]] = Field(
        default=['parameters', 'w_bi', 'graph', 'wiring',
                 'init_state', 'lut', 'weights'],
        description="Only save these model objects",
        json_schema_extra={'expand': False}  # Mark as non-expandable
    )

    @field_validator('save_keys', mode='before')
    @classmethod
    def _convert_string_to_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    @property
    def M(self):
        return self.kqgr

    @property
    def T(self):
        return self.train


class UniverseWrapper:
    """Lazy wrapper that deep-merges universe overrides into child Params on access."""

    def __init__(self, mother: 'Params', overrides: dict):
        self._mother = mother
        self._overrides = overrides or {}
        self._cache: dict = {}

    def __getattr__(self, name: str) -> 'Params':
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._cache:
            if name in self._overrides:
                merged = deep_merge(self._mother.model_dump(), self._overrides[name])
                self._cache[name] = Params(**merged)
            else:
                self._cache[name] = self._mother
        return self._cache[name]

# =============================================================================
# YAML CONFIGURATION GUIDE
# =============================================================================
# When crafting YAML configuration files, the `wiring` blocks can accept both 
# hardcoded numbers and mathematical string equations. 
# 
# Follow these three rules to avoid "Variable Soup" and crashes:
#
# 1. STRUCTURAL VARIABLES (Auto-injected - DO NOT declare in `variables`):
#    The system automatically calculates and provides the following variables 
#    based on your network size. You can use them freely in any string equation:
#      - I_n_nodes : Number of nodes in the Input layer
#      - R_n_nodes : Number of nodes in the Reservoir layer
#      - src_size  : Total number of nodes sending connections in a wiring block
#      - tgt_size  : Total number of nodes receiving connections in a wiring block
#      - n_nodes   : Alias for tgt_size
#    *Example usage:* k_max: "R_n_nodes"
#
# 2. CUSTOM HYPERPARAMETERS (MUST declare in `variables`):
#    If you invent a variable name because you want to grid-search it or easily
#    tweak it from the top of the file, you MUST define it in the `variables:` 
#    block under `model:`. If the math engine sees a string it doesn't recognize, 
#    it will crash.
#    *Example usage:* k_avg: "R_k_avg"  <-- R_k_avg must be in variables block
#
# 3. HARDCODED CONSTANTS (No quotes, no variables):
#    If a value is static and won't be modified by a grid search or override, 
#    just write the number. Do not create unnecessary variables.
#    *Example usage:* k_min: 0
#
# EXAMPLE WIRING BLOCK:
#    - type: random
#      source: R
#      target: R
#      k_avg: "R_k_avg"      # Custom hyperparameter (Rule 2)
#      k_min: 0              # Hardcoded constant (Rule 3)
#      k_max: "R_n_nodes"    # Auto-injected structural variable (Rule 1)
# =============================================================================

class Params(BaseModel):
    version: Literal['1.0'] = Field('1.0', description="Config schema version — bump when breaking changes require migration")
    model: ModelParams
    logging: LoggingParams = Field(LoggingParams())
    dataset: Optional[Union[KQGRTemporalDatasetParams, KQGRPathIntegrationDatasetParams, KQGRNeuroGymDatasetParams, PathIntegrationDatasetParams, TemporalDatasetParams, NeuroGymDatasetParams]] = None
    multiverse_overrides: Optional[dict] = Field(default=None, json_schema_extra={'expand': False})

    _universes: Optional[UniverseWrapper] = PrivateAttr(default=None)

    @model_validator(mode='after')
    def _sync_query_channel(self):
        """Give the input layer room for the dataset's query channel: query_bits =
        resolution × query.redundancy (0 without a query). Idempotent — a no-op when in
        sync, so dumps round-trip; skipped while fields are unexpanded grid lists."""
        Q = getattr(self.dataset, 'query', None)
        I = self.model.input_layer
        if isinstance(I.resolution, list) or (Q is not None and isinstance(Q.redundancy, list)):
            return self
        want = I.resolution * Q.redundancy if Q is not None else 0
        if I.query_bits != want:
            d = I.model_dump()
            d.update(query_bits=want, bits=None, chunk_size=None, n_nodes=None, chunks=None)
            self.model.input_layer = type(I).model_validate(d)
        return self

    @property
    def U(self) -> UniverseWrapper:
        if self._universes is None:
            self._universes = UniverseWrapper(self, self.multiverse_overrides)
        return self._universes

    @property
    def M(self):
        return self.model

    @property
    def L(self):
        return self.logging

    @property
    def D(self):
        return self.dataset

    @property
    def dataset_init_obj(self):
        return self.D.init_obj

    @property
    def accuracy_obj(self):
        from project.boolean_reservoir.code.train_model import EuclideanDistanceAccuracy, BooleanAccuracy
        from benchmark.path_integration.parameter import PathIntegrationDatasetParams
        if self.M.T.accuracy is not None:
            return self.M.T.accuracy_obj
        # Infer from dataset type so existing YAML configs need no changes
        if isinstance(self.D, PathIntegrationDatasetParams):
            return EuclideanDistanceAccuracy().accuracy
        return BooleanAccuracy().accuracy


def load_yaml_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    params = Params(**config)
    return params

def save_yaml_config(base_model: BaseModel, output_dir_path: Path, file_name='parameters', copy_from_original_file_path=None):
    if copy_from_original_file_path:
        shutil_copy(copy_from_original_file_path, output_dir_path / (file_name + '_original.yaml'))
    _save_yaml_config(base_model, output_dir_path / (file_name + '_short.yaml'), exclude_none=True, exclude_defaults=True)
    _save_yaml_config(base_model, output_dir_path / (file_name + '.yaml'))

def _save_yaml_config(base_model: BaseModel, filepath: Path, **kwargs):
    with open(filepath, 'w') as file:
        yaml.dump(base_model.model_dump(**kwargs), file)

if __name__ == '__main__':
    class Params(BaseModel):
        optim: Union[DynamicParams, List[DynamicParams]] = Field(
            # standard for RC in litterature (ridge)
            default=DynamicParams(name='adam', params={
                                  'lr': 1e-3, 'weight_decay': 1e-3}),
            description="Optimizer configuration"
        )
        save_keys: Optional[List[str]] = Field(
            default=['parameters'],
            description="Only save these model objects",
            json_schema_extra={'expand': False}  # Mark as non-expandable
        )
    p = Params(
        optim=[
            DynamicParams(
                name="adam",
                params=CallParams(
                    lr=[1e-1, 1e-2, 1e-3],
                    a=[1, 2],
                ),
            ),
            DynamicParams(
                name="adamw",
                params=CallParams(
                    lr=[1e-1, 1e-2, 1e-3],
                ),
            ),
        ]
    )
    from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
    P = generate_param_combinations(p)
    for p in P:
        print(p)