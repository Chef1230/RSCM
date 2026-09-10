"""Serializable prior-family plans shared by planning and generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Mapping

from rdb_prior.schema.semantics import SemanticSchemaPlan


class PriorFamily(str, Enum):
    """Compatibility label for the plan's primary executable family."""

    LEGACY_ROLE_SCM = "legacy_role_scm"
    RELATIONAL_SCM = "relational_scm"
    RELATIONAL_TREE = "relational_tree"
    TEMPORAL_EVENT = "temporal_event"
    RULE_PROCESS = "rule_process"


class AttributePriorKind(str, Enum):
    LEGACY_SCM = "legacy_scm"
    COLUMN_SCM = "column_scm"
    TREE = "tree"
    RULE = "rule"


class RelationPriorKind(str, Enum):
    UNIFORM = "uniform"
    LATENT_AFFINITY = "latent_affinity"
    SCM = "scm"
    TREE = "tree"
    COMMUNITY = "community"
    STATE_CONDITIONED_EVENT = "state_conditioned_event"


class TemporalPriorKind(str, Enum):
    STATIC = "static"
    STATIONARY = "stationary"
    SEASONAL = "seasonal"
    CHURN = "churn"
    RENEWAL = "renewal"


class ProcessPriorKind(str, Enum):
    NONE = "none"
    RULE = "rule"
    STATE_MACHINE = "state_machine"
    WORKFLOW = "workflow"


class NuisancePriorKind(str, Enum):
    LEGACY = "legacy"
    MCAR = "mcar"
    MAR = "mar"
    MNAR = "mnar"
    PROXY = "proxy"
    SPURIOUS = "spurious"
    DISTRACTOR = "distractor"


class StateVisibility(str, Enum):
    """How a temporal state may affect ordinary generated data."""

    HIDDEN = "hidden"
    EVENT_EMISSION = "event_emission"


class TransitionClock(str, Enum):
    """Clock which causes a temporal-state transition."""

    EVENT_DRIVEN = "event_driven"
    DURATION_DRIVEN = "duration_driven"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True, kw_only=True)
class StateSpacePlan:
    values: tuple[str, ...]
    terminal_states: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple) or len(self.values) < 2:
            raise ValueError("state-space values must contain at least two states")
        for value in self.values:
            _identifier("state value", value)
        if len(set(self.values)) != len(self.values):
            raise ValueError("state-space values must be unique")
        if not isinstance(self.terminal_states, tuple):
            raise TypeError("terminal_states must be a tuple")
        for value in self.terminal_states:
            _identifier("terminal state", value)
        if len(set(self.terminal_states)) != len(self.terminal_states):
            raise ValueError("terminal_states must be unique")
        if not set(self.terminal_states).issubset(self.values):
            raise ValueError("terminal_states must belong to state-space values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "terminal_states": list(self.terminal_states),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateSpacePlan":
        return cls(
            values=tuple(data["values"]),
            terminal_states=tuple(data.get("terminal_states", ())),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TransitionMechanismPlan:
    clock: TransitionClock
    family: str
    seed: int
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.clock, TransitionClock):
            raise TypeError("clock must be TransitionClock")
        _identifier("transition family", self.family)
        _seed("transition seed", self.seed)
        object.__setattr__(self, "parameters", _parameters(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "clock": self.clock.value,
            "family": self.family,
            "seed": self.seed,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransitionMechanismPlan":
        return cls(
            clock=TransitionClock(data["clock"]),
            family=data["family"],
            seed=data["seed"],
            parameters=tuple(data.get("parameters", {}).items()),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DurationMechanismPlan:
    family: str
    state_conditioned: bool
    seed: int
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _identifier("duration family", self.family)
        if not isinstance(self.state_conditioned, bool):
            raise TypeError("state_conditioned must be a bool")
        _seed("duration seed", self.seed)
        object.__setattr__(self, "parameters", _parameters(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "state_conditioned": self.state_conditioned,
            "seed": self.seed,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DurationMechanismPlan":
        return cls(
            family=data["family"],
            state_conditioned=data["state_conditioned"],
            seed=data["seed"],
            parameters=tuple(data.get("parameters", {}).items()),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TemporalStatePlan:
    """One entity-scoped state process; static and temporal state stay distinct."""

    state_id: str
    owner_table_id: str
    shared_state_id: str
    state_space: StateSpacePlan
    initial_family: str
    initial_parameters: tuple[tuple[str, object], ...]
    transition: TransitionMechanismPlan
    duration: DurationMechanismPlan
    visibility: StateVisibility
    seed: int

    def __post_init__(self) -> None:
        for name in ("state_id", "owner_table_id", "shared_state_id", "initial_family"):
            _identifier(name, getattr(self, name))
        if not isinstance(self.state_space, StateSpacePlan):
            raise TypeError("state_space must be StateSpacePlan")
        object.__setattr__(
            self,
            "initial_parameters",
            _parameters(self.initial_parameters),
        )
        if not isinstance(self.transition, TransitionMechanismPlan):
            raise TypeError("transition must be TransitionMechanismPlan")
        if not isinstance(self.duration, DurationMechanismPlan):
            raise TypeError("duration must be DurationMechanismPlan")
        if not isinstance(self.visibility, StateVisibility):
            raise TypeError("visibility must be StateVisibility")
        _seed("temporal state seed", self.seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "owner_table_id": self.owner_table_id,
            "shared_state_id": self.shared_state_id,
            "state_space": self.state_space.to_dict(),
            "initial_family": self.initial_family,
            "initial_parameters": dict(self.initial_parameters),
            "transition": self.transition.to_dict(),
            "duration": self.duration.to_dict(),
            "visibility": self.visibility.value,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TemporalStatePlan":
        return cls(
            state_id=data["state_id"],
            owner_table_id=data["owner_table_id"],
            shared_state_id=data["shared_state_id"],
            state_space=StateSpacePlan.from_dict(data["state_space"]),
            initial_family=data["initial_family"],
            initial_parameters=tuple(data.get("initial_parameters", {}).items()),
            transition=TransitionMechanismPlan.from_dict(data["transition"]),
            duration=DurationMechanismPlan.from_dict(data["duration"]),
            visibility=StateVisibility(data["visibility"]),
            seed=data["seed"],
        )

def _identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _optional_identifier_tuple(name: str, values: object) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    for value in values:
        _identifier(name, value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")


def _require_unique_state_owners(
    states: tuple[SharedStatePlan, ...] | tuple[TemporalStatePlan, ...],
    label: str,
) -> None:
    owners = [item.owner_table_id for item in states]
    if len(set(owners)) != len(owners):
        raise ValueError(f"{label} owners must be unique")


def _seed(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _parameters(values: tuple[tuple[str, object], ...]) -> tuple[tuple[str, object], ...]:
    if not isinstance(values, tuple):
        raise TypeError("parameters must be a tuple")
    normalized: list[tuple[str, object]] = []
    for item in values:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("parameters items must be pairs")
        key, value = item
        _identifier("parameter name", key)
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError("parameters must be JSON-safe") from error
        normalized.append((key, value))
    if len({key for key, _value in normalized}) != len(normalized):
        raise ValueError("parameter names must be unique")
    return tuple(sorted(normalized, key=lambda item: item[0]))


@dataclass(frozen=True, slots=True, kw_only=True)
class MechanismRef:
    """Versioned choice on one orthogonal prior axis."""

    kind: str
    version: str
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _identifier("mechanism kind", self.kind)
        _identifier("mechanism version", self.version)
        object.__setattr__(self, "parameters", _parameters(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MechanismRef":
        return cls(
            kind=data["kind"],
            version=data["version"],
            parameters=tuple(data.get("parameters", {}).items()),
        )


def _kind_ref(
    value: MechanismRef,
    allowed: type[Enum],
    axis: str,
) -> None:
    if not isinstance(value, MechanismRef):
        raise TypeError(f"{axis} mechanism must be MechanismRef")
    if value.kind not in {item.value for item in allowed}:
        allowed_values = ", ".join(item.value for item in allowed)
        raise ValueError(
            f"{axis} mechanism kind {value.kind!r} is not one of {allowed_values}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedStatePlan:
    state_id: str
    owner_table_id: str
    family: str
    dimension: int
    seed: int
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        for name in ("state_id", "owner_table_id", "family"):
            _identifier(name, getattr(self, name))
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension < 1:
            raise ValueError("dimension must be positive")
        _seed("seed", self.seed)
        object.__setattr__(self, "parameters", _parameters(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {"state_id": self.state_id, "owner_table_id": self.owner_table_id, "family": self.family, "dimension": self.dimension, "seed": self.seed, "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SharedStatePlan":
        return cls(state_id=data["state_id"], owner_table_id=data["owner_table_id"], family=data["family"], dimension=data["dimension"], seed=data["seed"], parameters=tuple(data.get("parameters", {}).items()))


@dataclass(frozen=True, slots=True, kw_only=True)
class TableMechanismBinding:
    table_id: str
    mechanism_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier("table_id", self.table_id)
        if not isinstance(self.mechanism_ids, tuple) or not all(isinstance(item, str) and item for item in self.mechanism_ids):
            raise TypeError("mechanism_ids must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {"table_id": self.table_id, "mechanism_ids": list(self.mechanism_ids)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TableMechanismBinding":
        return cls(table_id=data["table_id"], mechanism_ids=tuple(data.get("mechanism_ids", ())))


@dataclass(frozen=True, slots=True, kw_only=True)
class RelationMechanismBinding:
    foreign_key_id: str
    mechanism_id: str

    def __post_init__(self) -> None:
        _identifier("foreign_key_id", self.foreign_key_id)
        _identifier("mechanism_id", self.mechanism_id)

    def to_dict(self) -> dict[str, str]:
        return {"foreign_key_id": self.foreign_key_id, "mechanism_id": self.mechanism_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationMechanismBinding":
        return cls(foreign_key_id=data["foreign_key_id"], mechanism_id=data["mechanism_id"])


@dataclass(frozen=True, slots=True, kw_only=True)
class MotifMechanismBundle:
    """Mechanisms selected jointly for one schema motif occurrence.

    Family and population mechanism stay as compatibility provenance.
    The four MechanismRef fields are the compositional source of truth.
    """

    bundle_id: str
    motif_occurrence_id: str
    family: PriorFamily
    node_bindings: tuple[TableMechanismBinding, ...]
    edge_bindings: tuple[RelationMechanismBinding, ...]
    population_mechanism: str
    attribute_mechanism: MechanismRef
    relation_mechanism: MechanismRef
    temporal_mechanism: MechanismRef
    process_mechanism: MechanismRef
    shared_state_ids: tuple[str, ...] = ()
    compatible_task_families: tuple[str, ...] = ()
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        for name in ("bundle_id", "motif_occurrence_id", "population_mechanism"):
            _identifier(name, getattr(self, name))
        if not isinstance(self.family, PriorFamily):
            raise TypeError("family must be PriorFamily")
        if not isinstance(self.node_bindings, tuple) or not all(
            isinstance(item, TableMechanismBinding) for item in self.node_bindings
        ):
            raise TypeError("node_bindings must contain TableMechanismBinding values")
        if not isinstance(self.edge_bindings, tuple) or not all(
            isinstance(item, RelationMechanismBinding) for item in self.edge_bindings
        ):
            raise TypeError("edge_bindings must contain RelationMechanismBinding values")
        _kind_ref(self.attribute_mechanism, AttributePriorKind, "attribute")
        _kind_ref(self.relation_mechanism, RelationPriorKind, "relation")
        _kind_ref(self.temporal_mechanism, TemporalPriorKind, "temporal")
        _kind_ref(self.process_mechanism, ProcessPriorKind, "process")
        _optional_identifier_tuple("shared_state_ids", self.shared_state_ids)
        if not isinstance(self.compatible_task_families, tuple) or not all(
            isinstance(item, str) and item for item in self.compatible_task_families
        ):
            raise TypeError("compatible_task_families must contain strings")
        if (
            self.relation_mechanism.kind
            == RelationPriorKind.STATE_CONDITIONED_EVENT.value
            and self.temporal_mechanism.kind == TemporalPriorKind.STATIC.value
        ):
            raise ValueError(
                "state_conditioned_event relation requires a non-static temporal mechanism"
            )
        if (
            self.process_mechanism.kind
            in {
                ProcessPriorKind.STATE_MACHINE.value,
                ProcessPriorKind.WORKFLOW.value,
            }
            and self.temporal_mechanism.kind == TemporalPriorKind.STATIC.value
        ):
            raise ValueError(
                "state-machine and workflow processes require a temporal mechanism"
            )
        object.__setattr__(self, "parameters", _parameters(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "motif_occurrence_id": self.motif_occurrence_id,
            "family": self.family.value,
            "node_bindings": [item.to_dict() for item in self.node_bindings],
            "edge_bindings": [item.to_dict() for item in self.edge_bindings],
            "population_mechanism": self.population_mechanism,
            "attribute_mechanism": self.attribute_mechanism.to_dict(),
            "relation_mechanism": self.relation_mechanism.to_dict(),
            "temporal_mechanism": self.temporal_mechanism.to_dict(),
            "process_mechanism": self.process_mechanism.to_dict(),
            "shared_state_ids": list(self.shared_state_ids),
            "compatible_task_families": list(self.compatible_task_families),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MotifMechanismBundle":
        family = PriorFamily(data["family"])
        attribute_raw = data.get("attribute_mechanism", "legacy")
        temporal_raw = data.get("temporal_mechanism", "legacy")
        refs = _legacy_bundle_refs(family, attribute_raw, temporal_raw)
        return cls(
            bundle_id=data["bundle_id"],
            motif_occurrence_id=data["motif_occurrence_id"],
            family=family,
            node_bindings=tuple(
                TableMechanismBinding.from_dict(item)
                for item in data.get("node_bindings", ())
            ),
            edge_bindings=tuple(
                RelationMechanismBinding.from_dict(item)
                for item in data.get("edge_bindings", ())
            ),
            population_mechanism=data.get("population_mechanism", "legacy"),
            attribute_mechanism=(
                MechanismRef.from_dict(attribute_raw)
                if isinstance(attribute_raw, Mapping)
                else refs[0]
            ),
            relation_mechanism=(
                MechanismRef.from_dict(data["relation_mechanism"])
                if isinstance(data.get("relation_mechanism"), Mapping)
                else refs[1]
            ),
            temporal_mechanism=(
                MechanismRef.from_dict(temporal_raw)
                if isinstance(temporal_raw, Mapping)
                else refs[2]
            ),
            process_mechanism=(
                MechanismRef.from_dict(data["process_mechanism"])
                if isinstance(data.get("process_mechanism"), Mapping)
                else refs[3]
            ),
            shared_state_ids=tuple(data.get("shared_state_ids", ())),
            compatible_task_families=tuple(data.get("compatible_task_families", ())),
            parameters=tuple(data.get("parameters", {}).items()),
        )


def _legacy_bundle_refs(
    family: PriorFamily,
    attribute_name: object,
    temporal_name: object,
) -> tuple[MechanismRef, MechanismRef, MechanismRef, MechanismRef]:
    if family is PriorFamily.TEMPORAL_EVENT:
        return (
            MechanismRef(
                kind=AttributePriorKind.COLUMN_SCM.value,
                version="v1",
                parameters=(("legacy_name", str(attribute_name)),),
            ),
            MechanismRef(
                kind=RelationPriorKind.STATE_CONDITIONED_EVENT.value,
                version="v1",
            ),
            MechanismRef(
                kind=TemporalPriorKind.STATIONARY.value,
                version="v1",
                parameters=(("legacy_name", str(temporal_name)),),
            ),
            MechanismRef(kind=ProcessPriorKind.NONE.value, version="v1"),
        )
    return (
        MechanismRef(kind=AttributePriorKind.LEGACY_SCM.value, version="v1"),
        MechanismRef(kind=RelationPriorKind.LATENT_AFFINITY.value, version="v1"),
        MechanismRef(
            kind=TemporalPriorKind.STATIC.value,
            version="v1",
            parameters=(("legacy_name", str(temporal_name)),),
        ),
        MechanismRef(kind=ProcessPriorKind.NONE.value, version="v1"),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskPolicyPlan:
    programs_per_database: int = 1
    require_family_compatibility: bool = True
    sample_program_before_data: bool = True
    posthoc_horizon_selection: bool = False
    max_materialization_attempts: int = 8
    support_fraction: float = 0.7
    min_support_rows: int = 8
    min_query_rows: int = 4
    min_total_rows: int | None = 12
    min_class_count_per_split: int = 1
    cutoff_fraction_min: float = 0.45
    cutoff_fraction_max: float = 0.70
    horizon_fraction_min: float = 0.12
    horizon_fraction_max: float = 0.30
    positive_rate_min: float = 0.05
    positive_rate_max: float = 0.95

    def __post_init__(self) -> None:
        for name in ("programs_per_database", "max_materialization_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("min_support_rows", "min_query_rows", "min_class_count_per_split"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_total_rows is not None:
            if (
                isinstance(self.min_total_rows, bool)
                or not isinstance(self.min_total_rows, int)
                or self.min_total_rows < 1
            ):
                raise ValueError("min_total_rows must be a positive integer or None")
            if self.min_total_rows < self.min_support_rows + self.min_query_rows:
                raise ValueError(
                    "min_total_rows must be at least min_support_rows + min_query_rows"
                )
        for name in ("require_family_compatibility", "sample_program_before_data", "posthoc_horizon_selection"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for low, high, name in (
            (self.support_fraction, self.support_fraction, "support"),
            (self.cutoff_fraction_min, self.cutoff_fraction_max, "cutoff"),
            (self.horizon_fraction_min, self.horizon_fraction_max, "horizon"),
            (self.positive_rate_min, self.positive_rate_max, "positive rate"),
        ):
            if not 0 < low <= high < 1:
                raise ValueError(f"{name} range must satisfy 0 < low <= high < 1")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskPolicyPlan":
        return cls(**{name: data.get(name, field.default) for name, field in cls.__dataclass_fields__.items()})


class NuisanceColumnRole(str, Enum):
    """Private role of an observed nuisance column."""

    CAUSAL = "causal"
    PROXY = "proxy"
    SPURIOUS = "spurious"
    REDUNDANT = "redundant"
    IRRELEVANT = "irrelevant"
    PURE_NOISE = "pure_noise"


@dataclass(frozen=True, slots=True, kw_only=True)
class NuisanceColumnPlan:
    """Overlay contract for one physical feature column."""

    column_id: str
    role: NuisanceColumnRole
    source_column_ids: tuple[str, ...] = ()
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _identifier("column_id", self.column_id)
        if not isinstance(self.role, NuisanceColumnRole):
            raise TypeError("role must be NuisanceColumnRole")
        _optional_identifier_tuple("source_column_ids", self.source_column_ids)
        object.__setattr__(self, "parameters", _parameters(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_id": self.column_id,
            "role": self.role.value,
            "source_column_ids": list(self.source_column_ids),
            "parameters": dict(self.parameters),
        }

    @property
    def column_role(self) -> NuisanceColumnRole:
        return self.role

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NuisanceColumnPlan":
        return cls(
            column_id=data["column_id"],
            role=NuisanceColumnRole(data["role"]),
            source_column_ids=tuple(data.get("source_column_ids", ())),
            parameters=tuple(data.get("parameters", {}).items()),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MissingnessPlan:
    """One explicit MCAR/MAR/MNAR or structured missingness overlay."""

    column_id: str
    family: str
    rate: float
    driver_column_ids: tuple[str, ...] = ()
    block_size: int = 0
    time_column_id: str | None = None
    seed: int = 0
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _identifier("column_id", self.column_id)
        _identifier("missingness family", self.family)
        if isinstance(self.rate, bool) or not isinstance(self.rate, (int, float)):
            raise TypeError("missingness rate must be numeric")
        if not 0 <= self.rate < 1:
            raise ValueError("missingness rate must be in [0, 1)")
        _optional_identifier_tuple("driver_column_ids", self.driver_column_ids)
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int) or self.block_size < 0:
            raise ValueError("block_size must be a non-negative integer")
        if self.time_column_id is not None:
            _identifier("time_column_id", self.time_column_id)
        _seed("missingness seed", self.seed)
        object.__setattr__(self, "parameters", _parameters(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_id": self.column_id,
            "family": self.family,
            "rate": float(self.rate),
            "driver_column_ids": list(self.driver_column_ids),
            "block_size": self.block_size,
            "time_column_id": self.time_column_id,
            "seed": self.seed,
            "parameters": dict(self.parameters),
        }

    @property
    def mechanism(self) -> str:
        return self.family

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MissingnessPlan":
        return cls(
            column_id=data["column_id"],
            family=data["family"],
            rate=data.get("rate", 0.0),
            driver_column_ids=tuple(data.get("driver_column_ids", ())),
            block_size=data.get("block_size", 0),
            time_column_id=data.get("time_column_id"),
            seed=data.get("seed", 0),
            parameters=tuple(data.get("parameters", {}).items()),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DistractorRelationPlan:
    """An optional weak/irrelevant relation overlay on an existing FK."""

    relation_id: str
    family: str
    parent_table_id: str
    child_table_id: str
    strength: float = 0.0
    seed: int = 0
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        for name in ("relation_id", "family", "parent_table_id", "child_table_id"):
            _identifier(name, getattr(self, name))
        if isinstance(self.strength, bool) or not isinstance(self.strength, (int, float)):
            raise TypeError("relation distractor strength must be numeric")
        if not 0 <= self.strength <= 1:
            raise ValueError("relation distractor strength must be in [0, 1]")
        _seed("relation distractor seed", self.seed)
        object.__setattr__(self, "parameters", _parameters(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "family": self.family,
            "parent_table_id": self.parent_table_id,
            "child_table_id": self.child_table_id,
            "strength": float(self.strength),
            "seed": self.seed,
            "parameters": dict(self.parameters),
        }

    @property
    def relation_group_id(self) -> str:
        return self.relation_id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DistractorRelationPlan":
        return cls(
            relation_id=data["relation_id"],
            family=data["family"],
            parent_table_id=data["parent_table_id"],
            child_table_id=data["child_table_id"],
            strength=data.get("strength", 0.0),
            seed=data.get("seed", 0),
            parameters=tuple(data.get("parameters", {}).items()),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class NuisancePlan:
    """Orthogonal overlay plan; legacy is a deterministic no-op."""

    mechanism: MechanismRef = field(
        default_factory=lambda: MechanismRef(
            kind=NuisancePriorKind.LEGACY.value,
            version="v1",
        )
    )
    column_roles: tuple[NuisanceColumnPlan, ...] = ()
    missingness_plans: tuple[MissingnessPlan, ...] = ()
    distractor_relation_plans: tuple[DistractorRelationPlan, ...] = ()
    environment_id: str = "env_default"
    enabled: bool = False
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _kind_ref(self.mechanism, NuisancePriorKind, "nuisance")
        for name, item_type in (
            ("column_roles", NuisanceColumnPlan),
            ("missingness_plans", MissingnessPlan),
            ("distractor_relation_plans", DistractorRelationPlan),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not all(isinstance(item, item_type) for item in values):
                raise TypeError(f"{name} must contain {item_type.__name__} values")
        if not isinstance(self.environment_id, str) or not self.environment_id.strip():
            raise ValueError("environment_id must not be empty")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        object.__setattr__(self, "parameters", _parameters(self.parameters))
        if len({item.column_id for item in self.column_roles}) != len(self.column_roles):
            raise ValueError("nuisance column IDs must be unique")
        if len({item.column_id for item in self.missingness_plans}) != len(self.missingness_plans):
            raise ValueError("missingness column IDs must be unique")
        if len({item.relation_id for item in self.distractor_relation_plans}) != len(self.distractor_relation_plans):
            raise ValueError("distractor relation IDs must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism.to_dict(),
            "column_roles": [item.to_dict() for item in self.column_roles],
            "missingness_plans": [item.to_dict() for item in self.missingness_plans],
            "distractor_relation_plans": [item.to_dict() for item in self.distractor_relation_plans],
            "environment_id": self.environment_id,
            "enabled": self.enabled,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NuisancePlan":
        if not isinstance(data, Mapping):
            data = {}
        mechanism = data.get("mechanism")
        return cls(
            mechanism=(
                MechanismRef.from_dict(mechanism)
                if isinstance(mechanism, Mapping)
                else MechanismRef(kind=NuisancePriorKind.LEGACY.value, version="v1")
            ),
            column_roles=tuple(NuisanceColumnPlan.from_dict(item) for item in data.get("column_roles", ())),
            missingness_plans=tuple(MissingnessPlan.from_dict(item) for item in data.get("missingness_plans", ())),
            distractor_relation_plans=tuple(DistractorRelationPlan.from_dict(item) for item in data.get("distractor_relation_plans", ())),
            environment_id=data.get("environment_id", "env_default"),
            enabled=data.get("enabled", False),
            parameters=tuple(data.get("parameters", {}).items()),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PriorCompositionPlan:
    """Orthogonal prior selections for one database, independent of family label."""

    plan_id: str
    semantic_schema: SemanticSchemaPlan
    shared_states: tuple[SharedStatePlan, ...]
    motif_bundles: tuple[MotifMechanismBundle, ...]
    nuisance_plan: NuisancePlan
    task_policy: TaskPolicyPlan
    seed: int
    temporal_states: tuple[TemporalStatePlan, ...] = ()

    def __post_init__(self) -> None:
        _identifier("plan_id", self.plan_id)
        if not isinstance(self.semantic_schema, SemanticSchemaPlan):
            raise TypeError("semantic_schema must be SemanticSchemaPlan")
        if not isinstance(self.shared_states, tuple) or not all(
            isinstance(item, SharedStatePlan) for item in self.shared_states
        ):
            raise TypeError("shared_states must contain SharedStatePlan values")
        if not isinstance(self.temporal_states, tuple) or not all(
            isinstance(item, TemporalStatePlan) for item in self.temporal_states
        ):
            raise TypeError("temporal_states must contain TemporalStatePlan values")
        if not isinstance(self.motif_bundles, tuple) or not all(
            isinstance(item, MotifMechanismBundle) for item in self.motif_bundles
        ):
            raise TypeError("motif_bundles must contain MotifMechanismBundle values")
        if not isinstance(self.nuisance_plan, NuisancePlan):
            raise TypeError("nuisance_plan must be NuisancePlan")
        if not isinstance(self.task_policy, TaskPolicyPlan):
            raise TypeError("task_policy must be TaskPolicyPlan")
        _seed("seed", self.seed)
        if len({item.state_id for item in self.shared_states}) != len(
            self.shared_states
        ):
            raise ValueError("shared state IDs must be unique")
        _require_unique_state_owners(self.shared_states, "shared state")
        if len({item.state_id for item in self.temporal_states}) != len(
            self.temporal_states
        ):
            raise ValueError("temporal state IDs must be unique")
        _require_unique_state_owners(self.temporal_states, "temporal state")
        known_shared = {item.state_id for item in self.shared_states}
        if any(
            item.shared_state_id not in known_shared
            for item in self.temporal_states
        ):
            raise ValueError("temporal states must reference shared states")
        if len({item.bundle_id for item in self.motif_bundles}) != len(
            self.motif_bundles
        ):
            raise ValueError("bundle IDs must be unique")

    @property
    def primary_family(self) -> PriorFamily:
        if any(item.family is PriorFamily.RULE_PROCESS for item in self.motif_bundles):
            return PriorFamily.RULE_PROCESS
        if any(item.family is PriorFamily.TEMPORAL_EVENT for item in self.motif_bundles):
            return PriorFamily.TEMPORAL_EVENT
        return PriorFamily.LEGACY_ROLE_SCM

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "semantic_schema": self.semantic_schema.to_dict(),
            "shared_states": [item.to_dict() for item in self.shared_states],
            "temporal_states": [
                item.to_dict() for item in self.temporal_states
            ],
            "motif_bundles": [
                item.to_dict() for item in self.motif_bundles
            ],
            "nuisance_plan": self.nuisance_plan.to_dict(),
            "task_policy": self.task_policy.to_dict(),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PriorCompositionPlan":
        return cls(
            plan_id=data["plan_id"],
            semantic_schema=SemanticSchemaPlan.from_dict(data["semantic_schema"]),
            shared_states=tuple(
                SharedStatePlan.from_dict(item)
                for item in data.get("shared_states", ())
            ),
            temporal_states=tuple(
                TemporalStatePlan.from_dict(item)
                for item in data.get("temporal_states", ())
            ),
            motif_bundles=tuple(
                MotifMechanismBundle.from_dict(item)
                for item in data.get("motif_bundles", ())
            ),
            nuisance_plan=NuisancePlan.from_dict(data.get("nuisance_plan", {})),
            task_policy=TaskPolicyPlan.from_dict(data.get("task_policy", {})),
            seed=data["seed"],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DatabasePriorPlan:
    plan_id: str
    family: PriorFamily
    family_version: str
    semantic_schema: SemanticSchemaPlan
    shared_states: tuple[SharedStatePlan, ...]
    motif_bundles: tuple[MotifMechanismBundle, ...]
    task_policy: TaskPolicyPlan
    seed: int
    temporal_states: tuple[TemporalStatePlan, ...] = ()
    composition: PriorCompositionPlan | None = None

    def __post_init__(self) -> None:
        _identifier("plan_id", self.plan_id)
        _identifier("family_version", self.family_version)
        if not isinstance(self.family, PriorFamily):
            raise TypeError("family must be PriorFamily")
        if not isinstance(self.semantic_schema, SemanticSchemaPlan):
            raise TypeError("semantic_schema must be SemanticSchemaPlan")
        if not isinstance(self.shared_states, tuple) or not all(isinstance(item, SharedStatePlan) for item in self.shared_states):
            raise TypeError("shared_states must contain SharedStatePlan values")
        if not isinstance(self.temporal_states, tuple) or not all(
            isinstance(item, TemporalStatePlan) for item in self.temporal_states
        ):
            raise TypeError("temporal_states must contain TemporalStatePlan values")

        if not isinstance(self.motif_bundles, tuple) or not all(isinstance(item, MotifMechanismBundle) for item in self.motif_bundles):
            raise TypeError("motif_bundles must contain MotifMechanismBundle values")
        if not isinstance(self.task_policy, TaskPolicyPlan):
            raise TypeError("task_policy must be TaskPolicyPlan")
        _seed("seed", self.seed)
        if len({item.state_id for item in self.shared_states}) != len(self.shared_states):
            raise ValueError("shared state IDs must be unique")
        _require_unique_state_owners(self.shared_states, "shared state")
        if len({item.state_id for item in self.temporal_states}) != len(
            self.temporal_states
        ):
            raise ValueError("temporal state IDs must be unique")
        _require_unique_state_owners(self.temporal_states, "temporal state")
        known_shared = {item.state_id for item in self.shared_states}
        if any(item.shared_state_id not in known_shared for item in self.temporal_states):
            raise ValueError("temporal states must reference shared states")

        if len({item.bundle_id for item in self.motif_bundles}) != len(self.motif_bundles):
            raise ValueError("bundle IDs must be unique")
        composition = self.composition
        if composition is None:
            composition = PriorCompositionPlan(
                plan_id=self.plan_id,
                semantic_schema=self.semantic_schema,
                shared_states=self.shared_states,
                temporal_states=self.temporal_states,
                motif_bundles=self.motif_bundles,
                nuisance_plan=NuisancePlan(
                    mechanism=MechanismRef(
                        kind=NuisancePriorKind.LEGACY.value,
                        version="v1",
                    )
                ),
                task_policy=self.task_policy,
                seed=self.seed,
            )
            object.__setattr__(self, "composition", composition)
        elif not isinstance(composition, PriorCompositionPlan):
            raise TypeError("composition must be PriorCompositionPlan or None")
        elif (
            composition.semantic_schema != self.semantic_schema
            or composition.shared_states != self.shared_states
            or composition.temporal_states != self.temporal_states
            or composition.motif_bundles != self.motif_bundles
            or composition.task_policy != self.task_policy
            or composition.seed != self.seed
        ):
            raise ValueError("composition must mirror DatabasePriorPlan provenance")

    def bundle(self, bundle_id: str) -> MotifMechanismBundle:
        return next(item for item in self.motif_bundles if item.bundle_id == bundle_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "family": self.family.value,
            "family_version": self.family_version,
            "semantic_schema": self.semantic_schema.to_dict(),
            "shared_states": [item.to_dict() for item in self.shared_states],
            "temporal_states": [item.to_dict() for item in self.temporal_states],
            "motif_bundles": [item.to_dict() for item in self.motif_bundles],
            "task_policy": self.task_policy.to_dict(),
            "seed": self.seed,
            "composition": (
                None if self.composition is None else self.composition.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatabasePriorPlan":
        return cls(
            plan_id=data["plan_id"],
            family=PriorFamily(data["family"]),
            family_version=data["family_version"],
            semantic_schema=SemanticSchemaPlan.from_dict(data["semantic_schema"]),
            shared_states=tuple(
                SharedStatePlan.from_dict(item)
                for item in data.get("shared_states", ())
            ),
            motif_bundles=tuple(
                MotifMechanismBundle.from_dict(item)
                for item in data.get("motif_bundles", ())
            ),
            task_policy=TaskPolicyPlan.from_dict(data.get("task_policy", {})),
            seed=data["seed"],
            temporal_states=tuple(
                TemporalStatePlan.from_dict(item)
                for item in data.get("temporal_states", ())
            ),
            composition=(
                PriorCompositionPlan.from_dict(data["composition"])
                if data.get("composition") is not None
                else None
            ),
        )


__all__ = [
    "PriorFamily",
    "AttributePriorKind",
    "RelationPriorKind",
    "TemporalPriorKind",
    "ProcessPriorKind",
    "NuisancePriorKind",
    "NuisanceColumnRole",
    "NuisanceColumnPlan",
    "MissingnessPlan",
    "DistractorRelationPlan",
    "MechanismRef",
    "NuisancePlan",
    "PriorCompositionPlan",
    "StateVisibility",
    "TransitionClock",
    "StateSpacePlan",
    "TransitionMechanismPlan",
    "DurationMechanismPlan",
    "TemporalStatePlan",
    "SharedStatePlan",
    "TableMechanismBinding",
    "RelationMechanismBinding",
    "MotifMechanismBundle",
    "TaskPolicyPlan",
    "DatabasePriorPlan",
]
