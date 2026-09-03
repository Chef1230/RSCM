"""Serializable prior-family plans shared by planning and generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Mapping

from rdb_prior.schema.semantics import SemanticSchemaPlan


class PriorFamily(str, Enum):
    LEGACY_ROLE_SCM = "legacy_role_scm"
    RELATIONAL_SCM = "relational_scm"
    RELATIONAL_TREE = "relational_tree"
    TEMPORAL_EVENT = "temporal_event"
    RULE_PROCESS = "rule_process"


def _identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


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
    bundle_id: str
    motif_occurrence_id: str
    family: PriorFamily
    node_bindings: tuple[TableMechanismBinding, ...]
    edge_bindings: tuple[RelationMechanismBinding, ...]
    population_mechanism: str
    temporal_mechanism: str
    attribute_mechanism: str
    compatible_task_families: tuple[str, ...]
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        for name in ("bundle_id", "motif_occurrence_id", "population_mechanism", "temporal_mechanism", "attribute_mechanism"):
            _identifier(name, getattr(self, name))
        if not isinstance(self.family, PriorFamily):
            raise TypeError("family must be PriorFamily")
        if not isinstance(self.node_bindings, tuple) or not all(isinstance(item, TableMechanismBinding) for item in self.node_bindings):
            raise TypeError("node_bindings must contain TableMechanismBinding values")
        if not isinstance(self.edge_bindings, tuple) or not all(isinstance(item, RelationMechanismBinding) for item in self.edge_bindings):
            raise TypeError("edge_bindings must contain RelationMechanismBinding values")
        if not isinstance(self.compatible_task_families, tuple) or not all(isinstance(item, str) and item for item in self.compatible_task_families):
            raise TypeError("compatible_task_families must contain strings")
        object.__setattr__(self, "parameters", _parameters(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {"bundle_id": self.bundle_id, "motif_occurrence_id": self.motif_occurrence_id, "family": self.family.value, "node_bindings": [item.to_dict() for item in self.node_bindings], "edge_bindings": [item.to_dict() for item in self.edge_bindings], "population_mechanism": self.population_mechanism, "temporal_mechanism": self.temporal_mechanism, "attribute_mechanism": self.attribute_mechanism, "compatible_task_families": list(self.compatible_task_families), "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MotifMechanismBundle":
        return cls(bundle_id=data["bundle_id"], motif_occurrence_id=data["motif_occurrence_id"], family=PriorFamily(data["family"]), node_bindings=tuple(TableMechanismBinding.from_dict(item) for item in data["node_bindings"]), edge_bindings=tuple(RelationMechanismBinding.from_dict(item) for item in data["edge_bindings"]), population_mechanism=data["population_mechanism"], temporal_mechanism=data["temporal_mechanism"], attribute_mechanism=data["attribute_mechanism"], compatible_task_families=tuple(data.get("compatible_task_families", ())), parameters=tuple(data.get("parameters", {}).items()))


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskPolicyPlan:
    programs_per_database: int = 1
    require_family_compatibility: bool = True
    sample_program_before_data: bool = True
    posthoc_horizon_selection: bool = False
    max_materialization_attempts: int = 8
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
        for name in ("require_family_compatibility", "sample_program_before_data", "posthoc_horizon_selection"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for low, high, name in ((self.cutoff_fraction_min, self.cutoff_fraction_max, "cutoff"), (self.horizon_fraction_min, self.horizon_fraction_max, "horizon"), (self.positive_rate_min, self.positive_rate_max, "positive rate")):
            if not 0 < low <= high < 1:
                raise ValueError(f"{name} range must satisfy 0 < low <= high < 1")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskPolicyPlan":
        return cls(**{name: data.get(name, field.default) for name, field in cls.__dataclass_fields__.items()})


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

    def __post_init__(self) -> None:
        _identifier("plan_id", self.plan_id)
        _identifier("family_version", self.family_version)
        if not isinstance(self.family, PriorFamily):
            raise TypeError("family must be PriorFamily")
        if not isinstance(self.semantic_schema, SemanticSchemaPlan):
            raise TypeError("semantic_schema must be SemanticSchemaPlan")
        if not isinstance(self.shared_states, tuple) or not all(isinstance(item, SharedStatePlan) for item in self.shared_states):
            raise TypeError("shared_states must contain SharedStatePlan values")
        if not isinstance(self.motif_bundles, tuple) or not all(isinstance(item, MotifMechanismBundle) for item in self.motif_bundles):
            raise TypeError("motif_bundles must contain MotifMechanismBundle values")
        if not isinstance(self.task_policy, TaskPolicyPlan):
            raise TypeError("task_policy must be TaskPolicyPlan")
        _seed("seed", self.seed)
        if len({item.state_id for item in self.shared_states}) != len(self.shared_states):
            raise ValueError("shared state IDs must be unique")
        if len({item.bundle_id for item in self.motif_bundles}) != len(self.motif_bundles):
            raise ValueError("bundle IDs must be unique")

    def bundle(self, bundle_id: str) -> MotifMechanismBundle:
        return next(item for item in self.motif_bundles if item.bundle_id == bundle_id)

    def to_dict(self) -> dict[str, Any]:
        return {"plan_id": self.plan_id, "family": self.family.value, "family_version": self.family_version, "semantic_schema": self.semantic_schema.to_dict(), "shared_states": [item.to_dict() for item in self.shared_states], "motif_bundles": [item.to_dict() for item in self.motif_bundles], "task_policy": self.task_policy.to_dict(), "seed": self.seed}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatabasePriorPlan":
        return cls(plan_id=data["plan_id"], family=PriorFamily(data["family"]), family_version=data["family_version"], semantic_schema=SemanticSchemaPlan.from_dict(data["semantic_schema"]), shared_states=tuple(SharedStatePlan.from_dict(item) for item in data.get("shared_states", ())), motif_bundles=tuple(MotifMechanismBundle.from_dict(item) for item in data.get("motif_bundles", ())), task_policy=TaskPolicyPlan.from_dict(data.get("task_policy", {})), seed=data["seed"])


__all__ = ["PriorFamily", "SharedStatePlan", "TableMechanismBinding", "RelationMechanismBinding", "MotifMechanismBundle", "TaskPolicyPlan", "DatabasePriorPlan"]
