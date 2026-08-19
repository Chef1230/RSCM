"""Serializable task plans and support/query labels for stage 03."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from rdb_prior.task.composite import (
    AggregateOperator,
    AggregateSpec,
    CompositeTaskSpec,
)

# Re-exported composite enums for callers that previously imported them from
# the task model module.
from rdb_prior.task.composite import (
    CombineOperator,
    CompareOperator,
    CompositeFamily,
    LabelOperator,
    PredicateSpec,
)


class TaskMechanism(str, Enum):
    ENTITY_FUTURE_EVENT_EXISTENCE = "entity_future_event_existence"
    HISTORY_GATED_FUTURE_ACTIVITY = "history_gated_future_activity"
    HISTORY_GATED_FUTURE_INACTIVE = "history_gated_future_inactive"
    HISTORY_GATED_FUTURE_ACTIVE = "history_gated_future_active"
    RELATION_ATTRIBUTE = "relation_attribute"
    FUTURE_EVENT_ATTRIBUTE_CONDITION = "future_event_attribute_condition"
    TEMPORAL_RELATIONAL_AGGREGATE = "temporal_relational_aggregate"
    INTERACTION_RESPONSE = "interaction_response"
    RELATIONAL_CLASSIFICATION = "relational_classification"


class ClassificationKind(str, Enum):
    BINARY = "binary"
    CATEGORICAL = "categorical"

class PredictionType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class RouteRole(str, Enum):
    """Synthetic Task DSL supervision for one exact FK path."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    DISTRACTOR = "distractor"


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutePathLabel:
    foreign_key_ids: tuple[str, ...]
    role: RouteRole

    def __post_init__(self) -> None:
        if not isinstance(self.foreign_key_ids, tuple) or not (
            self.foreign_key_ids
        ):
            raise ValueError("foreign_key_ids must be a non-empty tuple")
        for foreign_key_id in self.foreign_key_ids:
            _identifier("route foreign key", foreign_key_id)
        if not isinstance(self.role, RouteRole):
            raise TypeError("role must be RouteRole")

    def to_dict(self) -> dict[str, Any]:
        return {
            "foreign_key_ids": list(self.foreign_key_ids),
            "role": self.role.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoutePathLabel:
        return cls(
            foreign_key_ids=tuple(data["foreign_key_ids"]),
            role=RouteRole(data["role"]),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationRule:
    table_id: str
    time_column_id: str
    max_timestamp: int

    def __post_init__(self) -> None:
        _identifier("table_id", self.table_id)
        _identifier("time_column_id", self.time_column_id)
        if isinstance(self.max_timestamp, bool) or not isinstance(
            self.max_timestamp, int
        ):
            raise TypeError("max_timestamp must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "time_column_id": self.time_column_id,
            "max_timestamp": self.max_timestamp,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObservationRule:
        return cls(
            table_id=data["table_id"],
            time_column_id=data["time_column_id"],
            max_timestamp=data["max_timestamp"],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskPlan:
    task_id: str
    sample_id: str
    instance_id: str
    schema_id: str
    mechanism: TaskMechanism
    prediction_type: PredictionType
    target_table_id: str
    source_table_id: str
    split_strategy: str
    seed: int
    target_column_id: str | None = None
    source_column_id: str | None = None
    foreign_key_id: str | None = None
    secondary_foreign_key_id: str | None = None
    time_column_id: str | None = None
    cutoff_time: int | None = None
    horizon_end_time: int | None = None
    row_cutoff_time_column_id: str | None = None
    masked_column_ids: tuple[str, ...] = ()
    observation_rules: tuple[ObservationRule, ...] = ()
    route_supervision: tuple[RoutePathLabel, ...] = ()
    classification_kind: ClassificationKind | None = None
    aggregate_operator: AggregateOperator | None = None
    threshold: float | None = None
    requested_positive_rate: float | None = None
    realized_positive_rate: float | None = None
    parameters: tuple[tuple[str, float], ...] = ()
    db_start_seconds: int | None = None
    db_end_seconds: int | None = None
    composite_spec: CompositeTaskSpec | None = None

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "sample_id",
            "instance_id",
            "schema_id",
            "target_table_id",
            "source_table_id",
            "split_strategy",
        ):
            _identifier(name, getattr(self, name))
        if not isinstance(self.mechanism, TaskMechanism):
            raise TypeError("mechanism must be TaskMechanism")
        if not isinstance(self.prediction_type, PredictionType):
            raise TypeError("prediction_type must be PredictionType")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in (
            "target_column_id",
            "source_column_id",
            "foreign_key_id",
            "secondary_foreign_key_id",
            "time_column_id",
            "row_cutoff_time_column_id",
        ):
            value = getattr(self, name)
            if value is not None:
                _identifier(name, value)
        if self.classification_kind is not None and not isinstance(
            self.classification_kind, ClassificationKind
        ):
            raise TypeError("classification_kind must be ClassificationKind or None")
        if self.aggregate_operator is not None and not isinstance(
            self.aggregate_operator, AggregateOperator
        ):
            raise TypeError("aggregate_operator must be AggregateOperator or None")
        for name in ("threshold", "requested_positive_rate", "realized_positive_rate"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise TypeError(f"{name} must be numeric or None")
        for name in ("requested_positive_rate", "realized_positive_rate"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("cutoff_time", "horizon_end_time"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{name} must be an integer or None")
        if (
            self.cutoff_time is not None
            and self.horizon_end_time is not None
            and self.horizon_end_time <= self.cutoff_time
        ):
            raise ValueError("horizon_end_time must be after cutoff_time")
        if not isinstance(self.masked_column_ids, tuple):
            raise TypeError("masked_column_ids must be a tuple")
        for column_id in self.masked_column_ids:
            _identifier("masked column", column_id)
        if len(set(self.masked_column_ids)) != len(self.masked_column_ids):
            raise ValueError("masked_column_ids must be unique")
        if not isinstance(self.observation_rules, tuple) or not all(
            isinstance(rule, ObservationRule) for rule in self.observation_rules
        ):
            raise TypeError("observation_rules must contain ObservationRule")
        observation_keys = tuple(
            (rule.table_id, rule.time_column_id)
            for rule in self.observation_rules
        )
        if len(set(observation_keys)) != len(observation_keys):
            raise ValueError("observation_rules must be unique by table/column")
        if (
            self.row_cutoff_time_column_id is not None
            and not self.observation_rules
        ):
            raise ValueError(
                "row-specific cutoff requires at least one observation rule"
            )
        if self.composite_spec is not None and not isinstance(
            self.composite_spec, CompositeTaskSpec
        ):
            raise TypeError("composite_spec must be CompositeTaskSpec or None")
        if not isinstance(self.route_supervision, tuple) or not all(
            isinstance(label, RoutePathLabel)
            for label in self.route_supervision
        ):
            raise TypeError("route_supervision must contain RoutePathLabel")
        route_paths = tuple(
            label.foreign_key_ids for label in self.route_supervision
        )
        if len(set(route_paths)) != len(route_paths):
            raise ValueError("route_supervision paths must be unique")
        object.__setattr__(self, "parameters", _parameters(self.parameters))
        self._validate_database_interval()
        self._validate_mechanism_contract()

    @property
    def parameter_map(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.parameters))

    @property
    def signature(self) -> tuple[Any, ...]:
        base = (
            self.mechanism.value,
            self.target_table_id,
            self.source_table_id,
            self.target_column_id,
            self.source_column_id,
            self.foreign_key_id,
            self.secondary_foreign_key_id,
            self.cutoff_time,
            self.horizon_end_time,
            self.row_cutoff_time_column_id,
            self.classification_kind,
            self.aggregate_operator,
            tuple(
                label.foreign_key_ids
                for label in self.route_supervision
                if label.role is RouteRole.REQUIRED
            ),
        )
        # Window / label-generation parameters are not part of parameters
        # in the base signature, so tasks on the same candidate would otherwise
        # collide in planner dedup. Append the mechanism-specific values only.
        if self.mechanism is TaskMechanism.TEMPORAL_RELATIONAL_AGGREGATE:
            extra = (
                self.parameter_map.get("window"),
                self.parameter_map.get("window_extended"),
            )
        elif self.mechanism in {
            TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE,
            TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE,
        }:
            extra = (
                self.parameter_map.get("cutoff_quantile"),
                self.parameter_map.get("support_fraction"),
                self.requested_positive_rate,
            )
        elif self.mechanism is TaskMechanism.INTERACTION_RESPONSE:
            extra = (
                self.parameter_map.get("beta_u"),
                self.parameter_map.get("beta_f"),
                self.parameter_map.get("beta_s"),
                self.parameter_map.get("beta_p"),
                self.parameter_map.get("base_rate"),
                self.parameter_map.get("invert"),
            )
        else:
            extra = ()
        # Composite tasks are fully described by their composite spec; include
        # its canonical content so distinct composite tasks never collide in
        # planner dedup even when their scalar fields coincide.
        if self.composite_spec is not None:
            extra += (self.composite_spec.canonical(),)
        return base + extra

    def _validate_database_interval(self) -> None:
        """Hard-check every task time lies inside the database calendar window.

        When the plan carries the database calendar interval, the task cutoff,
        horizon and every observation-rule cutoff must fall inside it, and any
        temporal aggregate window must fit within the interval.
        """
        for name in ("db_start_seconds", "db_end_seconds"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{name} must be an integer or None")
        if (self.db_start_seconds is None) != (self.db_end_seconds is None):
            raise ValueError(
                "db_start_seconds and db_end_seconds must both be set or both "
                "be None"
            )
        if self.db_start_seconds is None:
            return
        if self.db_end_seconds <= self.db_start_seconds:
            raise ValueError("db_end_seconds must be after db_start_seconds")
        for name in ("cutoff_time", "horizon_end_time"):
            value = getattr(self, name)
            if value is not None and not (
                self.db_start_seconds <= value <= self.db_end_seconds
            ):
                raise ValueError(
                    f"{name} must lie within the database calendar interval "
                    f"[{self.db_start_seconds}, {self.db_end_seconds}]"
                )
        for rule in self.observation_rules:
            if not (
                self.db_start_seconds
                <= rule.max_timestamp
                <= self.db_end_seconds
            ):
                raise ValueError(
                    "observation rule max_timestamp must lie within the "
                    "database calendar interval"
                )
        for key in ("window", "window_extended"):
            value = self.parameter_map.get(key)
            if value is not None and not (
                0 <= value <= (self.db_end_seconds - self.db_start_seconds)
            ):
                raise ValueError(
                    f"temporal aggregate {key} must fit inside the database "
                    "calendar interval"
                )

    def _validate_mechanism_contract(self) -> None:
        if self.mechanism is TaskMechanism.RELATION_ATTRIBUTE:
            if self.target_column_id is None:
                raise ValueError("relation attribute task requires target_column_id")
            if self.target_column_id not in self.masked_column_ids:
                raise ValueError("relation attribute target must be masked")
            # Temporal FK fields are allowed — when the target table carries a
            # TIME column the mechanism sets cutoff_time and observation_rules.
        elif self.mechanism is TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE:
            if self.prediction_type is not PredictionType.CLASSIFICATION:
                raise ValueError("future event existence must be classification")
            if None in (
                self.foreign_key_id,
                self.time_column_id,
                self.cutoff_time,
                self.horizon_end_time,
            ):
                raise ValueError("future event existence requires FK and horizon")
            if self.target_column_id is None:
                raise ValueError("future event existence requires target_column_id")
            if self.target_column_id not in self.masked_column_ids:
                raise ValueError("future event existence target must be masked")
        elif self.mechanism in {
            TaskMechanism.HISTORY_GATED_FUTURE_ACTIVITY,
            TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE,
            TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE,
        }:
            if self.prediction_type is not PredictionType.CLASSIFICATION:
                raise ValueError("history gated future activity must be classification")
            if None in (
                self.foreign_key_id,
                self.time_column_id,
                self.cutoff_time,
                self.horizon_end_time,
            ):
                raise ValueError("history gated future activity requires FK and horizon")
            if self.target_column_id is None:
                raise ValueError("history gated future activity requires target_column_id")
            if self.target_column_id not in self.masked_column_ids:
                raise ValueError("history gated future activity target must be masked")
        elif self.mechanism is TaskMechanism.FUTURE_EVENT_ATTRIBUTE_CONDITION:
            if self.target_column_id is None or self.row_cutoff_time_column_id is None:
                raise ValueError("future event attribute requires target and cutoff columns")
            if self.target_column_id not in self.masked_column_ids:
                raise ValueError("future event attribute target must be masked")
        elif self.mechanism is TaskMechanism.TEMPORAL_RELATIONAL_AGGREGATE:
            if self.target_column_id is None:
                raise ValueError("temporal aggregate requires target_column_id")
            if self.target_column_id not in self.masked_column_ids:
                raise ValueError("temporal aggregate target must be masked")
            if self.aggregate_operator is None or self.time_column_id is None:
                raise ValueError("temporal aggregate requires operator and time column")
            if not any(
                label.role is RouteRole.REQUIRED for label in self.route_supervision
            ):
                raise ValueError("temporal aggregate requires an exact required path")
        elif self.mechanism is TaskMechanism.INTERACTION_RESPONSE:
            if self.prediction_type is not PredictionType.CLASSIFICATION:
                raise ValueError("interaction response must be classification")
            if None in (
                self.foreign_key_id,
                self.time_column_id,
                self.row_cutoff_time_column_id,
                self.source_column_id,
            ):
                raise ValueError(
                    "interaction response requires FK, time, row cutoff and feature"
                )
            if self.target_column_id is None:
                raise ValueError("interaction response requires target_column_id")
            if self.target_column_id not in self.masked_column_ids:
                raise ValueError("interaction response target must be masked")
            if not any(
                label.role is RouteRole.REQUIRED for label in self.route_supervision
            ):
                raise ValueError("interaction response requires an exact required path")
        elif self.mechanism is TaskMechanism.RELATIONAL_CLASSIFICATION:
            if self.prediction_type is not PredictionType.CLASSIFICATION:
                raise ValueError("relational classification must be classification")
            if self.composite_spec is None:
                raise ValueError(
                    "relational classification requires composite_spec"
                )
            if self.target_column_id is None:
                raise ValueError(
                    "relational classification requires target_column_id"
                )
            if self.target_column_id not in self.masked_column_ids:
                raise ValueError(
                    "relational classification target must be masked"
                )
            if not any(
                label.role is RouteRole.REQUIRED for label in self.route_supervision
            ):
                raise ValueError(
                    "relational classification requires an exact required path"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sample_id": self.sample_id,
            "instance_id": self.instance_id,
            "schema_id": self.schema_id,
            "mechanism": self.mechanism.value,
            "prediction_type": self.prediction_type.value,
            "target_table_id": self.target_table_id,
            "source_table_id": self.source_table_id,
            "target_column_id": self.target_column_id,
            "source_column_id": self.source_column_id,
            "foreign_key_id": self.foreign_key_id,
            "secondary_foreign_key_id": self.secondary_foreign_key_id,
            "time_column_id": self.time_column_id,
            "cutoff_time": self.cutoff_time,
            "horizon_end_time": self.horizon_end_time,
            "row_cutoff_time_column_id": self.row_cutoff_time_column_id,
            "split_strategy": self.split_strategy,
            "seed": self.seed,
            "masked_column_ids": list(self.masked_column_ids),
            "observation_rules": [
                rule.to_dict() for rule in self.observation_rules
            ],
            "route_supervision": [
                label.to_dict() for label in self.route_supervision
            ],
            "classification_kind": (
                None if self.classification_kind is None else self.classification_kind.value
            ),
            "aggregate_operator": (
                None if self.aggregate_operator is None else self.aggregate_operator.value
            ),
            "threshold": self.threshold,
            "requested_positive_rate": self.requested_positive_rate,
            "realized_positive_rate": self.realized_positive_rate,
            "parameters": dict(self.parameters),
            "db_start_seconds": self.db_start_seconds,
            "db_end_seconds": self.db_end_seconds,
            "composite_spec": (
                None
                if self.composite_spec is None
                else self.composite_spec.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskPlan:
        return cls(
            task_id=data["task_id"],
            sample_id=data["sample_id"],
            instance_id=data["instance_id"],
            schema_id=data["schema_id"],
            mechanism=TaskMechanism(
                "entity_future_event_existence"
                if data["mechanism"] == "future_event_existence"
                else data["mechanism"]
            ),
            prediction_type=PredictionType(data["prediction_type"]),
            target_table_id=data["target_table_id"],
            source_table_id=data["source_table_id"],
            target_column_id=data.get("target_column_id"),
            source_column_id=data.get("source_column_id"),
            foreign_key_id=data.get("foreign_key_id"),
            secondary_foreign_key_id=data.get("secondary_foreign_key_id"),
            time_column_id=data.get("time_column_id"),
            cutoff_time=data.get("cutoff_time"),
            horizon_end_time=data.get("horizon_end_time"),
            row_cutoff_time_column_id=data.get(
                "row_cutoff_time_column_id"
            ),
            split_strategy=data["split_strategy"],
            seed=data["seed"],
            masked_column_ids=tuple(data.get("masked_column_ids", ())),
            observation_rules=tuple(
                ObservationRule.from_dict(item)
                for item in data.get("observation_rules", ())
            ),
            route_supervision=tuple(
                RoutePathLabel.from_dict(item)
                for item in data.get("route_supervision", ())
            ),
            classification_kind=(
                None
                if data.get("classification_kind") is None
                else ClassificationKind(data["classification_kind"])
            ),
            aggregate_operator=(
                None
                if data.get("aggregate_operator") is None
                else AggregateOperator(data["aggregate_operator"])
            ),
            threshold=data.get("threshold"),
            requested_positive_rate=data.get("requested_positive_rate"),
            realized_positive_rate=data.get("realized_positive_rate"),
            parameters=tuple(data.get("parameters", {}).items()),
            db_start_seconds=data.get("db_start_seconds"),
            db_end_seconds=data.get("db_end_seconds"),
            composite_spec=(
                None
                if data.get("composite_spec") is None
                else CompositeTaskSpec.from_dict(data["composite_spec"])
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskData:
    support_row_ids: np.ndarray
    support_labels: np.ndarray
    query_row_ids: np.ndarray
    query_labels: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "support_row_ids",
            "support_labels",
            "query_row_ids",
            "query_labels",
        ):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.ndim != 1:
                raise TypeError(f"{name} must be a one-dimensional ndarray")
            if value.dtype == object:
                raise TypeError(f"{name} cannot use object dtype")
        for name in ("support_row_ids", "query_row_ids"):
            value = getattr(self, name)
            if value.dtype.kind not in {"i", "u"}:
                raise TypeError(f"{name} must use an integer dtype")
            if np.any(value < 0) or len(np.unique(value)) != len(value):
                raise ValueError(f"{name} must contain unique non-negative rows")
        if len(self.support_row_ids) != len(self.support_labels):
            raise ValueError("support rows and labels must align")
        if len(self.query_row_ids) != len(self.query_labels):
            raise ValueError("query rows and labels must align")
        if not len(self.support_row_ids) or not len(self.query_row_ids):
            raise ValueError("support and query must both be non-empty")
        if np.intersect1d(self.support_row_ids, self.query_row_ids).size:
            raise ValueError("support and query rows must be disjoint")

    @property
    def total_rows(self) -> int:
        return len(self.support_row_ids) + len(self.query_row_ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlannedTask:
    plan: TaskPlan
    data: TaskData


def _identifier(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _parameters(
    values: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    if not isinstance(values, tuple):
        raise TypeError("parameters must be a tuple")
    result: list[tuple[str, float]] = []
    for name, value in values:
        _identifier("parameter name", name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("parameter values must be numeric")
        result.append((name, float(value)))
    if len({name for name, _value in result}) != len(result):
        raise ValueError("parameter names must be unique")
    return tuple(sorted(result))


__all__ = [
    "TaskMechanism",
    "ClassificationKind",
    "AggregateOperator",
    "PredictionType",
    "RouteRole",
    "RoutePathLabel",
    "ObservationRule",
    "TaskPlan",
    "TaskData",
    "PlannedTask",
    "CompositeFamily",
    "CompareOperator",
    "CombineOperator",
    "LabelOperator",
    "PredicateSpec",
    "AggregateSpec",
    "CompositeTaskSpec",
]
