"""Pre-data task programs and deterministic post-data execution.

The program layer is intentionally small: it stores a serialisable label
expression and calendar policies, while the executor delegates visibility and
split checks to the existing task contracts. Legacy program payloads keep
their original fields and are upgraded on read.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping as MappingABC
from typing import Any, Mapping

import numpy as np

from rdb_prior.compilation.model import ColumnKind, PhysicalSchema
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.instance.plan import InstancePlan
from rdb_prior.priors.model import DatabasePriorPlan, PriorFamily
from rdb_prior.runtime import RuntimeContext
from rdb_prior.task.mechanisms import (
    _SYNTHETIC_TARGET,
    _future_existence_values,
    _history_gated_future_values,
    _observation_rules,
    _numeric,
    _schema_route_labels,
    _stratified_split,
)
from rdb_prior.task.model import (
    ClassificationKind,
    PlannedTask,
    PredictionType,
    TaskData,
    TaskMechanism,
    TaskPlan,
)


def _encode(value: Any) -> Any:
    if isinstance(value, LabelExpression):
        return value.to_dict()
    if isinstance(value, (CutoffPolicy, HorizonPolicy)):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, MappingABC):
        return {str(key): _encode(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class LabelExpression:
    """A tiny serialisable expression node for a task label program."""

    kind: str
    args: tuple[Any, ...] = ()
    value: Any = None

    def __init__(
        self,
        *,
        kind: str | None = None,
        args: tuple[Any, ...] = (),
        value: Any = None,
        operator: str | None = None,
        operands: tuple[Any, ...] | None = None,
    ) -> None:
        chosen_kind = kind if kind is not None else operator
        if chosen_kind is None:
            raise TypeError("expression requires kind or operator")
        chosen_args = args if operands is None else operands
        object.__setattr__(self, "kind", chosen_kind)
        object.__setattr__(self, "args", chosen_args)
        object.__setattr__(self, "value", value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("expression kind must be a non-empty string")
        if not isinstance(self.args, tuple):
            raise TypeError("expression args must be a tuple")

    @property
    def operator(self) -> str:
        return self.kind

    @property
    def operands(self) -> tuple[Any, ...]:
        return self.args

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "args": [_encode(item) for item in self.args],
            "value": _encode(self.value),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelExpression":
        def decode(item: Any) -> Any:
            if isinstance(item, MappingABC) and "kind" in item and "args" in item:
                return cls.from_dict(item)
            if isinstance(item, list):
                return tuple(decode(child) for child in item)
            if isinstance(item, MappingABC):
                return {str(key): decode(child) for key, child in item.items()}
            return item

        return cls(
            kind=str(data["kind"]),
            args=tuple(decode(item) for item in data.get("args", ())),
            value=decode(data.get("value")),
        )


ProgramNode = LabelExpression


@dataclass(frozen=True, slots=True, kw_only=True)
class CutoffPolicy:
    family: str = "calendar_fraction"
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("cutoff policy family must be non-empty")
        if not isinstance(self.parameters, tuple):
            raise TypeError("cutoff policy parameters must be a tuple")

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "parameters": _encode(dict(self.parameters))}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CutoffPolicy":
        return cls(
            family=str(data.get("family", "calendar_fraction")),
            parameters=tuple(data.get("parameters", {}).items()),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class HorizonPolicy:
    family: str = "calendar_fraction"
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("horizon policy family must be non-empty")
        if not isinstance(self.parameters, tuple):
            raise TypeError("horizon policy parameters must be a tuple")

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "parameters": _encode(dict(self.parameters))}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HorizonPolicy":
        return cls(
            family=str(data.get("family", "calendar_fraction")),
            parameters=tuple(data.get("parameters", {}).items()),
        )


_PROGRAM_FAMILIES = (
    "future_event_existence",
    "history_gated_future_active",
    "history_gated_future_inactive",
    "future_event_attribute",
    "temporal_aggregate",
    "interaction_response",
    "multi_hop_program",
)

_FAMILY_ALIASES = {
    "future_event_existence": frozenset(
        {"future_event_existence", "entity_future_event_existence"}
    ),
    "history_gated_future_active": frozenset(
        {"history_gated_future_active", "history_gated_future_activity"}
    ),
    "history_gated_future_inactive": frozenset(
        {"history_gated_future_inactive"}
    ),
    "future_event_attribute": frozenset(
        {"future_event_attribute", "future_event_attribute_condition"}
    ),
    "temporal_aggregate": frozenset(
        {"temporal_aggregate", "temporal_relational_aggregate"}
    ),
    "interaction_response": frozenset({"interaction_response"}),
    "multi_hop_program": frozenset({"multi_hop_program"}),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskProgramPlan:
    """A pre-data task program with legacy fields retained as aliases."""

    program_id: str
    family: str
    target_table_id: str
    source_table_id: str = ""
    foreign_key_id: str = ""
    time_column_id: str = ""
    cutoff_time: int | None = None
    horizon_end_time: int | None = None
    required_mechanism_ids: tuple[str, ...] = ()
    seed: int = 0
    prior_plan_id: str = ""
    support_fraction: float = 0.7
    min_support_rows: int = 8
    min_query_rows: int = 4
    min_class_count_per_split: int = 1
    expression: LabelExpression | None = None
    cutoff_policy: CutoffPolicy | None = None
    horizon_policy: HorizonPolicy | None = None
    required_bundle_ids: tuple[str, ...] = ()
    source_paths: tuple[str, ...] = ()
    prior_family: str | None = None
    attribute_column_id: str | None = None
    threshold: float | None = None
    history_window_seconds: int | None = None

    def __post_init__(self) -> None:
        for name in ("program_id", "family", "target_table_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("source_table_id", "foreign_key_id", "time_column_id", "prior_plan_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
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
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in ("required_mechanism_ids", "required_bundle_ids", "source_paths"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not all(
                isinstance(item, str) and item for item in values
            ):
                raise ValueError(f"{name} must contain non-empty strings")
        if not self.required_mechanism_ids and not self.required_bundle_ids:
            raise ValueError("a task program must require a bundle or mechanism")
        if self.expression is not None and not isinstance(self.expression, LabelExpression):
            if isinstance(self.expression, MappingABC):
                object.__setattr__(
                    self, "expression", LabelExpression.from_dict(self.expression)
                )
            else:
                raise TypeError("expression must be LabelExpression or None")
        for name, policy_type in (
            ("cutoff_policy", CutoffPolicy),
            ("horizon_policy", HorizonPolicy),
        ):
            policy = getattr(self, name)
            if policy is not None and not isinstance(policy, policy_type):
                if isinstance(policy, MappingABC):
                    object.__setattr__(self, name, policy_type.from_dict(policy))
                else:
                    raise TypeError(f"{name} must be a policy or None")
        if not 0.0 < self.support_fraction < 1.0:
            raise ValueError("support_fraction must be in (0, 1)")
        for name in ("min_support_rows", "min_query_rows", "min_class_count_per_split"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.attribute_column_id is not None and not isinstance(
            self.attribute_column_id, str
        ):
            raise TypeError("attribute_column_id must be a string or None")
        if self.threshold is not None and not isinstance(self.threshold, (int, float)):
            raise TypeError("threshold must be numeric or None")
        if self.history_window_seconds is not None and (
            isinstance(self.history_window_seconds, bool)
            or not isinstance(self.history_window_seconds, int)
            or self.history_window_seconds < 1
        ):
            raise ValueError("history_window_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "family": self.family,
            "target_table_id": self.target_table_id,
            "source_table_id": self.source_table_id,
            "foreign_key_id": self.foreign_key_id,
            "time_column_id": self.time_column_id,
            "cutoff_time": self.cutoff_time,
            "horizon_end_time": self.horizon_end_time,
            "required_mechanism_ids": list(self.required_mechanism_ids),
            "seed": self.seed,
            "prior_plan_id": self.prior_plan_id,
            "support_fraction": self.support_fraction,
            "min_support_rows": self.min_support_rows,
            "min_query_rows": self.min_query_rows,
            "min_class_count_per_split": self.min_class_count_per_split,
            "expression": None if self.expression is None else self.expression.to_dict(),
            "cutoff_policy": None if self.cutoff_policy is None else self.cutoff_policy.to_dict(),
            "horizon_policy": None if self.horizon_policy is None else self.horizon_policy.to_dict(),
            "required_bundle_ids": list(self.required_bundle_ids),
            "source_paths": list(self.source_paths),
            "prior_family": self.prior_family,
            "attribute_column_id": self.attribute_column_id,
            "threshold": self.threshold,
            "history_window_seconds": self.history_window_seconds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskProgramPlan":
        required_mechanisms = tuple(data.get("required_mechanism_ids", ()))
        required_bundles = tuple(data.get("required_bundle_ids", ()))
        if not required_bundles and required_mechanisms:
            required_bundles = (required_mechanisms[0],)
        expression = data.get("expression")
        cutoff_policy = data.get("cutoff_policy")
        horizon_policy = data.get("horizon_policy")
        return cls(
            program_id=data["program_id"],
            family=data["family"],
            target_table_id=data["target_table_id"],
            source_table_id=data.get("source_table_id", ""),
            foreign_key_id=data.get("foreign_key_id", ""),
            time_column_id=data.get("time_column_id", ""),
            cutoff_time=data.get("cutoff_time"),
            horizon_end_time=data.get("horizon_end_time"),
            required_mechanism_ids=required_mechanisms,
            seed=data.get("seed", 0),
            prior_plan_id=data.get("prior_plan_id", ""),
            support_fraction=data.get("support_fraction", 0.7),
            min_support_rows=data.get("min_support_rows", 8),
            min_query_rows=data.get("min_query_rows", 4),
            min_class_count_per_split=data.get("min_class_count_per_split", 1),
            expression=None if expression is None else LabelExpression.from_dict(expression),
            cutoff_policy=None if cutoff_policy is None else CutoffPolicy.from_dict(cutoff_policy),
            horizon_policy=None if horizon_policy is None else HorizonPolicy.from_dict(horizon_policy),
            required_bundle_ids=required_bundles,
            source_paths=tuple(data.get("source_paths", ())),
            prior_family=data.get("prior_family"),
            attribute_column_id=data.get("attribute_column_id"),
            threshold=data.get("threshold"),
            history_window_seconds=data.get("history_window_seconds"),
        )


class TaskProgramPlanner:
    """Sample programs from compatible motif bundles before materialisation."""

    def plan(
        self,
        *,
        schema: PhysicalSchema,
        instance_plan: InstancePlan,
        prior_plan: DatabasePriorPlan,
        runtime: RuntimeContext,
    ) -> tuple[TaskProgramPlan, ...]:
        policy = prior_plan.task_policy
        if prior_plan.family not in {PriorFamily.TEMPORAL_EVENT, PriorFamily.RULE_PROCESS}:
            return ()
        if not policy.sample_program_before_data:
            return ()
        if policy.posthoc_horizon_selection:
            raise ValueError(
                "posthoc_horizon_selection cannot be enabled for pre-data programs"
            )
        start, end = instance_plan.calendar_start_seconds, instance_plan.calendar_end_seconds
        if start is None or end is None or end <= start:
            raise ValueError("pre-data task programs require a calendar")
        temporal_bundles = [
            bundle for bundle in prior_plan.motif_bundles
            if bundle.family in {PriorFamily.TEMPORAL_EVENT, PriorFamily.RULE_PROCESS}
        ]
        if not temporal_bundles:
            raise ValueError("temporal prior has no motif bundle")
        family_candidates = [
            family for family in _PROGRAM_FAMILIES
            if any(
                not policy.require_family_compatibility
                or any(name in _FAMILY_ALIASES[family] for name in bundle.compatible_task_families)
                for bundle in temporal_bundles
            )
        ]
        if not family_candidates:
            raise ValueError("temporal prior has no compatible task program family")
        span = end - start
        programs: list[TaskProgramPlan] = []
        for index in range(policy.programs_per_database):
            family = family_candidates[index % len(family_candidates)]
            compatible = [
                bundle for bundle in temporal_bundles
                if (
                    not policy.require_family_compatibility
                    or any(name in _FAMILY_ALIASES[family] for name in bundle.compatible_task_families)
                )
            ]
            if not compatible:
                raise ValueError(f"no motif bundle is compatible with {family}")
            bundle = compatible[index % len(compatible)]
            parameters = dict(bundle.parameters)
            source_table_id = str(parameters["event_table_id"])
            target_table_id = str(parameters["entity_table_id"])
            edge_id = bundle.edge_bindings[0].foreign_key_id
            time_column_id = next(
                column.column_id for column in schema.table(source_table_id).columns
                if column.kind is ColumnKind.TIME
            )
            rng = runtime.numpy_rng("task-program", index)
            cutoff_fraction = float(rng.uniform(policy.cutoff_fraction_min, policy.cutoff_fraction_max))
            horizon_fraction = float(rng.uniform(policy.horizon_fraction_min, policy.horizon_fraction_max))
            cutoff = start + int(round(span * cutoff_fraction))
            horizon = min(end, cutoff + max(1, int(round(span * horizon_fraction))))
            if horizon <= cutoff:
                raise ValueError("sampled task program has empty horizon")
            feature_columns = [
                column.column_id for column in schema.table(source_table_id).columns
                if column.kind is ColumnKind.FEATURE
            ]
            attribute_column_id = feature_columns[index % len(feature_columns)] if feature_columns else None
            threshold = (
                float(rng.uniform(-0.5, 0.5)) if family == "future_event_attribute"
                else (1.0 if family in {"temporal_aggregate", "multi_hop_program"} else None)
            )
            args: tuple[Any, ...] = (
                ("foreign_key_id", edge_id),
                ("time_column_id", time_column_id),
                ("cutoff_time", cutoff),
                ("horizon_end_time", horizon),
            )
            if attribute_column_id is not None:
                args += (("attribute_column_id", attribute_column_id),)
            if threshold is not None:
                args += (("threshold", threshold),)
            required = [bundle.bundle_id]
            required.extend({
                "future_event_existence": ("event_count", "event_time"),
                "history_gated_future_active": ("event_count", "event_time", "history"),
                "history_gated_future_inactive": ("event_count", "event_time", "history"),
                "future_event_attribute": ("event_count", "event_time", "event_attribute"),
                "temporal_aggregate": ("event_count",),
                "interaction_response": ("interaction",),
                "multi_hop_program": ("event_count", "relation_path"),
            }[family])
            programs.append(
                TaskProgramPlan(
                    program_id=f"program_{instance_plan.sample_id}_{index:03d}",
                    family=family,
                    target_table_id=target_table_id,
                    source_table_id=source_table_id,
                    foreign_key_id=edge_id,
                    time_column_id=time_column_id,
                    cutoff_time=cutoff,
                    horizon_end_time=horizon,
                    required_mechanism_ids=tuple(required),
                    seed=runtime.seed("task-program", index),
                    prior_plan_id=prior_plan.plan_id,
                    expression=LabelExpression(kind=family, args=args),
                    cutoff_policy=CutoffPolicy(
                        family="calendar_fraction",
                        parameters=(("fraction", cutoff_fraction),),
                    ),
                    horizon_policy=HorizonPolicy(
                        family="calendar_fraction",
                        parameters=(("fraction", horizon_fraction),),
                    ),
                    required_bundle_ids=(bundle.bundle_id,),
                    source_paths=(edge_id,),
                    prior_family=prior_plan.family.value,
                    attribute_column_id=attribute_column_id,
                    threshold=threshold,
                )
            )
        return tuple(programs)


def _normal_family(family: str) -> str:
    for canonical, aliases in _FAMILY_ALIASES.items():
        if family in aliases:
            return canonical
    return family


def _entity_labels(
    *, schema: PhysicalSchema, database: DatabaseInstance, program: TaskProgramPlan
) -> np.ndarray:
    if not program.source_table_id or not program.foreign_key_id or not program.time_column_id:
        raise ValueError("task program is missing source/FK/time metadata")
    foreign_key = next(
        item for item in schema.foreign_keys if item.foreign_key_id == program.foreign_key_id
    )
    event = database.table(program.source_table_id)
    assignments = np.asarray(event.column(foreign_key.child_column_id)).astype(np.int64)
    times = np.asarray(event.column(program.time_column_id)).astype(np.int64)
    entity_count = database.table(program.target_table_id).row_count
    candidate = type(
        "FutureEventCandidateCompat", (),
        {
            "foreign_key_id": program.foreign_key_id,
            "entity_table_id": program.target_table_id,
            "event_table_id": program.source_table_id,
            "time_column_id": program.time_column_id,
        },
    )()
    if program.cutoff_time is None or program.horizon_end_time is None:
        raise ValueError("task program requires cutoff and horizon")
    family = _normal_family(program.family)
    if family == "future_event_existence":
        return _future_existence_values(
            schema, database, candidate, int(program.cutoff_time), int(program.horizon_end_time)
        )
    if family == "history_gated_future_active":
        return _history_gated_future_values(
            schema, database, candidate,
            cutoff=int(program.cutoff_time), horizon=int(program.horizon_end_time),
            positive_is_active=True,
        )
    if family == "history_gated_future_inactive":
        return _history_gated_future_values(
            schema, database, candidate,
            cutoff=int(program.cutoff_time), horizon=int(program.horizon_end_time),
            positive_is_active=False,
        )
    valid = (assignments >= 0) & (times > int(program.cutoff_time)) & (times <= int(program.horizon_end_time))
    if family == "future_event_attribute":
        if not program.attribute_column_id:
            raise ValueError("future_event_attribute requires attribute_column_id")
        values = _numeric(event.column(program.attribute_column_id))
        threshold = 0.0 if program.threshold is None else float(program.threshold)
        labels = np.zeros(entity_count, dtype=np.int8)
        selected = valid & (values > threshold)
        if np.any(selected):
            labels[np.unique(assignments[selected])] = 1
        return labels
    if family in {"temporal_aggregate", "multi_hop_program", "interaction_response"}:
        counts = (
            np.bincount(assignments[valid], minlength=entity_count)
            if np.any(valid) else np.zeros(entity_count, dtype=np.int64)
        )
        if family == "interaction_response":
            return (counts > 0).astype(np.int8)
        threshold = 1.0 if program.threshold is None else float(program.threshold)
        return (counts > threshold).astype(np.int8)
    raise ValueError(f"unsupported task program family {program.family!r}")


def _mechanism_for_family(family: str) -> TaskMechanism:
    family = _normal_family(family)
    if family == "history_gated_future_active":
        return TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE
    if family == "history_gated_future_inactive":
        return TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE
    return TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE


class TaskExecutor:
    """Execute a fixed program; no horizon/threshold calibration uses labels."""

    def execute(
        self,
        *,
        sample_id: str,
        schema: PhysicalSchema,
        database: DatabaseInstance,
        program: TaskProgramPlan,
        support_fraction: float | None = None,
        min_support_rows: int | None = None,
        min_query_rows: int | None = None,
        min_class_count_per_split: int | None = None,
        positive_rate_min: float = 0.0,
        positive_rate_max: float = 1.0,
    ) -> PlannedTask | None:
        if not 0.0 <= positive_rate_min <= positive_rate_max <= 1.0:
            raise ValueError("positive-rate bounds must lie in [0, 1]")
        if program.cutoff_time is None or program.horizon_end_time is None:
            raise ValueError("task program requires a sampled cutoff and horizon")
        support_fraction = program.support_fraction if support_fraction is None else support_fraction
        min_support_rows = program.min_support_rows if min_support_rows is None else min_support_rows
        min_query_rows = program.min_query_rows if min_query_rows is None else min_query_rows
        min_class_count_per_split = (
            program.min_class_count_per_split
            if min_class_count_per_split is None else min_class_count_per_split
        )
        labels = _entity_labels(schema=schema, database=database, program=program)
        valid_mask = labels >= 0
        valid_indices = np.flatnonzero(valid_mask).astype(np.int64)
        if not len(valid_indices):
            return None
        valid_labels = np.asarray(labels[valid_mask], dtype=np.int8)
        positive_rate = float(np.mean(valid_labels))
        if not positive_rate_min <= positive_rate <= positive_rate_max:
            return None
        rng = np.random.Generator(np.random.PCG64DXSM(program.seed))
        split = _stratified_split(
            valid_labels, rng, support_fraction=support_fraction,
            min_support_rows=min_support_rows, min_query_rows=min_query_rows,
            min_class_count=min_class_count_per_split,
        )
        if split is None:
            return None
        support_local, query_local = split
        support = valid_indices[support_local]
        query = valid_indices[query_local]
        normalized = _normal_family(program.family)
        code = {
            "future_event_existence": 1,
            "history_gated_future_active": 1,
            "history_gated_future_inactive": 1,
            "future_event_attribute": 2,
            "temporal_aggregate": 3,
            "interaction_response": 4,
            "multi_hop_program": 5,
        }[normalized]
        parameters: list[tuple[str, float]] = [
            ("program_semantics_code", float(code)),
            ("support_fraction", float(support_fraction)),
        ]
        if program.threshold is not None:
            parameters.append(("program_threshold", float(program.threshold)))
        task_plan = TaskPlan(
            task_id=f"task_{sample_id}_{program.program_id}",
            sample_id=sample_id,
            instance_id=database.instance_id,
            schema_id=schema.schema_id,
            mechanism=_mechanism_for_family(program.family),
            prediction_type=PredictionType.CLASSIFICATION,
            target_table_id=program.target_table_id,
            source_table_id=program.source_table_id,
            target_column_id=_SYNTHETIC_TARGET,
            source_column_id=program.attribute_column_id,
            foreign_key_id=program.foreign_key_id,
            time_column_id=program.time_column_id,
            cutoff_time=program.cutoff_time,
            horizon_end_time=program.horizon_end_time,
            split_strategy="stratified_entities",
            seed=program.seed,
            masked_column_ids=(_SYNTHETIC_TARGET,),
            observation_rules=_observation_rules(schema, program.cutoff_time),
            route_supervision=_schema_route_labels(
                schema, target_table_id=program.target_table_id,
                required_paths=((program.foreign_key_id,),),
            ),
            classification_kind=ClassificationKind.BINARY,
            threshold=program.threshold,
            realized_positive_rate=positive_rate,
            parameters=tuple(parameters),
        )
        return PlannedTask(
            plan=task_plan,
            data=TaskData(
                support_row_ids=support,
                support_labels=labels[support],
                query_row_ids=query,
                query_labels=labels[query],
            ),
        )


__all__ = [
    "LabelExpression", "ProgramNode", "CutoffPolicy", "HorizonPolicy",
    "TaskProgramPlan", "TaskProgramPlanner", "TaskExecutor",
]
