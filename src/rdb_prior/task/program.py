"""Pre-data task programs and their deterministic post-data executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rdb_prior.compilation.model import ColumnKind, PhysicalSchema
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.instance.plan import InstancePlan
from rdb_prior.priors.model import DatabasePriorPlan, PriorFamily
from rdb_prior.runtime import RuntimeContext
from rdb_prior.task.mechanisms import (
    _SYNTHETIC_TARGET,
    _observation_rules,
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


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskProgramPlan:
    program_id: str
    family: str
    target_table_id: str
    source_table_id: str
    foreign_key_id: str
    time_column_id: str
    cutoff_time: int
    horizon_end_time: int
    required_mechanism_ids: tuple[str, ...]
    seed: int
    prior_plan_id: str
    support_fraction: float = 0.7
    min_support_rows: int = 8
    min_query_rows: int = 4
    min_class_count_per_split: int = 1

    def __post_init__(self) -> None:
        for name in ("program_id", "family", "target_table_id", "source_table_id", "foreign_key_id", "time_column_id", "prior_plan_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.horizon_end_time <= self.cutoff_time:
            raise ValueError("horizon_end_time must be after cutoff_time")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.required_mechanism_ids or not all(isinstance(item, str) and item for item in self.required_mechanism_ids):
            raise ValueError("required_mechanism_ids must be non-empty strings")
        if not 0.0 < self.support_fraction < 1.0:
            raise ValueError("support_fraction must be in (0, 1)")
        for name in ("min_support_rows", "min_query_rows", "min_class_count_per_split"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {"program_id": self.program_id, "family": self.family, "target_table_id": self.target_table_id, "source_table_id": self.source_table_id, "foreign_key_id": self.foreign_key_id, "time_column_id": self.time_column_id, "cutoff_time": self.cutoff_time, "horizon_end_time": self.horizon_end_time, "required_mechanism_ids": list(self.required_mechanism_ids), "seed": self.seed, "prior_plan_id": self.prior_plan_id, "support_fraction": self.support_fraction, "min_support_rows": self.min_support_rows, "min_query_rows": self.min_query_rows, "min_class_count_per_split": self.min_class_count_per_split}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskProgramPlan":
        return cls(program_id=data["program_id"], family=data["family"], target_table_id=data["target_table_id"], source_table_id=data["source_table_id"], foreign_key_id=data["foreign_key_id"], time_column_id=data["time_column_id"], cutoff_time=data["cutoff_time"], horizon_end_time=data["horizon_end_time"], required_mechanism_ids=tuple(data["required_mechanism_ids"]), seed=data["seed"], prior_plan_id=data["prior_plan_id"], support_fraction=data.get("support_fraction", 0.7), min_support_rows=data.get("min_support_rows", 8), min_query_rows=data.get("min_query_rows", 4), min_class_count_per_split=data.get("min_class_count_per_split", 1))


class TaskProgramPlanner:
    def plan(self, *, schema: PhysicalSchema, instance_plan: InstancePlan, prior_plan: DatabasePriorPlan, runtime: RuntimeContext) -> tuple[TaskProgramPlan, ...]:
        if prior_plan.family is not PriorFamily.TEMPORAL_EVENT:
            return ()
        start, end = instance_plan.calendar_start_seconds, instance_plan.calendar_end_seconds
        if start is None or end is None:
            raise ValueError("pre-data task programs require a calendar")
        candidates = [bundle for bundle in prior_plan.motif_bundles if bundle.family is PriorFamily.TEMPORAL_EVENT and "entity_future_event_existence" in bundle.compatible_task_families]
        if not candidates:
            raise ValueError("temporal prior has no compatible future-event bundle")
        span = end - start
        policy = prior_plan.task_policy
        programs: list[TaskProgramPlan] = []
        for index in range(policy.programs_per_database):
            bundle = candidates[index % len(candidates)]
            parameters = dict(bundle.parameters)
            source_table_id = str(parameters["event_table_id"])
            target_table_id = str(parameters["entity_table_id"])
            time_column_id = next(column.column_id for column in schema.table(source_table_id).columns if column.kind is ColumnKind.TIME)
            rng = runtime.numpy_rng("task-program", index)
            cutoff_fraction = float(rng.uniform(policy.cutoff_fraction_min, policy.cutoff_fraction_max))
            horizon_fraction = float(rng.uniform(policy.horizon_fraction_min, policy.horizon_fraction_max))
            cutoff = start + int(round(span * cutoff_fraction))
            horizon = min(end, cutoff + max(1, int(round(span * horizon_fraction))))
            if horizon <= cutoff:
                raise ValueError("sampled task program has empty horizon")
            programs.append(TaskProgramPlan(program_id=f"program_{instance_plan.sample_id}_{index:03d}", family="entity_future_event_existence", target_table_id=target_table_id, source_table_id=source_table_id, foreign_key_id=bundle.edge_bindings[0].foreign_key_id, time_column_id=time_column_id, cutoff_time=cutoff, horizon_end_time=horizon, required_mechanism_ids=(bundle.bundle_id, "event_count", "event_time"), seed=runtime.seed("task-program", index), prior_plan_id=prior_plan.plan_id))
        return tuple(programs)


class TaskExecutor:
    def execute(self, *, sample_id: str, schema: PhysicalSchema, database: DatabaseInstance, program: TaskProgramPlan, support_fraction: float | None = None, min_support_rows: int | None = None, min_query_rows: int | None = None, min_class_count_per_split: int | None = None, positive_rate_min: float = 0.0, positive_rate_max: float = 1.0) -> PlannedTask | None:
        if not 0.0 <= positive_rate_min <= positive_rate_max <= 1.0:
            raise ValueError("positive-rate bounds must lie in [0, 1]")
        support_fraction = program.support_fraction if support_fraction is None else support_fraction
        min_support_rows = program.min_support_rows if min_support_rows is None else min_support_rows
        min_query_rows = program.min_query_rows if min_query_rows is None else min_query_rows
        min_class_count_per_split = program.min_class_count_per_split if min_class_count_per_split is None else min_class_count_per_split
        foreign_key = next(item for item in schema.foreign_keys if item.foreign_key_id == program.foreign_key_id)
        event = database.table(program.source_table_id)
        assignments = event.column(foreign_key.child_column_id)
        times = event.column(program.time_column_id)
        labels = np.zeros(database.table(program.target_table_id).row_count, dtype=np.int8)
        selected = (assignments >= 0) & (times > program.cutoff_time) & (times <= program.horizon_end_time)
        labels[np.unique(assignments[selected])] = 1
        positive_rate = float(np.mean(labels))
        if not positive_rate_min <= positive_rate <= positive_rate_max:
            return None
        rng = np.random.Generator(np.random.PCG64DXSM(program.seed))
        split = _stratified_split(labels, rng, support_fraction=support_fraction, min_support_rows=min_support_rows, min_query_rows=min_query_rows, min_class_count=min_class_count_per_split)
        if split is None:
            return None
        support, query = split
        task_plan = TaskPlan(task_id=f"task_{sample_id}_{program.program_id}", sample_id=sample_id, instance_id=database.instance_id, schema_id=schema.schema_id, mechanism=TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE, prediction_type=PredictionType.CLASSIFICATION, target_table_id=program.target_table_id, source_table_id=program.source_table_id, target_column_id=_SYNTHETIC_TARGET, foreign_key_id=program.foreign_key_id, time_column_id=program.time_column_id, cutoff_time=program.cutoff_time, horizon_end_time=program.horizon_end_time, split_strategy="stratified_entities", seed=program.seed, masked_column_ids=(_SYNTHETIC_TARGET,), observation_rules=_observation_rules(schema, program.cutoff_time), route_supervision=_schema_route_labels(schema, target_table_id=program.target_table_id, required_paths=((program.foreign_key_id,),)), classification_kind=ClassificationKind.BINARY, realized_positive_rate=positive_rate, parameters=(("support_fraction", support_fraction),))
        return PlannedTask(plan=task_plan, data=TaskData(support_row_ids=support, support_labels=labels[support], query_row_ids=query, query_labels=labels[query]))


__all__ = ["TaskProgramPlan", "TaskProgramPlanner", "TaskExecutor"]
