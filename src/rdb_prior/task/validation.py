"""Hard correctness and leakage checks for generated task artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rdb_prior.compilation.model import ColumnKind, PhysicalSchema
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.schema.spec import TableRole
from rdb_prior.task.mechanisms import mechanism_labels, _traverse_path
from rdb_prior.task.model import (
    AggregateOperator,
    CompositeTaskSpec,
    PlannedTask,
    PredictionType,
    RoutePathLabel,
    RouteRole,
    TaskMechanism,
    TaskPlan,
)
from rdb_prior.task.view import build_task_view


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskValidationIssue:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskValidationIssue:
        return cls(code=data["code"], message=data["message"])


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskValidationReport:
    task_id: str
    issues: tuple[TaskValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskValidationReport:
        return cls(
            task_id=data["task_id"],
            issues=tuple(
                TaskValidationIssue.from_dict(item)
                for item in data.get("issues", ())
            ),
        )


def validate_task(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    task: PlannedTask,
) -> TaskValidationReport:
    plan = task.plan
    data = task.data
    issues: list[TaskValidationIssue] = []
    if plan.schema_id != schema.schema_id or plan.schema_id != database.schema_id:
        issues.append(_issue("schema_identity", "task schema identity mismatch"))
    if plan.instance_id != database.instance_id:
        issues.append(_issue("instance_identity", "task instance identity mismatch"))
    try:
        target_data = database.table(plan.target_table_id)
        schema.table(plan.source_table_id)
    except KeyError:
        issues.append(_issue("table_reference", "task references an unknown table"))
        return TaskValidationReport(task_id=plan.task_id, issues=tuple(issues))

    all_rows = np.concatenate([data.support_row_ids, data.query_row_ids])
    if np.any((all_rows < 0) | (all_rows >= target_data.row_count)):
        issues.append(_issue("row_bounds", "task row ID is outside target table"))
    else:
        try:
            target_mask = build_task_view(
                schema, database, plan
            ).row_masks[plan.target_table_id]
        except (IndexError, KeyError):
            issues.append(
                _issue(
                    "observation_visibility",
                    "task observation rules cannot be applied",
                )
            )
        else:
            if not np.all(target_mask[data.support_row_ids]):
                issues.append(
                    _issue("support_visibility", "support contains hidden rows")
                )
            if not np.all(target_mask[data.query_row_ids]):
                issues.append(
                    _issue("query_visibility", "query contains hidden rows")
                )
    for labels in (data.support_labels, data.query_labels):
        if labels.dtype.kind == "f" and np.any(~np.isfinite(labels)):
            issues.append(_issue("non_finite_label", "task labels are not finite"))
        if labels.dtype.kind in {"U", "S"} and np.any(labels == ""):
            issues.append(_issue("empty_label", "task labels contain empty strings"))
    if plan.prediction_type is PredictionType.CLASSIFICATION:
        if len(np.unique(data.support_labels)) < 2:
            issues.append(_issue("support_classes", "support has fewer than two classes"))
        if len(np.unique(data.query_labels)) < 2:
            issues.append(_issue("query_classes", "query has fewer than two classes"))

    if plan.row_cutoff_time_column_id is not None:
        try:
            cutoff_column = schema.table(plan.target_table_id).column(
                plan.row_cutoff_time_column_id
            )
        except KeyError:
            issues.append(
                _issue(
                    "row_cutoff_column",
                    "row-specific cutoff column does not exist on target table",
                )
            )
        else:
            if cutoff_column.kind is not ColumnKind.TIME:
                issues.append(
                    _issue(
                        "row_cutoff_kind",
                        "row-specific cutoff column must be a time column",
                    )
                )

    issues.extend(_validate_route_supervision(schema, plan.target_table_id, plan.route_supervision))
    if not (
        plan.mechanism is TaskMechanism.RANDOM_COLUMN
        or (
            plan.mechanism is TaskMechanism.RELATION_ATTRIBUTE
            and plan.source_column_id is None
        )
    ):
        # Composite tasks carry several required paths, so the single
        # source-table endpoint check does not apply; composite path
        # continuity is validated against each AggregateSpec instead.
        if plan.composite_spec is None:
            issues.extend(_validate_required_route_endpoint(schema, plan))

    issues.extend(_validate_mechanism_labels(schema, database, task))
    if plan.mechanism in {
        TaskMechanism.RELATION_ATTRIBUTE,
        TaskMechanism.RANDOM_COLUMN,
    }:
        issues.extend(_validate_relation_attribute(schema, task))
    if plan.mechanism in {
        TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE,
        TaskMechanism.HISTORY_GATED_FUTURE_ACTIVITY,
        TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE,
        TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE,
    }:
        issues.extend(_validate_future_visibility(schema, task))
    if plan.composite_spec is not None:
        issues.extend(_validate_composite(schema, database, task))
    return TaskValidationReport(task_id=plan.task_id, issues=tuple(issues))


def _validate_composite(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    task: PlannedTask,
) -> list[TaskValidationIssue]:
    """Structural, path, column, calendar and no-leakage checks for composite tasks.

    Label recomputation itself is covered by ``_validate_mechanism_labels``
    (which calls ``mechanism_labels`` for every mechanism).  This function
    verifies the spec structure, path continuity to each AggregateSpec source
    table, TIME/numeric value columns, calendar containment of every window and
    that observation rules cut every temporal Event/Detail table at the label
    cutoff.
    """
    plan = task.plan
    spec = plan.composite_spec
    issues: list[TaskValidationIssue] = []
    if spec is None:
        return [
            _issue(
                "composite_spec_missing",
                "composite mechanism requires composite_spec",
            )
        ]
    try:
        CompositeTaskSpec.from_dict(spec.to_dict())
    except (TypeError, ValueError) as error:
        return [
            _issue(
                "composite_structure",
                f"invalid composite spec structure: {error}",
            )
        ]
    aggregates = list(spec.label_aggregates)
    if spec.eligibility_aggregate is not None:
        aggregates.append(spec.eligibility_aggregate)
    for index, aggregate in enumerate(aggregates):
        prefix = (
            "eligibility"
            if spec.eligibility_aggregate is aggregate
            else f"label[{index}]"
        )
        try:
            _path, endpoint = _traverse_path(
                schema,
                database,
                plan.target_table_id,
                aggregate.required_path,
            )
        except (ValueError, StopIteration) as error:
            issues.append(
                _issue(
                    "composite_path",
                    f"{prefix} required path is not contiguous from "
                    f"target table: {error}",
                )
            )
            continue
        if endpoint != aggregate.source_table_id:
            issues.append(
                _issue(
                    "composite_endpoint",
                    f"{prefix} required path endpoint differs from "
                    "source_table_id",
                )
            )
        try:
            source_table = schema.table(aggregate.source_table_id)
        except KeyError:
            issues.append(
                _issue(
                    "composite_source",
                    f"{prefix} source table is unknown",
                )
            )
            continue
        try:
            time_column = source_table.column(aggregate.time_column_id)
        except KeyError:
            issues.append(
                _issue(
                    "composite_time_column",
                    f"{prefix} time column does not exist on source table",
                )
            )
        else:
            if time_column.kind is not ColumnKind.TIME:
                issues.append(
                    _issue(
                        "composite_time_kind",
                        f"{prefix} time column must be a TIME column",
                    )
                )
        column_ids = {column.column_id for column in source_table.columns}
        if aggregate.value_column_id is not None:
            if aggregate.value_column_id not in column_ids:
                issues.append(
                    _issue(
                        "composite_value_column",
                        f"{prefix} value column does not exist",
                    )
                )
        if aggregate.operator in {
            AggregateOperator.SUM,
            AggregateOperator.MEAN,
            AggregateOperator.MIN,
            AggregateOperator.MAX,
        } and aggregate.value_column_id is None:
            issues.append(
                _issue(
                    "composite_value_required",
                    f"{prefix} numeric aggregate requires a value column",
                )
            )
        if (
            aggregate.operator is AggregateOperator.COUNT_DISTINCT
            and aggregate.value_column_id is None
        ):
            issues.append(
                _issue(
                    "composite_count_distinct_value",
                    f"{prefix} COUNT_DISTINCT requires a value column",
                )
            )
        if plan.cutoff_time is not None and (
            plan.db_start_seconds is not None and plan.db_end_seconds is not None
        ):
            lower = plan.cutoff_time + aggregate.window_start
            upper = plan.cutoff_time + aggregate.window_end
            if not (
                plan.db_start_seconds
                <= lower
                <= plan.db_end_seconds
                and plan.db_start_seconds
                <= upper
                <= plan.db_end_seconds
            ):
                issues.append(
                    _issue(
                        "composite_window_calendar",
                        f"{prefix} aggregate window exceeds the instance "
                        "calendar interval",
                    )
                )

    expected_rules = {
        (table.table_id, column.column_id)
        for table in schema.tables
        if table.role in {TableRole.EVENT, TableRole.DETAIL}
        for column in table.columns
        if column.kind is ColumnKind.TIME
    }
    actual_rules = {
        (rule.table_id, rule.time_column_id)
        for rule in plan.observation_rules
        if plan.cutoff_time is not None
        and rule.max_timestamp == plan.cutoff_time
    }
    if actual_rules != expected_rules:
        issues.append(
            _issue(
                "composite_visibility",
                "every temporal Event/Detail table must use the task cutoff "
                "so no post-cutoff data reaches the observation view",
            )
        )
    data = task.data
    for labels, name in (
        (data.support_labels, "support"),
        (data.query_labels, "query"),
    ):
        if np.any(labels < 0):
            issues.append(
                _issue(
                    "composite_ineligible",
                    f"{name} contains rows that are not eligibility-eligible",
                )
            )
    return issues


def _validate_relation_attribute(
    schema: PhysicalSchema,
    task: PlannedTask,
) -> list[TaskValidationIssue]:
    plan = task.plan
    data = task.data
    issues: list[TaskValidationIssue] = []
    try:
        column = schema.table(plan.target_table_id).column(
            plan.target_column_id or ""
        )
    except KeyError:
        return [_issue("target_column", "target column does not exist")]
    if column.kind is not ColumnKind.FEATURE:
        issues.append(_issue("target_kind", "attribute target is not a feature"))
    if plan.target_column_id not in plan.masked_column_ids:
        issues.append(_issue("target_leakage", "query target column is not masked"))
    return issues


def _validate_mechanism_labels(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    task: PlannedTask,
) -> list[TaskValidationIssue]:
    plan = task.plan
    data = task.data
    issues: list[TaskValidationIssue] = []
    try:
        expected = mechanism_labels(schema, database, plan)
    except (KeyError, ValueError, StopIteration) as error:
        return [_issue("mechanism_recompute", f"cannot recompute labels: {error}")]
    if not np.array_equal(data.support_labels, expected[data.support_row_ids]):
        issues.append(_issue("support_labels", "support labels are inconsistent with mechanism"))
    if not np.array_equal(data.query_labels, expected[data.query_row_ids]):
        issues.append(_issue("query_labels", "query labels are inconsistent with mechanism"))
    return issues


def _validate_future_visibility(
    schema: PhysicalSchema,
    task: PlannedTask,
) -> list[TaskValidationIssue]:
    plan = task.plan
    issues: list[TaskValidationIssue] = []
    source_rules = [
        rule
        for rule in plan.observation_rules
        if rule.table_id == plan.source_table_id
        and rule.time_column_id == plan.time_column_id
    ]
    if len(source_rules) != 1 or source_rules[0].max_timestamp != plan.cutoff_time:
        issues.append(
            _issue(
                "future_visibility",
                "source event rows are not cut off at label cutoff",
            )
        )
    expected_rules = {
        (table.table_id, column.column_id)
        for table in schema.tables
        if table.role in {TableRole.EVENT, TableRole.DETAIL}
        for column in table.columns
        if column.kind is ColumnKind.TIME
    }
    actual_rules = {
        (rule.table_id, rule.time_column_id)
        for rule in plan.observation_rules
        if rule.max_timestamp == plan.cutoff_time
    }
    if actual_rules != expected_rules:
        issues.append(
            _issue(
                "global_future_visibility",
                "every temporal Event/Detail table must use the common task cutoff",
            )
        )
    return issues


def _validate_route_supervision(
    schema: PhysicalSchema,
    target_table_id: str,
    labels: tuple[RoutePathLabel, ...],
) -> list[TaskValidationIssue]:
    issues: list[TaskValidationIssue] = []
    foreign_keys = {
        foreign_key.foreign_key_id: foreign_key
        for foreign_key in schema.foreign_keys
    }
    for label in labels:
        current = target_table_id
        visited = {current}
        for foreign_key_id in label.foreign_key_ids:
            foreign_key = foreign_keys.get(foreign_key_id)
            if foreign_key is None:
                issues.append(
                    _issue(
                        "route_foreign_key",
                        f"route supervision references unknown FK {foreign_key_id}",
                    )
                )
                break
            if current == foreign_key.parent_table_id:
                following = foreign_key.child_table_id
            elif current == foreign_key.child_table_id:
                following = foreign_key.parent_table_id
            else:
                issues.append(
                    _issue(
                        "route_continuity",
                        "route supervision is not contiguous from target table",
                    )
                )
                break
            if following in visited:
                issues.append(
                    _issue("route_cycle", "route supervision contains a cycle")
                )
                break
            visited.add(following)
            current = following
    return issues


def _validate_required_route_endpoint(
    schema: PhysicalSchema,
    plan: TaskPlan,
) -> list[TaskValidationIssue]:
    required = [
        label for label in plan.route_supervision
        if label.role is RouteRole.REQUIRED
    ]
    if not required:
        return [_issue("required_route", "task has no required route")]
    foreign_keys = {fk.foreign_key_id: fk for fk in schema.foreign_keys}
    issues: list[TaskValidationIssue] = []
    for label in required:
        current = plan.target_table_id
        for foreign_key_id in label.foreign_key_ids:
            fk = foreign_keys[foreign_key_id]
            current = (
                fk.child_table_id
                if current == fk.parent_table_id
                else fk.parent_table_id
            )
        if current != plan.source_table_id:
            issues.append(
                _issue(
                    "required_route_endpoint",
                    "required route endpoint differs from source_table_id",
                )
            )
    return issues


def _issue(code: str, message: str) -> TaskValidationIssue:
    return TaskValidationIssue(code=code, message=message)


def _arrays_equal(first: np.ndarray, second: np.ndarray) -> bool:
    if first.dtype.kind == "f" or second.dtype.kind == "f":
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


__all__ = [
    "TaskValidationIssue",
    "TaskValidationReport",
    "validate_task",
]
