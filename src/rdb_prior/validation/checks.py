"""Hard checks for executable plans and generated database instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rdb_prior.compilation.model import (
    ColumnKind,
    PhysicalDataType,
    PhysicalSchema,
)
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.instance.plan import InstancePlan
from rdb_prior.priors.registry import mechanism_ref
from rdb_prior.schema.spec import Optionality
from rdb_prior.task.program import TaskProgramPlan


@dataclass(frozen=True, slots=True, kw_only=True)
class InstanceValidationIssue:
    code: str
    message: str
    table_id: str | None = None
    column_id: str | None = None
    foreign_key_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        for name in ("table_id", "column_id", "foreign_key_id"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstanceValidationIssue:
        return cls(
            code=data["code"],
            message=data["message"],
            table_id=data.get("table_id"),
            column_id=data.get("column_id"),
            foreign_key_id=data.get("foreign_key_id"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class InstanceValidationReport:
    schema_id: str
    plan_id: str
    issues: tuple[InstanceValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "plan_id": self.plan_id,
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstanceValidationReport:
        return cls(
            schema_id=data["schema_id"],
            plan_id=data["plan_id"],
            issues=tuple(
                InstanceValidationIssue.from_dict(item)
                for item in data.get("issues", ())
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskProgramValidationReport:
    program_id: str
    issues: tuple[InstanceValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues


def validate_instance_plan(
    schema: PhysicalSchema,
    plan: InstancePlan,
) -> InstanceValidationReport:
    issues: list[InstanceValidationIssue] = []
    if schema.schema_id != plan.schema_id:
        issues.append(_issue("plan_schema_mismatch", "plan schema ID differs"))
    schema_tables = {table.table_id: table for table in schema.tables}
    plan_tables = {table.table_id: table for table in plan.tables}
    if set(schema_tables) != set(plan_tables):
        issues.append(_issue("plan_table_coverage", "plan table set differs"))
    for table_id in set(schema_tables) & set(plan_tables):
        if schema_tables[table_id].role is not plan_tables[table_id].role:
            issues.append(
                _issue(
                    "plan_role_mismatch",
                    "planned role differs from physical role",
                    table_id=table_id,
                )
            )

    positions = {table_id: index for index, table_id in enumerate(plan.generation_order)}
    expected_fks = {foreign_key.foreign_key_id for foreign_key in schema.foreign_keys}
    planned_fks: list[str] = []
    schema_fk_map = {
        foreign_key.foreign_key_id: foreign_key
        for foreign_key in schema.foreign_keys
    }
    for relation in plan.relations:
        planned_fks.extend(relation.foreign_key_ids)
        for fk_id, parent_id in zip(
            relation.foreign_key_ids,
            relation.parent_table_ids,
            strict=True,
        ):
            foreign_key = schema_fk_map.get(fk_id)
            if foreign_key is None:
                continue
            if (
                foreign_key.parent_table_id != parent_id
                or foreign_key.child_table_id != relation.child_table_id
            ):
                issues.append(
                    _issue(
                        "plan_relation_binding",
                        "relation group does not match physical FK",
                        foreign_key_id=fk_id,
                    )
                )
    if set(planned_fks) != expected_fks or len(planned_fks) != len(expected_fks):
        issues.append(
            _issue(
                "plan_relation_coverage",
                "every physical FK must occur in exactly one relation group",
            )
        )
    for foreign_key in schema.foreign_keys:
        if positions.get(foreign_key.parent_table_id, -1) >= positions.get(
            foreign_key.child_table_id, -1
        ):
            issues.append(
                _issue(
                    "plan_generation_order",
                    "parent table must be generated before child table",
                    foreign_key_id=foreign_key.foreign_key_id,
                )
            )
    issues.extend(_validate_prior_references(schema, plan))
    return InstanceValidationReport(
        schema_id=schema.schema_id,
        plan_id=plan.plan_id,
        issues=tuple(issues),
    )


def validate_task_program(
    schema: PhysicalSchema,
    plan: InstancePlan,
    program: TaskProgramPlan,
) -> TaskProgramValidationReport:
    """Verify that a pre-data task refers only to its compatible bundle."""
    issues: list[InstanceValidationIssue] = []
    tables = {item.table_id for item in schema.tables}
    foreign_keys = {item.foreign_key_id: item for item in schema.foreign_keys}
    bundles = {item.bundle_id: item for item in plan.motif_bundles}
    if program.target_table_id not in tables:
        issues.append(_issue("task_program_target_table", "target table does not exist", table_id=program.target_table_id))
    if program.source_table_id and program.source_table_id not in tables:
        issues.append(_issue("task_program_source_table", "source table does not exist", table_id=program.source_table_id))
    if program.foreign_key_id and program.foreign_key_id not in foreign_keys:
        issues.append(_issue("task_program_foreign_key", "foreign key does not exist", foreign_key_id=program.foreign_key_id))
    if plan.prior_plan_id and program.prior_plan_id and plan.prior_plan_id != program.prior_plan_id:
        issues.append(_issue("task_program_prior_plan", "program prior plan differs from instance plan"))

    selected = []
    for bundle_id in program.required_bundle_ids:
        bundle = bundles.get(bundle_id)
        if bundle is None:
            issues.append(_issue("task_program_bundle", f"unknown required bundle: {bundle_id}"))
        else:
            selected.append(bundle)
    if not selected:
        issues.append(_issue("task_program_no_bundle", "program has no resolved mechanism bundle"))
    available: set[str] = set()
    for bundle in selected:
        if bundle.compatible_task_families and program.family not in bundle.compatible_task_families:
            issues.append(_issue("task_program_incompatible_family", f"{program.family} is not compatible with bundle {bundle.bundle_id}"))
        available.add(bundle.bundle_id)
        available.add(bundle.population_mechanism)
        available.update(
            mechanism_id
            for binding in bundle.node_bindings
            for mechanism_id in binding.mechanism_ids
        )
        available.update(binding.mechanism_id for binding in bundle.edge_bindings)
    available.update(_task_requirement_ids(program.family))
    missing = set(program.required_mechanism_ids) - available
    for mechanism_id in sorted(missing):
        issues.append(_issue("task_program_mechanism", f"unknown or incompatible mechanism: {mechanism_id}"))
    return TaskProgramValidationReport(program_id=program.program_id, issues=tuple(issues))


def validate_nuisance_target_leakage(
    plan: InstancePlan,
    target_column_id: str | None,
) -> tuple[InstanceValidationIssue, ...]:
    """Reject an unmasked nuisance feature derived from a task target column."""
    if target_column_id is None:
        return ()
    issues: list[InstanceValidationIssue] = []
    for column_plan in plan.nuisance_plan.column_roles:
        if (
            column_plan.column_id != target_column_id
            and target_column_id in column_plan.source_column_ids
        ):
            issues.append(
                _issue(
                    "nuisance_target_leakage",
                    "nuisance feature is derived from the task target column",
                    column_id=column_plan.column_id,
                )
            )
    for missingness in plan.nuisance_plan.missingness_plans:
        if (
            missingness.column_id != target_column_id
            and target_column_id in missingness.driver_column_ids
        ):
            issues.append(
                _issue(
                    "nuisance_target_leakage",
                    "nuisance missingness is driven by the task target column",
                    column_id=missingness.column_id,
                )
            )
    return tuple(issues)


def _validate_prior_references(
    schema: PhysicalSchema,
    plan: InstancePlan,
) -> list[InstanceValidationIssue]:
    issues: list[InstanceValidationIssue] = []
    table_ids = {item.table_id for item in schema.tables}
    foreign_key_ids = {item.foreign_key_id for item in schema.foreign_keys}
    column_ids = {column.column_id for table in schema.tables for column in table.columns}
    known_shared = {item.state_id for item in plan.shared_states}
    known_temporal = {item.state_id for item in plan.temporal_state_plans}

    bundle_ids: set[str] = set()
    table_claims: dict[str, set[str]] = {}
    relation_claims: dict[str, set[str]] = {}
    for bundle in plan.motif_bundles:
        if bundle.bundle_id in bundle_ids:
            issues.append(_issue("duplicate_motif_bundle", "motif bundle ID is duplicated"))
        bundle_ids.add(bundle.bundle_id)
        for reference in (
            bundle.attribute_mechanism,
            bundle.relation_mechanism,
            bundle.temporal_mechanism,
            bundle.process_mechanism,
        ):
            try:
                registered = mechanism_ref(reference.kind)
            except ValueError:
                issues.append(_issue("unknown_mechanism_reference", f"unknown mechanism: {reference.kind}"))
            else:
                if registered.version != reference.version:
                    issues.append(_issue("mechanism_version_mismatch", f"unexpected version for mechanism: {reference.kind}"))
        unknown_states = set(bundle.shared_state_ids) - known_shared
        for state_id in sorted(unknown_states):
            issues.append(_issue("unknown_shared_state", f"bundle references unknown state: {state_id}"))
        for binding in bundle.node_bindings:
            if binding.table_id not in table_ids:
                issues.append(_issue("bundle_table_binding", "bundle table binding does not exist", table_id=binding.table_id))
                continue
            table_claims.setdefault(binding.table_id, set()).add(bundle.attribute_mechanism.kind)
        for binding in bundle.edge_bindings:
            if binding.foreign_key_id not in foreign_key_ids:
                issues.append(_issue("bundle_relation_binding", "bundle FK binding does not exist", foreign_key_id=binding.foreign_key_id))
                continue
            relation_claims.setdefault(binding.foreign_key_id, set()).add(bundle.relation_mechanism.kind)
    for table_id, claims in table_claims.items():
        non_legacy = claims - {"legacy_scm"}
        if len(non_legacy) > 1:
            issues.append(_issue("motif_bundle_conflict", "table has conflicting attribute bundles", table_id=table_id))
    for foreign_key_id, claims in relation_claims.items():
        non_legacy = claims - {"latent_affinity"}
        if len(non_legacy) > 1:
            issues.append(_issue("motif_bundle_conflict", "relation has conflicting relation bundles", foreign_key_id=foreign_key_id))

    _validate_mechanism_columns(issues, plan, column_ids, known_shared)
    for mechanism in plan.population_mechanisms:
        if mechanism.table_id not in table_ids:
            issues.append(_issue("population_table_reference", "population mechanism table does not exist", table_id=mechanism.table_id))
        if mechanism.parent_table_id is not None and mechanism.parent_table_id not in table_ids:
            issues.append(_issue("population_parent_table_reference", "population parent table does not exist", table_id=mechanism.table_id))
        for state_id in set(mechanism.state_ids) - known_shared:
            issues.append(_issue("population_state_reference", f"unknown population state: {state_id}", table_id=mechanism.table_id))
    for process in plan.temporal_processes:
        if process.table_id not in table_ids:
            issues.append(_issue("temporal_table_reference", "temporal process table does not exist", table_id=process.table_id))
        for state_id in set(process.state_ids) - known_shared:
            issues.append(_issue("temporal_state_reference", f"unknown shared state: {state_id}", table_id=process.table_id))
        for state_id in set(process.temporal_state_ids) - known_temporal:
            issues.append(_issue("temporal_trajectory_reference", f"unknown temporal state: {state_id}", table_id=process.table_id))
        for column_id in set(process.covariate_column_ids) - column_ids:
            issues.append(_issue("temporal_covariate_reference", f"unknown temporal covariate: {column_id}", table_id=process.table_id, column_id=column_id))
    nuisance = plan.nuisance_plan
    try:
        registered = mechanism_ref(nuisance.mechanism.kind)
    except ValueError:
        issues.append(_issue("unknown_mechanism_reference", f"unknown nuisance mechanism: {nuisance.mechanism.kind}"))
    else:
        if registered.version != nuisance.mechanism.version:
            issues.append(_issue("mechanism_version_mismatch", "unexpected nuisance mechanism version"))
    for column_plan in nuisance.column_roles:
        if column_plan.column_id not in column_ids:
            issues.append(_issue("nuisance_column_reference", "nuisance column does not exist", column_id=column_plan.column_id))
        for source_id in set(column_plan.source_column_ids) - column_ids:
            issues.append(_issue("nuisance_source_reference", f"unknown nuisance source: {source_id}", column_id=column_plan.column_id))
    return issues


def _validate_mechanism_columns(
    issues: list[InstanceValidationIssue],
    plan: InstancePlan,
    column_ids: set[str],
    known_shared: set[str],
) -> None:
    graph: dict[str, tuple[str, ...]] = {}
    for mechanism in plan.column_mechanisms:
        if mechanism.column_id not in column_ids:
            issues.append(_issue("column_mechanism_reference", "mechanism column does not exist", column_id=mechanism.column_id))
            continue
        if mechanism.column_id in graph:
            issues.append(_issue("duplicate_column_mechanism", "multiple mechanisms target one column", column_id=mechanism.column_id))
        graph[mechanism.column_id] = mechanism.parent_column_ids
        for parent_id in set(mechanism.parent_column_ids) - column_ids:
            issues.append(_issue("column_parent_reference", f"unknown parent column: {parent_id}", column_id=mechanism.column_id))
        for state_id in set(mechanism.shared_state_ids) - known_shared:
            issues.append(_issue("column_state_reference", f"unknown shared state: {state_id}", column_id=mechanism.column_id))
    visiting: set[str] = set()
    complete: set[str] = set()
    for column_id in graph:
        if _column_dag_cycle(column_id, graph, visiting, complete):
            issues.append(_issue("column_dag_cycle", "column mechanism DAG contains a cycle", column_id=column_id))
            break


def _column_dag_cycle(
    column_id: str,
    graph: Mapping[str, tuple[str, ...]],
    visiting: set[str],
    complete: set[str],
) -> bool:
    if column_id in complete:
        return False
    if column_id in visiting:
        return True
    visiting.add(column_id)
    cyclic = any(
        parent_id in graph and _column_dag_cycle(parent_id, graph, visiting, complete)
        for parent_id in graph[column_id]
    )
    visiting.remove(column_id)
    if not cyclic:
        complete.add(column_id)
    return cyclic


def _task_requirement_ids(family: str) -> set[str]:
    requirements = {
        "future_event_existence": {"event_count", "event_time"},
        "history_gated_future_active": {"event_count", "event_time", "history"},
        "history_gated_future_inactive": {"event_count", "event_time", "history"},
        "future_event_attribute": {"event_count", "event_time", "event_attribute", "event_attributes"},
        "temporal_aggregate": {"event_count"},
        "interaction_response": {"interaction"},
        "multi_hop_program": {"event_count", "relation_path"},
    }
    return requirements.get(family, set())


def validate_database_instance(
    schema: PhysicalSchema,
    plan: InstancePlan,
    database: DatabaseInstance,
) -> InstanceValidationReport:
    plan_report = validate_instance_plan(schema, plan)
    issues = list(plan_report.issues)
    if database.schema_id != schema.schema_id or database.plan_id != plan.plan_id:
        issues.append(
            _issue("database_identity", "database IDs do not match schema and plan")
        )
    database_tables = {table.table_id: table for table in database.tables}
    schema_table_ids = {table.table_id for table in schema.tables}
    if set(database_tables) != schema_table_ids:
        issues.append(_issue("database_table_coverage", "database table set differs"))

    for table in schema.tables:
        data = database_tables.get(table.table_id)
        if data is None:
            continue
        expected_rows = plan.table(table.table_id).population.row_count
        if data.row_count != expected_rows:
            issues.append(
                _issue(
                    "row_count_mismatch",
                    f"expected {expected_rows} rows, found {data.row_count}",
                    table_id=table.table_id,
                )
            )
        expected_columns = {column.column_id for column in table.columns}
        if set(data.columns) != expected_columns:
            issues.append(
                _issue(
                    "column_coverage",
                    "generated columns differ from physical schema",
                    table_id=table.table_id,
                )
            )
            continue
        for column in table.columns:
            values = data.column(column.column_id)
            issues.extend(_validate_column(table.table_id, column, values))
            if (
                column.kind is ColumnKind.TIME
                and plan.calendar_start_seconds is not None
            ):
                calendar = np.asarray(values, dtype=np.int64)
                if np.any(
                    (calendar < plan.calendar_start_seconds)
                    | (calendar > plan.calendar_end_seconds)
                ):
                    issues.append(
                        _issue(
                            "time_out_of_calendar",
                            "TIME value outside database calendar interval",
                            table_id=table.table_id,
                            column_id=column.column_id,
                        )
                    )

    for foreign_key in schema.foreign_keys:
        child = database_tables.get(foreign_key.child_table_id)
        parent = database_tables.get(foreign_key.parent_table_id)
        if child is None or parent is None:
            continue
        values = child.column(foreign_key.child_column_id)
        invalid = (values < -1) | (values >= parent.row_count)
        if np.any(invalid):
            issues.append(
                _issue(
                    "foreign_key_bounds",
                    "foreign key contains an invalid parent row index",
                    foreign_key_id=foreign_key.foreign_key_id,
                )
            )
        if foreign_key.optionality is Optionality.REQUIRED and np.any(values < 0):
            issues.append(
                _issue(
                    "required_foreign_key_missing",
                    "required foreign key contains missing values",
                    foreign_key_id=foreign_key.foreign_key_id,
                )
            )

    for relation in plan.relations:
        if relation.family != "affinity_bridge" or any(relation.optional_rates):
            continue
        child = database_tables.get(relation.child_table_id)
        if child is None:
            continue
        fk_map = {
            foreign_key.foreign_key_id: foreign_key
            for foreign_key in schema.foreign_keys
        }
        matrix = np.column_stack(
            [child.column(fk_map[fk_id].child_column_id) for fk_id in relation.foreign_key_ids]
        )
        combinations = int(
            np.prod(
                [database_tables[parent_id].row_count for parent_id in relation.parent_table_ids]
            )
        )
        if len(matrix) <= combinations and len(np.unique(matrix, axis=0)) != len(matrix):
            issues.append(
                _issue(
                    "bridge_duplicate_tuple",
                    "bridge relation contains duplicate parent tuples",
                    table_id=relation.child_table_id,
                )
            )
    return InstanceValidationReport(
        schema_id=schema.schema_id,
        plan_id=plan.plan_id,
        issues=tuple(issues),
    )


def _validate_column(table_id: str, column: Any, values: np.ndarray) -> list[InstanceValidationIssue]:
    issues: list[InstanceValidationIssue] = []
    if values.dtype == object:
        issues.append(
            _issue(
                "object_dtype",
                "object arrays cannot be persisted safely",
                table_id=table_id,
                column_id=column.column_id,
            )
        )
        return issues
    if column.kind is ColumnKind.PRIMARY_KEY:
        expected = np.arange(len(values), dtype=np.int64)
        if not np.array_equal(values, expected):
            issues.append(
                _issue(
                    "primary_key_sequence",
                    "primary key must be a contiguous row index",
                    table_id=table_id,
                    column_id=column.column_id,
                )
            )
        return issues
    if column.kind is ColumnKind.FOREIGN_KEY:
        if values.dtype.kind not in {"i", "u"}:
            issues.append(
                _issue(
                    "foreign_key_dtype",
                    "foreign key must use an integer dtype",
                    table_id=table_id,
                    column_id=column.column_id,
                )
            )
        return issues

    if values.dtype.kind == "f":
        present = values[~np.isnan(values)]
        if np.any(~np.isfinite(present)):
            issues.append(
                _issue(
                    "non_finite_feature",
                    "feature contains infinite values",
                    table_id=table_id,
                    column_id=column.column_id,
                )
            )
    elif values.dtype.kind in {"U", "S"}:
        present = values[values != ""]
    else:
        present = values
    if len(present) == 0:
        issues.append(
            _issue(
                "all_missing_feature",
                "feature contains no observed values",
                table_id=table_id,
                column_id=column.column_id,
            )
        )
    if column.unique and len(np.unique(present)) != len(present):
        issues.append(
            _issue(
                "unique_feature_duplicate",
                "unique column contains duplicate observed values",
                table_id=table_id,
                column_id=column.column_id,
            )
        )
    if column.data_type is PhysicalDataType.TEXT and values.dtype.kind not in {"U", "S"}:
        issues.append(
            _issue(
                "text_dtype",
                "text column must use a fixed-width string dtype",
                table_id=table_id,
                column_id=column.column_id,
            )
        )
    return issues


def _issue(
    code: str,
    message: str,
    *,
    table_id: str | None = None,
    column_id: str | None = None,
    foreign_key_id: str | None = None,
) -> InstanceValidationIssue:
    return InstanceValidationIssue(
        code=code,
        message=message,
        table_id=table_id,
        column_id=column_id,
        foreign_key_id=foreign_key_id,
    )


__all__ = [
    "InstanceValidationIssue",
    "InstanceValidationReport",
    "TaskProgramValidationReport",
    "validate_instance_plan",
    "validate_database_instance",
    "validate_nuisance_target_leakage",
    "validate_task_program",
]
