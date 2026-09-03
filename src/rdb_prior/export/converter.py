"""Convert one generated task into RDBPFN's dbinfer_bench format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rdb_prior.compilation.model import (
    ColumnKind,
    PhysicalColumn,
    PhysicalDataType,
    PhysicalForeignKey,
    PhysicalSchema,
)
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.task.artifacts import TaskArtifact
from rdb_prior.task.model import PredictionType, TaskMechanism
from rdb_prior.task.view import TaskView, build_task_view
from rdb_prior.time_bounds import NS64_MAX_SECONDS, NS64_MIN_SECONDS

from .model import RDBPFNDataset


_LABEL_COLUMN = "label"
_CUTOFF_COLUMN = "cutoff_time"


@dataclass(frozen=True, slots=True, kw_only=True)
class RDBPFNConverter:
    validation_fraction: float = 0.2
    min_validation_rows: int = 8

    def __post_init__(self) -> None:
        if isinstance(self.validation_fraction, bool) or not isinstance(
            self.validation_fraction, (int, float)
        ):
            raise TypeError("validation_fraction must be numeric")
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if isinstance(self.min_validation_rows, bool) or not isinstance(
            self.min_validation_rows, int
        ):
            raise TypeError("min_validation_rows must be an integer")
        if self.min_validation_rows < 1:
            raise ValueError("min_validation_rows must be positive")

    def convert(
        self,
        *,
        task_artifact: TaskArtifact,
        schema: PhysicalSchema,
        database: DatabaseInstance,
    ) -> RDBPFNDataset:
        plan = task_artifact.task.plan
        if plan.schema_id != schema.schema_id:
            raise ValueError("task and schema identity mismatch")
        if plan.instance_id != database.instance_id:
            raise ValueError("task and database identity mismatch")

        view = build_task_view(schema, database, plan)
        table_payloads: dict[str, dict[str, np.ndarray]] = {}
        table_metadata: list[dict[str, Any]] = []
        masked_columns: list[tuple[str, str]] = []
        foreign_keys = {
            (item.child_table_id, item.child_column_id): item
            for item in schema.foreign_keys
        }

        for table in schema.tables:
            visible_rows = view.visible_rows(table.table_id)
            columns: dict[str, np.ndarray] = {}
            column_metadata: list[dict[str, Any]] = []
            time_column: str | None = None
            for column in table.columns:
                if view.is_column_masked(table.table_id, column.column_id):
                    masked_columns.append((table.name, column.name))
                    continue
                foreign_key = foreign_keys.get((table.table_id, column.column_id))
                values = self._column_values(
                    schema=schema,
                    database=database,
                    table_id=table.table_id,
                    column=column,
                    foreign_key=foreign_key,
                    visible_rows=visible_rows,
                )
                columns[column.name] = values
                column_metadata.append(
                    self._column_metadata(
                        schema=schema,
                        database=database,
                        view=view,
                        column=column,
                        values=values,
                        foreign_key=foreign_key,
                    )
                )
                if column.kind is ColumnKind.TIME:
                    time_column = column.name
            table_payloads[table.name] = columns
            table_metadata.append(
                {
                    "name": table.name,
                    "source": f"data/{table.name}.npz",
                    "format": "numpy",
                    "columns": column_metadata,
                    "time_column": time_column,
                }
            )

        target_table = schema.table(plan.target_table_id)
        target_primary_key = target_table.primary_key
        task_splits = self._task_splits(
            task_artifact=task_artifact,
            schema=schema,
            database=database,
            target_primary_key=target_primary_key,
            visible_target_rows=view.visible_rows(target_table.table_id),
        )
        task_columns = self._task_column_metadata(
            task_artifact=task_artifact,
            target_primary_key=target_primary_key,
            target_capacity=len(
                table_payloads[target_table.name][target_primary_key.name]
            ),
            has_cutoff=_CUTOFF_COLUMN in task_splits["train"],
        )
        labels = np.concatenate(
            [task_splits[name][_LABEL_COLUMN] for name in ("train", "validation", "test")]
        )
        prediction_type = plan.prediction_type.value
        task_metadata: dict[str, Any] = {
            "name": plan.task_id,
            "source": f"{plan.task_id}/{{split}}.npz",
            "format": "numpy",
            "columns": task_columns,
            "time_column": (
                _CUTOFF_COLUMN if _CUTOFF_COLUMN in task_splits["train"] else None
            ),
            "evaluation_metric": self._evaluation_metric(
                plan.prediction_type,
                labels,
            ),
            "target_column": _LABEL_COLUMN,
            "target_table": target_table.name,
            "task_type": prediction_type,
            "mechanism": plan.mechanism.value,
        }
        if plan.prediction_type is PredictionType.CLASSIFICATION:
            task_metadata["num_classes"] = int(len(np.unique(labels)))
        if plan.composite_spec is not None:
            composite = plan.composite_spec
            task_metadata["composite_family"] = composite.family.value
            task_metadata["composite_aggregate_operators"] = [
                aggregate.operator.value
                for aggregate in composite.label_aggregates
            ]
            task_metadata["composite_path_count"] = len(
                composite.label_aggregates
            )
            task_metadata["composite_max_path_depth"] = max(
                len(aggregate.required_path)
                for aggregate in composite.label_aggregates
            )
            task_metadata["composite_predicate_count"] = sum(
                len(aggregate.predicates)
                for aggregate in composite.label_aggregates
            )
            task_metadata["composite_has_eligibility"] = (
                composite.eligibility_aggregate is not None
            )
            task_metadata["requested_positive_rate"] = (
                plan.requested_positive_rate
            )
            task_metadata["realized_positive_rate"] = plan.realized_positive_rate

        metadata = {
            "dataset_name": plan.task_id,
            "tables": table_metadata,
            "tasks": [task_metadata],
        }
        return RDBPFNDataset(
            dataset_name=plan.task_id,
            task_name=plan.task_id,
            metadata=metadata,
            tables=table_payloads,
            splits=task_splits,
            masked_columns=tuple(masked_columns),
        )

    def _column_values(
        self,
        *,
        schema: PhysicalSchema,
        database: DatabaseInstance,
        table_id: str,
        column: PhysicalColumn,
        foreign_key: PhysicalForeignKey | None,
        visible_rows: np.ndarray,
    ) -> np.ndarray:
        raw = database.table(table_id).column(column.column_id)[visible_rows]
        if column.kind is ColumnKind.PRIMARY_KEY:
            return raw.copy()
        if foreign_key is not None:
            parent = schema.table(foreign_key.parent_table_id)
            parent_keys = database.table(parent.table_id).column(
                parent.primary_key.column_id
            )
            assignments = raw.astype(np.int64, copy=False)
            result = np.empty(len(assignments), dtype=object)
            result[:] = None
            valid = assignments >= 0
            result[valid] = parent_keys[assignments[valid]].tolist()
            return result
        return _cast_values(column, raw)

    def _column_metadata(
        self,
        *,
        schema: PhysicalSchema,
        database: DatabaseInstance,
        view: TaskView,
        column: PhysicalColumn,
        values: np.ndarray,
        foreign_key: PhysicalForeignKey | None,
    ) -> dict[str, Any]:
        if column.kind is ColumnKind.PRIMARY_KEY:
            return {
                "name": column.name,
                "dtype": "primary_key",
                "capacity": int(len(values)),
            }
        if foreign_key is not None:
            parent = schema.table(foreign_key.parent_table_id)
            return {
                "name": column.name,
                "dtype": "foreign_key",
                "link_to": f"{parent.name}.{parent.primary_key.name}",
                "capacity": int(len(view.visible_rows(parent.table_id))),
            }
        dtype = _dbb_dtype(column)
        metadata: dict[str, Any] = {"name": column.name, "dtype": dtype}
        if dtype == "float":
            metadata["in_size"] = 1
        elif dtype == "category":
            metadata["num_categories"] = _category_count(values)
        return metadata

    def _task_splits(
        self,
        *,
        task_artifact: TaskArtifact,
        schema: PhysicalSchema,
        database: DatabaseInstance,
        target_primary_key: PhysicalColumn,
        visible_target_rows: np.ndarray,
    ) -> dict[str, dict[str, np.ndarray]]:
        task = task_artifact.task
        plan = task.plan
        target_table = schema.table(plan.target_table_id)
        primary_keys = database.table(target_table.table_id).column(
            target_primary_key.column_id
        )
        visible_mask = np.zeros(len(primary_keys), dtype=bool)
        visible_mask[visible_target_rows] = True
        support_visible = np.flatnonzero(
            visible_mask[task.data.support_row_ids]
        ).astype(np.int64)
        query_visible = np.flatnonzero(
            visible_mask[task.data.query_row_ids]
        ).astype(np.int64)
        if not len(support_visible):
            raise ValueError(
                "task has no visible support rows after applying observation rules"
            )
        if not len(query_visible):
            raise ValueError(
                "task has no visible query rows after applying observation rules"
            )
        support_rows = task.data.support_row_ids[support_visible]
        support_labels = task.data.support_labels[support_visible]
        query_rows = task.data.query_row_ids[query_visible]
        query_labels = task.data.query_labels[query_visible]
        support_train, support_validation = self._split_support(
            labels=support_labels,
            prediction_type=plan.prediction_type,
            temporal=plan.split_strategy == "temporal_rows",
            seed=plan.seed,
        )
        split_indices = {
            "train": support_train,
            "validation": support_validation,
            "test": np.arange(len(query_rows), dtype=np.int64),
        }
        row_sources = {
            "train": support_rows,
            "validation": support_rows,
            "test": query_rows,
        }
        label_sources = {
            "train": support_labels,
            "validation": support_labels,
            "test": query_labels,
        }
        target_time_column = next(
            (
                column
                for column in target_table.columns
                if column.kind is ColumnKind.TIME
            ),
            None,
        )
        target_times = (
            database.table(target_table.table_id).column(
                target_time_column.column_id
            )
            if target_time_column is not None
            else None
        )

        result: dict[str, dict[str, np.ndarray]] = {}
        for split_name, indices in split_indices.items():
            rows = row_sources[split_name][indices]
            columns: dict[str, np.ndarray] = {
                target_primary_key.name: primary_keys[rows].copy(),
                _LABEL_COLUMN: _cast_labels(
                    label_sources[split_name][indices],
                    plan.prediction_type,
                ),
            }
            if plan.mechanism in {
                TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE,
                TaskMechanism.HISTORY_GATED_FUTURE_ACTIVITY,
                TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE,
                TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE,
            }:
                cutoff = np.full(len(rows), int(plan.cutoff_time), dtype=np.int64)
                columns[_CUTOFF_COLUMN] = _seconds_to_datetime(cutoff)
            elif plan.cutoff_time is not None and target_times is not None:
                columns[_CUTOFF_COLUMN] = _seconds_to_datetime(target_times[rows])
            elif plan.cutoff_time is not None:
                cutoff = np.full(len(rows), int(plan.cutoff_time), dtype=np.int64)
                columns[_CUTOFF_COLUMN] = _seconds_to_datetime(cutoff)
            elif target_times is not None:
                columns[_CUTOFF_COLUMN] = _seconds_to_datetime(target_times[rows])
            result[split_name] = columns
        return result

    def _split_support(
        self,
        *,
        labels: np.ndarray,
        prediction_type: PredictionType,
        temporal: bool,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        count = len(labels)
        classes = np.unique(labels) if prediction_type is PredictionType.CLASSIFICATION else ()
        minimum_train = max(1, len(classes))
        validation_count = max(
            self.min_validation_rows,
            int(round(count * float(self.validation_fraction))),
        )
        validation_count = min(validation_count, count - minimum_train)
        if validation_count < 1:
            raise ValueError("support split is too small for train and validation")

        if prediction_type is PredictionType.CLASSIFICATION:
            validation = _stratified_validation_indices(
                labels,
                validation_count=validation_count,
                seed=seed,
            )
        elif temporal:
            validation = np.arange(
                count - validation_count,
                count,
                dtype=np.int64,
            )
        else:
            rng = np.random.Generator(np.random.PCG64DXSM(seed ^ 0x52444250))
            validation = np.sort(
                rng.choice(count, size=validation_count, replace=False)
            ).astype(np.int64)
        train_mask = np.ones(count, dtype=bool)
        train_mask[validation] = False
        train = np.flatnonzero(train_mask).astype(np.int64)
        return train, validation

    @staticmethod
    def _task_column_metadata(
        *,
        task_artifact: TaskArtifact,
        target_primary_key: PhysicalColumn,
        target_capacity: int,
        has_cutoff: bool,
    ) -> list[dict[str, Any]]:
        prediction_type = task_artifact.task.plan.prediction_type
        labels = np.concatenate(
            [
                task_artifact.task.data.support_labels,
                task_artifact.task.data.query_labels,
            ]
        )
        columns: list[dict[str, Any]] = [
            {
                "name": target_primary_key.name,
                "dtype": "primary_key",
                "capacity": int(target_capacity),
            }
        ]
        if prediction_type is PredictionType.CLASSIFICATION:
            columns.append(
                {
                    "name": _LABEL_COLUMN,
                    "dtype": "category",
                    "num_categories": int(len(np.unique(labels))),
                }
            )
        else:
            columns.append(
                {"name": _LABEL_COLUMN, "dtype": "float", "in_size": 1}
            )
        if has_cutoff:
            columns.append({"name": _CUTOFF_COLUMN, "dtype": "datetime"})
        return columns

    @staticmethod
    def _evaluation_metric(
        prediction_type: PredictionType,
        labels: np.ndarray,
    ) -> str:
        if prediction_type is PredictionType.REGRESSION:
            return "mae"
        return "auroc" if len(np.unique(labels)) == 2 else "accuracy"


def _cast_values(column: PhysicalColumn, values: np.ndarray) -> np.ndarray:
    if column.kind is ColumnKind.TIME or column.data_type is PhysicalDataType.TIMESTAMP:
        return _seconds_to_datetime(values)
    if column.data_type in {
        PhysicalDataType.BIGINT,
        PhysicalDataType.INTEGER,
        PhysicalDataType.DOUBLE,
    }:
        return values.astype(np.float32, copy=True)
    if column.data_type is PhysicalDataType.BOOLEAN:
        return values.copy()
    if column.data_type is PhysicalDataType.TEXT:
        return values.copy()
    return values.copy()


def _dbb_dtype(column: PhysicalColumn) -> str:
    if column.kind is ColumnKind.TIME or column.data_type is PhysicalDataType.TIMESTAMP:
        return "datetime"
    if column.data_type in {
        PhysicalDataType.BIGINT,
        PhysicalDataType.INTEGER,
        PhysicalDataType.DOUBLE,
    }:
        return "float"
    return "category"


def _cast_labels(values: np.ndarray, prediction_type: PredictionType) -> np.ndarray:
    if prediction_type is PredictionType.REGRESSION:
        return values.astype(np.float32, copy=True)
    return values.copy()


def _seconds_to_datetime(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64):
        return array.astype("datetime64[ns]", copy=True)
    numeric = array.astype(np.float64, copy=False)
    result = np.full(len(numeric), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = np.isfinite(numeric)
    seconds = numeric[valid].astype(np.int64)
    if np.any((seconds < NS64_MIN_SECONDS) | (seconds > NS64_MAX_SECONDS)):
        bad = seconds[(seconds < NS64_MIN_SECONDS) | (seconds > NS64_MAX_SECONDS)][0]
        raise ValueError(
            "timestamp seconds outside datetime64[ns] range "
            f"[{NS64_MIN_SECONDS}, {NS64_MAX_SECONDS}]: {int(bad)}"
        )
    representable = seconds >= 0
    converted = seconds[representable].astype("datetime64[s]").astype(
        "datetime64[ns]"
    )
    target = np.zeros(len(numeric), dtype=bool)
    target[valid] = representable
    result[target] = converted
    return result


def _category_count(values: np.ndarray) -> int:
    if values.dtype.kind == "f":
        values = values[np.isfinite(values)]
    elif values.dtype == object:
        values = np.asarray(
            [
                item
                for item in values
                if item is not None
                and not (isinstance(item, float) and np.isnan(item))
            ],
            dtype=object,
        )
    return max(1, int(len(np.unique(values))))


def _stratified_validation_indices(
    labels: np.ndarray,
    *,
    validation_count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64DXSM(seed ^ 0x56414C49))
    classes = np.unique(labels)
    selected: list[int] = []
    remaining: list[int] = []
    for label in classes:
        indices = np.flatnonzero(labels == label).astype(np.int64)
        rng.shuffle(indices)
        if len(indices) >= 2 and len(selected) < validation_count:
            selected.append(int(indices[0]))
            remaining.extend(int(value) for value in indices[1:-1])
        else:
            remaining.extend(int(value) for value in indices[:-1])
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, validation_count - len(selected))])
    if len(selected) != validation_count:
        raise ValueError("cannot construct a non-degenerate validation split")
    return np.asarray(sorted(selected), dtype=np.int64)


__all__ = ["RDBPFNConverter"]
