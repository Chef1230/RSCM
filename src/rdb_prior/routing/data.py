"""Leakage-safe task loading and tensorization for sparse routing."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch

from rdb_prior.artifacts import load_instance_artifact, load_schema_artifact
from rdb_prior.compilation.model import (
    ColumnKind,
    PhysicalColumn,
    PhysicalDataType,
    PhysicalSchema,
)
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.task.artifacts import TaskArtifact, load_task_artifact
from rdb_prior.task.model import PredictionType, RouteRole, TaskMechanism
from rdb_prior.task.view import TaskView, build_task_view

from .catalog import (
    SchemaPath,
    enumerate_schema_paths,
    path_feature_vector,
    path_similarity_matrix,
)
from .config import RouterModelConfig


_TYPE_INDEX = {
    PhysicalDataType.BIGINT: 0,
    PhysicalDataType.INTEGER: 1,
    PhysicalDataType.DOUBLE: 2,
    PhysicalDataType.BOOLEAN: 3,
    PhysicalDataType.TEXT: 4,
    PhysicalDataType.TIMESTAMP: 5,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class RawRoutingTask:
    artifact_path: Path
    task_artifact: TaskArtifact
    schema: PhysicalSchema
    database: DatabaseInstance
    view: TaskView


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutedTaskTensors:
    task_id: str
    prediction_type: PredictionType
    row_ids: torch.Tensor
    target_values: torch.Tensor
    target_missing: torch.Tensor
    target_type_ids: torch.Tensor
    target_column_features: torch.Tensor
    support_mask: torch.Tensor
    labels: torch.Tensor
    num_classes: int
    class_values: tuple[Any, ...]
    path_features: torch.Tensor
    path_costs: torch.Tensor
    path_similarity: torch.Tensor
    route_targets: torch.Tensor
    route_weights: torch.Tensor
    source_values: torch.Tensor
    source_missing: torch.Tensor
    source_row_mask: torch.Tensor
    source_positions: torch.Tensor
    source_type_ids: torch.Tensor
    source_column_features: torch.Tensor
    source_column_mask: torch.Tensor
    path_signatures: tuple[str, ...]
    source_column_ids: tuple[tuple[str, ...], ...]
    label_center: float = 0.0
    label_scale: float = 1.0

    def to(self, device: torch.device | str) -> RoutedTaskTensors:
        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device) if torch.is_tensor(value) else value
        return RoutedTaskTensors(**values)

    @property
    def query_mask(self) -> torch.Tensor:
        return ~self.support_mask


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutedTaskBatch:
    """Padded heterogeneous routing tasks for one GPU optimizer step."""

    task_ids: tuple[str, ...]
    prediction_types: tuple[PredictionType, ...]
    num_classes: tuple[int, ...]
    row_ids: torch.Tensor
    target_values: torch.Tensor
    target_missing: torch.Tensor
    target_type_ids: torch.Tensor
    target_column_features: torch.Tensor
    target_column_mask: torch.Tensor
    row_mask: torch.Tensor
    support_mask: torch.Tensor
    labels: torch.Tensor
    path_features: torch.Tensor
    path_mask: torch.Tensor
    path_costs: torch.Tensor
    path_similarity: torch.Tensor
    route_targets: torch.Tensor
    route_weights: torch.Tensor
    source_values: torch.Tensor
    source_missing: torch.Tensor
    source_row_mask: torch.Tensor
    source_positions: torch.Tensor
    source_type_ids: torch.Tensor
    source_column_features: torch.Tensor
    source_column_mask: torch.Tensor

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> RoutedTaskBatch:
        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.to(device, non_blocking=non_blocking)
                if torch.is_tensor(value)
                else value
            )
        return RoutedTaskBatch(**values)

    def pin_memory(self) -> RoutedTaskBatch:
        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.pin_memory() if torch.is_tensor(value) else value
        return RoutedTaskBatch(**values)

    @property
    def query_mask(self) -> torch.Tensor:
        return self.row_mask & ~self.support_mask


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutingTaskReference:
    artifact_path: Path
    schema_id: str
    task_id: str


class RoutingTaskStore:
    """Lazy manifest reader with a shared LRU for schema/database artifacts."""

    def __init__(
        self,
        task_manifest: Path,
        *,
        start_index: int = 0,
        task_count: int | None = None,
        cache_size: int = 16,
    ) -> None:
        self.task_manifest = Path(task_manifest)
        self.cache_size = max(0, int(cache_size))
        self.references = load_routing_task_references(
            self.task_manifest,
            start_index=start_index,
            task_count=task_count,
        )
        self._instance_cache: OrderedDict[Path, Any] = OrderedDict()
        self._schema_cache: OrderedDict[Path, Any] = OrderedDict()
        self._cache_lock = Lock()

    def __len__(self) -> int:
        return len(self.references)

    def load(self, reference: RoutingTaskReference) -> RawRoutingTask:
        task_path = reference.artifact_path
        task_artifact = load_task_artifact(task_path)
        instance_path = _resolve_reference(
            task_path, task_artifact.instance_artifact
        )
        schema_path = _resolve_reference(task_path, task_artifact.schema_artifact)
        instance = self._cached_load(
            instance_path,
            load_instance_artifact,
            self._instance_cache,
        )
        schema_artifact = self._cached_load(
            schema_path,
            load_schema_artifact,
            self._schema_cache,
        )
        schema = schema_artifact.compilation.schema
        view = build_task_view(schema, instance.database, task_artifact.task.plan)
        return RawRoutingTask(
            artifact_path=task_path,
            task_artifact=task_artifact,
            schema=schema,
            database=instance.database,
            view=view,
        )

    def _cached_load(self, path: Path, loader, cache: OrderedDict[Path, Any]):
        if not self.cache_size:
            return loader(path)
        with self._cache_lock:
            cached = cache.get(path)
            if cached is not None:
                cache.move_to_end(path)
                return cached
        loaded = loader(path)
        with self._cache_lock:
            cached = cache.get(path)
            if cached is not None:
                cache.move_to_end(path)
                return cached
            cache[path] = loaded
            while len(cache) > self.cache_size:
                cache.popitem(last=False)
        return loaded


def load_routing_task_references(
    task_manifest: Path,
    *,
    start_index: int = 0,
    task_count: int | None = None,
) -> tuple[RoutingTaskReference, ...]:
    payload = json.loads(Path(task_manifest).read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "relational_task_manifest":
        raise ValueError("input is not a relational task manifest")
    if payload.get("artifact_version") not in {1, 2}:
        raise ValueError("unsupported relational task manifest version")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("task manifest entries must be a list")
    selected = entries[start_index:]
    if task_count is not None:
        selected = selected[:task_count]
    references: list[RoutingTaskReference] = []
    for entry in selected:
        if not isinstance(entry, dict) or not isinstance(entry.get("artifact"), str):
            raise ValueError("task manifest entry is malformed")
        schema_id = entry.get("schema_id")
        task_id = entry.get("task_id")
        if not isinstance(schema_id, str) or not schema_id:
            raise ValueError("task manifest entry is missing schema_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task manifest entry is missing task_id")
        references.append(
            RoutingTaskReference(
                artifact_path=(Path(task_manifest).parent / entry["artifact"]).resolve(),
                schema_id=schema_id,
                task_id=task_id,
            )
        )
    return tuple(references)


def load_routing_tasks(
    task_manifest: Path,
    *,
    start_index: int = 0,
    task_count: int | None = None,
) -> tuple[RawRoutingTask, ...]:
    store = RoutingTaskStore(
        task_manifest,
        start_index=start_index,
        task_count=task_count,
    )
    return tuple(store.load(reference) for reference in store.references)


def collate_routed_tasks(tasks: tuple[RoutedTaskTensors, ...] | list[RoutedTaskTensors]) -> RoutedTaskBatch:
    """Pad task-local row/column/path axes without merging their semantics."""
    if not tasks:
        raise ValueError("cannot collate an empty routed task batch")
    batch_size = len(tasks)
    max_rows = max(task.target_values.shape[0] for task in tasks)
    max_targets = max(task.target_values.shape[1] for task in tasks)
    max_paths = max(task.path_features.shape[0] for task in tasks)
    max_columns = max(task.source_values.shape[2] for task in tasks)
    max_samples = max(task.source_values.shape[3] for task in tasks)
    path_feature_dim = tasks[0].path_features.shape[-1]
    column_feature_dim = tasks[0].source_column_features.shape[-1]

    target_values = torch.zeros(batch_size, max_rows, max_targets)
    row_ids = torch.full((batch_size, max_rows), -1, dtype=torch.long)
    target_missing = torch.ones(batch_size, max_rows, max_targets, dtype=torch.bool)
    target_type_ids = torch.zeros(batch_size, max_targets, dtype=torch.long)
    target_column_features = torch.zeros(batch_size, max_targets, column_feature_dim)
    target_column_mask = torch.zeros(batch_size, max_targets, dtype=torch.bool)
    row_mask = torch.zeros(batch_size, max_rows, dtype=torch.bool)
    support_mask = torch.zeros(batch_size, max_rows, dtype=torch.bool)
    labels = torch.zeros(batch_size, max_rows, dtype=torch.float32)
    path_features = torch.zeros(batch_size, max_paths, path_feature_dim)
    path_mask = torch.zeros(batch_size, max_paths, dtype=torch.bool)
    path_costs = torch.zeros(batch_size, max_paths, 3)
    path_similarity = torch.zeros(batch_size, max_paths, max_paths)
    route_targets = torch.zeros(batch_size, max_paths)
    route_weights = torch.zeros(batch_size, max_paths)
    source_values = torch.zeros(
        batch_size, max_rows, max_paths, max_columns, max_samples
    )
    source_missing = torch.ones_like(source_values, dtype=torch.bool)
    source_row_mask = torch.zeros_like(source_values, dtype=torch.bool)
    source_positions = torch.zeros(batch_size, max_rows, max_paths, max_samples)
    source_type_ids = torch.zeros(
        batch_size, max_paths, max_columns, dtype=torch.long
    )
    source_column_features = torch.zeros(
        batch_size, max_paths, max_columns, column_feature_dim
    )
    source_column_mask = torch.zeros(
        batch_size, max_paths, max_columns, dtype=torch.bool
    )

    for index, task in enumerate(tasks):
        rows, targets = task.target_values.shape
        paths = task.path_features.shape[0]
        columns = task.source_values.shape[2]
        samples = task.source_values.shape[3]
        row_ids[index, :rows] = task.row_ids
        target_values[index, :rows, :targets] = task.target_values
        target_missing[index, :rows, :targets] = task.target_missing
        target_type_ids[index, :targets] = task.target_type_ids
        target_column_features[index, :targets] = task.target_column_features
        target_column_mask[index, :targets] = True
        row_mask[index, :rows] = True
        support_mask[index, :rows] = task.support_mask
        labels[index, :rows] = task.labels.float()
        path_features[index, :paths] = task.path_features
        path_mask[index, :paths] = True
        path_costs[index, :paths] = task.path_costs
        path_similarity[index, :paths, :paths] = task.path_similarity
        route_targets[index, :paths] = task.route_targets
        route_weights[index, :paths] = task.route_weights
        source_values[index, :rows, :paths, :columns, :samples] = task.source_values
        source_missing[index, :rows, :paths, :columns, :samples] = task.source_missing
        source_row_mask[index, :rows, :paths, :columns, :samples] = task.source_row_mask
        source_positions[index, :rows, :paths, :samples] = task.source_positions
        source_type_ids[index, :paths, :columns] = task.source_type_ids
        source_column_features[index, :paths, :columns] = task.source_column_features
        source_column_mask[index, :paths, :columns] = task.source_column_mask

    return RoutedTaskBatch(
        task_ids=tuple(task.task_id for task in tasks),
        prediction_types=tuple(task.prediction_type for task in tasks),
        num_classes=tuple(task.num_classes for task in tasks),
        row_ids=row_ids,
        target_values=target_values,
        target_missing=target_missing,
        target_type_ids=target_type_ids,
        target_column_features=target_column_features,
        target_column_mask=target_column_mask,
        row_mask=row_mask,
        support_mask=support_mask,
        labels=labels,
        path_features=path_features,
        path_mask=path_mask,
        path_costs=path_costs,
        path_similarity=path_similarity,
        route_targets=route_targets,
        route_weights=route_weights,
        source_values=source_values,
        source_missing=source_missing,
        source_row_mask=source_row_mask,
        source_positions=source_positions,
        source_type_ids=source_type_ids,
        source_column_features=source_column_features,
        source_column_mask=source_column_mask,
    )


class RoutingTaskTensorizer:
    """Build bounded dense tensors; aggregation remains sparse in the model."""

    def __init__(self, config: RouterModelConfig) -> None:
        self.config = config

    def tensorize(
        self,
        raw: RawRoutingTask,
        *,
        selected_path_mask: torch.Tensor | np.ndarray | None = None,
        selected_column_mask: torch.Tensor | np.ndarray | None = None,
        materialize_relations: bool = True,
    ) -> RoutedTaskTensors:
        plan = raw.task_artifact.task.plan
        data = raw.task_artifact.task.data
        preferred = tuple(
            label.foreign_key_ids
            for label in plan.route_supervision
            if label.role is not RouteRole.DISTRACTOR
        )
        if not preferred and plan.foreign_key_id is not None:
            preferred = ((plan.foreign_key_id,),)
        paths = enumerate_schema_paths(
            raw.schema,
            target_table_id=plan.target_table_id,
            max_depth=self.config.max_path_depth,
            max_candidates=self.config.max_candidates,
            preferred_paths=preferred,
        )
        if not paths:
            raise ValueError(f"task {plan.task_id} has no legal schema paths")

        (
            support_row_ids,
            support_labels,
            query_row_ids,
            query_labels,
        ) = _bounded_task_rows(
            data.support_row_ids,
            data.support_labels,
            data.query_row_ids,
            data.query_labels,
            max_rows=self.config.max_rows_per_task,
            seed_key=plan.task_id,
        )
        row_ids = np.concatenate(
            (support_row_ids, query_row_ids)
        ).astype(np.int64, copy=False)
        cost_cutoff = _minimum_row_cutoff(raw, row_ids)
        support_mask = np.zeros(len(row_ids), dtype=bool)
        support_mask[: len(support_row_ids)] = True
        target_columns = _visible_columns(
            raw.schema,
            raw.view,
            plan.target_table_id,
            max_columns=self.config.max_target_columns,
        )
        if not target_columns:
            raise ValueError(f"task {plan.task_id} has no visible target columns")
        target_table = raw.database.table(plan.target_table_id)
        target_values = np.zeros((len(row_ids), len(target_columns)), dtype=np.float32)
        target_missing = np.zeros_like(target_values, dtype=bool)
        for column_index, column in enumerate(target_columns):
            encoded, missing = _encode_column(
                target_table.column(column.column_id),
                reference_rows=support_row_ids,
            )
            target_values[:, column_index] = encoded[row_ids]
            target_missing[:, column_index] = missing[row_ids]

        path_columns = [
            _visible_columns(raw.schema, raw.view, path.source_table_id)
            for path in paths
        ]
        max_source_columns = max(1, max(map(len, path_columns)))
        rows = len(row_ids)
        path_count = len(paths)
        samples = self.config.rows_per_hop
        stored_samples = samples if materialize_relations else 0
        source_values = np.zeros(
            (rows, path_count, max_source_columns, stored_samples), dtype=np.float32
        )
        source_missing = np.ones_like(source_values, dtype=bool)
        source_row_mask = np.zeros_like(source_values, dtype=bool)
        source_positions = np.zeros(
            (rows, path_count, stored_samples), dtype=np.float32
        )
        source_type_ids = np.zeros(
            (path_count, max_source_columns), dtype=np.int64
        )
        source_column_features = np.zeros(
            (path_count, max_source_columns, 8), dtype=np.float32
        )
        source_column_mask = np.zeros(
            (path_count, max_source_columns), dtype=bool
        )
        path_costs = np.zeros((path_count, 3), dtype=np.float32)
        column_ids: list[tuple[str, ...]] = []
        active_paths = _selection_mask(
            selected_path_mask,
            shape=(path_count,),
            default=materialize_relations,
        )
        active_columns = _selection_mask(
            selected_column_mask,
            shape=(path_count, max_source_columns),
            default=materialize_relations,
        )

        for path_index, (path, columns) in enumerate(zip(paths, path_columns)):
            endpoint = raw.database.table(path.source_table_id)
            endpoint_time_column = next(
                (
                    column
                    for column in raw.schema.table(path.source_table_id).columns
                    if column.kind is ColumnKind.TIME
                ),
                None,
            )
            visible_endpoint_rows = raw.view.visible_rows(path.source_table_id)
            for column_index, column in enumerate(columns):
                source_type_ids[path_index, column_index] = _TYPE_INDEX[
                    column.data_type
                ]
                source_column_features[path_index, column_index] = (
                    column_feature_vector(column)
                )
                source_column_mask[path_index, column_index] = True
            column_ids.append(tuple(column.column_id for column in columns))
            path_costs[path_index] = _estimated_path_cost(
                raw,
                path,
                rows_per_hop=samples,
                max_depth=self.config.max_path_depth,
                max_timestamp=cost_cutoff,
            )
            active_column_indices = [
                column_index
                for column_index in range(len(columns))
                if active_columns[path_index, column_index]
            ]
            if not active_paths[path_index] or not active_column_indices:
                continue
            encoded_columns = {
                column_index: _encode_column(
                    endpoint.column(columns[column_index].column_id),
                    reference_rows=visible_endpoint_rows,
                )
                for column_index in active_column_indices
            }
            for row_index, target_row in enumerate(row_ids):
                related, _reads, _expanded = _traverse_path(
                    raw,
                    path,
                    int(target_row),
                    min_rows=self.config.min_rows_per_hop,
                    max_rows=samples,
                    seed_key=f"{plan.task_id}:{path.signature}:{target_row}",
                )
                count = min(samples, len(related))
                if not count:
                    continue
                related = related[:count]
                if (
                    self.config.aggregation == "sequence"
                    and endpoint_time_column is not None
                ):
                    times = endpoint.column(endpoint_time_column.column_id)
                    related = related[
                        np.argsort(times[related], kind="stable")
                    ]
                    observed_times = times[related].astype(np.float64)
                    span = float(observed_times[-1] - observed_times[0])
                    if span > 0:
                        positions = (
                            (observed_times - observed_times[0]) / span
                        ).astype(np.float32)
                    else:
                        positions = np.zeros(count, dtype=np.float32)
                else:
                    positions = np.linspace(
                        0.0, 1.0, count, dtype=np.float32
                    )
                source_positions[row_index, path_index, :count] = positions
                for column_index, (encoded, missing) in encoded_columns.items():
                    source_values[
                        row_index, path_index, column_index, :count
                    ] = encoded[related]
                    source_missing[
                        row_index, path_index, column_index, :count
                    ] = missing[related]
                    source_row_mask[
                        row_index, path_index, column_index, :count
                    ] = True

        route_targets, route_weights = _route_supervision(raw, paths)
        labels, num_classes, center, scale, class_values = _encode_labels(
            plan.prediction_type,
            support_labels,
            query_labels,
        )
        return RoutedTaskTensors(
            task_id=plan.task_id,
            prediction_type=plan.prediction_type,
            row_ids=torch.from_numpy(row_ids),
            target_values=torch.from_numpy(target_values),
            target_missing=torch.from_numpy(target_missing),
            target_type_ids=torch.tensor(
                [_TYPE_INDEX[column.data_type] for column in target_columns],
                dtype=torch.long,
            ),
            target_column_features=torch.from_numpy(
                np.stack([column_feature_vector(column) for column in target_columns])
            ),
            support_mask=torch.from_numpy(support_mask),
            labels=torch.from_numpy(labels),
            num_classes=num_classes,
            class_values=class_values,
            path_features=torch.from_numpy(
                np.stack(
                    [
                        path_feature_vector(
                            raw.schema,
                            path,
                            max_depth=self.config.max_path_depth,
                        )
                        for path in paths
                    ]
                )
            ),
            path_costs=torch.from_numpy(path_costs),
            path_similarity=torch.from_numpy(path_similarity_matrix(paths)),
            route_targets=torch.from_numpy(route_targets),
            route_weights=torch.from_numpy(route_weights),
            source_values=torch.from_numpy(source_values),
            source_missing=torch.from_numpy(source_missing),
            source_row_mask=torch.from_numpy(source_row_mask),
            source_positions=torch.from_numpy(source_positions),
            source_type_ids=torch.from_numpy(source_type_ids),
            source_column_features=torch.from_numpy(source_column_features),
            source_column_mask=torch.from_numpy(source_column_mask),
            path_signatures=tuple(path.signature for path in paths),
            source_column_ids=tuple(column_ids),
            label_center=center,
            label_scale=scale,
        )

    def tensorize_descriptors(self, raw: RawRoutingTask) -> RoutedTaskTensors:
        """Read target/support and schema descriptors, but no relation cells."""
        return self.tensorize(raw, materialize_relations=False)

    def materialize_selected(
        self,
        raw: RawRoutingTask,
        *,
        selected_path_mask: torch.Tensor,
        selected_column_mask: torch.Tensor,
    ) -> RoutedTaskTensors:
        """Read and traverse only hard-selected paths and source columns."""
        return self.tensorize(
            raw,
            selected_path_mask=selected_path_mask,
            selected_column_mask=selected_column_mask,
            materialize_relations=True,
        )


def column_feature_vector(column: PhysicalColumn) -> np.ndarray:
    vector = np.zeros(8, dtype=np.float32)
    vector[_TYPE_INDEX[column.data_type]] = 1.0
    vector[6] = float(column.nullable)
    vector[7] = float(column.unique)
    return vector


def _visible_columns(
    schema: PhysicalSchema,
    view: TaskView,
    table_id: str,
    *,
    max_columns: int | None = None,
) -> tuple[PhysicalColumn, ...]:
    columns = tuple(
        column
        for column in schema.table(table_id).columns
        if column.kind not in {ColumnKind.PRIMARY_KEY, ColumnKind.FOREIGN_KEY}
        and not view.is_column_masked(table_id, column.column_id)
    )
    return columns if max_columns is None else columns[:max_columns]


def _encode_column(
    values: np.ndarray,
    *,
    reference_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    missing = _missing_mask(values)
    reference = values[reference_rows]
    reference_missing = missing[reference_rows]
    if values.dtype.kind in {"U", "S", "b"}:
        observed = np.unique(reference[~reference_missing])
        mapping = {value: index + 1 for index, value in enumerate(observed.tolist())}
        encoded = np.asarray([mapping.get(value, 0) for value in values], dtype=np.float32)
        scale = max(1, len(mapping))
        encoded /= scale
    else:
        encoded = values.astype(np.float64, copy=True)
        observed = reference[~reference_missing].astype(np.float64, copy=False)
        center = float(np.median(observed)) if len(observed) else 0.0
        scale = float(np.std(observed)) if len(observed) else 1.0
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        encoded = ((encoded - center) / scale).astype(np.float32)
        encoded[~np.isfinite(encoded)] = 0.0
    encoded[missing] = 0.0
    return encoded, missing


def _missing_mask(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "f":
        return ~np.isfinite(values)
    if values.dtype.kind in {"U", "S"}:
        return values == ""
    return np.zeros(len(values), dtype=bool)


def _traverse_path(
    raw: RawRoutingTask,
    path: SchemaPath,
    start_row: int,
    *,
    min_rows: int,
    max_rows: int,
    seed_key: str,
) -> tuple[np.ndarray, int, int]:
    current = np.asarray([start_row], dtype=np.int64)
    row_cutoff = _row_cutoff(raw, start_row)
    total_reads = 0
    total_expanded = 0
    fks = {fk.foreign_key_id: fk for fk in raw.schema.foreign_keys}
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(digest[:8], "big")))
    for hop_index, hop in enumerate(path.hops):
        foreign_key = fks[hop.foreign_key_id]
        if hop.parent_to_child:
            assignments = raw.database.table(
                foreign_key.child_table_id
            ).column(foreign_key.child_column_id)
            visible = _visible_row_mask(
                raw,
                foreign_key.child_table_id,
                max_timestamp=row_cutoff,
            )
            following = np.flatnonzero(
                visible & np.isin(assignments, current)
            ).astype(np.int64)
        else:
            assignments = raw.database.table(
                foreign_key.child_table_id
            ).column(foreign_key.child_column_id)
            parents = assignments[current]
            following = np.unique(parents[parents >= 0]).astype(np.int64)
            visible = _visible_row_mask(
                raw,
                foreign_key.parent_table_id,
                max_timestamp=row_cutoff,
            )
            following = following[visible[following]]
        total_expanded += len(following)
        cap = int(rng.integers(min_rows, max_rows + 1))
        if len(following) > cap:
            following = np.sort(rng.choice(following, cap, replace=False))
        total_reads += len(following)
        current = following
        if not len(current):
            break
        # Mix the hop into the deterministic stream without global state.
        if hop_index + 1 < len(path.hops):
            rng.random()
    return current, total_reads, total_expanded


def _estimated_path_cost(
    raw: RawRoutingTask,
    path: SchemaPath,
    *,
    rows_per_hop: int,
    max_depth: int,
    max_timestamp: int | None = None,
) -> np.ndarray:
    """Estimate fanout/read cost without traversing candidate feature data."""
    fks = {fk.foreign_key_id: fk for fk in raw.schema.foreign_keys}
    expansions: list[float] = []
    expected_rows = 1.0
    for hop in path.hops:
        foreign_key = fks[hop.foreign_key_id]
        child_rows = np.flatnonzero(
            _visible_row_mask(
                raw,
                foreign_key.child_table_id,
                max_timestamp=max_timestamp,
            )
        ).astype(np.int64)
        assignments = raw.database.table(foreign_key.child_table_id).column(
            foreign_key.child_column_id
        )[child_rows]
        valid_mask = assignments >= 0
        if np.any(valid_mask):
            parent_visible = _visible_row_mask(
                raw,
                foreign_key.parent_table_id,
                max_timestamp=max_timestamp,
            )
            valid_indices = np.flatnonzero(valid_mask)
            valid_mask[valid_indices] &= parent_visible[
                assignments[valid_indices]
            ]
        valid = assignments[valid_mask]
        if hop.parent_to_child:
            parent_count = max(
                1,
                int(
                    np.count_nonzero(
                        _visible_row_mask(
                            raw,
                            foreign_key.parent_table_id,
                            max_timestamp=max_timestamp,
                        )
                    )
                ),
            )
            expansion = len(valid) / parent_count
        else:
            expansion = float(len(valid) / max(1, len(assignments)))
        expansions.append(expansion)
        expected_rows = min(
            float(rows_per_hop), expected_rows * max(expansion, 1e-3)
        )
    average_fanout = float(np.mean(expansions)) if expansions else 0.0
    return np.asarray(
        [
            len(path.hops) / max_depth,
            min(
                1.0,
                np.log1p(average_fanout) / np.log1p(rows_per_hop),
            ),
            min(1.0, expected_rows / rows_per_hop),
        ],
        dtype=np.float32,
    )


def _minimum_row_cutoff(
    raw: RawRoutingTask,
    row_ids: np.ndarray,
) -> int | None:
    column_id = raw.task_artifact.task.plan.row_cutoff_time_column_id
    if column_id is None:
        return None
    values = raw.database.table(
        raw.task_artifact.task.plan.target_table_id
    ).column(column_id)[row_ids]
    valid = values[values >= 0]
    return None if not len(valid) else int(np.min(valid))


def _row_cutoff(raw: RawRoutingTask, start_row: int) -> int | None:
    column_id = raw.task_artifact.task.plan.row_cutoff_time_column_id
    if column_id is None:
        return None
    value = raw.database.table(
        raw.task_artifact.task.plan.target_table_id
    ).column(column_id)[start_row]
    return None if int(value) < 0 else int(value)


def _visible_row_mask(
    raw: RawRoutingTask,
    table_id: str,
    *,
    max_timestamp: int | None,
) -> np.ndarray:
    visible = raw.view.row_masks[table_id]
    if max_timestamp is None:
        return visible
    rules = {
        rule.table_id: rule
        for rule in raw.task_artifact.task.plan.observation_rules
    }
    rule = rules.get(table_id)
    if rule is None:
        return visible
    values = raw.database.table(table_id).column(rule.time_column_id)
    return visible & (values >= 0) & (values <= max_timestamp)


def _selection_mask(
    value: torch.Tensor | np.ndarray | None,
    *,
    shape: tuple[int, ...],
    default: bool,
) -> np.ndarray:
    if value is None:
        return np.full(shape, default, dtype=bool)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=bool)
    if result.shape != shape:
        raise ValueError(
            f"selection mask shape {result.shape} does not match {shape}"
        )
    return result


def _route_supervision(
    raw: RawRoutingTask,
    paths: tuple[SchemaPath, ...],
) -> tuple[np.ndarray, np.ndarray]:
    plan = raw.task_artifact.task.plan
    roles = {label.foreign_key_ids: label.role for label in plan.route_supervision}
    if not roles and plan.mechanism in {
        TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE,
        TaskMechanism.HISTORY_GATED_FUTURE_ACTIVITY,
        TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE,
        TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE,
    }:
        if plan.foreign_key_id is not None:
            roles[(plan.foreign_key_id,)] = RouteRole.REQUIRED
    target_map = {
        RouteRole.REQUIRED: 1.0,
        RouteRole.OPTIONAL: 0.5,
        RouteRole.DISTRACTOR: 0.0,
    }
    weight_map = {
        RouteRole.REQUIRED: 1.0,
        RouteRole.OPTIONAL: 0.6,
        RouteRole.DISTRACTOR: 0.35,
    }
    targets = np.zeros(len(paths), dtype=np.float32)
    weights = np.full(len(paths), weight_map[RouteRole.DISTRACTOR], dtype=np.float32)
    for index, path in enumerate(paths):
        role = roles.get(path.foreign_key_ids, RouteRole.DISTRACTOR)
        targets[index] = target_map[role]
        weights[index] = weight_map[role]
    return targets, weights


def _encode_labels(
    prediction_type: PredictionType,
    support_labels: np.ndarray,
    query_labels: np.ndarray,
) -> tuple[np.ndarray, int, float, float, tuple[Any, ...]]:
    values = np.concatenate((support_labels, query_labels))
    if prediction_type is PredictionType.CLASSIFICATION:
        classes = np.unique(support_labels)
        mapping = {value: index for index, value in enumerate(classes.tolist())}
        try:
            encoded = np.asarray([mapping[value] for value in values], dtype=np.int64)
        except KeyError as error:
            raise ValueError("query label is absent from support classes") from error
        return encoded, len(classes), 0.0, 1.0, tuple(classes.tolist())
    support = support_labels.astype(np.float64, copy=False)
    center = float(np.mean(support))
    scale = float(np.std(support))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    encoded = ((values.astype(np.float64) - center) / scale).astype(np.float32)
    return encoded, 1, center, scale, ()


def _bounded_task_rows(
    support_row_ids: np.ndarray,
    support_labels: np.ndarray,
    query_row_ids: np.ndarray,
    query_labels: np.ndarray,
    *,
    max_rows: int,
    seed_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = len(support_row_ids) + len(query_row_ids)
    if total <= max_rows:
        return support_row_ids, support_labels, query_row_ids, query_labels
    support_count = round(max_rows * len(support_row_ids) / total)
    support_count = min(
        len(support_row_ids),
        max(1, min(max_rows - 1, support_count)),
    )
    query_count = min(len(query_row_ids), max_rows - support_count)
    if query_count < 1:
        query_count = 1
        support_count = max_rows - 1
    support_indices = _sample_row_indices(
        len(support_row_ids),
        support_count,
        seed_key=f"{seed_key}:support",
        labels=support_labels,
    )
    # Query selection is deliberately label-blind.
    query_indices = _sample_row_indices(
        len(query_row_ids),
        query_count,
        seed_key=f"{seed_key}:query",
        labels=None,
    )
    return (
        support_row_ids[support_indices],
        support_labels[support_indices],
        query_row_ids[query_indices],
        query_labels[query_indices],
    )


def _sample_row_indices(
    length: int,
    count: int,
    *,
    seed_key: str,
    labels: np.ndarray | None,
) -> np.ndarray:
    if count >= length:
        return np.arange(length, dtype=np.int64)
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    rng = np.random.Generator(
        np.random.PCG64DXSM(int.from_bytes(digest[:8], "big"))
    )
    mandatory: list[int] = []
    if labels is not None:
        for value in np.unique(labels):
            candidates = np.flatnonzero(labels == value)
            mandatory.append(int(candidates[int(rng.integers(len(candidates)))]))
    mandatory = mandatory[:count]
    remaining = np.setdiff1d(
        np.arange(length, dtype=np.int64),
        np.asarray(mandatory, dtype=np.int64),
        assume_unique=False,
    )
    needed = count - len(mandatory)
    sampled = (
        rng.choice(remaining, needed, replace=False).astype(np.int64)
        if needed
        else np.empty(0, dtype=np.int64)
    )
    return np.sort(
        np.concatenate((np.asarray(mandatory, dtype=np.int64), sampled))
    )


def _resolve_reference(artifact_path: Path, reference: str) -> Path:
    path = Path(reference)
    if not path.is_absolute():
        path = artifact_path.parent / path
    return path.resolve()


__all__ = [
    "RawRoutingTask",
    "RoutedTaskBatch",
    "RoutedTaskTensors",
    "RoutingTaskReference",
    "RoutingTaskStore",
    "RoutingTaskTensorizer",
    "collate_routed_tasks",
    "column_feature_vector",
    "load_routing_task_references",
    "load_routing_tasks",
]
