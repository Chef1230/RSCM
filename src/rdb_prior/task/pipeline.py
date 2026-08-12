"""Stage-03 pipeline from instance manifest to task artifacts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Callable

from rdb_prior.artifacts import load_instance_artifact, load_schema_artifact
from rdb_prior.runtime import digest_config
from rdb_prior.task.artifacts import TaskArtifactWriter
from rdb_prior.task.planner import TaskPlanner, TaskPlannerConfig
from rdb_prior.task.validation import validate_task
from rdb_prior.validation.checks import validate_database_instance


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskPipelineConfig:
    instance_manifest: Path
    output_root: Path
    database_count: int | None = None
    start_index: int = 0
    shard_id: int = 0
    num_shards: int = 1
    num_workers: int = 1
    overwrite: bool = False
    progress_every: int = 100
    project_version: str = "task-pipeline-v1"
    planner: TaskPlannerConfig = TaskPlannerConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.instance_manifest, Path):
            raise TypeError("instance_manifest must be pathlib.Path")
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be pathlib.Path")
        if self.database_count is not None and (
            isinstance(self.database_count, bool)
            or not isinstance(self.database_count, int)
        ):
            raise TypeError("database_count must be an integer or None")
        if self.database_count is not None and self.database_count < 1:
            raise ValueError("database_count must be positive")
        for name in (
            "start_index",
            "shard_id",
            "num_shards",
            "num_workers",
            "progress_every",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.start_index < 0 or self.progress_every < 0:
            raise ValueError("start_index and progress_every must be non-negative")
        if self.num_shards < 1 or not 0 <= self.shard_id < self.num_shards:
            raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
        if self.num_workers < 1:
            raise ValueError("num_workers must be positive")
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean")
        if not isinstance(self.planner, TaskPlannerConfig):
            raise TypeError("planner must be TaskPlannerConfig")

    def to_dict(self) -> dict[str, object]:
        planner = self.planner
        return {
            "instance_manifest": str(self.instance_manifest),
            "database_count": self.database_count,
            "start_index": self.start_index,
            "shard_id": self.shard_id,
            "num_shards": self.num_shards,
            "num_workers": self.num_workers,
            "overwrite": self.overwrite,
            "progress_every": self.progress_every,
            "project_version": self.project_version,
            "planner": {
                name: getattr(planner, name)
                for name in planner.__dataclass_fields__
                if name != "mechanism_weights"
            }
            | {
                "mechanism_weights": [
                    [mechanism.value, weight]
                    for mechanism, weight in planner.mechanism_weights
                ]
            },
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskPipelineResult:
    output_root: Path
    manifest_path: Path
    artifact_paths: tuple[Path, ...]
    database_count: int

    @property
    def task_count(self) -> int:
        return len(self.artifact_paths)


@dataclass(frozen=True, slots=True, kw_only=True)
class _TaskWorkItem:
    index: int
    entry: dict[str, Any]
    instance_manifest: Path
    output_root: Path
    overwrite: bool
    planner_config: TaskPlannerConfig
    config_digest: str
    project_version: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _TaskWorkResult:
    index: int
    sample_id: str
    artifact_paths: tuple[Path, ...]
    output_entries: tuple[dict[str, object], ...]
    task_count: int


def generate_tasks(
    config: TaskPipelineConfig,
    *,
    progress: Callable[[int, int, str, int], None] | None = None,
) -> TaskPipelineResult:
    if not isinstance(config, TaskPipelineConfig):
        raise TypeError("config must be TaskPipelineConfig")
    manifest = json.loads(config.instance_manifest.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "database_instance_manifest":
        raise ValueError("input is not a database instance manifest")
    if manifest.get("artifact_version") != 1:
        raise ValueError("unsupported database instance manifest version")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("instance manifest entries must be a list")

    indexed = list(enumerate(entries))[config.start_index :]
    if config.database_count is not None:
        indexed = indexed[: config.database_count]
    selected = [
        (index, entry)
        for index, entry in indexed
        if index % config.num_shards == config.shard_id
    ]
    configuration = config.to_dict()
    config_digest = digest_config(configuration)
    work_items = [
        _TaskWorkItem(
            index=index,
            entry=entry,
            instance_manifest=config.instance_manifest,
            output_root=config.output_root,
            overwrite=config.overwrite,
            planner_config=config.planner,
            config_digest=config_digest,
            project_version=config.project_version,
        )
        for index, entry in selected
    ]
    _LOGGER.info(
        "starting task pipeline: databases=%d tasks_per_database=%d "
        "workers=%d shard=%d/%d output=%s",
        len(selected),
        config.planner.tasks_per_database,
        config.num_workers,
        config.shard_id,
        config.num_shards,
        config.output_root,
    )

    results = _run_task_work_items(
        work_items,
        num_workers=config.num_workers,
        progress=progress,
    )
    artifact_paths = [
        artifact_path
        for result in results
        for artifact_path in result.artifact_paths
    ]
    output_entries = [
        output_entry
        for result in results
        for output_entry in result.output_entries
    ]
    writer = TaskArtifactWriter(
        output_root=config.output_root,
        overwrite=config.overwrite,
    )

    manifest_path = writer.write_manifest(
        configuration=configuration,
        database_count=len(selected),
        entries=output_entries,
        filename=(
            "manifest.json"
            if config.num_shards == 1
            else (
                f"manifest.shard_{config.shard_id:05d}"
                f"_of_{config.num_shards:05d}.json"
            )
        ),
    )
    _LOGGER.info(
        "task pipeline complete: databases=%d tasks=%d manifest=%s",
        len(selected),
        len(artifact_paths),
        manifest_path,
    )
    return TaskPipelineResult(
        output_root=config.output_root,
        manifest_path=manifest_path,
        artifact_paths=tuple(artifact_paths),
        database_count=len(selected),
    )


def _run_task_work_items(
    work_items: list[_TaskWorkItem],
    *,
    num_workers: int,
    progress: Callable[[int, int, str, int], None] | None,
) -> list[_TaskWorkResult]:
    if not work_items:
        return []
    completed = 0
    results: list[_TaskWorkResult] = []
    if num_workers == 1:
        for item in work_items:
            result = _generate_one_database_tasks(item)
            results.append(result)
            completed += 1
            if progress is not None:
                progress(
                    completed,
                    len(work_items),
                    result.sample_id,
                    result.task_count,
                )
    else:
        max_workers = min(num_workers, len(work_items))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_generate_one_database_tasks, item): item
                for item in work_items
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                if progress is not None:
                    progress(
                        completed,
                        len(work_items),
                        result.sample_id,
                        result.task_count,
                    )
    return sorted(results, key=lambda result: result.index)


def _generate_one_database_tasks(item: _TaskWorkItem) -> _TaskWorkResult:
    if not isinstance(item.entry, dict):
        raise ValueError("instance manifest entry must be an object")
    instance_path = (item.instance_manifest.parent / item.entry["artifact"]).resolve()
    instance_artifact = load_instance_artifact(instance_path)
    schema_path = Path(instance_artifact.schema_artifact)
    if not schema_path.is_absolute():
        schema_path = (instance_path.parent / schema_path).resolve()
    schema_artifact = load_schema_artifact(schema_path)
    schema = schema_artifact.compilation.schema
    database = instance_artifact.database
    instance_report = validate_database_instance(
        schema,
        instance_artifact.plan,
        database,
    )
    if not instance_report.is_valid:
        issues = [issue.to_dict() for issue in instance_report.issues]
        raise ValueError(
            f"invalid database instance {instance_artifact.instance_id}: "
            f"{issues}"
        )
    runtime = instance_artifact.runtime.restore_context().child("task")
    tasks = TaskPlanner(item.planner_config).generate(
        sample_id=instance_artifact.sample_id,
        schema=schema,
        database=database,
        runtime=runtime,
        instance_plan=instance_artifact.plan,
    )
    writer = TaskArtifactWriter(
        output_root=item.output_root,
        overwrite=item.overwrite,
    )
    artifact_paths: list[Path] = []
    output_entries: list[dict[str, object]] = []
    for task in tasks:
        report = validate_task(schema, database, task)
        if not report.is_valid:
            raise ValueError(
                f"invalid task {task.plan.task_id}: "
                f"{[issue.code for issue in report.issues]}"
            )
        task_runtime = runtime.child(task.plan.task_id)
        runtime_record = task_runtime.record(
            project_version=item.project_version,
            config_digest=item.config_digest,
            metadata={
                "stage": "relational_task",
                "sample_id": instance_artifact.sample_id,
                "task_id": task.plan.task_id,
                "mechanism": task.plan.mechanism.value,
            },
        )
        artifact_path = writer.commit(
            sample_id=instance_artifact.sample_id,
            instance_artifact=str(instance_path),
            schema_artifact=str(schema_path),
            runtime=runtime_record,
            task=task,
            report=report,
        )
        artifact_paths.append(artifact_path)
        output_entries.append(
            {
                "sample_id": instance_artifact.sample_id,
                "task_id": task.plan.task_id,
                "artifact": artifact_path.relative_to(
                    item.output_root
                ).as_posix(),
                "instance_artifact": str(instance_path),
                "schema_id": schema.schema_id,
                "mechanism": task.plan.mechanism.value,
                "prediction_type": task.plan.prediction_type.value,
                "support_count": len(task.data.support_row_ids),
                "query_count": len(task.data.query_row_ids),
            }
        )
    return _TaskWorkResult(
        index=item.index,
        sample_id=instance_artifact.sample_id,
        artifact_paths=tuple(artifact_paths),
        output_entries=tuple(output_entries),
        task_count=len(tasks),
    )


__all__ = [
    "TaskPipelineConfig",
    "TaskPipelineResult",
    "generate_tasks",
]
