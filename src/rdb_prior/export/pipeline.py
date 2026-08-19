"""Stage-04 export pipeline from task artifacts to RDBPFN datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Callable

from rdb_prior.artifacts import load_instance_artifact, load_schema_artifact
from rdb_prior.runtime import digest_config
from rdb_prior.task.artifacts import load_task_artifact

from .artifacts import RDBPFNArtifactWriter
from .converter import RDBPFNConverter
from .h5 import (
    H5ExportConfig,
    H5ExportResult,
    export_processed_dbb_to_h5,
    run_rdbpfn_dfs,
)
from .validation import validate_rdbpfn_dataset


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class RDBPFNExportConfig:
    task_manifest: Path
    output_root: Path
    task_count: int | None = None
    start_index: int = 0
    shard_id: int = 0
    num_shards: int = 1
    validation_fraction: float = 0.2
    min_validation_rows: int = 8
    compress: bool = True
    overwrite: bool = False
    progress_every: int = 100
    project_version: str = "rdbpfn-export-v1"
    h5_enabled: bool = False
    h5_output: Path | None = None
    rdbpfn_preprocessing_root: Path = Path("../RDBPFN/data_preprocessing")
    h5_run_dfs: bool = True
    dfs_depth: int = 1
    dfs_jobs: int = 4
    h5_total_rows: int = 600
    h5_max_columns: int = 60
    h5_seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.task_manifest, Path):
            raise TypeError("task_manifest must be pathlib.Path")
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be pathlib.Path")
        if self.h5_output is not None and not isinstance(self.h5_output, Path):
            raise TypeError("h5_output must be pathlib.Path or None")
        if not isinstance(self.rdbpfn_preprocessing_root, Path):
            raise TypeError("rdbpfn_preprocessing_root must be pathlib.Path")
        if self.task_count is not None and (
            isinstance(self.task_count, bool) or not isinstance(self.task_count, int)
        ):
            raise TypeError("task_count must be an integer or None")
        if self.task_count is not None and self.task_count < 1:
            raise ValueError("task_count must be positive")
        for name in (
            "start_index",
            "shard_id",
            "num_shards",
            "min_validation_rows",
            "progress_every",
            "dfs_depth",
            "dfs_jobs",
            "h5_total_rows",
            "h5_max_columns",
            "h5_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.start_index < 0 or self.progress_every < 0:
            raise ValueError("start_index and progress_every must be non-negative")
        if self.min_validation_rows < 1:
            raise ValueError("min_validation_rows must be positive")
        if self.dfs_depth not in (1, 2):
            raise ValueError("dfs_depth must be 1 or 2")
        if self.dfs_jobs < 1:
            raise ValueError("dfs_jobs must be positive")
        if self.h5_total_rows < 2:
            raise ValueError("h5_total_rows must be at least two")
        if self.h5_max_columns < 1:
            raise ValueError("h5_max_columns must be positive")
        if self.num_shards < 1 or not 0 <= self.shard_id < self.num_shards:
            raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
        if isinstance(self.validation_fraction, bool) or not isinstance(
            self.validation_fraction, (int, float)
        ):
            raise TypeError("validation_fraction must be numeric")
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        for name in ("compress", "overwrite", "h5_enabled", "h5_run_dfs"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if not isinstance(self.project_version, str) or not self.project_version:
            raise ValueError("project_version must be a non-empty string")

    def to_dict(self) -> dict[str, object]:
        return {
            "task_manifest": str(self.task_manifest),
            "task_count": self.task_count,
            "start_index": self.start_index,
            "shard_id": self.shard_id,
            "num_shards": self.num_shards,
            "validation_fraction": self.validation_fraction,
            "min_validation_rows": self.min_validation_rows,
            "compress": self.compress,
            "overwrite": self.overwrite,
            "progress_every": self.progress_every,
            "project_version": self.project_version,
            "h5_enabled": self.h5_enabled,
            "h5_output": str(self.h5_output) if self.h5_output is not None else None,
            "rdbpfn_preprocessing_root": str(self.rdbpfn_preprocessing_root),
            "h5_run_dfs": self.h5_run_dfs,
            "dfs_depth": self.dfs_depth,
            "dfs_jobs": self.dfs_jobs,
            "h5_total_rows": self.h5_total_rows,
            "h5_max_columns": self.h5_max_columns,
            "h5_seed": self.h5_seed,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RDBPFNExportResult:
    output_root: Path
    manifest_path: Path
    dataset_paths: tuple[Path, ...]
    h5_result: H5ExportResult | None = None

    @property
    def dataset_count(self) -> int:
        return len(self.dataset_paths)


def export_rdbpfn_tasks(
    config: RDBPFNExportConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> RDBPFNExportResult:
    if not isinstance(config, RDBPFNExportConfig):
        raise TypeError("config must be RDBPFNExportConfig")
    manifest = json.loads(config.task_manifest.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "relational_task_manifest":
        raise ValueError("input is not a relational task manifest")
    if manifest.get("artifact_version") != 1:
        raise ValueError("unsupported relational task manifest version")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("task manifest entries must be a list")

    indexed = list(enumerate(entries))[config.start_index :]
    if config.task_count is not None:
        indexed = indexed[: config.task_count]
    selected = [
        (index, entry)
        for index, entry in indexed
        if index % config.num_shards == config.shard_id
    ]
    configuration = config.to_dict()
    config_digest = digest_config(configuration)
    converter = RDBPFNConverter(
        validation_fraction=config.validation_fraction,
        min_validation_rows=config.min_validation_rows,
    )
    writer = RDBPFNArtifactWriter(
        output_root=config.output_root,
        overwrite=config.overwrite,
        compress=config.compress,
    )
    dataset_paths: list[Path] = []
    output_entries: list[dict[str, object]] = []
    _LOGGER.info(
        "starting RDBPFN export: tasks=%d shard=%d/%d output=%s",
        len(selected),
        config.shard_id,
        config.num_shards,
        config.output_root,
    )

    for completed, (_index, entry) in enumerate(selected, start=1):
        if not isinstance(entry, dict):
            raise ValueError("task manifest entry must be an object")
        task_path = (config.task_manifest.parent / entry["artifact"]).resolve()
        task_artifact = load_task_artifact(task_path)
        instance_path = _resolve_reference(task_path, task_artifact.instance_artifact)
        schema_path = _resolve_reference(task_path, task_artifact.schema_artifact)
        instance_artifact = load_instance_artifact(instance_path)
        schema_artifact = load_schema_artifact(schema_path)
        dataset = converter.convert(
            task_artifact=task_artifact,
            schema=schema_artifact.compilation.schema,
            database=instance_artifact.database,
        )
        report = validate_rdbpfn_dataset(dataset)
        if not report.is_valid:
            raise ValueError(
                f"invalid RDBPFN export {dataset.dataset_name}: {list(report.issues)}"
            )
        dataset_path = writer.commit(dataset)
        dataset_paths.append(dataset_path)
        task_plan = task_artifact.task.plan
        output_entries.append(
            {
                "dataset_name": dataset.dataset_name,
                "dataset": dataset_path.relative_to(config.output_root).as_posix(),
                "metadata": (
                    dataset_path.relative_to(config.output_root) / "metadata.yaml"
                ).as_posix(),
                "sample_id": task_artifact.sample_id,
                "task_id": task_plan.task_id,
                "mechanism": task_plan.mechanism.value,
                "composite_family": (
                    None
                    if task_plan.composite_spec is None
                    else task_plan.composite_spec.family.value
                ),
                "prediction_type": task_plan.prediction_type.value,
                "task_artifact": str(task_path),
                "instance_artifact": str(instance_path),
                "schema_artifact": str(schema_path),
                "config_digest": config_digest,
                "project_version": config.project_version,
                "table_count": len(dataset.tables),
                "train_count": len(dataset.splits["train"][_LABEL_COLUMN]),
                "validation_count": len(dataset.splits["validation"][_LABEL_COLUMN]),
                "test_count": len(dataset.splits["test"][_LABEL_COLUMN]),
            }
        )
        _LOGGER.debug(
            "RDBPFN dataset committed: task_id=%s path=%s",
            task_plan.task_id,
            dataset_path,
        )
        if progress is not None:
            progress(completed, len(selected), dataset.dataset_name)

    manifest_path = writer.write_manifest(
        configuration=configuration,
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
        "RDBPFN export complete: datasets=%d manifest=%s",
        len(dataset_paths),
        manifest_path,
    )
    h5_result = _export_h5(
        config=config,
        dataset_paths=tuple(dataset_paths),
        progress=progress,
    )
    return RDBPFNExportResult(
        output_root=config.output_root,
        manifest_path=manifest_path,
        dataset_paths=tuple(dataset_paths),
        h5_result=h5_result,
    )


def _export_h5(
    *,
    config: RDBPFNExportConfig,
    dataset_paths: tuple[Path, ...],
    progress: Callable[[int, int, str], None] | None,
) -> H5ExportResult | None:
    if not config.h5_enabled:
        return None
    if not dataset_paths:
        raise RuntimeError("cannot export H5 because no stage-04 datasets were selected")
    processed_root = Path(f"{config.output_root.resolve()}-processed")
    if config.h5_run_dfs:
        processed_root = run_rdbpfn_dfs(
            raw_root=config.output_root,
            preprocessing_root=config.rdbpfn_preprocessing_root,
            depth=config.dfs_depth,
            jobs=config.dfs_jobs,
        )
    output_path = config.h5_output or (
        config.output_root / _default_h5_filename(config)
    )
    expected_names = tuple(
        f"{path.name}-dfs-{config.dfs_depth}" for path in dataset_paths
    )
    return export_processed_dbb_to_h5(
        H5ExportConfig(
            processed_root=processed_root,
            output_path=output_path,
            total_rows=config.h5_total_rows,
            max_columns=config.h5_max_columns,
            seed=config.h5_seed,
            overwrite=config.overwrite,
            dataset_names=expected_names,
        ),
        progress=progress,
    )


def _default_h5_filename(config: RDBPFNExportConfig) -> str:
    if config.num_shards == 1:
        return "rdbpfn_tasks.h5"
    return (
        f"rdbpfn_tasks.shard_{config.shard_id:05d}"
        f"_of_{config.num_shards:05d}.h5"
    )


def _resolve_reference(artifact_path: Path, reference: str) -> Path:
    path = Path(reference)
    if not path.is_absolute():
        path = artifact_path.parent / path
    return path.resolve()


_LABEL_COLUMN = "label"


__all__ = [
    "RDBPFNExportConfig",
    "RDBPFNExportResult",
    "export_rdbpfn_tasks",
]
