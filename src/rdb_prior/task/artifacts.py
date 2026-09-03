"""Atomic artifact storage for generated task plans and labels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping

import numpy as np

from rdb_prior.runtime import RuntimeRecord
from rdb_prior.task.model import PlannedTask, TaskData, TaskPlan
from rdb_prior.task.program import TaskProgramPlan
from rdb_prior.task.validation import TaskValidationReport


_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskArtifactWriter:
    output_root: Path
    overwrite: bool = False

    @property
    def task_directory(self) -> Path:
        return self.output_root / "tasks"

    def commit(
        self,
        *,
        sample_id: str,
        instance_artifact: str,
        schema_artifact: str,
        runtime: RuntimeRecord,
        task: PlannedTask,
        report: TaskValidationReport,
        task_program: TaskProgramPlan | None = None,
    ) -> Path:
        for name, value in (
            ("sample_id", sample_id),
            ("task_id", task.plan.task_id),
        ):
            if not isinstance(value, str) or not _ARTIFACT_ID.fullmatch(value):
                raise ValueError(f"{name} is not artifact-safe")
        if report.task_id != task.plan.task_id or not report.is_valid:
            raise ValueError("Cannot commit an invalid task")

        target = self.task_directory / sample_id / task.plan.task_id
        temporary = target.parent / f".{task.plan.task_id}.tmp"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not self.overwrite:
            raise FileExistsError(
                f"Task artifact already exists: {target}; use overwrite=True"
            )
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        try:
            _write_json(temporary / "task_plan.json", task.plan.to_dict())
            if task_program is not None:
                _write_json(temporary / "task_program.json", task_program.to_dict())
            _write_json(temporary / "runtime.json", runtime.to_dict())
            _write_json(temporary / "validation.json", report.to_dict())
            with (temporary / "task_data.npz").open("wb") as handle:
                np.savez_compressed(
                    handle,
                    support_row_ids=task.data.support_row_ids,
                    support_labels=task.data.support_labels,
                    query_row_ids=task.data.query_row_ids,
                    query_labels=task.data.query_labels,
                )
            _write_json(
                temporary / "artifact.json",
                {
                    "artifact_type": "relational_task",
                    "artifact_version": 2,
                    "sample_id": sample_id,
                    "task_id": task.plan.task_id,
                    "instance_artifact": instance_artifact,
                    "schema_artifact": schema_artifact,
                    "plan": "task_plan.json",
                    "data": "task_data.npz",
                    "runtime": "runtime.json",
                    "validation": "validation.json",
                    "task_program": (
                        None if task_program is None else "task_program.json"
                    ),
                },
            )
            if target.exists():
                shutil.rmtree(target)
            temporary.replace(target)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return target / "artifact.json"

    def write_manifest(
        self,
        *,
        configuration: Mapping[str, Any],
        database_count: int,
        entries: Iterable[Mapping[str, Any]],
        filename: str = "manifest.json",
    ) -> Path:
        if not _ARTIFACT_ID.fullmatch(filename) or not filename.endswith(".json"):
            raise ValueError("manifest filename is not artifact-safe JSON")
        path = self.output_root / filename
        if path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Task manifest already exists: {path}; use overwrite=True"
            )
        encoded_entries = [dict(entry) for entry in entries]
        _write_json(
            path,
            {
                "artifact_type": "relational_task_manifest",
                "artifact_version": 1,
                "configuration": dict(configuration),
                "database_count": database_count,
                "task_count": len(encoded_entries),
                "entries": encoded_entries,
            },
        )
        return path


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskArtifact:
    sample_id: str
    instance_artifact: str
    schema_artifact: str
    runtime: RuntimeRecord
    task: PlannedTask
    validation: TaskValidationReport
    task_program: TaskProgramPlan | None = None


def load_task_artifact(path: str | Path) -> TaskArtifact:
    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "relational_task":
        raise ValueError("unsupported task artifact type")
    if payload.get("artifact_version") not in {1, 2}:
        raise ValueError("unsupported task artifact version")
    root = artifact_path.parent
    plan = TaskPlan.from_dict(
        json.loads((root / payload["plan"]).read_text(encoding="utf-8"))
    )
    with np.load(root / payload["data"], allow_pickle=False) as archive:
        data = TaskData(
            support_row_ids=archive["support_row_ids"],
            support_labels=archive["support_labels"],
            query_row_ids=archive["query_row_ids"],
            query_labels=archive["query_labels"],
        )
    runtime = RuntimeRecord.from_dict(
        json.loads((root / payload["runtime"]).read_text(encoding="utf-8"))
    )
    validation = TaskValidationReport.from_dict(
        json.loads((root / payload["validation"]).read_text(encoding="utf-8"))
    )
    task_program = (
        None
        if payload.get("task_program") is None
        else TaskProgramPlan.from_dict(
            json.loads((root / payload["task_program"]).read_text(encoding="utf-8"))
        )
    )
    if plan.task_id != payload["task_id"]:
        raise ValueError("task artifact plan identity mismatch")
    return TaskArtifact(
        sample_id=payload["sample_id"],
        instance_artifact=payload["instance_artifact"],
        schema_artifact=payload["schema_artifact"],
        runtime=runtime,
        task=PlannedTask(plan=plan, data=data),
        validation=validation,
        task_program=task_program,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "TaskArtifactWriter",
    "TaskArtifact",
    "load_task_artifact",
]
