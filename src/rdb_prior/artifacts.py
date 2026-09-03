# src/rdb_prior/artifacts.py
# -*- coding: utf-8 -*-
"""Atomic JSON artifact writing for the schema-only generation stage."""

from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping

from rdb_prior.compilation.model import (
    CompilationResult,
    CompilationTrace,
    PhysicalSchema,
)
from rdb_prior.runtime import RuntimeRecord
from rdb_prior.generation.model import DatabaseInstance, TableData
from rdb_prior.instance.plan import InstancePlan
from rdb_prior.priors.model import DatabasePriorPlan
from rdb_prior.schema.blueprint import SchemaBlueprint
from rdb_prior.schema.semantics import SemanticSchemaPlan
from rdb_prior.task.program import TaskProgramPlan
from rdb_prior.schema.validation import ValidationReport
from rdb_prior.schema.validation import (
    ValidationIssue,
    ValidationLayer,
    ValidationLevel,
)
from rdb_prior.validation.checks import InstanceValidationReport


_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def blueprint_to_dict(blueprint: SchemaBlueprint) -> dict[str, Any]:
    return blueprint.to_dict()


def validation_report_to_dict(
    report: ValidationReport,
) -> dict[str, Any]:
    encoded_issues: list[dict[str, Any]] = []
    for issue in report.issues:
        encoded = {
            "layer": issue.layer.value,
            "level": issue.level.value,
            "code": issue.code,
            "message": issue.message,
            "node_ids": list(issue.node_ids),
            "edge_ids": list(issue.edge_ids),
        }
        if issue.constraint_id is not None:
            encoded["constraint_id"] = issue.constraint_id
        if issue.motif_type is not None:
            encoded["motif_type"] = issue.motif_type
        encoded_issues.append(encoded)

    return {
        "blueprint_id": report.blueprint_id,
        "is_valid": report.is_valid,
        "issues": encoded_issues,
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaArtifactWriter:
    output_root: Path
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be pathlib.Path")
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean")

    @property
    def schema_directory(self) -> Path:
        return self.output_root / "schemas"

    def commit(
        self,
        *,
        sample_id: str,
        runtime: RuntimeRecord,
        blueprint: SchemaBlueprint,
        compilation: CompilationResult,
        report: ValidationReport,
        semantic_schema: SemanticSchemaPlan | None = None,
        process_plan: tuple[Mapping[str, Any], ...] = (),
        prior_compatibility: Mapping[str, Any] | None = None,
    ) -> Path:
        if not isinstance(sample_id, str) or not _ARTIFACT_ID.fullmatch(
            sample_id
        ):
            raise ValueError("sample_id is not safe for an artifact filename")
        if not isinstance(runtime, RuntimeRecord):
            raise TypeError("runtime must be RuntimeRecord")
        if not isinstance(blueprint, SchemaBlueprint):
            raise TypeError("blueprint must be SchemaBlueprint")
        if not isinstance(compilation, CompilationResult):
            raise TypeError("compilation must be CompilationResult")
        if not isinstance(report, ValidationReport):
            raise TypeError("report must be ValidationReport")
        if not report.is_valid:
            raise ValueError("Cannot commit an invalid schema blueprint")

        output_path = self.schema_directory / f"{sample_id}.json"
        payload = {
            "artifact_type": "physical_schema",
            "artifact_version": 3,
            "sample_id": sample_id,
            "runtime": runtime.to_dict(),
            "blueprint": blueprint_to_dict(blueprint),
            "physical_schema": compilation.schema.to_dict(),
            "compilation_trace": compilation.trace.to_dict(),
            "validation": validation_report_to_dict(report),
            "semantic_schema": (
                None if semantic_schema is None else semantic_schema.to_dict()
            ),
            "process_plan": [dict(item) for item in process_plan],
            "prior_compatibility": (
                {} if prior_compatibility is None else dict(prior_compatibility)
            ),
        }
        self._write_json(output_path, payload)
        return output_path

    def write_manifest(
        self,
        *,
        configuration: Mapping[str, Any],
        entries: Iterable[Mapping[str, Any]],
    ) -> Path:
        manifest_path = self.output_root / "manifest.json"
        payload = {
            "artifact_type": "physical_schema_manifest",
            "artifact_version": 2,
            "configuration": dict(configuration),
            "entries": [dict(entry) for entry in entries],
        }
        self._write_json(manifest_path, payload)
        return manifest_path

    def _write_json(
        self,
        path: Path,
        payload: Mapping[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Artifact already exists: {path}; use overwrite=True"
            )

        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
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
        temporary_path.replace(path)


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaArtifact:
    sample_id: str
    runtime: RuntimeRecord
    blueprint: SchemaBlueprint
    compilation: CompilationResult
    validation: ValidationReport
    semantic_schema: SemanticSchemaPlan | None = None
    process_plan: tuple[Mapping[str, Any], ...] = ()
    prior_compatibility: Mapping[str, Any] | None = None


def load_schema_artifact(path: str | Path) -> SchemaArtifact:
    """Load one V2 schema artifact for a later pipeline stage."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("schema artifact root must be an object")
    if payload.get("artifact_type") != "physical_schema":
        raise ValueError("unsupported schema artifact type")
    if payload.get("artifact_version") not in {2, 3}:
        raise ValueError("unsupported schema artifact version")

    blueprint = SchemaBlueprint.from_dict(payload["blueprint"])
    schema = PhysicalSchema.from_dict(payload["physical_schema"])
    trace = CompilationTrace.from_dict(payload["compilation_trace"])
    validation = _validation_report_from_dict(payload["validation"])
    artifact = SchemaArtifact(
        sample_id=payload["sample_id"],
        runtime=RuntimeRecord.from_dict(payload["runtime"]),
        blueprint=blueprint,
        compilation=CompilationResult(schema=schema, trace=trace),
        validation=validation,
        semantic_schema=(
            None
            if payload.get("semantic_schema") is None
            else SemanticSchemaPlan.from_dict(payload["semantic_schema"])
        ),
        process_plan=tuple(payload.get("process_plan", ())),
        prior_compatibility=payload.get("prior_compatibility"),
    )
    if artifact.compilation.schema.blueprint_id != blueprint.blueprint_id:
        raise ValueError("artifact blueprint and physical schema do not match")
    if validation.blueprint_id != blueprint.blueprint_id:
        raise ValueError("artifact blueprint and validation do not match")
    if artifact.semantic_schema is not None and artifact.semantic_schema.schema_id != schema.schema_id:
        raise ValueError("artifact semantic schema and physical schema do not match")
    return artifact


@dataclass(frozen=True, slots=True, kw_only=True)
class InstanceArtifactWriter:
    output_root: Path
    overwrite: bool = False

    @property
    def instance_directory(self) -> Path:
        return self.output_root / "instances"

    def commit(
        self,
        *,
        sample_id: str,
        schema_artifact: str,
        runtime: RuntimeRecord,
        schema: PhysicalSchema,
        plan: InstancePlan,
        database: DatabaseInstance,
        report: InstanceValidationReport,
        prior_plan: DatabasePriorPlan | None = None,
        task_programs: tuple[TaskProgramPlan, ...] = (),
    ) -> Path:
        if not _ARTIFACT_ID.fullmatch(sample_id):
            raise ValueError("sample_id is not artifact-safe")
        if not report.is_valid:
            raise ValueError("Cannot commit an invalid database instance")
        if schema.schema_id != plan.schema_id or schema.schema_id != database.schema_id:
            raise ValueError("schema, plan and database identity mismatch")

        target = self.instance_directory / sample_id
        temporary = self.instance_directory / f".{sample_id}.tmp"
        self.instance_directory.mkdir(parents=True, exist_ok=True)
        if target.exists() and not self.overwrite:
            raise FileExistsError(
                f"Artifact already exists: {target}; use overwrite=True"
            )
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        try:
            _write_json_file(temporary / "instance_plan.json", plan.to_dict())
            if prior_plan is not None:
                _write_json_file(temporary / "database_prior_plan.json", prior_plan.to_dict())
            if task_programs:
                _write_json_file(
                    temporary / "task_programs.json",
                    {"programs": [item.to_dict() for item in task_programs]},
                )
            _write_json_file(temporary / "runtime.json", runtime.to_dict())
            _write_json_file(temporary / "validation.json", report.to_dict())
            table_entries: list[dict[str, Any]] = []
            tables_directory = temporary / "tables"
            tables_directory.mkdir()
            for physical_table in schema.tables:
                table = database.table(physical_table.table_id)
                filename = f"{physical_table.name}.npz"
                table_path = tables_directory / filename
                with table_path.open("wb") as handle:
                    np.savez_compressed(handle, **dict(table.columns))
                table_entries.append(
                    {
                        "table_id": table.table_id,
                        "physical_name": physical_table.name,
                        "artifact": f"tables/{filename}",
                        "row_count": table.row_count,
                        "column_ids": list(table.columns),
                    }
                )
            _write_json_file(
                temporary / "artifact.json",
                {
                    "artifact_type": "database_instance",
                    "artifact_version": 2,
                    "sample_id": sample_id,
                    "instance_id": database.instance_id,
                    "schema_id": schema.schema_id,
                    "plan_id": plan.plan_id,
                    "schema_artifact": schema_artifact,
                    "plan": "instance_plan.json",
                    "runtime": "runtime.json",
                    "validation": "validation.json",
                    "database_prior_plan": (
                        None if prior_plan is None else "database_prior_plan.json"
                    ),
                    "task_programs": (
                        None if not task_programs else "task_programs.json"
                    ),
                    "tables": table_entries,
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
        entries: Iterable[Mapping[str, Any]],
        filename: str = "manifest.json",
    ) -> Path:
        if not _ARTIFACT_ID.fullmatch(filename) or not filename.endswith(".json"):
            raise ValueError("manifest filename is not artifact-safe JSON")
        path = self.output_root / filename
        if path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Manifest already exists: {path}; use overwrite=True"
            )
        _write_json_file(
            path,
            {
                "artifact_type": "database_instance_manifest",
                "artifact_version": 1,
                "configuration": dict(configuration),
                "entries": [dict(entry) for entry in entries],
            },
        )
        return path


@dataclass(frozen=True, slots=True, kw_only=True)
class InstanceArtifact:
    sample_id: str
    schema_artifact: str
    runtime: RuntimeRecord
    plan: InstancePlan
    database: DatabaseInstance
    validation: InstanceValidationReport
    prior_plan: DatabasePriorPlan | None = None
    task_programs: tuple[TaskProgramPlan, ...] = ()


def load_instance_artifact(path: str | Path) -> InstanceArtifact:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "database_instance":
        raise ValueError("unsupported instance artifact type")
    if payload.get("artifact_version") not in {1, 2}:
        raise ValueError("unsupported instance artifact version")
    root = manifest_path.parent
    plan = InstancePlan.from_dict(
        json.loads((root / payload["plan"]).read_text(encoding="utf-8"))
    )
    runtime = RuntimeRecord.from_dict(
        json.loads((root / payload["runtime"]).read_text(encoding="utf-8"))
    )
    validation = InstanceValidationReport.from_dict(
        json.loads((root / payload["validation"]).read_text(encoding="utf-8"))
    )
    tables: list[TableData] = []
    for entry in payload["tables"]:
        with np.load(root / entry["artifact"], allow_pickle=False) as archive:
            columns = {column_id: archive[column_id] for column_id in archive.files}
        tables.append(TableData(table_id=entry["table_id"], columns=columns))
    database = DatabaseInstance(
        instance_id=payload["instance_id"],
        schema_id=payload["schema_id"],
        plan_id=payload["plan_id"],
        tables=tuple(tables),
    )
    prior_file = payload.get("database_prior_plan")
    prior_plan = (
        None
        if prior_file is None
        else DatabasePriorPlan.from_dict(json.loads((root / prior_file).read_text(encoding="utf-8")))
    )
    programs_file = payload.get("task_programs")
    task_programs = ()
    if programs_file is not None:
        program_payload = json.loads((root / programs_file).read_text(encoding="utf-8"))
        task_programs = tuple(TaskProgramPlan.from_dict(item) for item in program_payload.get("programs", ()))
    return InstanceArtifact(
        sample_id=payload["sample_id"],
        schema_artifact=payload["schema_artifact"],
        runtime=runtime,
        plan=plan,
        database=database,
        validation=validation,
        prior_plan=prior_plan,
        task_programs=task_programs,
    )


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
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


def _validation_report_from_dict(data: Mapping[str, Any]) -> ValidationReport:
    if not isinstance(data, Mapping):
        raise TypeError("validation payload must be a mapping")
    issue_payloads = data.get("issues")
    if not isinstance(issue_payloads, list):
        raise ValueError("validation issues must be a list")
    issues: list[ValidationIssue] = []
    for item in issue_payloads:
        if not isinstance(item, Mapping):
            raise ValueError("validation issue must be an object")
        issues.append(
            ValidationIssue(
                layer=ValidationLayer(item["layer"]),
                level=ValidationLevel(item["level"]),
                code=item["code"],
                message=item["message"],
                node_ids=tuple(item.get("node_ids", ())),
                edge_ids=tuple(item.get("edge_ids", ())),
                constraint_id=item.get("constraint_id"),
                motif_type=item.get("motif_type"),
            )
        )
    return ValidationReport(
        blueprint_id=data.get("blueprint_id"),
        issues=tuple(issues),
    )


__all__ = [
    "blueprint_to_dict",
    "validation_report_to_dict",
    "SchemaArtifactWriter",
    "SchemaArtifact",
    "load_schema_artifact",
    "InstanceArtifactWriter",
    "InstanceArtifact",
    "load_instance_artifact",
]
