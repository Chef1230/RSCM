# src/rdb_prior/pipeline.py
# -*- coding: utf-8 -*-
"""Schema-only pipeline: Blueprint sampling -> PhysicalSchema artifacts."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Callable

from rdb_prior.artifacts import (
    InstanceArtifactWriter,
    SchemaArtifactWriter,
    load_schema_artifact,
)
from rdb_prior.compilation.compiler import PhysicalCompilerConfig
from rdb_prior.extensions.defaults import default_extension_bundle
from rdb_prior.extensions.interfaces import ExtensionBundle
from rdb_prior.generation.database import DatabaseGenerator
from rdb_prior.instance.planner import InstancePlanner, InstancePlannerConfig
from rdb_prior.priors.compatibility import entity_event_candidates
from rdb_prior.priors.model import PriorFamily
from rdb_prior.priors.planner import PriorPlanner, PriorPlannerConfig
from rdb_prior.runtime import RuntimeContext, digest_config
from rdb_prior.schema.domain_prototypes import sample_semantic_schema
from rdb_prior.schema.graph import (
    SchemaGraphArtifactWriter,
    SchemaGraphConfig,
)
from rdb_prior.schema.sampler import BlueprintSamplerConfig
from rdb_prior.schema.validation import validate_blueprint
from rdb_prior.validation.checks import (
    validate_database_instance,
    validate_instance_plan,
)
from rdb_prior.task.pipeline import (
    TaskPipelineConfig,
    TaskPipelineResult,
    generate_tasks,
)
from rdb_prior.task.program import TaskExecutor, TaskProgramPlanner


_SAMPLE_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaPipelineConfig:
    output_root: Path
    num_schemas: int = 100
    base_seed: int = 42
    start_index: int = 0
    sample_id_prefix: str = "sample"
    overwrite: bool = False
    progress_every: int = 100
    project_version: str = "schema-pipeline-v1"
    sampler: BlueprintSamplerConfig = BlueprintSamplerConfig()
    compiler: PhysicalCompilerConfig = PhysicalCompilerConfig()
    graph: SchemaGraphConfig = SchemaGraphConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be pathlib.Path")
        for name, value in (
            ("num_schemas", self.num_schemas),
            ("base_seed", self.base_seed),
            ("start_index", self.start_index),
            ("progress_every", self.progress_every),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.num_schemas < 1:
            raise ValueError("num_schemas must be positive")
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if self.progress_every < 0:
            raise ValueError("progress_every must be non-negative")
        if not isinstance(self.sample_id_prefix, str) or not (
            _SAMPLE_PREFIX.fullmatch(self.sample_id_prefix)
        ):
            raise ValueError("sample_id_prefix is not artifact-safe")
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean")
        if (
            not isinstance(self.project_version, str)
            or not self.project_version.strip()
        ):
            raise ValueError("project_version must not be empty")
        if not isinstance(self.sampler, BlueprintSamplerConfig):
            raise TypeError("sampler must be BlueprintSamplerConfig")
        if not isinstance(self.compiler, PhysicalCompilerConfig):
            raise TypeError("compiler must be PhysicalCompilerConfig")
        if not isinstance(self.graph, SchemaGraphConfig):
            raise TypeError("graph must be SchemaGraphConfig")

    def to_dict(self) -> dict[str, object]:
        return {
            "num_schemas": self.num_schemas,
            "base_seed": self.base_seed,
            "start_index": self.start_index,
            "sample_id_prefix": self.sample_id_prefix,
            "overwrite": self.overwrite,
            "progress_every": self.progress_every,
            "project_version": self.project_version,
            "sampler": {
                "min_tables": self.sampler.min_tables,
                "max_tables": self.sampler.max_tables,
                "table_count_values": list(
                    self.sampler.table_count_values
                ),
                "table_count_weights": list(
                    self.sampler.table_count_weights
                ),
                "max_rank": self.sampler.max_rank,
                "max_extra_edges": self.sampler.max_extra_edges,
                "extra_edge_probability": (
                    self.sampler.extra_edge_probability
                ),
                "min_motif_occurrences": (
                    self.sampler.min_motif_occurrences
                ),
                "max_motif_occurrences": (
                    self.sampler.max_motif_occurrences
                ),
                "background_attachment_probability": (
                    self.sampler.background_attachment_probability
                ),
                "blueprint_id_prefix": (
                    self.sampler.blueprint_id_prefix
                ),
                "motif_weights": [
                    [motif_type, weight]
                    for motif_type, weight in self.sampler.motif_weights
                ],
            },
            "compiler": {
                "min_feature_columns": (
                    self.compiler.min_feature_columns
                ),
                "max_feature_columns": (
                    self.compiler.max_feature_columns
                ),
                "feature_columns_by_table_count": [
                    {
                        "table_count_min": rule.table_count_min,
                        "table_count_max": rule.table_count_max,
                        "min_columns": rule.min_columns,
                        "max_columns": rule.max_columns,
                    }
                    for rule in (
                        self.compiler.feature_columns_by_table_count
                    )
                ],
                "feature_columns_by_role": {
                    rule.role.value: {
                        "min_columns": rule.min_columns,
                        "max_columns": rule.max_columns,
                    }
                    for rule in self.compiler.feature_columns_by_role
                },
                "feature_nullable_probability": (
                    self.compiler.feature_nullable_probability
                ),
                "primary_key_names": list(
                    self.compiler.primary_key_names
                ),
                "schema_id_prefix": self.compiler.schema_id_prefix,
            },
            "graph": self.graph.to_dict(),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaPipelineResult:
    output_root: Path
    manifest_path: Path
    artifact_paths: tuple[Path, ...]
    dot_paths: tuple[Path, ...] = ()
    image_paths: tuple[Path, ...] = ()

    @property
    def generated_count(self) -> int:
        return len(self.artifact_paths)


@dataclass(frozen=True, slots=True, kw_only=True)
class InstancePipelineConfig:
    schema_manifest: Path
    output_root: Path
    count: int | None = None
    start_index: int = 0
    shard_id: int = 0
    num_shards: int = 1
    num_workers: int = 1
    overwrite: bool = False
    progress_every: int = 100
    project_version: str = "instance-pipeline-v1"
    planner: InstancePlannerConfig = InstancePlannerConfig()
    prior: PriorPlannerConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.schema_manifest, Path):
            raise TypeError("schema_manifest must be pathlib.Path")
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be pathlib.Path")
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
        if self.count is not None and (
            isinstance(self.count, bool) or not isinstance(self.count, int)
        ):
            raise TypeError("count must be an integer or None")
        if self.count is not None and self.count < 1:
            raise ValueError("count must be positive")
        if self.start_index < 0 or self.progress_every < 0:
            raise ValueError("start_index and progress_every must be non-negative")
        if self.num_shards < 1 or not 0 <= self.shard_id < self.num_shards:
            raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
        if self.num_workers < 1:
            raise ValueError("num_workers must be positive")
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean")
        if not isinstance(self.planner, InstancePlannerConfig):
            raise TypeError("planner must be InstancePlannerConfig")
        if self.prior is not None and not isinstance(self.prior, PriorPlannerConfig):
            raise TypeError("prior must be PriorPlannerConfig or None")

    def to_dict(self) -> dict[str, object]:
        planner = self.planner
        return {
            "schema_manifest": str(self.schema_manifest),
            "count": self.count,
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
                if name
                not in {
                    "scm_weights",
                    "root_cause_weights",
                    "role_scm",
                }
            }
            | {
                "scm_weights": [
                    [family.value, weight]
                    for family, weight in planner.scm_weights
                ],
                "root_cause_weights": [
                    [family.value, weight]
                    for family, weight in planner.root_cause_weights
                ],
                "role_scm": {
                    role.value: {
                        "scm_weights": [
                            [family.value, weight]
                            for family, weight in prior.scm_weights
                        ],
                        "root_cause_weights": [
                            [family.value, weight]
                            for family, weight in prior.root_cause_weights
                        ],
                        "signal_scale_multiplier": (
                            prior.signal_scale_multiplier
                        ),
                        "noise_scale_multiplier": (
                            prior.noise_scale_multiplier
                        ),
                        "activation_scale_multiplier": (
                            prior.activation_scale_multiplier
                        ),
                        "output_scale_multiplier": (
                            prior.output_scale_multiplier
                        ),
                    }
                    for role, prior in planner.role_scm
                },
                "prior": (
                    None
                    if self.prior is None
                    else {
                        "database_family_weights": [
                            [family.value, weight]
                            for family, weight in self.prior.database_family_weights
                        ],
                        "task_policy": self.prior.task_policy.to_dict(),
                        "state_dimension": self.prior.state_dimension,
                    }
                ),
            },
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class InstancePipelineResult:
    output_root: Path
    manifest_path: Path
    artifact_paths: tuple[Path, ...]

    @property
    def generated_count(self) -> int:
        return len(self.artifact_paths)


@dataclass(frozen=True, slots=True, kw_only=True)
class _InstanceWorkItem:
    order: int
    schema_path: Path
    output_root: Path
    planner_config: InstancePlannerConfig
    prior_config: PriorPlannerConfig | None
    overwrite: bool
    project_version: str
    config_digest: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _InstanceWorkResult:
    order: int
    sample_id: str
    artifact_path: Path
    manifest_entry: dict[str, object]


def generate_physical_schemas(
    config: SchemaPipelineConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
    extensions: ExtensionBundle | None = None,
) -> SchemaPipelineResult:
    """Generate a deterministic batch of validated physical schemas."""
    if not isinstance(config, SchemaPipelineConfig):
        raise TypeError("config must be SchemaPipelineConfig")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    if extensions is not None and not isinstance(extensions, ExtensionBundle):
        raise TypeError("extensions must be ExtensionBundle or None")

    configuration = config.to_dict()
    config_digest = digest_config(configuration)
    root_runtime = RuntimeContext(config.base_seed)
    components = extensions or default_extension_bundle(
        config.sampler,
        config.compiler,
    )
    writer = SchemaArtifactWriter(
        output_root=config.output_root,
        overwrite=config.overwrite,
    )
    graph_writer = SchemaGraphArtifactWriter(
        output_root=config.output_root,
        config=config.graph,
        overwrite=config.overwrite,
    )

    paths: list[Path] = []
    dot_paths: list[Path] = []
    image_paths: list[Path] = []
    entries: list[dict[str, object]] = []
    _LOGGER.info(
        "starting schema pipeline: count=%d start_index=%d output=%s",
        config.num_schemas,
        config.start_index,
        config.output_root,
    )

    for offset in range(config.num_schemas):
        sample_index = config.start_index + offset
        sample_id = f"{config.sample_id_prefix}_{sample_index:06d}"
        runtime = root_runtime.for_sample(sample_id)
        domain = components.domain.sample(runtime.child("domain"))
        blueprint = components.blueprint.sample(sample_id, runtime, domain)
        report = validate_blueprint(blueprint, raise_on_error=True)
        processes = components.process.instantiate(
            domain,
            blueprint,
            runtime.child("process"),
        )
        task_plan = components.task.plan(
            domain,
            blueprint,
            processes,
            runtime.child("task"),
        )
        design = components.design.sample(
            blueprint,
            task_plan,
            runtime.child("design"),
        )
        compilation = components.compiler.compile(
            blueprint,
            design,
            sample_id,
            runtime,
        )
        physical_schema = compilation.schema
        semantic_schema = sample_semantic_schema(
            physical_schema,
            runtime.child("semantic-schema"),
        )
        runtime_record = runtime.record(
            project_version=config.project_version,
            config_digest=config_digest,
            metadata={
                "stage": "physical_schema",
                "sample_id": sample_id,
            },
        )
        artifact_path = writer.commit(
            sample_id=sample_id,
            runtime=runtime_record,
            blueprint=blueprint,
            compilation=compilation,
            report=report,
            semantic_schema=semantic_schema,
            prior_compatibility={
                "temporal_event_candidate_count": len(
                    entity_event_candidates(blueprint, physical_schema)
                ),
                "implemented_families": [
                    PriorFamily.LEGACY_ROLE_SCM.value,
                    PriorFamily.TEMPORAL_EVENT.value,
                ],
            },
        )
        graph_artifacts = graph_writer.commit(
            sample_id=sample_id,
            schema=physical_schema,
        )
        paths.append(artifact_path)
        if graph_artifacts.dot_path is not None:
            dot_paths.append(graph_artifacts.dot_path)
        if graph_artifacts.image_path is not None:
            image_paths.append(graph_artifacts.image_path)
        _LOGGER.debug(
            "schema artifact committed: sample_id=%s path=%s tables=%d fks=%d",
            sample_id,
            artifact_path,
            len(physical_schema.tables),
            len(physical_schema.foreign_keys),
        )

        role_counts = Counter(
            node.role.value
            for node in blueprint.nodes
        )
        entry: dict[str, object] = {
            "sample_id": sample_id,
            "artifact": artifact_path.relative_to(
                config.output_root
            ).as_posix(),
            "derived_seed": runtime.seed(),
            "blueprint_id": blueprint.blueprint_id,
            "physical_schema_id": physical_schema.schema_id,
            "table_count": len(physical_schema.tables),
            "foreign_key_count": len(
                physical_schema.foreign_keys
            ),
            "motif_counts": dict(
                sorted(
                    Counter(
                        occurrence.motif_type
                        for occurrence in blueprint.motif_occurrences
                    ).items()
                )
            ),
            "role_counts": dict(sorted(role_counts.items())),
        }
        graph_manifest: dict[str, Path] = {}
        if graph_artifacts.dot_path is not None:
            graph_manifest["dot"] = graph_artifacts.dot_path
        if (
            config.graph.render_format is not None
            and graph_artifacts.image_path is not None
        ):
            graph_manifest[config.graph.render_format] = (
                graph_artifacts.image_path
            )
        entry["graph_artifacts"] = {
            format_name: path.relative_to(config.output_root).as_posix()
            for format_name, path in graph_manifest.items()
        }
        entries.append(entry)

        completed = offset + 1
        if progress is not None:
            progress(completed, config.num_schemas, sample_id)

    manifest_path = writer.write_manifest(
        configuration=configuration,
        entries=entries,
    )
    _LOGGER.info(
        "schema pipeline complete: generated=%d manifest=%s",
        len(paths),
        manifest_path,
    )
    return SchemaPipelineResult(
        output_root=config.output_root,
        manifest_path=manifest_path,
        artifact_paths=tuple(paths),
        dot_paths=tuple(dot_paths),
        image_paths=tuple(image_paths),
    )


def generate_database_instances(
    config: InstancePipelineConfig,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> InstancePipelineResult:
    """Materialize validated database instances from a schema manifest."""
    if not isinstance(config, InstancePipelineConfig):
        raise TypeError("config must be InstancePipelineConfig")
    manifest = json.loads(config.schema_manifest.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "physical_schema_manifest":
        raise ValueError("input is not a physical schema manifest")
    if manifest.get("artifact_version") not in {None, 2}:
        raise ValueError("unsupported physical schema manifest version")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("schema manifest entries must be a list")

    indexed = list(enumerate(entries))[config.start_index :]
    if config.count is not None:
        indexed = indexed[: config.count]
    selected = [
        (index, entry)
        for index, entry in indexed
        if index % config.num_shards == config.shard_id
    ]
    configuration = config.to_dict()
    config_digest = digest_config(configuration)
    writer = InstanceArtifactWriter(
        output_root=config.output_root,
        overwrite=config.overwrite,
    )
    work_items: list[_InstanceWorkItem] = []
    for order, (_index, entry) in enumerate(selected):
        if not isinstance(entry, dict):
            raise ValueError("schema manifest entry must be an object")
        if "artifact" not in entry:
            raise ValueError("schema manifest entry is missing artifact")
        schema_path = (config.schema_manifest.parent / entry["artifact"]).resolve()
        work_items.append(
            _InstanceWorkItem(
                order=order,
                schema_path=schema_path,
                output_root=config.output_root,
                planner_config=config.planner,
                prior_config=config.prior,
                overwrite=config.overwrite,
                project_version=config.project_version,
                config_digest=config_digest,
            )
        )
    _LOGGER.info(
        "starting instance pipeline: selected=%d workers=%d shard=%d/%d output=%s",
        len(selected),
        config.num_workers,
        config.shard_id,
        config.num_shards,
        config.output_root,
    )

    results = _run_instance_work_items(
        work_items,
        num_workers=config.num_workers,
        progress=progress,
    )
    artifact_paths = [result.artifact_path for result in results]
    output_entries = [result.manifest_entry for result in results]
    for result in results:
        _LOGGER.debug(
            "instance artifact committed: sample_id=%s path=%s tables=%s rows=%s",
            result.sample_id,
            result.artifact_path,
            result.manifest_entry["table_count"],
            result.manifest_entry["row_count"],
        )

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
        "instance pipeline complete: generated=%d manifest=%s",
        len(artifact_paths),
        manifest_path,
    )
    return InstancePipelineResult(
        output_root=config.output_root,
        manifest_path=manifest_path,
        artifact_paths=tuple(artifact_paths),
    )


def _run_instance_work_items(
    work_items: list[_InstanceWorkItem],
    *,
    num_workers: int,
    progress: Callable[[int, int, str], None] | None,
) -> list[_InstanceWorkResult]:
    if not work_items:
        return []
    results: list[_InstanceWorkResult] = []
    if num_workers == 1:
        for completed, item in enumerate(work_items, start=1):
            result = _generate_one_database_instance(item)
            results.append(result)
            if progress is not None:
                progress(completed, len(work_items), result.sample_id)
    else:
        max_workers = min(num_workers, len(work_items))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_generate_one_database_instance, item)
                for item in work_items
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if progress is not None:
                    progress(completed, len(work_items), result.sample_id)
    results.sort(key=lambda result: result.order)
    return results


def _generate_one_database_instance(
    item: _InstanceWorkItem,
) -> _InstanceWorkResult:
    artifact = load_schema_artifact(item.schema_path)
    runtime = artifact.runtime.restore_context().child("database-instance")
    schema = artifact.compilation.schema
    prior_plan = None
    if item.prior_config is not None:
        semantic_schema = artifact.semantic_schema or sample_semantic_schema(
            schema,
            runtime.child("semantic-schema"),
        )
        prior_plan = PriorPlanner(item.prior_config).plan(
            blueprint=artifact.blueprint,
            physical_schema=schema,
            semantic_schema=semantic_schema,
            runtime=runtime.child("prior"),
        )
    attempts = 1
    if prior_plan is not None and prior_plan.family is PriorFamily.TEMPORAL_EVENT:
        attempts = prior_plan.task_policy.max_materialization_attempts

    last_error: ValueError | None = None
    for attempt in range(attempts):
        # The original seed path is intentionally preserved for legacy plans.
        # Temporal retries get an explicitly derived, materialization-only seed.
        attempt_runtime = runtime if attempt == 0 else runtime.child("materialization-retry", attempt)
        try:
            plan = InstancePlanner(item.planner_config).plan(
                sample_id=artifact.sample_id,
                schema=schema,
                runtime=attempt_runtime,
                prior_plan=prior_plan,
            )
            task_programs = ()
            if prior_plan is not None and prior_plan.family is PriorFamily.TEMPORAL_EVENT:
                # Programs are sampled from the plan calendar before any rows are
                # materialized.  They are never calibrated from realized labels.
                task_programs = TaskProgramPlanner().plan(
                    schema=schema,
                    instance_plan=plan,
                    prior_plan=prior_plan,
                    runtime=attempt_runtime.child("task-program"),
                )
            plan_report = validate_instance_plan(schema, plan)
            if not plan_report.is_valid:
                raise ValueError(
                    f"invalid instance plan: {[issue.code for issue in plan_report.issues]}"
                )
            materialization = DatabaseGenerator().materialize(schema=schema, plan=plan)
            plan = materialization.plan
            database = materialization.database
            report = validate_database_instance(schema, plan, database)
            if not report.is_valid:
                raise ValueError(
                    f"invalid database instance: {[issue.code for issue in report.issues]}"
                )
            if task_programs:
                policy = prior_plan.task_policy
                executor = TaskExecutor()
                if any(
                    executor.execute(
                        sample_id=artifact.sample_id,
                        schema=schema,
                        database=database,
                        program=program,
                        positive_rate_min=policy.positive_rate_min,
                        positive_rate_max=policy.positive_rate_max,
                    ) is None
                    for program in task_programs
                ):
                    raise ValueError("pre-sampled task program failed post-materialization validation")
            break
        except ValueError as error:
            last_error = error
    else:
        assert last_error is not None
        raise ValueError(
            f"unable to materialize {artifact.sample_id} after {attempts} attempts: {last_error}"
        ) from last_error
    runtime_record = runtime.record(
        project_version=item.project_version,
        config_digest=item.config_digest,
        metadata={
            "stage": "database_instance",
            "sample_id": artifact.sample_id,
            "schema_id": schema.schema_id,
        },
    )
    artifact_path = InstanceArtifactWriter(
        output_root=item.output_root,
        overwrite=item.overwrite,
    ).commit(
        sample_id=artifact.sample_id,
        schema_artifact=str(item.schema_path),
        runtime=runtime_record,
        schema=schema,
        plan=plan,
        database=database,
        report=report,
        prior_plan=prior_plan,
        task_programs=task_programs,
    )
    row_count = sum(table.row_count for table in database.tables)
    return _InstanceWorkResult(
        order=item.order,
        sample_id=artifact.sample_id,
        artifact_path=artifact_path,
        manifest_entry={
            "sample_id": artifact.sample_id,
            "artifact": artifact_path.relative_to(item.output_root).as_posix(),
            "schema_artifact": str(item.schema_path),
            "schema_id": schema.schema_id,
            "plan_id": plan.plan_id,
            "table_count": len(database.tables),
            "row_count": row_count,
            "scm_counts": dict(
                sorted(
                    Counter(
                        table.feature_family.value for table in plan.tables
                    ).items()
                )
            ),
            "prior_family": plan.prior_family,
        },
    )


__all__ = [
    "SchemaPipelineConfig",
    "SchemaPipelineResult",
    "generate_physical_schemas",
    "InstancePipelineConfig",
    "InstancePipelineResult",
    "generate_database_instances",
    "TaskPipelineConfig",
    "TaskPipelineResult",
    "generate_tasks",
]
