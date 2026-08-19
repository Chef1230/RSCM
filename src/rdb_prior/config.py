"""Strict configuration loading for the schema-generation stage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Final

try:
    import yaml
except ImportError:  # pragma: no cover - declared project dependency.
    yaml = None

from rdb_prior.compilation.compiler import (
    PhysicalCompilerConfig,
    RoleFeatureRule,
    TableCountFeatureRule,
)
from rdb_prior.export.pipeline import RDBPFNExportConfig
from rdb_prior.instance.plan import (
    EventTemporalMechanism,
    FeatureSCMFamily,
    RootCauseFamily,
)
from rdb_prior.instance.planner import InstancePlannerConfig, RoleSCMPrior
from rdb_prior.pipeline import InstancePipelineConfig, SchemaPipelineConfig
from rdb_prior.routing.config import (
    RoutedH5Config,
    RouterModelConfig,
    RouterTrainingConfig,
)
from rdb_prior.schema.graph import SchemaGraphConfig
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.schema.spec import TableRole
from rdb_prior.task.model import CompositeFamily, TaskMechanism
from rdb_prior.task.pipeline import TaskPipelineConfig
from rdb_prior.task.planner import TaskPlannerConfig


_CONFIG_VERSION = 1
_PATH_OPTIONS = {
    "output_root",
    # Legacy stage-specific keys remain loadable for existing run configs.
    "schema_output_root",
    "schema_manifest",
    "instance_output_root",
    "instance_manifest",
    "task_output_root",
    "task_manifest",
    "rdbpfn_output_root",
    "router_output_root",
    "routed_output_root",
}


class SchemaConfigError(ValueError):
    """Raised when a schema pipeline configuration is malformed."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaConfigOverrides:
    """Optional CLI overrides; ``None`` preserves the YAML value."""

    output_root: Path | None = None
    num_schemas: int | None = None
    base_seed: int | None = None
    start_index: int | None = None
    sample_id_prefix: str | None = None
    progress_every: int | None = None
    overwrite: bool | None = None
    min_tables: int | None = None
    max_tables: int | None = None
    max_rank: int | None = None
    max_extra_edges: int | None = None
    extra_edge_probability: float | None = None
    min_feature_columns: int | None = None
    max_feature_columns: int | None = None
    feature_nullable_probability: float | None = None
    blueprint_id_prefix: str | None = None
    schema_id_prefix: str | None = None
    write_schema_dot: bool | None = None
    schema_graph_format: str | None = None
    graphviz_command: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class InstanceConfigOverrides:
    schema_manifest: Path | None = None
    output_root: Path | None = None
    count: int | None = None
    start_index: int | None = None
    shard_id: int | None = None
    num_shards: int | None = None
    num_workers: int | None = None
    progress_every: int | None = None
    overwrite: bool | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskConfigOverrides:
    instance_manifest: Path | None = None
    output_root: Path | None = None
    database_count: int | None = None
    tasks_per_database: int | None = None
    start_index: int | None = None
    shard_id: int | None = None
    num_shards: int | None = None
    num_workers: int | None = None
    progress_every: int | None = None
    overwrite: bool | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RDBPFNExportConfigOverrides:
    task_manifest: Path | None = None
    output_root: Path | None = None
    task_count: int | None = None
    start_index: int | None = None
    shard_id: int | None = None
    num_shards: int | None = None
    validation_fraction: float | None = None
    min_validation_rows: int | None = None
    compress: bool | None = None
    progress_every: int | None = None
    overwrite: bool | None = None
    h5_enabled: bool | None = None
    h5_output: Path | None = None
    rdbpfn_preprocessing_root: Path | None = None
    h5_run_dfs: bool | None = None
    dfs_depth: int | None = None
    dfs_jobs: int | None = None
    h5_total_rows: int | None = None
    h5_max_columns: int | None = None
    h5_seed: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RouterTrainingConfigOverrides:
    task_manifest: Path | None = None
    output_root: Path | None = None
    task_count: int | None = None
    start_index: int | None = None
    epochs: int | None = None
    device: str | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    prefetch_factor: int | None = None
    mixed_precision: str | None = None
    overwrite: bool | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutedH5ConfigOverrides:
    task_manifest: Path | None = None
    checkpoint: Path | None = None
    output_path: Path | None = None
    task_count: int | None = None
    start_index: int | None = None
    device: str | None = None
    overwrite: bool | None = None


def load_schema_pipeline_config(
    path: str | Path,
    *,
    overrides: SchemaConfigOverrides | None = None,
) -> SchemaPipelineConfig:
    """Load one YAML/JSON file into validated immutable runtime config."""
    config_path = Path(path).resolve()
    document = _load_document(config_path)
    root = _mapping(document, "config")
    _reject_unknown(
        root,
        {
            "config_version",
            "seed",
            "paths",
            "generation",
            "schema",
            "schema_graph",
            "motifs",
            "physical_design",
            "instance",
            "instance_generation",
            "task",
            "task_generation",
            "rdbpfn_export",
            "path_router",
            "routed_h5",
        },
        "config",
    )

    version = root.get("config_version", _CONFIG_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise SchemaConfigError("config.config_version must be an integer")
    if version != _CONFIG_VERSION:
        raise SchemaConfigError(
            f"Unsupported config_version {version!r}; expected "
            f"{_CONFIG_VERSION}"
        )

    paths = _section(
        root,
        "paths",
        _PATH_OPTIONS,
    )
    generation = _section(
        root,
        "generation",
        {
            "num_schemas",
            "start_index",
            "sample_id_prefix",
            "progress_every",
            "overwrite",
            "project_version",
        },
    )
    schema = _section(
        root,
        "schema",
        {
            "min_tables",
            "max_tables",
            "table_count_values",
            "table_count_weights",
            "max_rank",
            "max_extra_edges",
            "extra_edge_probability",
            "min_motif_occurrences",
            "max_motif_occurrences",
            "background_attachment_probability",
            "blueprint_id_prefix",
        },
    )
    schema_graph = _section(
        root,
        "schema_graph",
        {
            "write_dot",
            "render_format",
            "graphviz_command",
            "include_columns",
            "include_role_metadata",
        },
    )
    motifs = _section(root, "motifs", {"weights"})
    physical = _section(
        root,
        "physical_design",
        {
            "schema_id_prefix",
            "min_feature_columns",
            "max_feature_columns",
            "feature_nullable_probability",
            "primary_key_names",
            "feature_columns_by_table_count",
            "feature_columns_by_role",
        },
    )
    _section(root, "instance", _INSTANCE_OPTIONS)
    _section(root, "instance_generation", _INSTANCE_GENERATION_OPTIONS)
    _section(root, "task", _TASK_OPTIONS)
    _section(root, "task_generation", _TASK_GENERATION_OPTIONS)
    _section(root, "rdbpfn_export", _RDBPFN_EXPORT_OPTIONS)
    _section(root, "path_router", _PATH_ROUTER_OPTIONS)
    _section(root, "routed_h5", _ROUTED_H5_OPTIONS)

    cli = overrides or SchemaConfigOverrides()
    if not isinstance(cli, SchemaConfigOverrides):
        raise TypeError("overrides must be SchemaConfigOverrides or None")

    sampler_defaults = BlueprintSamplerConfig()
    compiler_defaults = PhysicalCompilerConfig()
    graph_defaults = SchemaGraphConfig()

    cli_bounds_overridden = (
        cli.min_tables is not None or cli.max_tables is not None
    )
    template_min, template_max, template_values, template_weights = (
        _template_distribution(config_path)
    )
    # A run config that overrides the table-count bounds without supplying
    # its own prior inherits the template's fixed values, which then fall
    # outside the new bounds. Reset to a uniform draw in that case; keep
    # explicitly-configured lists so genuinely inconsistent ones still fail.
    yaml_bounds_overridden = (
        schema.get("min_tables", sampler_defaults.min_tables) != template_min
        or schema.get("max_tables", sampler_defaults.max_tables) != template_max
    )
    merged_values = schema.get(
        "table_count_values", sampler_defaults.table_count_values
    )
    merged_weights = schema.get(
        "table_count_weights", sampler_defaults.table_count_weights
    )
    lists_from_template = (
        isinstance(merged_values, (list, tuple))
        and tuple(merged_values) == template_values
        and isinstance(merged_weights, (list, tuple))
        and tuple(merged_weights) == template_weights
    )
    if cli_bounds_overridden or (yaml_bounds_overridden and lists_from_template):
        table_count_values: tuple[int, ...] = ()
        table_count_weights: tuple[int | float, ...] = ()
    else:
        table_count_values = _integer_tuple(
            merged_values,
            "config.schema.table_count_values",
        )
        table_count_weights = _numeric_tuple(
            merged_weights,
            "config.schema.table_count_weights",
        )

    motif_weights = _motif_weights(
        motifs.get("weights"),
        default=sampler_defaults.motif_weights,
    )

    feature_bounds_overridden = (
        cli.min_feature_columns is not None
        or cli.max_feature_columns is not None
    )
    if feature_bounds_overridden:
        table_feature_rules: tuple[TableCountFeatureRule, ...] = ()
        role_feature_rules: tuple[RoleFeatureRule, ...] = ()
    else:
        table_feature_rules = _table_feature_rules(
            physical.get("feature_columns_by_table_count", ())
        )
        role_feature_rules = _role_feature_rules(
            physical.get("feature_columns_by_role", {})
        )

    primary_key_names = _string_tuple(
        physical.get(
            "primary_key_names",
            compiler_defaults.primary_key_names,
        ),
        "config.physical_design.primary_key_names",
    )

    output_value = _stage_output_path(paths, "schema")
    output_root = _resolve_output_root(
        config_path=config_path,
        configured=output_value,
        override=cli.output_root,
    )

    try:
        sampler = BlueprintSamplerConfig(
            min_tables=_override(
                cli.min_tables,
                schema.get("min_tables", sampler_defaults.min_tables),
            ),
            max_tables=_override(
                cli.max_tables,
                schema.get("max_tables", sampler_defaults.max_tables),
            ),
            table_count_values=table_count_values,
            table_count_weights=table_count_weights,
            max_rank=_override(
                cli.max_rank,
                schema.get("max_rank", sampler_defaults.max_rank),
            ),
            max_extra_edges=_override(
                cli.max_extra_edges,
                schema.get(
                    "max_extra_edges",
                    sampler_defaults.max_extra_edges,
                ),
            ),
            extra_edge_probability=_override(
                cli.extra_edge_probability,
                schema.get(
                    "extra_edge_probability",
                    sampler_defaults.extra_edge_probability,
                ),
            ),
            min_motif_occurrences=schema.get(
                "min_motif_occurrences",
                sampler_defaults.min_motif_occurrences,
            ),
            max_motif_occurrences=schema.get(
                "max_motif_occurrences",
                sampler_defaults.max_motif_occurrences,
            ),
            background_attachment_probability=schema.get(
                "background_attachment_probability",
                sampler_defaults.background_attachment_probability,
            ),
            blueprint_id_prefix=_override(
                cli.blueprint_id_prefix,
                schema.get(
                    "blueprint_id_prefix",
                    sampler_defaults.blueprint_id_prefix,
                ),
            ),
            motif_weights=motif_weights,
        )
        compiler = PhysicalCompilerConfig(
            min_feature_columns=_override(
                cli.min_feature_columns,
                physical.get(
                    "min_feature_columns",
                    compiler_defaults.min_feature_columns,
                ),
            ),
            max_feature_columns=_override(
                cli.max_feature_columns,
                physical.get(
                    "max_feature_columns",
                    compiler_defaults.max_feature_columns,
                ),
            ),
            feature_columns_by_table_count=table_feature_rules,
            feature_columns_by_role=role_feature_rules,
            feature_nullable_probability=_override(
                cli.feature_nullable_probability,
                physical.get(
                    "feature_nullable_probability",
                    compiler_defaults.feature_nullable_probability,
                ),
            ),
            primary_key_names=primary_key_names,
            schema_id_prefix=_override(
                cli.schema_id_prefix,
                physical.get(
                    "schema_id_prefix",
                    compiler_defaults.schema_id_prefix,
                ),
            ),
        )
        render_format = _override(
            cli.schema_graph_format,
            schema_graph.get(
                "render_format",
                graph_defaults.render_format,
            ),
        )
        if (
            isinstance(render_format, str)
            and render_format.strip().lower() == "none"
        ):
            render_format = None
        graph = SchemaGraphConfig(
            write_dot=_override(
                cli.write_schema_dot,
                schema_graph.get("write_dot", graph_defaults.write_dot),
            ),
            render_format=render_format,
            graphviz_command=_override(
                cli.graphviz_command,
                schema_graph.get(
                    "graphviz_command",
                    graph_defaults.graphviz_command,
                ),
            ),
            include_columns=schema_graph.get(
                "include_columns",
                graph_defaults.include_columns,
            ),
            include_role_metadata=schema_graph.get(
                "include_role_metadata",
                graph_defaults.include_role_metadata,
            ),
        )
        # Construction validates configured motif names against the active
        # library, so --validate-config-only fails before any artifact write.
        BlueprintSampler(sampler)
        return SchemaPipelineConfig(
            output_root=output_root,
            num_schemas=_override(
                cli.num_schemas,
                generation.get("num_schemas", 100),
            ),
            base_seed=_override(
                cli.base_seed,
                root.get("seed", 42),
            ),
            start_index=_override(
                cli.start_index,
                generation.get("start_index", 0),
            ),
            sample_id_prefix=_override(
                cli.sample_id_prefix,
                generation.get("sample_id_prefix", "sample"),
            ),
            overwrite=_override(
                cli.overwrite,
                generation.get("overwrite", False),
            ),
            progress_every=_override(
                cli.progress_every,
                generation.get("progress_every", 100),
            ),
            project_version=generation.get(
                "project_version",
                "schema-pipeline-v1",
            ),
            sampler=sampler,
            compiler=compiler,
            graph=graph,
        )
    except (TypeError, ValueError) as error:
        raise SchemaConfigError(
            f"Invalid schema pipeline config {config_path}: {error}"
        ) from error


_INSTANCE_OPTIONS = {
    "entity_rows_min",
    "entity_rows_max",
    "entity_rows_distribution",
    "entity_rows_median",
    "entity_rows_log_sigma",
    "lookup_rows_min",
    "lookup_rows_max",
    "lookup_rows_distribution",
    "event_fanout_min",
    "event_fanout_max",
    "bridge_rows_factor_min",
    "bridge_rows_factor_max",
    "detail_fanout_min",
    "detail_fanout_max",
    "entity_child_factor_min",
    "entity_child_factor_max",
    "population_scale_distribution",
    "population_scale_log_sigma",
    "max_rows_per_table",
    "latent_dimension",
    "optional_rate_min",
    "optional_rate_max",
    "affinity_strength",
    "degree_strength",
    "feature_missing_rate_min",
    "feature_missing_rate_max",
    "feature_missing_zero_probability",
    "feature_missing_beta_alpha",
    "feature_missing_beta_beta",
    "feature_noise_scale_min",
    "feature_noise_scale_max",
    "scm_signal_scale_min",
    "scm_signal_scale_max",
    "scm_meta_relative_std_min",
    "scm_meta_relative_std_max",
    "scm_activation_scale_min",
    "scm_activation_scale_max",
    "scm_output_scale_log_std",
    "scm_long_tail_probability",
    "scm_long_tail_alpha_min",
    "scm_long_tail_alpha_max",
    "categorical_cardinality_min",
    "categorical_cardinality_max",
    "categorical_high_cardinality_probability",
    "categorical_high_cardinality_min",
    "categorical_high_cardinality_max",
    "categorical_dirichlet_alpha_min",
    "categorical_dirichlet_alpha_max",
    "categorical_signal_strength_min",
    "categorical_signal_strength_max",
    "time_scale_seconds_min",
    "time_scale_seconds_max",
    "calendar_start_seconds_min",
    "calendar_start_seconds_max",
    "calendar_span_seconds_min",
    "calendar_span_seconds_max",
    "event_temporal_mechanism_weights",
    "burst_max_clusters",
    "burst_cluster_width_min",
    "burst_cluster_width_max",
    "churn_exponent_min",
    "churn_exponent_max",
    "seasonal_period_days_min",
    "seasonal_period_days_max",
    "scm_weights",
    "role_scm",
    "mlp_depth_min",
    "mlp_depth_max",
    "mlp_hidden_factor_min",
    "mlp_hidden_factor_max",
    "mlp_dropout_probability",
    "mlp_dropout_rate_min",
    "mlp_dropout_rate_max",
    "root_cause_weights",
}

_ROLE_SCM_OPTIONS = {
    "scm_weights",
    "root_cause_weights",
    "signal_scale_multiplier",
    "noise_scale_multiplier",
    "activation_scale_multiplier",
    "output_scale_multiplier",
}

_INSTANCE_GENERATION_OPTIONS = {
    "count",
    "start_index",
    "shard_id",
    "num_shards",
    "num_workers",
    "progress_every",
    "overwrite",
    "project_version",
}

_TASK_OPTIONS = {
    "tasks_per_database",
    "mechanism_weights",
    "support_fraction",
    "min_support_rows",
    "min_query_rows",
    "min_class_count_per_split",
    "max_classification_categories",
    "cutoff_quantile_min",
    "cutoff_quantile_max",
    "horizon_fraction_min",
    "horizon_fraction_max",
    "positive_rate_min",
    "positive_rate_max",
    "history_gated_frequency_weight_min",
    "history_gated_frequency_weight_max",
    "history_gated_silence_weight_min",
    "history_gated_silence_weight_max",
    "window_repeated_probability",
    "window_short_probability",
    "window_short_fraction_min",
    "window_short_fraction_max",
    "window_long_fraction_min",
    "window_long_fraction_max",
    "interaction_u_weight_min",
    "interaction_u_weight_max",
    "interaction_frequency_weight_min",
    "interaction_frequency_weight_max",
    "interaction_silence_weight_min",
    "interaction_silence_weight_max",
    "interaction_item_weight_min",
    "interaction_item_weight_max",
    "interaction_invert_probability",
    "composite_family_weights",
    "composite_max_path_depth",
    "composite_max_predicates",
    "composite_candidate_limit",
    "max_attempts_per_database",
    "require_full_task_count",
}

_TASK_GENERATION_OPTIONS = {
    "database_count",
    "start_index",
    "shard_id",
    "num_shards",
    "num_workers",
    "progress_every",
    "overwrite",
    "project_version",
}

_RDBPFN_EXPORT_OPTIONS = {
    "task_count",
    "start_index",
    "shard_id",
    "num_shards",
    "validation_fraction",
    "min_validation_rows",
    "compress",
    "progress_every",
    "overwrite",
    "project_version",
    "h5_enabled",
    "h5_output",
    "rdbpfn_preprocessing_root",
    "h5_run_dfs",
    "dfs_depth",
    "dfs_jobs",
    "h5_total_rows",
    "h5_max_columns",
    "h5_seed",
}

_PATH_ROUTER_OPTIONS = {
    "max_path_depth",
    "max_candidates",
    "top_k_paths",
    "max_source_columns",
    "rows_per_hop",
    "min_rows_per_hop",
    "max_target_columns",
    "max_rows_per_task",
    "token_dim",
    "type_embedding_dim",
    "path_feature_dim",
    "column_feature_dim",
    "router_hidden_dim",
    "transformer_heads",
    "transformer_layers",
    "dropout",
    "max_classes",
    "aggregation",
    "temperature",
    "epochs",
    "learning_rate",
    "weight_decay",
    "gradient_clip",
    "validation_fraction",
    "task_count",
    "start_index",
    "seed",
    "device",
    "lambda_route",
    "lambda_cost",
    "lambda_sparse",
    "lambda_diversity",
    "overwrite",
    "progress_every",
    "batch_size",
    "num_workers",
    "prefetch_factor",
    "artifact_cache_size",
    "mixed_precision",
}

_ROUTED_H5_OPTIONS = {
    "checkpoint",
    "output_path",
    "task_count",
    "start_index",
    "device",
    "overwrite",
}


def load_instance_pipeline_config(
    path: str | Path,
    *,
    overrides: InstanceConfigOverrides | None = None,
) -> InstancePipelineConfig:
    """Load and validate stage-02 instance generation configuration."""
    config_path = Path(path).resolve()
    # Validate the shared schema sections too: one file remains the source of
    # truth for the staged shell pipeline.
    load_schema_pipeline_config(config_path)
    root = _mapping(_load_document(config_path), "config")
    paths = _section(
        root,
        "paths",
        _PATH_OPTIONS,
    )
    instance = _section(root, "instance", _INSTANCE_OPTIONS)
    generation = _section(
        root,
        "instance_generation",
        _INSTANCE_GENERATION_OPTIONS,
    )
    cli = overrides or InstanceConfigOverrides()
    if not isinstance(cli, InstanceConfigOverrides):
        raise TypeError("overrides must be InstanceConfigOverrides or None")
    defaults = InstancePlannerConfig()

    scm_weights = _weight_mapping(
        instance.get("scm_weights"),
        defaults.scm_weights,
        FeatureSCMFamily,
        "config.instance.scm_weights",
        sort_keys=True,
    )
    root_cause_weights = _weight_mapping(
        instance.get("root_cause_weights"),
        defaults.root_cause_weights,
        RootCauseFamily,
        "config.instance.root_cause_weights",
    )
    mechanism_weights = _weight_mapping(
        instance.get("event_temporal_mechanism_weights"),
        defaults.event_temporal_mechanism_weights,
        EventTemporalMechanism,
        "config.instance.event_temporal_mechanism_weights",
        sort_keys=True,
    )
    role_scm = _role_scm_mapping(
        instance.get("role_scm"),
        scm_weights=scm_weights,
        root_cause_weights=root_cause_weights,
    )

    schema_output = _stage_output_path(paths, "schema")
    schema_manifest_value = _stage_manifest_path(
        paths,
        "schema",
        schema_output,
    )
    output_value = _stage_output_path(paths, "instance")

    try:
        planner_values = {
            name: instance.get(name, getattr(defaults, name))
            for name in defaults.__dataclass_fields__
            if name
            not in {
                "scm_weights",
                "root_cause_weights",
                "role_scm",
                "event_temporal_mechanism_weights",
            }
        }
        planner = InstancePlannerConfig(
            **planner_values,
            scm_weights=scm_weights,
            root_cause_weights=root_cause_weights,
            role_scm=role_scm,
            event_temporal_mechanism_weights=mechanism_weights,
        )
        return InstancePipelineConfig(
            schema_manifest=_resolve_output_root(
                config_path=config_path,
                configured=schema_manifest_value,
                override=cli.schema_manifest,
            ),
            output_root=_resolve_output_root(
                config_path=config_path,
                configured=output_value,
                override=cli.output_root,
            ),
            count=_override(cli.count, generation.get("count")),
            start_index=_override(
                cli.start_index,
                generation.get("start_index", 0),
            ),
            shard_id=_override(cli.shard_id, generation.get("shard_id", 0)),
            num_shards=_override(
                cli.num_shards,
                generation.get("num_shards", 1),
            ),
            num_workers=_override(
                cli.num_workers,
                generation.get("num_workers", 1),
            ),
            progress_every=_override(
                cli.progress_every,
                generation.get("progress_every", 100),
            ),
            overwrite=_override(
                cli.overwrite,
                generation.get("overwrite", False),
            ),
            project_version=generation.get(
                "project_version", "instance-pipeline-v1"
            ),
            planner=planner,
        )
    except (TypeError, ValueError) as error:
        raise SchemaConfigError(
            f"Invalid instance pipeline config {config_path}: {error}"
        ) from error


def load_task_pipeline_config(
    path: str | Path,
    *,
    overrides: TaskConfigOverrides | None = None,
) -> TaskPipelineConfig:
    """Load and validate stage-03 task generation configuration."""
    config_path = Path(path).resolve()
    load_instance_pipeline_config(config_path)
    root = _mapping(_load_document(config_path), "config")
    paths = _section(
        root,
        "paths",
        _PATH_OPTIONS,
    )
    task = _section(root, "task", _TASK_OPTIONS)
    generation = _section(
        root,
        "task_generation",
        _TASK_GENERATION_OPTIONS,
    )
    cli = overrides or TaskConfigOverrides()
    if not isinstance(cli, TaskConfigOverrides):
        raise TypeError("overrides must be TaskConfigOverrides or None")
    defaults = TaskPlannerConfig()

    raw_weights = task.get("mechanism_weights")
    if raw_weights is None:
        mechanism_weights = defaults.mechanism_weights
    else:
        weights = _mapping(raw_weights, "config.task.mechanism_weights")
        try:
            mechanism_weights = tuple(
                (TaskMechanism(name), float(weight))
                for name, weight in sorted(weights.items())
            )
        except (TypeError, ValueError) as error:
            raise SchemaConfigError(
                f"Invalid config.task.mechanism_weights: {error}"
            ) from error

    raw_composite_weights = task.get("composite_family_weights")
    if raw_composite_weights is None:
        composite_family_weights = defaults.composite_family_weights
    else:
        composite_weights = _mapping(
            raw_composite_weights, "config.task.composite_family_weights"
        )
        try:
            composite_family_weights = tuple(
                (CompositeFamily(name), float(weight))
                for name, weight in sorted(composite_weights.items())
            )
        except (TypeError, ValueError) as error:
            raise SchemaConfigError(
                f"Invalid config.task.composite_family_weights: {error}"
            ) from error

    instance_output = _stage_output_path(paths, "instance")
    manifest_value = _stage_manifest_path(
        paths,
        "instance",
        instance_output,
    )
    output_value = _stage_output_path(paths, "task")

    try:
        planner_values = {
            name: task.get(name, getattr(defaults, name))
            for name in defaults.__dataclass_fields__
            if name
            not in {
                "mechanism_weights",
                "composite_family_weights",
                "tasks_per_database",
            }
        }
        planner = TaskPlannerConfig(
            **planner_values,
            tasks_per_database=_override(
                cli.tasks_per_database,
                task.get("tasks_per_database", defaults.tasks_per_database),
            ),
            mechanism_weights=mechanism_weights,
            composite_family_weights=composite_family_weights,
        )
        return TaskPipelineConfig(
            instance_manifest=_resolve_output_root(
                config_path=config_path,
                configured=manifest_value,
                override=cli.instance_manifest,
            ),
            output_root=_resolve_output_root(
                config_path=config_path,
                configured=output_value,
                override=cli.output_root,
            ),
            database_count=_override(
                cli.database_count,
                generation.get("database_count"),
            ),
            start_index=_override(
                cli.start_index,
                generation.get("start_index", 0),
            ),
            shard_id=_override(cli.shard_id, generation.get("shard_id", 0)),
            num_shards=_override(
                cli.num_shards,
                generation.get("num_shards", 1),
            ),
            num_workers=_override(
                cli.num_workers,
                generation.get("num_workers", 1),
            ),
            progress_every=_override(
                cli.progress_every,
                generation.get("progress_every", 100),
            ),
            overwrite=_override(
                cli.overwrite,
                generation.get("overwrite", False),
            ),
            project_version=generation.get(
                "project_version", "task-pipeline-v1"
            ),
            planner=planner,
        )
    except (TypeError, ValueError) as error:
        raise SchemaConfigError(
            f"Invalid task pipeline config {config_path}: {error}"
        ) from error


def load_rdbpfn_export_config(
    path: str | Path,
    *,
    overrides: RDBPFNExportConfigOverrides | None = None,
) -> RDBPFNExportConfig:
    """Load and validate stage-04 RDBPFN export configuration."""
    config_path = Path(path).resolve()
    load_task_pipeline_config(config_path)
    root = _mapping(_load_document(config_path), "config")
    paths = _section(
        root,
        "paths",
        _PATH_OPTIONS,
    )
    export = _section(root, "rdbpfn_export", _RDBPFN_EXPORT_OPTIONS)
    cli = overrides or RDBPFNExportConfigOverrides()
    if not isinstance(cli, RDBPFNExportConfigOverrides):
        raise TypeError(
            "overrides must be RDBPFNExportConfigOverrides or None"
        )

    task_output = _stage_output_path(paths, "task")
    manifest_value = _stage_manifest_path(paths, "task", task_output)
    output_value = _stage_output_path(paths, "rdbpfn")
    h5_output_value = export.get("h5_output")
    preprocessing_value = export.get(
        "rdbpfn_preprocessing_root",
        "../RDBPFN/data_preprocessing",
    )
    if h5_output_value is not None and not isinstance(
        h5_output_value, (str, Path)
    ):
        raise SchemaConfigError("config.rdbpfn_export.h5_output must be a path string")
    if not isinstance(preprocessing_value, (str, Path)):
        raise SchemaConfigError(
            "config.rdbpfn_export.rdbpfn_preprocessing_root must be a path string"
        )

    resolved_h5_output = None
    if cli.h5_output is not None or h5_output_value is not None:
        resolved_h5_output = _resolve_output_root(
            config_path=config_path,
            configured=Path(h5_output_value or "."),
            override=cli.h5_output,
        )

    try:
        return RDBPFNExportConfig(
            task_manifest=_resolve_output_root(
                config_path=config_path,
                configured=manifest_value,
                override=cli.task_manifest,
            ),
            output_root=_resolve_output_root(
                config_path=config_path,
                configured=output_value,
                override=cli.output_root,
            ),
            task_count=_override(cli.task_count, export.get("task_count")),
            start_index=_override(
                cli.start_index,
                export.get("start_index", 0),
            ),
            shard_id=_override(cli.shard_id, export.get("shard_id", 0)),
            num_shards=_override(
                cli.num_shards,
                export.get("num_shards", 1),
            ),
            validation_fraction=_override(
                cli.validation_fraction,
                export.get("validation_fraction", 0.2),
            ),
            min_validation_rows=_override(
                cli.min_validation_rows,
                export.get("min_validation_rows", 8),
            ),
            compress=_override(cli.compress, export.get("compress", True)),
            progress_every=_override(
                cli.progress_every,
                export.get("progress_every", 100),
            ),
            overwrite=_override(
                cli.overwrite,
                export.get("overwrite", False),
            ),
            project_version=export.get(
                "project_version", "rdbpfn-export-v1"
            ),
            h5_enabled=_override(
                cli.h5_enabled,
                export.get("h5_enabled", False),
            ),
            h5_output=resolved_h5_output,
            rdbpfn_preprocessing_root=_resolve_output_root(
                config_path=config_path,
                configured=Path(preprocessing_value),
                override=cli.rdbpfn_preprocessing_root,
            ),
            h5_run_dfs=_override(
                cli.h5_run_dfs,
                export.get("h5_run_dfs", True),
            ),
            dfs_depth=_override(cli.dfs_depth, export.get("dfs_depth", 1)),
            dfs_jobs=_override(cli.dfs_jobs, export.get("dfs_jobs", 4)),
            h5_total_rows=_override(
                cli.h5_total_rows,
                export.get("h5_total_rows", 600),
            ),
            h5_max_columns=_override(
                cli.h5_max_columns,
                export.get("h5_max_columns", 60),
            ),
            h5_seed=_override(cli.h5_seed, export.get("h5_seed", 42)),
        )
    except (TypeError, ValueError) as error:
        raise SchemaConfigError(
            f"Invalid RDBPFN export config {config_path}: {error}"
        ) from error


def load_router_training_config(
    path: str | Path,
    *,
    overrides: RouterTrainingConfigOverrides | None = None,
) -> RouterTrainingConfig:
    """Load the support-conditioned sparse MLP router configuration."""
    config_path = Path(path).resolve()
    load_task_pipeline_config(config_path)
    root = _mapping(_load_document(config_path), "config")
    paths = _section(root, "paths", _PATH_OPTIONS)
    router = _section(root, "path_router", _PATH_ROUTER_OPTIONS)
    cli = overrides or RouterTrainingConfigOverrides()
    if not isinstance(cli, RouterTrainingConfigOverrides):
        raise TypeError(
            "overrides must be RouterTrainingConfigOverrides or None"
        )
    task_output = _stage_output_path(paths, "task")
    task_manifest = _stage_manifest_path(paths, "task", task_output)
    router_output = _stage_output_path(paths, "router")
    model_defaults = RouterModelConfig()
    try:
        model = RouterModelConfig(
            **{
                name: router.get(name, getattr(model_defaults, name))
                for name in model_defaults.__dataclass_fields__
            }
        )
        return RouterTrainingConfig(
            task_manifest=_resolve_output_root(
                config_path=config_path,
                configured=task_manifest,
                override=cli.task_manifest,
            ),
            output_root=_resolve_output_root(
                config_path=config_path,
                configured=router_output,
                override=cli.output_root,
            ),
            model=model,
            epochs=_override(cli.epochs, router.get("epochs", 10)),
            learning_rate=router.get("learning_rate", 3e-4),
            weight_decay=router.get("weight_decay", 1e-4),
            gradient_clip=router.get("gradient_clip", 1.0),
            validation_fraction=router.get("validation_fraction", 0.1),
            task_count=_override(cli.task_count, router.get("task_count")),
            start_index=_override(
                cli.start_index, router.get("start_index", 0)
            ),
            seed=router.get("seed", root.get("seed", 42)),
            device=_override(cli.device, router.get("device", "auto")),
            lambda_route=router.get("lambda_route", 1.0),
            lambda_cost=router.get("lambda_cost", 0.05),
            lambda_sparse=router.get("lambda_sparse", 0.05),
            lambda_diversity=router.get("lambda_diversity", 0.05),
            overwrite=_override(
                cli.overwrite, router.get("overwrite", False)
            ),
            progress_every=router.get("progress_every", 50),
            batch_size=_override(cli.batch_size, router.get("batch_size", 1)),
            num_workers=_override(cli.num_workers, router.get("num_workers", 0)),
            prefetch_factor=_override(
                cli.prefetch_factor, router.get("prefetch_factor", 2)
            ),
            artifact_cache_size=router.get("artifact_cache_size", 16),
            mixed_precision=_override(
                cli.mixed_precision, router.get("mixed_precision", "none")
            ),
        )
    except (TypeError, ValueError) as error:
        raise SchemaConfigError(
            f"Invalid sparse router config {config_path}: {error}"
        ) from error


def load_routed_h5_config(
    path: str | Path,
    *,
    overrides: RoutedH5ConfigOverrides | None = None,
) -> RoutedH5Config:
    """Load token-native H5 export configuration."""
    config_path = Path(path).resolve()
    load_task_pipeline_config(config_path)
    root = _mapping(_load_document(config_path), "config")
    paths = _section(root, "paths", _PATH_OPTIONS)
    routed = _section(root, "routed_h5", _ROUTED_H5_OPTIONS)
    cli = overrides or RoutedH5ConfigOverrides()
    if not isinstance(cli, RoutedH5ConfigOverrides):
        raise TypeError("overrides must be RoutedH5ConfigOverrides or None")
    task_output = _stage_output_path(paths, "task")
    task_manifest = _stage_manifest_path(paths, "task", task_output)
    router_output = _stage_output_path(paths, "router")
    routed_output = _stage_output_path(paths, "routed")
    checkpoint_value = routed.get(
        "checkpoint", router_output / "checkpoints" / "best.pt"
    )
    output_value = routed.get(
        "output_path", routed_output / "routed_tasks.h5"
    )
    if not isinstance(checkpoint_value, (str, Path)):
        raise SchemaConfigError("config.routed_h5.checkpoint must be a path")
    if not isinstance(output_value, (str, Path)):
        raise SchemaConfigError("config.routed_h5.output_path must be a path")
    try:
        return RoutedH5Config(
            task_manifest=_resolve_output_root(
                config_path=config_path,
                configured=task_manifest,
                override=cli.task_manifest,
            ),
            checkpoint=_resolve_output_root(
                config_path=config_path,
                configured=Path(checkpoint_value),
                override=cli.checkpoint,
            ),
            output_path=_resolve_output_root(
                config_path=config_path,
                configured=Path(output_value),
                override=cli.output_path,
            ),
            task_count=_override(cli.task_count, routed.get("task_count")),
            start_index=_override(
                cli.start_index, routed.get("start_index", 0)
            ),
            device=_override(cli.device, routed.get("device", "auto")),
            overwrite=_override(
                cli.overwrite, routed.get("overwrite", False)
            ),
        )
    except (TypeError, ValueError) as error:
        raise SchemaConfigError(
            f"Invalid routed H5 config {config_path}: {error}"
        ) from error


def _weight_mapping(
    raw: Any,
    default: tuple[tuple[Any, float], ...],
    family: type,
    path: str,
    *,
    sort_keys: bool = False,
) -> tuple[tuple[Any, float], ...]:
    """Parse a ``{family_name: weight}`` YAML mapping into enum-weight pairs.

    ``sort_keys=True`` preserves the historical alphabetical ordering used
    by scm_weights; otherwise YAML insertion order is kept so a template can
    mirror the dataclass default ordering exactly.
    """
    if raw is None:
        return default
    weights = _mapping(raw, path)
    items = sorted(weights.items()) if sort_keys else weights.items()
    try:
        return tuple((family(name), float(weight)) for name, weight in items)
    except (TypeError, ValueError) as error:
        raise SchemaConfigError(f"Invalid {path}: {error}") from error


def _role_scm_mapping(
    raw: Any,
    *,
    scm_weights: tuple[tuple[FeatureSCMFamily, float], ...],
    root_cause_weights: tuple[tuple[RootCauseFamily, float], ...],
) -> tuple[tuple[TableRole, RoleSCMPrior], ...]:
    """Parse optional table-role-specific SCM family and scale priors."""
    if raw is None:
        return ()
    entries = _mapping(raw, "config.instance.role_scm")
    result: list[tuple[TableRole, RoleSCMPrior]] = []
    for role_name, entry in entries.items():
        path = f"config.instance.role_scm.{role_name}"
        try:
            role = TableRole(role_name)
        except ValueError as error:
            allowed_roles = ", ".join(role.value for role in TableRole)
            raise SchemaConfigError(
                f"Unknown role {role_name!r}; expected one of: "
                f"{allowed_roles}"
            ) from error
        data = _mapping(entry, path)
        _reject_unknown(data, _ROLE_SCM_OPTIONS, path)
        if role is TableRole.LOOKUP:
            default_scm_weights = ((FeatureSCMFamily.EXOGENOUS, 1.0),)
            default_root_cause_weights = (
                (RootCauseFamily.STANDARD_NORMAL, 1.0),
            )
        else:
            default_scm_weights = scm_weights
            default_root_cause_weights = root_cause_weights
        try:
            prior = RoleSCMPrior(
                scm_weights=_weight_mapping(
                    data.get("scm_weights"),
                    default_scm_weights,
                    FeatureSCMFamily,
                    f"{path}.scm_weights",
                    sort_keys=True,
                ),
                root_cause_weights=_weight_mapping(
                    data.get("root_cause_weights"),
                    default_root_cause_weights,
                    RootCauseFamily,
                    f"{path}.root_cause_weights",
                ),
                signal_scale_multiplier=data.get(
                    "signal_scale_multiplier",
                    1.0,
                ),
                noise_scale_multiplier=data.get(
                    "noise_scale_multiplier",
                    1.0,
                ),
                activation_scale_multiplier=data.get(
                    "activation_scale_multiplier",
                    1.0,
                ),
                output_scale_multiplier=data.get(
                    "output_scale_multiplier",
                    1.0,
                ),
            )
        except SchemaConfigError:
            raise
        except (TypeError, ValueError) as error:
            raise SchemaConfigError(f"Invalid {path}: {error}") from error
        result.append((role, prior))
    return tuple(result)


# Project-wide default template. Every loaded config is deep-merged on top
# of this base: fields present in the requested file win, missing fields
# fall back to the template values.
_DEFAULT_CONFIG_TEMPLATE: Final[Path] = (
    Path(__file__).resolve().parents[2] / "configs" / "template.yaml"
)


def _load_document(path: Path) -> Any:
    document = _read_config_file(path)
    template_path = _resolve_config_template(path)
    if template_path is None:
        return document
    base = _read_config_file(template_path)
    if not isinstance(base, Mapping) or not isinstance(document, Mapping):
        return document
    return _deep_merge(dict(base), dict(document))


def _resolve_config_template(path: Path) -> Path | None:
    """Locate the fallback template for *path*, or None to skip merging.

    The environment variable ``RDB_PRIOR_CONFIG_TEMPLATE`` overrides the
    default location; set it to an empty string to disable merging.
    """
    override = os.environ.get("RDB_PRIOR_CONFIG_TEMPLATE")
    if override is not None:
        if not override.strip():
            return None
        candidate = Path(override).expanduser().resolve()
    else:
        candidate = _DEFAULT_CONFIG_TEMPLATE
    if candidate == path or not candidate.is_file():
        return None
    return candidate


def _template_distribution(
    config_path: Path,
) -> tuple[int, int, tuple[int, ...], tuple[int | float, ...]]:
    """``(min_tables, max_tables, table_count_values, table_count_weights)``
    from the resolved fallback template.

    Falls back to the dataclass defaults when no template applies, i.e. when
    loading the template itself or when merging is disabled. Used to tell
    whether a run config overrode the table-count bounds on top of the
    template's own fixed prior.
    """
    defaults = BlueprintSamplerConfig()
    fallback = (
        defaults.min_tables,
        defaults.max_tables,
        defaults.table_count_values,
        defaults.table_count_weights,
    )
    template_path = _resolve_config_template(config_path)
    if template_path is None:
        return fallback
    document = _read_config_file(template_path)
    if not isinstance(document, Mapping):
        return fallback
    schema = document.get("schema")
    if not isinstance(schema, Mapping):
        return fallback
    values = schema.get("table_count_values", defaults.table_count_values)
    weights = schema.get("table_count_weights", defaults.table_count_weights)
    if not isinstance(values, (list, tuple)) or not isinstance(
        weights, (list, tuple)
    ):
        return fallback
    return (
        schema.get("min_tables", defaults.min_tables),
        schema.get("max_tables", defaults.max_tables),
        tuple(values),
        tuple(weights),
    )


def _deep_merge(base: Any, override: Any) -> Any:
    """Merge *override* onto *base*.

    Nested mappings merge recursively; every other value (scalars and
    lists alike) is replaced wholesale by the override, including explicit
    nulls.
    """
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged: dict[Any, Any] = dict(base)
        for key, value in override.items():
            merged[key] = (
                _deep_merge(base[key], value) if key in base else value
            )
        return merged
    return override


def _read_config_file(path: Path) -> Any:
    if not path.is_file():
        raise SchemaConfigError(f"Config file does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return json.loads(text)
        if yaml is None:
            raise SchemaConfigError(
                "PyYAML is required to load YAML configuration files"
            )
        return yaml.safe_load(text)
    except SchemaConfigError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise SchemaConfigError(f"Cannot read config {path}: {error}") from error
    except Exception as error:
        if yaml is not None and isinstance(error, yaml.YAMLError):
            raise SchemaConfigError(
                f"Cannot parse YAML config {path}: {error}"
            ) from error
        raise


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        raise SchemaConfigError(f"{path} must be a mapping, not null")
    if not isinstance(value, Mapping):
        raise SchemaConfigError(f"{path} must be a mapping")
    result = dict(value)
    if not all(isinstance(key, str) for key in result):
        raise SchemaConfigError(f"{path} keys must be strings")
    return result


def _section(
    root: Mapping[str, Any],
    name: str,
    allowed: set[str],
) -> dict[str, Any]:
    value = root.get(name, {})
    section = _mapping(value, f"config.{name}")
    _reject_unknown(section, allowed, f"config.{name}")
    return section


def _reject_unknown(
    values: Mapping[str, Any],
    allowed: set[str],
    path: str,
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise SchemaConfigError(
            f"{path} contains unknown option(s): {', '.join(unknown)}"
        )


def _sequence(value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise SchemaConfigError(f"{path} must be a list")
    return tuple(value)


def _integer_tuple(value: Any, path: str) -> tuple[int, ...]:
    values = _sequence(value, path)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        raise SchemaConfigError(f"{path} items must be integers")
    return values


def _numeric_tuple(value: Any, path: str) -> tuple[int | float, ...]:
    values = _sequence(value, path)
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in values
    ):
        raise SchemaConfigError(f"{path} items must be numeric")
    return values


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    values = _sequence(value, path)
    if any(not isinstance(item, str) for item in values):
        raise SchemaConfigError(f"{path} items must be strings")
    return values


def _motif_weights(
    value: Any,
    *,
    default: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, int | float], ...]:
    if value is None:
        return default
    weights = _mapping(value, "config.motifs.weights")
    result: list[tuple[str, int | float]] = []
    for motif_type, weight in sorted(weights.items()):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise SchemaConfigError(
                f"config.motifs.weights.{motif_type} must be numeric"
            )
        result.append((motif_type, weight))
    return tuple(result)


def _table_feature_rules(value: Any) -> tuple[TableCountFeatureRule, ...]:
    if value is None:
        # Explicit null clears the template's rules (back to fallback columns).
        return ()
    entries = _sequence(
        value,
        "config.physical_design.feature_columns_by_table_count",
    )
    rules: list[TableCountFeatureRule] = []
    allowed = {
        "table_count_min",
        "table_count_max",
        "min_columns",
        "max_columns",
    }
    for index, entry in enumerate(entries):
        path = (
            "config.physical_design.feature_columns_by_table_count"
            f"[{index}]"
        )
        data = _mapping(entry, path)
        _reject_unknown(data, allowed, path)
        missing = sorted(allowed - set(data))
        if missing:
            raise SchemaConfigError(
                f"{path} is missing: {', '.join(missing)}"
            )
        try:
            rules.append(TableCountFeatureRule(**data))
        except (TypeError, ValueError) as error:
            raise SchemaConfigError(f"Invalid {path}: {error}") from error
    return tuple(rules)


def _role_feature_rules(value: Any) -> tuple[RoleFeatureRule, ...]:
    if value is None:
        # Explicit null clears the template's rules (back to fallback columns).
        return ()
    roles = _mapping(
        value,
        "config.physical_design.feature_columns_by_role",
    )
    rules: list[RoleFeatureRule] = []
    allowed = {"min_columns", "max_columns"}
    for role_name, entry in sorted(roles.items()):
        path = (
            "config.physical_design.feature_columns_by_role."
            f"{role_name}"
        )
        try:
            role = TableRole(role_name)
        except ValueError as error:
            allowed_roles = ", ".join(role.value for role in TableRole)
            raise SchemaConfigError(
                f"Unknown role {role_name!r}; expected one of: "
                f"{allowed_roles}"
            ) from error
        data = _mapping(entry, path)
        _reject_unknown(data, allowed, path)
        missing = sorted(allowed - set(data))
        if missing:
            raise SchemaConfigError(
                f"{path} is missing: {', '.join(missing)}"
            )
        try:
            rules.append(RoleFeatureRule(role=role, **data))
        except (TypeError, ValueError) as error:
            raise SchemaConfigError(f"Invalid {path}: {error}") from error
    return tuple(rules)


def _stage_output_path(
    paths: Mapping[str, Any],
    stage: str,
) -> Path:
    legacy_key = f"{stage}_output_root"
    if legacy_key in paths:
        value = paths[legacy_key]
        field = f"config.paths.{legacy_key}"
    else:
        value = paths.get("output_root", "outputs/v1")
        field = "config.paths.output_root"
        if isinstance(value, (str, Path)):
            value = Path(value) / stage
    if not isinstance(value, (str, Path)):
        raise SchemaConfigError(f"{field} must be a path string")
    return Path(value)


def _stage_manifest_path(
    paths: Mapping[str, Any],
    stage: str,
    output_path: Path,
) -> Path:
    key = f"{stage}_manifest"
    value = paths.get(key, output_path / "manifest.json")
    if not isinstance(value, (str, Path)):
        raise SchemaConfigError(f"config.paths.{key} must be a path string")
    return Path(value)


def _resolve_output_root(
    *,
    config_path: Path,
    configured: Path,
    override: Path | None,
) -> Path:
    if override is not None:
        value = Path(override)
        return value.resolve()

    if configured.is_absolute():
        return configured.resolve()
    config_directory = next(
        (
            parent
            for parent in config_path.parents
            if parent.name == "configs"
        ),
        None,
    )
    project_root = (
        config_directory.parent
        if config_directory is not None
        else config_path.parent
    )
    return (project_root / configured).resolve()


def _override(override: Any, configured: Any) -> Any:
    return configured if override is None else override


__all__ = [
    "SchemaConfigError",
    "SchemaConfigOverrides",
    "InstanceConfigOverrides",
    "TaskConfigOverrides",
    "RDBPFNExportConfigOverrides",
    "RouterTrainingConfigOverrides",
    "RoutedH5ConfigOverrides",
    "load_schema_pipeline_config",
    "load_instance_pipeline_config",
    "load_task_pipeline_config",
    "load_rdbpfn_export_config",
    "load_router_training_config",
    "load_routed_h5_config",
]
