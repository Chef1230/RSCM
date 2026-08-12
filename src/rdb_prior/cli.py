# src/rdb_prior/cli.py
# -*- coding: utf-8 -*-
"""Command-line entry points for staged synthetic-data generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rdb_prior.config import (
    InstanceConfigOverrides,
    RDBPFNExportConfigOverrides,
    RoutedH5ConfigOverrides,
    RouterTrainingConfigOverrides,
    SchemaConfigError,
    SchemaConfigOverrides,
    TaskConfigOverrides,
    load_instance_pipeline_config,
    load_rdbpfn_export_config,
    load_routed_h5_config,
    load_router_training_config,
    load_schema_pipeline_config,
    load_task_pipeline_config,
)
from rdb_prior.export.pipeline import export_rdbpfn_tasks
from rdb_prior.observability import (
    ProgressReporter,
    close_logging,
    configure_logging,
)
from rdb_prior.pipeline import (
    generate_database_instances,
    generate_physical_schemas,
    generate_tasks,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rdb-prior")
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema = subparsers.add_parser(
        "schema",
        help="sample validated SchemaBlueprint and PhysicalSchema artifacts",
    )
    schema.add_argument(
        "--config",
        type=Path,
        default=Path("configs/refactor_v1.yaml"),
        help="YAML/JSON schema-pipeline config",
    )
    schema.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override the derived schema stage output directory",
    )
    schema.add_argument(
        "--count",
        type=int,
        default=None,
        help="override generation.num_schemas",
    )
    schema.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override top-level seed",
    )
    schema.add_argument("--start-index", type=int, default=None)
    schema.add_argument("--sample-id-prefix", default=None)
    schema.add_argument("--min-tables", type=int, default=None)
    schema.add_argument("--max-tables", type=int, default=None)
    schema.add_argument("--max-rank", type=int, default=None)
    schema.add_argument("--max-extra-edges", type=int, default=None)
    schema.add_argument(
        "--extra-edge-probability",
        type=float,
        default=None,
    )
    schema.add_argument("--min-feature-columns", type=int, default=None)
    schema.add_argument("--max-feature-columns", type=int, default=None)
    schema.add_argument(
        "--feature-nullable-probability",
        type=float,
        default=None,
    )
    schema.add_argument("--blueprint-id-prefix", default=None)
    schema.add_argument("--schema-id-prefix", default=None)
    schema.add_argument(
        "--schema-dot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="write one Graphviz DOT file per generated schema",
    )
    schema.add_argument(
        "--schema-graph-format",
        choices=("none", "png", "svg", "pdf"),
        default=None,
        help="optionally render DOT files with Graphviz",
    )
    schema.add_argument(
        "--graphviz-command",
        default=None,
        help="Graphviz dot executable name or path",
    )
    schema.add_argument("--progress-every", type=int, default=None)
    schema.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override generation.overwrite",
    )
    schema.add_argument(
        "--validate-config-only",
        action="store_true",
        help="validate and print resolved config without generating files",
    )
    _add_observability_arguments(schema)

    instance = subparsers.add_parser(
        "instance",
        help="materialize database instances from a physical schema manifest",
    )
    instance.add_argument(
        "--config",
        type=Path,
        default=Path("configs/refactor_v1.yaml"),
    )
    instance.add_argument("--schema-manifest", type=Path, default=None)
    instance.add_argument("--output-dir", type=Path, default=None)
    instance.add_argument("--count", type=int, default=None)
    instance.add_argument("--start-index", type=int, default=None)
    instance.add_argument("--shard-id", type=int, default=None)
    instance.add_argument("--num-shards", type=int, default=None)
    instance.add_argument(
        "--jobs",
        dest="num_workers",
        type=int,
        default=None,
        help="override instance_generation.num_workers",
    )
    instance.add_argument("--progress-every", type=int, default=None)
    instance.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    instance.add_argument("--validate-config-only", action="store_true")
    _add_observability_arguments(instance)

    task = subparsers.add_parser(
        "task",
        help="derive supervised relational tasks from an instance manifest",
    )
    task.add_argument(
        "--config",
        type=Path,
        default=Path("configs/refactor_v1.yaml"),
    )
    task.add_argument("--instance-manifest", type=Path, default=None)
    task.add_argument("--output-dir", type=Path, default=None)
    task.add_argument(
        "--database-count",
        "--count",
        dest="database_count",
        type=int,
        default=None,
    )
    task.add_argument("--tasks-per-database", type=int, default=None)
    task.add_argument("--start-index", type=int, default=None)
    task.add_argument("--shard-id", type=int, default=None)
    task.add_argument("--num-shards", type=int, default=None)
    task.add_argument(
        "--jobs",
        "--workers",
        dest="num_workers",
        type=int,
        default=None,
        help="override task_generation.num_workers",
    )
    task.add_argument("--progress-every", type=int, default=None)
    task.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    task.add_argument("--validate-config-only", action="store_true")
    _add_observability_arguments(task)

    export = subparsers.add_parser(
        "rdbpfn-export",
        help="export task artifacts as RDBPFN dbinfer_bench datasets",
    )
    export.add_argument(
        "--config",
        type=Path,
        default=Path("configs/refactor_v1.yaml"),
    )
    export.add_argument("--task-manifest", type=Path, default=None)
    export.add_argument("--output-dir", type=Path, default=None)
    export.add_argument("--count", dest="task_count", type=int, default=None)
    export.add_argument("--start-index", type=int, default=None)
    export.add_argument("--shard-id", type=int, default=None)
    export.add_argument("--num-shards", type=int, default=None)
    export.add_argument("--validation-fraction", type=float, default=None)
    export.add_argument("--min-validation-rows", type=int, default=None)
    export.add_argument(
        "--compress",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    export.add_argument("--progress-every", type=int, default=None)
    export.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    export.add_argument(
        "--h5",
        dest="h5_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run optional DFS/H5 packaging after DBB export",
    )
    export.add_argument("--h5-output", type=Path, default=None)
    export.add_argument(
        "--rdbpfn-preprocessing-root",
        type=Path,
        default=None,
    )
    export.add_argument(
        "--h5-run-dfs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    export.add_argument("--dfs-depth", type=int, choices=(1, 2), default=None)
    export.add_argument("--dfs-jobs", type=int, default=None)
    export.add_argument("--h5-total-rows", type=int, default=None)
    export.add_argument("--h5-max-columns", type=int, default=None)
    export.add_argument("--h5-seed", type=int, default=None)
    export.add_argument("--validate-config-only", action="store_true")
    _add_observability_arguments(export)

    router = subparsers.add_parser(
        "router-train",
        help="train the supervised sparse MLP path router and PFN backend",
    )
    router.add_argument(
        "--config", type=Path, default=Path("configs/refactor_v2.yaml")
    )
    router.add_argument("--task-manifest", type=Path, default=None)
    router.add_argument("--output-dir", type=Path, default=None)
    router.add_argument("--count", dest="task_count", type=int, default=None)
    router.add_argument("--start-index", type=int, default=None)
    router.add_argument("--epochs", type=int, default=None)
    router.add_argument("--device", default=None)
    router.add_argument("--batch-size", type=int, default=None)
    router.add_argument("--num-workers", type=int, default=None)
    router.add_argument("--prefetch-factor", type=int, default=None)
    router.add_argument(
        "--mixed-precision",
        choices=("none", "fp16", "bf16"),
        default=None,
    )
    router.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=None
    )
    router.add_argument("--validate-config-only", action="store_true")
    _add_observability_arguments(router)

    routed_h5 = subparsers.add_parser(
        "routed-h5",
        help="export selected relation-column tokens from a router checkpoint",
    )
    routed_h5.add_argument(
        "--config", type=Path, default=Path("configs/refactor_v2.yaml")
    )
    routed_h5.add_argument("--task-manifest", type=Path, default=None)
    routed_h5.add_argument("--checkpoint", type=Path, default=None)
    routed_h5.add_argument("--output", dest="output_path", type=Path, default=None)
    routed_h5.add_argument("--count", dest="task_count", type=int, default=None)
    routed_h5.add_argument("--start-index", type=int, default=None)
    routed_h5.add_argument("--device", default=None)
    routed_h5.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=None
    )
    routed_h5.add_argument("--validate-config-only", action="store_true")
    _add_observability_arguments(routed_h5)

    router_eval = subparsers.add_parser(
        "router-eval",
        help="evaluate a router checkpoint on real or synthetic task artifacts",
    )
    router_eval.add_argument("--task-manifest", type=Path, required=True)
    router_eval.add_argument("--checkpoint", type=Path, required=True)
    router_eval.add_argument("--output-dir", type=Path, required=True)
    router_eval.add_argument("--count", dest="task_count", type=int, default=None)
    router_eval.add_argument("--start-index", type=int, default=0)
    router_eval.add_argument("--device", default="auto")
    router_eval.add_argument(
        "--mixed-precision",
        choices=("none", "fp16", "bf16"),
        default="none",
    )
    router_eval.add_argument("--artifact-cache-size", type=int, default=16)
    router_eval.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    _add_observability_arguments(router_eval)

    relbench_import = subparsers.add_parser(
        "relbench-import",
        help="convert a RelBench EntityTask into native benchmark artifacts",
    )
    relbench_import.add_argument("--dataset", required=True)
    relbench_import.add_argument("--task", required=True)
    relbench_import.add_argument("--output-dir", type=Path, required=True)
    relbench_import.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="download/verify the registered RelBench dataset and task cache",
    )
    relbench_import.add_argument("--seed", type=int, default=0)
    relbench_import.add_argument("--max-rows-per-task", type=int, default=600)
    relbench_import.add_argument(
        "--query-rows-per-task", type=int, default=256
    )
    relbench_import.add_argument("--support-rows", type=int, default=None)
    relbench_import.add_argument("--max-classes", type=int, default=16)
    relbench_import.add_argument("--max-text-length", type=int, default=256)
    relbench_import.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    _add_observability_arguments(relbench_import)

    relbench_score = subparsers.add_parser(
        "relbench-score",
        help="score complete router predictions with official RelBench metrics",
    )
    relbench_score.add_argument("--metadata", type=Path, required=True)
    relbench_score.add_argument("--predictions", type=Path, required=True)
    relbench_score.add_argument("--output", type=Path, required=True)
    relbench_score.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=False
    )
    relbench_score.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    _add_observability_arguments(relbench_score)
    return parser


def _add_observability_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="write timestamped logs to this file",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="show a terminal progress bar; default is auto-detect",
    )
    parser.add_argument("--progress-width", type=int, default=28)


def _run_schema(args: argparse.Namespace) -> int:
    logger = configure_logging(
        level=args.log_level,
        log_file=args.log_file,
    ).getChild("schema")
    overrides = SchemaConfigOverrides(
        output_root=args.output_dir,
        num_schemas=args.count,
        base_seed=args.seed,
        start_index=args.start_index,
        sample_id_prefix=args.sample_id_prefix,
        progress_every=args.progress_every,
        overwrite=args.overwrite,
        min_tables=args.min_tables,
        max_tables=args.max_tables,
        max_rank=args.max_rank,
        max_extra_edges=args.max_extra_edges,
        extra_edge_probability=args.extra_edge_probability,
        min_feature_columns=args.min_feature_columns,
        max_feature_columns=args.max_feature_columns,
        feature_nullable_probability=(
            args.feature_nullable_probability
        ),
        blueprint_id_prefix=args.blueprint_id_prefix,
        schema_id_prefix=args.schema_id_prefix,
        write_schema_dot=args.schema_dot,
        schema_graph_format=args.schema_graph_format,
        graphviz_command=args.graphviz_command,
    )
    config = load_schema_pipeline_config(
        args.config,
        overrides=overrides,
    )

    if args.validate_config_only:
        logger.info("schema configuration validated: %s", args.config)
        payload = config.to_dict()
        payload["output_root"] = str(config.output_root)
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        close_logging()
        return 0

    reporter = ProgressReporter(
        stage="schema",
        total=config.num_schemas,
        logger=logger,
        log_every=config.progress_every,
        enabled=args.progress,
        width=args.progress_width,
    )
    try:
        result = generate_physical_schemas(
            config,
            progress=lambda completed, total, sample_id: reporter.update(
                completed,
                total,
                sample_id,
            ),
        )
    except Exception:
        logger.exception("schema pipeline failed")
        raise
    finally:
        reporter.close()
        close_logging()
    print(
        json.dumps(
            {
                "generated_count": result.generated_count,
                "dot_count": len(result.dot_paths),
                "image_count": len(result.image_paths),
                "output_root": str(result.output_root),
                "manifest": str(result.manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_instance(args: argparse.Namespace) -> int:
    logger = configure_logging(
        level=args.log_level,
        log_file=args.log_file,
    ).getChild("instance")
    config = load_instance_pipeline_config(
        args.config,
        overrides=InstanceConfigOverrides(
            schema_manifest=args.schema_manifest,
            output_root=args.output_dir,
            count=args.count,
            start_index=args.start_index,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            num_workers=args.num_workers,
            progress_every=args.progress_every,
            overwrite=args.overwrite,
        ),
    )
    if args.validate_config_only:
        logger.info("instance configuration validated: %s", args.config)
        payload = config.to_dict()
        payload["output_root"] = str(config.output_root)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        close_logging()
        return 0

    reporter = ProgressReporter(
        stage="instance",
        logger=logger,
        log_every=config.progress_every,
        enabled=args.progress,
        width=args.progress_width,
    )
    try:
        result = generate_database_instances(
            config,
            progress=lambda completed, total, sample_id: reporter.update(
                completed,
                total,
                sample_id,
            ),
        )
    except Exception:
        logger.exception("instance pipeline failed")
        raise
    finally:
        reporter.close()
        close_logging()
    print(
        json.dumps(
            {
                "generated_count": result.generated_count,
                "num_workers": config.num_workers,
                "output_root": str(result.output_root),
                "manifest": str(result.manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_task(args: argparse.Namespace) -> int:
    logger = configure_logging(
        level=args.log_level,
        log_file=args.log_file,
    ).getChild("task")
    config = load_task_pipeline_config(
        args.config,
        overrides=TaskConfigOverrides(
            instance_manifest=args.instance_manifest,
            output_root=args.output_dir,
            database_count=args.database_count,
            tasks_per_database=args.tasks_per_database,
            start_index=args.start_index,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            num_workers=args.num_workers,
            progress_every=args.progress_every,
            overwrite=args.overwrite,
        ),
    )
    if args.validate_config_only:
        logger.info("task configuration validated: %s", args.config)
        payload = config.to_dict()
        payload["output_root"] = str(config.output_root)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        close_logging()
        return 0

    reporter = ProgressReporter(
        stage="task",
        logger=logger,
        log_every=config.progress_every,
        enabled=args.progress,
        width=args.progress_width,
    )

    def progress(
        completed: int,
        total: int,
        sample_id: str,
        task_count: int,
    ) -> None:
        reporter.update(
            completed,
            total,
            sample_id,
            detail=f"tasks={task_count}",
        )

    try:
        result = generate_tasks(config, progress=progress)
    except Exception:
        logger.exception("task pipeline failed")
        raise
    finally:
        reporter.close()
        close_logging()
    print(
        json.dumps(
            {
                "database_count": result.database_count,
                "task_count": result.task_count,
                "num_workers": config.num_workers,
                "output_root": str(result.output_root),
                "manifest": str(result.manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_rdbpfn_export(args: argparse.Namespace) -> int:
    logger = configure_logging(
        level=args.log_level,
        log_file=args.log_file,
    ).getChild("rdbpfn-export")
    config = load_rdbpfn_export_config(
        args.config,
        overrides=RDBPFNExportConfigOverrides(
            task_manifest=args.task_manifest,
            output_root=args.output_dir,
            task_count=args.task_count,
            start_index=args.start_index,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            validation_fraction=args.validation_fraction,
            min_validation_rows=args.min_validation_rows,
            compress=args.compress,
            progress_every=args.progress_every,
            overwrite=args.overwrite,
            h5_enabled=args.h5_enabled,
            h5_output=args.h5_output,
            rdbpfn_preprocessing_root=args.rdbpfn_preprocessing_root,
            h5_run_dfs=args.h5_run_dfs,
            dfs_depth=args.dfs_depth,
            dfs_jobs=args.dfs_jobs,
            h5_total_rows=args.h5_total_rows,
            h5_max_columns=args.h5_max_columns,
            h5_seed=args.h5_seed,
        ),
    )
    if args.validate_config_only:
        logger.info("RDBPFN export configuration validated: %s", args.config)
        payload = config.to_dict()
        payload["output_root"] = str(config.output_root)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        close_logging()
        return 0

    reporter = ProgressReporter(
        stage="rdbpfn-export",
        logger=logger,
        log_every=config.progress_every,
        enabled=args.progress,
        width=args.progress_width,
    )
    try:
        result = export_rdbpfn_tasks(
            config,
            progress=lambda completed, total, dataset_name: reporter.update(
                completed,
                total,
                dataset_name,
            ),
        )
    except Exception:
        logger.exception("RDBPFN export pipeline failed")
        raise
    finally:
        reporter.close()
        close_logging()
    print(
        json.dumps(
            {
                "dataset_count": result.dataset_count,
                "output_root": str(result.output_root),
                "manifest": str(result.manifest_path),
                "h5_output": (
                    str(result.h5_result.output_path)
                    if result.h5_result is not None
                    else None
                ),
                "h5_task_count": (
                    result.h5_result.task_count
                    if result.h5_result is not None
                    else 0
                ),
                "h5_skipped_task_count": (
                    result.h5_result.skipped_task_count
                    if result.h5_result is not None
                    else 0
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_router_train(args: argparse.Namespace) -> int:
    logger = configure_logging(
        level=args.log_level, log_file=args.log_file
    ).getChild("router-train")
    config = load_router_training_config(
        args.config,
        overrides=RouterTrainingConfigOverrides(
            task_manifest=args.task_manifest,
            output_root=args.output_dir,
            task_count=args.task_count,
            start_index=args.start_index,
            epochs=args.epochs,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            mixed_precision=args.mixed_precision,
            overwrite=args.overwrite,
        ),
    )
    if args.validate_config_only:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        close_logging()
        return 0
    from rdb_prior.routing.trainer import train_sparse_router

    reporter = ProgressReporter(
        stage="router-train",
        logger=logger,
        log_every=config.progress_every,
        enabled=args.progress,
        overwrite=False,
        width=args.progress_width,
    )
    try:
        result = train_sparse_router(
            config,
            progress=lambda completed, total, task_id, detail: reporter.update(
                completed, total, task_id, detail=detail
            ),
        )
    except Exception:
        logger.exception("sparse router training failed")
        raise
    finally:
        reporter.close()
        close_logging()
    print(
        json.dumps(
            {
                "best_checkpoint": str(result.best_checkpoint),
                "last_checkpoint": str(result.last_checkpoint),
                "metrics": str(result.metrics_path),
                "train_task_count": result.train_task_count,
                "validation_task_count": result.validation_task_count,
                "best_validation_loss": result.best_validation_loss,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_routed_h5(args: argparse.Namespace) -> int:
    logger = configure_logging(
        level=args.log_level, log_file=args.log_file
    ).getChild("routed-h5")
    config = load_routed_h5_config(
        args.config,
        overrides=RoutedH5ConfigOverrides(
            task_manifest=args.task_manifest,
            checkpoint=args.checkpoint,
            output_path=args.output_path,
            task_count=args.task_count,
            start_index=args.start_index,
            device=args.device,
            overwrite=args.overwrite,
        ),
    )
    if args.validate_config_only:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        close_logging()
        return 0
    from rdb_prior.export.routed_h5 import export_routed_h5

    reporter = ProgressReporter(
        stage="routed-h5",
        logger=logger,
        log_every=50,
        enabled=args.progress,
        width=args.progress_width,
    )
    try:
        result = export_routed_h5(
            config,
            progress=lambda completed, total, task_id: reporter.update(
                completed, total, task_id
            ),
        )
    except Exception:
        logger.exception("routed H5 export failed")
        raise
    finally:
        reporter.close()
        close_logging()
    print(
        json.dumps(
            {"output": str(result.output_path), "task_count": result.task_count},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_router_eval(args: argparse.Namespace) -> int:
    logger = configure_logging(
        level=args.log_level, log_file=args.log_file
    ).getChild("router-eval")
    from rdb_prior.evaluation.router import evaluate_router_checkpoint
    from rdb_prior.routing.config import RouterEvaluationConfig

    config = RouterEvaluationConfig(
        task_manifest=args.task_manifest.resolve(),
        checkpoint=args.checkpoint.resolve(),
        output_root=args.output_dir.resolve(),
        task_count=args.task_count,
        start_index=args.start_index,
        device=args.device,
        mixed_precision=args.mixed_precision,
        artifact_cache_size=args.artifact_cache_size,
        overwrite=args.overwrite,
    )
    reporter = ProgressReporter(
        stage="router-eval",
        logger=logger,
        log_every=1,
        enabled=args.progress,
        width=args.progress_width,
    )
    try:
        result = evaluate_router_checkpoint(
            config,
            progress=lambda completed, total, task_id: reporter.update(
                completed, total, task_id
            ),
        )
    except Exception:
        logger.exception("router benchmark evaluation failed")
        raise
    finally:
        reporter.close()
        close_logging()
    print(
        json.dumps(
            {
                "metrics": str(result.metrics_path),
                "predictions": str(result.predictions_path),
                "task_count": result.task_count,
                "query_row_count": result.query_row_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_relbench_import(args: argparse.Namespace) -> int:
    logger = configure_logging(
        level=args.log_level, log_file=args.log_file
    ).getChild("relbench-import")
    from rdb_prior.importers.relbench import (
        RelBenchImportConfig,
        import_relbench,
    )

    config = RelBenchImportConfig(
        dataset_name=args.dataset,
        task_name=args.task,
        output_root=args.output_dir.resolve(),
        download=args.download,
        overwrite=args.overwrite,
        seed=args.seed,
        max_rows_per_task=args.max_rows_per_task,
        query_rows_per_task=args.query_rows_per_task,
        support_rows=args.support_rows,
        max_classes=args.max_classes,
        max_text_length=args.max_text_length,
    )
    reporter = ProgressReporter(
        stage="relbench-import",
        logger=logger,
        log_every=1,
        enabled=args.progress,
        width=args.progress_width,
    )
    try:
        result = import_relbench(
            config,
            progress=lambda completed, total, task_id: reporter.update(
                completed, total, task_id
            ),
        )
    except Exception:
        logger.exception("RelBench import failed")
        raise
    finally:
        reporter.close()
        close_logging()
    print(
        json.dumps(
            {
                "task_manifest": str(result.task_manifest),
                "metadata": str(result.metadata_path),
                "task_count": result.task_count,
                "support_row_count": result.support_row_count,
                "query_row_count": result.query_row_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_relbench_score(args: argparse.Namespace) -> int:
    logger = configure_logging(
        level=args.log_level, log_file=args.log_file
    ).getChild("relbench-score")
    from rdb_prior.evaluation.relbench import (
        RelBenchScoreConfig,
        score_relbench_predictions,
    )

    try:
        result = score_relbench_predictions(
            RelBenchScoreConfig(
                metadata_path=args.metadata.resolve(),
                predictions_path=args.predictions.resolve(),
                output_path=args.output.resolve(),
                download=args.download,
                overwrite=args.overwrite,
            )
        )
    except Exception:
        logger.exception("official RelBench scoring failed")
        raise
    finally:
        close_logging()
    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "prediction_count": result.prediction_count,
                "metrics": dict(result.metrics),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "schema":
        try:
            return _run_schema(args)
        except SchemaConfigError as error:
            parser.error(str(error))
    if args.command == "instance":
        try:
            return _run_instance(args)
        except SchemaConfigError as error:
            parser.error(str(error))
    if args.command == "task":
        try:
            return _run_task(args)
        except SchemaConfigError as error:
            parser.error(str(error))
    if args.command == "rdbpfn-export":
        try:
            return _run_rdbpfn_export(args)
        except SchemaConfigError as error:
            parser.error(str(error))
    if args.command == "router-train":
        try:
            return _run_router_train(args)
        except SchemaConfigError as error:
            parser.error(str(error))
    if args.command == "routed-h5":
        try:
            return _run_routed_h5(args)
        except SchemaConfigError as error:
            parser.error(str(error))
    if args.command == "router-eval":
        return _run_router_eval(args)
    if args.command == "relbench-import":
        return _run_relbench_import(args)
    if args.command == "relbench-score":
        return _run_relbench_score(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
