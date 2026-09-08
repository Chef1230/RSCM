from __future__ import annotations

from dataclasses import replace
import json
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
import warnings

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.artifacts import load_schema_artifact
from rdb_prior.compilation.compiler import PhysicalSchemaCompiler
from rdb_prior.export.converter import (
    RDBPFNConverter,
    _seconds_to_datetime,
    assert_export_has_no_private_metadata,
)
from rdb_prior.export.pipeline import RDBPFNExportConfig, export_rdbpfn_tasks
from rdb_prior.export.validation import validate_rdbpfn_dataset
from rdb_prior.generation.database import DatabaseGenerator
from rdb_prior.instance.planner import InstancePlanner, InstancePlannerConfig
from rdb_prior.pipeline import (
    InstancePipelineConfig,
    SchemaPipelineConfig,
    generate_database_instances,
    generate_physical_schemas,
)
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.task.artifacts import TaskArtifact, load_task_artifact
from rdb_prior.task.model import TaskMechanism
from rdb_prior.task.pipeline import TaskPipelineConfig, generate_tasks
from rdb_prior.task.planner import TaskPlanner, TaskPlannerConfig
from rdb_prior.task.view import build_task_view


class RDBPFNExportTests(unittest.TestCase):
    def test_conversion_filters_supervision_rows_hidden_by_task_view(self) -> None:
        for attempt in range(20):
            sample_id = f"vis_filter_{attempt}"
            runtime = RuntimeContext(42 + attempt * 101).for_sample(sample_id)
            blueprint = BlueprintSampler(
                BlueprintSamplerConfig(min_tables=6, max_tables=9)
            ).sample(sample_id, runtime)
            schema = PhysicalSchemaCompiler().compile(
                blueprint, sample_id, runtime,
            )
            instance_plan = InstancePlanner(
                InstancePlannerConfig(
                    entity_rows_min=32,
                    entity_rows_max=40,
                    lookup_rows_min=3,
                    lookup_rows_max=5,
                    max_rows_per_table=128,
                )
            ).plan(
                sample_id=sample_id,
                schema=schema,
                runtime=runtime.child("database-instance"),
            )
            database = DatabaseGenerator().generate(
                schema=schema,
                plan=instance_plan,
            )
            try:
                tasks = TaskPlanner(
                    TaskPlannerConfig(
                        tasks_per_database=2,
                        # Prefer temporal mechanisms whose observation_rules
                        # filter Event / Detail rows by cutoff.
                        mechanism_weights=(
                            (TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE, 0.4),
                            (TaskMechanism.TEMPORAL_RELATIONAL_AGGREGATE, 0.4),
                            (TaskMechanism.FUTURE_EVENT_ATTRIBUTE_CONDITION, 0.2),
                        ),
                        min_support_rows=8,
                        min_query_rows=4,
                        min_class_count_per_split=1,
                        max_attempts_per_database=256,
                    )
                ).generate(
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    runtime=runtime.child("task"),
                )
            except ValueError:
                continue
            # Find a task whose observation_rules actually hide rows on at
            # least one filtered table (Event or Detail with a TIME column).
            try:
                task = next(
                    item
                    for item in tasks
                    if any(
                        len(database.table(rule.table_id).column(rule.time_column_id))
                        > len(
                            build_task_view(
                                schema, database, item.plan,
                            ).visible_rows(rule.table_id)
                        )
                        for rule in item.plan.observation_rules
                    )
                )
            except StopIteration:
                continue
            artifact = TaskArtifact(
                sample_id=sample_id,
                instance_artifact="unused",
                schema_artifact="unused",
                runtime=runtime.record(
                    project_version="test",
                    config_digest="test",
                ),
                task=task,
                validation=None,  # type: ignore[arg-type]
            )
            dataset = RDBPFNConverter(min_validation_rows=2).convert(
                task_artifact=artifact,
                schema=schema,
                database=database,
            )
            self.assertTrue(validate_rdbpfn_dataset(dataset).is_valid)
            assert_export_has_no_private_metadata(dataset)
            with self.assertRaisesRegex(ValueError, "generator-private"):
                assert_export_has_no_private_metadata(
                    replace(
                        dataset,
                        metadata={
                            **dataset.metadata,
                            "prior_provenance": {"private": True},
                        },
                    )
                )
            # Verify that observation_rules actually filtered rows from at
            # least one Event/Detail table in the exported database.
            filtered_any = False
            for rule in task.plan.observation_rules:
                table_name = schema.table(rule.table_id).name
                if table_name in dataset.tables:
                    pk_col = schema.table(rule.table_id).primary_key.name
                    exported_count = len(dataset.tables[table_name][pk_col])
                    db_count = database.table(rule.table_id).row_count
                    if exported_count < db_count:
                        filtered_any = True
                        break
            self.assertTrue(
                filtered_any,
                "observation_rules should filter at least one table in the export",
            )
            return
        self.fail("no task with observation-filtered rows found in 20 attempts")

    def test_pipeline_writes_loadable_dbb_dataset_without_target_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            schema_result = generate_physical_schemas(
                SchemaPipelineConfig(
                    output_root=root / "schema",
                    num_schemas=1,
                    base_seed=91,
                    sampler=BlueprintSamplerConfig(min_tables=4, max_tables=4),
                )
            )
            instance_result = generate_database_instances(
                InstancePipelineConfig(
                    schema_manifest=schema_result.manifest_path,
                    output_root=root / "instance",
                    planner=InstancePlannerConfig(
                        entity_rows_min=24,
                        entity_rows_max=32,
                        lookup_rows_min=3,
                        lookup_rows_max=5,
                        max_rows_per_table=96,
                    ),
                )
            )
            task_result = generate_tasks(
                TaskPipelineConfig(
                    instance_manifest=instance_result.manifest_path,
                    output_root=root / "task",
                    planner=TaskPlannerConfig(
                        tasks_per_database=1,
                        mechanism_weights=(
                            (TaskMechanism.RELATION_ATTRIBUTE, 1.0),
                        ),
                        min_support_rows=8,
                        min_query_rows=4,
                    ),
                )
            )
            result = export_rdbpfn_tasks(
                RDBPFNExportConfig(
                    task_manifest=task_result.manifest_path,
                    output_root=root / "rdbpfn",
                    min_validation_rows=2,
                )
            )

            self.assertEqual(1, result.dataset_count)
            dataset_path = result.dataset_paths[0]
            metadata = yaml.safe_load(
                (dataset_path / "metadata.yaml").read_text(encoding="utf-8")
            )
            task_artifact = load_task_artifact(task_result.artifact_paths[0])
            schema = load_schema_artifact(task_artifact.schema_artifact).compilation.schema
            plan = task_artifact.task.plan
            target_table = schema.table(plan.target_table_id)
            target_column = target_table.column(plan.target_column_id or "")
            table_meta = {
                table["name"]: table for table in metadata["tables"]
            }

            self.assertEqual(plan.task_id, metadata["dataset_name"])
            self.assertEqual("label", metadata["tasks"][0]["target_column"])
            self.assertNotIn(
                target_column.name,
                {item["name"] for item in table_meta[target_table.name]["columns"]},
            )
            with np.load(
                dataset_path / "data" / f"{target_table.name}.npz",
                allow_pickle=True,
            ) as archive:
                self.assertNotIn(target_column.name, archive.files)
            for split in ("train", "validation", "test"):
                self.assertTrue(
                    (dataset_path / plan.task_id / f"{split}.npz").is_file()
                )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("rdbpfn_export_manifest", manifest["artifact_type"])
            self.assertEqual(1, manifest["dataset_count"])
            self._assert_local_dbb_loader_accepts(dataset_path, plan.task_id)

    def test_future_event_conversion_applies_common_cutoff(self) -> None:
        converter = RDBPFNConverter(min_validation_rows=2)
        for index in range(50):
            sample_id = f"export_future_{index}"
            runtime = RuntimeContext(404).for_sample(sample_id)
            blueprint = BlueprintSampler(
                BlueprintSamplerConfig(min_tables=5, max_tables=7)
            ).sample(sample_id, runtime)
            schema = PhysicalSchemaCompiler().compile(
                blueprint,
                sample_id,
                runtime,
            )
            instance_plan = InstancePlanner(
                InstancePlannerConfig(
                    entity_rows_min=24,
                    entity_rows_max=32,
                    lookup_rows_min=3,
                    lookup_rows_max=5,
                    max_rows_per_table=96,
                )
            ).plan(
                sample_id=sample_id,
                schema=schema,
                runtime=runtime.child("database-instance"),
            )
            database = DatabaseGenerator().generate(
                schema=schema,
                plan=instance_plan,
            )
            planner = TaskPlanner(
                TaskPlannerConfig(
                    tasks_per_database=1,
                    mechanism_weights=(
                        (TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE, 1.0),
                    ),
                    min_support_rows=8,
                    min_query_rows=4,
                    min_class_count_per_split=1,
                    max_attempts_per_database=256,
                )
            )
            try:
                task = planner.generate(
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    runtime=runtime.child("task"),
                )[0]
            except ValueError:
                continue
            artifact = TaskArtifact(
                sample_id=sample_id,
                instance_artifact="unused",
                schema_artifact="unused",
                runtime=runtime.record(
                    project_version="test",
                    config_digest="test",
                ),
                task=task,
                validation=None,  # type: ignore[arg-type]
            )
            dataset = converter.convert(
                task_artifact=artifact,
                schema=schema,
                database=database,
            )

            self.assertTrue(validate_rdbpfn_dataset(dataset).is_valid)
            cutoff = np.datetime64(int(task.plan.cutoff_time), "s").astype(
                "datetime64[ns]"
            )
            for split in dataset.splits.values():
                np.testing.assert_array_equal(
                    split["cutoff_time"],
                    np.full(len(split["cutoff_time"]), cutoff),
                )
            for rule in task.plan.observation_rules:
                table = schema.table(rule.table_id)
                time_column = table.column(rule.time_column_id)
                values = dataset.tables[table.name][time_column.name]
                self.assertTrue(np.all(values <= cutoff))
            return
        self.fail("no balanced future-event task found in 30 databases")

    def _assert_local_dbb_loader_accepts(
        self,
        dataset_path: Path,
        task_name: str,
    ) -> None:
        data_preprocessing = PROJECT_ROOT.parent / "RDBPFN" / "data_preprocessing"
        metadata_module_path = data_preprocessing / "dbinfer_bench" / "dataset_meta.py"
        if not metadata_module_path.is_file():
            self.skipTest("local RDBPFN checkout is not available")
        spec = importlib.util.spec_from_file_location(
            "rdbpfn_dataset_meta_compat",
            metadata_module_path,
        )
        if spec is None or spec.loader is None:
            self.fail("cannot load local RDBPFN metadata model")
        module = importlib.util.module_from_spec(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec.loader.exec_module(module)
        metadata = yaml.safe_load(
            (dataset_path / "metadata.yaml").read_text(encoding="utf-8")
        )
        model = module.DBBRDBDatasetMeta
        parsed = (
            model.model_validate(metadata)
            if hasattr(model, "model_validate")
            else model.parse_obj(metadata)
        )
        self.assertEqual(dataset_path.name, parsed.dataset_name)
        self.assertEqual(1, len(parsed.tasks))
        self.assertEqual(task_name, parsed.tasks[0].name)

        boto3_stubbed = importlib.util.find_spec("boto3") is None
        if boto3_stubbed:
            sys.modules["boto3"] = types.ModuleType("boto3")
        sys.path.insert(0, str(data_preprocessing))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dbb = importlib.import_module("dbinfer_bench")
                dataset = dbb.load_rdb_data(str(dataset_path))
        finally:
            sys.path.remove(str(data_preprocessing))
            if boto3_stubbed:
                sys.modules.pop("boto3", None)
        self.assertEqual(dataset_path.name, dataset.dataset_name)
        self.assertEqual(1, len(dataset.tasks))
        self.assertEqual(task_name, dataset.tasks[0].metadata.name)

    def test_seconds_to_datetime_rejects_out_of_ns_range(self) -> None:
        from rdb_prior.time_bounds import NS64_MAX_SECONDS

        out_of_range = np.asarray(
            [1_577_836_800, int(NS64_MAX_SECONDS) + 1_000],
            dtype=np.int64,
        )
        with self.assertRaises(ValueError):
            _seconds_to_datetime(out_of_range)

    def test_seconds_to_datetime_maps_negative_to_nat(self) -> None:
        result = _seconds_to_datetime(np.asarray([-1, np.nan, 1_577_836_800]))
        self.assertTrue(np.isnat(result[0]))
        self.assertTrue(np.isnat(result[1]))
        self.assertEqual(
            result[2],
            np.datetime64(1_577_836_800, "s").astype("datetime64[ns]"),
        )

    def test_seconds_to_datetime_handles_ns_bounds(self) -> None:
        from rdb_prior.time_bounds import NS64_MAX_SECONDS

        result = _seconds_to_datetime(np.asarray([NS64_MAX_SECONDS, 0]))
        self.assertFalse(np.isnat(result[0]))
        self.assertFalse(np.isnat(result[1]))
        self.assertEqual(
            result[0],
            np.datetime64(NS64_MAX_SECONDS, "s").astype("datetime64[ns]"),
        )


if __name__ == "__main__":
    unittest.main()
