from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.artifacts import load_instance_artifact, load_schema_artifact
from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.instance.planner import InstancePlannerConfig
from rdb_prior.pipeline import (
    InstancePipelineConfig,
    SchemaPipelineConfig,
    generate_database_instances,
    generate_physical_schemas,
)
from rdb_prior.schema.sampler import BlueprintSamplerConfig


class SchemaPipelineTests(unittest.TestCase):
    def test_end_to_end_pipeline_writes_reproducible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "schemas"
            config = SchemaPipelineConfig(
                output_root=output_root,
                num_schemas=3,
                base_seed=17,
                progress_every=1,
                sampler=BlueprintSamplerConfig(
                    min_tables=3,
                    max_tables=5,
                ),
            )

            result = generate_physical_schemas(config)

            self.assertEqual(3, result.generated_count)
            self.assertEqual(3, len(result.dot_paths))
            self.assertEqual((), result.image_paths)
            self.assertTrue(result.manifest_path.is_file())
            manifest = json.loads(
                result.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(3, len(manifest["entries"]))
            for entry in manifest["entries"]:
                graph_path = output_root / entry["graph_artifacts"]["dot"]
                self.assertTrue(graph_path.is_file())

            for artifact_path in result.artifact_paths:
                payload = json.loads(
                    artifact_path.read_text(encoding="utf-8")
                )
                self.assertEqual("physical_schema", payload["artifact_type"])
                self.assertEqual(3, payload["artifact_version"])
                self.assertTrue(payload["validation"]["is_valid"])
                self.assertNotIn("motifs", payload["blueprint"])
                self.assertIn(
                    "motif_occurrences",
                    payload["blueprint"],
                )
                self.assertIn("compilation_trace", payload)
                schema = PhysicalSchema.from_dict(payload["physical_schema"])
                artifact = load_schema_artifact(artifact_path)
                self.assertEqual(schema, artifact.compilation.schema)
                self.assertEqual(
                    artifact.blueprint.blueprint_id,
                    artifact.compilation.trace.blueprint_id,
                )

    def test_existing_artifact_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = SchemaPipelineConfig(
                output_root=Path(temporary_directory),
                num_schemas=1,
            )
            generate_physical_schemas(config)

            with self.assertRaises(FileExistsError):
                generate_physical_schemas(config)

    def test_schema_to_instance_pipeline_writes_reloadable_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            schema_result = generate_physical_schemas(
                SchemaPipelineConfig(
                    output_root=root / "schema",
                    num_schemas=2,
                    base_seed=33,
                    sampler=BlueprintSamplerConfig(min_tables=3, max_tables=4),
                )
            )
            result = generate_database_instances(
                InstancePipelineConfig(
                    schema_manifest=schema_result.manifest_path,
                    output_root=root / "instance",
                    num_workers=2,
                    progress_every=1,
                    planner=InstancePlannerConfig(
                        entity_rows_min=16,
                        entity_rows_max=20,
                        lookup_rows_min=3,
                        lookup_rows_max=5,
                        max_rows_per_table=48,
                    ),
                )
            )

            self.assertEqual(2, result.generated_count)
            schema_manifest = json.loads(
                schema_result.manifest_path.read_text(encoding="utf-8")
            )
            instance_manifest = json.loads(
                result.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                [entry["sample_id"] for entry in schema_manifest["entries"]],
                [entry["sample_id"] for entry in instance_manifest["entries"]],
            )
            for artifact_path in result.artifact_paths:
                artifact = load_instance_artifact(artifact_path)
                self.assertTrue(artifact.validation.is_valid)
                self.assertEqual(
                    artifact.plan.schema_id,
                    artifact.database.schema_id,
                )
                self.assertTrue(artifact.database.tables)

            serial_result = generate_database_instances(
                InstancePipelineConfig(
                    schema_manifest=schema_result.manifest_path,
                    output_root=root / "instance_serial",
                    num_workers=1,
                    planner=InstancePlannerConfig(
                        entity_rows_min=16,
                        entity_rows_max=20,
                        lookup_rows_min=3,
                        lookup_rows_max=5,
                        max_rows_per_table=48,
                    ),
                )
            )
            for parallel_path, serial_path in zip(
                result.artifact_paths,
                serial_result.artifact_paths,
                strict=True,
            ):
                parallel = load_instance_artifact(parallel_path)
                serial = load_instance_artifact(serial_path)
                self.assertEqual(parallel.plan.to_dict(), serial.plan.to_dict())
                for parallel_table, serial_table in zip(
                    parallel.database.tables,
                    serial.database.tables,
                    strict=True,
                ):
                    self.assertEqual(
                        set(parallel_table.columns),
                        set(serial_table.columns),
                    )
                    for column_id in parallel_table.columns:
                        np.testing.assert_array_equal(
                            parallel_table.columns[column_id],
                            serial_table.columns[column_id],
                        )

    def test_instance_shards_share_output_without_manifest_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            schema_result = generate_physical_schemas(
                SchemaPipelineConfig(
                    output_root=root / "schema",
                    num_schemas=4,
                    sampler=BlueprintSamplerConfig(min_tables=3, max_tables=3),
                )
            )
            planner = InstancePlannerConfig(
                entity_rows_min=8,
                entity_rows_max=10,
                lookup_rows_min=2,
                lookup_rows_max=3,
                max_rows_per_table=20,
            )
            results = [
                generate_database_instances(
                    InstancePipelineConfig(
                        schema_manifest=schema_result.manifest_path,
                        output_root=root / "instance",
                        shard_id=shard_id,
                        num_shards=2,
                        planner=planner,
                    )
                )
                for shard_id in range(2)
            ]

            self.assertEqual(4, sum(result.generated_count for result in results))
            self.assertNotEqual(results[0].manifest_path, results[1].manifest_path)
            self.assertTrue(all(result.manifest_path.is_file() for result in results))


if __name__ == "__main__":
    unittest.main()
