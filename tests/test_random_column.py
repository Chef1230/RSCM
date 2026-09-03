from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.compilation.compiler import PhysicalSchemaCompiler
from rdb_prior.export.converter import RDBPFNConverter
from rdb_prior.export.validation import validate_rdbpfn_dataset
from rdb_prior.generation.database import DatabaseGenerator
from rdb_prior.instance.planner import InstancePlanner, InstancePlannerConfig
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.task.mechanisms import (
    mechanism_labels,
    random_column_candidates,
)
from rdb_prior.task.artifacts import TaskArtifact
from rdb_prior.task.model import TaskMechanism, TaskPlan
from rdb_prior.task.planner import TaskPlanner, TaskPlannerConfig
from rdb_prior.task.validation import validate_task
from rdb_prior.task.view import build_task_view


class RandomColumnTaskTests(unittest.TestCase):
    def _database(self, sample_id: str):
        runtime = RuntimeContext(303).for_sample(sample_id)
        blueprint = BlueprintSampler(
            BlueprintSamplerConfig(min_tables=4, max_tables=4)
        ).sample(sample_id, runtime)
        schema = PhysicalSchemaCompiler().compile(blueprint, sample_id, runtime)
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
        return runtime, schema, DatabaseGenerator().generate(
            schema=schema, plan=instance_plan
        )

    def test_random_column_task_matches_rdbpfn_threshold_semantics(self) -> None:
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=1,
                mechanism_weights=((TaskMechanism.RANDOM_COLUMN, 1.0),),
                min_support_rows=8,
                min_query_rows=4,
                min_class_count_per_split=1,
                max_attempts_per_database=512,
            )
        )

        for index in range(40):
            sample_id = f"random_column_{index}"
            runtime, schema, database = self._database(sample_id)
            candidates = random_column_candidates(schema, database)
            if not candidates:
                continue
            try:
                tasks = planner.generate(
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    runtime=runtime.child("task"),
                )
            except ValueError:
                continue
            if not tasks:
                continue

            task = tasks[0]
            plan = task.plan
            self.assertIs(TaskMechanism.RANDOM_COLUMN, plan.mechanism)
            self.assertEqual(plan.target_table_id, plan.source_table_id)
            self.assertEqual((), plan.route_supervision)
            self.assertIn(plan.target_column_id, plan.masked_column_ids)
            self.assertIsNotNone(plan.threshold)
            candidate = next(
                candidate
                for candidate in candidates
                if candidate.table_id == plan.target_table_id
            )
            self.assertIn(
                plan.target_column_id,
                candidate.feature_column_ids,
            )

            expected = mechanism_labels(schema, database, plan)
            np.testing.assert_array_equal(
                task.data.support_labels,
                expected[task.data.support_row_ids],
            )
            np.testing.assert_array_equal(
                task.data.query_labels,
                expected[task.data.query_row_ids],
            )
            np.testing.assert_array_equal(
                mechanism_labels(
                    schema, database, TaskPlan.from_dict(plan.to_dict())
                ),
                expected,
            )
            self.assertEqual({0, 1}, set(expected.tolist()))
            validation = validate_task(schema, database, task)
            self.assertTrue(validation.is_valid, validation.to_dict())
            view = build_task_view(schema, database, plan)
            self.assertTrue(
                view.is_column_masked(
                    plan.target_table_id, plan.target_column_id or ""
                )
            )
            return

        self.fail("no valid random-column task found in 40 databases")

    def test_random_column_task_converts_to_dbb_without_target_leakage(self) -> None:
        for index in range(20):
            sample_id = f"random_column_export_{index}"
            runtime, schema, database = self._database(sample_id)
            planner = TaskPlanner(
                TaskPlannerConfig(
                    tasks_per_database=1,
                    mechanism_weights=((TaskMechanism.RANDOM_COLUMN, 1.0),),
                    min_support_rows=8,
                    min_query_rows=4,
                    min_class_count_per_split=1,
                    max_attempts_per_database=512,
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
                runtime=runtime.record(project_version="test"),
                task=task,
                validation=None,  # type: ignore[arg-type]
            )
            dataset = RDBPFNConverter(min_validation_rows=2).convert(
                task_artifact=artifact,
                schema=schema,
                database=database,
            )
            report = validate_rdbpfn_dataset(dataset)
            self.assertTrue(report.is_valid, report)
            target_table = schema.table(task.plan.target_table_id)
            target_column = target_table.column(task.plan.target_column_id or "")
            self.assertNotIn(
                target_column.name,
                dataset.tables[target_table.name],
            )
            self.assertEqual(
                TaskMechanism.RANDOM_COLUMN.value,
                dataset.metadata["tasks"][0]["mechanism"],
            )
            return

        self.fail("no valid random-column DBB dataset found in 20 databases")


if __name__ == "__main__":
    unittest.main()
