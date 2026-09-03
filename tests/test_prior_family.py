from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.artifacts import (
    InstanceArtifactWriter,
    load_instance_artifact,
    load_schema_artifact,
)
from rdb_prior.compilation.compiler import PhysicalSchemaCompiler
from rdb_prior.generation.database import DatabaseGenerator
from rdb_prior.instance.plan import InstancePlan
from rdb_prior.instance.planner import InstancePlanner, InstancePlannerConfig
from rdb_prior.pipeline import (
    InstancePipelineConfig,
    SchemaPipelineConfig,
    generate_database_instances,
    generate_physical_schemas,
)
from rdb_prior.priors.model import PriorFamily, TaskPolicyPlan
from rdb_prior.priors.planner import PriorPlanner, PriorPlannerConfig
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.domain_prototypes import sample_semantic_schema
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.task.program import TaskExecutor, TaskProgramPlanner
from rdb_prior.task.artifacts import load_task_artifact
from rdb_prior.task.pipeline import TaskPipelineConfig, generate_tasks
from rdb_prior.validation.checks import validate_database_instance, validate_instance_plan


class PriorFamilyTests(unittest.TestCase):
    def _temporal_fixture(self):
        runtime = RuntimeContext(412).for_sample("temporal_prior")
        blueprint = BlueprintSampler(
            BlueprintSamplerConfig(
                min_tables=3,
                max_tables=3,
                min_motif_occurrences=1,
                max_motif_occurrences=1,
                max_extra_edges=0,
                background_attachment_probability=0.0,
                motif_weights=(("entity_event", 1.0),),
            )
        ).sample("temporal_prior", runtime)
        schema = PhysicalSchemaCompiler().compile(
            blueprint,
            "temporal_prior",
            runtime,
        )
        semantic = sample_semantic_schema(schema, runtime.child("semantic"))
        policy = TaskPolicyPlan(
            programs_per_database=1,
            cutoff_fraction_min=0.25,
            cutoff_fraction_max=0.25,
            horizon_fraction_min=0.60,
            horizon_fraction_max=0.60,
            positive_rate_min=0.01,
            positive_rate_max=0.99,
        )
        prior = PriorPlanner(
            PriorPlannerConfig(
                database_family_weights=((PriorFamily.TEMPORAL_EVENT, 1.0),),
                task_policy=policy,
                state_dimension=3,
            )
        ).plan(
            blueprint=blueprint,
            physical_schema=schema,
            semantic_schema=semantic,
            runtime=runtime.child("prior"),
        )
        plan = InstancePlanner(
            InstancePlannerConfig(
                entity_rows_min=48,
                entity_rows_max=48,
                lookup_rows_min=4,
                lookup_rows_max=4,
                max_rows_per_table=256,
            )
        ).plan(
            sample_id="temporal_prior",
            schema=schema,
            runtime=runtime.child("instance"),
            prior_plan=prior,
        )
        materialization = DatabaseGenerator().materialize(schema=schema, plan=plan)
        return runtime, schema, prior, materialization.plan, materialization.database

    def test_temporal_entity_event_is_jointly_planned_and_valid(self) -> None:
        runtime, schema, prior, plan, database = self._temporal_fixture()

        self.assertIs(PriorFamily.TEMPORAL_EVENT, prior.family)
        self.assertEqual(prior, type(prior).from_dict(prior.to_dict()))
        self.assertEqual(plan, InstancePlan.from_dict(plan.to_dict()))
        self.assertEqual("temporal_event", plan.prior_family)
        self.assertTrue(plan.shared_state_ids)
        self.assertTrue(plan.population_mechanisms)
        self.assertTrue(plan.temporal_processes)
        self.assertTrue(plan.column_mechanisms)
        self.assertTrue(validate_instance_plan(schema, plan).is_valid)
        self.assertTrue(validate_database_instance(schema, plan, database).is_valid)

        population = plan.population_mechanisms[0]
        event = database.table(population.table_id)
        entity = database.table(population.parent_table_id)
        foreign_key_id = dict(population.parameters)["foreign_key_id"]
        foreign_key = next(item for item in schema.foreign_keys if item.foreign_key_id == foreign_key_id)
        assignments = event.column(foreign_key.child_column_id)
        self.assertTrue(np.all((assignments >= 0) & (assignments < entity.row_count)))
        counts = np.bincount(assignments, minlength=entity.row_count)
        self.assertGreater(float(np.var(counts)), float(np.mean(counts)))

        process = plan.temporal_processes[0]
        time_column = next(
            column.column_id
            for column in schema.table(event.table_id).columns
            if column.kind.value == "time"
        )
        times = event.column(time_column)
        self.assertTrue(np.all(times >= plan.calendar_start_seconds))
        self.assertTrue(np.all(times <= plan.calendar_end_seconds))
        for entity_id in range(entity.row_count):
            per_entity = times[assignments == entity_id]
            self.assertTrue(np.all(np.diff(per_entity) >= 0))
        self.assertIn(process.family, {"stationary", "seasonal", "churn"})
        self.assertTrue(all(mechanism.shared_state_ids for mechanism in plan.column_mechanisms))

        programs = TaskProgramPlanner().plan(
            schema=schema,
            instance_plan=plan,
            prior_plan=prior,
            runtime=runtime.child("task-program"),
        )
        self.assertEqual(1, len(programs))
        program = programs[0]
        self.assertGreater(program.horizon_end_time, program.cutoff_time)
        self.assertGreaterEqual(program.cutoff_time, plan.calendar_start_seconds)
        self.assertLessEqual(program.horizon_end_time, plan.calendar_end_seconds)
        task = TaskExecutor().execute(
            sample_id="temporal_prior",
            schema=schema,
            database=database,
            program=program,
        )
        self.assertIsNotNone(task)
        assert task is not None
        expected = np.zeros(entity.row_count, dtype=np.int8)
        selected = (times > program.cutoff_time) & (times <= program.horizon_end_time)
        expected[np.unique(assignments[selected])] = 1
        observed_ids = np.concatenate((task.data.support_row_ids, task.data.query_row_ids))
        observed_labels = np.concatenate((task.data.support_labels, task.data.query_labels))
        self.assertTrue(np.array_equal(expected[observed_ids], observed_labels))
        self.assertIsNone(
            TaskExecutor().execute(
                sample_id="temporal_prior",
                schema=schema,
                database=database,
                program=program,
                positive_rate_min=1.0,
                positive_rate_max=1.0,
            )
        )

    def test_legacy_instance_plan_and_artifact_readers_remain_compatible(self) -> None:
        runtime, schema, prior, plan, database = self._temporal_fixture()
        legacy_payload = plan.to_dict()
        for key in (
            "prior_plan_id",
            "prior_family",
            "motif_bundles",
            "shared_state_ids",
            "column_mechanisms",
            "population_mechanisms",
            "temporal_processes",
        ):
            legacy_payload.pop(key)
        legacy = InstancePlan.from_dict(legacy_payload)
        self.assertEqual("legacy_role_scm", legacy.prior_family)
        self.assertEqual((), legacy.motif_bundles)

        programs = TaskProgramPlanner().plan(
            schema=schema,
            instance_plan=plan,
            prior_plan=prior,
            runtime=runtime.child("task-program"),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_path = InstanceArtifactWriter(
                output_root=Path(temporary_directory),
                overwrite=True,
            ).commit(
                sample_id="temporal_prior",
                schema_artifact="schema.json",
                runtime=runtime.record(
                    project_version="test",
                    config_digest="test",
                    metadata={},
                ),
                schema=schema,
                plan=plan,
                database=database,
                report=validate_database_instance(schema, plan, database),
                prior_plan=prior,
                task_programs=programs,
            )
            restored = load_instance_artifact(artifact_path)
        self.assertEqual(prior, restored.prior_plan)
        self.assertEqual(programs, restored.task_programs)

    def test_reserved_prior_family_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved but not implemented"):
            PriorPlannerConfig(
                database_family_weights=((PriorFamily.RELATIONAL_TREE, 1.0),),
            )

    def test_temporal_program_is_persisted_before_task_pipeline_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sampler = BlueprintSamplerConfig(
                min_tables=3,
                max_tables=3,
                min_motif_occurrences=1,
                max_motif_occurrences=1,
                max_extra_edges=0,
                background_attachment_probability=0.0,
                motif_weights=(("entity_event", 1.0),),
            )
            schema_result = generate_physical_schemas(
                SchemaPipelineConfig(
                    output_root=root / "schema",
                    num_schemas=1,
                    base_seed=901,
                    sampler=sampler,
                )
            )
            # The schema artifact retains private semantics while the physical
            # schema remains anonymous.
            self.assertIsNotNone(load_schema_artifact(schema_result.artifact_paths[0]).semantic_schema)
            policy = TaskPolicyPlan(
                programs_per_database=1,
                cutoff_fraction_min=0.25,
                cutoff_fraction_max=0.25,
                horizon_fraction_min=0.60,
                horizon_fraction_max=0.60,
                positive_rate_min=0.01,
                positive_rate_max=0.99,
            )
            instance_result = generate_database_instances(
                InstancePipelineConfig(
                    schema_manifest=schema_result.manifest_path,
                    output_root=root / "instance",
                    planner=InstancePlannerConfig(
                        entity_rows_min=48,
                        entity_rows_max=48,
                        lookup_rows_min=4,
                        lookup_rows_max=4,
                        max_rows_per_table=256,
                    ),
                    prior=PriorPlannerConfig(
                        database_family_weights=((PriorFamily.TEMPORAL_EVENT, 1.0),),
                        task_policy=policy,
                    ),
                )
            )
            instance = load_instance_artifact(instance_result.artifact_paths[0])
            self.assertIsNotNone(instance.prior_plan)
            self.assertEqual(PriorFamily.TEMPORAL_EVENT, instance.prior_plan.family)
            self.assertEqual(1, len(instance.task_programs))
            self.assertEqual(
                instance.plan.population_mechanisms[0].parameters,
                tuple(sorted(instance.plan.population_mechanisms[0].parameters)),
            )
            self.assertIn(
                "entity_event_counts",
                dict(instance.plan.population_mechanisms[0].parameters),
            )

            task_result = generate_tasks(
                TaskPipelineConfig(
                    instance_manifest=instance_result.manifest_path,
                    output_root=root / "task",
                )
            )
            self.assertEqual(1, task_result.task_count)
            task_artifact = load_task_artifact(task_result.artifact_paths[0])
            self.assertEqual(instance.task_programs[0], task_artifact.task_program)


if __name__ == "__main__":
    unittest.main()
