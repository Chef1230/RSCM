from __future__ import annotations

from dataclasses import replace
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
from rdb_prior.compilation.model import ColumnKind, PhysicalDataType
from rdb_prior.generation.database import DatabaseGenerator
from rdb_prior.generation.encoding import encode_feature_score
from rdb_prior.generation.state import SharedStateRegistry
from rdb_prior.generation.temporal_processes import _temporal_ticks
from rdb_prior.instance.plan import InstancePlan
from rdb_prior.instance.planner import InstancePlanner, InstancePlannerConfig
from rdb_prior.pipeline import (
    InstancePipelineConfig,
    SchemaPipelineConfig,
    generate_database_instances,
    generate_physical_schemas,
)
from rdb_prior.priors.model import (
    AttributePriorKind,
    MechanismRef,
    NuisancePlan,
    NuisancePriorKind,
    PriorCompositionPlan,
    PriorFamily,
    ProcessPriorKind,
    RelationPriorKind,
    TaskPolicyPlan,
    TemporalPriorKind,
)
from rdb_prior.priors.planner import (
    PriorCompositionConfig,
    PriorPlanner,
    PriorPlannerConfig,
)
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.domain_prototypes import sample_semantic_schema
from rdb_prior.schema.semantics import (
    SemanticNodePlan,
    SemanticSchemaPlan,
    TableSemanticPlan,
    TableSemanticRole,
)
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.schema.spec import TableRole
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

    def test_semantic_roles_change_bundle_mechanism_weights(self) -> None:
        runtime = RuntimeContext(91).for_sample("semantic_weights")
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
        ).sample("semantic_weights", runtime)
        schema = PhysicalSchemaCompiler().compile(
            blueprint,
            "semantic_weights",
            runtime,
        )
        base = sample_semantic_schema(schema, runtime.child("semantic"))
        entity_id = next(item.table_id for item in base.tables if schema.table(item.table_id).role is TableRole.ENTITY)
        event_id = next(item.table_id for item in base.tables if schema.table(item.table_id).role is TableRole.EVENT)
        def with_roles(entity_role, event_role):
            tables = tuple(
                TableSemanticPlan(
                    table_id=item.table_id,
                    role=(
                        entity_role if item.table_id == entity_id
                        else event_role if item.table_id == event_id
                        else item.role
                    ),
                )
                for item in base.tables
            )
            nodes = tuple(
                SemanticNodePlan(node_id=item.table_id, role=item.role)
                for item in tables
            )
            return SemanticSchemaPlan(
                schema_id=schema.schema_id,
                prototype_id=base.prototype_id,
                seed=base.seed,
                tables=tables,
                columns=base.columns,
                nodes=nodes,
            )
        policy = TaskPolicyPlan(programs_per_database=1)
        planner = PriorPlanner(
            PriorPlannerConfig(
                database_family_weights=((PriorFamily.TEMPORAL_EVENT, 1.0),),
                task_policy=policy,
            )
        )
        actor_transaction = planner.plan(
            blueprint=blueprint,
            physical_schema=schema,
            semantic_schema=with_roles(TableSemanticRole.ACTOR, TableSemanticRole.TRANSACTION),
            runtime=runtime.child("prior-a"),
        )
        object_observation = planner.plan(
            blueprint=blueprint,
            physical_schema=schema,
            semantic_schema=with_roles(TableSemanticRole.OBJECT, TableSemanticRole.OBSERVATION),
            runtime=runtime.child("prior-b"),
        )
        first = next(item for item in actor_transaction.motif_bundles if item.family is PriorFamily.TEMPORAL_EVENT)
        second = next(item for item in object_observation.motif_bundles if item.family is PriorFamily.TEMPORAL_EVENT)
        self.assertNotEqual(
            dict(first.parameters)["semantic_mechanism_weights"],
            dict(second.parameters)["semantic_mechanism_weights"],
        )

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
        self.assertIn(process.family, {"stationary", "seasonal", "churn", "renewal"})
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

    def test_static_state_signal_drives_realized_event_counts(self) -> None:
        _runtime, _schema, _prior, plan, _database = self._temporal_fixture()
        mechanism = plan.population_mechanisms[0]
        state = SharedStateRegistry.from_plan(plan).state(
            mechanism.state_ids[0]
        )
        parameters = dict(mechanism.parameters)
        signal = state @ np.asarray(parameters["count_state_weights"])
        counts = np.asarray(parameters["entity_event_counts"])
        self.assertGreater(np.corrcoef(signal, counts)[0, 1], 0.25)

    def test_temporal_column_mechanism_family_changes_materialized_events(self) -> None:
        _runtime, schema, _prior, plan, _database = self._temporal_fixture()
        event_id = plan.population_mechanisms[0].table_id
        event_features = {
            column.column_id
            for column in schema.table(event_id).columns
            if column.kind is ColumnKind.FEATURE
        }
        linear = replace(
            plan,
            column_mechanisms=tuple(
                replace(item, family="linear")
                if item.column_id in event_features
                else item
                for item in plan.column_mechanisms
            ),
        )
        cam = replace(
            plan,
            column_mechanisms=tuple(
                replace(item, family="cam")
                if item.column_id in event_features
                else item
                for item in plan.column_mechanisms
            ),
        )
        linear_event = DatabaseGenerator().generate(
            schema=schema, plan=linear
        ).table(event_id)
        cam_event = DatabaseGenerator().generate(
            schema=schema, plan=cam
        ).table(event_id)
        changed = []
        for column_id in event_features:
            left = linear_event.column(column_id)
            right = cam_event.column(column_id)
            changed.append(
                not np.array_equal(
                    left,
                    right,
                    equal_nan=left.dtype.kind == "f",
                )
            )
        self.assertTrue(any(changed))

    def test_static_state_changes_every_temporal_time_family(self) -> None:
        parameters = {
            "seasonal_strength": 0.7,
            "churn_exponent": 1.3,
            "state_churn_weights": [0.8, 0.6, 0.4],
            "state_seasonal_phase_weights": [0.7, 0.5, 0.3],
            "state_active_interval_weights": [0.9, 0.7, 0.5],
            "state_renewal_scale_weights": [0.6, 0.4, 0.2],
        }
        low_state = np.asarray([-2.0, -2.0, -2.0])
        high_state = np.asarray([2.0, 2.0, 2.0])
        for family in ("stationary", "seasonal", "churn", "renewal"):
            left = _temporal_ticks(
                np.random.Generator(np.random.PCG64DXSM(81)),
                family,
                24,
                low_state,
                parameters,
            )
            right = _temporal_ticks(
                np.random.Generator(np.random.PCG64DXSM(81)),
                family,
                24,
                high_state,
                parameters,
            )
            self.assertTrue(np.all((left >= 0.0) & (left <= 1.0)))
            self.assertTrue(np.all(np.diff(left) >= 0.0))
            self.assertFalse(np.allclose(left, right), family)

    def test_temporal_encoder_preserves_categorical_missingness(self) -> None:
        _runtime, schema, _prior, plan, _database = self._temporal_fixture()
        event_id = plan.population_mechanisms[0].table_id
        source = next(
            column
            for column in schema.table(event_id).columns
            if column.kind is ColumnKind.FEATURE
        )
        column = replace(
            source,
            data_type=PhysicalDataType.TEXT,
            nullable=True,
        )
        values = encode_feature_score(
            np.linspace(-3.0, 3.0, 96),
            column,
            schema.table(event_id).role,
            np.random.Generator(np.random.PCG64DXSM(123)),
            cardinality=8,
            db_start=plan.calendar_start_seconds,
            db_end=plan.calendar_end_seconds,
            categorical_dirichlet_alpha=1.0,
            categorical_signal_strength=1.0,
            missing_rate=0.5,
        )
        self.assertEqual("U", values.dtype.kind)
        self.assertTrue(np.any(values == ""))
        self.assertTrue(np.any(values != ""))

    def test_prior_artifact_v1_is_upgraded_to_composition(self) -> None:
        _runtime, _schema, prior, plan, _database = self._temporal_fixture()
        legacy = prior.to_dict()
        legacy.pop("composition")
        for bundle in legacy["motif_bundles"]:
            attribute = bundle["attribute_mechanism"]["kind"]
            temporal = bundle["temporal_mechanism"]["kind"]
            bundle["attribute_mechanism"] = (
                "sampled_linear_cam"
                if attribute == AttributePriorKind.COLUMN_SCM.value
                else "legacy"
            )
            bundle["temporal_mechanism"] = (
                "sampled_stationary_seasonal_churn"
                if temporal != TemporalPriorKind.STATIC.value
                else "legacy"
            )
            bundle.pop("relation_mechanism")
            bundle.pop("process_mechanism")
            bundle.pop("shared_state_ids")
        restored = type(prior).from_dict(legacy)
        self.assertIsNotNone(restored.composition)
        assert restored.composition is not None
        self.assertEqual(
            NuisancePriorKind.LEGACY.value,
            restored.composition.nuisance_plan.mechanism.kind,
        )
        self.assertEqual(restored.plan_id, plan.prior_composition_id)
        self.assertEqual(restored, type(restored).from_dict(restored.to_dict()))

    def test_composition_plan_can_combine_tree_temporal_and_nuisance(self) -> None:
        _runtime, _schema, prior, _plan, _database = self._temporal_fixture()
        assert prior.composition is not None
        bundle = next(
            item
            for item in prior.composition.motif_bundles
            if item.family is PriorFamily.TEMPORAL_EVENT
        )
        composed_bundle = replace(
            bundle,
            attribute_mechanism=MechanismRef(
                kind=AttributePriorKind.TREE.value,
                version="v1",
            ),
            relation_mechanism=MechanismRef(
                kind=RelationPriorKind.TREE.value,
                version="v1",
            ),
            temporal_mechanism=MechanismRef(
                kind=TemporalPriorKind.CHURN.value,
                version="v1",
            ),
            process_mechanism=MechanismRef(
                kind=ProcessPriorKind.NONE.value,
                version="v1",
            ),
        )
        composition = replace(
            prior.composition,
            motif_bundles=tuple(
                composed_bundle if item.bundle_id == bundle.bundle_id else item
                for item in prior.composition.motif_bundles
            ),
            nuisance_plan=NuisancePlan(
                mechanism=MechanismRef(
                    kind=NuisancePriorKind.MNAR.value,
                    version="v1",
                )
            ),
        )
        restored = PriorCompositionPlan.from_dict(composition.to_dict())
        self.assertEqual(composition, restored)
        self.assertEqual(NuisancePriorKind.MNAR.value, restored.nuisance_plan.mechanism.kind)
        refs = (
            restored.motif_bundles[0].attribute_mechanism,
            restored.motif_bundles[0].relation_mechanism,
            restored.motif_bundles[0].temporal_mechanism,
            restored.motif_bundles[0].process_mechanism,
            restored.nuisance_plan.mechanism,
        )
        self.assertTrue(all(item.version for item in refs))

    def test_planner_uses_explicit_composition_for_temporal_bundle(self) -> None:
        runtime = RuntimeContext(614).for_sample("composed_prior")
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
        ).sample("composed_prior", runtime)
        schema = PhysicalSchemaCompiler().compile(
            blueprint,
            "composed_prior",
            runtime,
        )
        prior = PriorPlanner(
            PriorPlannerConfig(
                composition=PriorCompositionConfig(
                    attribute=AttributePriorKind.TREE,
                    relation=RelationPriorKind.TREE,
                    temporal=TemporalPriorKind.CHURN,
                    nuisance=NuisancePriorKind.MNAR,
                )
            )
        ).plan(
            blueprint=blueprint,
            physical_schema=schema,
            semantic_schema=sample_semantic_schema(
                schema,
                runtime.child("semantic"),
            ),
            runtime=runtime.child("prior"),
        )
        bundle = next(
            item
            for item in prior.motif_bundles
            if item.family is PriorFamily.TEMPORAL_EVENT
        )
        self.assertEqual(AttributePriorKind.TREE.value, bundle.attribute_mechanism.kind)
        self.assertEqual(RelationPriorKind.TREE.value, bundle.relation_mechanism.kind)
        self.assertEqual(TemporalPriorKind.CHURN.value, bundle.temporal_mechanism.kind)
        assert prior.composition is not None
        self.assertEqual(
            NuisancePriorKind.MNAR.value,
            prior.composition.nuisance_plan.mechanism.kind,
        )
        with self.assertRaisesRegex(ValueError, "only executes column_scm"):
            InstancePlanner(
                InstancePlannerConfig(
                    entity_rows_min=24,
                    entity_rows_max=24,
                    lookup_rows_min=4,
                    lookup_rows_max=4,
                    max_rows_per_table=256,
                )
            ).plan(
                sample_id="composed_prior",
                schema=schema,
                runtime=runtime.child("instance"),
                prior_plan=prior,
            )

    def test_composition_rejects_conflicting_axes(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a temporal prior"):
            PriorCompositionConfig(
                attribute=AttributePriorKind.TREE,
                relation=RelationPriorKind.STATE_CONDITIONED_EVENT,
                temporal=TemporalPriorKind.STATIC,
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
