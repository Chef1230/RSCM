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


from rdb_prior.artifacts import InstanceArtifactWriter, load_instance_artifact
from rdb_prior.compilation.compiler import PhysicalSchemaCompiler
from rdb_prior.config import SchemaConfigError, load_instance_pipeline_config
from rdb_prior.generation.database import DatabaseGenerator
from rdb_prior.generation.state_trajectory import StateTrajectory
from rdb_prior.generation.temporal_processes import _state_rate
from rdb_prior.instance.planner import InstancePlanner, InstancePlannerConfig
from rdb_prior.pipeline import (
    InstancePipelineConfig,
    SchemaPipelineConfig,
    generate_database_instances,
    generate_physical_schemas,
)
from rdb_prior.priors.model import (
    PriorFamily,
    StateSpacePlan,
    StateVisibility,
    TaskPolicyPlan,
    TransitionClock,
)
from rdb_prior.priors.planner import (
    PriorPlanner,
    PriorPlannerConfig,
    TemporalStateConfig,
)
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.domain_prototypes import sample_semantic_schema
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.validation.checks import validate_database_instance


class TemporalStateTests(unittest.TestCase):
    def test_before_after_and_terminal_intervals_are_unambiguous(self) -> None:
        space = StateSpacePlan(
            values=("active", "review", "blocked"),
            terminal_states=("blocked",),
        )
        trajectory = StateTrajectory(
            calendar_start=0,
            calendar_end=20,
            initial_state="active",
            state_space=space,
        )
        trajectory.transition(timestamp=10, ordinal=0, state="review")
        trajectory.transition(timestamp=10, ordinal=1, state="blocked")

        self.assertEqual("active", trajectory.state_before(10))
        self.assertEqual("active", trajectory.state_before(10, 0))
        self.assertEqual("review", trajectory.state_after(10, 0))
        self.assertEqual("review", trajectory.state_before(10, 1))
        self.assertEqual("blocked", trajectory.state_after(10, 1))
        self.assertTrue(trajectory.is_terminal)
        with self.assertRaisesRegex(ValueError, "terminal"):
            trajectory.transition(timestamp=11, ordinal=0, state="active")

        intervals = trajectory.intervals
        self.assertEqual(intervals[0].end, intervals[1].start)
        self.assertEqual(intervals[1].end, intervals[2].start)
        self.assertEqual(0, intervals[0].start.timestamp)
        self.assertEqual(21, intervals[-1].end.timestamp)

    def test_stateful_materialization_is_deterministic_and_private(self) -> None:
        schema, prior, plan, first = self._stateful_fixture()
        second = DatabaseGenerator().materialize(schema=schema, plan=plan)

        self.assertEqual(prior, type(prior).from_dict(prior.to_dict()))
        self.assertEqual(first.plan, second.plan)
        self.assertIsNotNone(first.temporal_state_registry)
        assert first.temporal_state_registry is not None
        assert second.temporal_state_registry is not None
        self.assertEqual(
            first.temporal_state_registry.to_dict(),
            second.temporal_state_registry.to_dict(),
        )
        for left, right in zip(first.database.tables, second.database.tables):
            self.assertEqual(left.table_id, right.table_id)
            for column_id in left.columns:
                actual = left.column(column_id)
                expected = right.column(column_id)
                self.assertTrue(
                    np.array_equal(
                        actual,
                        expected,
                        equal_nan=actual.dtype.kind == "f",
                    )
                )
        self.assertTrue(
            validate_database_instance(
                schema, first.plan, first.database
            ).is_valid
        )
        state_id = first.plan.temporal_state_plans[0].state_id
        self.assertTrue(
            all(state_id not in table.columns for table in first.database.tables)
        )

        temporal = first.plan.temporal_state_plans[0]
        low = _state_rate(temporal, 0, np.zeros(3), 0.0, 1.0)
        high = _state_rate(temporal, 0, np.ones(3), 0.0, 1.0)
        self.assertNotEqual(low, high)

    def test_stateless_temporal_plan_creates_no_trajectory(self) -> None:
        schema, prior, plan, _materialization = self._stateful_fixture(
            temporal_state_enabled=False
        )
        self.assertEqual((), prior.temporal_states)
        self.assertIsNone(
            DatabaseGenerator().materialize(
                schema=schema, plan=plan
            ).temporal_state_registry
        )

    def test_debug_artifact_round_trip_is_opt_in(self) -> None:
        schema, prior, _plan, materialization = self._stateful_fixture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_path = InstanceArtifactWriter(
                output_root=Path(temporary_directory),
                overwrite=True,
            ).commit(
                sample_id="stateful_prior",
                schema_artifact="schema.json",
                runtime=RuntimeContext(77).record(
                    project_version="test",
                    config_digest="test",
                    metadata={},
                ),
                schema=schema,
                plan=materialization.plan,
                database=materialization.database,
                report=validate_database_instance(
                    schema, materialization.plan, materialization.database
                ),
                prior_plan=prior,
                temporal_state_registry=materialization.temporal_state_registry,
            )
            restored = load_instance_artifact(artifact_path)
            self.assertIsNotNone(restored.temporal_state_registry)
            assert restored.temporal_state_registry is not None
            assert materialization.temporal_state_registry is not None
            self.assertEqual(
                materialization.temporal_state_registry.to_dict(),
                restored.temporal_state_registry.to_dict(),
            )

    def test_pipeline_persists_trajectory_only_when_opted_in(self) -> None:
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
            result = generate_database_instances(
                InstancePipelineConfig(
                    schema_manifest=schema_result.manifest_path,
                    output_root=root / "instance",
                    planner=InstancePlannerConfig(
                        entity_rows_min=24,
                        entity_rows_max=24,
                        lookup_rows_min=4,
                        lookup_rows_max=4,
                        max_rows_per_table=256,
                    ),
                    prior=PriorPlannerConfig(
                        database_family_weights=(
                            (PriorFamily.TEMPORAL_EVENT, 1.0),
                        ),
                        task_policy=TaskPolicyPlan(
                            positive_rate_min=0.01,
                            positive_rate_max=0.99,
                        ),
                        state_dimension=3,
                        temporal_state=TemporalStateConfig(
                            enabled=True,
                            state_space=StateSpacePlan(
                                values=("active", "blocked"),
                                terminal_states=("blocked",),
                            ),
                            transition_clock=TransitionClock.EVENT_DRIVEN,
                        ),
                    ),
                    persist_private_state_trajectory=True,
                )
            )
            restored = load_instance_artifact(result.artifact_paths[0])
            self.assertIsNotNone(restored.temporal_state_registry)


    def test_temporal_state_yaml_is_explicit_and_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "state.yaml"
            path.write_text(
                "prior:\n"
                "  mode: compositional\n"
                "  database_family_weights:\n"
                "    temporal_event: 1.0\n"
                "  shared_state:\n"
                "    family: gaussian_mixture\n"
                "    dimension: 3\n"
                "  temporal_state:\n"
                "    enabled: true\n"
                "    state_space:\n"
                "      values: [active, inactive, blocked]\n"
                "      terminal_states: [blocked]\n"
                "    initial_state:\n"
                "      family: state_conditioned_softmax\n"
                "    transition:\n"
                "      clock: hybrid\n"
                "      family: transition_matrix\n"
                "    duration:\n"
                "      family: exponential\n"
                "    visibility:\n"
                "      family: hidden\n"
                "debug:\n"
                "  persist_private_state_trajectory: true\n",
                encoding="utf-8",
            )
            configured = load_instance_pipeline_config(path)
            assert configured.prior is not None
            self.assertTrue(configured.prior.temporal_state.enabled)
            self.assertIs(
                TransitionClock.HYBRID,
                configured.prior.temporal_state.transition_clock,
            )
            self.assertTrue(configured.persist_private_state_trajectory)

            path.write_text(
                "prior:\n"
                "  mode: compositional\n"
                "  database_family_weights: {temporal_event: 1.0}\n"
                "  temporal_state:\n"
                "    enabled: false\n"
                "    state_space: {values: [active, blocked]}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SchemaConfigError, "disabled"):
                load_instance_pipeline_config(path)

    def _stateful_fixture(self, *, temporal_state_enabled: bool = True):
        runtime = RuntimeContext(1007).for_sample("stateful_prior")
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
        ).sample("stateful_prior", runtime)
        schema = PhysicalSchemaCompiler().compile(
            blueprint, "stateful_prior", runtime
        )
        semantic = sample_semantic_schema(schema, runtime.child("semantic"))
        temporal_state = (
            TemporalStateConfig(
                enabled=True,
                state_space=StateSpacePlan(
                    values=("active", "inactive", "blocked"),
                    terminal_states=("blocked",),
                ),
                transition_clock=TransitionClock.HYBRID,
                visibility=StateVisibility.EVENT_EMISSION,
            )
            if temporal_state_enabled
            else TemporalStateConfig()
        )
        prior = PriorPlanner(
            PriorPlannerConfig(
                database_family_weights=((PriorFamily.TEMPORAL_EVENT, 1.0),),
                task_policy=TaskPolicyPlan(max_materialization_attempts=3),
                state_dimension=3,
                temporal_state=temporal_state,
            )
        ).plan(
            blueprint=blueprint,
            physical_schema=schema,
            semantic_schema=semantic,
            runtime=runtime.child("prior"),
        )
        plan = InstancePlanner(
            InstancePlannerConfig(
                entity_rows_min=24,
                entity_rows_max=24,
                lookup_rows_min=4,
                lookup_rows_max=4,
                max_rows_per_table=256,
            )
        ).plan(
            sample_id="stateful_prior",
            schema=schema,
            runtime=runtime.child("instance"),
            prior_plan=prior,
        )
        return schema, prior, plan, DatabaseGenerator().materialize(
            schema=schema, plan=plan
        )


if __name__ == "__main__":
    unittest.main()
