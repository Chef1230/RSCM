from __future__ import annotations

from dataclasses import replace
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


from rdb_prior.artifacts import load_instance_artifact
from rdb_prior.compilation.compiler import PhysicalSchemaCompiler
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
from rdb_prior.task.artifacts import load_task_artifact
from rdb_prior.task.mechanisms import (
    _aggregate_values,
    _temporal_split,
    TemporalAggregateCandidate,
    build_interaction_response_task,
    build_history_gated_future_active_task,
    build_history_gated_future_inactive_task,
    interaction_candidates,
    future_event_candidates,
    future_event_labels,
    mechanism_labels,
)
from rdb_prior.task.model import (
    AggregateOperator,
    RouteRole,
    TaskMechanism,
    TaskPlan,
)
from rdb_prior.task.pipeline import TaskPipelineConfig, generate_tasks
from rdb_prior.task.planner import (
    TaskPlanner,
    TaskPlannerConfig,
    _bind_instance_calendar,
)
from rdb_prior.task.validation import validate_task
from rdb_prior.task.view import build_task_view


class TaskGenerationTests(unittest.TestCase):
    def _database(self, sample_id: str, *, min_tables: int = 4, max_tables: int = 4):
        runtime = RuntimeContext(303).for_sample(sample_id)
        blueprint = BlueprintSampler(
            BlueprintSamplerConfig(min_tables=min_tables, max_tables=max_tables)
        ).sample(sample_id, runtime)
        schema = PhysicalSchemaCompiler().compile(blueprint, sample_id, runtime)
        plan = InstancePlanner(
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
        database = DatabaseGenerator().generate(schema=schema, plan=plan)
        return runtime, schema, database

    def test_temporal_split_rejects_single_class_query(self) -> None:
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
        ordered = np.arange(len(labels), dtype=np.int64)

        split = _temporal_split(
            labels,
            ordered,
            np.random.default_rng(17),
            support_fraction=0.5,
            min_support_rows=4,
            min_query_rows=4,
            min_class_count=2,
        )

        self.assertIsNone(split)

    def test_relation_attribute_task_masks_target_and_round_trips(self) -> None:
        runtime, schema, database = self._database("attribute_task")
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=2,
                mechanism_weights=((TaskMechanism.RELATION_ATTRIBUTE, 1.0),),
                min_support_rows=12,
                min_query_rows=6,
            )
        )

        tasks = planner.generate(
            sample_id="attribute_task",
            schema=schema,
            database=database,
            runtime=runtime.child("task"),
        )

        self.assertEqual(2, len(tasks))
        for task in tasks:
            self.assertIs(TaskMechanism.RELATION_ATTRIBUTE, task.plan.mechanism)
            self.assertIn(
                task.plan.target_column_id,
                task.plan.masked_column_ids,
            )
            self.assertTrue(validate_task(schema, database, task).is_valid)
            view = build_task_view(schema, database, task.plan)
            self.assertTrue(
                view.is_column_masked(
                    task.plan.target_table_id,
                    task.plan.target_column_id or "",
                )
            )
            target_mask = view.row_masks[task.plan.target_table_id]
            self.assertTrue(
                np.all(target_mask[task.data.support_row_ids])
            )
            self.assertTrue(
                np.all(target_mask[task.data.query_row_ids])
            )
            self.assertEqual(task.plan, TaskPlan.from_dict(task.plan.to_dict()))

    def test_future_event_task_recomputes_labels_and_cuts_visibility(self) -> None:
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=1,
                mechanism_weights=(
                    (TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE, 1.0),
                ),
                min_support_rows=8,
                min_query_rows=4,
                min_class_count_per_split=1,
                max_attempts_per_database=512,
            )
        )
        for index in range(40):
            sample_id = f"future_task_{index}"
            runtime, schema, database = self._database(
                sample_id, min_tables=5, max_tables=7
            )
            try:
                tasks = planner.generate(
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    runtime=runtime.child("task"),
                )
            except ValueError:
                continue
            task = tasks[0]
            expected = future_event_labels(schema, database, task.plan)

            np.testing.assert_array_equal(
                task.data.support_labels,
                expected[task.data.support_row_ids],
            )
            np.testing.assert_array_equal(
                task.data.query_labels,
                expected[task.data.query_row_ids],
            )
            self.assertTrue(task.plan.observation_rules)
            self.assertTrue(
                all(
                    rule.max_timestamp == task.plan.cutoff_time
                    for rule in task.plan.observation_rules
                )
            )
            self.assertTrue(validate_task(schema, database, task).is_valid)
            view = build_task_view(schema, database, task.plan)
            for rule in task.plan.observation_rules:
                visible = view.visible_rows(rule.table_id)
                times = database.table(rule.table_id).column(rule.time_column_id)
                self.assertTrue(np.all(times[visible] <= rule.max_timestamp))
            for foreign_key in schema.foreign_keys:
                child_rows = view.visible_rows(foreign_key.child_table_id)
                assignments = database.table(foreign_key.child_table_id).column(
                    foreign_key.child_column_id
                )[child_rows]
                valid = assignments >= 0
                parent_mask = view.row_masks[foreign_key.parent_table_id]
                self.assertTrue(np.all(parent_mask[assignments[valid]]))
            return
        self.fail("no balanced future-event task found in 20 databases")

    def test_history_gated_future_activity_gates_on_prior_events(self) -> None:
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=1,
                mechanism_weights=(
                    (TaskMechanism.HISTORY_GATED_FUTURE_ACTIVITY, 1.0),
                ),
                min_support_rows=8,
                min_query_rows=4,
                min_class_count_per_split=1,
                max_attempts_per_database=512,
            )
        )
        for index in range(40):
            sample_id = f"history_gated_task_{index}"
            runtime, schema, database = self._database(
                sample_id, min_tables=5, max_tables=7
            )
            try:
                tasks = planner.generate(
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    runtime=runtime.child("task"),
                )
            except ValueError:
                continue
            task = tasks[0]
            plan = task.plan

            # Recompute the label from raw event rows: 1 iff the entity has
            # at least one event at or before the cutoff and none within
            # (cutoff, horizon].
            fk = next(
                foreign_key
                for foreign_key in schema.foreign_keys
                if foreign_key.foreign_key_id == plan.foreign_key_id
            )
            event = database.table(plan.source_table_id)
            times = event.column(plan.time_column_id or "")
            assignments = event.column(fk.child_column_id)
            entity_count = database.table(plan.target_table_id).row_count
            valid = assignments >= 0
            has_history = np.zeros(entity_count, dtype=bool)
            has_future = np.zeros(entity_count, dtype=bool)
            has_history[
                np.unique(assignments[valid & (times <= plan.cutoff_time)])
            ] = True
            has_future[np.unique(assignments[
                valid
                & (times > plan.cutoff_time)
                & (times <= plan.horizon_end_time)
            ])] = True
            expected = (has_history & ~has_future).astype(np.int8)

            np.testing.assert_array_equal(
                task.data.support_labels,
                expected[task.data.support_row_ids],
            )
            np.testing.assert_array_equal(
                task.data.query_labels,
                expected[task.data.query_row_ids],
            )
            np.testing.assert_array_equal(
                expected, mechanism_labels(schema, database, plan)
            )
            # Both classes must occur: labels are gated on having history.
            self.assertEqual({0, 1}, set(expected.tolist()))
            self.assertTrue(np.any(has_history))
            self.assertTrue(validate_task(schema, database, task).is_valid)
            return
        self.fail("no balanced history-gated task found in 40 databases")

    def _history_mask(
        self,
        schema: PhysicalSchema,
        database: DatabaseInstance,
        plan: TaskPlan,
    ) -> np.ndarray:
        """Independently compute which entities have prior history.

        This is the only non-stochastic part of the sub-mode labels; the
        ``-1`` sentinel gate always agrees with it.
        """
        fk = next(
            foreign_key
            for foreign_key in schema.foreign_keys
            if foreign_key.foreign_key_id == plan.foreign_key_id
        )
        event = database.table(plan.source_table_id)
        times = event.column(plan.time_column_id or "")
        assignments = event.column(fk.child_column_id)
        entity_count = database.table(plan.target_table_id).row_count
        valid = assignments >= 0
        has_history = np.zeros(entity_count, dtype=bool)
        has_history[
            np.unique(assignments[valid & (times <= plan.cutoff_time)])
        ] = True
        return has_history

    def _test_history_gated_submode(self, mechanism: TaskMechanism) -> None:
        """Shared test body for the two history-gated sub-modes.

        Both sub-modes restrict samples to entities with prior history and
        derive labels from actual events in the future window; the two only
        differ in which group is the positive class.
        """
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=1,
                mechanism_weights=((mechanism, 1.0),),
                min_support_rows=8,
                min_query_rows=4,
                min_class_count_per_split=1,
                max_attempts_per_database=512,
            )
        )
        for index in range(40):
            sample_id = f"hg_submode_{mechanism.value}_{index}"
            runtime, schema, database = self._database(
                sample_id, min_tables=5, max_tables=7
            )
            try:
                tasks = planner.generate(
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    runtime=runtime.child("task"),
                )
            except ValueError:
                continue
            task = tasks[0]
            plan = task.plan

            expected = mechanism_labels(schema, database, plan)
            history_mask = self._history_mask(schema, database, plan)
            fk = next(
                foreign_key
                for foreign_key in schema.foreign_keys
                if foreign_key.foreign_key_id == plan.foreign_key_id
            )
            event = database.table(plan.source_table_id)
            times = event.column(plan.time_column_id or "")
            assignments = event.column(fk.child_column_id)
            has_future = np.zeros(len(history_mask), dtype=bool)
            valid = assignments >= 0
            has_future[
                np.unique(assignments[
                    valid
                    & (times > plan.cutoff_time)
                    & (times <= plan.horizon_end_time)
                ])
            ] = True
            actual = np.full(len(history_mask), -1, dtype=np.int8)
            if mechanism is TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE:
                actual[history_mask] = has_future[history_mask].astype(np.int8)
            else:
                actual[history_mask] = (~has_future[history_mask]).astype(np.int8)
            # The label must be a direct function of the real future window.
            np.testing.assert_array_equal(expected, actual)
            # Entities without history are excluded from the sample.
            supervised = np.concatenate(
                (task.data.support_row_ids, task.data.query_row_ids)
            )
            self.assertTrue(
                np.all(expected[supervised] >= 0),
                f"Sub-mode {mechanism.value}: all supervised entities must have history",
            )
            # The -1 sentinel exactly marks no-history entities.
            self.assertEqual(
                {-1},
                set(np.unique(expected[~history_mask]).tolist()),
                f"Sub-mode {mechanism.value}: no-history entities must be -1",
            )
            # Both classes must be present.
            supervised_labels = expected[supervised]
            self.assertEqual(
                {0, 1},
                set(np.unique(supervised_labels).tolist()),
                f"Sub-mode {mechanism.value}: both classes must appear",
            )

            # The task's stored labels agree with the recompute path, and the
            # plan round-trips through serialization without changing labels.
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
            self.assertTrue(
                np.any(expected >= 0),
                f"Sub-mode {mechanism.value}: some entities must have history",
            )
            self.assertTrue(validate_task(schema, database, task).is_valid)
            return
        self.fail(f"no balanced {mechanism.value} task found in 40 databases")

    def test_history_gated_future_inactive_submode(self) -> None:
        """Sub-mode 1: entities with history, positive = no future events."""
        self._test_history_gated_submode(
            TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE
        )

    def test_history_gated_future_active_submode(self) -> None:
        """Sub-mode 2: entities with history, positive = has future events."""
        self._test_history_gated_submode(
            TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE
        )

    def test_history_gated_labels_use_the_future_window(self) -> None:
        """Labels must be the observed future-event result, not a propensity draw."""
        checked = 0
        for mechanism, build in (
            (
                TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE,
                build_history_gated_future_inactive_task,
            ),
            (
                TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE,
                build_history_gated_future_active_task,
            ),
        ):
            for index in range(20):
                sample_id = f"hg_future_labels_{mechanism.value}_{index}"
                runtime, schema, database = self._database(
                    sample_id, min_tables=5, max_tables=7
                )
                candidates = future_event_candidates(schema)
                if not candidates:
                    continue
                task = build(
                    task_id=f"hg_future_labels_{mechanism.value}_{index}",
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    candidate=candidates[0],
                    seed=runtime.seed("hg-future-labels"),
                    support_fraction=0.7,
                    min_support_rows=4,
                    min_query_rows=2,
                    min_class_count_per_split=1,
                    cutoff_quantile_min=0.45,
                    cutoff_quantile_max=0.7,
                    horizon_fraction_min=0.12,
                    horizon_fraction_max=0.3,
                    positive_rate_min=0.2,
                    positive_rate_max=0.8,
                    history_gated_frequency_weight_min=0.1,
                    history_gated_frequency_weight_max=10.0,
                    history_gated_silence_weight_min=0.1,
                    history_gated_silence_weight_max=10.0,
                )
                if task is None:
                    continue
                plan = task.plan
                labels = mechanism_labels(schema, database, plan)
                fk = next(
                    foreign_key
                    for foreign_key in schema.foreign_keys
                    if foreign_key.foreign_key_id == plan.foreign_key_id
                )
                event = database.table(plan.source_table_id)
                times = event.column(plan.time_column_id or "")
                assignments = event.column(fk.child_column_id)
                history = np.zeros(len(labels), dtype=bool)
                future = np.zeros(len(labels), dtype=bool)
                valid = assignments >= 0
                history[np.unique(assignments[valid & (times <= plan.cutoff_time)])] = True
                future[
                    np.unique(assignments[
                        valid
                        & (times > plan.cutoff_time)
                        & (times <= plan.horizon_end_time)
                    ])
                ] = True
                expected = np.full(len(labels), -1, dtype=np.int8)
                if mechanism is TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE:
                    expected[history] = future[history].astype(np.int8)
                else:
                    expected[history] = (~future[history]).astype(np.int8)
                np.testing.assert_array_equal(labels, expected)
                self.assertAlmostEqual(
                    plan.realized_positive_rate,
                    float(np.mean(expected[expected >= 0])),
                )
                checked += 1
        self.assertGreaterEqual(checked, 2, "need active and inactive future-window tasks")

    def test_history_gated_labels_reproducible_from_plan(self) -> None:
        """Serializing a plan must preserve the future-window labels."""
        for mechanism, build in (
            (
                TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE,
                build_history_gated_future_inactive_task,
            ),
            (
                TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE,
                build_history_gated_future_active_task,
            ),
        ):
            checked = 0
            for index in range(12):
                sample_id = f"hg_repro_{mechanism.value}_{index}"
                runtime, schema, database = self._database(
                    sample_id, min_tables=5, max_tables=7
                )
                candidates = future_event_candidates(schema)
                if not candidates:
                    continue
                for seed in range(20):
                    task = build(
                        task_id=f"hg_repro_{mechanism.value}_{index}_{seed}",
                        sample_id=sample_id,
                        schema=schema,
                        database=database,
                        candidate=candidates[0],
                        seed=runtime.seed("hg-repro", seed),
                        support_fraction=0.7,
                        min_support_rows=4,
                        min_query_rows=2,
                        min_class_count_per_split=1,
                        cutoff_quantile_min=0.45,
                        cutoff_quantile_max=0.7,
                        horizon_fraction_min=0.12,
                        horizon_fraction_max=0.3,
                        positive_rate_min=0.2,
                        positive_rate_max=0.8,
                    )
                    if task is None:
                        continue
                    plan = task.plan
                    np.testing.assert_array_equal(
                        mechanism_labels(
                            schema,
                            database,
                            TaskPlan.from_dict(plan.to_dict()),
                        ),
                        mechanism_labels(schema, database, plan),
                    )
                    checked += 1
                    if checked >= 8:
                        break
                if checked >= 8:
                    break
            self.assertGreaterEqual(
                checked, 8, f"need enough {mechanism.value} tasks"
            )

    def test_temporal_aggregate_mixed_window_regimes(self) -> None:
        """Aggregate tasks must mix SHORT, LONG and REPEATED windows."""
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=1,
                mechanism_weights=(
                    (TaskMechanism.TEMPORAL_RELATIONAL_AGGREGATE, 1.0),
                ),
                min_support_rows=8,
                min_query_rows=4,
                min_class_count_per_split=1,
                max_attempts_per_database=512,
            )
        )
        regimes: dict[int, int] = {0: 0, 1: 0, 2: 0}
        observed = 0
        for index in range(120):
            sample_id = f"tagg_regime_{index}"
            runtime, schema, database = self._database(
                sample_id, min_tables=5, max_tables=7
            )
            try:
                tasks = planner.generate(
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    runtime=runtime.child("task"),
                )
            except ValueError:
                continue
            for task in tasks:
                params = task.plan.parameter_map
                regime = int(params["window_regime"])
                self.assertIn(regime, (0, 1, 2))
                window = int(params["window"])
                extended = params.get("window_extended")
                if regime == 2:
                    self.assertIsNotNone(extended)
                    self.assertGreater(int(extended), window)
                else:
                    self.assertIsNone(extended)
                regimes[regime] += 1
                observed += 1
            if observed >= 60:
                break
        self.assertGreaterEqual(observed, 60, "need enough aggregate tasks")
        self.assertTrue(
            all(regimes.values()),
            f"all three window regimes must appear: {regimes}",
        )

    def test_temporal_aggregate_repeated_sums_short_long(self) -> None:
        """A REPEATED task's label equals the short+long summed aggregate."""
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=1,
                mechanism_weights=(
                    (TaskMechanism.TEMPORAL_RELATIONAL_AGGREGATE, 1.0),
                ),
                min_support_rows=8,
                min_query_rows=4,
                min_class_count_per_split=1,
                max_attempts_per_database=512,
            )
        )
        for index in range(80):
            sample_id = f"tagg_repeated_{index}"
            runtime, schema, database = self._database(
                sample_id, min_tables=5, max_tables=7
            )
            try:
                tasks = planner.generate(
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    runtime=runtime.child("task"),
                )
            except ValueError:
                continue
            for task in tasks:
                plan = task.plan
                params = plan.parameter_map
                if params.get("window_extended") is None:
                    continue
                # Reconstruct candidate and cutoffs exactly as the recompute
                # path does, then re-derive the nested multi-scale aggregate.
                required = next(
                    label.foreign_key_ids
                    for label in plan.route_supervision
                    if label.role is RouteRole.REQUIRED
                )
                target = schema.table(plan.target_table_id)
                row_time = plan.row_cutoff_time_column_id
                if row_time is None:
                    cutoffs = np.full(
                        database.table(target.table_id).row_count,
                        int(plan.cutoff_time),
                        dtype=np.int64,
                    )
                else:
                    cutoffs = (
                        database.table(target.table_id)
                        .column(row_time)
                        .astype(np.int64)
                    )
                candidate = TemporalAggregateCandidate(
                    target_table_id=plan.target_table_id,
                    source_table_id=plan.source_table_id,
                    required_path=required,
                    time_column_id=plan.time_column_id or "",
                    operator=plan.aggregate_operator or AggregateOperator.COUNT,
                    source_column_id=plan.source_column_id,
                )
                window = int(params["window"])
                extended = int(params["window_extended"])
                summed = (
                    _aggregate_values(
                        schema, database, candidate, cutoffs, window
                    )
                    + _aggregate_values(
                        schema, database, candidate, cutoffs, extended
                    )
                )
                expected = (summed > float(plan.threshold)).astype(np.int8)
                np.testing.assert_array_equal(
                    expected, mechanism_labels(schema, database, plan)
                )
                self.assertTrue(validate_task(schema, database, task).is_valid)
                return
        self.fail("no REPEATED aggregate task found in 80 databases")

    def test_temporal_aggregate_recomputable_exact(self) -> None:
        """Stored aggregate labels must exactly equal the recompute path."""
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=1,
                mechanism_weights=(
                    (TaskMechanism.TEMPORAL_RELATIONAL_AGGREGATE, 1.0),
                ),
                min_support_rows=8,
                min_query_rows=4,
                min_class_count_per_split=1,
                max_attempts_per_database=512,
            )
        )
        observed = 0
        regimes: set[int] = set()
        for index in range(60):
            sample_id = f"tagg_recompute_{index}"
            runtime, schema, database = self._database(
                sample_id, min_tables=5, max_tables=7
            )
            try:
                tasks = planner.generate(
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    runtime=runtime.child("task"),
                )
            except ValueError:
                continue
            for task in tasks:
                expected = mechanism_labels(schema, database, task.plan)
                np.testing.assert_array_equal(
                    task.data.support_labels,
                    expected[task.data.support_row_ids],
                )
                np.testing.assert_array_equal(
                    task.data.query_labels,
                    expected[task.data.query_row_ids],
                )
                self.assertTrue(validate_task(schema, database, task).is_valid)
                regimes.add(int(task.plan.parameter_map["window_regime"]))
                observed += 1
            if observed >= 30:
                break
        self.assertGreaterEqual(observed, 30, "need enough aggregate tasks")
        self.assertEqual({0, 1, 2}, regimes)

    def _interaction_covariates(
        self,
        schema: PhysicalSchema,
        database: DatabaseInstance,
        plan: TaskPlan,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Independent per-event-row covariates for interaction-response tasks.

        Returns ``(gate, counts, silence)`` where ``gate[e]`` is True when the
        row's entity has at least one interaction strictly before its own time,
        ``counts[e]`` is the entity's prior interaction count and ``silence[e]``
        the normalized gap since the last one. Only non-stochastic inputs to
        the label.
        """
        fk = next(
            foreign_key
            for foreign_key in schema.foreign_keys
            if foreign_key.foreign_key_id == plan.foreign_key_id
        )
        event = database.table(plan.target_table_id)
        times = event.column(plan.time_column_id or "").astype(np.int64)
        entity = event.column(fk.child_column_id)
        row_count = len(times)
        counts = np.zeros(row_count, dtype=np.int64)
        last = np.full(row_count, -1, dtype=np.int64)
        gate = np.zeros(row_count, dtype=bool)
        for e in range(row_count):
            ent = int(entity[e])
            if ent < 0:
                continue
            t_e = int(times[e])
            history = (entity == ent) & (times < t_e)
            if np.any(history):
                counts[e] = np.count_nonzero(history)
                last[e] = int(times[history].max())
                gate[e] = True
        span = max(1, int(times.max()) - int(times.min()))
        silence = np.zeros(row_count, dtype=np.float64)
        silence[gate] = np.clip((times[gate] - last[gate]) / span, 0.0, 1.0)
        return gate, counts, silence

    def test_interaction_response_gates_on_prior_interaction(self) -> None:
        """Interaction rows are gated on the entity's prior interaction history."""
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=1,
                mechanism_weights=(
                    (TaskMechanism.INTERACTION_RESPONSE, 1.0),
                ),
                min_support_rows=8,
                min_query_rows=4,
                min_class_count_per_split=1,
                max_attempts_per_database=512,
            )
        )
        for index in range(40):
            sample_id = f"interaction_gate_{index}"
            runtime, schema, database = self._database(
                sample_id, min_tables=5, max_tables=7
            )
            try:
                tasks = planner.generate(
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    runtime=runtime.child("task"),
                )
            except ValueError:
                continue
            task = tasks[0]
            plan = task.plan
            expected = mechanism_labels(schema, database, plan)
            gate, _counts, _silence = self._interaction_covariates(
                schema, database, plan
            )
            supervised = np.concatenate(
                (task.data.support_row_ids, task.data.query_row_ids)
            )
            self.assertTrue(
                np.all(gate[supervised]),
                "all supervised rows must be history-gated",
            )
            self.assertEqual(
                {-1},
                set(np.unique(expected[~gate]).tolist()),
                "ungated rows must be -1",
            )
            self.assertEqual(
                {0, 1},
                set(np.unique(expected[supervised]).tolist()),
                "both response classes must appear",
            )
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
            self.assertTrue(validate_task(schema, database, task).is_valid)
            return
        self.fail("no interaction-response task found in 40 databases")

    def test_interaction_response_label_direction_follows_invert(self) -> None:
        """Response probability rises with frequency and falls with silence.

        High-frequency recent interactions must be more likely positive when
        ``invert=0`` (positive = response) and less likely when ``invert=1``
        (positive = ignore). The engaged/dormant buckets align both propensity
        terms so the within-task ordering is strict.
        """
        stats = {
            0: {"engaged_pos": 0, "engaged_total": 0, "dormant_pos": 0, "dormant_total": 0},
            1: {"engaged_pos": 0, "engaged_total": 0, "dormant_pos": 0, "dormant_total": 0},
        }
        tasks = 0
        for index in range(16):
            sample_id = f"interaction_dir_{index}"
            runtime, schema, database = self._database(
                sample_id, min_tables=5, max_tables=7
            )
            candidates = interaction_candidates(schema, database)
            if not candidates:
                continue
            candidate = candidates[0]
            for seed in range(30):
                task = build_interaction_response_task(
                    task_id=f"interaction_dir_{index}_{seed}",
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    candidate=candidate,
                    seed=runtime.seed("interaction-dir", seed),
                    support_fraction=0.7,
                    min_support_rows=4,
                    min_query_rows=2,
                    min_class_count_per_split=1,
                    positive_rate_min=0.2,
                    positive_rate_max=0.8,
                    interaction_u_weight_min=0.25,
                    interaction_u_weight_max=2.0,
                    interaction_frequency_weight_min=0.5,
                    interaction_frequency_weight_max=3.0,
                    interaction_silence_weight_min=0.5,
                    interaction_silence_weight_max=3.0,
                    interaction_item_weight_min=0.5,
                    interaction_item_weight_max=3.0,
                    interaction_invert_probability=0.5,
                )
                if task is None:
                    continue
                plan = task.plan
                invert = int(plan.parameter_map["invert"])
                labels = mechanism_labels(schema, database, plan)
                _gate, counts, silence = self._interaction_covariates(
                    schema, database, plan
                )
                engaged = (counts >= 2) & (silence < 0.5)
                dormant = (counts == 1) & (silence > 0.5)
                stats[invert]["engaged_pos"] += int(
                    np.sum(labels[engaged] == 1)
                )
                stats[invert]["engaged_total"] += int(np.sum(engaged))
                stats[invert]["dormant_pos"] += int(
                    np.sum(labels[dormant] == 1)
                )
                stats[invert]["dormant_total"] += int(np.sum(dormant))
                tasks += 1
        self.assertGreaterEqual(tasks, 24, "need enough interaction tasks")
        for invert in (0, 1):
            entry = stats[invert]
            self.assertGreater(
                entry["engaged_total"], 0,
                f"invert={invert}: need engaged rows",
            )
            self.assertGreater(
                entry["dormant_total"], 0,
                f"invert={invert}: need dormant rows",
            )
            engaged_rate = entry["engaged_pos"] / entry["engaged_total"]
            dormant_rate = entry["dormant_pos"] / entry["dormant_total"]
            if invert == 0:
                self.assertGreater(
                    engaged_rate,
                    dormant_rate,
                    f"invert=0: engaged {engaged_rate:.3f} should exceed "
                    f"dormant {dormant_rate:.3f}",
                )
            else:
                self.assertLess(
                    engaged_rate,
                    dormant_rate,
                    f"invert=1: engaged {engaged_rate:.3f} should fall below "
                    f"dormant {dormant_rate:.3f}",
                )

    def test_interaction_response_both_invert_variants_appear(self) -> None:
        """The mechanism must emit both response and ignore tasks."""
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=1,
                mechanism_weights=(
                    (TaskMechanism.INTERACTION_RESPONSE, 1.0),
                ),
                min_support_rows=8,
                min_query_rows=4,
                min_class_count_per_split=1,
                max_attempts_per_database=512,
            )
        )
        observed = 0
        inverts: set[int] = set()
        for index in range(60):
            sample_id = f"interaction_invert_{index}"
            runtime, schema, database = self._database(
                sample_id, min_tables=5, max_tables=7
            )
            try:
                tasks = planner.generate(
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    runtime=runtime.child("task"),
                )
            except ValueError:
                continue
            for task in tasks:
                inverts.add(int(task.plan.parameter_map["invert"]))
                observed += 1
            if observed >= 40:
                break
        self.assertGreaterEqual(observed, 40, "need enough interaction tasks")
        self.assertEqual(
            {0, 1},
            inverts,
            "both response (invert=0) and ignore (invert=1) variants must appear",
        )

    def test_interaction_response_labels_reproducible_from_plan(self) -> None:
        """Serializing a plan must not shift the replayed label stream."""
        checked = 0
        for index in range(12):
            sample_id = f"interaction_repro_{index}"
            runtime, schema, database = self._database(
                sample_id, min_tables=5, max_tables=7
            )
            candidates = interaction_candidates(schema, database)
            if not candidates:
                continue
            candidate = candidates[0]
            for seed in range(20):
                task = build_interaction_response_task(
                    task_id=f"interaction_repro_{index}_{seed}",
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    candidate=candidate,
                    seed=runtime.seed("interaction-repro", seed),
                    support_fraction=0.7,
                    min_support_rows=4,
                    min_query_rows=2,
                    min_class_count_per_split=1,
                    positive_rate_min=0.2,
                    positive_rate_max=0.8,
                    interaction_u_weight_min=0.25,
                    interaction_u_weight_max=2.0,
                    interaction_frequency_weight_min=0.5,
                    interaction_frequency_weight_max=3.0,
                    interaction_silence_weight_min=0.5,
                    interaction_silence_weight_max=3.0,
                    interaction_item_weight_min=0.5,
                    interaction_item_weight_max=3.0,
                    interaction_invert_probability=0.35,
                )
                if task is None:
                    continue
                plan = task.plan
                np.testing.assert_array_equal(
                    mechanism_labels(
                        schema,
                        database,
                        TaskPlan.from_dict(plan.to_dict()),
                    ),
                    mechanism_labels(schema, database, plan),
                )
                checked += 1
                if checked >= 8:
                    break
            if checked >= 8:
                break
        self.assertGreaterEqual(checked, 8, "need enough interaction tasks")

    def test_interaction_response_soft_positive_rate(self) -> None:
        """The requested rate is soft, but labels must remain non-saturated."""
        observed = 0
        soft = False
        for index in range(16):
            sample_id = f"interaction_soft_{index}"
            runtime, schema, database = self._database(
                sample_id, min_tables=5, max_tables=7
            )
            candidates = interaction_candidates(schema, database)
            if not candidates:
                continue
            candidate = candidates[0]
            for seed in range(30):
                task = build_interaction_response_task(
                    task_id=f"interaction_soft_{index}_{seed}",
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    candidate=candidate,
                    seed=runtime.seed("interaction-soft", seed),
                    support_fraction=0.7,
                    min_support_rows=4,
                    min_query_rows=2,
                    min_class_count_per_split=1,
                    positive_rate_min=0.2,
                    positive_rate_max=0.8,
                    interaction_u_weight_min=0.25,
                    interaction_u_weight_max=2.0,
                    interaction_frequency_weight_min=0.5,
                    interaction_frequency_weight_max=3.0,
                    interaction_silence_weight_min=0.5,
                    interaction_silence_weight_max=3.0,
                    interaction_item_weight_min=0.5,
                    interaction_item_weight_max=3.0,
                    interaction_invert_probability=0.35,
                )
                if task is None:
                    continue
                plan = task.plan
                self.assertGreater(
                    plan.realized_positive_rate,
                    0.05,
                    "interaction labels must not saturate at all-zero",
                )
                self.assertLess(
                    plan.realized_positive_rate,
                    0.95,
                    "interaction labels must not saturate at all-one",
                )
                observed += 1
                if plan.realized_positive_rate != plan.requested_positive_rate:
                    soft = True
                if observed >= 40:
                    break
            if observed >= 40:
                break
        self.assertGreaterEqual(observed, 40, "need enough interaction tasks")
        self.assertTrue(
            soft, "realized rate must deviate from the requested base rate"
        )

    def test_full_task_pipeline_honors_tasks_per_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            schema_result = generate_physical_schemas(
                SchemaPipelineConfig(
                    output_root=root / "schema",
                    num_schemas=2,
                    base_seed=71,
                    sampler=BlueprintSamplerConfig(min_tables=3, max_tables=4),
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
            serial_result = generate_tasks(
                TaskPipelineConfig(
                    instance_manifest=instance_result.manifest_path,
                    output_root=root / "task_serial",
                    num_workers=1,
                    planner=TaskPlannerConfig(
                        tasks_per_database=2,
                        mechanism_weights=(
                            (TaskMechanism.RELATION_ATTRIBUTE, 1.0),
                        ),
                        min_support_rows=8,
                        min_query_rows=4,
                    ),
                )
            )
            result = generate_tasks(
                TaskPipelineConfig(
                    instance_manifest=instance_result.manifest_path,
                    output_root=root / "task",
                    num_workers=2,
                    planner=TaskPlannerConfig(
                        tasks_per_database=2,
                        mechanism_weights=(
                            (TaskMechanism.RELATION_ATTRIBUTE, 1.0),
                        ),
                        min_support_rows=8,
                        min_query_rows=4,
                    ),
                )
            )

            self.assertEqual(2, result.database_count)
            self.assertEqual(4, result.task_count)
            self.assertEqual(serial_result.task_count, result.task_count)
            for serial_path, parallel_path in zip(
                serial_result.artifact_paths,
                result.artifact_paths,
            ):
                serial_task = load_task_artifact(serial_path)
                parallel_task = load_task_artifact(parallel_path)
                self.assertEqual(serial_task.task.plan, parallel_task.task.plan)
                np.testing.assert_array_equal(
                    serial_task.task.data.support_row_ids,
                    parallel_task.task.data.support_row_ids,
                )
                np.testing.assert_array_equal(
                    serial_task.task.data.query_row_ids,
                    parallel_task.task.data.query_row_ids,
                )
                np.testing.assert_array_equal(
                    serial_task.task.data.support_labels,
                    parallel_task.task.data.support_labels,
                )
                np.testing.assert_array_equal(
                    serial_task.task.data.query_labels,
                    parallel_task.task.data.query_labels,
                )
            manifest = json.loads(
                result.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(2, manifest["database_count"])
            self.assertEqual(4, manifest["task_count"])
            for artifact_path in result.artifact_paths:
                artifact = load_task_artifact(artifact_path)
                self.assertTrue(artifact.validation.is_valid)
                instance = load_instance_artifact(artifact.instance_artifact)
                self.assertEqual(
                    instance.database.instance_id,
                    artifact.task.plan.instance_id,
                )

    def test_all_mechanisms_emit_recomputable_exact_required_paths(self) -> None:
        # A small 6-8 table schema does not always admit every mechanism,
        # and the eligible set shifts whenever the motif library changes.
        # Audit the first deterministic sample that supports all mechanisms.
        chosen = None
        for suffix in range(32):
            sample_id = (
                "mechanism_route_audit"
                if suffix == 0
                else f"mechanism_route_audit_{suffix}"
            )
            runtime = RuntimeContext(991).for_sample(sample_id)
            blueprint = BlueprintSampler(
                BlueprintSamplerConfig(min_tables=6, max_tables=8)
            ).sample(sample_id, runtime)
            schema = PhysicalSchemaCompiler().compile(
                blueprint, sample_id, runtime
            )
            instance_plan = InstancePlanner(
                InstancePlannerConfig(
                    entity_rows_min=32,
                    entity_rows_max=48,
                    max_rows_per_table=160,
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
            tasks_by_mechanism = {}
            for mechanism in TaskMechanism:
                try:
                    tasks = TaskPlanner(
                        TaskPlannerConfig(
                            tasks_per_database=1,
                            mechanism_weights=((mechanism, 1.0),),
                            min_support_rows=8,
                            min_query_rows=4,
                            min_class_count_per_split=1,
                            max_attempts_per_database=512,
                        )
                    ).generate(
                        sample_id=sample_id,
                        schema=schema,
                        database=database,
                        runtime=runtime.child("task", mechanism.value),
                    )
                except ValueError:
                    break
                tasks_by_mechanism[mechanism] = tasks[0]
            if len(tasks_by_mechanism) == len(TaskMechanism):
                chosen = (schema, database, tasks_by_mechanism)
                break

        self.assertIsNotNone(
            chosen, "no sampled database supported all mechanisms"
        )
        schema, database, tasks_by_mechanism = chosen

        for mechanism in TaskMechanism:
            task = tasks_by_mechanism[mechanism]
            required = [
                label
                for label in task.plan.route_supervision
                if label.role is RouteRole.REQUIRED
            ]
            if mechanism is TaskMechanism.RANDOM_COLUMN:
                self.assertFalse(required, mechanism.value)
            else:
                self.assertTrue(required, mechanism.value)
            expected = mechanism_labels(schema, database, task.plan)
            np.testing.assert_array_equal(
                task.data.support_labels,
                expected[task.data.support_row_ids],
            )
            np.testing.assert_array_equal(
                task.data.query_labels,
                expected[task.data.query_row_ids],
            )
            view = build_task_view(schema, database, task.plan)
            target_mask = view.row_masks[task.plan.target_table_id]
            self.assertTrue(
                np.all(target_mask[task.data.support_row_ids])
            )
            self.assertTrue(
                np.all(target_mask[task.data.query_row_ids])
            )
            self.assertTrue(validate_task(schema, database, task).is_valid)

    def test_task_cutoff_and_horizon_within_calendar_interval(self) -> None:
        generated: list[TaskPlan] = []
        for suffix in range(8):
            sample_id = (
                "task_calendar_bounds"
                if suffix == 0
                else f"task_calendar_bounds_{suffix}"
            )
            runtime = RuntimeContext(404).for_sample(sample_id)
            blueprint = BlueprintSampler(
                BlueprintSamplerConfig(min_tables=4, max_tables=4)
            ).sample(sample_id, runtime)
            schema = PhysicalSchemaCompiler().compile(
                blueprint, sample_id, runtime
            )
            plan = InstancePlanner(
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
            database = DatabaseGenerator().generate(schema=schema, plan=plan)
            try:
                tasks = TaskPlanner(
                    TaskPlannerConfig(
                        tasks_per_database=2,
                        mechanism_weights=(
                            (TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE, 1.0),
                        ),
                        min_support_rows=8,
                        min_query_rows=4,
                        min_class_count_per_split=1,
                        max_attempts_per_database=512,
                    )
                ).generate(
                    sample_id=sample_id,
                    schema=schema,
                    database=database,
                    runtime=runtime.child("task"),
                    instance_plan=plan,
                )
            except ValueError:
                continue
            for task in tasks:
                task_plan = task.plan
                self.assertEqual(
                    task_plan.db_start_seconds,
                    plan.calendar_start_seconds,
                )
                self.assertEqual(
                    task_plan.db_end_seconds,
                    plan.calendar_end_seconds,
                )
                for name in ("cutoff_time", "horizon_end_time"):
                    value = getattr(task_plan, name)
                    if value is not None:
                        self.assertGreaterEqual(
                            value, task_plan.db_start_seconds
                        )
                        self.assertLessEqual(value, task_plan.db_end_seconds)
                for rule in task_plan.observation_rules:
                    self.assertGreaterEqual(
                        rule.max_timestamp, task_plan.db_start_seconds
                    )
                    self.assertLessEqual(
                        rule.max_timestamp, task_plan.db_end_seconds
                    )
                generated.append(task_plan)
            if len(generated) >= 3:
                break
        self.assertTrue(generated)

    def test_calendar_binding_rejects_invalid_candidate_without_clipping(self) -> None:
        runtime, schema, database = self._database("calendar_binding")
        instance_plan = InstancePlanner(
            InstancePlannerConfig(
                entity_rows_min=32,
                entity_rows_max=40,
                lookup_rows_min=3,
                lookup_rows_max=5,
                max_rows_per_table=128,
            )
        ).plan(
            sample_id="calendar_binding",
            schema=schema,
            runtime=runtime.child("database-instance"),
        )
        task = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=1,
                mechanism_weights=((TaskMechanism.RELATION_ATTRIBUTE, 1.0),),
                min_support_rows=8,
                min_query_rows=4,
            )
        ).generate(
            sample_id="calendar_binding",
            schema=schema,
            database=database,
            runtime=runtime.child("task"),
        )[0]

        invalid = replace(
            task,
            plan=replace(task.plan, cutoff_time=0),
        )
        self.assertIsNone(_bind_instance_calendar(invalid, instance_plan))


if __name__ == "__main__":
    unittest.main()
