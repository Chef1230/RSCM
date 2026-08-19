"""Composite relational-classification task tests.

Covers the serializable DSL, the generic aggregate executor, every composite
family, the planner/config wiring, backward compatibility and the
no-leakage guarantees (post-cutoff data may drive labels but never the
observation view).
"""

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

from rdb_prior.compilation.model import ColumnKind, PhysicalSchema
from rdb_prior.compilation.compiler import PhysicalSchemaCompiler
from rdb_prior.config import load_task_pipeline_config
from rdb_prior.generation.database import DatabaseGenerator
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.instance.planner import InstancePlanner, InstancePlannerConfig
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.schema.spec import TableRole
from rdb_prior.task.mechanisms import (
    _evaluate_aggregate_spec,
    _evaluate_composite_scores,
    _observation_rules,
    _schema_route_labels,
    _traverse_path,
    _SYNTHETIC_TARGET,
    build_composite_relational_classification_task,
    composite_labels,
    generate_composite_candidates,
    mechanism_labels,
)
from rdb_prior.task.model import (
    AggregateOperator,
    AggregateSpec,
    CombineOperator,
    CompareOperator,
    CompositeFamily,
    CompositeTaskSpec,
    LabelOperator,
    PredicateSpec,
    RoutePathLabel,
    RouteRole,
    TaskMechanism,
    TaskPlan,
)
from rdb_prior.task.planner import TaskPlanner, TaskPlannerConfig
from rdb_prior.task.validation import validate_task
from rdb_prior.task.view import build_task_view


def _database(sample_id: str):
    runtime = RuntimeContext(303).for_sample(sample_id)
    blueprint = BlueprintSampler(
        BlueprintSamplerConfig(min_tables=4, max_tables=6)
    ).sample(sample_id, runtime)
    schema = PhysicalSchemaCompiler().compile(blueprint, sample_id, runtime)
    plan = InstancePlanner(
        InstancePlannerConfig(
            entity_rows_min=60,
            entity_rows_max=90,
            lookup_rows_min=4,
            lookup_rows_max=6,
            max_rows_per_table=220,
        )
    ).plan(
        sample_id=sample_id,
        schema=schema,
        runtime=runtime.child("db"),
    )
    database = DatabaseGenerator().generate(schema=schema, plan=plan)
    return runtime, schema, database


def _event_time_column(table) -> str | None:
    return next(
        (
            column.column_id
            for column in table.columns
            if column.kind is ColumnKind.TIME
        ),
        None,
    )


def _find_entity_event(schema):
    """First ENTITY -> EVENT edge where the event has a TIME column."""
    for foreign_key in schema.foreign_keys:
        parent = schema.table(foreign_key.parent_table_id)
        child = schema.table(foreign_key.child_table_id)
        if parent.role is not TableRole.ENTITY:
            continue
        if child.role is not TableRole.EVENT:
            continue
        time_column = _event_time_column(child)
        if time_column is not None:
            return (
                parent.table_id,
                child.table_id,
                foreign_key.foreign_key_id,
                time_column,
            )
    return None


def _find_entity_two_events(schema):
    """An ENTITY with two EVENT children, both carrying a TIME column."""
    for entity in schema.tables:
        if entity.role is not TableRole.ENTITY:
            continue
        children: list[tuple[str, str, str]] = []
        for foreign_key in schema.foreign_keys:
            if foreign_key.parent_table_id != entity.table_id:
                continue
            child = schema.table(foreign_key.child_table_id)
            if child.role is not TableRole.EVENT:
                continue
            time_column = _event_time_column(child)
            if time_column is not None:
                children.append(
                    (
                        child.table_id,
                        foreign_key.foreign_key_id,
                        time_column,
                    )
                )
        if len(children) >= 2:
            return entity.table_id, children[0], children[1]
    return None


def _find_two_hop(schema):
    """An ENTITY -> EVENT -> EVENT/DETAIL path of length two."""
    for first in schema.foreign_keys:
        parent = schema.table(first.parent_table_id)
        if parent.role is not TableRole.ENTITY:
            continue
        middle = schema.table(first.child_table_id)
        if middle.role not in {TableRole.EVENT, TableRole.DETAIL}:
            continue
        for second in schema.foreign_keys:
            if second.parent_table_id != middle.table_id:
                continue
            endpoint = schema.table(second.child_table_id)
            if endpoint.role not in {TableRole.EVENT, TableRole.DETAIL}:
                continue
            time_column = _event_time_column(endpoint)
            if time_column is not None:
                return (
                    parent.table_id,
                    (first.foreign_key_id, second.foreign_key_id),
                    endpoint.table_id,
                    time_column,
                )
    return None


def _hand_plan(
    *,
    schema: PhysicalSchema,
    database: DatabaseInstance,
    spec: CompositeTaskSpec,
    cutoff: int,
    target_table_id: str,
    source_table_id: str,
    time_column_id: str,
    path: tuple[str, ...],
    seed: int = 7,
) -> TaskPlan:
    return TaskPlan(
        task_id="composite_test_task",
        sample_id=database.instance_id,
        instance_id=database.instance_id,
        schema_id=schema.schema_id,
        mechanism=TaskMechanism.RELATIONAL_CLASSIFICATION,
        prediction_type=_prediction_type(),
        target_table_id=target_table_id,
        source_table_id=source_table_id,
        target_column_id=_SYNTHETIC_TARGET,
        source_column_id=None,
        time_column_id=time_column_id,
        cutoff_time=cutoff,
        split_strategy="stratified_rows",
        seed=seed,
        masked_column_ids=(_SYNTHETIC_TARGET,),
        observation_rules=_observation_rules(schema, cutoff),
        route_supervision=_schema_route_labels(
            schema,
            target_table_id=target_table_id,
            required_paths=(path,),
        ),
        classification_kind=_binary_kind(),
        composite_spec=spec,
    )


def _prediction_type():
    from rdb_prior.task.model import PredictionType

    return PredictionType.CLASSIFICATION


def _binary_kind():
    from rdb_prior.task.model import ClassificationKind

    return ClassificationKind.BINARY


class CompositeDslTests(unittest.TestCase):
    def test_dsl_json_round_trip(self) -> None:
        predicate = PredicateSpec(
            column_id="cat",
            operator=CompareOperator.EQ,
            value="active",
        )
        aggregate = AggregateSpec(
            source_table_id="events",
            required_path=("fk1",),
            time_column_id="ts",
            window_start=-3600,
            window_end=0,
            operator=AggregateOperator.SUM,
            value_column_id="amount",
            predicates=(predicate,),
        )
        eligibility = AggregateSpec(
            source_table_id="events",
            required_path=("fk1",),
            time_column_id="ts",
            window_start=-86400,
            window_end=0,
            operator=AggregateOperator.COUNT,
        )
        spec = CompositeTaskSpec(
            family=CompositeFamily.HISTORY_CONDITIONED_FUTURE,
            label_aggregates=(aggregate,),
            combine_operator=CombineOperator.SUM,
            label_operator=LabelOperator.GT,
            label_threshold=1.5,
            eligibility_aggregate=eligibility,
            eligibility_operator=LabelOperator.GT,
            eligibility_threshold=0.0,
        )
        self.assertEqual(spec, CompositeTaskSpec.from_dict(spec.to_dict()))
        self.assertEqual(
            spec.canonical(),
            CompositeTaskSpec.from_dict(spec.to_dict()).canonical(),
        )
        self.assertEqual(
            aggregate, AggregateSpec.from_dict(aggregate.to_dict())
        )
        self.assertEqual(
            predicate, PredicateSpec.from_dict(predicate.to_dict())
        )

    def test_old_task_plan_json_is_backward_compatible(self) -> None:
        # A complete pre-composite plan (no composite_spec key) must still load.
        payload = {
            "task_id": "t1",
            "sample_id": "s1",
            "instance_id": "i1",
            "schema_id": "sc1",
            "mechanism": "relation_attribute",
            "prediction_type": "classification",
            "target_table_id": "N0",
            "source_table_id": "N1",
            "target_column_id": "col",
            "masked_column_ids": ["col"],
            "split_strategy": "stratified_rows",
            "seed": 0,
        }
        plan = TaskPlan.from_dict(payload)
        self.assertIsNone(plan.composite_spec)
        self.assertEqual(payload["mechanism"], plan.mechanism.value)
        # A dict that round-trips through to_dict also keeps composite None.
        self.assertIsNone(TaskPlan.from_dict(plan.to_dict()).composite_spec)

    def test_count_needs_no_value_but_numeric_aggregates_do(self) -> None:
        AggregateSpec(
            source_table_id="e",
            required_path=("fk",),
            time_column_id="ts",
            window_start=-10,
            window_end=0,
            operator=AggregateOperator.COUNT,
        )
        with self.assertRaises(ValueError):
            AggregateSpec(
                source_table_id="e",
                required_path=("fk",),
                time_column_id="ts",
                window_start=-10,
                window_end=0,
                operator=AggregateOperator.SUM,
            )
        with self.assertRaises(ValueError):
            AggregateSpec(
                source_table_id="e",
                required_path=("fk",),
                time_column_id="ts",
                window_start=-10,
                window_end=0,
                operator=AggregateOperator.COUNT,
                value_column_id="v",
            )

    def test_composite_plan_round_trip_preserves_spec(self) -> None:
        spec = CompositeTaskSpec(
            family=CompositeFamily.FILTERED_AGGREGATE,
            label_aggregates=(
                AggregateSpec(
                    source_table_id="e",
                    required_path=("fk",),
                    time_column_id="ts",
                    window_start=0,
                    window_end=100,
                    operator=AggregateOperator.COUNT,
                ),
            ),
            combine_operator=CombineOperator.SUM,
            label_operator=LabelOperator.GT,
            label_threshold=0.0,
        )
        plan = TaskPlan(
            task_id="t", sample_id="s", instance_id="i", schema_id="sc",
            mechanism=TaskMechanism.RELATIONAL_CLASSIFICATION,
            prediction_type=_prediction_type(),
            target_table_id="N0", source_table_id="e",
            target_column_id=_SYNTHETIC_TARGET,
            time_column_id="ts", cutoff_time=10,
            split_strategy="stratified_rows", seed=1,
            masked_column_ids=(_SYNTHETIC_TARGET,),
            route_supervision=(
                RoutePathLabel(foreign_key_ids=("fk",), role=RouteRole.REQUIRED),
            ),
            classification_kind=_binary_kind(),
            composite_spec=spec,
        )
        restored = TaskPlan.from_dict(plan.to_dict())
        self.assertEqual(plan, restored)
        self.assertEqual(spec, restored.composite_spec)


class CompositeExecutorTests(unittest.TestCase):
    def test_filtered_aggregate_count_labels(self) -> None:
        for index in range(30):
            sample_id = f"filtered_agg_{index}"
            runtime, schema, database = _database(sample_id)
            pair = _find_entity_event(schema)
            if pair is None:
                continue
            entity_id, event_id, fk_id, event_time = pair
            times = database.table(event_id).column(event_time)
            cutoff = int(np.quantile(times, 0.6))
            horizon = max(1, int((int(times.max()) - cutoff) / 2))
            window = AggregateSpec(
                source_table_id=event_id,
                required_path=(fk_id,),
                time_column_id=event_time,
                window_start=0,
                window_end=horizon,
                operator=AggregateOperator.COUNT,
            )
            cutoffs = np.full(
                database.table(entity_id).row_count, cutoff, dtype=np.int64
            )
            values = _evaluate_aggregate_spec(
                schema, database, window, entity_id, cutoffs
            )
            assignments = database.table(event_id).column(
                _fk_child_column(schema, fk_id)
            )
            expected = _independent_count(
                assignments,
                times,
                database.table(entity_id).row_count,
                cutoff,
                cutoff + horizon,
            )
            np.testing.assert_array_equal(values, expected)

            spec = CompositeTaskSpec(
                family=CompositeFamily.FILTERED_AGGREGATE,
                label_aggregates=(window,),
                combine_operator=CombineOperator.SUM,
                label_operator=LabelOperator.GT,
                label_threshold=0.0,
            )
            plan = _hand_plan(
                schema=schema, database=database, spec=spec, cutoff=cutoff,
                target_table_id=entity_id, source_table_id=event_id,
                time_column_id=event_time, path=(fk_id,),
            )
            labels = mechanism_labels(schema, database, plan)
            np.testing.assert_array_equal(labels, (expected > 0).astype(np.int8))
            return
        self.fail("no entity/event pair found in 30 databases")

    def test_count_distinct_differs_from_count(self) -> None:
        for index in range(30):
            sample_id = f"count_distinct_{index}"
            runtime, schema, database = _database(sample_id)
            pair = _find_entity_event(schema)
            if pair is None:
                continue
            entity_id, event_id, fk_id, event_time = pair
            value_column = next(
                (
                    column.column_id
                    for column in schema.table(event_id).columns
                    if column.kind is ColumnKind.FEATURE
                ),
                None,
            )
            if value_column is None:
                continue
            times = database.table(event_id).column(event_time)
            cutoff = int(np.quantile(times, 0.5))
            window = max(1, int((int(times.max()) - int(times.min())) / 4))
            kwargs = dict(
                source_table_id=event_id,
                required_path=(fk_id,),
                time_column_id=event_time,
                window_start=0,
                window_end=window,
            )
            cutoffs = np.full(
                database.table(entity_id).row_count, cutoff, dtype=np.int64
            )
            count_values = _evaluate_aggregate_spec(
                schema, database,
                AggregateSpec(**kwargs, operator=AggregateOperator.COUNT),
                entity_id, cutoffs,
            )
            distinct_values = _evaluate_aggregate_spec(
                schema, database,
                AggregateSpec(**kwargs, operator=AggregateOperator.COUNT_DISTINCT, value_column_id=value_column),
                entity_id, cutoffs,
            )
            # COUNT_DISTINCT never exceeds COUNT and every multi-row window
            # with at least two distinct values shows a strict difference.
            self.assertTrue(np.all(distinct_values <= count_values))
            if np.any((count_values >= 2) & (distinct_values < count_values)):
                return
        self.fail("no duplicate-valued window found in 30 databases")

    def test_quantified_event_is_filtered_count_gt_zero(self) -> None:
        for index in range(30):
            sample_id = f"quantified_{index}"
            runtime, schema, database = _database(sample_id)
            pair = _find_entity_event(schema)
            if pair is None:
                continue
            entity_id, event_id, fk_id, event_time = pair
            feature_columns = [
                column.column_id
                for column in schema.table(event_id).columns
                if column.kind is ColumnKind.FEATURE
            ]
            if not feature_columns:
                continue
            predicate_column = feature_columns[0]
            raw = database.table(event_id).column(predicate_column)
            observed = raw[raw == raw]
            if not len(observed):
                continue
            unique = np.unique(observed)
            target = unique[0]
            predicate = PredicateSpec(
                column_id=predicate_column,
                operator=CompareOperator.EQ,
                value=str(target) if raw.dtype.kind in {"U", "S"} else float(target),
            )
            times = database.table(event_id).column(event_time)
            cutoff = int(np.quantile(times, 0.5))
            horizon = max(1, int((int(times.max()) - cutoff) / 2))
            window = AggregateSpec(
                source_table_id=event_id,
                required_path=(fk_id,),
                time_column_id=event_time,
                window_start=0,
                window_end=horizon,
                operator=AggregateOperator.COUNT,
                predicates=(predicate,),
            )
            cutoffs = np.full(
                database.table(entity_id).row_count, cutoff, dtype=np.int64
            )
            filtered_count = _evaluate_aggregate_spec(
                schema, database, window, entity_id, cutoffs
            )
            spec = CompositeTaskSpec(
                family=CompositeFamily.QUANTIFIED_EVENT,
                label_aggregates=(window,),
                combine_operator=CombineOperator.SUM,
                label_operator=LabelOperator.GT,
                label_threshold=0.0,
            )
            plan = _hand_plan(
                schema=schema, database=database, spec=spec, cutoff=cutoff,
                target_table_id=entity_id, source_table_id=event_id,
                time_column_id=event_time, path=(fk_id,),
            )
            labels = mechanism_labels(schema, database, plan)
            expected = (filtered_count > 0).astype(np.int8)
            np.testing.assert_array_equal(labels, expected)
            return
        self.fail("no quantified-event case found in 30 databases")

    def test_multi_source_combines_two_event_tables(self) -> None:
        for index in range(30):
            sample_id = f"multi_source_{index}"
            runtime, schema, database = _database(sample_id)
            found = _find_entity_two_events(schema)
            if found is None:
                continue
            entity_id, (a_id, a_fk, a_time), (b_id, b_fk, b_time) = found
            times_a = database.table(a_id).column(a_time)
            times_b = database.table(b_id).column(b_time)
            cutoff = int(np.quantile(np.concatenate([times_a, times_b]), 0.5))
            horizon = max(1, int((int(times_a.max()) - cutoff) / 2))
            agg_a = AggregateSpec(
                source_table_id=a_id, required_path=(a_fk,),
                time_column_id=a_time, window_start=0, window_end=horizon,
                operator=AggregateOperator.COUNT,
            )
            agg_b = AggregateSpec(
                source_table_id=b_id, required_path=(b_fk,),
                time_column_id=b_time, window_start=0, window_end=horizon,
                operator=AggregateOperator.COUNT,
            )
            cutoffs = np.full(
                database.table(entity_id).row_count, cutoff, dtype=np.int64
            )
            spec = CompositeTaskSpec(
                family=CompositeFamily.MULTI_SOURCE,
                label_aggregates=(agg_a, agg_b),
                combine_operator=CombineOperator.SUM,
                label_operator=LabelOperator.GT,
                label_threshold=0.0,
            )
            scores = _evaluate_composite_scores(
                schema, database, spec, entity_id, cutoffs
            )
            count_a = _evaluate_aggregate_spec(
                schema, database, agg_a, entity_id, cutoffs
            )
            count_b = _evaluate_aggregate_spec(
                schema, database, agg_b, entity_id, cutoffs
            )
            np.testing.assert_array_equal(scores, count_a + count_b)
            return
        self.fail("no two-event entity found in 30 databases")

    def test_multi_hop_filtered_traverses_two_hops(self) -> None:
        for index in range(30):
            sample_id = f"multi_hop_{index}"
            runtime, schema, database = _database(sample_id)
            found = _find_two_hop(schema)
            if found is None:
                continue
            entity_id, path, endpoint_id, endpoint_time = found
            times = database.table(endpoint_id).column(endpoint_time)
            cutoff = int(np.quantile(times, 0.5))
            horizon = max(1, int((int(times.max()) - cutoff) / 2))
            aggregate = AggregateSpec(
                source_table_id=endpoint_id,
                required_path=path,
                time_column_id=endpoint_time,
                window_start=0,
                window_end=horizon,
                operator=AggregateOperator.COUNT,
            )
            cutoffs = np.full(
                database.table(entity_id).row_count, cutoff, dtype=np.int64
            )
            values = _evaluate_aggregate_spec(
                schema, database, aggregate, entity_id, cutoffs
            )
            expected = _independent_two_hop_count(
                schema, database, path, entity_id, endpoint_id,
                cutoff, cutoff + horizon,
            )
            np.testing.assert_array_equal(values, expected)
            # Two-hop path endpoint must be reachable via _traverse_path.
            _row_sets, endpoint = _traverse_path(
                schema, database, entity_id, path
            )
            self.assertEqual(endpoint_id, endpoint)
            return
        self.fail("no two-hop path found in 30 databases")

    def test_history_conditioned_future_excludes_no_history_entities(self) -> None:
        for index in range(30):
            sample_id = f"history_conditioned_{index}"
            runtime, schema, database = _database(sample_id)
            pair = _find_entity_event(schema)
            if pair is None:
                continue
            entity_id, event_id, fk_id, event_time = pair
            times = database.table(event_id).column(event_time)
            cutoff = int(np.quantile(times, 0.5))
            horizon = max(1, int((int(times.max()) - cutoff) / 2))
            lookback = max(1, int((cutoff - int(times.min())) / 2))
            future = AggregateSpec(
                source_table_id=event_id, required_path=(fk_id,),
                time_column_id=event_time, window_start=0, window_end=horizon,
                operator=AggregateOperator.COUNT,
            )
            history = AggregateSpec(
                source_table_id=event_id, required_path=(fk_id,),
                time_column_id=event_time, window_start=-lookback, window_end=0,
                operator=AggregateOperator.COUNT,
            )
            spec = CompositeTaskSpec(
                family=CompositeFamily.HISTORY_CONDITIONED_FUTURE,
                label_aggregates=(future,),
                combine_operator=CombineOperator.SUM,
                label_operator=LabelOperator.GT,
                label_threshold=0.0,
                eligibility_aggregate=history,
                eligibility_operator=LabelOperator.GT,
                eligibility_threshold=0.0,
            )
            plan = _hand_plan(
                schema=schema, database=database, spec=spec, cutoff=cutoff,
                target_table_id=entity_id, source_table_id=event_id,
                time_column_id=event_time, path=(fk_id,),
            )
            labels = mechanism_labels(schema, database, plan)
            cutoffs = np.full(
                database.table(entity_id).row_count, cutoff, dtype=np.int64
            )
            history_counts = _evaluate_aggregate_spec(
                schema, database, history, entity_id, cutoffs
            )
            eligible = history_counts > 0
            self.assertTrue(np.all(labels[~eligible] == -1))
            future_counts = _evaluate_aggregate_spec(
                schema, database, future, entity_id, cutoffs
            )
            np.testing.assert_array_equal(
                labels[eligible], (future_counts[eligible] > 0).astype(np.int8)
            )
            return
        self.fail("no history-conditioned case found in 30 databases")

    def test_builder_labels_recompute_exactly(self) -> None:
        for index in range(20):
            sample_id = f"builder_recompute_{index}"
            runtime, schema, database = _database(sample_id)
            candidates = generate_composite_candidates(
                schema, database,
                families=tuple(CompositeFamily),
                max_path_depth=3,
                candidate_limit=30,
            )
            for candidate in candidates:
                task = build_composite_relational_classification_task(
                    task_id=f"task_{sample_id}", sample_id=sample_id,
                    schema=schema, database=database, candidate=candidate,
                    seed=7, support_fraction=0.7, min_support_rows=8,
                    min_query_rows=4, min_class_count_per_split=1,
                    positive_rate_min=0.2, positive_rate_max=0.8,
                    max_predicates=2,
                )
                if task is None:
                    continue
                expected = composite_labels(schema, database, task.plan)
                np.testing.assert_array_equal(
                    expected, mechanism_labels(schema, database, task.plan)
                )
                np.testing.assert_array_equal(
                    task.data.support_labels,
                    expected[task.data.support_row_ids],
                )
                np.testing.assert_array_equal(
                    task.data.query_labels,
                    expected[task.data.query_row_ids],
                )
                self.assertTrue(
                    np.all(task.data.support_labels >= 0)
                    and np.all(task.data.query_labels >= 0)
                )
                self.assertTrue(validate_task(schema, database, task).is_valid)
                # Round trip keeps the spec that reproduced the labels.
                plan_roundtrip = TaskPlan.from_dict(task.plan.to_dict())
                np.testing.assert_array_equal(
                    mechanism_labels(schema, database, plan_roundtrip),
                    expected,
                )
                return
        self.fail("no composite task built in 20 databases")


class CompositeLeakageTests(unittest.TestCase):
    def _future_count_task(self, index: int):
        sample_id = f"leakage_{index}"
        runtime, schema, database = _database(sample_id)
        pair = _find_entity_event(schema)
        if pair is None:
            return None
        entity_id, event_id, fk_id, event_time = pair
        times = database.table(event_id).column(event_time)
        cutoff = int(np.quantile(times, 0.5))
        horizon = max(1, int((int(times.max()) - cutoff) / 2))
        future = AggregateSpec(
            source_table_id=event_id, required_path=(fk_id,),
            time_column_id=event_time, window_start=0, window_end=horizon,
            operator=AggregateOperator.COUNT,
        )
        spec = CompositeTaskSpec(
            family=CompositeFamily.QUANTIFIED_EVENT,
            label_aggregates=(future,),
            combine_operator=CombineOperator.SUM,
            label_operator=LabelOperator.GT,
            label_threshold=0.0,
        )
        plan = _hand_plan(
            schema=schema, database=database, spec=spec, cutoff=cutoff,
            target_table_id=entity_id, source_table_id=event_id,
            time_column_id=event_time, path=(fk_id,),
        )
        return schema, database, plan, event_id, event_time, cutoff, horizon

    def test_view_cuts_post_cutoff_rows_and_tracks_visible_changes(self) -> None:
        for index in range(30):
            found = self._future_count_task(index)
            if found is None:
                continue
            schema, database, plan, event_id, event_time, cutoff, _horizon = found
            view = build_task_view(schema, database, plan)
            visible = view.visible_rows(event_id)
            times = database.table(event_id).column(event_time)
            # Every visible event row lies at or before the cutoff.
            self.assertTrue(np.all(times[visible] <= cutoff))
            self.assertTrue(np.any(times > cutoff))
            # A visible row pushed past the cutoff leaves the view.
            hidden_after = int(np.flatnonzero(times > cutoff)[0])
            candidate = np.flatnonzero(times <= cutoff)
            if not len(candidate):
                continue
            row = int(candidate[0])
            times[row] = times[hidden_after] + 1
            view2 = build_task_view(schema, database, plan)
            self.assertNotIn(row, view2.visible_rows(event_id).tolist())
            return
        self.fail("no leakage-view case found in 30 databases")

    def test_between_cutoff_and_horizon_changes_labels_not_view(self) -> None:
        for index in range(30):
            found = self._future_count_task(index)
            if found is None:
                continue
            schema, database, plan, event_id, event_time, cutoff, horizon = found
            times = database.table(event_id).column(event_time)
            assignments = database.table(event_id).column(
                _fk_child_column(schema, plan.route_supervision[0].foreign_key_ids[0])
            )
            in_window = np.flatnonzero(
                (times > cutoff) & (times <= cutoff + horizon)
            )
            if not len(in_window):
                continue
            # Choose an entity with exactly one in-window row so removing it
            # flips that entity's count from 1 to 0 and therefore its label.
            in_window_counts = np.bincount(
                assignments[in_window].astype(np.int64)
            )
            single_entity = np.flatnonzero(in_window_counts == 1)
            rows_for_single = [
                int(row)
                for row in in_window
                if int(assignments[row]) in single_entity
            ]
            if not rows_for_single:
                continue
            row = rows_for_single[0]
            before = mechanism_labels(schema, database, plan)
            view_before = build_task_view(schema, database, plan)
            # Move the row beyond the horizon: it leaves the label window but
            # was already hidden from the observation view.
            times[row] = cutoff + horizon + 1000
            after = mechanism_labels(schema, database, plan)
            self.assertFalse(np.array_equal(before, after))
            view_after = build_task_view(schema, database, plan)
            for table_id in view_before.row_masks:
                np.testing.assert_array_equal(
                    view_before.row_masks[table_id],
                    view_after.row_masks[table_id],
                )
            return
        self.fail("no in-window row found in 30 databases")

    def test_after_horizon_changes_do_not_change_labels(self) -> None:
        for index in range(30):
            found = self._future_count_task(index)
            if found is None:
                continue
            schema, database, plan, event_id, event_time, cutoff, horizon = found
            times = database.table(event_id).column(event_time)
            beyond = np.flatnonzero(times > cutoff + horizon)
            if not len(beyond):
                continue
            row = int(beyond[0])
            before = mechanism_labels(schema, database, plan)
            times[row] = times[row] + 100000
            after = mechanism_labels(schema, database, plan)
            np.testing.assert_array_equal(before, after)
            return
        self.fail("no beyond-horizon row found in 30 databases")


class CompositePlannerConfigTests(unittest.TestCase):
    def test_planner_generates_full_count_with_composite_only(self) -> None:
        for index in range(30):
            sample_id = f"planner_composite_{index}"
            runtime, schema, database = _database(sample_id)
            planner = TaskPlanner(
                TaskPlannerConfig(
                    tasks_per_database=4,
                    mechanism_weights=(
                        (TaskMechanism.RELATIONAL_CLASSIFICATION, 1.0),
                    ),
                    min_support_rows=8,
                    min_query_rows=4,
                    min_class_count_per_split=1,
                    max_attempts_per_database=512,
                )
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
            self.assertEqual(4, len(tasks))
            for task in tasks:
                self.assertIs(
                    TaskMechanism.RELATIONAL_CLASSIFICATION,
                    task.plan.mechanism,
                )
                self.assertIsNotNone(task.plan.composite_spec)
                self.assertTrue(validate_task(schema, database, task).is_valid)
                np.testing.assert_array_equal(
                    mechanism_labels(schema, database, task.plan),
                    composite_labels(schema, database, task.plan),
                )
            return
        self.fail("composite-only planner failed in 30 databases")

    def test_fixed_seed_is_fully_deterministic(self) -> None:
        runtime, schema, database = _database("determinism")
        planner = TaskPlanner(
            TaskPlannerConfig(
                tasks_per_database=3,
                mechanism_weights=(
                    (TaskMechanism.RELATIONAL_CLASSIFICATION, 1.0),
                ),
                min_support_rows=8,
                min_query_rows=4,
                min_class_count_per_split=1,
                max_attempts_per_database=512,
            )
        )
        first = planner.generate(
            sample_id="determinism",
            schema=schema,
            database=database,
            runtime=runtime.child("task"),
        )
        second = planner.generate(
            sample_id="determinism",
            schema=schema,
            database=database,
            runtime=runtime.child("task"),
        )
        self.assertEqual(len(first), len(second))
        for task_a, task_b in zip(first, second):
            self.assertEqual(task_a.plan, task_b.plan)
            np.testing.assert_array_equal(
                task_a.data.support_labels, task_b.data.support_labels
            )
            np.testing.assert_array_equal(
                task_a.data.query_labels, task_b.data.query_labels
            )
            np.testing.assert_array_equal(
                task_a.data.support_row_ids, task_b.data.support_row_ids
            )

    def test_config_rejects_duplicate_or_non_positive_families(self) -> None:
        with self.assertRaises(ValueError):
            TaskPlannerConfig(
                composite_family_weights=(
                    (CompositeFamily.FILTERED_AGGREGATE, 1.0),
                    (CompositeFamily.FILTERED_AGGREGATE, 2.0),
                )
            )
        with self.assertRaises(ValueError):
            TaskPlannerConfig(
                composite_family_weights=(
                    (CompositeFamily.FILTERED_AGGREGATE, 0.0),
                )
            )

    def test_config_loader_rejects_unknown_family_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                "config_version: 1\n"
                "seed: 42\n"
                "paths:\n"
                "  output_root: outputs/bad\n"
                "task:\n"
                "  composite_family_weights:\n"
                "    not_a_family: 1.0\n",
                encoding="utf-8",
            )
            from rdb_prior.config import SchemaConfigError

            with self.assertRaises(SchemaConfigError):
                load_task_pipeline_config(path)


def _fk_child_column(schema, fk_id: str) -> str:
    for foreign_key in schema.foreign_keys:
        if foreign_key.foreign_key_id == fk_id:
            return foreign_key.child_column_id
    raise AssertionError(f"unknown FK {fk_id}")


def _independent_count(
    assignments: np.ndarray,
    times: np.ndarray,
    entity_count: int,
    lower: int,
    upper: int,
) -> np.ndarray:
    result = np.zeros(entity_count, dtype=np.float64)
    for entity in range(entity_count):
        result[entity] = np.count_nonzero(
            (assignments == entity)
            & (times >= lower)
            & (times <= upper)
        )
    return result


def _independent_two_hop_count(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    path: tuple[str, ...],
    entity_id: str,
    endpoint_id: str,
    lower: int,
    upper: int,
) -> np.ndarray:
    first, second = path
    first_fk = _foreign_key(schema, first)
    second_fk = _foreign_key(schema, second)
    middle = (
        first_fk.child_table_id
        if first_fk.child_table_id != entity_id
        else first_fk.parent_table_id
    )
    first_assign = database.table(first_fk.child_table_id).column(
        first_fk.child_column_id
    )
    second_assign = database.table(second_fk.child_table_id).column(
        second_fk.child_column_id
    )
    endpoint_times = database.table(endpoint_id).column(
        _event_time_column(schema.table(endpoint_id))
    )
    result = np.zeros(database.table(entity_id).row_count, dtype=np.float64)
    middle_ids = database.table(middle).column(
        schema.table(middle).primary_key.column_id
    )
    for entity in range(database.table(entity_id).row_count):
        middle_rows = np.flatnonzero(first_assign == entity)
        middle_values = middle_ids[middle_rows]
        endpoint_rows = np.flatnonzero(
            np.isin(second_assign, middle_values)
        )
        result[entity] = np.count_nonzero(
            (endpoint_times[endpoint_rows] >= lower)
            & (endpoint_times[endpoint_rows] <= upper)
        )
    return result


def _foreign_key(schema, fk_id: str):
    for foreign_key in schema.foreign_keys:
        if foreign_key.foreign_key_id == fk_id:
            return foreign_key
    raise AssertionError(f"unknown FK {fk_id}")


if __name__ == "__main__":
    unittest.main()
