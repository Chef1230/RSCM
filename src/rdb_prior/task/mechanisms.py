"""Classification task mechanisms with exact route provenance.

Every mechanism now explicitly samples three things from its target table
node: a primary key (row_id), a target column, and — when the target table
carries a TIME column — a time column.  Synthetic-label mechanisms use the
well-known sentinel ``__label__`` as their ``target_column_id`` so that the
export layer can always emit ``row_id | label | cutoff_time``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from rdb_prior.compilation.model import ColumnKind, PhysicalForeignKey, PhysicalSchema
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.schema.spec import TableRole
from rdb_prior.task.model import (
    AggregateOperator,
    AggregateSpec,
    ClassificationKind,
    CombineOperator,
    CompareOperator,
    CompositeFamily,
    CompositeTaskSpec,
    LabelOperator,
    ObservationRule,
    PlannedTask,
    PredicateSpec,
    PredictionType,
    RoutePathLabel,
    RouteRole,
    TaskData,
    TaskMechanism,
    TaskPlan,
)
from rdb_prior.task.view import build_task_view

# Sentinel column id used by mechanisms whose label is computed rather than
# sourced from an existing feature column.
_SYNTHETIC_TARGET = "__label__"

# Interaction-response tasks derive a dedicated label stream by XOR-mixing the
# plan seed (int) with this fixed constant; the derived stream is never stored
# in ``TaskPlan.parameters`` because those values are coerced to float and a
# 64-bit seed would lose precision above 2^53, breaking exact label replay.
_LABEL_SEED_MAGIC = 0x9E3779B1
_MAX_LABEL_RETRIES = 16

# Keep interaction-response logits in a useful stochastic range. Raw
# log-frequency/item counts can otherwise grow with table size and drive the
# sigmoid to 0/1 for almost every eligible row.
_INTERACTION_FEATURE_LOGIT_SCALE = 2.0
_INTERACTION_COUNT_QUANTILE = 0.95


# ---------------------------------------------------------------------------
# Unified target-table sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class TargetSample:
    """The three columns every task samples from its target table node."""

    table_id: str
    primary_key_column_id: str
    target_column_id: str
    time_column_id: str | None


def _sample_target_columns(
    schema: PhysicalSchema,
    target_table_id: str,
    *,
    target_column_id: str,
    time_column_id: str | None = None,
) -> TargetSample:
    """Bundle primary key, target column, and optional time column."""
    target_table = schema.table(target_table_id)
    return TargetSample(
        table_id=target_table_id,
        primary_key_column_id=target_table.primary_key.column_id,
        target_column_id=target_column_id,
        time_column_id=time_column_id,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RelationAttributeCandidate:
    table_id: str
    column_id: str
    source_table_id: str
    source_column_id: str
    required_path: tuple[str, ...]
    classification_kind: ClassificationKind
    class_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RandomColumnCandidate:
    """A target table with feature columns eligible for random selection."""

    table_id: str
    feature_column_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class FutureEventCandidate:
    foreign_key_id: str
    entity_table_id: str
    event_table_id: str
    time_column_id: str
    target_column_id: str = _SYNTHETIC_TARGET


@dataclass(frozen=True, slots=True, kw_only=True)
class FutureEventAttributeCandidate:
    event_table_id: str
    target_column_id: str
    parent_table_id: str
    source_column_id: str
    foreign_key_id: str
    time_column_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TemporalAggregateCandidate:
    target_table_id: str
    source_table_id: str
    required_path: tuple[str, ...]
    time_column_id: str
    operator: AggregateOperator
    source_column_id: str | None
    target_column_id: str = _SYNTHETIC_TARGET


@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionCandidate:
    """One interaction-response task shape: an Event row is an interaction whose
    response is predicted from the entity's historical interaction behavior and
    (optionally) the item/context's historical interaction count."""

    event_table_id: str
    entity_foreign_key_id: str
    item_foreign_key_id: str | None
    time_column_id: str
    feature_column_id: str
    target_column_id: str = _SYNTHETIC_TARGET


def relation_attribute_candidates(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    *,
    max_classification_categories: int,
) -> tuple[RelationAttributeCandidate, ...]:
    """Use Event/Detail target columns and an actual parent source column."""
    result: list[RelationAttributeCandidate] = []
    for table in schema.tables:
        if table.role not in {TableRole.EVENT, TableRole.DETAIL}:
            continue
        target_columns = [
            column
            for column in table.columns
            if column.kind is ColumnKind.FEATURE and not column.unique
        ]
        if not target_columns:
            continue
        for fk in schema.foreign_keys:
            if fk.child_table_id != table.table_id:
                continue
            parent = schema.table(fk.parent_table_id)
            sources = _usable_feature_columns(schema, database, parent.table_id)
            for target in target_columns:
                observed = _observed_values(database.table(table.table_id).column(target.column_id))
                unique_count = len(np.unique(observed))
                kind = (
                    ClassificationKind.CATEGORICAL
                    if 2 < unique_count <= max_classification_categories
                    else ClassificationKind.BINARY
                )
                class_count = min(max(2, unique_count), max_classification_categories)
                for source in sources:
                    result.append(
                        RelationAttributeCandidate(
                            table_id=table.table_id,
                            column_id=target.column_id,
                            source_table_id=parent.table_id,
                            source_column_id=source,
                            required_path=(fk.foreign_key_id,),
                            classification_kind=kind,
                            class_count=class_count,
                        )
                    )
    return tuple(result)


def random_column_candidates(
    schema: PhysicalSchema, database: DatabaseInstance
) -> tuple[RandomColumnCandidate, ...]:
    """Enumerate tables that can support RDBPFN-style random target tasks.

    A task masks one feature column and learns its thresholded values from the
    remaining columns. Requiring at least two usable features leaves at least
    one physical feature after masking, matching the intended pretraining
    setting rather than producing target-only tasks.
    """
    result: list[RandomColumnCandidate] = []
    for table in schema.tables:
        feature_column_ids = _usable_feature_columns(
            schema, database, table.table_id
        )
        if len(feature_column_ids) >= 2:
            result.append(
                RandomColumnCandidate(
                    table_id=table.table_id,
                    feature_column_ids=feature_column_ids,
                )
            )
    return tuple(result)


def future_event_candidates(schema: PhysicalSchema) -> tuple[FutureEventCandidate, ...]:
    result: list[FutureEventCandidate] = []
    for fk in schema.foreign_keys:
        parent = schema.table(fk.parent_table_id)
        child = schema.table(fk.child_table_id)
        time_column = _time_column(child)
        if (
            parent.role is TableRole.ENTITY
            and child.role is TableRole.EVENT
            and time_column is not None
        ):
            result.append(
                FutureEventCandidate(
                    foreign_key_id=fk.foreign_key_id,
                    entity_table_id=parent.table_id,
                    event_table_id=child.table_id,
                    time_column_id=time_column,
                )
            )
    return tuple(result)


def future_event_attribute_candidates(
    schema: PhysicalSchema, database: DatabaseInstance
) -> tuple[FutureEventAttributeCandidate, ...]:
    result: list[FutureEventAttributeCandidate] = []
    for event in schema.tables:
        if event.role is not TableRole.EVENT:
            continue
        time_column = _time_column(event)
        targets = [c.column_id for c in event.columns if c.kind is ColumnKind.FEATURE]
        if time_column is None or not targets:
            continue
        for fk in schema.foreign_keys:
            if fk.child_table_id != event.table_id:
                continue
            sources = _usable_feature_columns(schema, database, fk.parent_table_id)
            for target in targets:
                for source in sources:
                    result.append(
                        FutureEventAttributeCandidate(
                            event_table_id=event.table_id,
                            target_column_id=target,
                            parent_table_id=fk.parent_table_id,
                            source_column_id=source,
                            foreign_key_id=fk.foreign_key_id,
                            time_column_id=time_column,
                        )
                    )
    return tuple(result)


def interaction_candidates(
    schema: PhysicalSchema, database: DatabaseInstance
) -> tuple[InteractionCandidate, ...]:
    """Enumerate interaction-response candidate shapes.

    Each Event row is treated as one interaction linking an entity parent (the
    ``user``) to an optional second parent (the ``item``/context). The entity
    FK must point at an ENTITY-role parent carrying at least one usable
    feature (the ``U_e`` user-propensity input); the optional item FK points
    at a different parent table. Candidates cover every usable entity feature
    and both the no-item and each-item variant.
    """
    result: list[InteractionCandidate] = []
    for event in schema.tables:
        if event.role is not TableRole.EVENT:
            continue
        time_column = _time_column(event)
        if time_column is None:
            continue
        outgoing = [
            foreign_key
            for foreign_key in schema.foreign_keys
            if foreign_key.child_table_id == event.table_id
        ]
        for entity_fk in outgoing:
            parent = schema.table(entity_fk.parent_table_id)
            if parent.role is not TableRole.ENTITY:
                continue
            features = _usable_feature_columns(
                schema, database, parent.table_id
            )
            if not features:
                continue
            item_options = [None] + [
                foreign_key
                for foreign_key in outgoing
                if foreign_key.foreign_key_id != entity_fk.foreign_key_id
                and foreign_key.parent_table_id != entity_fk.parent_table_id
            ]
            for feature in features:
                for item_fk in item_options:
                    result.append(
                        InteractionCandidate(
                            event_table_id=event.table_id,
                            entity_foreign_key_id=entity_fk.foreign_key_id,
                            item_foreign_key_id=(
                                None
                                if item_fk is None
                                else item_fk.foreign_key_id
                            ),
                            time_column_id=time_column,
                            feature_column_id=feature,
                        )
                    )
    return tuple(result)


def temporal_aggregate_candidates(
    schema: PhysicalSchema, database: DatabaseInstance
) -> tuple[TemporalAggregateCandidate, ...]:
    result: list[TemporalAggregateCandidate] = []
    for target in schema.tables:
        if target.role not in {TableRole.ENTITY, TableRole.EVENT}:
            continue
        if not _has_incoming_fk(schema, target.table_id):
            continue
        for path, endpoint_id in _enumerate_paths(schema, target.table_id, max_depth=2):
            endpoint = schema.table(endpoint_id)
            if endpoint.role not in {TableRole.EVENT, TableRole.DETAIL}:
                continue
            if target.role is TableRole.EVENT and endpoint.role is not TableRole.DETAIL:
                continue
            time_column = _time_column(endpoint)
            if time_column is None:
                continue
            result.append(
                TemporalAggregateCandidate(
                    target_table_id=target.table_id,
                    source_table_id=endpoint.table_id,
                    required_path=path,
                    time_column_id=time_column,
                    operator=AggregateOperator.COUNT,
                    source_column_id=None,
                )
            )
            for source in _usable_feature_columns(schema, database, endpoint.table_id):
                for operator in (
                    AggregateOperator.SUM,
                    AggregateOperator.MAX,
                    AggregateOperator.MIN,
                ):
                    result.append(
                        TemporalAggregateCandidate(
                            target_table_id=target.table_id,
                            source_table_id=endpoint.table_id,
                            required_path=path,
                            time_column_id=time_column,
                            operator=operator,
                            source_column_id=source,
                        )
                    )
    return tuple(result)


def build_relation_attribute_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: RelationAttributeCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
    positive_rate_min: float = 0.2, positive_rate_max: float = 0.8,
) -> PlannedTask | None:
    rng = _rng(seed)
    scores = _path_source_scores(
        schema, database, candidate.table_id, candidate.required_path,
        candidate.source_column_id,
    )
    if candidate.classification_kind is ClassificationKind.CATEGORICAL:
        labels = _categorical_quantiles(scores, candidate.class_count)
        threshold = None
        requested_rate = None
    else:
        requested_rate = float(rng.uniform(positive_rate_min, positive_rate_max))
        labels, threshold = _threshold_labels(scores, requested_rate)
    # When the target table carries a TIME column, use temporal ordering for
    # the support/query split so that the export layer can emit cutoff_time.
    target_table = schema.table(candidate.table_id)
    time_col = _time_column(target_table)
    if time_col is not None:
        times = database.table(candidate.table_id).column(time_col)
        ordered = np.argsort(times, kind="stable").astype(np.int64)
        split = _temporal_split(
            labels, ordered, rng, support_fraction=support_fraction,
            min_support_rows=min_support_rows, min_query_rows=min_query_rows,
            min_class_count=min_class_count_per_split,
        )
        split_strategy = "temporal_rows"
        visibility_cutoff = int(times[ordered[-1]]) if len(ordered) else 0
        obs_rules = _observation_rules(schema, visibility_cutoff)
    else:
        split = _stratified_split(
            labels, rng, support_fraction=support_fraction,
            min_support_rows=min_support_rows, min_query_rows=min_query_rows,
            min_class_count=min_class_count_per_split,
        )
        split_strategy = "stratified_rows"
        visibility_cutoff = None
        obs_rules = ()
    if split is None:
        return None
    support, query = split
    plan = TaskPlan(
        task_id=task_id, sample_id=sample_id, instance_id=database.instance_id,
        schema_id=schema.schema_id, mechanism=TaskMechanism.RELATION_ATTRIBUTE,
        prediction_type=PredictionType.CLASSIFICATION,
        target_table_id=candidate.table_id,
        source_table_id=candidate.source_table_id,
        target_column_id=candidate.column_id,
        source_column_id=candidate.source_column_id,
        time_column_id=time_col,
        cutoff_time=visibility_cutoff,
        split_strategy=split_strategy, seed=seed,
        masked_column_ids=(candidate.column_id,),
        observation_rules=obs_rules,
        route_supervision=_schema_route_labels(
            schema, target_table_id=candidate.table_id,
            required_paths=(candidate.required_path,),
        ),
        classification_kind=candidate.classification_kind,
        threshold=threshold, requested_positive_rate=requested_rate,
        realized_positive_rate=(
            None
            if candidate.classification_kind is ClassificationKind.CATEGORICAL
            else float(np.mean(labels == 1))
        ),
        parameters=(("class_count", candidate.class_count), ("support_fraction", support_fraction)),
    )
    return _planned_if_visible(
        schema, database, plan, labels, support, query
    )


def build_random_column_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: RandomColumnCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
) -> PlannedTask | None:
    """Build one RDBPFN-compatible random target-column task.

    This mirrors ``_augment_batch_with_random_targets`` in RDBPFN: choose one
    usable feature, draw a threshold uniformly between its observed minimum
    and maximum, and define the binary target as ``value > threshold``.
    """
    if not candidate.feature_column_ids:
        return None
    rng = _rng(seed)
    target_index = int(rng.integers(0, len(candidate.feature_column_ids)))
    target_column_id = candidate.feature_column_ids[target_index]
    raw_values = database.table(candidate.table_id).column(target_column_id)
    scores = _numeric(raw_values)
    col_min = float(np.min(scores))
    col_max = float(np.max(scores))
    diff = col_max - col_min
    threshold = (
        col_min
        if diff <= 1e-6
        else col_min + float(rng.uniform()) * diff
    )
    labels = (scores > threshold).astype(np.int8)
    split = _stratified_split(
        labels, rng, support_fraction=support_fraction,
        min_support_rows=min_support_rows, min_query_rows=min_query_rows,
        min_class_count=min_class_count_per_split,
    )
    if split is None:
        return None
    support, query = split
    plan = TaskPlan(
        task_id=task_id, sample_id=sample_id, instance_id=database.instance_id,
        schema_id=schema.schema_id, mechanism=TaskMechanism.RANDOM_COLUMN,
        prediction_type=PredictionType.CLASSIFICATION,
        target_table_id=candidate.table_id,
        source_table_id=candidate.table_id,
        target_column_id=target_column_id,
        split_strategy="stratified_rows", seed=seed,
        masked_column_ids=(target_column_id,),
        classification_kind=ClassificationKind.BINARY,
        threshold=threshold,
        realized_positive_rate=float(np.mean(labels == 1)),
        parameters=(
            ("target_column_index", target_index),
            ("support_fraction", support_fraction),
        ),
    )
    return _planned_if_visible(
        schema, database, plan, labels, support, query
    )


def _build_history_gated_submode_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: FutureEventCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
    cutoff_quantile_min: float, cutoff_quantile_max: float,
    horizon_fraction_min: float, horizon_fraction_max: float,
    positive_rate_min: float, positive_rate_max: float,
    history_gated_frequency_weight_min: float,
    history_gated_frequency_weight_max: float,
    history_gated_silence_weight_min: float,
    history_gated_silence_weight_max: float,
    mechanism: TaskMechanism,
) -> PlannedTask | None:
    """Build a history-gated task from the actual future event window.

    Only entities with prior history are eligible (-1 otherwise). For an
    ACTIVE task, an eligible entity is positive iff it has an event in
    (cutoff, horizon]; INACTIVE uses the complement. The requested rate is
    used only to choose the horizon whose observed future-event rate is closest
    to the sampled target. It never creates or flips labels.

    The frequency/silence weight arguments remain in the public builder
    signature for compatibility with older configs and are not used to
    synthesize labels.
    """
    rng = _rng(seed)
    event = database.table(candidate.event_table_id)
    times = event.column(candidate.time_column_id)
    if len(times) < 2 or int(times.max()) <= int(times.min()):
        return None
    cutoff_q = float(rng.uniform(cutoff_quantile_min, cutoff_quantile_max))
    cutoff = int(np.quantile(times, cutoff_q))
    span = int(times.max()) - int(times.min())
    desired = float(rng.uniform(positive_rate_min, positive_rate_max))
    low = cutoff + max(1, round(span * float(horizon_fraction_min)))
    high = cutoff + max(1, round(span * float(horizon_fraction_max)))
    horizon_candidates = np.unique(np.clip(times[times > cutoff], low, high))
    if not len(horizon_candidates):
        return None
    positive_is_active = mechanism is TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE
    best: tuple[float, int, np.ndarray] | None = None
    for horizon_candidate in horizon_candidates:
        candidate_labels = _history_gated_future_values(
            schema, database, candidate,
            cutoff=cutoff, horizon=int(horizon_candidate),
            positive_is_active=positive_is_active,
        )
        valid_labels = candidate_labels[candidate_labels >= 0]
        if not len(valid_labels):
            continue
        item = (
            abs(float(np.mean(valid_labels)) - desired),
            int(horizon_candidate),
            candidate_labels,
        )
        if best is None or item[0] < best[0]:
            best = item
    if best is None:
        return None
    _distance, horizon, labels = best
    # Recompute the rate for the selected horizon, not the last candidate.
    valid_labels = labels[labels >= 0]
    # Filter to only entities with history for the split.
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return None
    valid_indices = np.flatnonzero(valid_mask).astype(np.int64)
    valid_labels = labels[valid_mask]
    split = _stratified_split(
        valid_labels, rng, support_fraction=support_fraction,
        min_support_rows=min_support_rows, min_query_rows=min_query_rows,
        min_class_count=min_class_count_per_split,
    )
    if split is None:
        return None
    valid_support, valid_query = split
    support = valid_indices[valid_support]
    query = valid_indices[valid_query]
    plan = TaskPlan(
        task_id=task_id, sample_id=sample_id, instance_id=database.instance_id,
        schema_id=schema.schema_id,
        mechanism=mechanism,
        prediction_type=PredictionType.CLASSIFICATION,
        target_table_id=candidate.entity_table_id,
        source_table_id=candidate.event_table_id,
        target_column_id=candidate.target_column_id,
        foreign_key_id=candidate.foreign_key_id,
        time_column_id=candidate.time_column_id,
        cutoff_time=cutoff, horizon_end_time=horizon,
        split_strategy="stratified_entities", seed=seed,
        masked_column_ids=(candidate.target_column_id,),
        observation_rules=_observation_rules(schema, cutoff),
        route_supervision=_schema_route_labels(
            schema, target_table_id=candidate.entity_table_id,
            required_paths=((candidate.foreign_key_id,),),
        ),
        classification_kind=ClassificationKind.BINARY,
        requested_positive_rate=desired,
        realized_positive_rate=float(np.mean(valid_labels)),
        parameters=(
            ("cutoff_quantile", cutoff_q),
            ("support_fraction", support_fraction),
        ),
    )
    return _planned_if_visible(
        schema, database, plan, labels, support, query
    )


def build_future_event_existence_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: FutureEventCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
    cutoff_quantile_min: float, cutoff_quantile_max: float,
    horizon_fraction_min: float, horizon_fraction_max: float,
    positive_rate_min: float = 0.15, positive_rate_max: float = 0.65,
) -> PlannedTask | None:
    return _build_entity_future_horizon_task(
        task_id=task_id, sample_id=sample_id, schema=schema,
        database=database, candidate=candidate, seed=seed,
        support_fraction=support_fraction, min_support_rows=min_support_rows,
        min_query_rows=min_query_rows,
        min_class_count_per_split=min_class_count_per_split,
        cutoff_quantile_min=cutoff_quantile_min,
        cutoff_quantile_max=cutoff_quantile_max,
        horizon_fraction_min=horizon_fraction_min,
        horizon_fraction_max=horizon_fraction_max,
        positive_rate_min=positive_rate_min,
        positive_rate_max=positive_rate_max,
        mechanism=TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE,
        label_values=_future_existence_values,
    )


def build_history_gated_future_activity_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: FutureEventCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
    cutoff_quantile_min: float, cutoff_quantile_max: float,
    horizon_fraction_min: float, horizon_fraction_max: float,
    positive_rate_min: float = 0.15, positive_rate_max: float = 0.65,
) -> PlannedTask | None:
    """Positive label: the entity has events at or before the cutoff but
    produces no events within (cutoff, horizon]."""
    return _build_entity_future_horizon_task(
        task_id=task_id, sample_id=sample_id, schema=schema,
        database=database, candidate=candidate, seed=seed,
        support_fraction=support_fraction, min_support_rows=min_support_rows,
        min_query_rows=min_query_rows,
        min_class_count_per_split=min_class_count_per_split,
        cutoff_quantile_min=cutoff_quantile_min,
        cutoff_quantile_max=cutoff_quantile_max,
        horizon_fraction_min=horizon_fraction_min,
        horizon_fraction_max=horizon_fraction_max,
        positive_rate_min=positive_rate_min,
        positive_rate_max=positive_rate_max,
        mechanism=TaskMechanism.HISTORY_GATED_FUTURE_ACTIVITY,
        label_values=_history_gated_values,
    )


def build_history_gated_future_inactive_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: FutureEventCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
    cutoff_quantile_min: float, cutoff_quantile_max: float,
    horizon_fraction_min: float, horizon_fraction_max: float,
    positive_rate_min: float = 0.15, positive_rate_max: float = 0.65,
    history_gated_frequency_weight_min: float = 0.5,
    history_gated_frequency_weight_max: float = 3.0,
    history_gated_silence_weight_min: float = 0.5,
    history_gated_silence_weight_max: float = 3.0,
) -> PlannedTask | None:
    """Sub-mode 1: only entities with history. Positive = churned (silent,
    low-frequency entities are more likely positive)."""
    return _build_history_gated_submode_task(
        task_id=task_id, sample_id=sample_id, schema=schema,
        database=database, candidate=candidate, seed=seed,
        support_fraction=support_fraction, min_support_rows=min_support_rows,
        min_query_rows=min_query_rows,
        min_class_count_per_split=min_class_count_per_split,
        cutoff_quantile_min=cutoff_quantile_min,
        cutoff_quantile_max=cutoff_quantile_max,
        horizon_fraction_min=horizon_fraction_min,
        horizon_fraction_max=horizon_fraction_max,
        positive_rate_min=positive_rate_min,
        positive_rate_max=positive_rate_max,
        history_gated_frequency_weight_min=history_gated_frequency_weight_min,
        history_gated_frequency_weight_max=history_gated_frequency_weight_max,
        history_gated_silence_weight_min=history_gated_silence_weight_min,
        history_gated_silence_weight_max=history_gated_silence_weight_max,
        mechanism=TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE,
    )


def build_history_gated_future_active_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: FutureEventCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
    cutoff_quantile_min: float, cutoff_quantile_max: float,
    horizon_fraction_min: float, horizon_fraction_max: float,
    positive_rate_min: float = 0.15, positive_rate_max: float = 0.65,
    history_gated_frequency_weight_min: float = 0.5,
    history_gated_frequency_weight_max: float = 3.0,
    history_gated_silence_weight_min: float = 0.5,
    history_gated_silence_weight_max: float = 3.0,
) -> PlannedTask | None:
    """Sub-mode 2: only entities with history. Positive = retained (recent,
    high-frequency entities are more likely positive)."""
    return _build_history_gated_submode_task(
        task_id=task_id, sample_id=sample_id, schema=schema,
        database=database, candidate=candidate, seed=seed,
        support_fraction=support_fraction, min_support_rows=min_support_rows,
        min_query_rows=min_query_rows,
        min_class_count_per_split=min_class_count_per_split,
        cutoff_quantile_min=cutoff_quantile_min,
        cutoff_quantile_max=cutoff_quantile_max,
        horizon_fraction_min=horizon_fraction_min,
        horizon_fraction_max=horizon_fraction_max,
        positive_rate_min=positive_rate_min,
        positive_rate_max=positive_rate_max,
        history_gated_frequency_weight_min=history_gated_frequency_weight_min,
        history_gated_frequency_weight_max=history_gated_frequency_weight_max,
        history_gated_silence_weight_min=history_gated_silence_weight_min,
        history_gated_silence_weight_max=history_gated_silence_weight_max,
        mechanism=TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE,
    )


def _build_entity_future_horizon_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: FutureEventCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
    cutoff_quantile_min: float, cutoff_quantile_max: float,
    horizon_fraction_min: float, horizon_fraction_max: float,
    positive_rate_min: float, positive_rate_max: float,
    mechanism: TaskMechanism,
    label_values: Callable[
        [PhysicalSchema, DatabaseInstance, FutureEventCandidate, int, int],
        np.ndarray,
    ],
) -> PlannedTask | None:
    rng = _rng(seed)
    event = database.table(candidate.event_table_id)
    times = event.column(candidate.time_column_id)
    if len(times) < 2 or int(times.max()) <= int(times.min()):
        return None
    cutoff_q = float(rng.uniform(cutoff_quantile_min, cutoff_quantile_max))
    cutoff = int(np.quantile(times, cutoff_q))
    desired = float(rng.uniform(positive_rate_min, positive_rate_max))
    low = cutoff + max(1, round((int(times.max()) - int(times.min())) * horizon_fraction_min))
    high = cutoff + max(1, round((int(times.max()) - int(times.min())) * horizon_fraction_max))
    horizon_candidates = np.unique(np.clip(times[(times > cutoff)], low, high))
    if not len(horizon_candidates):
        return None
    best: tuple[float, int, np.ndarray] | None = None
    for horizon in horizon_candidates:
        plan_stub = (cutoff, int(horizon))
        labels = label_values(schema, database, candidate, *plan_stub)
        item = (abs(float(np.mean(labels)) - desired), int(horizon), labels)
        if best is None or item[0] < best[0]:
            best = item
    assert best is not None
    _distance, horizon, labels = best
    split = _stratified_split(
        labels, rng, support_fraction=support_fraction,
        min_support_rows=min_support_rows, min_query_rows=min_query_rows,
        min_class_count=min_class_count_per_split,
    )
    if split is None:
        return None
    support, query = split
    plan = TaskPlan(
        task_id=task_id, sample_id=sample_id, instance_id=database.instance_id,
        schema_id=schema.schema_id,
        mechanism=mechanism,
        prediction_type=PredictionType.CLASSIFICATION,
        target_table_id=candidate.entity_table_id,
        source_table_id=candidate.event_table_id,
        target_column_id=candidate.target_column_id,
        foreign_key_id=candidate.foreign_key_id,
        time_column_id=candidate.time_column_id,
        cutoff_time=cutoff, horizon_end_time=horizon,
        split_strategy="stratified_entities", seed=seed,
        masked_column_ids=(candidate.target_column_id,),
        observation_rules=_observation_rules(schema, cutoff),
        route_supervision=_schema_route_labels(
            schema, target_table_id=candidate.entity_table_id,
            required_paths=((candidate.foreign_key_id,),),
        ),
        classification_kind=ClassificationKind.BINARY,
        requested_positive_rate=desired,
        realized_positive_rate=float(np.mean(labels)),
        parameters=(("cutoff_quantile", cutoff_q), ("support_fraction", support_fraction)),
    )
    return _planned_if_visible(
        schema, database, plan, labels, support, query
    )


def build_future_event_attribute_condition_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: FutureEventAttributeCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
    positive_rate_min: float = 0.2, positive_rate_max: float = 0.8,
) -> PlannedTask | None:
    rng = _rng(seed)
    path = (candidate.foreign_key_id,)
    scores = _path_source_scores(
        schema, database, candidate.event_table_id, path, candidate.source_column_id
    )
    desired = float(rng.uniform(positive_rate_min, positive_rate_max))
    labels, threshold = _threshold_labels(scores, desired)
    split = _stratified_split(
        labels, rng, support_fraction=support_fraction,
        min_support_rows=min_support_rows, min_query_rows=min_query_rows,
        min_class_count=min_class_count_per_split,
    )
    if split is None:
        return None
    support, query = split
    event_times = database.table(candidate.event_table_id).column(candidate.time_column_id)
    plan = TaskPlan(
        task_id=task_id, sample_id=sample_id, instance_id=database.instance_id,
        schema_id=schema.schema_id,
        mechanism=TaskMechanism.FUTURE_EVENT_ATTRIBUTE_CONDITION,
        prediction_type=PredictionType.CLASSIFICATION,
        target_table_id=candidate.event_table_id,
        source_table_id=candidate.parent_table_id,
        target_column_id=candidate.target_column_id,
        source_column_id=candidate.source_column_id,
        time_column_id=candidate.time_column_id,
        row_cutoff_time_column_id=candidate.time_column_id,
        split_strategy="stratified_event_rows", seed=seed,
        masked_column_ids=(candidate.target_column_id,),
        observation_rules=_observation_rules(schema, int(np.max(event_times))),
        route_supervision=_schema_route_labels(
            schema, target_table_id=candidate.event_table_id,
            required_paths=(path,),
        ),
        classification_kind=ClassificationKind.BINARY,
        threshold=threshold, requested_positive_rate=desired,
        realized_positive_rate=float(np.mean(labels)),
        parameters=(("support_fraction", support_fraction),),
    )
    return _planned_if_visible(
        schema, database, plan, labels, support, query
    )


def build_temporal_relational_aggregate_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: TemporalAggregateCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
    cutoff_quantile_min: float, cutoff_quantile_max: float,
    window_repeated_probability: float, window_short_probability: float,
    window_short_fraction_min: float, window_short_fraction_max: float,
    window_long_fraction_min: float, window_long_fraction_max: float,
    positive_rate_min: float = 0.2, positive_rate_max: float = 0.8,
) -> PlannedTask | None:
    """Build a temporal relational aggregate task with mixed window regimes.

    The aggregation window is sampled per task from three regimes: SHORT
    (a small single window), LONG (a large single window) or REPEATED (a
    short *and* a long window nested at the same cutoff, summed). This mixes
    near-term and longer-term activity signals across the task corpus.
    """
    rng = _rng(seed)
    target = schema.table(candidate.target_table_id)
    source_times = database.table(candidate.source_table_id).column(candidate.time_column_id)
    if len(source_times) < 2 or int(source_times.max()) <= int(source_times.min()):
        return None
    row_cutoff_column = _time_column(target) if target.role is TableRole.EVENT else None
    span = int(source_times.max()) - int(source_times.min())

    def _fraction_window(lo: float, hi: float) -> int:
        return max(1, round(span * float(rng.uniform(lo, hi))))

    if rng.random() < window_repeated_probability:
        regime = 2
        window = _fraction_window(window_short_fraction_min, window_short_fraction_max)
        window_extended = _fraction_window(window_long_fraction_min, window_long_fraction_max)
    else:
        regime = 0 if rng.random() < window_short_probability else 1
        if regime == 0:
            window = _fraction_window(window_short_fraction_min, window_short_fraction_max)
        else:
            window = _fraction_window(window_long_fraction_min, window_long_fraction_max)
        window_extended = None
    windows = (window,) if window_extended is None else (window, window_extended)
    if row_cutoff_column is None:
        cutoff = int(np.quantile(source_times, float(rng.uniform(cutoff_quantile_min, cutoff_quantile_max))))
        horizon = min(int(source_times.max()), cutoff + max(windows))
        cutoffs = np.full(database.table(target.table_id).row_count, cutoff, dtype=np.int64)
    else:
        cutoff = None
        horizon = None
        cutoffs = database.table(target.table_id).column(row_cutoff_column).astype(np.int64)
    aggregates = _windowed_aggregates(schema, database, candidate, cutoffs, windows)
    desired = float(rng.uniform(positive_rate_min, positive_rate_max))
    labels, threshold = _threshold_labels(aggregates, desired)
    split = _stratified_split(
        labels, rng, support_fraction=support_fraction,
        min_support_rows=min_support_rows, min_query_rows=min_query_rows,
        min_class_count=min_class_count_per_split,
    )
    if split is None:
        return None
    support, query = split
    visibility_cutoff = int(np.max(cutoffs)) if cutoff is None else int(cutoff)
    parameters = [
        ("window", float(window)),
        ("window_regime", float(regime)),
        ("support_fraction", support_fraction),
    ]
    if window_extended is not None:
        parameters.append(("window_extended", float(window_extended)))
    plan = TaskPlan(
        task_id=task_id, sample_id=sample_id, instance_id=database.instance_id,
        schema_id=schema.schema_id,
        mechanism=TaskMechanism.TEMPORAL_RELATIONAL_AGGREGATE,
        prediction_type=PredictionType.CLASSIFICATION,
        target_table_id=candidate.target_table_id,
        source_table_id=candidate.source_table_id,
        target_column_id=candidate.target_column_id,
        source_column_id=candidate.source_column_id,
        time_column_id=candidate.time_column_id,
        cutoff_time=cutoff, horizon_end_time=horizon,
        row_cutoff_time_column_id=row_cutoff_column,
        split_strategy="stratified_rows", seed=seed,
        masked_column_ids=(candidate.target_column_id,),
        observation_rules=_observation_rules(schema, visibility_cutoff),
        route_supervision=_schema_route_labels(
            schema, target_table_id=candidate.target_table_id,
            required_paths=(candidate.required_path,),
        ),
        classification_kind=ClassificationKind.BINARY,
        aggregate_operator=candidate.operator,
        threshold=threshold, requested_positive_rate=desired,
        realized_positive_rate=float(np.mean(labels)),
        parameters=tuple(parameters),
    )
    return _planned_if_visible(
        schema, database, plan, labels, support, query
    )


def build_interaction_response_task(
    *, task_id: str, sample_id: str, schema: PhysicalSchema,
    database: DatabaseInstance, candidate: InteractionCandidate,
    seed: int, support_fraction: float, min_support_rows: int,
    min_query_rows: int, min_class_count_per_split: int,
    positive_rate_min: float, positive_rate_max: float,
    interaction_u_weight_min: float, interaction_u_weight_max: float,
    interaction_frequency_weight_min: float, interaction_frequency_weight_max: float,
    interaction_silence_weight_min: float, interaction_silence_weight_max: float,
    interaction_item_weight_min: float, interaction_item_weight_max: float,
    interaction_invert_probability: float,
) -> PlannedTask | None:
    """Build an interaction-response task over an Event table.

    Every Event row is an interaction; rows whose entity has no prior interaction
    are excluded (``-1``) and the rest draw ``Bernoulli(propensity)`` from a
    base rate plus per-task frequency/recency/entity/item weights. Each task
    independently samples its beta weights and flips the label (response vs
    ignore) with ``interaction_invert_probability``.
    """
    rng = _rng(seed)
    event = database.table(candidate.event_table_id)
    times = event.column(candidate.time_column_id)
    if len(times) < 2 or int(times.max()) <= int(times.min()):
        return None
    beta_u = float(rng.uniform(interaction_u_weight_min, interaction_u_weight_max))
    beta_f = float(rng.uniform(
        interaction_frequency_weight_min, interaction_frequency_weight_max
    ))
    beta_s = float(rng.uniform(
        interaction_silence_weight_min, interaction_silence_weight_max
    ))
    beta_p = float(rng.uniform(
        interaction_item_weight_min, interaction_item_weight_max
    ))
    base_rate = float(rng.uniform(positive_rate_min, positive_rate_max))
    invert = bool(rng.random() < interaction_invert_probability)
    label_rng = _label_rng(seed)
    labels = _interaction_response_values(
        schema, database, candidate, beta_u=beta_u, beta_f=beta_f,
        beta_s=beta_s, beta_p=beta_p, base_rate=base_rate,
        invert=invert, rng=label_rng,
    )
    retries = 0
    while len(np.unique(labels[labels >= 0])) < 2 and retries < _MAX_LABEL_RETRIES:
        labels = _interaction_response_values(
            schema, database, candidate, beta_u=beta_u, beta_f=beta_f,
            beta_s=beta_s, beta_p=beta_p, base_rate=base_rate,
            invert=invert, rng=label_rng,
        )
        retries += 1
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return None
    valid_indices = np.flatnonzero(valid_mask).astype(np.int64)
    split = _stratified_split(
        labels[valid_mask], rng, support_fraction=support_fraction,
        min_support_rows=min_support_rows, min_query_rows=min_query_rows,
        min_class_count=min_class_count_per_split,
    )
    if split is None:
        return None
    valid_support, valid_query = split
    support = valid_indices[valid_support]
    query = valid_indices[valid_query]
    entity_fk = _foreign_key(schema, candidate.entity_foreign_key_id)
    plan = TaskPlan(
        task_id=task_id, sample_id=sample_id, instance_id=database.instance_id,
        schema_id=schema.schema_id,
        mechanism=TaskMechanism.INTERACTION_RESPONSE,
        prediction_type=PredictionType.CLASSIFICATION,
        target_table_id=candidate.event_table_id,
        source_table_id=entity_fk.parent_table_id,
        target_column_id=candidate.target_column_id,
        source_column_id=candidate.feature_column_id,
        foreign_key_id=candidate.entity_foreign_key_id,
        secondary_foreign_key_id=candidate.item_foreign_key_id,
        time_column_id=candidate.time_column_id,
        row_cutoff_time_column_id=candidate.time_column_id,
        split_strategy="stratified_rows", seed=seed,
        masked_column_ids=(candidate.target_column_id,),
        observation_rules=_observation_rules(schema, int(times.max())),
        route_supervision=_schema_route_labels(
            schema, target_table_id=candidate.event_table_id,
            required_paths=((candidate.entity_foreign_key_id,),),
            optional_paths=(
                ((candidate.item_foreign_key_id,),)
                if candidate.item_foreign_key_id is not None
                else ()
            ),
        ),
        classification_kind=ClassificationKind.BINARY,
        requested_positive_rate=base_rate if not invert else (1.0 - base_rate),
        realized_positive_rate=float(np.mean(labels[valid_mask])),
        parameters=(
            ("beta_u", beta_u), ("beta_f", beta_f), ("beta_s", beta_s),
            ("beta_p", beta_p), ("base_rate", base_rate),
            ("invert", float(invert)), ("label_retries", float(retries)),
            ("support_fraction", support_fraction),
        ),
    )
    return _planned_if_visible(
        schema, database, plan, labels, support, query
    )


def mechanism_labels(
    schema: PhysicalSchema, database: DatabaseInstance, plan: TaskPlan
) -> np.ndarray:
    required = tuple(
        label.foreign_key_ids for label in plan.route_supervision
        if label.role is RouteRole.REQUIRED
    )
    if plan.mechanism is TaskMechanism.RELATIONAL_CLASSIFICATION:
        return composite_labels(schema, database, plan)
    if plan.mechanism is TaskMechanism.RANDOM_COLUMN:
        if plan.target_column_id is None or plan.threshold is None:
            raise ValueError("random column plan is missing target metadata")
        values = database.table(plan.target_table_id).column(
            plan.target_column_id
        )
        return (_numeric(values) > float(plan.threshold)).astype(np.int8)
    if plan.mechanism is TaskMechanism.INTERACTION_RESPONSE:
        candidate = InteractionCandidate(
            event_table_id=plan.target_table_id,
            entity_foreign_key_id=plan.foreign_key_id or "",
            item_foreign_key_id=plan.secondary_foreign_key_id,
            time_column_id=plan.time_column_id or "",
            feature_column_id=plan.source_column_id or "",
        )
        # Replay the same per-row label stream the builder drew. Each pass
        # consumes exactly one uniform per history-gated row in row order, so
        # label_retries+1 passes land on the builder's final stream position.
        params = plan.parameter_map
        label_rng = _label_rng(plan.seed)
        labels = None
        for _ in range(int(params.get("label_retries", 0.0)) + 1):
            labels = _interaction_response_values(
                schema, database, candidate,
                beta_u=float(params["beta_u"]),
                beta_f=float(params["beta_f"]),
                beta_s=float(params["beta_s"]),
                beta_p=float(params["beta_p"]),
                base_rate=float(params["base_rate"]),
                invert=bool(params["invert"]),
                rng=label_rng,
            )
        assert labels is not None
        return labels
    if plan.mechanism in {
        TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE,
        TaskMechanism.HISTORY_GATED_FUTURE_ACTIVITY,
        TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE,
        TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE,
    }:
        candidate = FutureEventCandidate(
            foreign_key_id=plan.foreign_key_id or "",
            entity_table_id=plan.target_table_id,
            event_table_id=plan.source_table_id,
            time_column_id=plan.time_column_id or "",
        )
        if plan.mechanism is TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE:
            return _future_existence_values(
                schema, database, candidate,
                int(plan.cutoff_time), int(plan.horizon_end_time),
            )
        if plan.mechanism is TaskMechanism.HISTORY_GATED_FUTURE_ACTIVITY:
            return _history_gated_values(
                schema, database, candidate,
                int(plan.cutoff_time), int(plan.horizon_end_time),
            )
        if plan.mechanism is TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE:
            return _history_gated_future_values(
                schema, database, candidate,
                cutoff=int(plan.cutoff_time),
                horizon=int(plan.horizon_end_time),
                positive_is_active=False,
            )
        return _history_gated_future_values(
            schema, database, candidate,
            cutoff=int(plan.cutoff_time),
            horizon=int(plan.horizon_end_time),
            positive_is_active=True,
        )
    # Imported/legacy autocomplete tasks use the observed target column as the
    # benchmark label and therefore do not carry synthetic SCM source metadata.
    if (
        plan.mechanism is TaskMechanism.RELATION_ATTRIBUTE
        and plan.source_column_id is None
        and plan.target_column_id is not None
    ):
        return database.table(plan.target_table_id).column(plan.target_column_id)
    if not required:
        raise ValueError("task has no required path")
    if plan.mechanism in {
        TaskMechanism.RELATION_ATTRIBUTE,
        TaskMechanism.FUTURE_EVENT_ATTRIBUTE_CONDITION,
    }:
        scores = _path_source_scores(
            schema, database, plan.target_table_id, required[0], plan.source_column_id or ""
        )
        if plan.classification_kind is ClassificationKind.CATEGORICAL:
            return _categorical_quantiles(scores, int(plan.parameter_map["class_count"]))
        return (scores > float(plan.threshold)).astype(np.int8)
    if plan.mechanism is TaskMechanism.TEMPORAL_RELATIONAL_AGGREGATE:
        target = schema.table(plan.target_table_id)
        row_time = plan.row_cutoff_time_column_id
        if row_time is None:
            cutoffs = np.full(database.table(target.table_id).row_count, int(plan.cutoff_time), dtype=np.int64)
        else:
            cutoffs = database.table(target.table_id).column(row_time).astype(np.int64)
        candidate = TemporalAggregateCandidate(
            target_table_id=plan.target_table_id,
            source_table_id=plan.source_table_id,
            required_path=required[0], time_column_id=plan.time_column_id or "",
            operator=plan.aggregate_operator or AggregateOperator.COUNT,
            source_column_id=plan.source_column_id,
        )
        window = int(plan.parameter_map["window"])
        extended = plan.parameter_map.get("window_extended")
        windows = (window,) if extended is None else (window, int(extended))
        values = _windowed_aggregates(schema, database, candidate, cutoffs, windows)
        return (values > float(plan.threshold)).astype(np.int8)
    raise ValueError(f"unsupported mechanism {plan.mechanism.value}")


def future_event_labels(schema: PhysicalSchema, database: DatabaseInstance, plan: TaskPlan) -> np.ndarray:
    return mechanism_labels(schema, database, plan)


def _future_existence_values(
    schema: PhysicalSchema, database: DatabaseInstance,
    candidate: FutureEventCandidate, cutoff: int, horizon: int,
) -> np.ndarray:
    fk = _foreign_key(schema, candidate.foreign_key_id)
    event = database.table(candidate.event_table_id)
    times = event.column(candidate.time_column_id)
    assignments = event.column(fk.child_column_id)
    selected = (times > cutoff) & (times <= horizon) & (assignments >= 0)
    labels = np.zeros(database.table(candidate.entity_table_id).row_count, dtype=np.int8)
    labels[np.unique(assignments[selected])] = 1
    return labels


def _history_gated_values(
    schema: PhysicalSchema, database: DatabaseInstance,
    candidate: FutureEventCandidate, cutoff: int, horizon: int,
) -> np.ndarray:
    fk = _foreign_key(schema, candidate.foreign_key_id)
    event = database.table(candidate.event_table_id)
    times = event.column(candidate.time_column_id)
    assignments = event.column(fk.child_column_id)
    entity_count = database.table(candidate.entity_table_id).row_count
    valid = assignments >= 0
    has_history = np.zeros(entity_count, dtype=bool)
    has_future = np.zeros(entity_count, dtype=bool)
    has_history[np.unique(assignments[valid & (times <= cutoff)])] = True
    has_future[
        np.unique(assignments[valid & (times > cutoff) & (times <= horizon)])
    ] = True
    return (has_history & ~has_future).astype(np.int8)


def _history_gated_future_values(
    schema: PhysicalSchema, database: DatabaseInstance,
    candidate: FutureEventCandidate, *, cutoff: int, horizon: int,
    positive_is_active: bool,
) -> np.ndarray:
    """Return history-gated labels from real events in the future window.

    Entities without an event at or before cutoff receive -1 and are excluded
    from support/query. For eligible entities, ACTIVE is positive exactly when
    an event occurs in (cutoff, horizon]; INACTIVE is the exact complement. No
    random label stream is involved.
    """
    fk = _foreign_key(schema, candidate.foreign_key_id)
    event = database.table(candidate.event_table_id)
    times = event.column(candidate.time_column_id)
    assignments = event.column(fk.child_column_id)
    entity_count = database.table(candidate.entity_table_id).row_count
    valid = assignments >= 0
    has_history = np.zeros(entity_count, dtype=bool)
    has_future = np.zeros(entity_count, dtype=bool)
    has_history[np.unique(assignments[valid & (times <= cutoff)])] = True
    has_future[
        np.unique(assignments[valid & (times > cutoff) & (times <= horizon)])
    ] = True
    labels = np.full(entity_count, -1, dtype=np.int8)
    if positive_is_active:
        labels[has_history] = has_future[has_history].astype(np.int8)
    else:
        labels[has_history] = (~has_future[has_history]).astype(np.int8)
    return labels


def _interaction_response_values(
    schema: PhysicalSchema, database: DatabaseInstance,
    candidate: InteractionCandidate, *, beta_u: float, beta_f: float,
    beta_s: float, beta_p: float, base_rate: float, invert: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate bounded stochastic response labels for eligible interactions.

    Entity/item history counts are log-transformed and normalized per task.
    Their weighted score is centered and passed through tanh before being added
    to the base-rate logit. This preserves the frequency/recency direction
    without allowing table size to force probabilities to 0 or 1.
    """
    event = database.table(candidate.event_table_id)
    times = event.column(candidate.time_column_id).astype(np.int64)
    row_count = len(times)
    entity_fk = _foreign_key(schema, candidate.entity_foreign_key_id)
    entity = event.column(entity_fk.child_column_id).astype(np.int64)
    item_fk = (
        None
        if candidate.item_foreign_key_id is None
        else _foreign_key(schema, candidate.item_foreign_key_id)
    )
    item = (
        event.column(item_fk.child_column_id).astype(np.int64)
        if item_fk is not None
        else None
    )
    entity_values = database.table(entity_fk.parent_table_id).column(
        candidate.feature_column_id
    )
    raw_feature = _numeric(entity_values)
    observed = raw_feature[_observed_mask(entity_values)]
    labels = np.full(row_count, -1, dtype=np.int8)
    if not len(observed):
        return labels
    low = float(observed.min())
    high = float(observed.max())
    u_norm = np.clip((raw_feature - low) / max(high - low, 1e-12), 0.0, 1.0)
    span = max(1, int(times.max()) - int(times.min()))

    gated = np.zeros(row_count, dtype=bool)
    frequency = np.zeros(row_count, dtype=np.float64)
    silence = np.zeros(row_count, dtype=np.float64)
    item_frequency = np.zeros(row_count, dtype=np.float64)
    for row in range(row_count):
        ent = int(entity[row])
        if ent < 0:
            continue
        t_row = int(times[row])
        entity_history = (entity == ent) & (times < t_row)
        if not np.any(entity_history):
            continue
        gated[row] = True
        count_f = np.count_nonzero(entity_history)
        last = int(times[entity_history].max())
        frequency[row] = np.log1p(count_f)
        silence[row] = np.clip((t_row - last) / span, 0.0, 1.0)
        if item is not None:
            item_id = int(item[row])
            if item_id >= 0:
                item_frequency[row] = np.log1p(
                    np.count_nonzero((item == item_id) & (times < t_row))
                )
    if not np.any(gated):
        return labels

    def _normalize_count(values: np.ndarray) -> np.ndarray:
        scale = max(
            float(np.quantile(values[gated], _INTERACTION_COUNT_QUANTILE)),
            1.0,
        )
        return np.clip(values / scale, 0.0, 1.0)

    frequency_norm = _normalize_count(frequency)
    item_frequency_norm = _normalize_count(item_frequency)
    entity_feature = np.zeros(row_count, dtype=np.float64)
    valid_entity = gated & (entity >= 0) & (entity < len(u_norm))
    entity_feature[valid_entity] = u_norm[entity[valid_entity]]
    score = (
        beta_u * entity_feature
        + beta_f * frequency_norm
        - beta_s * silence
        + beta_p * item_frequency_norm
    )
    score[gated] -= float(np.median(score[gated]))
    logits = _logit(base_rate) + _INTERACTION_FEATURE_LOGIT_SCALE * np.tanh(score)
    if invert:
        logits = -logits
    probabilities = _sigmoid(np.clip(logits, -20.0, 20.0))
    for row in np.flatnonzero(gated):
        labels[row] = 1 if rng.random() < probabilities[row] else 0
    return labels


def _path_source_scores(
    schema: PhysicalSchema, database: DatabaseInstance, target_table_id: str,
    path: tuple[str, ...], source_column_id: str,
) -> np.ndarray:
    row_sets, endpoint = _traverse_path(schema, database, target_table_id, path)
    values = _numeric(database.table(endpoint).column(source_column_id))
    scores = np.zeros(len(row_sets), dtype=np.float64)
    for index, rows in enumerate(row_sets):
        if len(rows):
            scores[index] = float(np.mean(values[rows]))
    # Stable tiny tie-breaker; it prevents discrete parent columns collapsing a quantile task.
    scores += np.arange(len(scores), dtype=np.float64) * 1e-9
    return scores


def _aggregate_values(
    schema: PhysicalSchema, database: DatabaseInstance,
    candidate: TemporalAggregateCandidate, cutoffs: np.ndarray, window: int,
) -> np.ndarray:
    row_sets, endpoint = _traverse_path(
        schema, database, candidate.target_table_id, candidate.required_path
    )
    if endpoint != candidate.source_table_id:
        raise ValueError("required path endpoint does not match source table")
    source = database.table(endpoint)
    times = source.column(candidate.time_column_id)
    values = (
        None if candidate.source_column_id is None
        else _numeric(source.column(candidate.source_column_id))
    )
    output = np.zeros(len(row_sets), dtype=np.float64)
    for index, rows in enumerate(row_sets):
        selected = rows[(times[rows] > cutoffs[index]) & (times[rows] <= cutoffs[index] + window)]
        if candidate.operator is AggregateOperator.COUNT:
            output[index] = len(selected)
        elif not len(selected):
            output[index] = 0.0
        elif candidate.operator is AggregateOperator.SUM:
            output[index] = float(np.sum(values[selected]))
        elif candidate.operator is AggregateOperator.MAX:
            output[index] = float(np.max(values[selected]))
        else:
            output[index] = float(np.min(values[selected]))
    return output


def _windowed_aggregates(
    schema: PhysicalSchema, database: DatabaseInstance,
    candidate: TemporalAggregateCandidate, cutoffs: np.ndarray,
    windows: tuple[int, ...],
) -> np.ndarray:
    """Sum the aggregate over several windows anchored at the same cutoff.

    Used by the REPEATED (nested multi-scale) window regime: the label then
    reflects both near-term (short window) and longer-term (extended window)
    activity. The recompute path must supply the exact same window tuple.
    """
    total = np.zeros(len(cutoffs), dtype=np.float64)
    for window in windows:
        total += _aggregate_values(schema, database, candidate, cutoffs, int(window))
    return total


# ---------------------------------------------------------------------------
# Composite relational classification: generic aggregate executor
# ---------------------------------------------------------------------------


def _evaluate_predicate(
    values: np.ndarray, predicate: PredicateSpec
) -> np.ndarray:
    """Boolean row mask for one predicate; missing values never match.

    EQ/NE compare raw values (string-safe). Order predicates compare the
    ``_numeric`` conversion, which matches how value columns are read by the
    numeric aggregate operators.
    """
    if predicate.operator in {CompareOperator.EQ, CompareOperator.NE}:
        mask = values == predicate.value
        if predicate.operator is CompareOperator.NE:
            mask = ~mask
        return mask
    numeric = _numeric(values)
    target = float(predicate.value)
    if predicate.operator is CompareOperator.LT:
        return numeric < target
    if predicate.operator is CompareOperator.LE:
        return numeric <= target
    if predicate.operator is CompareOperator.GT:
        return numeric > target
    return numeric >= target


def _combine_aggregate_terms(
    terms: list[np.ndarray], operator: CombineOperator
) -> np.ndarray:
    """Combine per-aggregate target-row vectors with *operator*."""
    if operator is CombineOperator.MAX:
        result = terms[0]
        for other in terms[1:]:
            result = np.maximum(result, other)
        return result
    if operator is CombineOperator.MIN:
        result = terms[0]
        for other in terms[1:]:
            result = np.minimum(result, other)
        return result
    total = terms[0]
    for other in terms[1:]:
        total = total + other
    if operator is CombineOperator.MEAN:
        return total / len(terms)
    return total


def _compare_values(
    values: np.ndarray, operator: LabelOperator, threshold: float
) -> np.ndarray:
    if operator is LabelOperator.GE:
        return values >= threshold
    return values > threshold


def _evaluate_aggregate_spec(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    spec: AggregateSpec,
    target_table_id: str,
    cutoffs: np.ndarray,
) -> np.ndarray:
    """Per-target-row aggregate value for one :class:`AggregateSpec`.

    Time bounds are inclusive ``[cutoff + window_start, cutoff + window_end]``.
    An empty selected set yields ``0.0`` (never NaN). ``COUNT`` counts rows;
    ``COUNT_DISTINCT`` counts unique observed (non-missing) values; numeric
    operators reuse the shared ``_numeric`` conversion.
    """
    row_sets, endpoint = _traverse_path(
        schema, database, target_table_id, spec.required_path
    )
    if endpoint != spec.source_table_id:
        raise ValueError("required path endpoint does not match source table")
    source = database.table(endpoint)
    times = source.column(spec.time_column_id).astype(np.int64)
    output = np.zeros(len(row_sets), dtype=np.float64)
    predicate_mask = None
    if spec.predicates:
        predicate_mask = np.ones(len(times), dtype=bool)
        for predicate in spec.predicates:
            predicate_mask &= _evaluate_predicate(
                source.column(predicate.column_id), predicate
            )
    raw_values = (
        None
        if spec.operator is AggregateOperator.COUNT
        else source.column(spec.value_column_id)
    )
    numeric_values = (
        None if raw_values is None else _numeric(raw_values)
    )
    for index, rows in enumerate(row_sets):
        if not len(rows):
            continue
        lower = int(cutoffs[index]) + spec.window_start
        upper = int(cutoffs[index]) + spec.window_end
        selected = rows[(times[rows] >= lower) & (times[rows] <= upper)]
        if predicate_mask is not None:
            selected = selected[predicate_mask[selected]]
        if spec.operator is AggregateOperator.COUNT:
            output[index] = len(selected)
        elif not len(selected):
            output[index] = 0.0
        elif spec.operator is AggregateOperator.COUNT_DISTINCT:
            observed = raw_values[selected][_observed_mask(raw_values[selected])]
            output[index] = len(np.unique(observed))
        elif spec.operator is AggregateOperator.SUM:
            output[index] = float(np.sum(numeric_values[selected]))
        elif spec.operator is AggregateOperator.MEAN:
            output[index] = float(np.mean(numeric_values[selected]))
        elif spec.operator is AggregateOperator.MAX:
            output[index] = float(np.max(numeric_values[selected]))
        else:
            output[index] = float(np.min(numeric_values[selected]))
    return output


def _evaluate_composite_scores(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    spec: CompositeTaskSpec,
    target_table_id: str,
    cutoffs: np.ndarray,
) -> np.ndarray:
    """Combine every label aggregate into one per-target-row score vector."""
    terms = [
        _evaluate_aggregate_spec(
            schema, database, aggregate, target_table_id, cutoffs
        )
        for aggregate in spec.label_aggregates
    ]
    return _combine_aggregate_terms(terms, spec.combine_operator)


def _composite_score_state(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    spec: CompositeTaskSpec,
    target_table_id: str,
    cutoffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """``(scores, eligible_mask, eligibility_scores)`` for one composite spec."""
    scores = _evaluate_composite_scores(
        schema, database, spec, target_table_id, cutoffs
    )
    if spec.eligibility_aggregate is None:
        return (
            scores,
            np.ones(len(scores), dtype=bool),
            None,
        )
    eligibility = _evaluate_aggregate_spec(
        schema,
        database,
        spec.eligibility_aggregate,
        target_table_id,
        cutoffs,
    )
    eligible = _compare_values(
        eligibility, spec.eligibility_operator, spec.eligibility_threshold
    )
    return scores, eligible, eligibility


def _composite_cutoffs(
    schema: PhysicalSchema, database: DatabaseInstance, plan: TaskPlan
) -> np.ndarray:
    """Per-target-row cutoffs for a composite plan (global cutoff)."""
    if plan.row_cutoff_time_column_id is not None:
        return (
            database.table(plan.target_table_id)
            .column(plan.row_cutoff_time_column_id)
            .astype(np.int64)
        )
    return np.full(
        database.table(plan.target_table_id).row_count,
        int(plan.cutoff_time),
        dtype=np.int64,
    )


def composite_labels(
    schema: PhysicalSchema, database: DatabaseInstance, plan: TaskPlan
) -> np.ndarray:
    """Recompute composite labels exactly from the persisted composite_spec.

    Uses the stored label threshold/operator and never re-samples, so a fixed
    seed plus the plan reproduces labels bit-for-bit.
    """
    spec = plan.composite_spec
    if spec is None:
        raise ValueError("plan has no composite_spec")
    target_table_id = plan.target_table_id
    cutoffs = _composite_cutoffs(schema, database, plan)
    scores, eligible, _ = _composite_score_state(
        schema, database, spec, target_table_id, cutoffs
    )
    labels = np.full(
        database.table(target_table_id).row_count, -1, dtype=np.int8
    )
    labels[eligible] = _compare_values(
        scores[eligible], spec.label_operator, spec.label_threshold
    ).astype(np.int8)
    return labels


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeCandidate:
    """A legal composite path skeleton; the builder samples the rest.

    ``required_paths``, ``source_table_ids`` and ``time_column_ids`` align
    elementwise (one label aggregate per entry).  ``MULTI_SOURCE`` carries two
    or more paths; every other family carries exactly one.
    """

    target_table_id: str
    family: CompositeFamily
    required_paths: tuple[tuple[str, ...], ...]
    source_table_ids: tuple[str, ...]
    time_column_ids: tuple[str, ...]


def generate_composite_candidates(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    *,
    families: tuple[CompositeFamily, ...],
    max_path_depth: int = 3,
    candidate_limit: int = 256,
) -> tuple[CompositeCandidate, ...]:
    """Enumerate legal composite path skeletons without a full cross product.

    Only targets that are ENTITY or EVENT and endpoints that are EVENT/DETAIL
    carrying a TIME column are kept.  Per family the skeleton list is capped at
    ``candidate_limit``.
    """
    skeletons: list[tuple[str, tuple[str, ...], str, str]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for target in schema.tables:
        if target.role not in {TableRole.ENTITY, TableRole.EVENT}:
            continue
        for path, endpoint_id in _enumerate_paths(
            schema, target.table_id, max_depth=max_path_depth
        ):
            endpoint = schema.table(endpoint_id)
            if endpoint.role not in {TableRole.EVENT, TableRole.DETAIL}:
                continue
            time_column = _time_column(endpoint)
            if time_column is None:
                continue
            key = (target.table_id, path)
            if key in seen:
                continue
            seen.add(key)
            skeletons.append(
                (target.table_id, path, endpoint_id, time_column)
            )
    result: list[CompositeCandidate] = []
    for family in families:
        if family is CompositeFamily.MULTI_SOURCE:
            candidates = _multi_source_candidates(skeletons)
        elif family is CompositeFamily.MULTI_HOP_FILTERED:
            candidates = [
                CompositeCandidate(
                    target_table_id=target,
                    family=family,
                    required_paths=(path,),
                    source_table_ids=(endpoint,),
                    time_column_ids=(time_column,),
                )
                for (target, path, endpoint, time_column) in skeletons
                if len(path) >= 2
            ]
        else:
            candidates = [
                CompositeCandidate(
                    target_table_id=target,
                    family=family,
                    required_paths=(path,),
                    source_table_ids=(endpoint,),
                    time_column_ids=(time_column,),
                )
                for (target, path, endpoint, time_column) in skeletons
            ]
        if len(candidates) > candidate_limit:
            candidates = candidates[:candidate_limit]
        result.extend(candidates)
    return tuple(result)


def _multi_source_candidates(
    skeletons: list[tuple[str, tuple[str, ...], str, str]],
) -> list[CompositeCandidate]:
    """Pairs of skeletons sharing a target but with distinct source endpoints."""
    by_target: dict[str, list[tuple[str, tuple[str, ...], str, str]]] = {}
    for skeleton in skeletons:
        by_target.setdefault(skeleton[0], []).append(skeleton)
    result: list[CompositeCandidate] = []
    for target, group in by_target.items():
        for index in range(len(group)):
            for other in range(index + 1, len(group)):
                first, second = group[index], group[other]
                if first[2] == second[2]:
                    continue
                result.append(
                    CompositeCandidate(
                        target_table_id=target,
                        family=CompositeFamily.MULTI_SOURCE,
                        required_paths=(first[1], second[1]),
                        source_table_ids=(first[2], second[2]),
                        time_column_ids=(first[3], second[3]),
                    )
                )
    return result


def _sample_predicates(
    rng: np.random.Generator,
    schema: PhysicalSchema,
    database: DatabaseInstance,
    source_id: str,
    *,
    count: int,
    exclude_columns: set[str],
) -> tuple[PredicateSpec, ...]:
    """Sample up to *count* predicates from the source's feature columns."""
    table = schema.table(source_id)
    predicate_columns = [
        column.column_id
        for column in table.columns
        if column.kind is ColumnKind.FEATURE
        and column.column_id not in exclude_columns
    ]
    result: list[PredicateSpec] = []
    for _ in range(count):
        if not predicate_columns:
            break
        column_id = str(
            predicate_columns[int(rng.integers(0, len(predicate_columns)))]
        )
        values = _observed_values(
            database.table(source_id).column(column_id)
        )
        if not len(values):
            continue
        if values.dtype.kind in {"U", "S", "O"}:
            uniques = values.astype(str)
            unique_values = list(np.unique(uniques))
            if not unique_values:
                continue
            value = unique_values[int(rng.integers(0, len(unique_values)))]
            operator = (
                CompareOperator.EQ
                if rng.random() < 0.5
                else CompareOperator.NE
            )
        else:
            unique_values = list(np.unique(values))
            if len(unique_values) < 2:
                continue
            value = float(unique_values[int(rng.integers(0, len(unique_values)))])
            operator = (
                CompareOperator.LT,
                CompareOperator.LE,
                CompareOperator.GT,
                CompareOperator.GE,
            )[int(rng.integers(0, 4))]
        result.append(
            PredicateSpec(column_id=column_id, operator=operator, value=value)
        )
    return tuple(result)


def _sample_aggregate_spec(
    rng: np.random.Generator,
    schema: PhysicalSchema,
    database: DatabaseInstance,
    path: tuple[str, ...],
    source_id: str,
    time_column_id: str,
    cutoff: int,
    *,
    operators: tuple[AggregateOperator, ...],
    require_predicate: bool,
    max_predicates: int,
    window_kind: str,
) -> AggregateSpec | None:
    """Sample one aggregate (window, operator, value column, predicates).

    Windows are relative seconds to *cutoff* and must fit inside the source
    table's observed time range; an out-of-range window returns ``None`` so the
    planner retries instead of clipping (clipping would change label
    semantics).
    """
    times = np.asarray(
        database.table(source_id).column(time_column_id), dtype=np.float64
    )
    times = times[np.isfinite(times) & (times >= 0)]
    if len(times) < 1 or int(times.max()) <= int(times.min()):
        return None
    source_min = int(times.min())
    source_max = int(times.max())
    span = source_max - source_min

    if window_kind == "history":
        size = max(1, round(span * float(rng.uniform(0.05, 0.4))))
        window_start, window_end = -size, 0
    elif window_kind == "future":
        size = max(1, round(span * float(rng.uniform(0.05, 0.4))))
        window_start, window_end = 0, size
    else:
        if rng.random() < 0.5:
            size = max(1, round(span * float(rng.uniform(0.05, 0.4))))
            window_start, window_end = -size, 0
        else:
            size = max(1, round(span * float(rng.uniform(0.05, 0.4))))
            window_start, window_end = 0, size
    if cutoff + window_start < source_min or cutoff + window_end > source_max:
        return None

    operator = operators[int(rng.integers(0, len(operators)))]
    if operator is AggregateOperator.COUNT:
        value_column_id = None
        exclude_columns: set[str] = set()
    else:
        value_columns = _usable_feature_columns(schema, database, source_id)
        if not value_columns:
            return None
        value_column_id = str(
            value_columns[int(rng.integers(0, len(value_columns)))]
        )
        exclude_columns = {value_column_id}

    if require_predicate:
        predicate_count = max(1, int(rng.integers(1, max_predicates + 1)))
    elif max_predicates >= 1:
        predicate_count = int(rng.integers(0, max_predicates + 1))
    else:
        predicate_count = 0
    predicates = _sample_predicates(
        rng,
        schema,
        database,
        source_id,
        count=predicate_count,
        exclude_columns=exclude_columns,
    )
    if require_predicate and not predicates:
        return None
    return AggregateSpec(
        source_table_id=source_id,
        required_path=path,
        time_column_id=time_column_id,
        window_start=window_start,
        window_end=window_end,
        operator=operator,
        value_column_id=value_column_id,
        predicates=predicates,
    )


def _sample_composite_spec(
    rng: np.random.Generator,
    schema: PhysicalSchema,
    database: DatabaseInstance,
    candidate: CompositeCandidate,
    cutoff: int,
    *,
    max_predicates: int,
) -> CompositeTaskSpec | None:
    """Sample the family-specific composite spec for one candidate.

    Returns a spec with a placeholder ``label_threshold`` (0.0) and the final
    ``label_operator``; the builder replaces the threshold from the eligible
    score distribution.
    """
    family = candidate.family
    if family is CompositeFamily.FILTERED_AGGREGATE:
        aggregate = _sample_aggregate_spec(
            rng, schema, database, candidate.required_paths[0],
            candidate.source_table_ids[0], candidate.time_column_ids[0], cutoff,
            operators=(
                AggregateOperator.COUNT,
                AggregateOperator.SUM,
                AggregateOperator.MEAN,
                AggregateOperator.MIN,
                AggregateOperator.MAX,
            ),
            require_predicate=False,
            max_predicates=max_predicates,
            window_kind="either",
        )
        if aggregate is None:
            return None
        label_operator = _label_operator_for(aggregate.operator)
        return CompositeTaskSpec(
            family=family,
            label_aggregates=(aggregate,),
            combine_operator=CombineOperator.SUM,
            label_operator=label_operator,
            label_threshold=0.0,
        )
    if family is CompositeFamily.COUNT_DISTINCT:
        aggregate = _sample_aggregate_spec(
            rng, schema, database, candidate.required_paths[0],
            candidate.source_table_ids[0], candidate.time_column_ids[0], cutoff,
            operators=(AggregateOperator.COUNT_DISTINCT,),
            require_predicate=False,
            max_predicates=max_predicates,
            window_kind="either",
        )
        if aggregate is None:
            return None
        return CompositeTaskSpec(
            family=family,
            label_aggregates=(aggregate,),
            combine_operator=CombineOperator.SUM,
            label_operator=LabelOperator.GE,
            label_threshold=0.0,
        )
    if family is CompositeFamily.QUANTIFIED_EVENT:
        aggregate = _sample_aggregate_spec(
            rng, schema, database, candidate.required_paths[0],
            candidate.source_table_ids[0], candidate.time_column_ids[0], cutoff,
            operators=(AggregateOperator.COUNT,),
            require_predicate=True,
            max_predicates=max_predicates,
            window_kind="future",
        )
        if aggregate is None:
            return None
        return CompositeTaskSpec(
            family=family,
            label_aggregates=(aggregate,),
            combine_operator=CombineOperator.SUM,
            label_operator=LabelOperator.GT,
            label_threshold=0.0,
        )
    if family is CompositeFamily.MULTI_SOURCE:
        aggregates: list[AggregateSpec] = []
        for path, source_id, time_column_id in zip(
            candidate.required_paths,
            candidate.source_table_ids,
            candidate.time_column_ids,
        ):
            aggregate = _sample_aggregate_spec(
                rng, schema, database, path, source_id, time_column_id, cutoff,
                operators=(
                    AggregateOperator.COUNT,
                    AggregateOperator.SUM,
                    AggregateOperator.MEAN,
                    AggregateOperator.MIN,
                    AggregateOperator.MAX,
                ),
                require_predicate=False,
                max_predicates=max_predicates,
                window_kind="either",
            )
            if aggregate is None:
                return None
            aggregates.append(aggregate)
        combine_operator = (
            CombineOperator.SUM,
            CombineOperator.MEAN,
            CombineOperator.MAX,
            CombineOperator.MIN,
        )[int(rng.integers(0, 4))]
        label_operator = _label_operator_for(aggregates[0].operator)
        return CompositeTaskSpec(
            family=family,
            label_aggregates=tuple(aggregates),
            combine_operator=combine_operator,
            label_operator=label_operator,
            label_threshold=0.0,
        )
    if family is CompositeFamily.MULTI_HOP_FILTERED:
        aggregate = _sample_aggregate_spec(
            rng, schema, database, candidate.required_paths[0],
            candidate.source_table_ids[0], candidate.time_column_ids[0], cutoff,
            operators=(
                AggregateOperator.COUNT,
                AggregateOperator.COUNT_DISTINCT,
            ),
            require_predicate=True,
            max_predicates=max_predicates,
            window_kind="either",
        )
        if aggregate is None:
            return None
        return CompositeTaskSpec(
            family=family,
            label_aggregates=(aggregate,),
            combine_operator=CombineOperator.SUM,
            label_operator=_label_operator_for(aggregate.operator),
            label_threshold=0.0,
        )
    # HISTORY_CONDITIONED_FUTURE
    future = _sample_aggregate_spec(
        rng, schema, database, candidate.required_paths[0],
        candidate.source_table_ids[0], candidate.time_column_ids[0], cutoff,
        operators=(
            AggregateOperator.COUNT,
            AggregateOperator.SUM,
            AggregateOperator.MEAN,
        ),
        require_predicate=False,
        max_predicates=max_predicates,
        window_kind="future",
    )
    if future is None:
        return None
    history = _sample_aggregate_spec(
        rng, schema, database, candidate.required_paths[0],
        candidate.source_table_ids[0], candidate.time_column_ids[0], cutoff,
        operators=(AggregateOperator.COUNT,),
        require_predicate=False,
        max_predicates=0,
        window_kind="history",
    )
    if history is None:
        return None
    return CompositeTaskSpec(
        family=family,
        label_aggregates=(future,),
        combine_operator=CombineOperator.SUM,
        label_operator=_label_operator_for(future.operator),
        label_threshold=0.0,
        eligibility_aggregate=history,
        eligibility_operator=LabelOperator.GT,
        eligibility_threshold=0.0,
    )


def _label_operator_for(operator: AggregateOperator) -> LabelOperator:
    """Builder thresholding mirrors ``_threshold_labels`` (strict ``>``).

    A strict threshold splits discrete count scores cleanly at 0; a ``>=``
    threshold would label every row positive when the sampled quantile is 0.
    ``GE`` stays available in the DSL for hand-authored tasks.
    """
    return LabelOperator.GT


def build_composite_relational_classification_task(
    *,
    task_id: str,
    sample_id: str,
    schema: PhysicalSchema,
    database: DatabaseInstance,
    candidate: CompositeCandidate,
    seed: int,
    support_fraction: float,
    min_support_rows: int,
    min_query_rows: int,
    min_class_count_per_split: int,
    positive_rate_min: float = 0.2,
    positive_rate_max: float = 0.8,
    max_predicates: int = 2,
) -> PlannedTask | None:
    """Build one composite relational-classification task from a candidate.

    A single global cutoff is sampled inside the pooled source time range so
    observation rules can cut every temporal Event/Detail table at exactly the
    cutoff, guaranteeing that no future information reaches the observation
    view.  Eligibility is computed first; only eligible rows receive labels,
    the composite score is thresholded from the eligible-score quantile, and a
    degenerate single-class label set returns ``None`` so the planner retries.
    """
    rng = _rng(seed)
    target_table_id = candidate.target_table_id
    row_count = database.table(target_table_id).row_count
    if row_count == 0:
        return None

    pooled = [
        np.asarray(
            database.table(source_id).column(time_column_id), dtype=np.float64
        )
        for source_id, time_column_id in zip(
            candidate.source_table_ids, candidate.time_column_ids
        )
    ]
    pooled = [
        times[np.isfinite(times) & (times >= 0)] for times in pooled
    ]
    pooled = [times for times in pooled if len(times)]
    if not pooled:
        return None
    all_times = np.concatenate(pooled).astype(np.int64)
    if len(all_times) < 2 or int(all_times.max()) <= int(all_times.min()):
        return None
    cutoff_quantile = float(rng.uniform(0.35, 0.65))
    cutoff = int(np.quantile(all_times, cutoff_quantile))
    if cutoff <= int(all_times.min()) or cutoff >= int(all_times.max()):
        return None
    cutoffs = np.full(row_count, cutoff, dtype=np.int64)

    spec = _sample_composite_spec(
        rng, schema, database, candidate, cutoff,
        max_predicates=max_predicates,
    )
    if spec is None:
        return None

    scores, eligible, _ = _composite_score_state(
        schema, database, spec, target_table_id, cutoffs
    )
    eligible_count = int(np.count_nonzero(eligible))
    if eligible_count == 0:
        return None
    desired = float(rng.uniform(positive_rate_min, positive_rate_max))
    threshold = float(np.quantile(scores[eligible], 1.0 - desired))
    spec = replace(
        spec,
        label_threshold=threshold,
    )
    labels = np.full(row_count, -1, dtype=np.int8)
    labels[eligible] = _compare_values(
        scores[eligible], spec.label_operator, spec.label_threshold
    ).astype(np.int8)
    if len(np.unique(labels[eligible])) < 2:
        return None

    split = _stratified_split(
        labels[eligible],
        rng,
        support_fraction=support_fraction,
        min_support_rows=min_support_rows,
        min_query_rows=min_query_rows,
        min_class_count=min_class_count_per_split,
    )
    if split is None:
        return None
    valid_support, valid_query = split
    eligible_rows = np.flatnonzero(eligible).astype(np.int64)
    support = eligible_rows[valid_support]
    query = eligible_rows[valid_query]

    route_depth = max(
        (len(path) for path in candidate.required_paths), default=1
    )
    plan = TaskPlan(
        task_id=task_id,
        sample_id=sample_id,
        instance_id=database.instance_id,
        schema_id=schema.schema_id,
        mechanism=TaskMechanism.RELATIONAL_CLASSIFICATION,
        prediction_type=PredictionType.CLASSIFICATION,
        target_table_id=target_table_id,
        source_table_id=candidate.source_table_ids[0],
        target_column_id=_SYNTHETIC_TARGET,
        source_column_id=None,
        time_column_id=candidate.time_column_ids[0],
        cutoff_time=cutoff,
        split_strategy="stratified_rows",
        seed=seed,
        masked_column_ids=(_SYNTHETIC_TARGET,),
        observation_rules=_observation_rules(schema, cutoff),
        route_supervision=_schema_route_labels(
            schema,
            target_table_id=target_table_id,
            required_paths=candidate.required_paths,
            max_depth=max(2, route_depth),
        ),
        classification_kind=ClassificationKind.BINARY,
        requested_positive_rate=desired,
        realized_positive_rate=float(np.mean(labels[eligible])),
        parameters=(("support_fraction", support_fraction),),
        composite_spec=spec,
    )
    return _planned_if_visible(
        schema, database, plan, labels, support, query
    )


def _traverse_path(
    schema: PhysicalSchema, database: DatabaseInstance,
    target_table_id: str, path: tuple[str, ...],
) -> tuple[list[np.ndarray], str]:
    groups = [np.asarray([row], dtype=np.int64) for row in range(database.table(target_table_id).row_count)]
    current = target_table_id
    for fk_id in path:
        fk = _foreign_key(schema, fk_id)
        if current == fk.child_table_id:
            assignments = database.table(current).column(fk.child_column_id)
            groups = [
                np.unique(assignments[rows][assignments[rows] >= 0]).astype(np.int64)
                for rows in groups
            ]
            current = fk.parent_table_id
        elif current == fk.parent_table_id:
            assignments = database.table(fk.child_table_id).column(fk.child_column_id)
            child_by_parent = [np.flatnonzero(assignments == row).astype(np.int64) for row in range(database.table(current).row_count)]
            groups = [
                np.unique(np.concatenate([child_by_parent[row] for row in rows])).astype(np.int64)
                if len(rows) else np.asarray([], dtype=np.int64)
                for rows in groups
            ]
            current = fk.child_table_id
        else:
            raise ValueError("required path is not contiguous from target table")
    return groups, current


def _threshold_labels(values: np.ndarray, positive_rate: float) -> tuple[np.ndarray, float]:
    threshold = float(np.quantile(values, 1.0 - positive_rate))
    return (values > threshold).astype(np.int8), threshold


def _categorical_quantiles(values: np.ndarray, class_count: int) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), dtype=np.int16)
    labels[order] = np.minimum(
        class_count - 1,
        (np.arange(len(values), dtype=np.int64) * class_count) // max(1, len(values)),
    )
    return labels


def _stratified_split(
    labels: np.ndarray, rng: np.random.Generator, *, support_fraction: float,
    min_support_rows: int, min_query_rows: int, min_class_count: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    groups = [np.flatnonzero(labels == value).astype(np.int64) for value in np.unique(labels)]
    if len(groups) < 2 or any(len(group) < 2 * min_class_count for group in groups):
        return None
    support_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []
    for group in groups:
        rng.shuffle(group)
        count = min(len(group) - min_class_count, max(min_class_count, round(len(group) * support_fraction)))
        support_parts.append(group[:count])
        query_parts.append(group[count:])
    support = np.concatenate(support_parts)
    query = np.concatenate(query_parts)
    rng.shuffle(support)
    rng.shuffle(query)
    if len(support) < min_support_rows or len(query) < min_query_rows:
        return None
    return support, query


def _planned(plan: TaskPlan, labels: np.ndarray, support: np.ndarray, query: np.ndarray) -> PlannedTask:
    return PlannedTask(
        plan=plan,
        data=TaskData(
            support_row_ids=support, support_labels=labels[support],
            query_row_ids=query, query_labels=labels[query],
        ),
    )


def _planned_if_visible(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    plan: TaskPlan,
    labels: np.ndarray,
    support: np.ndarray,
    query: np.ndarray,
) -> PlannedTask | None:
    task = _planned(plan, labels, support, query)
    target_mask = build_task_view(
        schema, database, plan
    ).row_masks[plan.target_table_id]
    supervised_rows = np.concatenate((support, query))
    if not np.all(target_mask[supervised_rows]):
        return None
    return task


def _usable_feature_columns(
    schema: PhysicalSchema, database: DatabaseInstance, table_id: str
) -> tuple[str, ...]:
    table = schema.table(table_id)
    return tuple(
        column.column_id for column in table.columns
        if column.kind is ColumnKind.FEATURE
        and len(_observed_values(database.table(table_id).column(column.column_id)))
        and len(np.unique(_observed_values(database.table(table_id).column(column.column_id)))) > 1
    )


def _time_column(table) -> str | None:
    return next((column.column_id for column in table.columns if column.kind is ColumnKind.TIME), None)


def _numeric(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"i", "u", "f", "b"}:
        result = values.astype(np.float64)
        if result.dtype.kind == "f":
            finite = np.isfinite(result)
            result[~finite] = float(np.mean(result[finite])) if np.any(finite) else 0.0
        return result
    uniques = sorted(value for value in np.unique(values).tolist() if value != "")
    mapping = {value: index for index, value in enumerate(uniques)}
    return np.asarray([mapping.get(value, -1) for value in values], dtype=np.float64)


def _observed_mask(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "f":
        return np.isfinite(values)
    if values.dtype.kind in {"U", "S"}:
        return values != ""
    return np.ones(len(values), dtype=bool)


def _observed_values(values: np.ndarray) -> np.ndarray:
    return values[_observed_mask(values)]


def _observation_rules(schema: PhysicalSchema, cutoff: int) -> tuple[ObservationRule, ...]:
    return tuple(
        ObservationRule(table_id=table.table_id, time_column_id=column.column_id, max_timestamp=cutoff)
        for table in schema.tables
        for column in table.columns
        if table.role in {TableRole.EVENT, TableRole.DETAIL} and column.kind is ColumnKind.TIME
    )


def _has_incoming_fk(schema: PhysicalSchema, table_id: str) -> bool:
    """Return True when *table_id* has at least one incoming FK edge."""
    return any(
        fk.child_table_id == table_id for fk in schema.foreign_keys
    )


def _foreign_key(schema: PhysicalSchema, foreign_key_id: str) -> PhysicalForeignKey:
    return next(fk for fk in schema.foreign_keys if fk.foreign_key_id == foreign_key_id)


def _enumerate_paths(
    schema: PhysicalSchema, target_table_id: str, *, max_depth: int
) -> tuple[tuple[tuple[str, ...], str], ...]:
    adjacent: dict[str, list[tuple[str, str]]] = {table.table_id: [] for table in schema.tables}
    for fk in schema.foreign_keys:
        adjacent[fk.parent_table_id].append((fk.foreign_key_id, fk.child_table_id))
        adjacent[fk.child_table_id].append((fk.foreign_key_id, fk.parent_table_id))
    frontier = [(target_table_id, (), frozenset({target_table_id}))]
    result: list[tuple[tuple[str, ...], str]] = []
    for _ in range(max_depth):
        following = []
        for current, path, visited in frontier:
            for fk_id, destination in sorted(adjacent[current]):
                if destination in visited:
                    continue
                candidate = path + (fk_id,)
                result.append((candidate, destination))
                following.append((destination, candidate, visited | {destination}))
        frontier = following
    return tuple(result)


def _schema_route_labels(
    schema: PhysicalSchema, *, target_table_id: str,
    required_paths: tuple[tuple[str, ...], ...] = (),
    optional_paths: tuple[tuple[str, ...], ...] = (), max_depth: int = 2,
) -> tuple[RoutePathLabel, ...]:
    required = set(required_paths)
    optional = set(optional_paths)
    paths = [path for path, _endpoint in _enumerate_paths(schema, target_table_id, max_depth=max_depth)]
    missing = required - set(paths)
    if missing:
        raise ValueError(f"required paths are not legal schema paths: {sorted(missing)!r}")
    return tuple(
        RoutePathLabel(
            foreign_key_ids=path,
            role=(RouteRole.REQUIRED if path in required else RouteRole.OPTIONAL if path in optional else RouteRole.DISTRACTOR),
        )
        for path in paths
    )


def _temporal_split(
    labels: np.ndarray,
    ordered: np.ndarray,
    rng: np.random.Generator,
    *,
    support_fraction: float,
    min_support_rows: int,
    min_query_rows: int,
    min_class_count: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Support/query split respecting temporal order for cutoff_time export."""
    class_values = np.unique(labels)
    groups = [np.flatnonzero(labels == value).astype(np.int64) for value in class_values]
    if len(groups) < 2 or any(len(group) < 2 * min_class_count for group in groups):
        return None
    support_count = max(min_support_rows, round(len(ordered) * support_fraction))
    support_count = min(support_count, len(ordered) - min_query_rows)
    if support_count < min_support_rows:
        return None
    support = ordered[:support_count].copy()
    query = ordered[support_count:].copy()
    if len(query) < min_query_rows:
        return None
    for rows in (support, query):
        if any(
            np.count_nonzero(labels[rows] == value) < min_class_count
            for value in class_values
        ):
            return None
    rng.shuffle(support)
    rng.shuffle(query)
    return support, query


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(seed))


def _label_rng(seed: int) -> np.random.Generator:
    """Dedicated label stream for stochastic interaction-response labels.

    Derived from the int plan seed so mechanism_labels can replay the exact
    same per-row draws independently of the builder's main RNG. Never stored
    in TaskPlan.parameters because those values are coerced to float and a
    64-bit seed loses precision above 2^53.
    """
    return _rng(seed ^ _LABEL_SEED_MAGIC)


def _logit(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


__all__ = [
    "TargetSample",
    "RelationAttributeCandidate", "RandomColumnCandidate", "FutureEventCandidate",
    "FutureEventAttributeCandidate", "TemporalAggregateCandidate",
    "CompositeCandidate",
    "relation_attribute_candidates", "future_event_candidates",
    "random_column_candidates",
    "future_event_attribute_candidates", "temporal_aggregate_candidates",
    "generate_composite_candidates",
    "build_relation_attribute_task", "build_random_column_task",
    "build_future_event_existence_task",
    "build_history_gated_future_activity_task",
    "build_history_gated_future_inactive_task",
    "build_history_gated_future_active_task",
    "build_future_event_attribute_condition_task",
    "build_temporal_relational_aggregate_task",
    "build_composite_relational_classification_task",
    "composite_labels", "mechanism_labels",
    "future_event_labels",
]
