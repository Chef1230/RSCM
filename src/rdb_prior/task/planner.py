"""Deterministic eligibility matching and task sampling for stage 03."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging

import numpy as np

from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.instance.plan import InstancePlan
from rdb_prior.runtime import RuntimeContext
from rdb_prior.task.mechanisms import (
    build_composite_relational_classification_task,
    build_interaction_response_task,
    build_future_event_attribute_condition_task,
    build_future_event_existence_task,
    build_history_gated_future_activity_task,
    build_history_gated_future_inactive_task,
    build_history_gated_future_active_task,
    build_relation_attribute_task,
    build_temporal_relational_aggregate_task,
    generate_composite_candidates,
    interaction_candidates,
    future_event_attribute_candidates,
    future_event_candidates,
    relation_attribute_candidates,
    temporal_aggregate_candidates,
)
from rdb_prior.task.model import (
    CompositeFamily,
    PlannedTask,
    TaskMechanism,
)


_DEFAULT_MECHANISM_WEIGHTS = (
    (TaskMechanism.RELATION_ATTRIBUTE, 0.35),
    (TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE, 0.25),
    (TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE, 0.08),
    (TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE, 0.07),
    (TaskMechanism.FUTURE_EVENT_ATTRIBUTE_CONDITION, 0.20),
    (TaskMechanism.TEMPORAL_RELATIONAL_AGGREGATE, 0.20),
    (TaskMechanism.INTERACTION_RESPONSE, 0.15),
)


_LOGGER = logging.getLogger(__name__)


_DEFAULT_COMPOSITE_FAMILY_WEIGHTS = (
    (CompositeFamily.FILTERED_AGGREGATE, 0.25),
    (CompositeFamily.COUNT_DISTINCT, 0.15),
    (CompositeFamily.QUANTIFIED_EVENT, 0.15),
    (CompositeFamily.MULTI_SOURCE, 0.15),
    (CompositeFamily.MULTI_HOP_FILTERED, 0.15),
    (CompositeFamily.HISTORY_CONDITIONED_FUTURE, 0.15),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskPlannerConfig:
    tasks_per_database: int = 2
    mechanism_weights: tuple[tuple[TaskMechanism, float], ...] = _DEFAULT_MECHANISM_WEIGHTS
    support_fraction: float = 0.7
    min_support_rows: int = 32
    min_query_rows: int = 16
    min_class_count_per_split: int = 2
    max_classification_categories: int = 12
    cutoff_quantile_min: float = 0.45
    cutoff_quantile_max: float = 0.7
    horizon_fraction_min: float = 0.12
    horizon_fraction_max: float = 0.3
    positive_rate_min: float = 0.2
    positive_rate_max: float = 0.8
    history_gated_frequency_weight_min: float = 0.5
    history_gated_frequency_weight_max: float = 3.0
    history_gated_silence_weight_min: float = 0.5
    history_gated_silence_weight_max: float = 3.0
    window_repeated_probability: float = 0.25
    window_short_probability: float = 0.5
    window_short_fraction_min: float = 0.05
    window_short_fraction_max: float = 0.15
    window_long_fraction_min: float = 0.30
    window_long_fraction_max: float = 0.60
    interaction_u_weight_min: float = 0.25
    interaction_u_weight_max: float = 2.0
    interaction_frequency_weight_min: float = 0.5
    interaction_frequency_weight_max: float = 3.0
    interaction_silence_weight_min: float = 0.5
    interaction_silence_weight_max: float = 3.0
    interaction_item_weight_min: float = 0.5
    interaction_item_weight_max: float = 3.0
    interaction_invert_probability: float = 0.35
    composite_family_weights: tuple[
        tuple[CompositeFamily, float], ...
    ] = _DEFAULT_COMPOSITE_FAMILY_WEIGHTS
    composite_max_path_depth: int = 3
    composite_max_predicates: int = 2
    composite_candidate_limit: int = 256
    max_attempts_per_database: int = 128
    require_full_task_count: bool = True

    def __post_init__(self) -> None:
        for name in (
            "tasks_per_database", "min_support_rows", "min_query_rows",
            "min_class_count_per_split", "max_classification_categories",
            "max_attempts_per_database",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.support_fraction < 1:
            raise ValueError("support_fraction must be in (0, 1)")
        for name, low, high in (
            ("cutoff quantile", self.cutoff_quantile_min, self.cutoff_quantile_max),
            ("horizon fraction", self.horizon_fraction_min, self.horizon_fraction_max),
            ("positive rate", self.positive_rate_min, self.positive_rate_max),
        ):
            _fraction_range(name, low, high)
        for name, low, high in (
            (
                "history gated frequency weight",
                self.history_gated_frequency_weight_min,
                self.history_gated_frequency_weight_max,
            ),
            (
                "history gated silence weight",
                self.history_gated_silence_weight_min,
                self.history_gated_silence_weight_max,
            ),
        ):
            _weight_range(name, low, high)
        _fraction_range(
            "window short fraction",
            self.window_short_fraction_min,
            self.window_short_fraction_max,
        )
        _fraction_range(
            "window long fraction",
            self.window_long_fraction_min,
            self.window_long_fraction_max,
        )
        for name, low, high in (
            (
                "interaction user weight",
                self.interaction_u_weight_min,
                self.interaction_u_weight_max,
            ),
            (
                "interaction frequency weight",
                self.interaction_frequency_weight_min,
                self.interaction_frequency_weight_max,
            ),
            (
                "interaction silence weight",
                self.interaction_silence_weight_min,
                self.interaction_silence_weight_max,
            ),
            (
                "interaction item weight",
                self.interaction_item_weight_min,
                self.interaction_item_weight_max,
            ),
        ):
            _weight_range(name, low, high)
        for name, value in (
            ("window repeated probability", self.window_repeated_probability),
            ("window short probability", self.window_short_probability),
            ("interaction invert probability", self.interaction_invert_probability),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not 0 < value < 1:
                raise ValueError(f"{name} must be in (0, 1)")
        if self.window_short_fraction_max > self.window_long_fraction_min:
            raise ValueError(
                "window_short_fraction_max must not exceed window_long_fraction_min"
            )
        if not isinstance(self.require_full_task_count, bool):
            raise TypeError("require_full_task_count must be a boolean")
        for name in ("composite_max_path_depth", "composite_candidate_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.composite_max_predicates, bool) or not isinstance(
            self.composite_max_predicates, int
        ):
            raise TypeError("composite_max_predicates must be an integer")
        if self.composite_max_predicates < 0:
            raise ValueError("composite_max_predicates must be non-negative")
        if (
            not isinstance(self.composite_family_weights, tuple)
            or not self.composite_family_weights
        ):
            raise ValueError(
                "composite_family_weights must be a non-empty tuple"
            )
        families = tuple(item[0] for item in self.composite_family_weights)
        if len(set(families)) != len(families):
            raise ValueError(
                "composite_family_weights contains duplicate families"
            )
        for family, weight in self.composite_family_weights:
            if not isinstance(family, CompositeFamily):
                raise TypeError(
                    "composite_family_weights keys must be CompositeFamily"
                )
            if weight <= 0:
                raise ValueError("composite family weights must be positive")
        if not isinstance(self.mechanism_weights, tuple) or not self.mechanism_weights:
            raise ValueError("mechanism_weights must be a non-empty tuple")
        mechanisms = tuple(item[0] for item in self.mechanism_weights)
        if len(set(mechanisms)) != len(mechanisms):
            raise ValueError("mechanism_weights contains duplicate mechanisms")
        for mechanism, weight in self.mechanism_weights:
            if not isinstance(mechanism, TaskMechanism):
                raise TypeError("mechanism_weights keys must be TaskMechanism")
            if weight <= 0:
                raise ValueError("mechanism weights must be positive")


class TaskPlanner:
    def __init__(self, config: TaskPlannerConfig | None = None) -> None:
        self.config = config or TaskPlannerConfig()

    def generate(
        self, *, sample_id: str, schema: PhysicalSchema,
        database: DatabaseInstance, runtime: RuntimeContext,
        instance_plan: InstancePlan | None = None,
    ) -> tuple[PlannedTask, ...]:
        pools: dict[TaskMechanism, list[object]] = {
            TaskMechanism.RELATION_ATTRIBUTE: list(
                relation_attribute_candidates(
                    schema, database,
                    max_classification_categories=self.config.max_classification_categories,
                )
            ),
            TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE: list(future_event_candidates(schema)),
            TaskMechanism.HISTORY_GATED_FUTURE_ACTIVITY: list(future_event_candidates(schema)),
            TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE: list(future_event_candidates(schema)),
            TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE: list(future_event_candidates(schema)),
            TaskMechanism.FUTURE_EVENT_ATTRIBUTE_CONDITION: list(
                future_event_attribute_candidates(schema, database)
            ),
            TaskMechanism.TEMPORAL_RELATIONAL_AGGREGATE: list(
                temporal_aggregate_candidates(schema, database)
            ),
            TaskMechanism.INTERACTION_RESPONSE: list(
                interaction_candidates(schema, database)
            ),
        }
        composite_pools = _composite_candidate_pools(schema, database, self.config)
        pools[TaskMechanism.RELATIONAL_CLASSIFICATION] = [
            candidate
            for candidates in composite_pools.values()
            for candidate in candidates
        ]
        rng = runtime.numpy_rng("task-selection")
        for pool in pools.values():
            rng.shuffle(pool)
        mechanisms, weights = zip(*self.config.mechanism_weights)
        generated: list[PlannedTask] = []
        signatures: set[tuple[object, ...]] = set()
        calendar_rejections = 0
        for attempt in range(self.config.max_attempts_per_database):
            if len(generated) >= self.config.tasks_per_database:
                break
            available = [mechanism for mechanism in mechanisms if pools.get(mechanism)]
            if not available:
                break
            available_weights = [weights[mechanisms.index(mechanism)] for mechanism in available]
            mechanism = available[int(rng.choice(len(available), p=_normalize(available_weights)))]
            candidate_pool = pools[mechanism]
            candidate = candidate_pool[int(rng.integers(0, len(candidate_pool)))]
            task_index = len(generated)
            common = dict(
                task_id=f"task_{sample_id}_{task_index:03d}", sample_id=sample_id,
                schema=schema, database=database, candidate=candidate,
                seed=runtime.seed("task", task_index, "attempt", attempt),
                support_fraction=self.config.support_fraction,
                min_support_rows=self.config.min_support_rows,
                min_query_rows=self.config.min_query_rows,
                min_class_count_per_split=self.config.min_class_count_per_split,
            )
            if mechanism is TaskMechanism.RELATION_ATTRIBUTE:
                task = build_relation_attribute_task(
                    **common, positive_rate_min=self.config.positive_rate_min,
                    positive_rate_max=self.config.positive_rate_max,
                )
            elif mechanism is TaskMechanism.ENTITY_FUTURE_EVENT_EXISTENCE:
                task = build_future_event_existence_task(
                    **common, cutoff_quantile_min=self.config.cutoff_quantile_min,
                    cutoff_quantile_max=self.config.cutoff_quantile_max,
                    horizon_fraction_min=self.config.horizon_fraction_min,
                    horizon_fraction_max=self.config.horizon_fraction_max,
                    positive_rate_min=self.config.positive_rate_min,
                    positive_rate_max=min(self.config.positive_rate_max, 0.8),
                )
            elif mechanism is TaskMechanism.HISTORY_GATED_FUTURE_ACTIVITY:
                task = build_history_gated_future_activity_task(
                    **common, cutoff_quantile_min=self.config.cutoff_quantile_min,
                    cutoff_quantile_max=self.config.cutoff_quantile_max,
                    horizon_fraction_min=self.config.horizon_fraction_min,
                    horizon_fraction_max=self.config.horizon_fraction_max,
                    positive_rate_min=self.config.positive_rate_min,
                    positive_rate_max=min(self.config.positive_rate_max, 0.8),
                )
            elif mechanism is TaskMechanism.HISTORY_GATED_FUTURE_INACTIVE:
                task = build_history_gated_future_inactive_task(
                    **common, cutoff_quantile_min=self.config.cutoff_quantile_min,
                    cutoff_quantile_max=self.config.cutoff_quantile_max,
                    horizon_fraction_min=self.config.horizon_fraction_min,
                    horizon_fraction_max=self.config.horizon_fraction_max,
                    positive_rate_min=self.config.positive_rate_min,
                    positive_rate_max=min(self.config.positive_rate_max, 0.8),
                    history_gated_frequency_weight_min=self.config.history_gated_frequency_weight_min,
                    history_gated_frequency_weight_max=self.config.history_gated_frequency_weight_max,
                    history_gated_silence_weight_min=self.config.history_gated_silence_weight_min,
                    history_gated_silence_weight_max=self.config.history_gated_silence_weight_max,
                )
            elif mechanism is TaskMechanism.HISTORY_GATED_FUTURE_ACTIVE:
                task = build_history_gated_future_active_task(
                    **common, cutoff_quantile_min=self.config.cutoff_quantile_min,
                    cutoff_quantile_max=self.config.cutoff_quantile_max,
                    horizon_fraction_min=self.config.horizon_fraction_min,
                    horizon_fraction_max=self.config.horizon_fraction_max,
                    positive_rate_min=self.config.positive_rate_min,
                    positive_rate_max=min(self.config.positive_rate_max, 0.8),
                    history_gated_frequency_weight_min=self.config.history_gated_frequency_weight_min,
                    history_gated_frequency_weight_max=self.config.history_gated_frequency_weight_max,
                    history_gated_silence_weight_min=self.config.history_gated_silence_weight_min,
                    history_gated_silence_weight_max=self.config.history_gated_silence_weight_max,
                )
            elif mechanism is TaskMechanism.FUTURE_EVENT_ATTRIBUTE_CONDITION:
                task = build_future_event_attribute_condition_task(
                    **common, positive_rate_min=self.config.positive_rate_min,
                    positive_rate_max=self.config.positive_rate_max,
                )
            elif mechanism is TaskMechanism.INTERACTION_RESPONSE:
                task = build_interaction_response_task(
                    **common, positive_rate_min=self.config.positive_rate_min,
                    positive_rate_max=self.config.positive_rate_max,
                    interaction_u_weight_min=self.config.interaction_u_weight_min,
                    interaction_u_weight_max=self.config.interaction_u_weight_max,
                    interaction_frequency_weight_min=self.config.interaction_frequency_weight_min,
                    interaction_frequency_weight_max=self.config.interaction_frequency_weight_max,
                    interaction_silence_weight_min=self.config.interaction_silence_weight_min,
                    interaction_silence_weight_max=self.config.interaction_silence_weight_max,
                    interaction_item_weight_min=self.config.interaction_item_weight_min,
                    interaction_item_weight_max=self.config.interaction_item_weight_max,
                    interaction_invert_probability=self.config.interaction_invert_probability,
                )
            elif mechanism is TaskMechanism.RELATIONAL_CLASSIFICATION:
                task = _generate_composite_task(
                    task_id=common["task_id"], sample_id=common["sample_id"],
                    schema=schema, database=database, seed=common["seed"],
                    support_fraction=common["support_fraction"],
                    min_support_rows=common["min_support_rows"],
                    min_query_rows=common["min_query_rows"],
                    min_class_count_per_split=common["min_class_count_per_split"],
                    composite_pools=composite_pools,
                    family_weights=dict(self.config.composite_family_weights),
                    rng=rng,
                    max_predicates=self.config.composite_max_predicates,
                    positive_rate_min=self.config.positive_rate_min,
                    positive_rate_max=self.config.positive_rate_max,
                )
            else:
                task = build_temporal_relational_aggregate_task(
                    **common, cutoff_quantile_min=self.config.cutoff_quantile_min,
                    cutoff_quantile_max=self.config.cutoff_quantile_max,
                    window_repeated_probability=self.config.window_repeated_probability,
                    window_short_probability=self.config.window_short_probability,
                    window_short_fraction_min=self.config.window_short_fraction_min,
                    window_short_fraction_max=self.config.window_short_fraction_max,
                    window_long_fraction_min=self.config.window_long_fraction_min,
                    window_long_fraction_max=self.config.window_long_fraction_max,
                    positive_rate_min=self.config.positive_rate_min,
                    positive_rate_max=self.config.positive_rate_max,
                )
            if task is None:
                continue
            task = _bind_instance_calendar(task, instance_plan)
            if task is None:
                calendar_rejections += 1
                continue
            if task.plan.signature in signatures:
                continue
            signatures.add(task.plan.signature)
            generated.append(task)
        if self.config.require_full_task_count and len(generated) != self.config.tasks_per_database:
            detail = (
                f"; rejected {calendar_rejections} candidate(s) whose time "
                "fields were outside the instance calendar"
                if calendar_rejections
                else ""
            )
            raise ValueError(
                f"database {sample_id!r} yielded {len(generated)} valid tasks; "
                f"required {self.config.tasks_per_database}{detail}"
            )
        return tuple(generated)


def _composite_candidate_pools(
    schema: PhysicalSchema,
    database: DatabaseInstance,
    config: TaskPlannerConfig,
) -> dict[CompositeFamily, list[object]]:
    """Composite candidate skeletons grouped by family."""
    families = tuple(family for family, _weight in config.composite_family_weights)
    candidates = generate_composite_candidates(
        schema,
        database,
        families=families,
        max_path_depth=config.composite_max_path_depth,
        candidate_limit=config.composite_candidate_limit,
    )
    pools: dict[CompositeFamily, list[object]] = {}
    for candidate in candidates:
        pools.setdefault(candidate.family, []).append(candidate)
    return pools


def _generate_composite_task(
    *,
    task_id: str,
    sample_id: str,
    schema: PhysicalSchema,
    database: DatabaseInstance,
    seed: int,
    support_fraction: float,
    min_support_rows: int,
    min_query_rows: int,
    min_class_count_per_split: int,
    composite_pools: dict[CompositeFamily, list[object]],
    family_weights: dict[CompositeFamily, float],
    rng: np.random.Generator,
    max_predicates: int,
    positive_rate_min: float,
    positive_rate_max: float,
) -> PlannedTask | None:
    """Build one composite task, trying families in weighted order.

    When a sampled family's candidates are infeasible (e.g. window out of
    range, degenerate labels), later families are tried within the same
    attempt so the planner does not waste attempts on an exhausted family.
    """
    families = [
        family for family, candidates in composite_pools.items() if candidates
    ]
    if not families:
        return None
    weights = [family_weights[family] for family in families]
    order = rng.choice(
        len(families), size=len(families), replace=False, p=_normalize(weights)
    )
    builds = 0
    for family_index in order:
        family = families[int(family_index)]
        candidates = list(composite_pools[family])
        rng.shuffle(candidates)
        for candidate in candidates:
            if builds >= 64:
                return None
            builds += 1
            task = build_composite_relational_classification_task(
                task_id=task_id,
                sample_id=sample_id,
                schema=schema,
                database=database,
                candidate=candidate,
                seed=seed,
                support_fraction=support_fraction,
                min_support_rows=min_support_rows,
                min_query_rows=min_query_rows,
                min_class_count_per_split=min_class_count_per_split,
                positive_rate_min=positive_rate_min,
                positive_rate_max=positive_rate_max,
                max_predicates=max_predicates,
            )
            if task is not None:
                return task
    return None


def _bind_instance_calendar(
    task: PlannedTask,
    instance_plan: InstancePlan | None,
) -> PlannedTask | None:
    """Attach an instance calendar without allowing invalid task plans through.

    Task mechanisms are intentionally generated from observed timestamps and
    therefore do not need to know the instance plan while calculating labels.
    Before adding the hard database bounds, however, reject a candidate whose
    cutoff, horizon, observation rule, or aggregate window is inconsistent with
    those bounds.  Rejecting lets the planner try another candidate; clipping
    would silently change the task labels and data semantics.
    """
    if instance_plan is None or instance_plan.calendar_start_seconds is None:
        return task
    start = instance_plan.calendar_start_seconds
    end = instance_plan.calendar_end_seconds
    assert end is not None
    plan = task.plan
    invalid_fields: list[str] = []
    for name in ("cutoff_time", "horizon_end_time"):
        value = getattr(plan, name)
        if value is not None and not start <= value <= end:
            invalid_fields.append(name)
    for index, rule in enumerate(plan.observation_rules):
        if not start <= rule.max_timestamp <= end:
            invalid_fields.append(f"observation_rules[{index}].max_timestamp")
    window = plan.parameter_map.get("window")
    if window is not None and not 0 <= window <= end - start:
        invalid_fields.append("window")
    if plan.composite_spec is not None:
        aggregates = list(plan.composite_spec.label_aggregates)
        if plan.composite_spec.eligibility_aggregate is not None:
            aggregates.append(plan.composite_spec.eligibility_aggregate)
        for index, aggregate in enumerate(aggregates):
            if plan.cutoff_time is None:
                continue
            lower = plan.cutoff_time + aggregate.window_start
            upper = plan.cutoff_time + aggregate.window_end
            if not (start <= lower <= end and start <= upper <= end):
                invalid_fields.append(f"composite window[{index}]")
    if invalid_fields:
        _LOGGER.debug(
            "skipping task candidate %s for calendar [%d, %d]: %s",
            plan.task_id,
            start,
            end,
            ", ".join(invalid_fields),
        )
        return None
    return replace(
        task,
        plan=replace(
            plan,
            db_start_seconds=start,
            db_end_seconds=end,
        ),
    )


def _fraction_range(name: str, low: float, high: float) -> None:
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (low, high)):
        raise TypeError(f"{name} bounds must be numeric")
    if not 0 < low <= high < 1:
        raise ValueError(f"{name} bounds must satisfy 0 < min <= max < 1")


def _weight_range(name: str, low: float, high: float) -> None:
    """Range check for propensity weights (no ``< 1`` upper bound)."""
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (low, high)):
        raise TypeError(f"{name} bounds must be numeric")
    if not 0 < low <= high:
        raise ValueError(f"{name} bounds must satisfy 0 < min <= max")


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


__all__ = ["TaskPlannerConfig", "TaskPlanner"]
