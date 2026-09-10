"""Materialization of P1 state-conditioned Entity--Event processes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import heapq

import numpy as np

from rdb_prior.compilation.model import ColumnKind, PhysicalSchema
from rdb_prior.generation.encoding import encode_feature_score
from rdb_prior.generation.model import DatabaseInstance, TableData
from rdb_prior.generation.state import SharedStateRegistry
from rdb_prior.generation.trees.executor import evaluate_forest
from rdb_prior.generation.trees.model import ForestPlan
from rdb_prior.process.executor import RuleEvaluationContext, RuleEffect, RuleProcessExecutor
from rdb_prior.process.model import RulePlan
from rdb_prior.generation.state_trajectory import (
    StateTrajectory,
    TemporalStateRegistry,
)
from rdb_prior.instance.plan import (
    ColumnMechanismPlan,
    InstancePlan,
    PopulationPlan,
)
from rdb_prior.priors.model import StateVisibility, TransitionClock


def resolve_temporal_population_plan(
    schema: PhysicalSchema,
    plan: InstancePlan,
    entity_database: DatabaseInstance,
) -> InstancePlan:
    """Resolve the P1 draft after entity state and attributes exist.

    The draft plan provides a safe provisional Event size so the old generator
    remains usable.  This final pass is the causal P1 point: ``z`` from the
    shared latent and anonymous entity attributes ``x`` jointly determine the
    Negative-Binomial count for every entity.  The realized count vector is
    provenance, never exported as a model feature.
    """
    if plan.temporal_state_plans:
        return _resolve_stateful_temporal_population_plan(schema, plan, entity_database)
    if not plan.population_mechanisms or not (
        any(
            dict(item.parameters).get("population_source") == "temporal_event"
            for item in plan.population_mechanisms
        )
        or any(item.temporal_state_ids for item in plan.temporal_processes)
        or plan.prior_family in {"temporal_event", "rule_process"}
    ):
        return plan
    shared_states = SharedStateRegistry.from_plan(plan)
    table_plans = {item.table_id: item for item in plan.tables}
    resolved: list = []
    changed_table_ids: set[str] = set()
    for mechanism in plan.population_mechanisms:
        if mechanism.family != "negative_binomial" or mechanism.parent_table_id is None:
            resolved.append(mechanism)
            continue
        parameters = dict(mechanism.parameters)
        entity_id = mechanism.parent_table_id
        event_id = mechanism.table_id
        entity = entity_database.table(entity_id)
        if not mechanism.state_ids:
            raise ValueError("temporal population mechanism lacks shared state")
        state = shared_states.state(mechanism.state_ids[0])
        row_count = entity.row_count
        rng = np.random.Generator(
            np.random.PCG64DXSM(plan.table(event_id).temporal_seed ^ 0x5EED5EED)
        )
        z_weights = np.asarray(
            parameters.get("count_state_weights", rng.normal(0.0, 0.45, size=state.shape[1])),
            dtype=np.float64,
        )
        z_score = _standardize(state @ z_weights)
        x_score = _entity_attribute_score(schema, entity_id, entity.columns, rng)
        family = str(parameters.get("intensity_family", "linear"))
        forest_payload = parameters.get("event_intensity_forest")
        if family == "tree" or isinstance(forest_payload, Mapping):
            if not isinstance(forest_payload, Mapping):
                raise ValueError("tree event intensity lacks a serialized forest")
            score = _tree_intensity_score(
                ForestPlan.from_dict(forest_payload),
                schema,
                entity_id,
                entity.columns,
                state,
            )
        elif family == "cam":
            score = (
                0.55 * z_score
                + 0.45 * x_score
                + 0.25 * np.sin(z_score * x_score)
            )
        else:
            score = 0.60 * z_score + 0.40 * x_score
        baseline = float(parameters["baseline_intensity"])
        intensity = baseline * np.exp(0.55 * _standardize(score))
        rule_effect = _rule_effect_for_entities(
            parameters.get("rule_plan"),
            entity_database,
            mechanism.state_ids[0],
            state,
            row_count,
        )
        if rule_effect is not None:
            intensity = intensity * rule_effect.intensity_multiplier + rule_effect.intensity_addition
        intensity = np.clip(
            intensity,
            0.03,
            max(12.0, baseline * 6.0 * (float(np.max(rule_effect.intensity_multiplier)) if rule_effect is not None else 1.0)),
        )
        dispersion = float(parameters["dispersion"])
        probability = dispersion / (dispersion + intensity)
        counts = rng.negative_binomial(dispersion, probability).astype(np.int64)
        total = int(np.sum(counts))
        draft_table = table_plans[event_id]
        hard_limit = max(draft_table.population.row_count * 4, 128)
        if total < 1:
            raise ValueError("temporal prior produced no events; retry materialization")
        if total > hard_limit:
            raise ValueError("temporal prior event population exceeded its hard limit")
        table_plans[event_id] = replace(
            draft_table,
            population=PopulationPlan(
                strategy="state_attribute_negative_binomial",
                row_count=total,
                parameters=draft_table.population.parameters + (("state_attribute_conditioned", 1.0),),
            ),
        )
        changed_table_ids.add(event_id)
        resolved.append(
            replace(
                mechanism,
                parameters=mechanism.parameters
                + (
                    ("entity_event_counts", [int(item) for item in counts]),
                    ("realized_event_count", total),
                    ("entity_attribute_signal_std", float(np.std(x_score))),
                ),
            )
        )
    resolved_plan = replace(
        plan,
        tables=tuple(table_plans[table_id] for table_id in plan.generation_order),
        population_mechanisms=tuple(resolved),
    )
    return _replan_downstream_populations(schema, resolved_plan, changed_table_ids)


@dataclass(frozen=True, slots=True)
class TemporalEventMaterialization:
    database: DatabaseInstance
    temporal_states: TemporalStateRegistry | None = None


def apply_temporal_event_processes(
    schema: PhysicalSchema,
    plan: InstancePlan,
    database: DatabaseInstance,
    *,
    include_private_state: bool = False,
) -> DatabaseInstance | TemporalEventMaterialization:
    """Apply the compatible P1 process or the opt-in temporal-state process."""
    if plan.temporal_state_plans:
        result = _apply_stateful_temporal_event_processes(schema, plan, database)
    else:
        result = TemporalEventMaterialization(
            database=_apply_stateless_temporal_event_processes(
                schema,
                plan,
                database,
            ),
        )
    return result if include_private_state else result.database



def _apply_stateless_temporal_event_processes(
    schema: PhysicalSchema,
    plan: InstancePlan,
    database: DatabaseInstance,
) -> DatabaseInstance:
    """Replace P1 Event FK/time/feature values with one shared process."""
    if not plan.population_mechanisms or not (
        any(
            dict(item.parameters).get("population_source") == "temporal_event"
            for item in plan.population_mechanisms
        )
        or any(item.temporal_state_ids for item in plan.temporal_processes)
        or plan.prior_family in {"temporal_event", "rule_process"}
    ):
        return database
    shared_states = SharedStateRegistry.from_plan(plan)
    tables = {item.table_id: item for item in database.tables}
    process_by_table = {item.table_id: item for item in plan.temporal_processes}
    columns_by_id = {item.column_id: item for item in plan.column_mechanisms}
    foreign_keys = {item.foreign_key_id: item for item in schema.foreign_keys}
    for mechanism in plan.population_mechanisms:
        if mechanism.family != "negative_binomial" or mechanism.parent_table_id is None:
            continue
        parameters = dict(mechanism.parameters)
        foreign_key = foreign_keys[str(parameters["foreign_key_id"])]
        event_table = schema.table(mechanism.table_id)
        entity_id = mechanism.parent_table_id
        if not mechanism.state_ids:
            raise ValueError("temporal population mechanism lacks shared state")
        entity_state = shared_states.state(mechanism.state_ids[0])
        entity_table = tables[entity_id]
        event = tables[mechanism.table_id]
        row_count = event.row_count
        table_seed = sum((index + 1) * ord(char) for index, char in enumerate(mechanism.table_id))
        rng = np.random.Generator(np.random.PCG64DXSM(plan.global_seed ^ table_seed))
        count_values = parameters.get("entity_event_counts")
        if isinstance(count_values, list) and len(count_values) == len(entity_state):
            assignments = np.repeat(
                np.arange(len(entity_state), dtype=np.int64),
                np.asarray(count_values, dtype=np.int64),
            )
            if len(assignments) != row_count:
                raise ValueError("resolved temporal population no longer matches Event table rows")
        else:
            coefficients = rng.normal(0.0, 0.5, size=entity_state.shape[1])
            score = entity_state @ coefficients
            probability = np.exp(0.55 * (score - score.max()))
            probability /= probability.sum()
            assignments = rng.choice(len(entity_state), size=row_count, p=probability).astype(np.int64)
        values = dict(event.columns)
        values[foreign_key.child_column_id] = assignments
        process = process_by_table.get(mechanism.table_id)
        if process is not None:
            values.update(
                _event_values(
                    event_table,
                    plan,
                    assignments,
                    entity_state,
                    entity_table,
                    event,
                    process,
                    columns_by_id,
                    rng,
                )
            )
        tables[mechanism.table_id] = TableData(table_id=mechanism.table_id, columns=values)
    return DatabaseInstance(instance_id=database.instance_id, schema_id=database.schema_id, plan_id=database.plan_id, tables=tuple(tables[table.table_id] for table in schema.tables))


def _event_values(
    event_table,
    plan: InstancePlan,
    assignments: np.ndarray,
    entity_state: np.ndarray,
    entity_table: TableData,
    event_data: TableData | None,
    process,
    column_mechanisms: dict[str, ColumnMechanismPlan],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    start = plan.calendar_start_seconds
    end = plan.calendar_end_seconds
    if start is None or end is None:
        raise ValueError("temporal prior requires a database calendar")
    rows = len(assignments)
    parameters = dict(process.parameters)
    raw = np.empty(rows, dtype=np.float64)
    for entity_index in range(len(entity_state)):
        indices = np.flatnonzero(assignments == entity_index)
        if len(indices):
            raw[indices] = _temporal_ticks(
                rng, process.family, len(indices), entity_state[entity_index], parameters
            )
    times = (start + raw * (end - start)).astype(np.int64)
    output = {
        column.column_id: times
        for column in event_table.columns
        if column.kind is ColumnKind.TIME
    }
    history = _event_history(assignments, times)
    time_score = (times - start) / max(end - start, 1)
    table_plan = plan.table(event_table.table_id)
    state = entity_state[assignments]
    rule_effect = None
    rule_payload = parameters.get("rule_plan")
    if isinstance(rule_payload, Mapping):
        state_id = next(iter(process.state_ids), None)
        if state_id is not None:
            time_ids = {
                event_table.table_id: next(
                    (column.column_id for column in event_table.columns if column.kind is ColumnKind.TIME),
                    "",
                )
            }
            rule_context = RuleEvaluationContext(
                tables={
                    entity_table.table_id: entity_table,
                    **({event_data.table_id: event_data} if event_data is not None else {}),
                },
                states={state_id: entity_state},
                cutoff_time=end,
                time_column_ids=time_ids,
            )
            rule_effect = RuleProcessExecutor().effects(
                RulePlan.from_dict(rule_payload),
                rule_context,
                row_count=len(entity_state),
            )
    for column in event_table.columns:
        if column.kind is not ColumnKind.FEATURE:
            continue
        mechanism = column_mechanisms.get(column.column_id)
        score = _execute_column_mechanism(
            mechanism,
            state,
            entity_table.columns,
            assignments,
            time_score,
            history,
            rng,
        )
        if rule_effect is not None:
            score = score + rule_effect.attribute_offset[assignments]
        output[column.column_id] = encode_feature_score(
            score,
            column,
            event_table.role,
            rng,
            cardinality=int(table_plan.parameter_map["categorical_cardinality"]),
            db_start=start,
            db_end=end,
            categorical_dirichlet_alpha=float(
                table_plan.parameter_map.get("categorical_dirichlet_alpha", 1.0)
            ),
            categorical_signal_strength=float(
                table_plan.parameter_map.get("categorical_signal_strength", 1.0)
            ),
            missing_rate=float(table_plan.parameter_map["missing_rate"]),
            long_tail_enabled=bool(
                table_plan.parameter_map.get("long_tail_enabled", 0.0)
            ),
            long_tail_alpha=float(
                table_plan.parameter_map.get("long_tail_alpha", 1.0)
            ),
        )
    return output


def _tree_intensity_score(
    forest: ForestPlan,
    schema: PhysicalSchema,
    entity_id: str,
    columns: Mapping[str, np.ndarray],
    state: np.ndarray,
) -> np.ndarray:
    """Evaluate a sampled count-intensity forest without inspecting labels."""
    feature = next(
        (
            column
            for column in schema.table(entity_id).columns
            if column.kind is ColumnKind.FEATURE
        ),
        None,
    )
    if feature is None:
        entity_feature = np.zeros(state.shape[0], dtype=np.float64)
    else:
        entity_feature = _numeric_column(columns[feature.column_id])
    return evaluate_forest(
        forest,
        {
            "state_0": state[:, 0],
            "entity_feature_0": entity_feature,
        },
        row_count=state.shape[0],
    )


def _rule_effect_for_entities(
    payload: object,
    database: DatabaseInstance,
    state_id: str,
    state: np.ndarray,
    row_count: int,
) -> RuleEffect | None:
    if not isinstance(payload, Mapping):
        return None
    rule = RulePlan.from_dict(payload)
    context = RuleEvaluationContext(
        tables={table.table_id: table for table in database.tables},
        states={state_id: state},
    )
    return RuleProcessExecutor().effects(rule, context, row_count=row_count)


def _numeric_column(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind in {"b", "i", "u", "f"}:
        numeric = raw.astype(np.float64)
        finite = np.isfinite(numeric)
        fill = float(np.mean(numeric[finite])) if np.any(finite) else 0.0
        numeric = np.where(finite, numeric, fill)
    else:
        _categories, numeric = np.unique(raw.astype(str), return_inverse=True)
        numeric = numeric.astype(np.float64)
    return _standardize(numeric)


def _entity_attribute_score(schema, table_id, columns, rng):
    table = schema.table(table_id)
    features: list[np.ndarray] = []
    for column in table.columns:
        if column.kind is not ColumnKind.FEATURE:
            continue
        values = np.asarray(columns[column.column_id])
        if values.dtype.kind in {"b", "i", "u", "f"}:
            encoded = values.astype(np.float64)
            if not np.isfinite(encoded).all():
                finite = np.isfinite(encoded)
                fill = float(np.mean(encoded[finite])) if np.any(finite) else 0.0
                encoded = np.where(finite, encoded, fill)
        else:
            _unique, encoded = np.unique(values.astype(str), return_inverse=True)
            encoded = encoded.astype(np.float64)
        features.append(_standardize(encoded))
    if not features:
        return np.zeros(next(iter(columns.values())).shape[0], dtype=np.float64)
    matrix = np.column_stack(features)
    return _standardize(matrix @ rng.normal(0.0, 0.5, size=matrix.shape[1]))


def _standardize(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(float(values.std()), 1e-6)




def _event_history(assignments: np.ndarray, times: np.ndarray) -> np.ndarray:
    history = np.zeros(len(assignments), dtype=np.float64)
    for entity_index in np.unique(assignments):
        indices = np.flatnonzero(assignments == entity_index)
        ranked = indices[np.argsort(times[indices], kind="stable")]
        history[ranked] = np.arange(len(ranked), dtype=np.float64)
    return history / max(float(history.max(initial=0.0)), 1.0)


def _temporal_ticks(rng, family: str, count: int, state: np.ndarray, parameters: dict[str, object]) -> np.ndarray:
    """Sample one ordered state-conditioned time process in [0, 1]."""
    active = 0.15 + 0.85 * _sigmoid(_state_projection(state, parameters, "state_active_interval_weights"))
    if family == "seasonal":
        strength = float(parameters.get("seasonal_strength", 0.5))
        phase = _state_projection(state, parameters, "state_seasonal_phase_weights")
        candidates = rng.random(max(count * 4, 12))
        acceptance = (1.0 + strength * np.sin(2.0 * np.pi * (candidates + phase))) / (1.0 + strength)
        accepted = candidates[rng.random(len(candidates)) <= acceptance]
        if len(accepted) < count:
            accepted = np.concatenate((accepted, rng.random(count - len(accepted))))
        return np.sort(accepted[:count] * active)
    if family == "churn":
        exponent = max(0.45, float(parameters.get("churn_exponent", 2.0)) + _softplus(_state_projection(state, parameters, "state_churn_weights")))
        return np.sort(rng.random(count) ** exponent * active)
    if family == "renewal":
        scale = active * np.exp(np.clip(_state_projection(state, parameters, "state_renewal_scale_weights"), -2.0, 2.0)) / max(count, 1)
        return np.clip(np.cumsum(rng.exponential(scale, size=count)), 0.0, active)
    if family != "stationary":
        raise ValueError(f"unsupported temporal process family: {family}")
    return np.sort(rng.random(count) * active)


def _state_projection(state: np.ndarray, parameters: dict[str, object], key: str) -> float:
    weights = np.asarray(parameters.get(key, ()), dtype=np.float64)
    if weights.shape != state.shape:
        return 0.0
    return float(np.dot(state, weights) / max(np.sqrt(len(state)), 1.0))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -20.0, 20.0))))


def _softplus(value: float) -> float:
    return float(np.logaddexp(0.0, np.clip(value, -30.0, 30.0)))


def _execute_column_mechanism(mechanism: ColumnMechanismPlan | None, state: np.ndarray, entity_columns: dict[str, np.ndarray], assignments: np.ndarray, time_score: np.ndarray, history: np.ndarray, fallback_rng: np.random.Generator) -> np.ndarray:
    """Execute the sampled Linear or CAM event-attribute mechanism."""
    if mechanism is None:
        return state.mean(axis=1) + 0.5 * time_score + 0.5 * history + fallback_rng.normal(0.0, 0.25, size=len(time_score))
    parameters = dict(mechanism.parameters)
    parent = _column_parent_matrix(
        mechanism.parent_column_ids,
        entity_columns,
        assignments,
    )
    if mechanism.family == "tree":
        forest_payload = parameters.get("forest")
        if not isinstance(forest_payload, Mapping):
            raise ValueError("tree temporal column lacks a serialized forest")
        return evaluate_forest(
            ForestPlan.from_dict(forest_payload),
            {
                "state_0": state[:, 0],
                "parent_feature_0": (
                    parent[:, 0]
                    if parent.shape[1]
                    else np.zeros(len(state), dtype=np.float64)
                ),
                "time": time_score,
                "history": history,
            },
            row_count=len(state),
        )
    rng = np.random.Generator(
        np.random.PCG64DXSM(int(parameters.get("mechanism_seed", 0)))
    )
    if mechanism.family == "linear":
        score = state @ rng.normal(0.0, 0.55, size=state.shape[1])
        if parent.shape[1]:
            score += parent @ rng.normal(0.0, 0.45, size=parent.shape[1])
    elif mechanism.family == "cam":
        state_frequency = np.exp(rng.uniform(-0.45, 0.45, size=state.shape[1]))
        score = np.sum(rng.normal(0.0, 0.7, size=state.shape[1]) * np.sin(state_frequency * state), axis=1)
        if parent.shape[1]:
            parent_frequency = np.exp(rng.uniform(-0.45, 0.45, size=parent.shape[1]))
            score += np.sum(rng.normal(0.0, 0.55, size=parent.shape[1]) * np.tanh(parent_frequency * parent), axis=1)
    else:
        raise ValueError(f"unsupported temporal column mechanism: {mechanism.family}")
    score += float(parameters.get("time_weight", 0.5)) * time_score
    score += float(parameters.get("history_weight", 0.5)) * history
    score += rng.normal(0.0, float(parameters.get("noise_scale", 0.25)), size=len(score))
    return np.asarray(score, dtype=np.float64)


def _column_parent_matrix(column_ids: tuple[str, ...], columns: dict[str, np.ndarray], assignments: np.ndarray) -> np.ndarray:
    values: list[np.ndarray] = []
    for column_id in column_ids:
        if column_id not in columns:
            continue
        raw = np.asarray(columns[column_id])
        if raw.dtype.kind in {"b", "i", "u", "f"}:
            numeric = raw.astype(np.float64)
        else:
            _categories, numeric = np.unique(raw.astype(str), return_inverse=True)
            numeric = numeric.astype(np.float64)
        values.append(_standardize(np.nan_to_num(numeric, nan=0.0))[assignments])
    if not values:
        return np.empty((len(assignments), 0), dtype=np.float64)
    return np.column_stack(values)


@dataclass(frozen=True, slots=True)
class _StatefulEventSchedule:
    assignments: np.ndarray
    times: np.ndarray
    ordinals: np.ndarray
    state_indices: np.ndarray
    history: np.ndarray
    trajectories: tuple[StateTrajectory, ...]


def _stateful_event_schedules(
    schema: PhysicalSchema,
    plan: InstancePlan,
    database: DatabaseInstance,
    *,
    mechanisms: tuple,
) -> dict[str, _StatefulEventSchedule]:
    """Generate all Event channels from one Entity-owned state trajectory.

    Channels are grouped by ``temporal_state_id`` and share a per-Entity
    priority queue.  The queue key is ``(time, kind, channel, sequence)``;
    this gives deterministic same-timestamp ordering while allowing any
    channel's emission to advance the common state machine.
    """
    process_by_table = {item.table_id: item for item in plan.temporal_processes}
    grouped: dict[str, list[tuple[object, object]]] = {}
    for mechanism in mechanisms:
        process = process_by_table.get(mechanism.table_id)
        if (
            mechanism.family != "negative_binomial"
            or mechanism.parent_table_id is None
            or process is None
            or not process.temporal_state_ids
        ):
            continue
        grouped.setdefault(process.temporal_state_ids[0], []).append(
            (mechanism, process)
        )

    schedules: dict[str, _StatefulEventSchedule] = {}
    for temporal_state_id, channels in sorted(grouped.items()):
        temporal_plan = next(
            item
            for item in plan.temporal_state_plans
            if item.state_id == temporal_state_id
        )
        owner_id = temporal_plan.owner_table_id
        if any(item.parent_table_id != owner_id for item, _process in channels):
            raise ValueError("all Event channels for a temporal state must share its Entity owner")
        schedules.update(
            _stateful_event_schedule_group(
                schema,
                plan,
                database,
                temporal_plan,
                tuple(sorted(channels, key=lambda item: item[0].table_id)),
            )
        )
    return schedules


def _stateful_event_schedule_group(
    schema: PhysicalSchema,
    plan: InstancePlan,
    database: DatabaseInstance,
    temporal_plan,
    channels: tuple[tuple[object, object], ...],
) -> dict[str, _StatefulEventSchedule]:
    if plan.calendar_start_seconds is None or plan.calendar_end_seconds is None:
        raise ValueError("temporal state process requires a calendar")
    schedules: dict[str, _StatefulEventSchedule] = {}
    start = plan.calendar_start_seconds
    end = plan.calendar_end_seconds
    span = float(end - start)
    static = SharedStateRegistry.from_plan(plan).state(temporal_plan.shared_state_id)
    entity = database.table(temporal_plan.owner_table_id)
    if len(static) != entity.row_count:
        raise ValueError("static state must align with temporal-state owner")
    state_to_index = {
        value: index for index, value in enumerate(temporal_plan.state_space.values)
    }
    channel_data: dict[str, dict[str, object]] = {}
    for mechanism, process in channels:
        parameters = dict(mechanism.parameters)
        baseline = float(parameters["baseline_intensity"])
        base_rate = max(baseline / span, 1.0 / max(span * 100.0, 1.0))
        score_rng = np.random.Generator(
            np.random.PCG64DXSM(
                temporal_plan.seed ^ _stable_channel_seed(mechanism.table_id)
            )
        )
        attribute_score = _entity_attribute_score(
            schema,
            temporal_plan.owner_table_id,
            entity.columns,
            score_rng,
        )
        forest_payload = parameters.get("event_intensity_forest")
        if isinstance(forest_payload, Mapping):
            intensity_score = _standardize(
                _tree_intensity_score(
                    ForestPlan.from_dict(forest_payload),
                    schema,
                    temporal_plan.owner_table_id,
                    entity.columns,
                    static,
                )
            )
        else:
            intensity_score = np.zeros(entity.row_count, dtype=np.float64)
        rule_effect = _rule_effect_for_entities(
            parameters.get("rule_plan"),
            database,
            temporal_plan.shared_state_id,
            static,
            entity.row_count,
        )
        channel_data[mechanism.table_id] = {
            "mechanism": mechanism,
            "process": process,
            "parameters": parameters,
            "base_rate": base_rate,
            "attribute_score": attribute_score,
            "intensity_score": intensity_score,
            "rule_effect": rule_effect,
            "assignments": [],
            "times": [],
            "ordinals": [],
            "state_indices": [],
            "history": [],
        }

    trajectories: list[StateTrajectory] = []
    for entity_index, z in enumerate(static):
        entity_seed = temporal_plan.seed ^ (entity_index + 1)
        initial_rng = np.random.Generator(np.random.PCG64DXSM(entity_seed))
        initial = _initial_state(temporal_plan, z, initial_rng)
        trajectory = StateTrajectory(
            calendar_start=start,
            calendar_end=end,
            initial_state=initial,
            state_space=temporal_plan.state_space,
        )
        trajectories.append(trajectory)
        queue: list[tuple[float, int, str, int, str, int]] = []
        channel_rngs: dict[str, np.random.Generator] = {}
        channel_history: dict[str, int] = {}
        channel_sequence: dict[str, int] = {}
        for channel_index, (mechanism, process) in enumerate(channels):
            channel_seed = _stable_channel_seed(mechanism.table_id)
            channel_rng = (
                initial_rng
                if channel_index == 0
                else np.random.Generator(np.random.PCG64DXSM(entity_seed ^ channel_seed))
            )
            channel_rngs[mechanism.table_id] = channel_rng
            channel_history[mechanism.table_id] = 0
            channel_sequence[mechanism.table_id] = 0
            current_state = initial
            rate = _channel_event_rate(
                temporal_plan,
                current_state,
                z,
                entity_index,
                channel_data[mechanism.table_id],
            )
            wait = _stateful_wait(
                channel_rng,
                process,
                z,
                rate,
                float(start),
                start,
                end,
            )
            heapq.heappush(
                queue,
                (start + wait, 0, mechanism.table_id, channel_sequence[mechanism.table_id], "event", channel_index),
            )
            channel_sequence[mechanism.table_id] += 1

        transition_rng = channel_rngs[channels[0][0].table_id]
        transition_data = channel_data[channels[0][0].table_id]
        transition_version = 0
        transition_deadline: float | None = None
        if temporal_plan.transition.clock is not TransitionClock.EVENT_DRIVEN:
            transition_rate = _transition_rate(
                temporal_plan,
                initial,
                z,
                entity_index,
                transition_data,
            )
            transition_deadline = float(
                start + transition_rng.exponential(1.0 / max(transition_rate, 1e-12))
            )
            heapq.heappush(
                queue,
                (transition_deadline, 1, "", transition_version, "transition", -1),
            )

        next_ordinal: dict[int, int] = {}
        last_transition: tuple[int, int] | None = None
        steps = 0
        while queue and not trajectory.is_terminal:
            event_time, kind, channel_id, sequence, event_kind, channel_index = heapq.heappop(queue)
            if event_kind == "transition" and sequence != transition_version:
                continue
            if event_time > end:
                break
            timestamp = _calendar_time(event_time, start, end)
            if event_kind == "transition":
                ordinal = _duration_transition_ordinal(
                    timestamp,
                    next_ordinal,
                    last_transition,
                )
                state = trajectory.state_after(timestamp, ordinal)
                target = _next_state(temporal_plan, state, z, transition_rng)
                override = _first_state_override(channel_data, entity_index, state_to_index)
                if override is not None:
                    target = override
                trajectory.transition(timestamp=timestamp, ordinal=ordinal, state=target)
                last_transition = (timestamp, ordinal)
                transition_version += 1
                if not trajectory.is_terminal:
                    state = trajectory.state_after(timestamp, ordinal)
                    transition_rate = _transition_rate(
                        temporal_plan,
                        state,
                        z,
                        entity_index,
                        transition_data,
                    )
                    transition_deadline = float(
                        event_time
                        + transition_rng.exponential(1.0 / max(transition_rate, 1e-12))
                    )
                    heapq.heappush(
                        queue,
                        (transition_deadline, 1, "", transition_version, "transition", -1),
                    )
                steps += 1
                continue

            data = channel_data[channel_id]
            ordinal = next_ordinal.get(timestamp, 0)
            next_ordinal[timestamp] = ordinal + 1
            state = trajectory.state_before(timestamp, ordinal)
            payload = data["assignments"]
            assert isinstance(payload, list)
            payload.append(entity_index)
            for key, value in (
                ("times", timestamp),
                ("ordinals", ordinal),
                ("state_indices", state_to_index[state]),
                ("history", float(channel_history[channel_id])),
            ):
                target = data[key]
                assert isinstance(target, list)
                target.append(value)
            channel_history[channel_id] += 1
            current_state = state
            if temporal_plan.transition.clock in {
                TransitionClock.EVENT_DRIVEN,
                TransitionClock.HYBRID,
            }:
                target_state = _next_state(
                    temporal_plan,
                    current_state,
                    z,
                    channel_rngs[channel_id],
                )
                effect = data["rule_effect"]
                if effect is not None:
                    override = effect.state_overrides[entity_index]
                    if override in state_to_index:
                        target_state = str(override)
                trajectory.transition(
                    timestamp=timestamp,
                    ordinal=ordinal,
                    state=target_state,
                )
                last_transition = (timestamp, ordinal)
                if temporal_plan.transition.clock is TransitionClock.HYBRID:
                    transition_version += 1
                    state = target_state
                    transition_rate = _transition_rate(
                        temporal_plan,
                        state,
                        z,
                        entity_index,
                        transition_data,
                    )
                    transition_deadline = float(
                        event_time
                        + transition_rng.exponential(1.0 / max(transition_rate, 1e-12))
                    )
                    heapq.heappush(
                        queue,
                        (transition_deadline, 1, "", transition_version, "transition", -1),
                    )
            steps += 1
            if steps >= 8192:
                raise ValueError("temporal state process exceeded transition limit")
            if trajectory.is_terminal:
                break
            state = trajectory.state_after(timestamp, ordinal)
            rate = _channel_event_rate(
                temporal_plan,
                state,
                z,
                entity_index,
                data,
            )
            process = data["process"]
            assert process is not None
            rng = channel_rngs[channel_id]
            wait = _stateful_wait(
                rng,
                process,
                z,
                rate,
                event_time,
                start,
                end,
            )
            channel_sequence[channel_id] += 1
            heapq.heappush(
                queue,
                (event_time + wait, 0, channel_id, channel_sequence[channel_id], "event", channel_index),
            )

    for table_id, data in channel_data.items():
        schedules[table_id] = _StatefulEventSchedule(
            assignments=np.asarray(data["assignments"], dtype=np.int64),
            times=np.asarray(data["times"], dtype=np.int64),
            ordinals=np.asarray(data["ordinals"], dtype=np.int64),
            state_indices=np.asarray(data["state_indices"], dtype=np.int64),
            history=np.asarray(data["history"], dtype=np.float64),
            trajectories=tuple(trajectories),
        )
    return schedules


def _stateful_event_schedule(
    schema: PhysicalSchema,
    plan: InstancePlan,
    database: DatabaseInstance,
    mechanism,
    process,
) -> _StatefulEventSchedule:
    """Compatibility view over the unified multi-channel scheduler."""
    schedule = _stateful_event_schedules(
        schema,
        plan,
        database,
        mechanisms=tuple(plan.population_mechanisms),
    ).get(mechanism.table_id)
    if schedule is None:
        raise ValueError(f"no shared temporal schedule for {mechanism.table_id}")
    return schedule


def _stable_channel_seed(table_id: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(table_id))


def _channel_event_rate(
    temporal_plan,
    state: str,
    z: np.ndarray,
    entity_index: int,
    data: dict[str, object],
) -> float:
    base_rate = float(data["base_rate"])
    attribute_score = np.asarray(data["attribute_score"])[entity_index]
    intensity_score = np.asarray(data["intensity_score"])[entity_index]
    effect = data["rule_effect"]
    multiplier = float(effect.intensity_multiplier[entity_index]) if effect is not None else 1.0
    addition = float(effect.intensity_addition[entity_index]) if effect is not None else 0.0
    state_index = temporal_plan.state_space.values.index(state)
    return _state_rate(
        temporal_plan,
        state_index,
        z,
        float(attribute_score),
        base_rate * float(np.exp(0.55 * np.clip(intensity_score, -5.0, 5.0))) * multiplier + addition,
    )


def _transition_rate(
    temporal_plan,
    state: str,
    z: np.ndarray,
    entity_index: int,
    data: dict[str, object],
) -> float:
    return _channel_event_rate(
        temporal_plan,
        state,
        z,
        entity_index,
        data,
    ) * float(
        dict(temporal_plan.duration.parameters).get(
            "transition_rate_multiplier", 1.0
        )
    )


def _first_state_override(
    channel_data: dict[str, dict[str, object]],
    entity_index: int,
    state_to_index: dict[str, int],
) -> str | None:
    for table_id in sorted(channel_data):
        effect = channel_data[table_id]["rule_effect"]
        if effect is not None:
            override = effect.state_overrides[entity_index]
            if override in state_to_index:
                return str(override)
    return None


def _resolve_stateful_temporal_population_plan(
    schema: PhysicalSchema,
    plan: InstancePlan,
    entity_database: DatabaseInstance,
) -> InstancePlan:
    table_plans = {item.table_id: item for item in plan.tables}
    process_by_table = {item.table_id: item for item in plan.temporal_processes}
    resolved = []
    schedules = _stateful_event_schedules(
        schema,
        plan,
        entity_database,
        mechanisms=tuple(plan.population_mechanisms),
    )
    for mechanism in plan.population_mechanisms:
        process = process_by_table.get(mechanism.table_id)
        if (
            mechanism.family != "negative_binomial"
            or mechanism.parent_table_id is None
            or process is None
            or not process.temporal_state_ids
        ):
            resolved.append(mechanism)
            continue
        schedule = schedules[mechanism.table_id]
        total = len(schedule.times)
        if total < 1:
            raise ValueError("temporal state process produced no events")
        table = table_plans[mechanism.table_id]
        hard_limit = max(table.population.row_count * 4, 128)
        if total > hard_limit:
            raise ValueError("temporal state process exceeded its event limit")
        counts = np.bincount(
            schedule.assignments,
            minlength=entity_database.table(mechanism.parent_table_id).row_count,
        )
        table_plans[mechanism.table_id] = replace(
            table,
            population=PopulationPlan(
                strategy="temporal_state_event_process",
                row_count=total,
                parameters=table.population.parameters
                + (("temporal_state_conditioned", 1.0),),
            ),
        )
        resolved.append(
            replace(
                mechanism,
                parameters=mechanism.parameters
                + (
                    ("entity_event_counts", [int(item) for item in counts]),
                    ("realized_event_count", int(total)),
                ),
            )
        )
    resolved_plan = replace(
        plan,
        tables=tuple(table_plans[table_id] for table_id in plan.generation_order),
        population_mechanisms=tuple(resolved),
    )
    changed_table_ids = {
        item.table_id
        for item in plan.population_mechanisms
        if item.family == "negative_binomial"
        and item.parent_table_id is not None
        and item.table_id in table_plans
    }
    return _replan_downstream_populations(schema, resolved_plan, changed_table_ids)


def _apply_stateful_temporal_event_processes(
    schema: PhysicalSchema,
    plan: InstancePlan,
    database: DatabaseInstance,
) -> TemporalEventMaterialization:
    if not plan.temporal_state_plans:
        return TemporalEventMaterialization(database=database)
    tables = {item.table_id: item for item in database.tables}
    processes = {item.table_id: item for item in plan.temporal_processes}
    foreign_keys = {item.foreign_key_id: item for item in schema.foreign_keys}
    trajectories: dict[str, tuple[StateTrajectory, ...]] = {}
    schedules = _stateful_event_schedules(
        schema,
        plan,
        database,
        mechanisms=tuple(plan.population_mechanisms),
    )
    for mechanism in plan.population_mechanisms:
        process = processes.get(mechanism.table_id)
        if (
            mechanism.family != "negative_binomial"
            or mechanism.parent_table_id is None
            or process is None
            or not process.temporal_state_ids
        ):
            continue
        schedule = schedules[mechanism.table_id]
        event = tables[mechanism.table_id]
        if len(schedule.times) != event.row_count:
            raise ValueError("stateful event schedule does not match final population")
        foreign_key = foreign_keys[str(dict(mechanism.parameters)["foreign_key_id"])]
        values = dict(event.columns)
        values[foreign_key.child_column_id] = schedule.assignments
        values.update(
            _stateful_event_values(
                schema,
                plan,
                mechanism,
                process,
                schedule,
                tables[mechanism.parent_table_id],
            )
        )
        tables[mechanism.table_id] = TableData(
            table_id=mechanism.table_id,
            columns=values,
        )
        trajectories[process.temporal_state_ids[0]] = schedule.trajectories
    return TemporalEventMaterialization(
        database=DatabaseInstance(
            instance_id=database.instance_id,
            schema_id=database.schema_id,
            plan_id=database.plan_id,
            tables=tuple(tables[item.table_id] for item in schema.tables),
        ),
        temporal_states=TemporalStateRegistry(trajectories),
    )


def _stateful_event_schedule_legacy(
    schema: PhysicalSchema,
    plan: InstancePlan,
    database: DatabaseInstance,
    mechanism,
    process,
) -> _StatefulEventSchedule:
    if plan.calendar_start_seconds is None or plan.calendar_end_seconds is None:
        raise ValueError("temporal state process requires a calendar")
    temporal_state_id = process.temporal_state_ids[0]
    temporal_plan = next(
        item for item in plan.temporal_state_plans if item.state_id == temporal_state_id
    )
    entity_id = mechanism.parent_table_id
    assert entity_id is not None
    entity = database.table(entity_id)
    static = SharedStateRegistry.from_plan(plan).state(
        temporal_plan.shared_state_id
    )
    if len(static) != entity.row_count:
        raise ValueError("static state must align with temporal-state owner")
    rng = np.random.Generator(
        np.random.PCG64DXSM(temporal_plan.seed ^ plan.global_seed)
    )
    attribute_score = _entity_attribute_score(
        schema,
        entity_id,
        entity.columns,
        rng,
    )
    start = plan.calendar_start_seconds
    end = plan.calendar_end_seconds
    span = float(end - start)
    mechanism_parameters = dict(mechanism.parameters)
    baseline = float(mechanism_parameters["baseline_intensity"])
    base_rate = max(baseline / span, 1.0 / max(span * 100.0, 1.0))
    forest_payload = mechanism_parameters.get("event_intensity_forest")
    if isinstance(forest_payload, Mapping):
        intensity_score = _standardize(
            _tree_intensity_score(
                ForestPlan.from_dict(forest_payload),
                schema,
                entity_id,
                entity.columns,
                static,
            )
        )
    else:
        intensity_score = np.zeros(entity.row_count, dtype=np.float64)
    rule_effect = _rule_effect_for_entities(
        mechanism_parameters.get("rule_plan"),
        database,
        temporal_plan.shared_state_id,
        static,
        entity.row_count,
    )

    assignments: list[int] = []
    times: list[int] = []
    ordinals: list[int] = []
    state_indices: list[int] = []
    history: list[float] = []
    trajectories: list[StateTrajectory] = []
    state_to_index = {
        value: index for index, value in enumerate(temporal_plan.state_space.values)
    }
    for entity_index, z in enumerate(static):
        entity_rng = np.random.Generator(
            np.random.PCG64DXSM(temporal_plan.seed ^ (entity_index + 1))
        )
        initial = _initial_state(temporal_plan, z, entity_rng)
        trajectory = StateTrajectory(
            calendar_start=start,
            calendar_end=end,
            initial_state=initial,
            state_space=temporal_plan.state_space,
        )
        now = float(start)
        deadline: float | None = None
        next_ordinal: dict[int, int] = {}
        last_transition: tuple[int, int] | None = None
        entity_history = 0
        for _step in range(8192):
            if trajectory.is_terminal or now > end:
                break
            state = trajectory.state_after(
                min(end, max(start, int(round(now))))
            )
            rate = _state_rate(
                temporal_plan,
                state_to_index[state],
                z,
                float(attribute_score[entity_index]),
                base_rate
                * float(np.exp(0.55 * np.clip(intensity_score[entity_index], -5.0, 5.0)))
                * (
                    float(rule_effect.intensity_multiplier[entity_index])
                    if rule_effect is not None
                    else 1.0
                )
                + (
                    float(rule_effect.intensity_addition[entity_index])
                    if rule_effect is not None
                    else 0.0
                ),
            )
            event_time = now + _stateful_wait(
                entity_rng,
                process,
                z,
                rate,
                now,
                start,
                end,
            )
            clock = temporal_plan.transition.clock
            if clock is TransitionClock.EVENT_DRIVEN:
                if event_time > end:
                    break
                timestamp = _calendar_time(event_time, start, end)
                ordinal = next_ordinal.get(timestamp, 0)
                next_ordinal[timestamp] = ordinal + 1
                _append_scheduled_event(
                    assignments,
                    times,
                    ordinals,
                    state_indices,
                    history,
                    entity_index,
                    timestamp,
                    ordinal,
                    state_to_index[
                        trajectory.state_before(timestamp, ordinal)
                    ],
                    entity_history,
                )
                target = _next_state(temporal_plan, state, z, entity_rng)
                if rule_effect is not None:
                    override = rule_effect.state_overrides[entity_index]
                    if override in state_to_index:
                        target = str(override)
                trajectory.transition(
                    timestamp=timestamp,
                    ordinal=ordinal,
                    state=target,
                )
                last_transition = (timestamp, ordinal)
                now = event_time
                entity_history += 1
                continue

            if deadline is None:
                transition_rate = rate * float(
                    dict(temporal_plan.duration.parameters).get(
                        "transition_rate_multiplier",
                        1.0,
                    )
                )
                deadline = now + float(entity_rng.exponential(1.0 / transition_rate))
            if event_time <= deadline and event_time <= end:
                timestamp = _calendar_time(event_time, start, end)
                ordinal = next_ordinal.get(timestamp, 0)
                next_ordinal[timestamp] = ordinal + 1
                _append_scheduled_event(
                    assignments,
                    times,
                    ordinals,
                    state_indices,
                    history,
                    entity_index,
                    timestamp,
                    ordinal,
                    state_to_index[
                        trajectory.state_before(timestamp, ordinal)
                    ],
                    entity_history,
                )
                if clock is TransitionClock.HYBRID:
                    target = _next_state(temporal_plan, state, z, entity_rng)
                    if rule_effect is not None:
                        override = rule_effect.state_overrides[entity_index]
                        if override in state_to_index:
                            target = str(override)
                    trajectory.transition(
                        timestamp=timestamp,
                        ordinal=ordinal,
                        state=target,
                    )
                    last_transition = (timestamp, ordinal)
                    deadline = None
                now = event_time
                entity_history += 1
                continue
            if deadline > end:
                break
            timestamp = _calendar_time(deadline, start, end)
            ordinal = _duration_transition_ordinal(
                timestamp,
                next_ordinal,
                last_transition,
            )
            target = _next_state(temporal_plan, state, z, entity_rng)
            if rule_effect is not None:
                override = rule_effect.state_overrides[entity_index]
                if override in state_to_index:
                    target = str(override)
            trajectory.transition(
                timestamp=timestamp,
                ordinal=ordinal,
                state=target,
            )
            last_transition = (timestamp, ordinal)
            now = deadline
            deadline = None
        else:
            raise ValueError("temporal state process exceeded transition limit")
        trajectories.append(trajectory)

    return _StatefulEventSchedule(
        assignments=np.asarray(assignments, dtype=np.int64),
        times=np.asarray(times, dtype=np.int64),
        ordinals=np.asarray(ordinals, dtype=np.int64),
        state_indices=np.asarray(state_indices, dtype=np.int64),
        history=np.asarray(history, dtype=np.float64),
        trajectories=tuple(trajectories),
    )


def _append_scheduled_event(
    assignments,
    times,
    ordinals,
    state_indices,
    history,
    entity_index: int,
    timestamp: int,
    ordinal: int,
    state_index: int,
    entity_history: int,
) -> None:
    assignments.append(entity_index)
    times.append(timestamp)
    ordinals.append(ordinal)
    state_indices.append(state_index)
    history.append(float(entity_history))


def _duration_transition_ordinal(
    timestamp: int,
    next_ordinal: dict[int, int],
    last_transition: tuple[int, int] | None,
) -> int:
    if last_transition is not None and last_transition[0] == timestamp:
        ordinal = max(next_ordinal.get(timestamp, 0), last_transition[1] + 1)
    elif timestamp in next_ordinal:
        ordinal = next_ordinal[timestamp]
    else:
        ordinal = -1
    next_ordinal[timestamp] = max(next_ordinal.get(timestamp, 0), ordinal + 1)
    return ordinal


def _calendar_time(value: float, start: int, end: int) -> int:
    return min(end, max(start, int(round(value))))


def _initial_state(temporal_plan, z: np.ndarray, rng) -> str:
    parameters = dict(temporal_plan.initial_parameters)
    logits = np.asarray(parameters["bias"], dtype=np.float64)
    weights = np.asarray(parameters["z_weights"], dtype=np.float64)
    return temporal_plan.state_space.values[
        int(rng.choice(len(logits), p=_softmax(logits + weights @ z)))
    ]


def _next_state(temporal_plan, state: str, z: np.ndarray, rng) -> str:
    values = temporal_plan.state_space.values
    source = values.index(state)
    if temporal_plan.transition.family == "transition_matrix":
        matrix = np.asarray(
            dict(temporal_plan.transition.parameters)["matrix"],
            dtype=np.float64,
        )
        probabilities = matrix[source]
    else:
        parameters = dict(temporal_plan.transition.parameters)
        bias = np.asarray(parameters["bias"], dtype=np.float64)[source]
        weights = np.asarray(parameters["z_weights"], dtype=np.float64)[source]
        probabilities = _softmax(bias + weights @ z)
    return values[int(rng.choice(len(values), p=probabilities))]


def _state_rate(
    temporal_plan,
    state_index: int,
    z: np.ndarray,
    attribute_score: float,
    base_rate: float,
) -> float:
    parameters = dict(temporal_plan.duration.parameters)
    multiplier = np.asarray(
        parameters["state_rate_multipliers"],
        dtype=np.float64,
    )[state_index]
    z_weights = np.asarray(parameters["z_weights"], dtype=np.float64)
    z_effect = float(np.dot(z, z_weights) / max(np.sqrt(len(z)), 1.0))
    conditioned = multiplier if temporal_plan.duration.state_conditioned else 1.0
    attribute_score = float(np.nan_to_num(attribute_score, nan=0.0))
    return max(base_rate * conditioned * np.exp(0.35 * z_effect + 0.20 * attribute_score), 1e-12)


def _stateful_wait(
    rng: np.random.Generator,
    process,
    state: np.ndarray,
    rate: float,
    now: float,
    start: int,
    end: int,
) -> float:
    """Draw one inter-arrival duration from the configured state-aware family."""
    parameters = dict(process.parameters)
    elapsed = (now - start) / max(float(end - start), 1.0)
    active = 0.15 + 0.85 * _sigmoid(
        _state_projection(state, parameters, "state_active_interval_weights")
    )
    adjusted_rate = max(rate * active, 1e-12)
    if process.family == "seasonal":
        strength = float(parameters.get("seasonal_strength", 0.5))
        phase = _state_projection(
            state, parameters, "state_seasonal_phase_weights"
        )
        adjusted_rate *= max(
            0.05,
            1.0 + strength * np.sin(2.0 * np.pi * (elapsed + phase)),
        )
    elif process.family == "churn":
        exponent = max(
            0.1,
            float(parameters.get("churn_exponent", 2.0))
            + _softplus(
                _state_projection(state, parameters, "state_churn_weights")
            ),
        )
        adjusted_rate *= np.exp(-exponent * elapsed)
    elif process.family == "renewal":
        scale = np.exp(
            -np.clip(
                _state_projection(
                    state, parameters, "state_renewal_scale_weights"
                ),
                -2.0,
                2.0,
            )
        )
        return float(rng.gamma(shape=2.0, scale=scale / (2.0 * adjusted_rate)))
    elif process.family != "stationary":
        raise ValueError(f"unsupported temporal process family: {process.family}")
    return float(rng.exponential(1.0 / max(adjusted_rate, 1e-12)))


def _replan_downstream_populations(
    schema: PhysicalSchema,
    plan: InstancePlan,
    changed_table_ids: set[str],
) -> InstancePlan:
    """Propagate final Event population changes to dependent child tables."""
    if not changed_table_ids:
        return plan
    tables = {item.table_id: item for item in plan.tables}
    changed = set(changed_table_ids)
    for table_id in plan.generation_order:
        if table_id in changed:
            continue
        incoming = [
            foreign_key
            for foreign_key in schema.foreign_keys
            if foreign_key.child_table_id == table_id
            and foreign_key.parent_table_id in changed
            and foreign_key.relation_strategy != "lookup_assignment"
        ]
        if not incoming:
            continue
        table = tables[table_id]
        parent_counts = [
            tables[foreign_key.parent_table_id].population.row_count
            for foreign_key in incoming
        ]
        multiplier = float(table.population.parameter_map.get("multiplier", 1.0))
        if table.population.strategy == "joint_bridge_population" and len(parent_counts) > 1:
            target = int(
                round(np.sqrt(np.prod(parent_counts, dtype=np.float64)) * multiplier)
            )
        else:
            target = int(round(max(parent_counts) * multiplier))
        target = max(1, target)
        parameters = tuple(
            item
            for item in table.population.parameters
            if item[0] != "upstream_population_replanned"
        ) + (("upstream_population_replanned", 1.0),)
        tables[table_id] = replace(
            table,
            population=PopulationPlan(
                strategy=table.population.strategy,
                row_count=target,
                parameters=parameters,
            ),
        )
        changed.add(table_id)
    return replace(
        plan,
        tables=tuple(tables[table_id] for table_id in plan.generation_order),
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    result = np.exp(np.clip(shifted, -40.0, 40.0))
    return result / result.sum()


def _stateful_event_values(
    schema: PhysicalSchema,
    plan: InstancePlan,
    mechanism,
    process,
    schedule: _StatefulEventSchedule,
    entity_table: TableData,
) -> dict[str, np.ndarray]:
    """Use the same sampled column mechanism and physical encoder as P1."""
    table = schema.table(mechanism.table_id)
    temporal_state_id = process.temporal_state_ids[0]
    temporal_plan = next(
        item for item in plan.temporal_state_plans if item.state_id == temporal_state_id
    )
    static = SharedStateRegistry.from_plan(plan).state(
        temporal_plan.shared_state_id
    )[schedule.assignments]
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            temporal_plan.seed ^ plan.table(mechanism.table_id).feature_seed
        )
    )
    state_score = _standardize(schedule.state_indices.astype(np.float64))
    time_score = (
        (schedule.times - plan.calendar_start_seconds)
        / max(plan.calendar_end_seconds - plan.calendar_start_seconds, 1)
    )
    history = schedule.history / max(float(schedule.history.max(initial=0.0)), 1.0)
    mechanisms = {
        item.column_id: item for item in plan.column_mechanisms
    }
    table_plan = plan.table(mechanism.table_id)
    feature_columns = [
        item for item in table.columns if item.kind is ColumnKind.FEATURE
    ]
    emission_column_id = feature_columns[0].column_id if feature_columns else None
    rule_effect = None
    rule_payload = dict(mechanism.parameters).get("rule_plan")
    if isinstance(rule_payload, Mapping):
        owner_state = SharedStateRegistry.from_plan(plan).state(temporal_plan.shared_state_id)
        row_count = len(next(iter(entity_table.columns.values())))
        rule_effect = RuleProcessExecutor().effects(
            RulePlan.from_dict(rule_payload),
            RuleEvaluationContext(
                tables={entity_table.table_id: entity_table},
                states={temporal_plan.shared_state_id: owner_state},
            ),
            row_count=row_count,
        )
    output: dict[str, np.ndarray] = {}
    for column in table.columns:
        if column.kind is ColumnKind.TIME:
            output[column.column_id] = schedule.times
            continue
        if column.kind is not ColumnKind.FEATURE:
            continue
        score = _execute_column_mechanism(
            mechanisms.get(column.column_id),
            static,
            entity_table.columns,
            schedule.assignments,
            time_score,
            history,
            rng,
        )
        if rule_effect is not None:
            score = score + rule_effect.attribute_offset[schedule.assignments]
        if column.column_id == emission_column_id and (
            temporal_plan.visibility is StateVisibility.EVENT_EMISSION
        ):
            score += 1.25 * state_score
        output[column.column_id] = encode_feature_score(
            score,
            column,
            table.role,
            rng,
            cardinality=int(table_plan.parameter_map["categorical_cardinality"]),
            db_start=plan.calendar_start_seconds,
            db_end=plan.calendar_end_seconds,
            categorical_dirichlet_alpha=float(
                table_plan.parameter_map.get("categorical_dirichlet_alpha", 1.0)
            ),
            categorical_signal_strength=float(
                table_plan.parameter_map.get("categorical_signal_strength", 1.0)
            ),
            missing_rate=float(table_plan.parameter_map["missing_rate"]),
            long_tail_enabled=bool(
                table_plan.parameter_map.get("long_tail_enabled", 0.0)
            ),
            long_tail_alpha=float(
                table_plan.parameter_map.get("long_tail_alpha", 1.0)
            ),
        )
    return output
__all__ = [
    "TemporalEventMaterialization",
    "apply_temporal_event_processes",
    "resolve_temporal_population_plan",
]
