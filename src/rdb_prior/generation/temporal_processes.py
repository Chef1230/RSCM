"""Materialization of P1 state-conditioned Entity--Event processes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from rdb_prior.compilation.model import ColumnKind, PhysicalDataType, PhysicalSchema
from rdb_prior.generation.latent import generate_latent_registry
from rdb_prior.generation.model import DatabaseInstance, TableData
from rdb_prior.generation.state import SharedStateRegistry
from rdb_prior.generation.state_trajectory import (
    StateTrajectory,
    TemporalStateRegistry,
)
from rdb_prior.instance.plan import InstancePlan, PopulationPlan
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
    if plan.prior_family != "temporal_event" or not plan.population_mechanisms:
        return plan
    latents = generate_latent_registry(plan)
    table_plans = {item.table_id: item for item in plan.tables}
    resolved: list = []
    for mechanism in plan.population_mechanisms:
        if mechanism.family != "negative_binomial" or mechanism.parent_table_id is None:
            resolved.append(mechanism)
            continue
        parameters = dict(mechanism.parameters)
        entity_id = mechanism.parent_table_id
        event_id = mechanism.table_id
        entity = entity_database.table(entity_id)
        state = latents.table(entity_id).values
        row_count = entity.row_count
        rng = np.random.Generator(
            np.random.PCG64DXSM(plan.table(event_id).temporal_seed ^ 0x5EED5EED)
        )
        z_weights = rng.normal(0.0, 0.45, size=state.shape[1])
        z_score = _standardize(state @ z_weights)
        x_score = _entity_attribute_score(schema, entity_id, entity.columns, rng)
        family = str(parameters.get("intensity_family", "linear"))
        if family == "cam":
            score = 0.55 * z_score + 0.45 * x_score + 0.25 * np.sin(z_score * x_score)
        else:
            score = 0.60 * z_score + 0.40 * x_score
        baseline = float(parameters["baseline_intensity"])
        intensity = np.clip(
            baseline * np.exp(0.55 * _standardize(score)),
            0.03,
            max(12.0, baseline * 6.0),
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
    return replace(
        plan,
        tables=tuple(table_plans[table_id] for table_id in plan.generation_order),
        population_mechanisms=tuple(resolved),
    )


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
    if plan.prior_family != "temporal_event" or not plan.population_mechanisms:
        return database
    latents = generate_latent_registry(plan)
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
        entity_latent = latents.table(entity_id).values
        entity_table = tables[entity_id]
        event = tables[mechanism.table_id]
        row_count = event.row_count
        table_seed = sum((index + 1) * ord(char) for index, char in enumerate(mechanism.table_id))
        rng = np.random.Generator(np.random.PCG64DXSM(plan.global_seed ^ table_seed))
        count_values = parameters.get("entity_event_counts")
        if isinstance(count_values, list) and len(count_values) == len(entity_latent):
            assignments = np.repeat(
                np.arange(len(entity_latent), dtype=np.int64),
                np.asarray(count_values, dtype=np.int64),
            )
            if len(assignments) != row_count:
                raise ValueError("resolved temporal population no longer matches Event table rows")
        else:
            coefficients = rng.normal(0.0, 0.5, size=entity_latent.shape[1])
            score = entity_latent @ coefficients
            probability = np.exp(0.55 * (score - score.max()))
            probability /= probability.sum()
            assignments = rng.choice(len(entity_latent), size=row_count, p=probability).astype(np.int64)
        values = dict(event.columns)
        values[foreign_key.child_column_id] = assignments
        process = process_by_table.get(mechanism.table_id)
        if process is not None:
            values.update(_event_values(schema, event_table, plan, assignments, entity_latent, _entity_attribute_score(schema, entity_id, entity_table.columns, rng), process, columns_by_id, rng))
        tables[mechanism.table_id] = TableData(table_id=mechanism.table_id, columns=values)
    return DatabaseInstance(instance_id=database.instance_id, schema_id=database.schema_id, plan_id=database.plan_id, tables=tuple(tables[table.table_id] for table in schema.tables))


def _event_values(schema, event_table, plan, assignments, entity_latent, entity_attributes, process, column_mechanisms, rng):
    start = plan.calendar_start_seconds
    end = plan.calendar_end_seconds
    if start is None or end is None:
        raise ValueError("temporal prior requires a database calendar")
    rows = len(assignments)
    family = process.family
    raw = np.empty(rows, dtype=np.float64)
    for entity_index in range(len(entity_latent)):
        indices = np.flatnonzero(assignments == entity_index)
        if not len(indices):
            continue
        if family == "churn":
            exponent = float(dict(process.parameters).get("churn_exponent", 2.0))
            ticks = rng.random(len(indices)) ** exponent
        elif family == "seasonal":
            strength = float(dict(process.parameters).get("seasonal_strength", 0.5))
            candidates = rng.random(max(len(indices) * 3, 8))
            acceptance = (1.0 + strength * np.sin(2.0 * np.pi * candidates)) / (1.0 + strength)
            accepted = candidates[rng.random(len(candidates)) <= acceptance]
            if len(accepted) < len(indices):
                accepted = np.concatenate((accepted, rng.random(len(indices) - len(accepted))))
            ticks = accepted[: len(indices)]
        else:
            ticks = rng.random(len(indices))
        raw[indices] = np.sort(ticks)
    times = (start + raw * (end - start)).astype(np.int64)
    output: dict[str, np.ndarray] = {}
    for column in event_table.columns:
        if column.kind is ColumnKind.TIME:
            output[column.column_id] = times
    history = np.zeros(rows, dtype=np.float64)
    for entity_index in np.unique(assignments):
        indices = np.flatnonzero(assignments == entity_index)
        ranked = indices[np.argsort(times[indices], kind="stable")]
        history[ranked] = np.arange(len(ranked), dtype=np.float64)
    history /= max(float(np.max(history)), 1.0)
    state = entity_latent[assignments]
    state_score = state @ rng.normal(0.0, 0.45, size=state.shape[1])
    attribute_score = entity_attributes[assignments]
    time_score = (times - start) / max(end - start, 1)
    for column in event_table.columns:
        if column.kind is not ColumnKind.FEATURE:
            continue
        mechanism = column_mechanisms.get(column.column_id)
        parameters = {} if mechanism is None else dict(mechanism.parameters)
        score = state_score + 0.45 * attribute_score + float(parameters.get("time_weight", 0.5)) * time_score + float(parameters.get("history_weight", 0.5)) * history + rng.normal(0.0, 0.25, size=rows)
        output[column.column_id] = _encode(score, column.data_type)
    return output


def _encode(score: np.ndarray, data_type: PhysicalDataType) -> np.ndarray:
    score = np.nan_to_num(np.asarray(score, dtype=np.float64), nan=0.0, posinf=12.0, neginf=-12.0)
    if data_type is PhysicalDataType.DOUBLE:
        return score.astype(np.float64)
    if data_type is PhysicalDataType.INTEGER:
        return np.rint(10.0 + 3.0 * score).astype(np.int64)
    if data_type is PhysicalDataType.BOOLEAN:
        return (score > np.median(score)).astype(bool)
    levels = np.mod(np.floor((score - score.min()) * 3.0), 8).astype(np.int64)
    return np.asarray([f"v_{item}" for item in levels], dtype="U8")


def _entity_attribute_score(schema, table_id, columns, rng):
    table = schema.table(table_id)
    features: list[np.ndarray] = []
    for column in table.columns:
        if column.kind is not ColumnKind.FEATURE:
            continue
        values = np.asarray(columns[column.column_id])
        if values.dtype.kind in {"b", "i", "u", "f"}:
            encoded = values.astype(np.float64)
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




@dataclass(frozen=True, slots=True)
class _StatefulEventSchedule:
    assignments: np.ndarray
    times: np.ndarray
    ordinals: np.ndarray
    state_indices: np.ndarray
    history: np.ndarray
    trajectories: tuple[StateTrajectory, ...]


def _resolve_stateful_temporal_population_plan(
    schema: PhysicalSchema,
    plan: InstancePlan,
    entity_database: DatabaseInstance,
) -> InstancePlan:
    table_plans = {item.table_id: item for item in plan.tables}
    process_by_table = {item.table_id: item for item in plan.temporal_processes}
    resolved = []
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
        schedule = _stateful_event_schedule(
            schema,
            plan,
            entity_database,
            mechanism,
            process,
        )
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
    return replace(
        plan,
        tables=tuple(table_plans[table_id] for table_id in plan.generation_order),
        population_mechanisms=tuple(resolved),
    )


def _apply_stateful_temporal_event_processes(
    schema: PhysicalSchema,
    plan: InstancePlan,
    database: DatabaseInstance,
) -> TemporalEventMaterialization:
    if plan.prior_family != "temporal_event":
        return TemporalEventMaterialization(database=database)
    tables = {item.table_id: item for item in database.tables}
    processes = {item.table_id: item for item in plan.temporal_processes}
    foreign_keys = {item.foreign_key_id: item for item in schema.foreign_keys}
    trajectories: dict[str, tuple[StateTrajectory, ...]] = {}
    for mechanism in plan.population_mechanisms:
        process = processes.get(mechanism.table_id)
        if (
            mechanism.family != "negative_binomial"
            or mechanism.parent_table_id is None
            or process is None
            or not process.temporal_state_ids
        ):
            continue
        schedule = _stateful_event_schedule(
            schema,
            plan,
            database,
            mechanism,
            process,
        )
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


def _stateful_event_schedule(
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
    baseline = float(dict(mechanism.parameters)["baseline_intensity"])
    base_rate = max(baseline / span, 1.0 / max(span * 100.0, 1.0))

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
                base_rate,
            )
            event_time = now + float(entity_rng.exponential(1.0 / rate))
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
) -> dict[str, np.ndarray]:
    table = schema.table(mechanism.table_id)
    temporal_state_id = process.temporal_state_ids[0]
    temporal_plan = next(
        item for item in plan.temporal_state_plans if item.state_id == temporal_state_id
    )
    static = SharedStateRegistry.from_plan(plan).state(
        temporal_plan.shared_state_id
    )[schedule.assignments]
    rng = np.random.Generator(
        np.random.PCG64DXSM(temporal_plan.seed ^ plan.table(mechanism.table_id).feature_seed)
    )
    state_score = schedule.state_indices.astype(np.float64)
    state_score = _standardize(state_score)
    z_score = static @ rng.normal(0.0, 0.4, size=static.shape[1])
    time_score = (
        (schedule.times - plan.calendar_start_seconds)
        / max(plan.calendar_end_seconds - plan.calendar_start_seconds, 1)
    )
    history = schedule.history / max(float(schedule.history.max(initial=0.0)), 1.0)
    output: dict[str, np.ndarray] = {}
    feature_columns = [
        item for item in table.columns if item.kind is ColumnKind.FEATURE
    ]
    emission_column_id = (
        feature_columns[0].column_id if feature_columns else None
    )
    for column in table.columns:
        if column.kind is ColumnKind.TIME:
            output[column.column_id] = schedule.times
        elif column.kind is ColumnKind.FEATURE:
            score = (
                z_score
                + 0.7 * state_score
                + 0.5 * time_score
                + 0.4 * history
                + rng.normal(0.0, 0.2, size=len(schedule.times))
            )
            if column.column_id == emission_column_id and (
                temporal_plan.visibility is StateVisibility.EVENT_EMISSION
            ):
                score += 1.25 * state_score
            output[column.column_id] = _encode(score, column.data_type)
    return output
__all__ = [
    "TemporalEventMaterialization",
    "apply_temporal_event_processes",
    "resolve_temporal_population_plan",
]
