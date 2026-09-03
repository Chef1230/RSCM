"""Materialization of P1 state-conditioned Entity--Event processes."""

from __future__ import annotations

import numpy as np

from rdb_prior.compilation.model import ColumnKind, PhysicalDataType, PhysicalSchema
from rdb_prior.generation.latent import generate_latent_registry
from rdb_prior.generation.model import DatabaseInstance, TableData
from rdb_prior.instance.plan import InstancePlan, PopulationPlan
from dataclasses import replace


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


def apply_temporal_event_processes(
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


__all__ = ["apply_temporal_event_processes", "resolve_temporal_population_plan"]
