"""Executable relational SCM propensity and population mechanisms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from rdb_prior.compilation.model import ColumnKind, PhysicalSchema
from rdb_prior.generation.latent import LatentRegistry, generate_latent_registry
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.instance.plan import (
    InstancePlan,
    PopulationMechanismPlan,
    PopulationPlan,
    RelationMechanismPlan,
)


def generate_relation_scm(
    plan: RelationMechanismPlan,
    *,
    child_rows: int,
    latents: LatentRegistry,
    population_counts: np.ndarray | None = None,
) -> np.ndarray:
    """Sample FK assignments from direct propensity scores."""
    if len(plan.foreign_key_ids) != 1:
        raise ValueError("SCM propensity currently supports one FK")
    parent = latents.table(plan.parent_table_ids[0])
    child = latents.table(plan.child_table_id).values
    rng = np.random.Generator(np.random.PCG64DXSM(plan.seed))
    if population_counts is not None:
        counts = np.asarray(population_counts, dtype=np.int64)
        if counts.shape == (len(parent.activity),) and int(counts.sum()) == child_rows:
            assignments = np.repeat(
                np.arange(len(parent.activity), dtype=np.int64),
                counts,
            )
            rng.shuffle(assignments)
            return _apply_optional(assignments, plan.optional_rates[0], rng)
    values = np.empty(child_rows, dtype=np.int64)
    parent_activity = _standardize(np.log(np.maximum(parent.activity, 1e-12)))
    for index, child_value in enumerate(child[:child_rows]):
        scores = _propensity_scores(
            plan.family,
            child_value,
            parent.values,
            parent_activity,
        )
        values[index] = _categorical(scores, rng)
    return _apply_optional(values, plan.optional_rates[0], rng)


def _propensity_scores(
    family: str,
    child: np.ndarray,
    parent: np.ndarray,
    parent_activity: np.ndarray,
) -> np.ndarray:
    latent = parent[:, 0]
    child_value = float(child[0])
    if family == "scm_softmax_affinity":
        return child_value * latent + 0.8 * parent_activity
    if family == "scm_cpt":
        state = int(child_value > 0.0)
        return (1.0 - 2.0 * state) * latent + 0.35 * parent_activity
    if family == "scm_community":
        community = latent >= float(np.median(latent))
        return np.where(community == (child_value >= 0.0), 1.5, -1.5) + 0.35 * parent_activity
    if family != "scm_logistic_propensity":
        raise ValueError(f"unsupported relational SCM relation family: {family}")
    return 0.9 * child_value * latent + 0.8 * parent_activity


def _categorical(scores: np.ndarray, rng: np.random.Generator) -> int:
    shifted = np.clip(scores - float(np.max(scores)), -40.0, 0.0)
    weights = np.exp(shifted)
    weights /= weights.sum()
    return int(rng.choice(len(weights), p=weights))


def _apply_optional(
    values: np.ndarray,
    rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    result = values.astype(np.int64, copy=True)
    if rate > 0:
        result[rng.random(len(result)) < rate] = -1
    return result


def resolve_relational_scm_population(
    schema: PhysicalSchema,
    plan: InstancePlan,
    draft_database: DatabaseInstance,
) -> InstancePlan:
    """Materialize Poisson/NB/ZINB child counts after parent attributes exist."""
    if plan.prior_family != "relational_scm":
        return plan
    latents = generate_latent_registry(plan)
    tables = {item.table_id: item for item in plan.tables}
    resolved: list[PopulationMechanismPlan] = []
    changed: set[str] = set()
    for mechanism in plan.population_mechanisms:
        if mechanism.parent_table_id is None:
            resolved.append(mechanism)
            continue
        parent_data = draft_database.table(mechanism.parent_table_id)
        parent_latent = latents.table(mechanism.parent_table_id).values[:, 0]
        parent_feature = _first_feature(schema, parent_data, mechanism.parent_table_id)
        score = _standardize(0.65 * parent_latent + 0.35 * parent_feature)
        parameters = dict(mechanism.parameters)
        baseline = float(parameters.get("baseline", 1.0))
        intensity = np.clip(baseline * np.exp(0.45 * score), 0.05, max(12.0, baseline * 6.0))
        rng = np.random.Generator(
            np.random.PCG64DXSM(
                plan.table(mechanism.table_id).temporal_seed ^ plan.global_seed
            )
        )
        family = mechanism.family
        if family == "poisson":
            counts = rng.poisson(intensity)
        elif family == "zero_inflated_negative_binomial":
            dispersion = float(parameters.get("dispersion", 2.5))
            probability = dispersion / (dispersion + intensity)
            counts = rng.negative_binomial(dispersion, probability)
            counts[rng.random(len(counts)) < float(parameters.get("zero_probability", 0.25))] = 0
        elif family == "negative_binomial":
            dispersion = float(parameters.get("dispersion", 2.5))
            probability = dispersion / (dispersion + intensity)
            counts = rng.negative_binomial(dispersion, probability)
        else:
            raise ValueError(f"unsupported relational SCM population family: {family}")
        total = int(counts.sum())
        draft_table = tables[mechanism.table_id]
        hard_limit = max(draft_table.population.row_count * 4, 128)
        if total < 1:
            raise ValueError("relational SCM produced no child rows; retry materialization")
        if total > hard_limit:
            raise ValueError("relational SCM child population exceeded its hard limit")
        changed.add(mechanism.table_id)
        base_parameters = tuple(
            item
            for item in mechanism.parameters
            if item[0] not in {"entity_event_counts", "realized_child_count"}
        )
        resolved.append(
            PopulationMechanismPlan(
                table_id=mechanism.table_id,
                family=mechanism.family,
                parent_table_id=mechanism.parent_table_id,
                state_ids=mechanism.state_ids,
                parameters=base_parameters
                + (
                    ("entity_event_counts", [int(item) for item in counts]),
                    ("realized_child_count", total),
                ),
            )
        )
        tables[mechanism.table_id] = _replace_table_population(
            tables[mechanism.table_id],
            total,
        )
    result = plan
    if changed:
        result = replace(
            result,
            tables=tuple(tables[table_id] for table_id in result.generation_order),
            population_mechanisms=tuple(resolved),
        )
    return result


def _replace_table_population(table, row_count: int):
    parameters = tuple(
        item
        for item in table.population.parameters
        if item[0] != "relational_scm_conditioned"
    )
    return replace(
        table,
        population=PopulationPlan(
            strategy="relational_scm_population",
            row_count=row_count,
            parameters=parameters + (("relational_scm_conditioned", 1.0),),
        ),
    )


def _first_feature(schema, table_data, table_id: str) -> np.ndarray:
    feature = next(
        (
            item
            for item in schema.table(table_id).columns
            if item.kind is ColumnKind.FEATURE
        ),
        None,
    )
    if feature is None:
        return np.zeros(table_data.row_count, dtype=np.float64)
    raw = np.asarray(table_data.column(feature.column_id))
    if raw.dtype.kind in {"b", "i", "u", "f"}:
        values = raw.astype(np.float64)
    else:
        _categories, values = np.unique(raw.astype(str), return_inverse=True)
        values = values.astype(np.float64)
    return _standardize(np.nan_to_num(values, nan=0.0))


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - float(values.mean())) / max(float(values.std()), 1e-6)


__all__ = ["generate_relation_scm", "resolve_relational_scm_population"]
