"""Executable discrete relation mechanisms backed by shared latents."""

from __future__ import annotations

import numpy as np

from rdb_prior.generation.latent import LatentRegistry
from rdb_prior.generation.trees.executor import evaluate_forest
from rdb_prior.instance.plan import RelationMechanismPlan


def generate_single_relation(
    plan: RelationMechanismPlan,
    *,
    child_rows: int,
    latents: LatentRegistry,
) -> dict[str, np.ndarray]:
    if len(plan.foreign_key_ids) != 1:
        raise ValueError("single relation strategy requires one foreign key")
    rng = np.random.Generator(np.random.PCG64DXSM(plan.seed))
    parent_id = plan.parent_table_ids[0]
    child = latents.table(plan.child_table_id).values
    parent = latents.table(parent_id)

    if plan.family == "tree_propensity":
        assignments = _tree_propensity_assignments(child, parent.values, parent.activity, plan, rng)
    elif plan.family == "lookup_cpt":
        assignments = _cpt_assignments(child, len(parent.activity), rng)
    elif plan.family == "lookup_transition":
        assignments = _transition_assignments(child, len(parent.activity), rng)
    else:
        assignments = _softmax_assignments(
            child,
            parent.values,
            parent.activity,
            rng,
            affinity=plan.parameter_map.get("affinity_strength", 1.0),
            degree=plan.parameter_map.get("degree_strength", 0.8),
        )
    assignments = _apply_optional(
        assignments,
        plan.optional_rates[0],
        rng,
    )
    if len(assignments) != child_rows:
        raise RuntimeError("relation strategy returned wrong row count")
    return {plan.foreign_key_ids[0]: assignments}


def generate_affinity_bridge(
    plan: RelationMechanismPlan,
    *,
    child_rows: int,
    latents: LatentRegistry,
) -> dict[str, np.ndarray]:
    if len(plan.foreign_key_ids) < 2:
        raise ValueError("affinity bridge requires at least two foreign keys")
    rng = np.random.Generator(np.random.PCG64DXSM(plan.seed))
    child = latents.table(plan.child_table_id).values
    parent_latents = [
        latents.table(parent_id)
        for parent_id in plan.parent_table_ids
    ]
    affinity = plan.parameter_map.get("affinity_strength", 1.0)
    degree = plan.parameter_map.get("degree_strength", 0.8)
    results = [np.empty(child_rows, dtype=np.int64) for _ in parent_latents]
    seen: set[tuple[int, ...]] = set()
    pair_matrices = [
        rng.normal(
            scale=1.0 / np.sqrt(child.shape[1]),
            size=(child.shape[1], child.shape[1]),
        )
        for _ in range(max(0, len(parent_latents) - 1))
    ]
    child_parent_projections = [
        rng.normal(
            scale=1.0 / np.sqrt(child.shape[1]),
            size=(child.shape[1], child.shape[1]),
        )
        for _parent in parent_latents
    ]

    for row_index in range(child_rows):
        selected: list[int] = []
        first = _draw_parent(
            child[row_index],
            parent_latents[0].values,
            parent_latents[0].activity,
            rng,
            affinity,
            degree,
            projection=child_parent_projections[0],
        )
        selected.append(first)
        for parent_index, parent in enumerate(parent_latents[1:], start=1):
            compatibility = (
                parent_latents[0].values[first]
                @ pair_matrices[parent_index - 1]
                @ parent.values.T
            )
            value = _draw_parent(
                child[row_index],
                parent.values,
                parent.activity,
                rng,
                affinity,
                degree,
                projection=child_parent_projections[parent_index],
                extra_scores=compatibility,
            )
            selected.append(value)

        candidate = tuple(selected)
        if candidate in seen:
            candidate = _repair_duplicate_tuple(
                candidate,
                seen,
                tuple(len(parent.activity) for parent in parent_latents),
            )
        seen.add(candidate)
        for index, value in enumerate(candidate):
            results[index][row_index] = value

    encoded: dict[str, np.ndarray] = {}
    for fk_id, values, optional_rate in zip(
        plan.foreign_key_ids,
        results,
        plan.optional_rates,
        strict=True,
    ):
        encoded[fk_id] = _apply_optional(values, optional_rate, rng)
    return encoded


def _tree_propensity_assignments(
    child: np.ndarray,
    parent: np.ndarray,
    activity: np.ndarray,
    plan: RelationMechanismPlan,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw each parent from directly sampled tree propensity scores."""
    forest = plan.tree_forest
    if forest is None:
        raise ValueError("tree_propensity relation requires a sampled forest")
    if parent.shape[0] < 1:
        raise ValueError("tree_propensity relation requires at least one parent")
    parent_activity = np.log(np.maximum(activity, 1e-12))
    parent_activity = (parent_activity - parent_activity.mean()) / max(
        float(parent_activity.std()), 1e-6
    )
    values = np.empty(child.shape[0], dtype=np.int64)
    for row_index, child_value in enumerate(child):
        scores = evaluate_forest(
            forest,
            {
                "child_latent_0": np.full(parent.shape[0], child_value[0]),
                "parent_latent_0": parent[:, 0],
                "parent_activity": parent_activity,
            },
            row_count=parent.shape[0],
        )
        values[row_index] = _categorical(scores, rng)
    return values


def _softmax_assignments(
    child: np.ndarray,
    parent: np.ndarray,
    activity: np.ndarray,
    rng: np.random.Generator,
    *,
    affinity: float,
    degree: float,
) -> np.ndarray:
    dimension = child.shape[1]
    projection = rng.normal(
        scale=1.0 / np.sqrt(dimension),
        size=(dimension, dimension),
    )
    results = np.empty(child.shape[0], dtype=np.int64)
    degree_scores = degree * np.log(activity + 1e-12)
    for row_index, child_value in enumerate(child):
        scores = (
            affinity * (child_value @ projection @ parent.T)
            / np.sqrt(dimension)
            + degree_scores
        )
        results[row_index] = _categorical(scores, rng)
    return results


def _cpt_assignments(
    child: np.ndarray,
    parent_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    state_count = min(6, max(2, round(np.sqrt(child.shape[0]))))
    boundaries = np.quantile(
        child[:, 0],
        np.linspace(0, 1, state_count + 1)[1:-1],
    )
    states = np.digitize(child[:, 0], boundaries)
    concentration = rng.lognormal(mean=-0.4, sigma=0.8, size=parent_count)
    probabilities = np.vstack(
        [rng.dirichlet(concentration + 0.05) for _ in range(state_count)]
    )
    return np.asarray(
        [rng.choice(parent_count, p=probabilities[state]) for state in states],
        dtype=np.int64,
    )


def _transition_assignments(
    child: np.ndarray,
    parent_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    order = np.argsort(child[:, 0], kind="stable")
    concentration = np.full(parent_count, 0.25, dtype=np.float64)
    transitions = np.vstack(
        [rng.dirichlet(concentration) for _ in range(parent_count)]
    )
    persistence = float(rng.uniform(0.55, 0.9))
    transitions *= 1.0 - persistence
    transitions[np.arange(parent_count), np.arange(parent_count)] += persistence
    transitions /= transitions.sum(axis=1, keepdims=True)

    values = np.empty(len(child), dtype=np.int64)
    state = int(rng.integers(0, parent_count))
    for row_index in order:
        values[row_index] = state
        state = int(rng.choice(parent_count, p=transitions[state]))
    return values


def _draw_parent(
    child: np.ndarray,
    parent: np.ndarray,
    activity: np.ndarray,
    rng: np.random.Generator,
    affinity: float,
    degree: float,
    *,
    projection: np.ndarray,
    extra_scores: np.ndarray | None = None,
) -> int:
    scores = (
        affinity * (child @ projection @ parent.T) / np.sqrt(child.shape[0])
        + degree * np.log(activity + 1e-12)
    )
    if extra_scores is not None:
        scores = scores + affinity * extra_scores / np.sqrt(child.shape[0])
    return _categorical(scores, rng)


def _categorical(scores: np.ndarray, rng: np.random.Generator) -> int:
    stable = np.clip(scores - np.max(scores), -40.0, 0.0)
    weights = np.exp(stable)
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


def _repair_duplicate_tuple(
    candidate: tuple[int, ...],
    seen: set[tuple[int, ...]],
    cardinalities: tuple[int, ...],
) -> tuple[int, ...]:
    values = list(candidate)
    combinations = int(np.prod(cardinalities, dtype=np.int64))
    if len(seen) >= combinations:
        return candidate
    for offset in range(1, combinations + 1):
        carry = offset
        repaired = list(values)
        for index in range(len(repaired) - 1, -1, -1):
            repaired[index] = (repaired[index] + carry) % cardinalities[index]
            carry //= cardinalities[index]
        result = tuple(repaired)
        if result not in seen:
            return result
    return candidate


__all__ = ["generate_single_relation", "generate_affinity_bridge"]
