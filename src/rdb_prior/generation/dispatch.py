"""Family dispatch for final plan resolution before database export."""

from __future__ import annotations

from collections.abc import Callable

from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.generation.populations import resolve_state_conditioned_populations
from rdb_prior.generation.relation_scm import resolve_relational_scm_population
from rdb_prior.instance.plan import InstancePlan


_Finalizer = Callable[[PhysicalSchema, InstancePlan, DatabaseInstance], InstancePlan]


def _has_temporal_population(plan: InstancePlan) -> bool:
    return bool(plan.temporal_state_plans) or any(
        dict(item.parameters).get("population_source") == "temporal_event"
        for item in plan.population_mechanisms
    ) or any(item.temporal_state_ids for item in plan.temporal_processes) or (
        plan.prior_family in {"temporal_event", "rule_process"}
        and bool(plan.population_mechanisms)
    )


def _has_relational_scm_population(plan: InstancePlan) -> bool:
    return any(
        dict(item.parameters).get("population_source") == "relational_scm"
        for item in plan.population_mechanisms
    ) or any(item.family.startswith("scm_") for item in plan.relations)


def requires_finalization(plan: InstancePlan) -> bool:
    return bool(plan.population_mechanisms) and (
        _has_temporal_population(plan)
        or _has_relational_scm_population(plan)
    )


def finalize_plan(
    schema: PhysicalSchema,
    plan: InstancePlan,
    draft_database: DatabaseInstance,
) -> InstancePlan:
    result = plan
    if _has_temporal_population(plan):
        result = resolve_state_conditioned_populations(schema, result, draft_database)
    if _has_relational_scm_population(result):
        result = resolve_relational_scm_population(schema, result, draft_database)
    return result


__all__ = ["finalize_plan", "requires_finalization"]
