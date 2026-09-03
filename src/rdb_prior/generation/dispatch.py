"""Family dispatch for final plan resolution before database export."""

from __future__ import annotations

from collections.abc import Callable

from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.generation.populations import resolve_state_conditioned_populations
from rdb_prior.instance.plan import InstancePlan


_Finalizer = Callable[[PhysicalSchema, InstancePlan, DatabaseInstance], InstancePlan]
_FINALIZERS: dict[str, _Finalizer] = {
    "temporal_event": resolve_state_conditioned_populations,
}


def requires_finalization(plan: InstancePlan) -> bool:
    return plan.prior_family in _FINALIZERS and bool(plan.population_mechanisms)


def finalize_plan(
    schema: PhysicalSchema,
    plan: InstancePlan,
    draft_database: DatabaseInstance,
) -> InstancePlan:
    finalizer = _FINALIZERS.get(plan.prior_family)
    return plan if finalizer is None else finalizer(schema, plan, draft_database)


__all__ = ["finalize_plan", "requires_finalization"]
