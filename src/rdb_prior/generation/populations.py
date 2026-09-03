"""Conditional population materializers for registered prior families."""

from __future__ import annotations

from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.generation.model import DatabaseInstance
from rdb_prior.generation.temporal_processes import resolve_temporal_population_plan
from rdb_prior.instance.plan import InstancePlan


def resolve_state_conditioned_populations(
    schema: PhysicalSchema,
    plan: InstancePlan,
    entity_database: DatabaseInstance,
) -> InstancePlan:
    """Resolve Entity--Event P1 counts after entity attributes are available."""
    return resolve_temporal_population_plan(schema, plan, entity_database)


__all__ = ["resolve_state_conditioned_populations"]
