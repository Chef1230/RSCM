"""Private shared-state views used by prior-family materialization."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import numpy as np

from rdb_prior.generation.latent import LatentRegistry, generate_latent_registry
from rdb_prior.instance.plan import InstancePlan


def shared_state_values(
    plan: InstancePlan,
    latents: LatentRegistry | None = None,
) -> Mapping[str, np.ndarray]:
    """Return private ``state_id -> z_i`` arrays without exposing data columns."""
    registry = latents or generate_latent_registry(plan)
    values: dict[str, np.ndarray] = {}
    for mechanism in plan.population_mechanisms:
        if mechanism.parent_table_id is None:
            continue
        parent = registry.table(mechanism.parent_table_id).values
        for state_id in mechanism.state_ids:
            values[state_id] = parent
    return MappingProxyType(values)


__all__ = ["shared_state_values"]
