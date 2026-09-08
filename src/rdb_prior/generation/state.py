"""Private static latent-state registry."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import numpy as np

from rdb_prior.generation.latent import LatentRegistry, generate_latent_registry
from rdb_prior.instance.plan import InstancePlan
from rdb_prior.priors.model import SharedStatePlan


class SharedStateRegistry:
    """Deterministic private static latents keyed by SharedStatePlan ID."""

    def __init__(self, values: Mapping[str, np.ndarray]) -> None:
        self._values = {
            state_id: np.asarray(value, dtype=np.float64)
            for state_id, value in values.items()
        }

    @classmethod
    def from_plan(
        cls,
        plan: InstancePlan,
        *,
        latents: LatentRegistry | None = None,
    ) -> "SharedStateRegistry":
        values: dict[str, np.ndarray] = {}
        if plan.shared_states:
            for state in plan.shared_states:
                values[state.state_id] = _sample_shared_state(plan, state)
            return cls(values)

        # P1 artifacts only record state IDs.  Preserve their historical
        # private view by falling back to their owner-table latent matrix.
        registry = latents or generate_latent_registry(plan)
        for mechanism in plan.population_mechanisms:
            if mechanism.parent_table_id is None:
                continue
            parent = registry.table(mechanism.parent_table_id).values
            for state_id in mechanism.state_ids:
                values[state_id] = parent
        return cls(values)

    def state(self, state_id: str) -> np.ndarray:
        try:
            return self._values[state_id]
        except KeyError as error:
            raise KeyError(f"No shared state for {state_id!r}") from error

    def values(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(self._values)


def _sample_shared_state(plan: InstancePlan, state: SharedStatePlan) -> np.ndarray:
    rows = plan.table(state.owner_table_id).population.row_count
    rng = np.random.Generator(np.random.PCG64DXSM(state.seed))
    if state.family not in {"gaussian_mixture", "gaussian_latent"}:
        raise ValueError(f"unsupported shared-state family: {state.family}")
    components = min(3, max(2, rows // 8))
    means = rng.normal(scale=0.8, size=(components, state.dimension))
    scales = np.exp(rng.uniform(-0.35, 0.35, size=(components, state.dimension)))
    weights = rng.dirichlet(np.ones(components))
    assignment = rng.choice(components, size=rows, p=weights)
    values = rng.normal(
        loc=means[assignment],
        scale=scales[assignment],
        size=(rows, state.dimension),
    )
    values -= values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return values / std


def shared_state_values(
    plan: InstancePlan,
    latents: LatentRegistry | None = None,
) -> Mapping[str, np.ndarray]:
    """Compatibility view for callers that used the P1 helper."""
    return SharedStateRegistry.from_plan(plan, latents=latents).values()


__all__ = ["SharedStateRegistry", "shared_state_values"]
