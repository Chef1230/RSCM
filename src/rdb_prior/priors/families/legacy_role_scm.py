"""Compatibility binder for the historical role-SCM implementation."""

from __future__ import annotations

from dataclasses import replace

from rdb_prior.instance.plan import InstancePlan
from rdb_prior.priors.model import DatabasePriorPlan


def bind_legacy_plan(plan: InstancePlan, prior_plan: DatabasePriorPlan) -> InstancePlan:
    """Attach provenance without changing the historical generated values."""
    return replace(
        plan,
        prior_plan_id=prior_plan.plan_id,
        prior_composition_id=prior_plan.composition.plan_id,
        prior_family=prior_plan.family.value,
        motif_bundles=prior_plan.motif_bundles,
        nuisance_plan=prior_plan.composition.nuisance_plan,
        shared_state_ids=tuple(item.state_id for item in prior_plan.shared_states),
    )


__all__ = ["bind_legacy_plan"]
