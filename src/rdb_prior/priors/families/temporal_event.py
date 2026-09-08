"""P1 state-conditioned Entity--Event prior binder."""

from __future__ import annotations

from dataclasses import replace
from math import exp

import numpy as np

from rdb_prior.compilation.model import ColumnKind, PhysicalSchema
from rdb_prior.generation.state import SharedStateRegistry
from rdb_prior.instance.plan import (
    ColumnMechanismPlan,
    InstancePlan,
    PopulationMechanismPlan,
    PopulationPlan,
    TemporalProcessPlan,
)
from rdb_prior.priors.model import (
    AttributePriorKind,
    DatabasePriorPlan,
    PriorFamily,
    RelationPriorKind,
    TemporalPriorKind,
)


_ATTRIBUTE_FAMILIES = ("linear", "cam")


def bind_temporal_event_plan(
    schema: PhysicalSchema,
    plan: InstancePlan,
    prior_plan: DatabasePriorPlan,
) -> InstancePlan:
    """Resolve P1 event counts from parent latent state before relation draws.

    Counts are planned deterministically here so the existing FK generator can
    preserve its integrity guarantees.  The subsequent database generator
    reuses the same seeds to materialize times and event attributes.
    """
    if prior_plan.family is not PriorFamily.TEMPORAL_EVENT:
        raise ValueError("temporal binder requires temporal_event prior")
    state_plan = replace(
        plan,
        shared_state_ids=tuple(item.state_id for item in prior_plan.shared_states),
        shared_states=prior_plan.shared_states,
    )
    states = SharedStateRegistry.from_plan(state_plan)
    table_plans = {item.table_id: item for item in plan.tables}
    population_mechanisms: list[PopulationMechanismPlan] = []
    temporal_processes: list[TemporalProcessPlan] = []
    column_mechanisms: list[ColumnMechanismPlan] = []
    for bundle in prior_plan.motif_bundles:
        if bundle.family is not PriorFamily.TEMPORAL_EVENT:
            continue
        if bundle.attribute_mechanism.kind != AttributePriorKind.COLUMN_SCM.value:
            raise ValueError(
                "temporal_event/v1 only executes column_scm attributes; "
                "tree and rule attributes require their own executor"
            )
        if (
            bundle.relation_mechanism.kind
            != RelationPriorKind.STATE_CONDITIONED_EVENT.value
        ):
            raise ValueError(
                "temporal_event/v1 only executes state_conditioned_event relations"
            )
        temporal_kind = TemporalPriorKind(bundle.temporal_mechanism.kind)
        if temporal_kind is TemporalPriorKind.STATIC:
            raise ValueError("temporal_event/v1 requires a non-static temporal mechanism")
        parameters = dict(bundle.parameters)
        entity_id = str(parameters["entity_table_id"])
        event_id = str(parameters["event_table_id"])
        state_id = str(parameters["state_id"])
        temporal_state_id = (
            str(parameters["temporal_state_id"])
            if "temporal_state_id" in parameters else None
        )
        foreign_key_id = bundle.edge_bindings[0].foreign_key_id
        entity_state = states.state(state_id)
        old_table = table_plans[event_id]
        entity_count = len(entity_state)
        baseline = max(0.20, old_table.population.row_count / max(1, entity_count))
        rng = np.random.Generator(np.random.PCG64DXSM(prior_plan.seed ^ old_table.temporal_seed))
        coefficients = rng.normal(0.0, 0.45, size=entity_state.shape[1])
        score = entity_state @ coefficients
        score = (score - np.mean(score)) / max(float(np.std(score)), 1e-6)
        intensity = np.clip(baseline * np.exp(0.55 * score), 0.03, max(12.0, baseline * 6.0))
        dispersion = float(rng.uniform(1.5, 5.0))
        probability = dispersion / (dispersion + intensity)
        counts = rng.negative_binomial(dispersion, probability).astype(np.int64)
        total = int(np.sum(counts))
        hard_limit = max(old_table.population.row_count * 4, 128)
        if total < 1:
            raise ValueError("temporal prior produced no events; retry materialization")
        if total > hard_limit:
            raise ValueError("temporal prior event population exceeded its hard limit")
        table_plans[event_id] = replace(
            old_table,
            population=PopulationPlan(
                strategy="state_conditioned_negative_binomial",
                row_count=total,
                parameters=old_table.population.parameters + (("state_conditioned", 1.0),),
            ),
        )
        population_mechanisms.append(
            PopulationMechanismPlan(
                table_id=event_id,
                family="negative_binomial",
                parent_table_id=entity_id,
                state_ids=(state_id,),
                parameters=(
                    ("foreign_key_id", foreign_key_id),
                    ("dispersion", dispersion),
                    ("baseline_intensity", baseline),
                    ("intensity_family", str(rng.choice(_ATTRIBUTE_FAMILIES))),
                    ("planned_event_count", total),
                    (
                        "count_state_weights",
                        rng.normal(0.0, 0.45, size=entity_state.shape[1]).tolist(),
                    ),
                ),
            )
        )
        time_family = temporal_kind.value
        temporal_processes.append(
            TemporalProcessPlan(
                table_id=event_id,
                family=str(time_family),
                state_ids=(state_id,),
                temporal_state_ids=(
                    () if temporal_state_id is None else (temporal_state_id,)
                ),
                parameters=(
                    ("foreign_key_id", foreign_key_id),
                    ("seasonal_strength", float(rng.uniform(0.25, 0.75))),
                    ("churn_exponent", float(rng.uniform(1.25, 3.0))),
                    (
                        "state_churn_weights",
                        rng.normal(0.0, 0.35, size=entity_state.shape[1]).tolist(),
                    ),
                    (
                        "state_seasonal_phase_weights",
                        rng.normal(0.0, 0.45, size=entity_state.shape[1]).tolist(),
                    ),
                    (
                        "state_active_interval_weights",
                        rng.normal(0.0, 0.35, size=entity_state.shape[1]).tolist(),
                    ),
                    (
                        "state_renewal_scale_weights",
                        rng.normal(0.0, 0.35, size=entity_state.shape[1]).tolist(),
                    ),
                ),
            )
        )
        parent_columns = tuple(
            column.column_id
            for column in schema.table(entity_id).columns
            if column.kind is ColumnKind.FEATURE
        )
        for column in schema.table(event_id).columns:
            if column.kind is ColumnKind.FEATURE:
                column_mechanisms.append(
                    ColumnMechanismPlan(
                        column_id=column.column_id,
                        family=str(rng.choice(_ATTRIBUTE_FAMILIES)),
                        parent_column_ids=parent_columns,
                        shared_state_ids=(state_id,),
                        parameters=(
                            ("time_weight", float(rng.uniform(0.2, 1.0))),
                            ("history_weight", float(rng.uniform(0.2, 1.0))),
                            ("mechanism_seed", int(rng.integers(0, 2**63 - 1))),
                            ("noise_scale", float(rng.uniform(0.12, 0.35))),
                        ),
                    )
                )
    return replace(
        plan,
        tables=tuple(table_plans[table_id] for table_id in plan.generation_order),
        prior_plan_id=prior_plan.plan_id,
        prior_composition_id=prior_plan.composition.plan_id,
        prior_family=prior_plan.family.value,
        motif_bundles=prior_plan.motif_bundles,
        shared_state_ids=tuple(item.state_id for item in prior_plan.shared_states),
        shared_states=prior_plan.shared_states,
        temporal_state_plans=prior_plan.temporal_states,
        population_mechanisms=tuple(population_mechanisms),
        temporal_processes=tuple(temporal_processes),
        column_mechanisms=tuple(column_mechanisms),
    )


__all__ = ["bind_temporal_event_plan"]
