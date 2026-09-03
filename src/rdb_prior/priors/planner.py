"""Database-level prior planning from blueprint motifs and semantic schema."""

from __future__ import annotations

from dataclasses import dataclass

from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.priors.compatibility import entity_event_candidates
from rdb_prior.priors.model import (
    DatabasePriorPlan,
    MotifMechanismBundle,
    PriorFamily,
    RelationMechanismBinding,
    SharedStatePlan,
    TableMechanismBinding,
    TaskPolicyPlan,
)
from rdb_prior.priors.registry import descriptor, is_implemented
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.blueprint import SchemaBlueprint
from rdb_prior.schema.semantics import SemanticSchemaPlan


@dataclass(frozen=True, slots=True, kw_only=True)
class PriorPlannerConfig:
    # No ``prior`` config must be byte-for-byte legacy in its seed path.
    database_family_weights: tuple[tuple[PriorFamily, float], ...] = ((PriorFamily.LEGACY_ROLE_SCM, 1.0),)
    task_policy: TaskPolicyPlan = TaskPolicyPlan()
    state_dimension: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.database_family_weights, tuple) or not self.database_family_weights:
            raise ValueError("database_family_weights must be non-empty")
        seen: set[PriorFamily] = set()
        for family, weight in self.database_family_weights:
            if not isinstance(family, PriorFamily) or family in seen:
                raise ValueError("database_family_weights must have unique PriorFamily keys")
            if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0:
                raise ValueError("prior family weights must be non-negative")
            if weight > 0 and not is_implemented(family):
                raise ValueError(f"prior family {family.value} is reserved but not implemented")
            seen.add(family)
        if not any(weight > 0 for _family, weight in self.database_family_weights):
            raise ValueError("at least one prior family weight must be positive")
        if not isinstance(self.task_policy, TaskPolicyPlan):
            raise TypeError("task_policy must be TaskPolicyPlan")
        if isinstance(self.state_dimension, bool) or not isinstance(self.state_dimension, int) or self.state_dimension < 1:
            raise ValueError("state_dimension must be positive")


class PriorPlanner:
    def __init__(self, config: PriorPlannerConfig | None = None) -> None:
        self.config = config or PriorPlannerConfig()

    def plan(self, *, blueprint: SchemaBlueprint, physical_schema: PhysicalSchema, semantic_schema: SemanticSchemaPlan, runtime: RuntimeContext) -> DatabasePriorPlan:
        if semantic_schema.schema_id != physical_schema.schema_id:
            raise ValueError("semantic schema does not belong to physical schema")
        candidates = entity_event_candidates(blueprint, physical_schema)
        family = self._family(runtime, bool(candidates))
        bundles: list[MotifMechanismBundle] = []
        states: list[SharedStatePlan] = []
        temporal_by_occurrence = {item.motif_occurrence_id: item for item in candidates}
        for occurrence in blueprint.motif_occurrences:
            candidate = temporal_by_occurrence.get(occurrence.occurrence_id)
            if family is PriorFamily.TEMPORAL_EVENT and candidate is not None:
                state_id = f"state_{candidate.entity_table_id}_{occurrence.occurrence_id}"
                states.append(SharedStatePlan(state_id=state_id, owner_table_id=candidate.entity_table_id, family="gaussian_latent", dimension=self.config.state_dimension, seed=runtime.seed("prior", "state", state_id)))
                bundles.append(MotifMechanismBundle(bundle_id=f"bundle_{occurrence.occurrence_id}", motif_occurrence_id=occurrence.occurrence_id, family=PriorFamily.TEMPORAL_EVENT, node_bindings=(TableMechanismBinding(table_id=candidate.entity_table_id, mechanism_ids=(state_id, "entity_state")), TableMechanismBinding(table_id=candidate.event_table_id, mechanism_ids=("event_count", "event_time", "event_attributes"))), edge_bindings=(RelationMechanismBinding(foreign_key_id=candidate.foreign_key_id, mechanism_id="state_conditioned_event_fk"),), population_mechanism="negative_binomial", temporal_mechanism="sampled_stationary_seasonal_churn", attribute_mechanism="sampled_linear_cam", compatible_task_families=("entity_future_event_existence",), parameters=(("entity_table_id", candidate.entity_table_id), ("event_table_id", candidate.event_table_id), ("state_id", state_id))))
            else:
                node_bindings = tuple(TableMechanismBinding(table_id=table_id, mechanism_ids=("legacy_role_scm",)) for _slot, table_id in occurrence.node_bindings)
                bundles.append(MotifMechanismBundle(bundle_id=f"bundle_{occurrence.occurrence_id}", motif_occurrence_id=occurrence.occurrence_id, family=PriorFamily.LEGACY_ROLE_SCM, node_bindings=node_bindings, edge_bindings=(), population_mechanism="legacy", temporal_mechanism="legacy", attribute_mechanism="legacy", compatible_task_families=()))
        return DatabasePriorPlan(plan_id=f"prior_plan_{physical_schema.schema_id}", family=family, family_version=descriptor(family).version, semantic_schema=semantic_schema, shared_states=tuple(states), motif_bundles=tuple(bundles), task_policy=self.config.task_policy, seed=runtime.seed("prior", "global"))

    def _family(self, runtime: RuntimeContext, temporal_available: bool) -> PriorFamily:
        values = [(family, weight) for family, weight in self.config.database_family_weights if weight > 0 and (family is not PriorFamily.TEMPORAL_EVENT or temporal_available)]
        if not values:
            raise ValueError("no configured prior family is compatible with this schema")
        families, weights = zip(*values)
        return runtime.python_rng("prior", "database-family").choices(families, weights=weights, k=1)[0]


__all__ = ["PriorPlannerConfig", "PriorPlanner"]
