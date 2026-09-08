"""Database-level prior planning from blueprint motifs and semantic schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.priors.compatibility import entity_event_candidates
from rdb_prior.priors.model import (
    DatabasePriorPlan,
    DurationMechanismPlan,
    MotifMechanismBundle,
    PriorFamily,
    RelationMechanismBinding,
    SharedStatePlan,
    StateSpacePlan,
    StateVisibility,
    TableMechanismBinding,
    TaskPolicyPlan,
    TemporalStatePlan,
    TransitionClock,
    TransitionMechanismPlan,
)
from rdb_prior.priors.registry import descriptor, is_implemented
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.blueprint import SchemaBlueprint
from rdb_prior.schema.semantics import SemanticSchemaPlan


@dataclass(frozen=True, slots=True, kw_only=True)
class TemporalStateConfig:
    enabled: bool = False
    state_space: StateSpacePlan | None = None
    initial_family: str = "state_conditioned_softmax"
    transition_clock: TransitionClock = TransitionClock.EVENT_DRIVEN
    transition_family: str = "transition_matrix"
    duration_family: str = "exponential"
    duration_state_conditioned: bool = True
    visibility: StateVisibility = StateVisibility.HIDDEN

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("temporal_state.enabled must be a bool")
        if not self.enabled:
            if self.state_space is not None:
                raise ValueError("disabled temporal state cannot define state_space")
            return
        if not isinstance(self.state_space, StateSpacePlan):
            raise TypeError("enabled temporal state requires StateSpacePlan")
        if self.initial_family != "state_conditioned_softmax":
            raise ValueError("unsupported initial-state family")
        if not isinstance(self.transition_clock, TransitionClock):
            raise TypeError("transition_clock must be TransitionClock")
        if self.transition_family not in {
            "transition_matrix",
            "state_conditioned_softmax",
        }:
            raise ValueError("unsupported transition family")
        if self.duration_family != "exponential":
            raise ValueError("unsupported duration family")
        if not isinstance(self.duration_state_conditioned, bool):
            raise TypeError("duration_state_conditioned must be a bool")
        if not isinstance(self.visibility, StateVisibility):
            raise TypeError("visibility must be StateVisibility")



@dataclass(frozen=True, slots=True, kw_only=True)
class PriorPlannerConfig:
    # No ``prior`` config must be byte-for-byte legacy in its seed path.
    database_family_weights: tuple[tuple[PriorFamily, float], ...] = ((PriorFamily.LEGACY_ROLE_SCM, 1.0),)
    task_policy: TaskPolicyPlan = TaskPolicyPlan()
    state_dimension: int = 4
    shared_state_family: str = "gaussian_mixture"
    temporal_state: TemporalStateConfig = TemporalStateConfig()

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
        if self.shared_state_family != "gaussian_mixture":
            raise ValueError("unsupported shared-state family")
        if not isinstance(self.temporal_state, TemporalStateConfig):
            raise TypeError("temporal_state must be TemporalStateConfig")
        if self.temporal_state.enabled and not any(
            family is PriorFamily.TEMPORAL_EVENT and weight > 0
            for family, weight in self.database_family_weights
        ):
            raise ValueError(
                "enabled temporal state requires temporal_event prior weight"
            )



class PriorPlanner:
    def __init__(self, config: PriorPlannerConfig | None = None) -> None:
        self.config = config or PriorPlannerConfig()

    def plan(
        self,
        *,
        blueprint: SchemaBlueprint,
        physical_schema: PhysicalSchema,
        semantic_schema: SemanticSchemaPlan,
        runtime: RuntimeContext,
    ) -> DatabasePriorPlan:
        if semantic_schema.schema_id != physical_schema.schema_id:
            raise ValueError("semantic schema does not belong to physical schema")
        candidates = entity_event_candidates(blueprint, physical_schema)
        family = self._family(runtime, bool(candidates))
        bundles: list[MotifMechanismBundle] = []
        states: list[SharedStatePlan] = []
        temporal_states: list[TemporalStatePlan] = []
        temporal_by_occurrence = {
            item.motif_occurrence_id: item for item in candidates
        }
        for occurrence in blueprint.motif_occurrences:
            candidate = temporal_by_occurrence.get(occurrence.occurrence_id)
            if family is not PriorFamily.TEMPORAL_EVENT or candidate is None:
                bundles.append(self._legacy_bundle(occurrence))
                continue

            state_id = f"state_{candidate.entity_table_id}_{occurrence.occurrence_id}"
            states.append(
                SharedStatePlan(
                    state_id=state_id,
                    owner_table_id=candidate.entity_table_id,
                    family=self.config.shared_state_family,
                    dimension=self.config.state_dimension,
                    seed=runtime.seed("prior", "state", state_id),
                )
            )
            temporal_state_id: str | None = None
            if self.config.temporal_state.enabled:
                temporal_state_id = (
                    f"temporal_{candidate.entity_table_id}_{occurrence.occurrence_id}"
                )
                temporal_states.append(
                    self._temporal_state_plan(
                        state_id=temporal_state_id,
                        owner_table_id=candidate.entity_table_id,
                        shared_state_id=state_id,
                        runtime=runtime,
                    )
                )
            bundles.append(
                self._temporal_bundle(
                    occurrence_id=occurrence.occurrence_id,
                    entity_table_id=candidate.entity_table_id,
                    event_table_id=candidate.event_table_id,
                    foreign_key_id=candidate.foreign_key_id,
                    shared_state_id=state_id,
                    temporal_state_id=temporal_state_id,
                )
            )
        return DatabasePriorPlan(
            plan_id=f"prior_plan_{physical_schema.schema_id}",
            family=family,
            family_version=descriptor(family).version,
            semantic_schema=semantic_schema,
            shared_states=tuple(states),
            motif_bundles=tuple(bundles),
            task_policy=self.config.task_policy,
            seed=runtime.seed("prior", "global"),
            temporal_states=tuple(temporal_states),
        )

    def _family(
        self,
        runtime: RuntimeContext,
        temporal_available: bool,
    ) -> PriorFamily:
        values = [
            (family, weight)
            for family, weight in self.config.database_family_weights
            if weight > 0
            and (
                family is not PriorFamily.TEMPORAL_EVENT
                or temporal_available
            )
        ]
        if not values:
            raise ValueError("no configured prior family is compatible with this schema")
        families, weights = zip(*values)
        return runtime.python_rng("prior", "database-family").choices(
            families,
            weights=weights,
            k=1,
        )[0]

    def _legacy_bundle(self, occurrence: object) -> MotifMechanismBundle:
        node_bindings = tuple(
            TableMechanismBinding(
                table_id=table_id,
                mechanism_ids=("legacy_role_scm",),
            )
            for _slot, table_id in occurrence.node_bindings
        )
        return MotifMechanismBundle(
            bundle_id=f"bundle_{occurrence.occurrence_id}",
            motif_occurrence_id=occurrence.occurrence_id,
            family=PriorFamily.LEGACY_ROLE_SCM,
            node_bindings=node_bindings,
            edge_bindings=(),
            population_mechanism="legacy",
            temporal_mechanism="legacy",
            attribute_mechanism="legacy",
            compatible_task_families=(),
        )

    def _temporal_bundle(
        self,
        *,
        occurrence_id: str,
        entity_table_id: str,
        event_table_id: str,
        foreign_key_id: str,
        shared_state_id: str,
        temporal_state_id: str | None,
    ) -> MotifMechanismBundle:
        parameters: tuple[tuple[str, object], ...] = (
            ("entity_table_id", entity_table_id),
            ("event_table_id", event_table_id),
            ("state_id", shared_state_id),
        )
        entity_mechanisms = (shared_state_id, "entity_state")
        event_mechanisms = ("event_count", "event_time", "event_attributes")
        temporal_mechanism = "sampled_stationary_seasonal_churn"
        if temporal_state_id is not None:
            parameters += (("temporal_state_id", temporal_state_id),)
            entity_mechanisms += (temporal_state_id,)
            event_mechanisms += ("temporal_state_process",)
            temporal_mechanism = "state_trajectory"
        return MotifMechanismBundle(
            bundle_id=f"bundle_{occurrence_id}",
            motif_occurrence_id=occurrence_id,
            family=PriorFamily.TEMPORAL_EVENT,
            node_bindings=(
                TableMechanismBinding(
                    table_id=entity_table_id,
                    mechanism_ids=entity_mechanisms,
                ),
                TableMechanismBinding(
                    table_id=event_table_id,
                    mechanism_ids=event_mechanisms,
                ),
            ),
            edge_bindings=(
                RelationMechanismBinding(
                    foreign_key_id=foreign_key_id,
                    mechanism_id="state_conditioned_event_fk",
                ),
            ),
            population_mechanism="negative_binomial",
            temporal_mechanism=temporal_mechanism,
            attribute_mechanism="sampled_linear_cam",
            compatible_task_families=("entity_future_event_existence",),
            parameters=parameters,
        )

    def _temporal_state_plan(
        self,
        *,
        state_id: str,
        owner_table_id: str,
        shared_state_id: str,
        runtime: RuntimeContext,
    ) -> TemporalStatePlan:
        config = self.config.temporal_state
        assert config.state_space is not None
        values = config.state_space.values
        count = len(values)
        rng = runtime.numpy_rng("prior", "temporal-state", state_id)
        initial_parameters = (
            ("bias", rng.normal(0.0, 0.45, size=count).tolist()),
            (
                "z_weights",
                rng.normal(
                    0.0,
                    0.35,
                    size=(count, self.config.state_dimension),
                ).tolist(),
            ),
        )
        transition_parameters: tuple[tuple[str, object], ...]
        if config.transition_family == "transition_matrix":
            transition_parameters = (
                ("matrix", _sample_transition_matrix(rng, config.state_space)),
            )
        else:
            transition_parameters = (
                (
                    "bias",
                    rng.normal(0.0, 0.35, size=(count, count)).tolist(),
                ),
                (
                    "z_weights",
                    rng.normal(
                        0.0,
                        0.25,
                        size=(count, count, self.config.state_dimension),
                    ).tolist(),
                ),
            )
        duration_parameters = (
            (
                "state_rate_multipliers",
                np.exp(rng.uniform(-0.55, 0.55, size=count)).tolist(),
            ),
            (
                "z_weights",
                rng.normal(
                    0.0,
                    0.25,
                    size=self.config.state_dimension,
                ).tolist(),
            ),
            ("transition_rate_multiplier", float(rng.uniform(0.6, 1.4))),
        )
        return TemporalStatePlan(
            state_id=state_id,
            owner_table_id=owner_table_id,
            shared_state_id=shared_state_id,
            state_space=config.state_space,
            initial_family=config.initial_family,
            initial_parameters=initial_parameters,
            transition=TransitionMechanismPlan(
                clock=config.transition_clock,
                family=config.transition_family,
                seed=runtime.seed("prior", "transition", state_id),
                parameters=transition_parameters,
            ),
            duration=DurationMechanismPlan(
                family=config.duration_family,
                state_conditioned=config.duration_state_conditioned,
                seed=runtime.seed("prior", "duration", state_id),
                parameters=duration_parameters,
            ),
            visibility=config.visibility,
            seed=runtime.seed("prior", "temporal-state", state_id),
        )


def _sample_transition_matrix(
    rng: np.random.Generator,
    state_space: StateSpacePlan,
) -> list[list[float]]:
    values = state_space.values
    terminal = set(state_space.terminal_states)
    matrix: list[list[float]] = []
    for source_index, source in enumerate(values):
        if source in terminal:
            row = [0.0] * len(values)
            row[source_index] = 1.0
        else:
            row = rng.uniform(0.15, 1.0, size=len(values))
            row[source_index] *= 0.7
            for target_index, target in enumerate(values):
                if target in terminal:
                    row[target_index] *= 0.25
            row = row / row.sum()
        matrix.append([float(value) for value in row])
    return matrix



__all__ = ["TemporalStateConfig", "PriorPlannerConfig", "PriorPlanner"]
