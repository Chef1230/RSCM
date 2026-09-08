"""Database-level prior planning from blueprint motifs and semantic schema."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from rdb_prior.compilation.model import ColumnKind, PhysicalSchema
from rdb_prior.generation.trees.model import ForestPlan
from rdb_prior.generation.trees.sampler import sample_forest
from rdb_prior.priors.compatibility import entity_event_candidates
from rdb_prior.priors.model import (
    AttributePriorKind,
    DatabasePriorPlan,
    DurationMechanismPlan,
    MotifMechanismBundle,
    NuisancePlan,
    NuisancePriorKind,
    PriorCompositionPlan,
    PriorFamily,
    ProcessPriorKind,
    RelationMechanismBinding,
    RelationPriorKind,
    SharedStatePlan,
    StateSpacePlan,
    StateVisibility,
    TableMechanismBinding,
    TaskPolicyPlan,
    TemporalPriorKind,
    TemporalStatePlan,
    TransitionClock,
    TransitionMechanismPlan,
)
from rdb_prior.priors.registry import descriptor, is_implemented, mechanism_ref
from rdb_prior.process.sampler import RuleProcessConfig, sample_rule_plan
from rdb_prior.nuisance.planner import NuisanceOverlayConfig, NuisancePlanner
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
class PriorCompositionConfig:
    """One database-level selection for every independent prior axis."""

    attribute: AttributePriorKind
    relation: RelationPriorKind
    temporal: TemporalPriorKind
    process: ProcessPriorKind = ProcessPriorKind.NONE
    nuisance: NuisancePriorKind = NuisancePriorKind.LEGACY

    def __post_init__(self) -> None:
        for name, expected in (
            ("attribute", AttributePriorKind),
            ("relation", RelationPriorKind),
            ("temporal", TemporalPriorKind),
            ("process", ProcessPriorKind),
            ("nuisance", NuisancePriorKind),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if (
            self.relation is RelationPriorKind.STATE_CONDITIONED_EVENT
            and self.temporal is TemporalPriorKind.STATIC
        ):
            raise ValueError(
                "state_conditioned_event relation requires a temporal prior"
            )
        if (
            self.process
            in {ProcessPriorKind.STATE_MACHINE, ProcessPriorKind.WORKFLOW}
            and self.temporal is TemporalPriorKind.STATIC
        ):
            raise ValueError(
                "state-machine and workflow processes require a temporal prior"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class RelationSCMConfig:
    """Controls direct sampling of column, relation, and population SCMs."""

    column_dag_depth: tuple[int, int] = (1, 4)
    parent_count: tuple[int, int] = (1, 5)
    mechanism_weights: tuple[tuple[str, float], ...] = (
        ("linear", 0.35),
        ("cam", 0.30),
        ("mlp", 0.20),
        ("exogenous", 0.15),
    )
    relation_mechanism_weights: tuple[tuple[str, float], ...] = (
        ("scm_logistic_propensity", 0.40),
        ("scm_softmax_affinity", 0.30),
        ("scm_cpt", 0.15),
        ("scm_community", 0.15),
    )
    population_mechanism_weights: tuple[tuple[str, float], ...] = (
        ("poisson", 0.35),
        ("negative_binomial", 0.45),
        ("zero_inflated_negative_binomial", 0.20),
    )

    def __post_init__(self) -> None:
        for name, bounds in (
            ("column_dag_depth", self.column_dag_depth),
            ("parent_count", self.parent_count),
        ):
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in bounds
                )
                or bounds[0] < 1
                or bounds[1] < bounds[0]
            ):
                raise ValueError(f"{name} must be an increasing positive integer pair")
        self._validate_weights(self.mechanism_weights, "mechanism_weights", {"exogenous", "linear", "cam", "mlp"})
        self._validate_weights(
            self.relation_mechanism_weights,
            "relation_mechanism_weights",
            {"scm_logistic_propensity", "scm_softmax_affinity", "scm_cpt", "scm_community"},
        )
        self._validate_weights(
            self.population_mechanism_weights,
            "population_mechanism_weights",
            {"poisson", "negative_binomial", "zero_inflated_negative_binomial"},
        )

    @staticmethod
    def _validate_weights(
        values: tuple[tuple[str, float], ...],
        name: str,
        allowed: set[str],
    ) -> None:
        if not isinstance(values, tuple) or not values:
            raise ValueError(f"{name} must be non-empty")
        seen: set[str] = set()
        for item in values:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0] in allowed
                or item[0] in seen
                or isinstance(item[1], bool)
                or not isinstance(item[1], (int, float))
                or item[1] < 0
            ):
                raise ValueError(f"{name} contains an invalid mechanism")
            seen.add(item[0])
        if not any(item[1] > 0 for item in values):
            raise ValueError(f"{name} must contain a positive weight")


@dataclass(frozen=True, slots=True, kw_only=True)
class RelationTreeConfig:
    """Bounded direct-prior controls for random relation/attribute forests."""

    tree_count_min: int = 1
    tree_count_max: int = 8
    depth_min: int = 2
    depth_max: int = 6
    threshold_strategy: str = "extra"
    use_for_attributes: bool = True
    use_for_event_intensity: bool = True
    use_for_relation_propensity: bool = True

    def __post_init__(self) -> None:
        for name in ("tree_count_min", "tree_count_max", "depth_min", "depth_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.tree_count_max < self.tree_count_min:
            raise ValueError("tree_count_max must be at least tree_count_min")
        if self.depth_max < self.depth_min:
            raise ValueError("depth_max must be at least depth_min")
        if self.threshold_strategy not in {"random", "extra"}:
            raise ValueError("threshold_strategy must be 'random' or 'extra'")
        for name in (
            "use_for_attributes",
            "use_for_event_intensity",
            "use_for_relation_propensity",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")


@dataclass(frozen=True, slots=True, kw_only=True)
class PriorPlannerConfig:
    # No ``prior`` config must be byte-for-byte legacy in its seed path.
    database_family_weights: tuple[tuple[PriorFamily, float], ...] = ((PriorFamily.LEGACY_ROLE_SCM, 1.0),)
    task_policy: TaskPolicyPlan = TaskPolicyPlan()
    state_dimension: int = 4
    shared_state_family: str = "gaussian_mixture"
    temporal_state: TemporalStateConfig = TemporalStateConfig()
    composition: PriorCompositionConfig | None = None
    relational_scm: RelationSCMConfig = RelationSCMConfig()
    relational_tree: RelationTreeConfig = RelationTreeConfig()
    rule_process: RuleProcessConfig = RuleProcessConfig()
    nuisance: NuisanceOverlayConfig = NuisanceOverlayConfig()

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
        if not isinstance(self.relational_scm, RelationSCMConfig):
            raise TypeError("relational_scm must be RelationSCMConfig")
        if not isinstance(self.relational_tree, RelationTreeConfig):
            raise TypeError("relational_tree must be RelationTreeConfig")
        if not isinstance(self.rule_process, RuleProcessConfig):
            raise TypeError("rule_process must be RuleProcessConfig")
        if not isinstance(self.nuisance, NuisanceOverlayConfig):
            raise TypeError("nuisance must be NuisanceOverlayConfig")
        if (
            self.composition is not None
            and not isinstance(self.composition, PriorCompositionConfig)
        ):
            raise TypeError("composition must be PriorCompositionConfig or None")
        if (
            self.temporal_state.enabled
            and self.composition is not None
            and self.composition.temporal is TemporalPriorKind.STATIC
        ):
            raise ValueError("temporal state requires a non-static composition")
        if self.temporal_state.enabled and self.composition is None and not any(
            family in {PriorFamily.TEMPORAL_EVENT, PriorFamily.RULE_PROCESS} and weight > 0
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
        candidates = entity_event_candidates(
            blueprint,
            physical_schema,
            semantic_schema,
        )
        if self.config.composition is None:
            family = self._family(runtime, bool(candidates))
            components = self._default_composition(family, runtime)
        else:
            components = self.config.composition
            if components.temporal is TemporalPriorKind.STATIC:
                family = (
                    PriorFamily.RELATIONAL_TREE
                    if (
                        components.attribute is AttributePriorKind.TREE
                        or components.relation is RelationPriorKind.TREE
                    )
                    else (
                        PriorFamily.RELATIONAL_SCM
                        if (
                            components.attribute is AttributePriorKind.COLUMN_SCM
                            or components.relation is RelationPriorKind.SCM
                        )
                        else PriorFamily.LEGACY_ROLE_SCM
                    )
                )
            elif candidates:
                family = (
                    PriorFamily.RULE_PROCESS
                    if components.process is ProcessPriorKind.RULE
                    else PriorFamily.TEMPORAL_EVENT
                )
            else:
                raise ValueError(
                    "a non-static compositional temporal prior requires an entity-event motif"
                )
        bundles: list[MotifMechanismBundle] = []
        states: list[SharedStatePlan] = []
        temporal_states: list[TemporalStatePlan] = []
        temporal_by_occurrence = {
            item.motif_occurrence_id: item for item in candidates
        }
        for occurrence in blueprint.motif_occurrences:
            candidate = temporal_by_occurrence.get(occurrence.occurrence_id)
            if family in {PriorFamily.TEMPORAL_EVENT, PriorFamily.RULE_PROCESS} and candidate is not None:
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
                        components=components,
                        semantic_schema=semantic_schema,
                        physical_schema=physical_schema,
                        runtime=runtime,
                        bundle_family=family,
                    )
                )
                continue
            if family is PriorFamily.RELATIONAL_SCM:
                bundles.append(
                    self._relational_scm_bundle(
                        occurrence=occurrence,
                        physical_schema=physical_schema,
                        semantic_schema=semantic_schema,
                        runtime=runtime,
                    )
                )
                continue
            if family is PriorFamily.RELATIONAL_TREE and candidate is not None:
                bundles.append(
                    self._tree_bundle(
                        occurrence=occurrence,
                        entity_table_id=candidate.entity_table_id,
                        event_table_id=candidate.event_table_id,
                        foreign_key_id=candidate.foreign_key_id,
                        components=components,
                        physical_schema=physical_schema,
                        semantic_schema=semantic_schema,
                        runtime=runtime,
                    )
                )
                continue
            bundles.append(self._legacy_bundle(occurrence, semantic_schema))
        plan_id = f"prior_plan_{physical_schema.schema_id}"
        seed = runtime.seed("prior", "global")
        nuisance_plan = NuisancePlanner(self.config.nuisance).plan(
            schema=physical_schema,
            semantic_schema=semantic_schema,
            runtime=runtime.child("nuisance"),
            mechanism=components.nuisance,
        )
        composition = PriorCompositionPlan(
            plan_id=plan_id,
            semantic_schema=semantic_schema,
            shared_states=tuple(states),
            temporal_states=tuple(temporal_states),
            motif_bundles=tuple(bundles),
            nuisance_plan=nuisance_plan,
            task_policy=self.config.task_policy,
            seed=seed,
        )
        return DatabasePriorPlan(
            plan_id=plan_id,
            family=family,
            family_version=descriptor(family).version,
            semantic_schema=semantic_schema,
            shared_states=tuple(states),
            motif_bundles=tuple(bundles),
            task_policy=self.config.task_policy,
            seed=seed,
            temporal_states=tuple(temporal_states),
            composition=composition,
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
                family not in {PriorFamily.TEMPORAL_EVENT, PriorFamily.RULE_PROCESS}
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

    def _default_composition(
        self,
        family: PriorFamily,
        runtime: RuntimeContext,
    ) -> PriorCompositionConfig:
        if family in {PriorFamily.TEMPORAL_EVENT, PriorFamily.RULE_PROCESS}:
            temporal = runtime.python_rng(
                "prior", "temporal-component"
            ).choice(
                (
                    TemporalPriorKind.STATIONARY,
                    TemporalPriorKind.SEASONAL,
                    TemporalPriorKind.CHURN,
                    TemporalPriorKind.RENEWAL,
                )
            )
            return PriorCompositionConfig(
                attribute=AttributePriorKind.COLUMN_SCM,
                relation=RelationPriorKind.STATE_CONDITIONED_EVENT,
                temporal=temporal,
                process=(
                    ProcessPriorKind.RULE
                    if family is PriorFamily.RULE_PROCESS
                    else ProcessPriorKind.NONE
                ),
                nuisance=self.config.nuisance.kind,
            )
        if family is PriorFamily.RELATIONAL_TREE:
            return PriorCompositionConfig(
                attribute=AttributePriorKind.TREE,
                relation=RelationPriorKind.TREE,
                temporal=TemporalPriorKind.STATIC,
                nuisance=self.config.nuisance.kind,
            )
        if family is PriorFamily.RELATIONAL_SCM:
            return PriorCompositionConfig(
                attribute=AttributePriorKind.COLUMN_SCM,
                relation=RelationPriorKind.SCM,
                temporal=TemporalPriorKind.STATIC,
                nuisance=self.config.nuisance.kind,
            )
        return PriorCompositionConfig(
            attribute=AttributePriorKind.LEGACY_SCM,
            relation=RelationPriorKind.LATENT_AFFINITY,
            temporal=TemporalPriorKind.STATIC,
            nuisance=self.config.nuisance.kind,
        )

    def _relational_scm_bundle(
        self,
        *,
        occurrence: object,
        physical_schema: PhysicalSchema,
        semantic_schema: SemanticSchemaPlan,
        runtime: RuntimeContext,
    ) -> MotifMechanismBundle:
        config = self.config.relational_scm
        table_ids = tuple(table_id for _slot, table_id in occurrence.node_bindings)
        table_id_set = set(table_ids)
        column_payloads: list[dict[str, object]] = []
        for table_id in table_ids:
            table = physical_schema.table(table_id)
            prior_features = [
                column.column_id
                for column in table.columns
                if column.kind is ColumnKind.FEATURE
            ]
            parent_features: list[str] = []
            for foreign_key in physical_schema.foreign_keys:
                if (
                    foreign_key.child_table_id == table_id
                    and foreign_key.parent_table_id in table_id_set
                ):
                    parent_features.extend(
                        column.column_id
                        for column in physical_schema.table(
                            foreign_key.parent_table_id
                        ).columns
                        if column.kind is ColumnKind.FEATURE
                    )
            depth_by_column: dict[str, int] = {}
            for ordinal, column in enumerate(
                item for item in table.columns if item.kind is ColumnKind.FEATURE
            ):
                rng = runtime.numpy_rng(
                    "prior",
                    "relational-scm-column",
                    f"{occurrence.occurrence_id}:{column.column_id}",
                )
                candidates = tuple(
                    item
                    for item in dict.fromkeys(
                        prior_features[:ordinal] + parent_features
                    )
                    if depth_by_column.get(item, 0)
                    < config.column_dag_depth[1]
                )
                lower, upper = config.parent_count
                count = min(
                    len(candidates),
                    int(rng.integers(lower, upper + 1)),
                )
                selected = (
                    tuple(
                        str(item)
                        for item in rng.choice(
                            np.asarray(candidates, dtype=object),
                            size=count,
                            replace=False,
                        )
                    )
                    if count
                    else ()
                )
                families, weights = zip(*config.mechanism_weights)
                family = runtime.python_rng(
                    "prior",
                    "relational-scm-family",
                    f"{occurrence.occurrence_id}:{column.column_id}",
                ).choices(families, weights=weights, k=1)[0]
                depth_by_column[column.column_id] = (
                    0
                    if not selected
                    else min(
                        config.column_dag_depth[1],
                        1 + max(
                            (
                                depth_by_column.get(parent_id, 0)
                                for parent_id in selected
                            ),
                            default=0,
                        ),
                    )
                )
                column_payloads.append(
                    {
                        "column_id": column.column_id,
                        "parent_column_ids": list(selected),
                        "parent_state_ids": [],
                        "family": family,
                        "parameters": {
                            "mechanism_seed": int(rng.integers(0, 2**63 - 1)),
                            "noise_scale": float(rng.uniform(0.08, 0.30)),
                            "signal_scale": float(rng.uniform(0.6, 1.4)),
                            "activation_scale": float(rng.uniform(0.7, 1.5)),
                            "output_scale": float(rng.uniform(0.8, 1.2)),
                            "mlp_depth": int(rng.integers(1, 3)),
                            "mlp_hidden_factor": float(rng.uniform(1.5, 3.0)),
                        },
                    }
                )
        relation_payloads: dict[str, object] = {}
        relation_families, relation_weights = zip(
            *config.relation_mechanism_weights
        )
        population_payloads: dict[str, object] = {}
        population_families, population_weights = zip(
            *config.population_mechanism_weights
        )
        edges: list[RelationMechanismBinding] = []
        for foreign_key in physical_schema.foreign_keys:
            if foreign_key.child_table_id not in table_id_set:
                continue
            rng = runtime.numpy_rng(
                "prior",
                "relational-scm-relation",
                f"{occurrence.occurrence_id}:{foreign_key.foreign_key_id}",
            )
            family = runtime.python_rng(
                "prior",
                "relational-scm-relation-family",
                f"{occurrence.occurrence_id}:{foreign_key.foreign_key_id}",
            ).choices(relation_families, weights=relation_weights, k=1)[0]
            relation_payloads[foreign_key.foreign_key_id] = {
                "foreign_key_id": foreign_key.foreign_key_id,
                "family": family,
            }
            edges.append(
                RelationMechanismBinding(
                    foreign_key_id=foreign_key.foreign_key_id,
                    mechanism_id=family,
                )
            )
        for table_id in table_ids:
            table = physical_schema.table(table_id)
            if table.role.value != "event":
                continue
            parent = next(
                (
                    item
                    for item in physical_schema.foreign_keys
                    if item.child_table_id == table_id
                    and item.relation_strategy != "lookup_assignment"
                    and item.parent_table_id in table_id_set
                ),
                None,
            )
            if parent is None:
                continue
            rng = runtime.numpy_rng(
                "prior",
                "relational-scm-population",
                f"{occurrence.occurrence_id}:{table_id}",
            )
            family = runtime.python_rng(
                "prior",
                "relational-scm-population-family",
                f"{occurrence.occurrence_id}:{table_id}",
            ).choices(population_families, weights=population_weights, k=1)[0]
            population_payloads[table_id] = {
                "table_id": table_id,
                "parent_table_id": parent.parent_table_id,
                "family": family,
                "baseline": float(rng.uniform(0.5, 2.0)),
                "dispersion": float(rng.uniform(1.5, 5.0)),
                "zero_probability": float(rng.uniform(0.15, 0.40)),
            }
        return MotifMechanismBundle(
            bundle_id=f"bundle_{occurrence.occurrence_id}",
            motif_occurrence_id=occurrence.occurrence_id,
            family=PriorFamily.RELATIONAL_SCM,
            node_bindings=tuple(
                TableMechanismBinding(
                    table_id=table_id,
                    mechanism_ids=("column_scm_dag",),
                )
                for table_id in table_ids
            ),
            edge_bindings=tuple(edges),
            population_mechanism="population_scm",
            attribute_mechanism=mechanism_ref(AttributePriorKind.COLUMN_SCM),
            relation_mechanism=mechanism_ref(RelationPriorKind.SCM),
            temporal_mechanism=mechanism_ref(TemporalPriorKind.STATIC),
            process_mechanism=mechanism_ref(ProcessPriorKind.NONE),
            compatible_task_families=(),
            parameters=(
                ("column_mechanisms", column_payloads),
                ("relation_mechanisms", relation_payloads),
                ("population_mechanisms", population_payloads),
                (
                    "semantic_table_roles",
                    [
                        [table_id, semantic_schema.table_role(table_id).value]
                        for table_id in table_ids
                    ],
                ),
            ),
        )

    def _forest(
        self,
        *,
        forest_id: str,
        feature_refs: tuple[str, ...],
        runtime: RuntimeContext,
    ) -> ForestPlan:
        config = self.config.relational_tree
        return sample_forest(
            forest_id=forest_id,
            feature_refs=feature_refs,
            rng=runtime.numpy_rng("prior", "relational-tree", forest_id),
            tree_count=(config.tree_count_min, config.tree_count_max),
            depth=(config.depth_min, config.depth_max),
            threshold_strategy=config.threshold_strategy,
        )

    def _tree_bundle(
        self,
        *,
        occurrence: object,
        entity_table_id: str,
        event_table_id: str,
        foreign_key_id: str,
        components: PriorCompositionConfig,
        physical_schema: PhysicalSchema,
        semantic_schema: SemanticSchemaPlan,
        runtime: RuntimeContext,
    ) -> MotifMechanismBundle:
        config = self.config.relational_tree
        relation_forests: dict[str, object] = {}
        attribute_forests: dict[str, object] = {}
        if config.use_for_relation_propensity:
            relation_forests[foreign_key_id] = self._forest(
                forest_id=f"forest_relation_{foreign_key_id}",
                feature_refs=(
                    "child_latent_0",
                    "parent_latent_0",
                    "parent_activity",
                ),
                runtime=runtime,
            ).to_dict()
        if config.use_for_attributes:
            for column in physical_schema.table(event_table_id).columns:
                if column.kind is ColumnKind.FEATURE:
                    attribute_forests[column.column_id] = self._forest(
                        forest_id=f"forest_attribute_{column.column_id}",
                        feature_refs=(
                            "self_latent_0",
                            "parent_latent_0",
                            "parent_feature_0",
                            "time",
                            "history",
                        ),
                        runtime=runtime,
                    ).to_dict()
        entity_role = semantic_schema.table_role(entity_table_id).value
        event_role = semantic_schema.table_role(event_table_id).value
        attribute_kind = (
            AttributePriorKind.TREE
            if config.use_for_attributes
            else AttributePriorKind.LEGACY_SCM
        )
        relation_kind = (
            RelationPriorKind.TREE
            if config.use_for_relation_propensity
            else RelationPriorKind.LATENT_AFFINITY
        )
        return MotifMechanismBundle(
            bundle_id=f"bundle_{occurrence.occurrence_id}",
            motif_occurrence_id=occurrence.occurrence_id,
            family=PriorFamily.RELATIONAL_TREE,
            node_bindings=(
                TableMechanismBinding(
                    table_id=entity_table_id,
                    mechanism_ids=("tree_parent_context",),
                ),
                TableMechanismBinding(
                    table_id=event_table_id,
                    mechanism_ids=("tree_relation", "tree_attributes"),
                ),
            ),
            edge_bindings=(
                RelationMechanismBinding(
                    foreign_key_id=foreign_key_id,
                    mechanism_id=(
                        "tree_propensity"
                        if config.use_for_relation_propensity
                        else "latent_affinity"
                    ),
                ),
            ),
            population_mechanism="legacy",
            attribute_mechanism=mechanism_ref(attribute_kind),
            relation_mechanism=mechanism_ref(relation_kind),
            temporal_mechanism=mechanism_ref(TemporalPriorKind.STATIC),
            process_mechanism=mechanism_ref(components.process),
            compatible_task_families=(),
            parameters=(
                ("entity_table_id", entity_table_id),
                ("event_table_id", event_table_id),
                ("semantic_entity_role", entity_role),
                ("semantic_event_role", event_role),
                ("relation_forests", relation_forests),
                ("attribute_forests", attribute_forests),
            ),
        )

    def _legacy_bundle(
        self,
        occurrence: object,
        semantic_schema: SemanticSchemaPlan | None = None,
    ) -> MotifMechanismBundle:
        node_bindings = tuple(
            TableMechanismBinding(
                table_id=table_id,
                mechanism_ids=("legacy_role_scm",),
            )
            for _slot, table_id in occurrence.node_bindings
        )
        semantic_roles = (
            [
                [slot, semantic_schema.table_role(table_id).value]
                for slot, table_id in occurrence.node_bindings
            ]
            if semantic_schema is not None
            else []
        )
        return MotifMechanismBundle(
            bundle_id=f"bundle_{occurrence.occurrence_id}",
            motif_occurrence_id=occurrence.occurrence_id,
            family=PriorFamily.LEGACY_ROLE_SCM,
            node_bindings=node_bindings,
            edge_bindings=(),
            population_mechanism="legacy",
            attribute_mechanism=mechanism_ref(AttributePriorKind.LEGACY_SCM),
            relation_mechanism=mechanism_ref(RelationPriorKind.LATENT_AFFINITY),
            temporal_mechanism=mechanism_ref(TemporalPriorKind.STATIC),
            process_mechanism=mechanism_ref(ProcessPriorKind.NONE),
            compatible_task_families=(),
            parameters=(("semantic_table_roles", semantic_roles),),
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
        components: PriorCompositionConfig,
        semantic_schema: SemanticSchemaPlan,
        physical_schema: PhysicalSchema,
        runtime: RuntimeContext,
        bundle_family: PriorFamily = PriorFamily.TEMPORAL_EVENT,
    ) -> MotifMechanismBundle:
        entity_role = semantic_schema.table_role(entity_table_id)
        event_role = semantic_schema.table_role(event_table_id)
        event_column_roles = tuple(
            item.role.value
            for item in semantic_schema.columns
            if item.column_id.startswith(f"{event_table_id}_")
        )
        semantic_weights = [
            [key, value]
            for key, value in self._semantic_mechanism_weights(
                entity_role,
                event_role,
                event_column_roles,
            )
        ]
        parameters: tuple[tuple[str, object], ...] = (
            ("entity_table_id", entity_table_id),
            ("event_table_id", event_table_id),
            ("state_id", shared_state_id),
            ("semantic_entity_role", entity_role.value),
            ("semantic_event_role", event_role.value),
            ("semantic_event_column_roles", list(event_column_roles)),
            ("semantic_mechanism_weights", semantic_weights),
        )
        tree_attribute_forests: dict[str, object] = {}
        tree_relation_forests: dict[str, object] = {}
        if components.attribute is AttributePriorKind.TREE:
            for column in physical_schema.table(event_table_id).columns:
                if column.kind is ColumnKind.FEATURE:
                    tree_attribute_forests[column.column_id] = self._forest(
                        forest_id=f"forest_temporal_attribute_{column.column_id}",
                        feature_refs=(
                            "state_0",
                            "parent_feature_0",
                            "time",
                            "history",
                        ),
                        runtime=runtime,
                    ).to_dict()
        if components.relation is RelationPriorKind.TREE:
            tree_relation_forests[foreign_key_id] = self._forest(
                forest_id=f"forest_temporal_relation_{foreign_key_id}",
                feature_refs=(
                    "child_latent_0",
                    "parent_latent_0",
                    "parent_activity",
                ),
                runtime=runtime,
            ).to_dict()
        if (
            components.attribute is AttributePriorKind.TREE
            and self.config.relational_tree.use_for_event_intensity
        ):
            parameters += (
                (
                    "event_intensity_forest",
                    self._forest(
                        forest_id=f"forest_temporal_intensity_{event_table_id}",
                        feature_refs=("state_0", "entity_feature_0"),
                        runtime=runtime,
                    ).to_dict(),
                ),
            )
        if tree_attribute_forests:
            parameters += (("attribute_forests", tree_attribute_forests),)
        if tree_relation_forests:
            parameters += (("relation_forests", tree_relation_forests),)
        entity_mechanisms = (shared_state_id, "entity_state")
        event_mechanisms = ("event_count", "event_time", "event_attributes")
        temporal_parameters: tuple[tuple[str, object], ...] = (
            ("selection", components.temporal.value),
        )
        process = components.process
        rule_plan = None
        if process is ProcessPriorKind.RULE:
            rule_plan = sample_rule_plan(
                schema=physical_schema,
                entity_table_id=entity_table_id,
                event_table_id=event_table_id,
                foreign_key_id=foreign_key_id,
                runtime=runtime,
                state_id=shared_state_id,
                config=self.config.rule_process,
            )
            parameters += (("rule_plan", rule_plan.to_dict()),)
            entity_mechanisms += ("rule_process",)
            event_mechanisms += ("rule_process",)
        if temporal_state_id is not None:
            parameters += (("temporal_state_id", temporal_state_id),)
            entity_mechanisms += (temporal_state_id,)
            event_mechanisms += ("temporal_state_process",)
            temporal_parameters = (("selection", "state_trajectory"),)
            if process is ProcessPriorKind.NONE:
                process = ProcessPriorKind.STATE_MACHINE
        return MotifMechanismBundle(
            bundle_id=f"bundle_{occurrence_id}",
            motif_occurrence_id=occurrence_id,
            family=bundle_family,
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
                    mechanism_id=(
                        "tree_propensity"
                        if components.relation is RelationPriorKind.TREE
                        else "state_conditioned_event_fk"
                    ),
                ),
            ),
            population_mechanism="negative_binomial",
            attribute_mechanism=mechanism_ref(components.attribute),
            relation_mechanism=mechanism_ref(components.relation),
            temporal_mechanism=mechanism_ref(
                components.temporal,
                parameters=temporal_parameters,
            ),
            process_mechanism=mechanism_ref(
                process,
                parameters=(
                    (("rule_id", rule_plan.rule_id), ("rule_plan", rule_plan.to_dict()))
                    if rule_plan is not None
                    else ()
                ),
            ),
            shared_state_ids=(shared_state_id,),
            compatible_task_families=(
                "future_event_existence",
                "entity_future_event_existence",
                "history_gated_future_active",
                "history_gated_future_inactive",
                "future_event_attribute",
                "temporal_aggregate",
                "interaction_response",
                "multi_hop_program",
            ),
            parameters=parameters,
        )

    @staticmethod
    def _semantic_mechanism_weights(
        entity_role: object,
        event_role: object,
        event_column_roles: tuple[str, ...] = (),
    ) -> tuple[tuple[str, float], ...]:
        entity = getattr(entity_role, "value", str(entity_role))
        event = getattr(event_role, "value", str(event_role))
        if entity == "actor" and event == "transaction":
            weights = {
                "attribute": 0.35,
                "intensity": 0.40,
                "temporal": 0.25,
            }
        elif entity == "object" and event == "observation":
            weights = {
                "attribute": 0.25,
                "intensity": 0.20,
                "temporal": 0.55,
            }
        elif event == "state_change":
            weights = {
                "attribute": 0.20,
                "intensity": 0.25,
                "temporal": 0.55,
            }
        else:
            weights = {
                "attribute": 0.34,
                "intensity": 0.33,
                "temporal": 0.33,
            }
        if "amount" in event_column_roles:
            weights["intensity"] += 0.08
        if "measurement" in event_column_roles:
            weights["temporal"] += 0.08
        if "outcome" in event_column_roles:
            weights["attribute"] += 0.04
        total = sum(weights.values())
        return tuple(
            (key, float(value / total))
            for key, value in sorted(weights.items())
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



__all__ = ["RelationSCMConfig", "RelationTreeConfig", "TemporalStateConfig", "PriorPlannerConfig", "PriorPlanner"]
