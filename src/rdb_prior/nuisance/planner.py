"""Deterministic planner for orthogonal nuisance overlays."""
from __future__ import annotations
from dataclasses import dataclass
from rdb_prior.compilation.model import ColumnKind, PhysicalSchema
from rdb_prior.priors.model import (
    DistractorRelationPlan, MissingnessPlan, NuisanceColumnPlan,
    NuisanceColumnRole, NuisancePlan, NuisancePriorKind,
)
from rdb_prior.priors.registry import mechanism_ref
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.semantics import SemanticSchemaPlan

@dataclass(frozen=True, slots=True, kw_only=True)
class NuisanceOverlayConfig:
    enabled: bool = False
    kind: NuisancePriorKind = NuisancePriorKind.LEGACY
    environment_id: str = "env_default"
    missing_rate_min: float = 0.05
    missing_rate_max: float = 0.20
    proxy_noise_scale: float = 0.15
    spurious_strength: float = 0.65
    distractor_relation_count: int = 0
    distractor_strength: float = 0.20

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("nuisance.enabled must be a bool")
        if not isinstance(self.kind, NuisancePriorKind):
            raise TypeError("nuisance.kind must be NuisancePriorKind")
        if not isinstance(self.environment_id, str) or not self.environment_id.strip():
            raise ValueError("nuisance.environment_id must not be empty")
        if not 0 <= self.missing_rate_min <= self.missing_rate_max < 1:
            raise ValueError("nuisance missing-rate bounds must satisfy 0 <= min <= max < 1")
        if self.proxy_noise_scale < 0 or self.spurious_strength < 0:
            raise ValueError("nuisance strengths must be non-negative")
        if self.distractor_relation_count < 0:
            raise ValueError("distractor_relation_count must be non-negative")
        if not 0 <= self.distractor_strength <= 1:
            raise ValueError("distractor_strength must lie in [0, 1]")

class NuisancePlanner:
    def __init__(self, config: NuisanceOverlayConfig | None = None) -> None:
        self.config = config or NuisanceOverlayConfig()

    def plan(self, *, schema: PhysicalSchema, semantic_schema: SemanticSchemaPlan, runtime: RuntimeContext, mechanism: NuisancePriorKind | None = None) -> NuisancePlan:
        kind = mechanism or self.config.kind
        if not isinstance(kind, NuisancePriorKind):
            kind = NuisancePriorKind(kind)
        enabled = self.config.enabled or (
            mechanism is not None and kind is not self.config.kind
        )
        if not enabled or kind is NuisancePriorKind.LEGACY:
            return NuisancePlan(mechanism=mechanism_ref(NuisancePriorKind.LEGACY), environment_id=self.config.environment_id, enabled=False)
        features = [
            (table, column)
            for table in schema.tables
            for column in table.columns
            if column.kind is ColumnKind.FEATURE
        ]
        by_table: dict[str, list[str]] = {}
        for table, column in features:
            by_table.setdefault(table.table_id, []).append(column.column_id)
        target_by_table = {table_id: ids[0] for table_id, ids in by_table.items() if ids}
        role_values: list[NuisanceColumnPlan] = []
        for table, column in features:
            ids = by_table[table.table_id]
            source_ids = tuple(item for item in ids if item != column.column_id)[:1]
            role = NuisanceColumnRole.CAUSAL
            try:
                semantic_role = semantic_schema.column_role(column.column_id).value
            except KeyError:
                semantic_role = "unknown"
            parameters: tuple[tuple[str, object], ...] = (
                ("semantic_role", semantic_role),
            )
            if kind is NuisancePriorKind.PROXY and column.column_id == target_by_table.get(table.table_id):
                role = NuisanceColumnRole.PROXY
                parameters = (("semantic_role", semantic_role), ("noise_scale", self.config.proxy_noise_scale))
            elif kind is NuisancePriorKind.SPURIOUS and column.column_id == target_by_table.get(table.table_id):
                role = NuisanceColumnRole.SPURIOUS
                parameters = (("semantic_role", semantic_role), ("strength", self.config.spurious_strength))
            elif kind is NuisancePriorKind.DISTRACTOR and column.column_id == target_by_table.get(table.table_id):
                role = NuisanceColumnRole.PURE_NOISE
            elif kind is NuisancePriorKind.PROXY and source_ids and column.column_id == source_ids[0]:
                role = NuisanceColumnRole.REDUNDANT
            role_values.append(NuisanceColumnPlan(column_id=column.column_id, role=role, source_column_ids=source_ids, parameters=parameters))
        missingness: list[MissingnessPlan] = []
        if kind in {NuisancePriorKind.MCAR, NuisancePriorKind.MAR, NuisancePriorKind.MNAR}:
            for table, column in features:
                if not column.nullable:
                    continue
                rng = runtime.numpy_rng("nuisance", "missingness", column.column_id)
                rate = float(rng.uniform(self.config.missing_rate_min, self.config.missing_rate_max))
                drivers = tuple(item for item in by_table[table.table_id] if item != column.column_id)[:2]
                missingness.append(MissingnessPlan(column_id=column.column_id, family=kind.value, rate=rate, driver_column_ids=drivers if kind is NuisancePriorKind.MAR else (), seed=runtime.seed("nuisance", "missingness", column.column_id)))
        distractors: list[DistractorRelationPlan] = []
        if kind is NuisancePriorKind.DISTRACTOR:
            candidates = [fk for fk in schema.foreign_keys if fk.parent_table_id != fk.child_table_id][:self.config.distractor_relation_count]
            distractors = [
                DistractorRelationPlan(
                    relation_id=fk.foreign_key_id,
                    family="weakly_correlated_path",
                    parent_table_id=fk.parent_table_id,
                    child_table_id=fk.child_table_id,
                    strength=self.config.distractor_strength,
                    seed=runtime.seed("nuisance", "relation", fk.foreign_key_id),
                )
                for fk in candidates
            ]
        return NuisancePlan(
            mechanism=mechanism_ref(kind),
            column_roles=tuple(role_values),
            missingness_plans=tuple(missingness),
            distractor_relation_plans=tuple(distractors),
            environment_id=self.config.environment_id,
            enabled=True,
            parameters=(("config_kind", kind.value), ("semantic_schema_id", semantic_schema.schema_id)),
        )

__all__ = ["NuisanceOverlayConfig", "NuisancePlanner"]
