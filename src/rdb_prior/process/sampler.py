"""Deterministic sampling of bounded Rule/process programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdb_prior.compilation.model import ColumnKind, PhysicalDataType, PhysicalSchema
from rdb_prior.process.model import ColumnRef, Compare, Constant, RuleAction, RulePlan
from rdb_prior.runtime import RuntimeContext


@dataclass(frozen=True, slots=True, kw_only=True)
class RuleProcessConfig:
    enabled: bool = True
    condition_probability: float = 0.65
    intensity_multiplier_min: float = 1.25
    intensity_multiplier_max: float = 2.25
    attribute_offset_min: float = 0.15
    attribute_offset_max: float = 0.75

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("rule_process.enabled must be a bool")
        for name in (
            "condition_probability",
            "intensity_multiplier_min",
            "intensity_multiplier_max",
            "attribute_offset_min",
            "attribute_offset_max",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
        if not 0.0 <= self.condition_probability <= 1.0:
            raise ValueError("condition_probability must be in [0, 1]")
        if self.intensity_multiplier_min <= 0 or self.intensity_multiplier_max < self.intensity_multiplier_min:
            raise ValueError("intensity multiplier range is invalid")
        if self.attribute_offset_max < self.attribute_offset_min:
            raise ValueError("attribute offset range is invalid")


class RuleProcessSampler:
    """Sample a small process rule from anonymous physical schema metadata."""

    def __init__(self, config: RuleProcessConfig | None = None) -> None:
        self.config = config or RuleProcessConfig()

    def sample(
        self,
        *,
        schema: PhysicalSchema,
        entity_table_id: str,
        event_table_id: str,
        foreign_key_id: str,
        runtime: RuntimeContext,
        state_id: str | None = None,
    ) -> RulePlan:
        if not self.config.enabled:
            raise ValueError("rule process prior is disabled")
        entity = schema.table(entity_table_id)
        feature = next((column for column in entity.columns if column.kind is ColumnKind.FEATURE), None)
        if feature is None:
            # A constant rule is still useful for schemas with key-only parents.
            condition = Compare("=", Constant(1), Constant(1))
            source_column = ""
        else:
            source_column = feature.column_id
            rng = runtime.numpy_rng("prior", "rule-process", entity_table_id, event_table_id)
            if feature.data_type in {
                PhysicalDataType.DOUBLE,
                PhysicalDataType.INTEGER,
                PhysicalDataType.BIGINT,
            }:
                threshold = float(rng.normal(0.0, 0.25))
                condition = Compare(">", ColumnRef(entity_table_id, source_column), Constant(threshold))
            elif feature.data_type is PhysicalDataType.BOOLEAN:
                condition = Compare("==", ColumnRef(entity_table_id, source_column), Constant(True))
            else:
                # Physical text values are strings; an empty-string comparison is
                # deterministic and type-safe before rows are materialized.
                condition = Compare("!=", ColumnRef(entity_table_id, source_column), Constant(""))
        rng = runtime.numpy_rng("prior", "rule-process-parameters", entity_table_id, event_table_id)
        multiplier = float(rng.uniform(self.config.intensity_multiplier_min, self.config.intensity_multiplier_max))
        offset = float(rng.uniform(self.config.attribute_offset_min, self.config.attribute_offset_max))
        actions = (
            RuleAction(target="event_intensity", operation="multiply", value=Constant(multiplier)),
            RuleAction(target="event_attribute", operation="add", value=Constant(offset)),
        )
        return RulePlan(
            rule_id=f"rule_{entity_table_id}_{event_table_id}",
            condition=condition,
            actions=actions,
            seed=runtime.seed("prior", "rule-process", entity_table_id, event_table_id),
            parameters=(
                ("entity_table_id", entity_table_id),
                ("event_table_id", event_table_id),
                ("foreign_key_id", foreign_key_id),
                ("source_column_id", source_column),
                ("state_id", state_id),
            ),
        )


def sample_rule_plan(
    *,
    schema: PhysicalSchema,
    entity_table_id: str,
    event_table_id: str,
    foreign_key_id: str,
    runtime: RuntimeContext,
    state_id: str | None = None,
    config: RuleProcessConfig | None = None,
) -> RulePlan:
    return RuleProcessSampler(config).sample(
        schema=schema,
        entity_table_id=entity_table_id,
        event_table_id=event_table_id,
        foreign_key_id=foreign_key_id,
        runtime=runtime,
        state_id=state_id,
    )


__all__ = ["RuleProcessConfig", "RuleProcessSampler", "sample_rule_plan"]
