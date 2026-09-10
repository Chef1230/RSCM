"""P7 binder for sampled relation/attribute tree mechanisms."""

from __future__ import annotations

from dataclasses import replace

from rdb_prior.compilation.model import ColumnKind, PhysicalSchema
from rdb_prior.generation.trees.model import ForestPlan
from rdb_prior.instance.plan import (
    ColumnMechanismPlan,
    InstancePlan,
)
from rdb_prior.priors.model import DatabasePriorPlan, PriorFamily


def bind_relational_tree_plan(
    schema: PhysicalSchema,
    plan: InstancePlan,
    prior_plan: DatabasePriorPlan,
) -> InstancePlan:
    """Attach sampled forests to relation and ordinary-attribute execution plans."""
    if prior_plan.family is not PriorFamily.RELATIONAL_TREE:
        raise ValueError("relational-tree binder requires relational_tree prior")
    relation_forests: dict[str, ForestPlan] = {}
    relation_families: dict[str, str] = {}
    attribute_forests: dict[str, ForestPlan] = {}
    for bundle in prior_plan.motif_bundles:
        if bundle.family is not PriorFamily.RELATIONAL_TREE:
            continue
        parameters = dict(bundle.parameters)
        for foreign_key_id, payload in dict(
            parameters.get("relation_forests", {})
        ).items():
            relation_forests[str(foreign_key_id)] = ForestPlan.from_dict(payload)
        for edge in bundle.edge_bindings:
            relation_families[edge.foreign_key_id] = edge.mechanism_id
        for column_id, payload in dict(
            parameters.get("attribute_forests", {})
        ).items():
            attribute_forests[str(column_id)] = ForestPlan.from_dict(payload)

    relations = tuple(
        replace(
            relation,
            family=relation_families.get(
                relation.foreign_key_ids[0], "tree_propensity"
            ),
            tree_forest=relation_forests.get(relation.foreign_key_ids[0]),
        )
        if len(relation.foreign_key_ids) == 1
        and (
            relation.foreign_key_ids[0] in relation_forests
            or relation.foreign_key_ids[0] in relation_families
        )
        else relation
        for relation in plan.relations
    )
    column_mechanisms = tuple(
        ColumnMechanismPlan(
            column_id=column_id,
            family="tree",
            parameters=(("forest", forest.to_dict()),),
        )
        for column_id, forest in sorted(attribute_forests.items())
        if _is_feature(schema, column_id)
    )
    return replace(
        plan,
        relations=relations,
        prior_plan_id=prior_plan.plan_id,
        prior_composition_id=prior_plan.composition.plan_id,
        prior_family=prior_plan.family.value,
        motif_bundles=prior_plan.motif_bundles,
        nuisance_plan=prior_plan.composition.nuisance_plan,
        column_mechanisms=column_mechanisms,
    )


def _is_feature(schema: PhysicalSchema, column_id: str) -> bool:
    for table in schema.tables:
        for column in table.columns:
            if column.column_id == column_id:
                return column.kind is ColumnKind.FEATURE
    return False


__all__ = ["bind_relational_tree_plan"]
