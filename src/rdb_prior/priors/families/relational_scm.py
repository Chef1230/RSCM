"""Binder for executable relational SCM mechanisms."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping

from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.generation.column_dag import ColumnSCMPlan, validate_column_dag
from rdb_prior.instance.plan import (
    ColumnMechanismPlan,
    InstancePlan,
    PopulationMechanismPlan,
)
from rdb_prior.priors.model import DatabasePriorPlan, PriorFamily


def bind_relational_scm_plan(
    schema: PhysicalSchema,
    plan: InstancePlan,
    prior_plan: DatabasePriorPlan,
) -> InstancePlan:
    """Materialize column DAG, relation propensity, and population bindings."""
    if prior_plan.family is not PriorFamily.RELATIONAL_SCM:
        raise ValueError("relational SCM binder requires relational_scm prior")
    column_values: dict[str, ColumnMechanismPlan] = {}
    relation_values: dict[str, str] = {}
    population_values: dict[str, Mapping[str, object]] = {}
    for bundle in prior_plan.motif_bundles:
        if bundle.family is not PriorFamily.RELATIONAL_SCM:
            continue
        parameters = dict(bundle.parameters)
        raw_columns = parameters.get("column_mechanisms", ())
        if not isinstance(raw_columns, list):
            raise TypeError("relational SCM column_mechanisms must be a list")
        for payload in raw_columns:
            if not isinstance(payload, Mapping):
                raise TypeError("relational SCM column mechanism must be a mapping")
            column_plan = ColumnSCMPlan.from_dict(payload)
            column_values[column_plan.column_id] = column_plan.to_mechanism()
        raw_relations = parameters.get("relation_mechanisms", {})
        if not isinstance(raw_relations, Mapping):
            raise TypeError("relational SCM relation_mechanisms must be a mapping")
        for foreign_key_id, payload in raw_relations.items():
            if isinstance(payload, Mapping):
                family = payload.get("family")
            else:
                family = payload
            if not isinstance(family, str) or not family:
                raise TypeError("relation SCM relation family must be a string")
            relation_values[str(foreign_key_id)] = family
        raw_populations = parameters.get("population_mechanisms", {})
        if not isinstance(raw_populations, Mapping):
            raise TypeError("relational SCM population_mechanisms must be a mapping")
        for table_id, payload in raw_populations.items():
            if not isinstance(payload, Mapping):
                raise TypeError("relational SCM population mechanism must be a mapping")
            population_values[str(table_id)] = payload

    column_plans = tuple(
        column_values[column_id]
        for column_id in sorted(column_values)
    )
    dag_plans = tuple(
        ColumnSCMPlan(
            column_id=item.column_id,
            parent_column_ids=item.parent_column_ids,
            parent_state_ids=item.shared_state_ids,
            family=item.family,
            parameters=item.parameters,
        )
        for item in column_plans
    )
    validate_column_dag(dag_plans)
    relations = tuple(
        replace(
            relation,
            family=relation_values[relation.foreign_key_ids[0]],
        )
        if len(relation.foreign_key_ids) == 1
        and relation.foreign_key_ids[0] in relation_values
        else relation
        for relation in plan.relations
    )
    existing_by_table = {
        item.table_id: item for item in plan.population_mechanisms
    }
    population_mechanisms = dict(existing_by_table)
    for table_id, payload in population_values.items():
        population_mechanisms[table_id] = PopulationMechanismPlan(
            table_id=table_id,
            family=str(payload["family"]),
            parent_table_id=(
                None
                if payload.get("parent_table_id") is None
                else str(payload["parent_table_id"])
            ),
            state_ids=(),
            parameters=tuple(
                (str(key), value)
                for key, value in payload.items()
                if key not in {"table_id", "parent_table_id", "family"}
            ),
        )
    return replace(
        plan,
        relations=relations,
        prior_plan_id=prior_plan.plan_id,
        prior_composition_id=prior_plan.composition.plan_id,
        prior_family=prior_plan.family.value,
        motif_bundles=prior_plan.motif_bundles,
        nuisance_plan=prior_plan.composition.nuisance_plan,
        column_mechanisms=column_plans,
        population_mechanisms=tuple(
            population_mechanisms[item]
            for item in sorted(population_mechanisms)
        ),
    )


__all__ = ["bind_relational_scm_plan"]
