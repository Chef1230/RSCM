"""Anonymous domain prototypes for generator-private semantic plans."""

from __future__ import annotations

from rdb_prior.compilation.model import ColumnKind, PhysicalDataType, PhysicalSchema
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.semantics import (
    ColumnSemanticPlan,
    ColumnSemanticRole,
    SemanticSchemaPlan,
    TableSemanticPlan,
    TableSemanticRole,
)
from rdb_prior.schema.spec import TableRole


_TABLE_ROLES = {
    TableRole.ENTITY: (TableSemanticRole.ACTOR, TableSemanticRole.OBJECT),
    TableRole.EVENT: (
        TableSemanticRole.TRANSACTION,
        TableSemanticRole.OBSERVATION,
        TableSemanticRole.INTERACTION,
        TableSemanticRole.STATE_CHANGE,
    ),
    TableRole.BRIDGE: (TableSemanticRole.INTERACTION,),
    TableRole.LOOKUP: (TableSemanticRole.REFERENCE,),
    TableRole.DETAIL: (TableSemanticRole.OBSERVATION,),
}


def sample_semantic_schema(schema: PhysicalSchema, runtime: RuntimeContext) -> SemanticSchemaPlan:
    """Sample a deterministic semantic annotation without reading names."""
    if not isinstance(schema, PhysicalSchema):
        raise TypeError("schema must be PhysicalSchema")
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be RuntimeContext")
    table_plans: list[TableSemanticPlan] = []
    column_plans: list[ColumnSemanticPlan] = []
    for table in schema.tables:
        table_rng = runtime.python_rng("semantic-schema", "table", table.table_id)
        table_plans.append(
            TableSemanticPlan(table_id=table.table_id, role=table_rng.choice(_TABLE_ROLES[table.role]))
        )
        for column in table.columns:
            if column.kind in {ColumnKind.PRIMARY_KEY, ColumnKind.FOREIGN_KEY}:
                continue
            column_plans.append(
                ColumnSemanticPlan(
                    column_id=column.column_id,
                    role=_column_role(column.kind, column.data_type, runtime, column.column_id),
                )
            )
    return SemanticSchemaPlan(
        schema_id=schema.schema_id,
        prototype_id="anonymous_relational_v1",
        seed=runtime.seed("semantic-schema", "global"),
        tables=tuple(table_plans),
        columns=tuple(column_plans),
    )


def _column_role(kind: ColumnKind, data_type: PhysicalDataType, runtime: RuntimeContext, column_id: str) -> ColumnSemanticRole:
    if kind is ColumnKind.TIME:
        return ColumnSemanticRole.TIMESTAMP
    rng = runtime.python_rng("semantic-schema", "column", column_id)
    if data_type is PhysicalDataType.BOOLEAN:
        return rng.choice((ColumnSemanticRole.STATE, ColumnSemanticRole.OUTCOME))
    if data_type in {PhysicalDataType.INTEGER, PhysicalDataType.DOUBLE}:
        return rng.choice((ColumnSemanticRole.MEASUREMENT, ColumnSemanticRole.AMOUNT, ColumnSemanticRole.STATE))
    return rng.choice((ColumnSemanticRole.STATIC_ATTRIBUTE, ColumnSemanticRole.CATEGORY, ColumnSemanticRole.ACTION_TYPE))


__all__ = ["sample_semantic_schema"]
