"""Private semantic plans used to drive compilation and mechanism selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from rdb_prior.compilation.model import (
    ColumnKind,
    PhysicalDataType,
    PhysicalSchema,
)
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.blueprint import SchemaBlueprint
from rdb_prior.schema.spec import TableRole


class TableSemanticRole(str, Enum):
    ACTOR = "actor"
    OBJECT = "object"
    TRANSACTION = "transaction"
    OBSERVATION = "observation"
    STATE_CHANGE = "state_change"
    INTERACTION = "interaction"
    REFERENCE = "reference"


class ColumnSemanticRole(str, Enum):
    STATIC_ATTRIBUTE = "static_attribute"
    STATE = "state"
    MEASUREMENT = "measurement"
    AMOUNT = "amount"
    CATEGORY = "category"
    ACTION_TYPE = "action_type"
    OUTCOME = "outcome"
    TIMESTAMP = "timestamp"


def _identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticNodePlan:
    node_id: str
    role: TableSemanticRole
    required_column_roles: tuple[ColumnSemanticRole, ...] = ()
    optional_column_roles: tuple[ColumnSemanticRole, ...] = ()

    def __post_init__(self) -> None:
        _identifier("node_id", self.node_id)
        if not isinstance(self.role, TableSemanticRole):
            raise TypeError("role must be TableSemanticRole")
        for name in ("required_column_roles", "optional_column_roles"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not all(
                isinstance(item, ColumnSemanticRole) for item in values
            ):
                raise TypeError(f"{name} must contain ColumnSemanticRole values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "required_column_roles": [
                item.value for item in self.required_column_roles
            ],
            "optional_column_roles": [
                item.value for item in self.optional_column_roles
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticNodePlan":
        return cls(
            node_id=data["node_id"],
            role=TableSemanticRole(data["role"]),
            required_column_roles=tuple(
                ColumnSemanticRole(item)
                for item in data.get("required_column_roles", ())
            ),
            optional_column_roles=tuple(
                ColumnSemanticRole(item)
                for item in data.get("optional_column_roles", ())
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TableSemanticPlan:
    table_id: str
    role: TableSemanticRole

    def __post_init__(self) -> None:
        _identifier("table_id", self.table_id)
        if not isinstance(self.role, TableSemanticRole):
            raise TypeError("role must be TableSemanticRole")

    def to_dict(self) -> dict[str, str]:
        return {"table_id": self.table_id, "role": self.role.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TableSemanticPlan":
        return cls(table_id=data["table_id"], role=TableSemanticRole(data["role"]))


@dataclass(frozen=True, slots=True, kw_only=True)
class ColumnSemanticPlan:
    column_id: str
    role: ColumnSemanticRole
    semantic_column_id: str | None = None
    physical_column_id: str | None = None

    def __post_init__(self) -> None:
        _identifier("column_id", self.column_id)
        if not isinstance(self.role, ColumnSemanticRole):
            raise TypeError("role must be ColumnSemanticRole")
        semantic_id = self.semantic_column_id or f"semantic_{self.column_id}"
        physical_id = self.physical_column_id or self.column_id
        _identifier("semantic_column_id", semantic_id)
        _identifier("physical_column_id", physical_id)
        object.__setattr__(self, "semantic_column_id", semantic_id)
        object.__setattr__(self, "physical_column_id", physical_id)

    def to_dict(self) -> dict[str, str]:
        return {
            "column_id": self.column_id,
            "semantic_column_id": self.semantic_column_id,
            "physical_column_id": self.physical_column_id,
            "role": self.role.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ColumnSemanticPlan":
        return cls(
            column_id=data["column_id"],
            role=ColumnSemanticRole(data["role"]),
            semantic_column_id=data.get("semantic_column_id"),
            physical_column_id=data.get("physical_column_id"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticSchemaPlan:
    schema_id: str
    prototype_id: str
    seed: int
    tables: tuple[TableSemanticPlan, ...]
    columns: tuple[ColumnSemanticPlan, ...]
    nodes: tuple[SemanticNodePlan, ...] = ()

    def __post_init__(self) -> None:
        _identifier("schema_id", self.schema_id)
        _identifier("prototype_id", self.prototype_id)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.tables, tuple) or not self.tables:
            raise ValueError("tables must be a non-empty tuple")
        if not isinstance(self.columns, tuple):
            raise TypeError("columns must be a tuple")
        if not isinstance(self.nodes, tuple):
            raise TypeError("nodes must be a tuple")
        if not all(isinstance(item, TableSemanticPlan) for item in self.tables):
            raise TypeError("tables must contain TableSemanticPlan values")
        if not all(isinstance(item, ColumnSemanticPlan) for item in self.columns):
            raise TypeError("columns must contain ColumnSemanticPlan values")
        if not all(isinstance(item, SemanticNodePlan) for item in self.nodes):
            raise TypeError("nodes must contain SemanticNodePlan values")
        if len({item.table_id for item in self.tables}) != len(self.tables):
            raise ValueError("table semantic IDs must be unique")
        if len({item.column_id for item in self.columns}) != len(self.columns):
            raise ValueError("column semantic IDs must be unique")
        if len({item.semantic_column_id for item in self.columns}) != len(self.columns):
            raise ValueError("semantic column IDs must be unique")
        if len({item.physical_column_id for item in self.columns}) != len(self.columns):
            raise ValueError("physical column bindings must be unique")
        if len({item.node_id for item in self.nodes}) != len(self.nodes):
            raise ValueError("semantic node IDs must be unique")
        table_ids = {item.table_id for item in self.tables}
        if not {item.node_id for item in self.nodes}.issubset(table_ids):
            raise ValueError("semantic nodes must refer to semantic tables")
        object.__setattr__(
            self,
            "tables",
            tuple(sorted(self.tables, key=lambda item: item.table_id)),
        )
        object.__setattr__(
            self,
            "columns",
            tuple(sorted(self.columns, key=lambda item: item.column_id)),
        )
        object.__setattr__(
            self,
            "nodes",
            tuple(sorted(self.nodes, key=lambda item: item.node_id)),
        )

    def table_role(self, table_id: str) -> TableSemanticRole:
        for item in self.tables:
            if item.table_id == table_id:
                return item.role
        raise KeyError(table_id)

    def node_plan(self, node_id: str) -> SemanticNodePlan | None:
        return next((item for item in self.nodes if item.node_id == node_id), None)

    def column_role(self, column_id: str) -> ColumnSemanticRole:
        for item in self.columns:
            if column_id in {
                item.column_id,
                item.semantic_column_id,
                item.physical_column_id,
            }:
                return item.role
        raise KeyError(column_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "prototype_id": self.prototype_id,
            "seed": self.seed,
            "tables": [item.to_dict() for item in self.tables],
            "columns": [item.to_dict() for item in self.columns],
            "nodes": [item.to_dict() for item in self.nodes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticSchemaPlan":
        return cls(
            schema_id=data["schema_id"],
            prototype_id=data["prototype_id"],
            seed=data["seed"],
            tables=tuple(TableSemanticPlan.from_dict(item) for item in data["tables"]),
            columns=tuple(ColumnSemanticPlan.from_dict(item) for item in data.get("columns", ())),
            nodes=tuple(SemanticNodePlan.from_dict(item) for item in data.get("nodes", ())),
        )


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


def _required_roles(role: TableSemanticRole) -> tuple[ColumnSemanticRole, ...]:
    return {
        TableSemanticRole.ACTOR: (
            ColumnSemanticRole.STATIC_ATTRIBUTE,
            ColumnSemanticRole.STATE,
            ColumnSemanticRole.CATEGORY,
        ),
        TableSemanticRole.OBJECT: (
            ColumnSemanticRole.STATIC_ATTRIBUTE,
            ColumnSemanticRole.CATEGORY,
        ),
        TableSemanticRole.TRANSACTION: (
            ColumnSemanticRole.AMOUNT,
            ColumnSemanticRole.ACTION_TYPE,
            ColumnSemanticRole.OUTCOME,
        ),
        TableSemanticRole.OBSERVATION: (
            ColumnSemanticRole.MEASUREMENT,
            ColumnSemanticRole.OUTCOME,
        ),
        TableSemanticRole.STATE_CHANGE: (
            ColumnSemanticRole.STATE,
            ColumnSemanticRole.ACTION_TYPE,
        ),
        TableSemanticRole.INTERACTION: (
            ColumnSemanticRole.ACTION_TYPE,
            ColumnSemanticRole.OUTCOME,
        ),
        TableSemanticRole.REFERENCE: (ColumnSemanticRole.CATEGORY,),
    }[role]


def _column_role(
    kind: ColumnKind,
    data_type: PhysicalDataType,
    runtime: object,
    column_id: str,
) -> ColumnSemanticRole:
    if kind is ColumnKind.TIME:
        return ColumnSemanticRole.TIMESTAMP
    rng = runtime.python_rng("semantic-schema", "column", column_id)
    if data_type is PhysicalDataType.BOOLEAN:
        return rng.choice((ColumnSemanticRole.STATE, ColumnSemanticRole.OUTCOME))
    if data_type in {PhysicalDataType.INTEGER, PhysicalDataType.DOUBLE}:
        return rng.choice(
            (
                ColumnSemanticRole.MEASUREMENT,
                ColumnSemanticRole.AMOUNT,
                ColumnSemanticRole.STATE,
            )
        )
    return rng.choice(
        (
            ColumnSemanticRole.STATIC_ATTRIBUTE,
            ColumnSemanticRole.CATEGORY,
            ColumnSemanticRole.ACTION_TYPE,
        )
    )


def sample_semantic_schema(
    source: PhysicalSchema | SchemaBlueprint,
    runtime: object,
    *,
    schema_id: str | None = None,
) -> SemanticSchemaPlan:
    """Sample logical semantics from a blueprint or complete a physical schema."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be RuntimeContext")
    if isinstance(source, SchemaBlueprint):
        tables = tuple(
            TableSemanticPlan(
                table_id=node.node_id,
                role=(rng := runtime.python_rng(
                    "semantic-schema", "table", node.node_id
                )).choice(_TABLE_ROLES[node.role]),
            )
            for node in source.nodes
        )
        nodes = tuple(
            SemanticNodePlan(
                node_id=table.table_id,
                role=table.role,
                required_column_roles=_required_roles(table.role),
            )
            for table in tables
        )
        return SemanticSchemaPlan(
            schema_id=schema_id or source.blueprint_id,
            prototype_id="anonymous_relational_v2",
            seed=runtime.seed("semantic-schema", "global"),
            tables=tables,
            columns=(),
            nodes=nodes,
        )
    if not isinstance(source, PhysicalSchema):
        raise TypeError("source must be PhysicalSchema or SchemaBlueprint")
    table_plans = tuple(
        TableSemanticPlan(
            table_id=table.table_id,
            role=(
                runtime.python_rng(
                    "semantic-schema", "table", table.table_id
                ).choice(_TABLE_ROLES[table.role])
            ),
        )
        for table in source.tables
    )
    columns = tuple(
        ColumnSemanticPlan(
            column_id=column.column_id,
            role=_column_role(
                column.kind, column.data_type, runtime, column.column_id
            ),
            semantic_column_id=f"semantic_{column.column_id}",
            physical_column_id=column.column_id,
        )
        for table in source.tables
        for column in table.columns
        if column.kind not in {ColumnKind.PRIMARY_KEY, ColumnKind.FOREIGN_KEY}
    )
    return SemanticSchemaPlan(
        schema_id=source.schema_id,
        prototype_id="anonymous_relational_v2",
        seed=runtime.seed("semantic-schema", "global"),
        tables=table_plans,
        columns=columns,
    )


def complete_semantic_schema(
    logical: SemanticSchemaPlan,
    schema: PhysicalSchema,
) -> SemanticSchemaPlan:
    """Bind logical roles to physical columns without consuming built-ins."""
    role_by_table = {item.table_id: item.role for item in logical.tables}
    node_by_table = {item.node_id: item for item in logical.nodes}
    logical_by_column = {
        item.physical_column_id or item.column_id: item
        for item in logical.columns
    }
    columns: list[ColumnSemanticPlan] = []
    for table in schema.tables:
        node = node_by_table.get(table.table_id)
        planned = (
            list(node.required_column_roles) + list(node.optional_column_roles)
            if node is not None
            else []
        )
        builtin_features = {
            TableRole.LOOKUP: 2,
            TableRole.DETAIL: 1,
        }.get(table.role, 0)
        feature_ordinal = 0
        planned_index = 0
        semantic_index = 0
        for column in table.columns:
            if column.kind in {ColumnKind.PRIMARY_KEY, ColumnKind.FOREIGN_KEY}:
                continue
            if column.kind is ColumnKind.TIME:
                role = ColumnSemanticRole.TIMESTAMP
            elif feature_ordinal < builtin_features:
                role = (
                    ColumnSemanticRole.CATEGORY
                    if table.role is TableRole.LOOKUP and feature_ordinal == 0
                    else ColumnSemanticRole.STATIC_ATTRIBUTE
                )
                feature_ordinal += 1
            elif planned_index < len(planned):
                role = planned[planned_index]
                planned_index += 1
                feature_ordinal += 1
            else:
                role = ColumnSemanticRole.STATIC_ATTRIBUTE
                feature_ordinal += 1
            existing = logical_by_column.get(column.column_id)
            semantic_id = (
                existing.semantic_column_id
                if existing is not None
                else f"semantic_{table.table_id}_{semantic_index:03d}"
            )
            if existing is not None:
                role = existing.role
            columns.append(
                ColumnSemanticPlan(
                    column_id=column.column_id,
                    role=role,
                    semantic_column_id=semantic_id,
                    physical_column_id=column.column_id,
                )
            )
            semantic_index += 1
    return SemanticSchemaPlan(
        schema_id=schema.schema_id,
        prototype_id=logical.prototype_id,
        seed=logical.seed,
        tables=tuple(
            TableSemanticPlan(table_id=table_id, role=role)
            for table_id, role in role_by_table.items()
        ),
        columns=tuple(columns),
        nodes=logical.nodes,
    )


__all__ = [
    "TableSemanticRole",
    "ColumnSemanticRole",
    "SemanticNodePlan",
    "TableSemanticPlan",
    "ColumnSemanticPlan",
    "SemanticSchemaPlan",
    "sample_semantic_schema",
    "complete_semantic_schema",
]
