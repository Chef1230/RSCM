"""Private semantic annotations used only by the synthetic generator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


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

    def __post_init__(self) -> None:
        _identifier("column_id", self.column_id)
        if not isinstance(self.role, ColumnSemanticRole):
            raise TypeError("role must be ColumnSemanticRole")

    def to_dict(self) -> dict[str, str]:
        return {"column_id": self.column_id, "role": self.role.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ColumnSemanticPlan":
        return cls(column_id=data["column_id"], role=ColumnSemanticRole(data["role"]))


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticSchemaPlan:
    schema_id: str
    prototype_id: str
    seed: int
    tables: tuple[TableSemanticPlan, ...]
    columns: tuple[ColumnSemanticPlan, ...]

    def __post_init__(self) -> None:
        _identifier("schema_id", self.schema_id)
        _identifier("prototype_id", self.prototype_id)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.tables or not isinstance(self.tables, tuple):
            raise ValueError("tables must be a non-empty tuple")
        if not isinstance(self.columns, tuple):
            raise TypeError("columns must be a tuple")
        if not all(isinstance(item, TableSemanticPlan) for item in self.tables):
            raise TypeError("tables must contain TableSemanticPlan values")
        if not all(isinstance(item, ColumnSemanticPlan) for item in self.columns):
            raise TypeError("columns must contain ColumnSemanticPlan values")
        if len({item.table_id for item in self.tables}) != len(self.tables):
            raise ValueError("table semantic IDs must be unique")
        if len({item.column_id for item in self.columns}) != len(self.columns):
            raise ValueError("column semantic IDs must be unique")
        object.__setattr__(self, "tables", tuple(sorted(self.tables, key=lambda item: item.table_id)))
        object.__setattr__(self, "columns", tuple(sorted(self.columns, key=lambda item: item.column_id)))

    def table_role(self, table_id: str) -> TableSemanticRole:
        for item in self.tables:
            if item.table_id == table_id:
                return item.role
        raise KeyError(table_id)

    def column_role(self, column_id: str) -> ColumnSemanticRole:
        for item in self.columns:
            if item.column_id == column_id:
                return item.role
        raise KeyError(column_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "prototype_id": self.prototype_id,
            "seed": self.seed,
            "tables": [item.to_dict() for item in self.tables],
            "columns": [item.to_dict() for item in self.columns],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticSchemaPlan":
        return cls(
            schema_id=data["schema_id"],
            prototype_id=data["prototype_id"],
            seed=data["seed"],
            tables=tuple(TableSemanticPlan.from_dict(item) for item in data["tables"]),
            columns=tuple(ColumnSemanticPlan.from_dict(item) for item in data.get("columns", ())),
        )


__all__ = [
    "TableSemanticRole",
    "ColumnSemanticRole",
    "TableSemanticPlan",
    "ColumnSemanticPlan",
    "SemanticSchemaPlan",
]
