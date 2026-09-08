"""Explicit acyclic column-SCM planning and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from rdb_prior.compilation.model import ColumnKind, PhysicalSchema, PhysicalTable
from rdb_prior.generation.feature_strategies import generate_feature_signal
from rdb_prior.generation.latent import LatentRegistry
from rdb_prior.generation.model import TableData
from rdb_prior.instance.plan import ColumnMechanismPlan, FeatureSCMFamily


@dataclass(frozen=True, slots=True, kw_only=True)
class ColumnSCMPlan:
    """Public column-DAG representation used by the relational SCM prior."""

    column_id: str
    parent_column_ids: tuple[str, ...] = ()
    parent_state_ids: tuple[str, ...] = ()
    family: str = "exogenous"
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.column_id, str) or not self.column_id:
            raise ValueError("column_id must be a non-empty string")
        if not isinstance(self.parent_column_ids, tuple):
            raise TypeError("parent_column_ids must be a tuple")
        if not isinstance(self.parent_state_ids, tuple):
            raise TypeError("parent_state_ids must be a tuple")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("family must be a non-empty string")
        allowed = {item.value for item in FeatureSCMFamily}
        if self.family not in allowed:
            raise ValueError(f"unsupported column SCM family: {self.family}")
        if not isinstance(self.parameters, tuple):
            raise TypeError("parameters must be a tuple")

    def to_dict(self) -> dict[str, object]:
        return {
            "column_id": self.column_id,
            "parent_column_ids": list(self.parent_column_ids),
            "parent_state_ids": list(self.parent_state_ids),
            "family": self.family,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ColumnSCMPlan":
        return cls(
            column_id=str(data["column_id"]),
            parent_column_ids=tuple(data.get("parent_column_ids", ())),
            parent_state_ids=tuple(data.get("parent_state_ids", ())),
            family=str(data.get("family", "exogenous")),
            parameters=tuple(dict(data.get("parameters", {})).items()),
        )

    def to_mechanism(self) -> ColumnMechanismPlan:
        return ColumnMechanismPlan(
            column_id=self.column_id,
            family=self.family,
            parent_column_ids=self.parent_column_ids,
            shared_state_ids=self.parent_state_ids,
            parameters=self.parameters,
        )


def validate_column_dag(plans: tuple[ColumnSCMPlan, ...]) -> None:
    """Reject cycles among known columns; external parent columns are roots."""
    if not isinstance(plans, tuple):
        raise TypeError("column DAG plans must be a tuple")
    by_id = {item.column_id: item for item in plans}
    if len(by_id) != len(plans):
        raise ValueError("column DAG column IDs must be unique")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(column_id: str) -> None:
        if column_id in visiting:
            raise ValueError("column SCM dependency graph must be acyclic")
        if column_id in visited:
            return
        visiting.add(column_id)
        for parent_id in by_id[column_id].parent_column_ids:
            if parent_id in by_id:
                visit(parent_id)
        visiting.remove(column_id)
        visited.add(column_id)

    for column_id in by_id:
        visit(column_id)


def execute_column_dag(
    *,
    schema: PhysicalSchema,
    table: PhysicalTable,
    mechanisms: Mapping[str, ColumnMechanismPlan],
    latents: LatentRegistry,
    relations: Mapping[str, np.ndarray],
    generated_tables: Mapping[str, TableData],
) -> dict[str, np.ndarray]:
    """Execute one table's sampled DAG in physical column order."""
    rows = latents.table(table.table_id).values.shape[0]
    own_latent = latents.table(table.table_id).values
    local_values: dict[str, np.ndarray] = {}
    result: dict[str, np.ndarray] = {}
    for column in table.columns:
        if column.kind is not ColumnKind.FEATURE:
            continue
        mechanism = mechanisms.get(column.column_id)
        if mechanism is None:
            continue
        parent_values = [
            _resolve_parent_column(
                parent_id=parent_id,
                schema=schema,
                table=table,
                relations=relations,
                generated_tables=generated_tables,
                local_values=local_values,
                rows=rows,
            )
            for parent_id in mechanism.parent_column_ids
        ]
        context = [own_latent]
        if parent_values:
            context.extend(parent_values)
        matrix = np.column_stack(context)
        parameters = dict(mechanism.parameters)
        family = FeatureSCMFamily(mechanism.family)
        signal = generate_feature_signal(
            family,
            matrix,
            np.random.Generator(
                np.random.PCG64DXSM(
                    int(parameters.get("mechanism_seed", 0))
                )
            ),
            noise_scale=float(parameters.get("noise_scale", 0.20)),
            signal_scale=float(parameters.get("signal_scale", 1.0)),
            activation_scale=float(parameters.get("activation_scale", 1.0)),
            output_scale=float(parameters.get("output_scale", 1.0)),
            long_tail_enabled=False,
            long_tail_alpha=1.5,
            mlp_depth=int(parameters.get("mlp_depth", 1)),
            mlp_hidden_factor=float(parameters.get("mlp_hidden_factor", 2.0)),
            mlp_dropout_rate=float(parameters.get("mlp_dropout_rate", 0.0)),
        )
        local_values[column.column_id] = signal
        result[column.column_id] = signal
    return result


def _resolve_parent_column(
    *,
    parent_id: str,
    schema: PhysicalSchema,
    table: PhysicalTable,
    relations: Mapping[str, np.ndarray],
    generated_tables: Mapping[str, TableData],
    local_values: Mapping[str, np.ndarray],
    rows: int,
) -> np.ndarray:
    if parent_id in local_values:
        return np.asarray(local_values[parent_id], dtype=np.float64)
    owner = next(
        (
            candidate
            for candidate in schema.tables
            if any(item.column_id == parent_id for item in candidate.columns)
        ),
        None,
    )
    if owner is None:
        raise KeyError(f"column DAG references unknown column {parent_id!r}")
    parent_data = generated_tables.get(owner.table_id)
    if parent_data is None:
        return np.zeros(rows, dtype=np.float64)
    raw = _numeric(parent_data.column(parent_id))
    if owner.table_id == table.table_id:
        return raw
    foreign_key = next(
        (
            item
            for item in schema.foreign_keys
            if item.child_table_id == table.table_id
            and item.parent_table_id == owner.table_id
        ),
        None,
    )
    if foreign_key is None:
        return np.zeros(rows, dtype=np.float64)
    assignments = relations[foreign_key.foreign_key_id]
    selected = np.zeros(rows, dtype=np.float64)
    valid = assignments >= 0
    selected[valid] = raw[assignments[valid]]
    return selected


def _numeric(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind in {"b", "i", "u", "f"}:
        numeric = raw.astype(np.float64)
        finite = np.isfinite(numeric)
        fill = float(np.mean(numeric[finite])) if np.any(finite) else 0.0
        numeric = np.where(finite, numeric, fill)
    else:
        _categories, encoded = np.unique(raw.astype(str), return_inverse=True)
        numeric = encoded.astype(np.float64)
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=8.0, neginf=-8.0)
    return _standardize(numeric)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - float(values.mean())) / max(float(values.std()), 1e-6)


__all__ = ["ColumnSCMPlan", "execute_column_dag", "validate_column_dag"]
