"""Execution of nuisance overlays after the causal database is materialized."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from rdb_prior.compilation.model import ColumnKind, PhysicalSchema
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rdb_prior.generation.latent import LatentRegistry
from rdb_prior.generation.model import DatabaseInstance, TableData
from rdb_prior.priors.model import NuisanceColumnRole, NuisancePlan
from .distractors import apply_distractor_relation
from .missingness import apply_missing_mask, sample_missing_mask

@dataclass(frozen=True, slots=True)
class NuisanceExecutionReport:
    enabled: bool
    column_overlays: int
    missing_values: int
    relation_overlays: int
    provenance: tuple[tuple[str, object], ...] = ()

def _stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")

def _numeric(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"b", "i", "u", "f"}:
        return np.nan_to_num(values.astype(np.float64), nan=0.0)
    if values.dtype.kind in {"U", "S"}:
        _, codes = np.unique(values.astype(str), return_inverse=True)
        return codes.astype(np.float64)
    return np.zeros(len(values), dtype=np.float64)

def _coerce_like(signal: np.ndarray, original: np.ndarray) -> np.ndarray:
    raw = np.asarray(original)
    signal = np.nan_to_num(np.asarray(signal, dtype=np.float64), nan=0.0)
    if raw.dtype.kind in {"U", "S"}:
        cardinality = max(1, min(max(2, len(raw)), len(signal)))
        if cardinality == 1:
            codes = np.zeros(len(signal), dtype=np.int64)
        else:
            low = float(signal.min()) if len(signal) else 0.0
            high = float(signal.max()) if len(signal) else 1.0
            if high <= low:
                high = low + 1.0
            bins = np.linspace(low, high, cardinality + 1)
            codes = np.clip(np.digitize(signal, bins[1:-1]), 0, cardinality-1)
        return np.char.add("v", codes.astype(str))
    if raw.dtype.kind == "b":
        return (signal > np.median(signal)).astype(np.int8)
    if raw.dtype.kind in {"i", "u"}:
        return np.rint(signal).astype(np.int64)
    return signal.astype(np.float64)

def _table_for_column(schema: PhysicalSchema, column_id: str):
    for table in schema.tables:
        if any(column.column_id == column_id for column in table.columns):
            return table
    return None

class NuisanceExecutor:
    """Apply a planned nuisance overlay without replacing causal values."""
    def apply(self, schema: PhysicalSchema, database: DatabaseInstance, plan: InstancePlan | NuisancePlan, latents: LatentRegistry | None = None) -> DatabaseInstance:
        from rdb_prior.instance.plan import InstancePlan
        nuisance = plan.nuisance_plan if isinstance(plan, InstancePlan) else plan
        if nuisance is None or not nuisance.enabled or nuisance.mechanism.kind == "legacy":
            return database
        if not isinstance(schema, PhysicalSchema) or not isinstance(database, DatabaseInstance):
            raise TypeError("schema and database must be native immutable models")
        tables = {
            table.table_id: {column_id: values.copy() for column_id, values in table.columns.items()}
            for table in database.tables
        }
        original = {
            table_id: {column_id: values.copy() for column_id, values in columns.items()}
            for table_id, columns in tables.items()
        }
        table_by_column = {
            column.column_id: table.table_id
            for table in schema.tables
            for column in table.columns
        }
        applied_columns = 0
        missing_values = 0
        relation_values = 0
        base_seed = _stable_int(nuisance.environment_id) ^ _stable_int(nuisance.mechanism.version)
        for item in nuisance.column_roles:
            table = _table_for_column(schema, item.column_id)
            if table is None or item.column_id not in tables[table.table_id]:
                continue
            column = table.column(item.column_id)
            if column.kind is not ColumnKind.FEATURE or item.role is NuisanceColumnRole.CAUSAL:
                continue
            rng = np.random.Generator(np.random.PCG64DXSM(base_seed ^ _stable_int(item.column_id)))
            raw = original[table.table_id][item.column_id]
            source = None
            for source_id in item.source_column_ids:
                source_table_id = table_by_column.get(source_id)
                if source_table_id is not None and source_id in original[source_table_id] and len(original[source_table_id][source_id]) == len(raw):
                    source = original[source_table_id][source_id]
                    break
            if source is None and latents is not None:
                try:
                    latent = latents.table(table.table_id).values
                    if len(latent) == len(raw):
                        source = latent[:, 0]
                except KeyError:
                    pass
            source_numeric = np.zeros(len(raw), dtype=np.float64) if source is None else _numeric(source)
            raw_numeric = _numeric(raw)
            parameters = dict(item.parameters)
            if item.role is NuisanceColumnRole.PROXY:
                signal = source_numeric + rng.normal(0.0, float(parameters.get("noise_scale", 0.15)), len(raw))
            elif item.role is NuisanceColumnRole.REDUNDANT:
                signal = source_numeric + rng.normal(0.0, float(parameters.get("noise_scale", 0.02)), len(raw))
            elif item.role is NuisanceColumnRole.SPURIOUS:
                env_sign = -1.0 if (_stable_int(nuisance.environment_id) & 1) else 1.0
                strength = float(parameters.get("strength", 0.65))
                signal = strength * source_numeric + env_sign * rng.normal(0.0, 1.0, len(raw))
            else:
                scale = max(float(np.nanstd(raw_numeric)), 1.0)
                signal = rng.normal(0.0, scale, len(raw))
            tables[table.table_id][item.column_id] = _coerce_like(signal, raw)
            applied_columns += 1
        for item in nuisance.missingness_plans:
            table = _table_for_column(schema, item.column_id)
            if table is None or item.column_id not in tables[table.table_id]:
                continue
            column = table.column(item.column_id)
            values = tables[table.table_id][item.column_id]
            driver_values = tuple(
                original[table_by_column[source_id]][source_id]
                for source_id in item.driver_column_ids
                if source_id in table_by_column and source_id in original[table_by_column[source_id]]
            )
            time_values = None
            if item.time_column_id is not None and item.time_column_id in original[table.table_id]:
                time_values = original[table.table_id][item.time_column_id]
            else:
                for candidate in table.columns:
                    if candidate.kind.value == "time" and candidate.column_id in original[table.table_id]:
                        time_values = original[table.table_id][candidate.column_id]
                        break
            latent_values = None
            if latents is not None:
                try:
                    candidate = latents.table(table.table_id).values
                    if len(candidate) == len(values):
                        latent_values = candidate[:, 0]
                except KeyError:
                    pass
            rng = np.random.Generator(np.random.PCG64DXSM(item.seed ^ base_seed))
            mask = sample_missing_mask(item, values, driver_values=driver_values, time_values=time_values, latent_values=latent_values, rng=rng)
            if column.nullable:
                tables[table.table_id][item.column_id] = apply_missing_mask(values, column, mask)
                missing_values += int(mask.sum())
        for item in nuisance.distractor_relation_plans:
            rng = np.random.Generator(np.random.PCG64DXSM(item.seed ^ base_seed))
            relation_values += apply_distractor_relation(schema=schema, tables=tables, plan=item, rng=rng)
        result = DatabaseInstance(
            instance_id=database.instance_id,
            schema_id=database.schema_id,
            plan_id=database.plan_id,
            tables=tuple(TableData(table_id=table_id, columns=columns) for table_id, columns in tables.items()),
        )
        return result

    def execute(self, **kwargs) -> DatabaseInstance:
        return self.apply(**kwargs)

def apply_nuisance_overlay(schema: PhysicalSchema, database: DatabaseInstance, plan: InstancePlan | NuisancePlan, latents: LatentRegistry | None = None) -> DatabaseInstance:
    return NuisanceExecutor().apply(schema, database, plan, latents)

__all__ = ["NuisanceExecutionReport", "NuisanceExecutor", "apply_nuisance_overlay"]
