"""Relation distractor overlays that preserve referential integrity."""
from __future__ import annotations
import numpy as np
from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.priors.model import DistractorRelationPlan

def apply_distractor_relation(*, schema: PhysicalSchema, tables: dict[str, dict[str, np.ndarray]], plan: DistractorRelationPlan, rng: np.random.Generator) -> int:
    fk = next((item for item in schema.foreign_keys if item.foreign_key_id == plan.relation_id), None)
    if fk is None:
        return 0
    child = tables.get(fk.child_table_id)
    parent = tables.get(fk.parent_table_id)
    if child is None or parent is None or fk.child_column_id not in child:
        return 0
    parent_key = parent.get(fk.parent_column_id)
    if parent_key is None or len(parent_key) == 0:
        return 0
    values = np.asarray(child[fk.child_column_id]).copy()
    valid = values >= 0
    mask = valid & (rng.random(len(values)) < np.clip(float(plan.strength), 0.0, 1.0))
    if not np.any(mask):
        return 0
    parent_indices = rng.integers(0, len(parent_key), size=int(mask.sum()))
    values[mask] = np.asarray(parent_key)[parent_indices]
    child[fk.child_column_id] = values
    return int(mask.sum())

__all__ = ["apply_distractor_relation"]
