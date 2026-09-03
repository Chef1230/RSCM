"""Compatibility checks between anonymous motifs and prior families."""

from __future__ import annotations

from dataclasses import dataclass

from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.schema.blueprint import SchemaBlueprint
from rdb_prior.schema.spec import TableRole


@dataclass(frozen=True, slots=True, kw_only=True)
class EntityEventCandidate:
    motif_occurrence_id: str
    entity_table_id: str
    event_table_id: str
    foreign_key_id: str


def entity_event_candidates(blueprint: SchemaBlueprint, schema: PhysicalSchema) -> tuple[EntityEventCandidate, ...]:
    """Return P1-safe event edges with exactly one structural Entity parent."""
    candidates: list[EntityEventCandidate] = []
    for occurrence in blueprint.motif_occurrences:
        bindings = occurrence.nodes
        if "entity" not in bindings or "event" not in bindings:
            continue
        entity_id, event_id = bindings["entity"], bindings["event"]
        incoming = [fk for fk in schema.foreign_keys if fk.child_table_id == event_id and fk.relation_strategy != "lookup_assignment"]
        if len(incoming) != 1:
            continue
        foreign_key = incoming[0]
        if foreign_key.parent_table_id != entity_id:
            continue
        if schema.table(entity_id).role is not TableRole.ENTITY or schema.table(event_id).role is not TableRole.EVENT:
            continue
        candidates.append(EntityEventCandidate(motif_occurrence_id=occurrence.occurrence_id, entity_table_id=entity_id, event_table_id=event_id, foreign_key_id=foreign_key.foreign_key_id))
    return tuple(sorted(candidates, key=lambda item: (item.motif_occurrence_id, item.foreign_key_id)))


__all__ = ["EntityEventCandidate", "entity_event_candidates"]
