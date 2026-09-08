"""Compatibility checks between anonymous motifs and prior families."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.schema.blueprint import SchemaBlueprint
from rdb_prior.schema.semantics import SemanticSchemaPlan
from rdb_prior.schema.spec import TableRole


@dataclass(frozen=True, slots=True, kw_only=True)
class EntityEventCandidate:
    motif_occurrence_id: str
    entity_table_id: str
    event_table_id: str
    foreign_key_id: str


def _value(item: object) -> str:
    value = getattr(item, "value", item)
    return str(value).lower()


def compatible(
    motif_type: str,
    structural_roles: Mapping[str, object] | Iterable[object],
    semantic_roles: Mapping[str, object] | Iterable[object],
    selected_components: Iterable[object] = (),
) -> bool:
    """Return whether a semantic process can bind to a structural motif.

    The check is intentionally anonymous: callers provide slot/role mappings,
    never business table or column names. Unknown motifs remain compatible so
    legacy fallback can handle them.
    """
    if _value(motif_type) not in {"entity_event", "entity_event_temporal"}:
        return True
    if isinstance(semantic_roles, Mapping):
        entity_role = _value(semantic_roles.get("entity", ""))
        event_role = _value(semantic_roles.get("event", ""))
    else:
        values = tuple(_value(item) for item in semantic_roles)
        entity_role, event_role = (values + ("", ""))[:2]
    if not entity_role or not event_role:
        return True
    allowed = {
        "actor": {"transaction", "observation", "state_change", "interaction"},
        "object": {"observation", "transaction", "interaction"},
    }
    return event_role in allowed.get(entity_role, {event_role})


def entity_event_candidates(
    blueprint: SchemaBlueprint,
    schema: PhysicalSchema,
    semantic_schema: SemanticSchemaPlan | None = None,
) -> tuple[EntityEventCandidate, ...]:
    """Return P1-safe event edges with structural and semantic compatibility."""
    candidates: list[EntityEventCandidate] = []
    for occurrence in blueprint.motif_occurrences:
        bindings = occurrence.nodes
        if "entity" not in bindings or "event" not in bindings:
            continue
        entity_id, event_id = bindings["entity"], bindings["event"]
        incoming = [
            fk
            for fk in schema.foreign_keys
            if fk.child_table_id == event_id
            and fk.relation_strategy != "lookup_assignment"
        ]
        if len(incoming) != 1:
            continue
        foreign_key = incoming[0]
        if foreign_key.parent_table_id != entity_id:
            continue
        if (
            schema.table(entity_id).role is not TableRole.ENTITY
            or schema.table(event_id).role is not TableRole.EVENT
        ):
            continue
        if semantic_schema is not None and semantic_schema.nodes and not compatible(
            occurrence.motif_type,
            {
                "entity": schema.table(entity_id).role,
                "event": schema.table(event_id).role,
            },
            {
                "entity": semantic_schema.table_role(entity_id),
                "event": semantic_schema.table_role(event_id),
            },
        ):
            continue
        candidates.append(
            EntityEventCandidate(
                motif_occurrence_id=occurrence.occurrence_id,
                entity_table_id=entity_id,
                event_table_id=event_id,
                foreign_key_id=foreign_key.foreign_key_id,
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.motif_occurrence_id, item.foreign_key_id),
        )
    )


__all__ = ["EntityEventCandidate", "compatible", "entity_event_candidates"]
