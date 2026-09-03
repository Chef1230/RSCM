"""Explicit registry of supported prior executors."""

from __future__ import annotations

from dataclasses import dataclass

from rdb_prior.priors.model import PriorFamily


@dataclass(frozen=True, slots=True, kw_only=True)
class PriorFamilyDescriptor:
    family: PriorFamily
    version: str
    implemented: bool


_DESCRIPTORS = {
    PriorFamily.LEGACY_ROLE_SCM: PriorFamilyDescriptor(family=PriorFamily.LEGACY_ROLE_SCM, version="v1", implemented=True),
    PriorFamily.TEMPORAL_EVENT: PriorFamilyDescriptor(family=PriorFamily.TEMPORAL_EVENT, version="v1", implemented=True),
    PriorFamily.RELATIONAL_SCM: PriorFamilyDescriptor(family=PriorFamily.RELATIONAL_SCM, version="reserved", implemented=False),
    PriorFamily.RELATIONAL_TREE: PriorFamilyDescriptor(family=PriorFamily.RELATIONAL_TREE, version="reserved", implemented=False),
    PriorFamily.RULE_PROCESS: PriorFamilyDescriptor(family=PriorFamily.RULE_PROCESS, version="reserved", implemented=False),
}


def descriptor(family: PriorFamily) -> PriorFamilyDescriptor:
    return _DESCRIPTORS[family]


def is_implemented(family: PriorFamily) -> bool:
    return descriptor(family).implemented


__all__ = ["PriorFamilyDescriptor", "descriptor", "is_implemented"]
