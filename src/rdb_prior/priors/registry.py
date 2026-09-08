"""Explicit registry of supported prior executors."""

from __future__ import annotations

from dataclasses import dataclass

from rdb_prior.priors.model import MechanismRef, PriorFamily


@dataclass(frozen=True, slots=True, kw_only=True)
class PriorFamilyDescriptor:
    family: PriorFamily
    version: str
    implemented: bool


_DESCRIPTORS = {
    PriorFamily.LEGACY_ROLE_SCM: PriorFamilyDescriptor(family=PriorFamily.LEGACY_ROLE_SCM, version="v1", implemented=True),
    PriorFamily.TEMPORAL_EVENT: PriorFamilyDescriptor(family=PriorFamily.TEMPORAL_EVENT, version="v1", implemented=True),
    PriorFamily.RELATIONAL_SCM: PriorFamilyDescriptor(family=PriorFamily.RELATIONAL_SCM, version="reserved", implemented=False),
    PriorFamily.RELATIONAL_TREE: PriorFamilyDescriptor(family=PriorFamily.RELATIONAL_TREE, version="v1", implemented=True),
    PriorFamily.RULE_PROCESS: PriorFamilyDescriptor(family=PriorFamily.RULE_PROCESS, version="reserved", implemented=False),
}


_MECHANISM_VERSIONS = {
    "legacy_scm": "v1",
    "column_scm": "v1",
    "tree": "v1",
    "rule": "v1",
    "uniform": "v1",
    "latent_affinity": "v1",
    "scm": "v1",
    "community": "v1",
    "state_conditioned_event": "v1",
    "static": "v1",
    "stationary": "v1",
    "seasonal": "v1",
    "churn": "v1",
    "renewal": "v1",
    "none": "v1",
    "state_machine": "v1",
    "workflow": "v1",
    "legacy": "v1",
    "mcar": "v1",
    "mar": "v1",
    "mnar": "v1",
    "proxy": "v1",
    "spurious": "v1",
    "distractor": "v1",
}


def mechanism_ref(
    kind: object,
    *,
    parameters: tuple[tuple[str, object], ...] = (),
) -> MechanismRef:
    """Return a serializable, explicitly versioned mechanism reference."""
    value = getattr(kind, "value", kind)
    if not isinstance(value, str) or value not in _MECHANISM_VERSIONS:
        raise ValueError(f"unknown mechanism kind: {value!r}")
    return MechanismRef(
        kind=value,
        version=_MECHANISM_VERSIONS[value],
        parameters=parameters,
    )


def descriptor(family: PriorFamily) -> PriorFamilyDescriptor:
    return _DESCRIPTORS[family]


def is_implemented(family: PriorFamily) -> bool:
    return descriptor(family).implemented


__all__ = ["PriorFamilyDescriptor", "descriptor", "is_implemented", "mechanism_ref"]
