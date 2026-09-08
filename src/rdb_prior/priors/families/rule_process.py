"""Rule/process prior family adapter.

The actual program is represented by :mod:`rdb_prior.process.model`; this
module is the family-facing boundary used by the prior registry and planner.
"""

from __future__ import annotations

from rdb_prior.priors.model import MechanismRef, ProcessPriorKind
from rdb_prior.priors.registry import mechanism_ref
from rdb_prior.process.model import RulePlan


RULE_PROCESS_VERSION = "v1"


def rule_process_ref(rule: RulePlan) -> MechanismRef:
    if not isinstance(rule, RulePlan):
        raise TypeError("rule must be RulePlan")
    return mechanism_ref(
        ProcessPriorKind.RULE,
        parameters=(("rule_id", rule.rule_id), ("rule_plan", rule.to_dict())),
    )


def is_rule_process_ref(ref: MechanismRef) -> bool:
    return isinstance(ref, MechanismRef) and ref.kind == ProcessPriorKind.RULE.value


__all__ = ["RULE_PROCESS_VERSION", "rule_process_ref", "is_rule_process_ref"]
