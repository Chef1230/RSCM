"""Small, JSON-serializable AST for rule/process priors.

The AST is intentionally closed: the executor knows every node type and never
evaluates Python source.  Plans can therefore be persisted and replayed on a
different host without importing user code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, ClassVar, Mapping


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _json_value(value: object, name: str = "value") -> object:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-safe") from error
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class RuleNode:
    """Base class for the closed rule AST."""

    _types: ClassVar[dict[str, type["RuleNode"]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        kind = getattr(cls, "kind", None)
        if isinstance(kind, str):
            RuleNode._types[kind] = cls

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - abstract contract
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuleNode":
        if not isinstance(data, Mapping):
            raise TypeError("rule node must be a mapping")
        kind = data.get("type", data.get("kind"))
        aliases = {
            "Constant": "constant",
            "ColumnRef": "column_ref",
            "StateRef": "state_ref",
            "Compare": "compare",
            "And": "and",
            "Or": "or",
            "Not": "not",
            "Add": "add",
            "Multiply": "multiply",
            "Aggregate": "aggregate",
            "Exists": "exists",
            "Lag": "lag",
            "StateTransition": "state_transition",
        }
        kind = aliases.get(kind, kind)
        node_type = cls._types.get(kind)
        if node_type is None:
            raise ValueError(f"unsupported rule node type: {kind!r}")
        return node_type._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "RuleNode":
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Constant(RuleNode):
    kind: ClassVar[str] = "constant"
    value: Any

    def __post_init__(self) -> None:
        _json_value(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "value": self.value}

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "Constant":
        return cls(data.get("value"))


@dataclass(frozen=True, slots=True)
class ColumnRef(RuleNode):
    kind: ClassVar[str] = "column_ref"
    table_id: str
    column_id: str

    def __post_init__(self) -> None:
        _identifier(self.table_id, "table_id")
        _identifier(self.column_id, "column_id")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "table_id": self.table_id, "column_id": self.column_id}

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "ColumnRef":
        return cls(table_id=data["table_id"], column_id=data["column_id"])


@dataclass(frozen=True, slots=True)
class StateRef(RuleNode):
    kind: ClassVar[str] = "state_ref"
    state_id: str

    def __post_init__(self) -> None:
        _identifier(self.state_id, "state_id")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "state_id": self.state_id}

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "StateRef":
        return cls(state_id=data["state_id"])


@dataclass(frozen=True, slots=True)
class Compare(RuleNode):
    kind: ClassVar[str] = "compare"
    operator: str
    left: RuleNode
    right: RuleNode
    _operators: ClassVar[frozenset[str]] = frozenset({"=", "==", "!=", "<", "<=", ">", ">="})

    def __post_init__(self) -> None:
        if self.operator not in self._operators:
            raise ValueError(f"unsupported comparison operator: {self.operator!r}")
        if not isinstance(self.left, RuleNode) or not isinstance(self.right, RuleNode):
            raise TypeError("Compare operands must be RuleNode values")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "operator": self.operator, "left": self.left.to_dict(), "right": self.right.to_dict()}

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "Compare":
        return cls(operator=data["operator"], left=RuleNode.from_dict(data["left"]), right=RuleNode.from_dict(data["right"]))


@dataclass(frozen=True, slots=True)
class And(RuleNode):
    kind: ClassVar[str] = "and"
    operands: tuple[RuleNode, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operands, tuple) or not self.operands:
            raise ValueError("And requires at least one operand")
        if not all(isinstance(item, RuleNode) for item in self.operands):
            raise TypeError("And operands must be RuleNode values")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "operands": [item.to_dict() for item in self.operands]}

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "And":
        return cls(operands=tuple(RuleNode.from_dict(item) for item in data["operands"]))


@dataclass(frozen=True, slots=True)
class Or(RuleNode):
    kind: ClassVar[str] = "or"
    operands: tuple[RuleNode, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operands, tuple) or not self.operands:
            raise ValueError("Or requires at least one operand")
        if not all(isinstance(item, RuleNode) for item in self.operands):
            raise TypeError("Or operands must be RuleNode values")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "operands": [item.to_dict() for item in self.operands]}

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "Or":
        return cls(operands=tuple(RuleNode.from_dict(item) for item in data["operands"]))


@dataclass(frozen=True, slots=True)
class Not(RuleNode):
    kind: ClassVar[str] = "not"
    operand: RuleNode

    def __post_init__(self) -> None:
        if not isinstance(self.operand, RuleNode):
            raise TypeError("Not operand must be a RuleNode")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "operand": self.operand.to_dict()}

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "Not":
        return cls(operand=RuleNode.from_dict(data["operand"]))


@dataclass(frozen=True, slots=True)
class Add(RuleNode):
    kind: ClassVar[str] = "add"
    left: RuleNode
    right: RuleNode

    def __post_init__(self) -> None:
        if not isinstance(self.left, RuleNode) or not isinstance(self.right, RuleNode):
            raise TypeError("Add operands must be RuleNode values")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "left": self.left.to_dict(), "right": self.right.to_dict()}

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "Add":
        return cls(left=RuleNode.from_dict(data["left"]), right=RuleNode.from_dict(data["right"]))


@dataclass(frozen=True, slots=True)
class Multiply(RuleNode):
    kind: ClassVar[str] = "multiply"
    left: RuleNode
    right: RuleNode

    def __post_init__(self) -> None:
        if not isinstance(self.left, RuleNode) or not isinstance(self.right, RuleNode):
            raise TypeError("Multiply operands must be RuleNode values")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "left": self.left.to_dict(), "right": self.right.to_dict()}

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "Multiply":
        return cls(left=RuleNode.from_dict(data["left"]), right=RuleNode.from_dict(data["right"]))


@dataclass(frozen=True, slots=True)
class Aggregate(RuleNode):
    kind: ClassVar[str] = "aggregate"
    function: str
    table_id: str
    value: RuleNode | None = None
    predicate: RuleNode | None = None
    window_seconds: int | None = None
    relation_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.function not in {"count", "sum", "mean", "max", "min"}:
            raise ValueError(f"unsupported aggregate function: {self.function!r}")
        _identifier(self.table_id, "table_id")
        if self.value is not None and not isinstance(self.value, RuleNode):
            raise TypeError("aggregate value must be a RuleNode")
        if self.predicate is not None and not isinstance(self.predicate, RuleNode):
            raise TypeError("aggregate predicate must be a RuleNode")
        if self.window_seconds is not None and (isinstance(self.window_seconds, bool) or self.window_seconds < 0):
            raise ValueError("window_seconds must be non-negative")
        if not isinstance(self.relation_path, tuple) or not all(isinstance(item, str) and item for item in self.relation_path):
            raise TypeError("relation_path must be a tuple of column IDs")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "function": self.function, "table_id": self.table_id, "value": None if self.value is None else self.value.to_dict(), "predicate": None if self.predicate is None else self.predicate.to_dict(), "window_seconds": self.window_seconds, "relation_path": list(self.relation_path)}

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "Aggregate":
        return cls(function=data["function"], table_id=data["table_id"], value=None if data.get("value") is None else RuleNode.from_dict(data["value"]), predicate=None if data.get("predicate") is None else RuleNode.from_dict(data["predicate"]), window_seconds=data.get("window_seconds"), relation_path=tuple(data.get("relation_path", ())))


@dataclass(frozen=True, slots=True)
class Exists(RuleNode):
    kind: ClassVar[str] = "exists"
    table_id: str
    predicate: RuleNode | None = None
    relation_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.table_id, "table_id")
        if self.predicate is not None and not isinstance(self.predicate, RuleNode):
            raise TypeError("exists predicate must be a RuleNode")
        if not isinstance(self.relation_path, tuple) or not all(isinstance(item, str) and item for item in self.relation_path):
            raise TypeError("relation_path must be a tuple of column IDs")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "table_id": self.table_id, "predicate": None if self.predicate is None else self.predicate.to_dict(), "relation_path": list(self.relation_path)}

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "Exists":
        return cls(table_id=data["table_id"], predicate=None if data.get("predicate") is None else RuleNode.from_dict(data["predicate"]), relation_path=tuple(data.get("relation_path", ())))


@dataclass(frozen=True, slots=True)
class Lag(RuleNode):
    kind: ClassVar[str] = "lag"
    source: RuleNode
    steps: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.source, RuleNode):
            raise TypeError("Lag source must be a RuleNode")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("Lag steps must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "source": self.source.to_dict(), "steps": self.steps}

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "Lag":
        return cls(source=RuleNode.from_dict(data["source"]), steps=data.get("steps", 1))


@dataclass(frozen=True, slots=True)
class StateTransition(RuleNode):
    kind: ClassVar[str] = "state_transition"
    state_id: str
    target_state: str
    condition: RuleNode | None = None

    def __post_init__(self) -> None:
        _identifier(self.state_id, "state_id")
        _identifier(self.target_state, "target_state")
        if self.condition is not None and not isinstance(self.condition, RuleNode):
            raise TypeError("StateTransition condition must be a RuleNode")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "state_id": self.state_id, "target_state": self.target_state, "condition": None if self.condition is None else self.condition.to_dict()}

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "StateTransition":
        return cls(state_id=data["state_id"], target_state=data["target_state"], condition=None if data.get("condition") is None else RuleNode.from_dict(data["condition"]))


@dataclass(frozen=True, slots=True)
class RuleAction:
    target: str
    operation: str
    value: RuleNode | Any

    def __post_init__(self) -> None:
        _identifier(self.target, "action target")
        if self.operation not in {"multiply", "add", "set", "transition"}:
            raise ValueError(f"unsupported rule action operation: {self.operation!r}")
        if isinstance(self.value, RuleNode):
            return
        _json_value(self.value, "action value")

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "operation": self.operation, "value": self.value.to_dict() if isinstance(self.value, RuleNode) else self.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuleAction":
        value = data.get("value")
        if isinstance(value, Mapping):
            value = RuleNode.from_dict(value)
        return cls(target=data["target"], operation=data["operation"], value=value)


@dataclass(frozen=True, slots=True)
class RulePlan:
    rule_id: str
    condition: RuleNode
    actions: tuple[RuleAction, ...]
    seed: int
    version: str = "v1"
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.rule_id, "rule_id")
        _identifier(self.version, "version")
        if not isinstance(self.condition, RuleNode):
            raise TypeError("condition must be a RuleNode")
        if not isinstance(self.actions, tuple) or not self.actions or not all(isinstance(item, RuleAction) for item in self.actions):
            raise TypeError("actions must be a non-empty tuple of RuleAction")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("rule seed must be a non-negative integer")
        if not isinstance(self.parameters, tuple):
            raise TypeError("parameters must be a tuple")
        normalized = []
        for key, value in self.parameters:
            _identifier(key, "parameter name")
            _json_value(value, "parameter value")
            normalized.append((key, value))
        if len({key for key, _ in normalized}) != len(normalized):
            raise ValueError("rule parameter names must be unique")
        object.__setattr__(self, "parameters", tuple(sorted(normalized)))

    def to_dict(self) -> dict[str, Any]:
        return {"rule_id": self.rule_id, "version": self.version, "seed": self.seed, "condition": self.condition.to_dict(), "actions": [item.to_dict() for item in self.actions], "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RulePlan":
        return cls(rule_id=data["rule_id"], version=data.get("version", "v1"), seed=data.get("seed", 0), condition=RuleNode.from_dict(data["condition"]), actions=tuple(RuleAction.from_dict(item) for item in data.get("actions", ())), parameters=tuple(data.get("parameters", {}).items()))


ProcessPlan = RulePlan


__all__ = [
    "RuleNode", "Constant", "ColumnRef", "StateRef", "Compare", "And", "Or", "Not",
    "Add", "Multiply", "Aggregate", "Exists", "Lag", "StateTransition",
    "RuleAction", "RulePlan", "ProcessPlan",
]
