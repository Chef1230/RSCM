"""Safe interpreter for the closed Rule/process AST."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

from rdb_prior.process.model import (
    Add,
    Aggregate,
    And,
    ColumnRef,
    Compare,
    Constant,
    Exists,
    Lag,
    Multiply,
    Not,
    Or,
    RuleNode,
    RulePlan,
    StateRef,
    StateTransition,
)


@dataclass(frozen=True, slots=True)
class RuleEvaluationContext:
    """Read-only values visible to a rule.

    ``row_index`` is supplied by the executor for one entity/parent row.  A
    relation path is a tuple of FK column IDs; this keeps the AST independent
    of physical table object implementations.
    """

    tables: Mapping[str, object]
    states: Mapping[str, np.ndarray] = field(default_factory=dict)
    cutoff_time: int | float | None = None
    time_column_ids: Mapping[str, str] = field(default_factory=dict)

    def table_columns(self, table_id: str) -> Mapping[str, np.ndarray]:
        table = self.tables[table_id]
        return table.columns if hasattr(table, "columns") else table


@dataclass(frozen=True, slots=True)
class RuleEffect:
    """Vectorized effects of a rule for each owner entity."""

    intensity_multiplier: np.ndarray
    intensity_addition: np.ndarray
    attribute_offset: np.ndarray
    state_overrides: tuple[str | None, ...]
    matched: np.ndarray


class RuleProcessExecutor:
    """Evaluate RulePlan values without ``eval``, ``exec`` or user code."""

    def evaluate(
        self,
        node: RuleNode,
        context: RuleEvaluationContext,
        *,
        row_index: int | None = None,
    ) -> Any:
        if not isinstance(node, RuleNode):
            raise TypeError("rule expressions must be RuleNode values")
        return self._evaluate(node, context, row_index)

    def evaluate_condition_per_row(
        self,
        plan: RulePlan,
        context: RuleEvaluationContext,
        row_count: int,
    ) -> np.ndarray:
        if row_count < 0:
            raise ValueError("row_count must be non-negative")
        return np.asarray(
            [bool(self._evaluate(plan.condition, context, index)) for index in range(row_count)],
            dtype=bool,
        )

    def effects(
        self,
        plan: RulePlan,
        context: RuleEvaluationContext,
        *,
        row_count: int,
        entity_indices: np.ndarray | None = None,
        attribute_column_id: str | None = None,
    ) -> RuleEffect:
        """Apply all actions to owner rows and return pure numpy effects."""
        matched = self.evaluate_condition_per_row(plan, context, row_count)
        multiplier = np.ones(row_count, dtype=np.float64)
        addition = np.zeros(row_count, dtype=np.float64)
        attribute = np.zeros(row_count, dtype=np.float64)
        overrides: list[str | None] = [None] * row_count
        for action in plan.actions:
            target = action.target
            for index in np.flatnonzero(matched):
                value = action.value
                if isinstance(value, RuleNode):
                    value = self._evaluate(value, context, int(index))
                if isinstance(value, np.ndarray):
                    value = value[index]
                if action.operation == "transition" or target in {"process_state", "workflow_transition", "state"}:
                    if action.operation not in {"transition", "set"}:
                        raise ValueError("state actions must use transition or set")
                    overrides[int(index)] = str(value)
                    continue
                numeric = float(value)
                if target in {"event_intensity", "future_intensity", "intensity"}:
                    if action.operation == "multiply":
                        multiplier[index] *= numeric
                    elif action.operation == "add":
                        addition[index] += numeric
                    elif action.operation == "set":
                        multiplier[index] = numeric
                    else:
                        raise ValueError(f"unsupported intensity operation: {action.operation}")
                    continue
                if target == "event_attribute" or target.startswith("event_attribute:"):
                    if attribute_column_id is not None and target.startswith("event_attribute:") and not target.endswith(attribute_column_id):
                        continue
                    if action.operation == "add":
                        attribute[index] += numeric
                    elif action.operation == "multiply":
                        attribute[index] += numeric - 1.0
                    elif action.operation == "set":
                        attribute[index] = numeric
                    else:
                        raise ValueError(f"unsupported attribute operation: {action.operation}")
        return RuleEffect(multiplier, addition, attribute, tuple(overrides), matched)

    def _evaluate(self, node: RuleNode, context: RuleEvaluationContext, row_index: int | None) -> Any:
        if isinstance(node, Constant):
            return node.value
        if isinstance(node, ColumnRef):
            values = np.asarray(context.table_columns(node.table_id)[node.column_id])
            return values if row_index is None else values[row_index]
        if isinstance(node, StateRef):
            values = np.asarray(context.states[node.state_id])
            return values if row_index is None else values[row_index]
        if isinstance(node, Compare):
            left = self._evaluate(node.left, context, row_index)
            right = self._evaluate(node.right, context, row_index)
            return self._compare(node.operator, left, right)
        if isinstance(node, And):
            result = True
            for operand in node.operands:
                result = np.logical_and(result, self._evaluate(operand, context, row_index))
            return result
        if isinstance(node, Or):
            result = False
            for operand in node.operands:
                result = np.logical_or(result, self._evaluate(operand, context, row_index))
            return result
        if isinstance(node, Not):
            return np.logical_not(self._evaluate(node.operand, context, row_index))
        if isinstance(node, Add):
            return np.asarray(self._evaluate(node.left, context, row_index)) + np.asarray(self._evaluate(node.right, context, row_index))
        if isinstance(node, Multiply):
            return np.asarray(self._evaluate(node.left, context, row_index)) * np.asarray(self._evaluate(node.right, context, row_index))
        if isinstance(node, Aggregate):
            return self._aggregate(node, context, row_index)
        if isinstance(node, Exists):
            return self._exists(node, context, row_index)
        if isinstance(node, Lag):
            values = np.asarray(self._evaluate(node.source, context, None))
            if values.ndim == 0:
                return values.item()
            shifted = np.empty_like(values)
            shifted[: node.steps] = 0
            shifted[node.steps :] = values[: -node.steps]
            return shifted if row_index is None else shifted[row_index]
        if isinstance(node, StateTransition):
            if node.condition is None:
                return node.target_state
            condition = self._evaluate(node.condition, context, row_index)
            return node.target_state if bool(condition) else None
        raise TypeError(f"unsupported rule node: {type(node).__name__}")

    @staticmethod
    def _compare(operator: str, left: Any, right: Any) -> Any:
        if operator in {"=", "=="}:
            return np.equal(left, right)
        if operator == "!=":
            return np.not_equal(left, right)
        if operator == "<":
            return np.less(left, right)
        if operator == "<=":
            return np.less_equal(left, right)
        if operator == ">":
            return np.greater(left, right)
        if operator == ">=":
            return np.greater_equal(left, right)
        raise ValueError(f"unsupported comparison operator: {operator!r}")

    def _candidate_rows(
        self,
        table_id: str,
        context: RuleEvaluationContext,
        row_index: int | None,
        relation_path: tuple[str, ...],
    ) -> np.ndarray:
        columns = context.table_columns(table_id)
        count = len(next(iter(columns.values())))
        rows = np.arange(count, dtype=np.int64)
        if relation_path and row_index is not None:
            fk_column = relation_path[-1]
            if fk_column not in columns:
                raise KeyError(f"relation path column {fk_column!r} is absent from {table_id!r}")
            rows = rows[np.asarray(columns[fk_column]) == row_index]
        return rows

    def _aggregate(self, node: Aggregate, context: RuleEvaluationContext, row_index: int | None) -> Any:
        rows = self._candidate_rows(node.table_id, context, row_index, node.relation_path)
        columns = context.table_columns(node.table_id)
        if node.window_seconds is not None and context.cutoff_time is not None:
            time_id = context.time_column_ids.get(node.table_id)
            if time_id is not None and time_id in columns:
                lower = context.cutoff_time - node.window_seconds
                times = np.asarray(columns[time_id])
                rows = rows[(times[rows] > lower) & (times[rows] <= context.cutoff_time)]
        if node.predicate is not None:
            rows = np.asarray([index for index in rows if bool(self._evaluate(node.predicate, context, int(index)))] , dtype=np.int64)
        if node.function == "count":
            return int(len(rows))
        if node.value is None:
            raise ValueError(f"aggregate {node.function!r} requires a value expression")
        values = np.asarray([self._evaluate(node.value, context, int(index)) for index in rows], dtype=np.float64)
        if len(values) == 0:
            return 0.0
        return {"sum": np.sum, "mean": np.mean, "max": np.max, "min": np.min}[node.function](values).item()

    def _exists(self, node: Exists, context: RuleEvaluationContext, row_index: int | None) -> bool:
        rows = self._candidate_rows(node.table_id, context, row_index, node.relation_path)
        if node.predicate is None:
            return bool(len(rows))
        return any(bool(self._evaluate(node.predicate, context, int(index))) for index in rows)


__all__ = ["RuleEvaluationContext", "RuleEffect", "RuleProcessExecutor"]
