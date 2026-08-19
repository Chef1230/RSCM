"""Serializable composite relational-classification task DSL.

The composite DSL describes a binary relational-classification task purely in
terms of:

* FK paths from a target table to one or more source tables,
* a time window per aggregate, expressed as seconds relative to each target
  row's cutoff, and
* an aggregate operator applied inside that window after optional row
  predicates.

A :class:`TaskPlan` using the ``RELATIONAL_CLASSIFICATION`` mechanism carries a
:class:`CompositeTaskSpec`.  ``mechanism_labels()`` recomputes the exact labels
from the spec plus the database alone (no stored label array), so a fixed
seed and a persisted spec are sufficient to reproduce labels exactly.

This module is intentionally a leaf: it depends only on the standard library,
so ``rdb_prior.task.model`` can import it without creating an import cycle.
:class:`AggregateOperator` lives here (it is re-exported by ``model`` for
backward compatibility with earlier import paths).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class AggregateOperator(str, Enum):
    COUNT = "count"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    COUNT_DISTINCT = "count_distinct"


class CompositeFamily(str, Enum):
    FILTERED_AGGREGATE = "filtered_aggregate"
    COUNT_DISTINCT = "count_distinct"
    QUANTIFIED_EVENT = "quantified_event"
    MULTI_SOURCE = "multi_source"
    MULTI_HOP_FILTERED = "multi_hop_filtered"
    HISTORY_CONDITIONED_FUTURE = "history_conditioned_future"


class CompareOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class CombineOperator(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


class LabelOperator(str, Enum):
    GT = "gt"
    GE = "ge"


# Aggregate operators that read a value column (COUNT only counts rows).
_VALUE_AGGREGATE_OPERATORS = frozenset(
    {
        AggregateOperator.COUNT_DISTINCT,
        AggregateOperator.SUM,
        AggregateOperator.MEAN,
        AggregateOperator.MIN,
        AggregateOperator.MAX,
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PredicateSpec:
    """One row predicate applied inside an aggregate window.

    The first version combines all predicates of an aggregate with logical
    AND.  ``value`` is a raw scalar (str/int/float) that must live inside the
    composite spec — it must not be stored in ``TaskPlan.parameters`` because
    parameter values are coerced to float, which would corrupt string
    predicates.
    """

    column_id: str
    operator: CompareOperator
    value: int | float | str

    def __post_init__(self) -> None:
        _identifier("column_id", self.column_id)
        if not isinstance(self.operator, CompareOperator):
            raise TypeError("operator must be CompareOperator")
        if isinstance(self.value, bool) or not isinstance(
            self.value, (int, float, str)
        ):
            raise TypeError("value must be a non-boolean scalar")
        if isinstance(self.value, str) and not self.value:
            raise ValueError("value must not be an empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_id": self.column_id,
            "operator": self.operator.value,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PredicateSpec:
        return cls(
            column_id=data["column_id"],
            operator=CompareOperator(data["operator"]),
            value=data["value"],
        )

    def canonical(self) -> tuple[Any, ...]:
        return (
            "PredicateSpec",
            self.column_id,
            self.operator.value,
            self.value,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AggregateSpec:
    """One filtered temporal aggregate over one FK path.

    Windows are relative seconds from each target row's cutoff:

    * history window: ``window_start < 0`` and ``window_end <= 0``
    * future window:  ``window_start >= 0`` and ``window_end > 0``

    The window uses inclusive bounds ``[cutoff + window_start,
    cutoff + window_end]``.  ``COUNT`` does not read a value column;
    ``COUNT_DISTINCT``/``SUM``/``MEAN``/``MIN``/``MAX`` require one.  All
    predicates combine with AND.  The path must be a contiguous FK walk from
    the target table to ``source_table_id`` (schema-level continuity is
    verified by task validation).
    """

    source_table_id: str
    required_path: tuple[str, ...]
    time_column_id: str
    window_start: int
    window_end: int
    operator: AggregateOperator
    value_column_id: str | None = None
    predicates: tuple[PredicateSpec, ...] = ()

    def __post_init__(self) -> None:
        _identifier("source_table_id", self.source_table_id)
        _identifier("time_column_id", self.time_column_id)
        if not isinstance(self.required_path, tuple) or not self.required_path:
            raise ValueError("required_path must be a non-empty tuple")
        for foreign_key_id in self.required_path:
            _identifier("required path foreign key", foreign_key_id)
        for name in ("window_start", "window_end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        if self.window_end <= 0:
            if not (self.window_start < 0 and self.window_end <= 0):
                raise ValueError(
                    "history windows must satisfy window_start < 0 <= window_end <= 0"
                )
        elif not (self.window_start >= 0 and self.window_end > 0):
            raise ValueError(
                "future windows must satisfy 0 <= window_start < window_end"
            )
        if not isinstance(self.operator, AggregateOperator):
            raise TypeError("operator must be AggregateOperator")
        if (
            self.operator in _VALUE_AGGREGATE_OPERATORS
            and self.value_column_id is None
        ):
            raise ValueError(f"{self.operator.value} requires value_column_id")
        if self.operator is AggregateOperator.COUNT:
            if self.value_column_id is not None:
                raise ValueError("COUNT does not use value_column_id")
        if self.value_column_id is not None:
            _identifier("value_column_id", self.value_column_id)
        if not isinstance(self.predicates, tuple) or not all(
            isinstance(predicate, PredicateSpec)
            for predicate in self.predicates
        ):
            raise TypeError("predicates must be a tuple of PredicateSpec")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_table_id": self.source_table_id,
            "required_path": list(self.required_path),
            "time_column_id": self.time_column_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "operator": self.operator.value,
            "value_column_id": self.value_column_id,
            "predicates": [predicate.to_dict() for predicate in self.predicates],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AggregateSpec:
        return cls(
            source_table_id=data["source_table_id"],
            required_path=tuple(data["required_path"]),
            time_column_id=data["time_column_id"],
            window_start=data["window_start"],
            window_end=data["window_end"],
            operator=AggregateOperator(data["operator"]),
            value_column_id=data.get("value_column_id"),
            predicates=tuple(
                PredicateSpec.from_dict(item)
                for item in data.get("predicates", ())
            ),
        )

    def canonical(self) -> tuple[Any, ...]:
        return (
            "AggregateSpec",
            self.source_table_id,
            self.required_path,
            self.time_column_id,
            self.window_start,
            self.window_end,
            self.operator.value,
            self.value_column_id,
            tuple(predicate.canonical() for predicate in self.predicates),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeTaskSpec:
    """The complete specification of one composite relational task.

    ``label_aggregates`` each produce one per-target-row score; the scores are
    combined with ``combine_operator``, then thresholded with
    ``label_operator``/``label_threshold`` into binary labels.  When
    ``eligibility_*`` fields are set, the eligibility aggregate is evaluated
    first (as a history gate) and ineligible rows receive ``-1`` labels.
    """

    family: CompositeFamily
    label_aggregates: tuple[AggregateSpec, ...]
    combine_operator: CombineOperator
    label_operator: LabelOperator
    label_threshold: float
    eligibility_aggregate: AggregateSpec | None = None
    eligibility_operator: LabelOperator | None = None
    eligibility_threshold: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.family, CompositeFamily):
            raise TypeError("family must be CompositeFamily")
        if not isinstance(self.label_aggregates, tuple) or not self.label_aggregates:
            raise ValueError("label_aggregates must be a non-empty tuple")
        if not all(
            isinstance(aggregate, AggregateSpec)
            for aggregate in self.label_aggregates
        ):
            raise TypeError("label_aggregates must contain AggregateSpec")
        if not isinstance(self.combine_operator, CombineOperator):
            raise TypeError("combine_operator must be CombineOperator")
        if not isinstance(self.label_operator, LabelOperator):
            raise TypeError("label_operator must be LabelOperator")
        if isinstance(self.label_threshold, bool) or not isinstance(
            self.label_threshold, (int, float)
        ):
            raise TypeError("label_threshold must be numeric")
        eligibility = (
            self.eligibility_aggregate,
            self.eligibility_operator,
            self.eligibility_threshold,
        )
        if any(item is None for item in eligibility) and not all(
            item is None for item in eligibility
        ):
            raise ValueError(
                "eligibility_aggregate, eligibility_operator and "
                "eligibility_threshold must all be set or all be None"
            )
        if self.eligibility_aggregate is not None:
            if not isinstance(self.eligibility_aggregate, AggregateSpec):
                raise TypeError(
                    "eligibility_aggregate must be AggregateSpec or None"
                )
            if not isinstance(self.eligibility_operator, LabelOperator):
                raise TypeError(
                    "eligibility_operator must be LabelOperator or None"
                )
            if isinstance(self.eligibility_threshold, bool) or not isinstance(
                self.eligibility_threshold, (int, float)
            ):
                raise TypeError(
                    "eligibility_threshold must be numeric or None"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "label_aggregates": [
                aggregate.to_dict() for aggregate in self.label_aggregates
            ],
            "combine_operator": self.combine_operator.value,
            "label_operator": self.label_operator.value,
            "label_threshold": self.label_threshold,
            "eligibility_aggregate": (
                None
                if self.eligibility_aggregate is None
                else self.eligibility_aggregate.to_dict()
            ),
            "eligibility_operator": (
                None
                if self.eligibility_operator is None
                else self.eligibility_operator.value
            ),
            "eligibility_threshold": self.eligibility_threshold,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompositeTaskSpec:
        eligibility_aggregate = data.get("eligibility_aggregate")
        eligibility_operator = data.get("eligibility_operator")
        eligibility_threshold = data.get("eligibility_threshold")
        if eligibility_aggregate is not None:
            eligibility_aggregate = AggregateSpec.from_dict(eligibility_aggregate)
        return cls(
            family=CompositeFamily(data["family"]),
            label_aggregates=tuple(
                AggregateSpec.from_dict(item)
                for item in data["label_aggregates"]
            ),
            combine_operator=CombineOperator(data["combine_operator"]),
            label_operator=LabelOperator(data["label_operator"]),
            label_threshold=data["label_threshold"],
            eligibility_aggregate=eligibility_aggregate,
            eligibility_operator=(
                None
                if eligibility_operator is None
                else LabelOperator(eligibility_operator)
            ),
            eligibility_threshold=eligibility_threshold,
        )

    def canonical(self) -> tuple[Any, ...]:
        return (
            "CompositeTaskSpec",
            self.family.value,
            tuple(aggregate.canonical() for aggregate in self.label_aggregates),
            self.combine_operator.value,
            self.label_operator.value,
            self.label_threshold,
            (
                None
                if self.eligibility_aggregate is None
                else self.eligibility_aggregate.canonical()
            ),
            (
                None
                if self.eligibility_operator is None
                else self.eligibility_operator.value
            ),
            self.eligibility_threshold,
        )


def _identifier(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


__all__ = [
    "AggregateOperator",
    "CompositeFamily",
    "CompareOperator",
    "CombineOperator",
    "LabelOperator",
    "PredicateSpec",
    "AggregateSpec",
    "CompositeTaskSpec",
]
