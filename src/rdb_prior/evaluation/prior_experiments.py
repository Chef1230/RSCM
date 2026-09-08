"""Small, serializable experiment manifests for prior-mixture studies.

This module intentionally does not train a model.  It fixes the comparison
cells, data/task budgets and seeds so any trainer can write results back in a
common format without coupling generation code to a training framework.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


_COMPONENTS = frozenset({"scm", "tree", "temporal", "rule", "nuisance"})


@dataclass(frozen=True, slots=True, kw_only=True)
class PriorExperimentCell:
    cell_id: str
    components: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id:
            raise ValueError("cell_id must be non-empty")
        if not isinstance(self.components, tuple) or not self.components:
            raise ValueError("components must be a non-empty tuple")
        if len(set(self.components)) != len(self.components):
            raise ValueError("components must be unique")
        unknown = set(self.components) - _COMPONENTS
        if unknown:
            raise ValueError(f"unknown prior components: {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return {"cell_id": self.cell_id, "components": list(self.components)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PriorExperimentCell":
        return cls(cell_id=data["cell_id"], components=tuple(data["components"]))


@dataclass(frozen=True, slots=True, kw_only=True)
class PriorExperimentPlan:
    """A fixed comparison grid, independent of a training implementation."""

    experiment_id: str
    cells: tuple[PriorExperimentCell, ...]
    database_count: int
    tasks_per_database: int
    training_budget: int
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id:
            raise ValueError("experiment_id must be non-empty")
        if not isinstance(self.cells, tuple) or not self.cells:
            raise ValueError("cells must be non-empty")
        if len({item.cell_id for item in self.cells}) != len(self.cells):
            raise ValueError("cell IDs must be unique")
        for name in ("database_count", "tasks_per_database", "training_budget"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.seeds, tuple) or not self.seeds:
            raise ValueError("seeds must be non-empty")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in self.seeds):
            raise ValueError("seeds must contain non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "prior_experiment_plan",
            "artifact_version": 1,
            "experiment_id": self.experiment_id,
            "cells": [item.to_dict() for item in self.cells],
            "database_count": self.database_count,
            "tasks_per_database": self.tasks_per_database,
            "training_budget": self.training_budget,
            "seeds": list(self.seeds),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PriorExperimentPlan":
        if data.get("artifact_type", "prior_experiment_plan") != "prior_experiment_plan":
            raise ValueError("unsupported experiment plan type")
        if data.get("artifact_version", 1) != 1:
            raise ValueError("unsupported experiment plan version")
        return cls(
            experiment_id=data["experiment_id"],
            cells=tuple(PriorExperimentCell.from_dict(item) for item in data["cells"]),
            database_count=data["database_count"],
            tasks_per_database=data["tasks_per_database"],
            training_budget=data["training_budget"],
            seeds=tuple(data["seeds"]),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PriorExperimentResult:
    """One train-prior × test-prior observation from a fixed plan."""

    train_cell_id: str
    test_cell_id: str
    seed: int
    metrics: tuple[tuple[str, float], ...]
    error_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.train_cell_id or not self.test_cell_id:
            raise ValueError("train_cell_id and test_cell_id must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be non-negative")
        if len({name for name, _value in self.metrics}) != len(self.metrics):
            raise ValueError("metric names must be unique")
        for name, value in self.metrics:
            if not isinstance(name, str) or not name or not isinstance(value, (int, float)):
                raise ValueError("metrics must contain non-empty numeric entries")
        if len(set(self.error_ids)) != len(self.error_ids):
            raise ValueError("error_ids must be unique")

    @property
    def metric_map(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.metrics}


def standard_mixture_plan(
    *,
    experiment_id: str,
    database_count: int,
    tasks_per_database: int,
    training_budget: int,
    seeds: tuple[int, ...],
) -> PriorExperimentPlan:
    """The fixed SCM/Tree/Temporal comparison grid described by PR11."""
    components = (
        ("scm",),
        ("tree",),
        ("temporal",),
        ("scm", "tree"),
        ("scm", "temporal"),
        ("tree", "temporal"),
        ("scm", "tree", "temporal"),
    )
    return PriorExperimentPlan(
        experiment_id=experiment_id,
        cells=tuple(
            PriorExperimentCell(
                cell_id="+".join(items), components=items
            )
            for items in components
        ),
        database_count=database_count,
        tasks_per_database=tasks_per_database,
        training_budget=training_budget,
        seeds=seeds,
    )


def summarize_prior_experiment(
    plan: PriorExperimentPlan,
    results: tuple[PriorExperimentResult, ...],
    *,
    metric: str,
) -> dict[str, Any]:
    """Produce a train×test mean-score matrix and pairwise error overlap."""
    cell_ids = {item.cell_id for item in plan.cells}
    buckets: dict[tuple[str, str], list[float]] = {}
    error_sets: dict[tuple[str, str, int], set[str]] = {}
    for result in results:
        if result.train_cell_id not in cell_ids or result.test_cell_id not in cell_ids:
            raise ValueError("result references a cell outside the experiment plan")
        if result.seed not in plan.seeds:
            raise ValueError("result seed is outside the experiment plan")
        if metric not in result.metric_map:
            continue
        buckets.setdefault((result.train_cell_id, result.test_cell_id), []).append(
            result.metric_map[metric]
        )
        error_sets[(result.train_cell_id, result.test_cell_id, result.seed)] = set(result.error_ids)
    matrix = {
        train: {
            test: (
                None if not buckets.get((train, test))
                else sum(buckets[(train, test)]) / len(buckets[(train, test)])
            )
            for test in sorted(cell_ids)
        }
        for train in sorted(cell_ids)
    }
    complementarity: dict[str, float] = {}
    for test in sorted(cell_ids):
        for left in sorted(cell_ids):
            for right in sorted(cell_ids):
                if left >= right:
                    continue
                overlaps = []
                for seed in plan.seeds:
                    first = error_sets.get((left, test, seed))
                    second = error_sets.get((right, test, seed))
                    if first is None or second is None:
                        continue
                    union = first | second
                    overlaps.append(0.0 if not union else 1.0 - len(first & second) / len(union))
                if overlaps:
                    complementarity[f"{test}:{left}|{right}"] = sum(overlaps) / len(overlaps)
    return {
        "metric": metric,
        "train_test_matrix": matrix,
        "error_complementarity": dict(sorted(complementarity.items())),
    }


def write_prior_experiment_plan(path: Path, plan: PriorExperimentPlan) -> Path:
    """Atomically persist a plan so runs can be reproduced by any trainer."""
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(plan, PriorExperimentPlan):
        raise TypeError("plan must be PriorExperimentPlan")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


__all__ = [
    "PriorExperimentCell",
    "PriorExperimentPlan",
    "PriorExperimentResult",
    "standard_mixture_plan",
    "summarize_prior_experiment",
    "write_prior_experiment_plan",
]
