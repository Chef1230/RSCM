"""Deterministic sampling of random decision-tree and forest priors."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from rdb_prior.generation.trees.model import ForestPlan, TreeNodePlan, TreePlan


def _bounds(name: str, values: tuple[int, int], minimum: int) -> tuple[int, int]:
    if not isinstance(values, tuple) or len(values) != 2:
        raise TypeError(f"{name} must be an integer pair")
    lower, upper = values
    if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        raise TypeError(f"{name} must contain integers")
    if lower < minimum or upper < lower:
        raise ValueError(f"{name} must be within valid bounds")
    return lower, upper


def sample_forest(
    *,
    forest_id: str,
    feature_refs: Iterable[str],
    rng: np.random.Generator,
    tree_count: tuple[int, int] = (1, 1),
    depth: tuple[int, int] = (2, 4),
    threshold_strategy: str = "extra",
) -> ForestPlan:
    """Sample a prior directly; no generated labels are inspected or fitted."""
    references = tuple(sorted(set(feature_refs)))
    if not references or not all(isinstance(item, str) and item for item in references):
        raise ValueError("feature_refs must contain non-empty strings")
    lower_trees, upper_trees = _bounds("tree_count", tree_count, 1)
    lower_depth, upper_depth = _bounds("depth", depth, 1)
    if threshold_strategy not in {"random", "extra"}:
        raise ValueError("threshold_strategy must be 'random' or 'extra'")

    count = int(rng.integers(lower_trees, upper_trees + 1))
    trees = tuple(
        _sample_tree(
            tree_id=f"{forest_id}_T{index:03d}",
            feature_refs=references,
            rng=rng,
            min_depth=lower_depth,
            max_depth=int(rng.integers(lower_depth, upper_depth + 1)),
            threshold_strategy=threshold_strategy,
        )
        for index in range(count)
    )
    return ForestPlan(
        forest_id=forest_id,
        trees=trees,
        output_scale=float(rng.uniform(0.65, 1.35)),
    )


def _sample_tree(
    *,
    tree_id: str,
    feature_refs: tuple[str, ...],
    rng: np.random.Generator,
    min_depth: int,
    max_depth: int,
    threshold_strategy: str,
) -> TreePlan:
    nodes: list[TreeNodePlan | None] = []

    def build(level: int) -> int:
        index = len(nodes)
        nodes.append(None)
        should_leaf = level >= max_depth or (
            level >= min_depth and float(rng.random()) < 0.35
        )
        if should_leaf:
            nodes[index] = TreeNodePlan(
                feature_ref=None,
                threshold=None,
                left_index=None,
                right_index=None,
                leaf_value=float(rng.normal(0.0, 1.0)),
            )
            return index
        if threshold_strategy == "extra":
            threshold = float(rng.uniform(-2.25, 2.25))
        else:
            threshold = float(rng.normal(0.0, 1.0))
        feature = feature_refs[int(rng.integers(0, len(feature_refs)))]
        left = build(level + 1)
        right = build(level + 1)
        nodes[index] = TreeNodePlan(
            feature_ref=feature,
            threshold=threshold,
            left_index=left,
            right_index=right,
            leaf_value=None,
        )
        return index

    build(0)
    return TreePlan(
        tree_id=tree_id,
        nodes=tuple(node for node in nodes if node is not None),
    )


__all__ = ["sample_forest"]
