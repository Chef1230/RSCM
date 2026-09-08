"""Safe execution of sampled tree priors against anonymous numeric contexts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rdb_prior.generation.trees.model import ForestPlan, TreePlan


def evaluate_tree(
    tree: TreePlan,
    features: Mapping[str, np.ndarray | float | int],
    *,
    row_count: int | None = None,
) -> np.ndarray:
    """Evaluate one indexed tree; missing feature references are explicit errors."""
    if not isinstance(tree, TreePlan):
        raise TypeError("tree must be TreePlan")
    arrays, rows = _feature_arrays(features, row_count)
    result = np.empty(rows, dtype=np.float64)
    pending: list[tuple[int, np.ndarray]] = [
        (0, np.arange(rows, dtype=np.int64))
    ]
    while pending:
        node_index, indices = pending.pop()
        node = tree.nodes[node_index]
        if node.is_leaf:
            assert node.leaf_value is not None
            result[indices] = node.leaf_value
            continue
        assert node.feature_ref is not None
        assert node.threshold is not None
        assert node.left_index is not None
        assert node.right_index is not None
        if node.feature_ref not in arrays:
            raise KeyError(
                f"tree {tree.tree_id!r} references unavailable feature "
                f"{node.feature_ref!r}"
            )
        left_mask = arrays[node.feature_ref][indices] <= node.threshold
        pending.append((node.right_index, indices[~left_mask]))
        pending.append((node.left_index, indices[left_mask]))
    return result


def evaluate_forest(
    forest: ForestPlan,
    features: Mapping[str, np.ndarray | float | int],
    *,
    row_count: int | None = None,
) -> np.ndarray:
    if not isinstance(forest, ForestPlan):
        raise TypeError("forest must be ForestPlan")
    values = [
        evaluate_tree(tree, features, row_count=row_count)
        for tree in forest.trees
    ]
    return forest.output_scale * np.mean(np.column_stack(values), axis=1)


def _feature_arrays(
    features: Mapping[str, np.ndarray | float | int],
    row_count: int | None,
) -> tuple[dict[str, np.ndarray], int]:
    if not isinstance(features, Mapping):
        raise TypeError("features must be a mapping")
    inferred = row_count
    for value in features.values():
        array = np.asarray(value)
        if array.ndim == 1 and array.size:
            inferred = int(array.shape[0])
            break
    if inferred is None or inferred < 1:
        raise ValueError("row_count must be positive when features are scalar")
    arrays: dict[str, np.ndarray] = {}
    for name, value in features.items():
        if not isinstance(name, str) or not name:
            raise ValueError("feature names must be non-empty strings")
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(inferred, float(array), dtype=np.float64)
        if array.ndim != 1 or array.shape[0] != inferred:
            raise ValueError(f"feature {name!r} does not align with row_count")
        arrays[name] = np.nan_to_num(array, nan=0.0, posinf=8.0, neginf=-8.0)
    return arrays, inferred


__all__ = ["evaluate_tree", "evaluate_forest"]
