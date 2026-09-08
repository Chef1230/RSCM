"""Immutable, JSON-serializable random decision-tree plans."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


def _identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True, kw_only=True)
class TreeNodePlan:
    """One indexed node in a sampled decision tree.

    A node is either a split (feature/threshold/children) or a finite-valued
    leaf. Indexed children make the prior compact and safe to serialize.
    """

    feature_ref: str | None
    threshold: float | None
    left_index: int | None
    right_index: int | None
    leaf_value: float | None

    def __post_init__(self) -> None:
        is_leaf = self.leaf_value is not None
        if is_leaf:
            if any(
                value is not None
                for value in (
                    self.feature_ref,
                    self.threshold,
                    self.left_index,
                    self.right_index,
                )
            ):
                raise ValueError("leaf nodes cannot define a split")
            if not isinstance(self.leaf_value, (int, float)) or isinstance(
                self.leaf_value, bool
            ) or not isfinite(float(self.leaf_value)):
                raise ValueError("leaf_value must be finite")
            return
        if not isinstance(self.feature_ref, str) or not self.feature_ref:
            raise ValueError("split nodes require feature_ref")
        if not isinstance(self.threshold, (int, float)) or isinstance(
            self.threshold, bool
        ) or not isfinite(float(self.threshold)):
            raise ValueError("split threshold must be finite")
        for name, value in (
            ("left_index", self.left_index),
            ("right_index", self.right_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def is_leaf(self) -> bool:
        return self.leaf_value is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_ref": self.feature_ref,
            "threshold": self.threshold,
            "left_index": self.left_index,
            "right_index": self.right_index,
            "leaf_value": self.leaf_value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TreeNodePlan":
        return cls(
            feature_ref=data.get("feature_ref"),
            threshold=data.get("threshold"),
            left_index=data.get("left_index"),
            right_index=data.get("right_index"),
            leaf_value=data.get("leaf_value"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TreePlan:
    tree_id: str
    nodes: tuple[TreeNodePlan, ...]

    def __post_init__(self) -> None:
        _identifier("tree_id", self.tree_id)
        if not isinstance(self.nodes, tuple) or not self.nodes:
            raise ValueError("nodes must be a non-empty tuple")
        if not all(isinstance(node, TreeNodePlan) for node in self.nodes):
            raise TypeError("nodes must contain TreeNodePlan values")
        self._validate_graph()

    def _validate_graph(self) -> None:
        count = len(self.nodes)
        visited: set[int] = set()
        active: set[int] = set()

        def visit(index: int) -> None:
            if index < 0 or index >= count:
                raise ValueError("tree child index is out of range")
            if index in active:
                raise ValueError("tree nodes must not contain cycles")
            if index in visited:
                return
            active.add(index)
            node = self.nodes[index]
            if not node.is_leaf:
                assert node.left_index is not None
                assert node.right_index is not None
                visit(node.left_index)
                visit(node.right_index)
            active.remove(index)
            visited.add(index)

        visit(0)
        if len(visited) != count:
            raise ValueError("all tree nodes must be reachable from root")

    @property
    def leaf_count(self) -> int:
        return sum(node.is_leaf for node in self.nodes)

    @property
    def depth(self) -> int:
        def recurse(index: int) -> int:
            node = self.nodes[index]
            if node.is_leaf:
                return 0
            assert node.left_index is not None
            assert node.right_index is not None
            return 1 + max(recurse(node.left_index), recurse(node.right_index))

        return recurse(0)

    def to_dict(self) -> dict[str, object]:
        return {
            "tree_id": self.tree_id,
            "nodes": [node.to_dict() for node in self.nodes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TreePlan":
        nodes = data.get("nodes")
        if not isinstance(nodes, list):
            raise TypeError("tree nodes must be a list")
        return cls(
            tree_id=str(data["tree_id"]),
            nodes=tuple(TreeNodePlan.from_dict(item) for item in nodes),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ForestPlan:
    """A sampled forest whose output is the scaled mean of its trees."""

    forest_id: str
    trees: tuple[TreePlan, ...]
    output_scale: float = 1.0

    def __post_init__(self) -> None:
        _identifier("forest_id", self.forest_id)
        if not isinstance(self.trees, tuple) or not self.trees:
            raise ValueError("trees must be a non-empty tuple")
        if not all(isinstance(tree, TreePlan) for tree in self.trees):
            raise TypeError("trees must contain TreePlan values")
        if len({tree.tree_id for tree in self.trees}) != len(self.trees):
            raise ValueError("tree IDs must be unique")
        if not isinstance(self.output_scale, (int, float)) or isinstance(
            self.output_scale, bool
        ) or not isfinite(float(self.output_scale)) or float(self.output_scale) <= 0:
            raise ValueError("output_scale must be a positive finite number")

    @property
    def leaf_count(self) -> int:
        return sum(tree.leaf_count for tree in self.trees)

    @property
    def max_depth(self) -> int:
        return max(tree.depth for tree in self.trees)

    def to_dict(self) -> dict[str, object]:
        return {
            "forest_id": self.forest_id,
            "trees": [tree.to_dict() for tree in self.trees],
            "output_scale": self.output_scale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ForestPlan":
        trees = data.get("trees")
        if not isinstance(trees, list):
            raise TypeError("forest trees must be a list")
        return cls(
            forest_id=str(data["forest_id"]),
            trees=tuple(TreePlan.from_dict(item) for item in trees),
            output_scale=float(data.get("output_scale", 1.0)),
        )


__all__ = ["TreeNodePlan", "TreePlan", "ForestPlan"]
