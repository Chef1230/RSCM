"""Compatibility import for generator-private semantic prototypes.

The semantic schema is sampled before physical compilation; this module keeps the
legacy import path working for callers that annotate an already compiled schema.
"""

from rdb_prior.schema.semantics import (
    ColumnSemanticPlan,
    ColumnSemanticRole,
    SemanticNodePlan,
    SemanticSchemaPlan,
    TableSemanticPlan,
    TableSemanticRole,
    complete_semantic_schema,
    sample_semantic_schema,
)

__all__ = [
    "ColumnSemanticPlan",
    "ColumnSemanticRole",
    "SemanticNodePlan",
    "SemanticSchemaPlan",
    "TableSemanticPlan",
    "TableSemanticRole",
    "complete_semantic_schema",
    "sample_semantic_schema",
]
