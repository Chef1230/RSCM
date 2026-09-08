from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.compilation.compiler import PhysicalSchemaCompiler
from rdb_prior.generation.database import DatabaseGenerator
from rdb_prior.generation.trees.executor import evaluate_forest
from rdb_prior.generation.trees.model import ForestPlan, TreeNodePlan, TreePlan
from rdb_prior.generation.trees.sampler import sample_forest
from rdb_prior.instance.plan import InstancePlan
from rdb_prior.instance.planner import InstancePlanner, InstancePlannerConfig
from rdb_prior.priors.model import (
    AttributePriorKind,
    NuisancePriorKind,
    PriorFamily,
    RelationPriorKind,
    TemporalPriorKind,
)
from rdb_prior.priors.planner import (
    PriorCompositionConfig,
    PriorPlanner,
    PriorPlannerConfig,
    RelationTreeConfig,
)
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.domain_prototypes import sample_semantic_schema
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.validation.checks import validate_database_instance


class RelationTreePriorTests(unittest.TestCase):
    def _schema(self, sample_id: str, seed: int):
        runtime = RuntimeContext(seed).for_sample(sample_id)
        blueprint = BlueprintSampler(
            BlueprintSamplerConfig(
                min_tables=3,
                max_tables=3,
                min_motif_occurrences=1,
                max_motif_occurrences=1,
                max_extra_edges=0,
                background_attachment_probability=0.0,
                motif_weights=(("entity_event", 1.0),),
            )
        ).sample(sample_id, runtime)
        schema = PhysicalSchemaCompiler().compile(blueprint, sample_id, runtime)
        semantic = sample_semantic_schema(schema, runtime.child("semantic"))
        return runtime, blueprint, schema, semantic

    def test_tree_model_round_trip_and_execution(self) -> None:
        tree = TreePlan(
            tree_id="known",
            nodes=(
                TreeNodePlan(
                    feature_ref="x",
                    threshold=0.0,
                    left_index=1,
                    right_index=2,
                    leaf_value=None,
                ),
                TreeNodePlan(
                    feature_ref=None,
                    threshold=None,
                    left_index=None,
                    right_index=None,
                    leaf_value=-2.0,
                ),
                TreeNodePlan(
                    feature_ref=None,
                    threshold=None,
                    left_index=None,
                    right_index=None,
                    leaf_value=3.0,
                ),
            ),
        )
        forest = ForestPlan(forest_id="known_forest", trees=(tree,))
        self.assertEqual(forest, ForestPlan.from_dict(forest.to_dict()))
        np.testing.assert_allclose(
            evaluate_forest(forest, {"x": np.asarray((-1.0, 2.0))}),
            np.asarray((-2.0, 3.0)),
        )
        self.assertEqual(2, forest.leaf_count)
        self.assertEqual(1, forest.max_depth)

    def test_sampler_is_seed_deterministic_and_bounded(self) -> None:
        first = sample_forest(
            forest_id="deterministic",
            feature_refs=("a", "b"),
            rng=np.random.Generator(np.random.PCG64DXSM(91)),
            tree_count=(3, 3),
            depth=(2, 2),
            threshold_strategy="extra",
        )
        second = sample_forest(
            forest_id="deterministic",
            feature_refs=("a", "b"),
            rng=np.random.Generator(np.random.PCG64DXSM(91)),
            tree_count=(3, 3),
            depth=(2, 2),
            threshold_strategy="extra",
        )
        self.assertEqual(first, second)
        self.assertEqual(3, len(first.trees))
        self.assertEqual(2, first.max_depth)
        self.assertGreaterEqual(first.leaf_count, 3)

    def test_tree_only_prior_materializes_relations_and_attributes(self) -> None:
        runtime, blueprint, schema, semantic = self._schema("tree_only", 812)
        prior = PriorPlanner(
            PriorPlannerConfig(
                database_family_weights=((PriorFamily.RELATIONAL_TREE, 1.0),),
                relational_tree=RelationTreeConfig(
                    tree_count_min=2,
                    tree_count_max=2,
                    depth_min=2,
                    depth_max=2,
                ),
            )
        ).plan(
            blueprint=blueprint,
            physical_schema=schema,
            semantic_schema=semantic,
            runtime=runtime.child("prior"),
        )
        plan = InstancePlanner(
            InstancePlannerConfig(
                entity_rows_min=32,
                entity_rows_max=32,
                lookup_rows_min=4,
                lookup_rows_max=4,
                max_rows_per_table=160,
            )
        ).plan(
            sample_id="tree_only",
            schema=schema,
            runtime=runtime.child("instance"),
            prior_plan=prior,
        )
        self.assertEqual(PriorFamily.RELATIONAL_TREE.value, plan.prior_family)
        self.assertTrue(any(item.family == "tree_propensity" for item in plan.relations))
        self.assertTrue(any(item.family == "tree" for item in plan.column_mechanisms))
        materialized = DatabaseGenerator().materialize(schema=schema, plan=plan)
        report = validate_database_instance(
            schema,
            materialized.plan,
            materialized.database,
        )
        self.assertTrue(report.is_valid, report.issues)
        self.assertEqual(plan, InstancePlan.from_dict(plan.to_dict()))
        self.assertEqual(prior, type(prior).from_dict(prior.to_dict()))
        replay = DatabaseGenerator().materialize(schema=schema, plan=plan)
        for expected, actual in zip(
            materialized.database.tables,
            replay.database.tables,
            strict=True,
        ):
            self.assertEqual(expected.table_id, actual.table_id)
            for column_id, values in expected.columns.items():
                np.testing.assert_array_equal(values, actual.columns[column_id])

    def test_tree_combines_with_temporal_and_nuisance_axes(self) -> None:
        runtime, blueprint, schema, semantic = self._schema("tree_temporal", 813)
        prior = PriorPlanner(
            PriorPlannerConfig(
                composition=PriorCompositionConfig(
                    attribute=AttributePriorKind.TREE,
                    relation=RelationPriorKind.STATE_CONDITIONED_EVENT,
                    temporal=TemporalPriorKind.CHURN,
                    nuisance=NuisancePriorKind.MNAR,
                ),
                relational_tree=RelationTreeConfig(
                    tree_count_min=1,
                    tree_count_max=1,
                    depth_min=2,
                    depth_max=2,
                ),
            )
        ).plan(
            blueprint=blueprint,
            physical_schema=schema,
            semantic_schema=semantic,
            runtime=runtime.child("prior"),
        )
        plan = InstancePlanner(
            InstancePlannerConfig(
                entity_rows_min=32,
                entity_rows_max=32,
                lookup_rows_min=4,
                lookup_rows_max=4,
                max_rows_per_table=192,
            )
        ).plan(
            sample_id="tree_temporal",
            schema=schema,
            runtime=runtime.child("instance"),
            prior_plan=prior,
        )
        self.assertEqual(PriorFamily.TEMPORAL_EVENT.value, plan.prior_family)
        self.assertTrue(any(item.family == "tree" for item in plan.column_mechanisms))
        population = next(item for item in plan.population_mechanisms)
        self.assertIn("event_intensity_forest", dict(population.parameters))
        materialized = DatabaseGenerator().materialize(schema=schema, plan=plan)
        report = validate_database_instance(
            schema,
            materialized.plan,
            materialized.database,
        )
        self.assertTrue(report.is_valid, report.issues)


if __name__ == "__main__":
    unittest.main()
