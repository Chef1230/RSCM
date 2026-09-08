from __future__ import annotations
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdb_prior.compilation.compiler import PhysicalSchemaCompiler
from rdb_prior.compilation.model import ColumnKind
from rdb_prior.generation.database import DatabaseGenerator
from rdb_prior.nuisance.executor import NuisanceExecutor
from rdb_prior.nuisance.planner import NuisanceOverlayConfig, NuisancePlanner
from rdb_prior.priors.model import NuisanceColumnRole, NuisancePriorKind, PriorFamily
from rdb_prior.priors.planner import PriorPlanner, PriorPlannerConfig
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.domain_prototypes import sample_semantic_schema
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig

class NuisancePriorTests(unittest.TestCase):
    def _schema(self, seed=913):
        runtime = RuntimeContext(seed).for_sample("nuisance")
        blueprint = BlueprintSampler(BlueprintSamplerConfig(
            min_tables=3, max_tables=3, min_motif_occurrences=1,
            max_motif_occurrences=1, max_extra_edges=0,
            background_attachment_probability=0.0,
            motif_weights=(("entity_event", 1.0),),
        )).sample("nuisance", runtime)
        schema = PhysicalSchemaCompiler().compile(blueprint, "nuisance", runtime)
        semantic = sample_semantic_schema(schema, runtime.child("semantic"))
        return runtime, blueprint, schema, semantic

    def test_nuisance_plan_round_trip_and_roles(self):
        runtime, _blueprint, schema, semantic = self._schema()
        config = NuisanceOverlayConfig(
            enabled=True, kind=NuisancePriorKind.PROXY,
            missing_rate_min=0.1, missing_rate_max=0.1,
        )
        plan = NuisancePlanner(config).plan(
            schema=schema, semantic_schema=semantic,
            runtime=runtime.child("nuisance"),
        )
        self.assertTrue(plan.enabled)
        self.assertEqual(plan, type(plan).from_dict(plan.to_dict()))
        self.assertEqual("proxy", plan.mechanism.kind)
        self.assertTrue(any(item.role is NuisanceColumnRole.PROXY for item in plan.column_roles))

    def test_overlay_is_deterministic_and_orthogonal(self):
        runtime, blueprint, schema, semantic = self._schema(914)
        prior_config = PriorPlannerConfig(
            database_family_weights=((PriorFamily.RELATIONAL_SCM, 1.0),),
            nuisance=NuisanceOverlayConfig(
                enabled=True, kind=NuisancePriorKind.MCAR,
                missing_rate_min=0.2, missing_rate_max=0.2,
            ),
        )
        prior = PriorPlanner(prior_config).plan(
            blueprint=blueprint, physical_schema=schema,
            semantic_schema=semantic, runtime=runtime.child("prior"),
        )
        self.assertEqual("mcar", prior.composition.nuisance_plan.mechanism.kind)
        self.assertTrue(prior.composition.nuisance_plan.enabled)
        self.assertFalse(any("nuisance" in bundle.attribute_mechanism.kind for bundle in prior.motif_bundles))
        from rdb_prior.instance.planner import InstancePlanner, InstancePlannerConfig
        first_plan = InstancePlanner(InstancePlannerConfig(
            entity_rows_min=20, entity_rows_max=20,
            lookup_rows_min=5, lookup_rows_max=5,
            max_rows_per_table=200,
        )).plan(
            sample_id="nuisance", schema=schema,
            runtime=runtime.child("instance"), prior_plan=prior,
        )
        db1 = DatabaseGenerator().generate(schema=schema, plan=first_plan)
        db2 = DatabaseGenerator().generate(schema=schema, plan=first_plan)
        for left, right in zip(db1.tables, db2.tables):
            for column_id in left.columns:
                
                left_values = left.column(column_id)
                right_values = right.column(column_id)
                if left_values.dtype.kind in {"f"}:
                    self.assertTrue(np.all((left_values == right_values) | (np.isnan(left_values) & np.isnan(right_values))))
                else:
                    self.assertTrue(np.array_equal(left_values, right_values))
        self.assertEqual(first_plan.nuisance_plan, prior.composition.nuisance_plan)

    def test_nuisance_can_be_disabled_explicitly(self):
        runtime, _blueprint, schema, semantic = self._schema(916)
        plan = NuisancePlanner(NuisanceOverlayConfig(
            enabled=False, kind=NuisancePriorKind.MCAR,
        )).plan(
            schema=schema, semantic_schema=semantic,
            runtime=runtime.child("nuisance"),
        )
        self.assertFalse(plan.enabled)
        self.assertEqual("legacy", plan.mechanism.kind)

    def test_all_nuisance_modes_can_be_planned(self):
        runtime, _blueprint, schema, semantic = self._schema(915)
        for kind in (
            NuisancePriorKind.MCAR, NuisancePriorKind.MAR,
            NuisancePriorKind.MNAR, NuisancePriorKind.PROXY,
            NuisancePriorKind.SPURIOUS, NuisancePriorKind.DISTRACTOR,
        ):
            plan = NuisancePlanner(NuisanceOverlayConfig(
                enabled=True, kind=kind, distractor_relation_count=1,
            )).plan(
                schema=schema, semantic_schema=semantic,
                runtime=runtime.child("nuisance", kind.value),
            )
            self.assertTrue(plan.enabled)
            self.assertEqual(kind.value, plan.mechanism.kind)

if __name__ == "__main__":
    unittest.main()
