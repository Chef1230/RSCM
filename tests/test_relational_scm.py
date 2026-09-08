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
from rdb_prior.generation.column_dag import ColumnSCMPlan, validate_column_dag
from rdb_prior.generation.database import DatabaseGenerator
from rdb_prior.instance.plan import InstancePlan
from rdb_prior.instance.planner import InstancePlanner, InstancePlannerConfig
from rdb_prior.priors.model import PriorFamily
from rdb_prior.priors.planner import PriorPlanner, PriorPlannerConfig, RelationSCMConfig
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.domain_prototypes import sample_semantic_schema
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.validation.checks import validate_database_instance


class RelationalSCMTests(unittest.TestCase):
    def _fixture(self, seed: int = 901):
        runtime = RuntimeContext(seed).for_sample("relational_scm")
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
        ).sample("relational_scm", runtime)
        schema = PhysicalSchemaCompiler().compile(
            blueprint, "relational_scm", runtime
        )
        semantic = sample_semantic_schema(schema, runtime.child("semantic"))
        prior = PriorPlanner(
            PriorPlannerConfig(
                database_family_weights=((PriorFamily.RELATIONAL_SCM, 1.0),),
                relational_scm=RelationSCMConfig(
                    column_dag_depth=(1, 3),
                    parent_count=(1, 3),
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
                entity_rows_min=36,
                entity_rows_max=36,
                lookup_rows_min=4,
                lookup_rows_max=4,
                max_rows_per_table=192,
            )
        ).plan(
            sample_id="relational_scm",
            schema=schema,
            runtime=runtime.child("instance"),
            prior_plan=prior,
        )
        return runtime, schema, prior, plan

    def test_column_scm_plan_rejects_cycles(self) -> None:
        with self.assertRaisesRegex(ValueError, "acyclic"):
            validate_column_dag(
                (
                    ColumnSCMPlan(
                        column_id="a",
                        parent_column_ids=("b",),
                        family="linear",
                    ),
                    ColumnSCMPlan(
                        column_id="b",
                        parent_column_ids=("a",),
                        family="cam",
                    ),
                )
            )

    def test_relational_scm_materializes_and_round_trips(self) -> None:
        _runtime, schema, prior, plan = self._fixture()
        self.assertEqual(PriorFamily.RELATIONAL_SCM.value, plan.prior_family)
        self.assertTrue(plan.column_mechanisms)
        self.assertTrue(all(item.family in {"exogenous", "linear", "cam", "mlp"} for item in plan.column_mechanisms))
        self.assertTrue(plan.population_mechanisms)
        self.assertTrue(any(item.family.startswith("scm_") for item in plan.relations))
        materialized = DatabaseGenerator().materialize(schema=schema, plan=plan)
        self.assertTrue(validate_database_instance(
            schema,
            materialized.plan,
            materialized.database,
        ).is_valid)
        self.assertEqual(plan, InstancePlan.from_dict(plan.to_dict()))
        self.assertEqual(prior, type(prior).from_dict(prior.to_dict()))
        self.assertTrue(
            any(
                "realized_child_count" in dict(item.parameters)
                for item in materialized.plan.population_mechanisms
            )
        )
        replayed = DatabaseGenerator().materialize(
            schema=schema,
            plan=materialized.plan,
        )
        for left, right in zip(
            materialized.database.tables,
            replayed.database.tables,
            strict=True,
        ):
            for column_id, values in left.columns.items():
                np.testing.assert_array_equal(values, right.columns[column_id])

    def test_fixed_seed_replays_relation_scm_database(self) -> None:
        _runtime, schema, _prior, plan = self._fixture(902)
        first = DatabaseGenerator().materialize(schema=schema, plan=plan).database
        second = DatabaseGenerator().materialize(schema=schema, plan=plan).database
        for left, right in zip(first.tables, second.tables, strict=True):
            for column_id, values in left.columns.items():
                np.testing.assert_array_equal(values, right.columns[column_id])


if __name__ == "__main__":
    unittest.main()
