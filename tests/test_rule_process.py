from __future__ import annotations

from dataclasses import replace
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
from rdb_prior.generation.model import TableData
from rdb_prior.instance.planner import InstancePlanner, InstancePlannerConfig
from rdb_prior.priors.model import PriorFamily, TaskPolicyPlan
from rdb_prior.priors.planner import PriorPlanner, PriorPlannerConfig
from rdb_prior.priors.registry import descriptor
from rdb_prior.process import (
    Add,
    Aggregate,
    And,
    ColumnRef,
    Compare,
    Constant,
    RuleAction,
    RuleEvaluationContext,
    RuleNode,
    RulePlan,
    RuleProcessExecutor,
)
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.domain_prototypes import sample_semantic_schema
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig


class RuleProcessTests(unittest.TestCase):
    def test_closed_ast_round_trip_and_rejects_python(self) -> None:
        expression = And(
            operands=(
                Compare(">", ColumnRef("entity", "score"), Constant(0.2)),
                Compare(">", Add(Constant(1), Constant(2)), Constant(2)),
            )
        )
        plan = RulePlan(
            rule_id="rule_a",
            condition=expression,
            actions=(
                RuleAction(
                    target="event_intensity",
                    operation="multiply",
                    value=Constant(1.5),
                ),
            ),
            seed=17,
        )
        self.assertEqual(plan, RulePlan.from_dict(plan.to_dict()))
        with self.assertRaises(ValueError):
            RuleNode.from_dict({"type": "python", "source": "__import__('os')"})

    def test_cross_table_aggregate_respects_cutoff(self) -> None:
        context = RuleEvaluationContext(
            tables={
                "entity": {"score": np.asarray([0.0, 1.0])},
                "event": {
                    "fk": np.asarray([0, 0, 1]),
                    "time": np.asarray([10, 16, 12]),
                    "amount": np.asarray([5.0, 8.0, 1.0]),
                },
            },
            cutoff_time=15,
            time_column_ids={"event": "time"},
        )
        node = Aggregate(
            function="count",
            table_id="event",
            predicate=Compare(">", ColumnRef("event", "amount"), Constant(2.0)),
            relation_path=("fk",),
            window_seconds=10,
        )
        condition = Compare(">=", node, Constant(1))
        rule = RulePlan(
            rule_id="aggregate_rule",
            condition=condition,
            actions=(RuleAction(target="event_intensity", operation="multiply", value=Constant(2.0)),),
            seed=1,
        )
        effect = RuleProcessExecutor().effects(rule, context, row_count=2)
        self.assertTrue(np.array_equal(effect.matched, np.asarray([True, False])))
        self.assertTrue(np.array_equal(effect.intensity_multiplier, np.asarray([2.0, 1.0])))

    def test_rule_family_is_registered_and_changes_materialization(self) -> None:
        runtime = RuntimeContext(8801).for_sample("rule_prior")
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
        ).sample("rule_prior", runtime)
        schema = PhysicalSchemaCompiler().compile(blueprint, "rule_prior", runtime)
        semantic = sample_semantic_schema(schema, runtime.child("semantic"))
        prior = PriorPlanner(
            PriorPlannerConfig(
                database_family_weights=((PriorFamily.RULE_PROCESS, 1.0),),
                task_policy=TaskPolicyPlan(programs_per_database=1),
            )
        ).plan(
            blueprint=blueprint,
            physical_schema=schema,
            semantic_schema=semantic,
            runtime=runtime.child("prior"),
        )
        self.assertTrue(descriptor(PriorFamily.RULE_PROCESS).implemented)
        self.assertEqual(PriorFamily.RULE_PROCESS, prior.family)
        self.assertEqual(prior, type(prior).from_dict(prior.to_dict()))
        bundle = next(item for item in prior.motif_bundles if item.family is PriorFamily.RULE_PROCESS)
        self.assertEqual("rule", bundle.process_mechanism.kind)
        self.assertIn("rule_plan", dict(bundle.parameters))

        instance_plan = InstancePlanner(
            InstancePlannerConfig(
                entity_rows_min=32,
                entity_rows_max=32,
                lookup_rows_min=4,
                lookup_rows_max=4,
                max_rows_per_table=256,
            )
        ).plan(
            sample_id="rule_prior",
            schema=schema,
            runtime=runtime.child("instance"),
            prior_plan=prior,
        )
        materialization = DatabaseGenerator().materialize(
            schema=schema,
            plan=instance_plan,
        )
        population = materialization.plan.population_mechanisms[0]
        self.assertGreater(int(dict(population.parameters)["realized_event_count"]), 0)
        self.assertIn("rule_plan", dict(population.parameters))

        original_rule = RulePlan.from_dict(dict(population.parameters)["rule_plan"])
        stronger_rule = replace(
            original_rule,
            actions=(
                RuleAction(target="event_intensity", operation="multiply", value=Constant(4.0)),
                RuleAction(target="event_attribute", operation="add", value=Constant(1.0)),
            ),
        )
        def with_rule(mechanism):
            parameters = dict(mechanism.parameters)
            if "rule_plan" not in parameters:
                return mechanism
            parameters["rule_plan"] = stronger_rule.to_dict()
            return replace(mechanism, parameters=tuple(parameters.items()))

        modified_plan = replace(
            instance_plan,
            population_mechanisms=tuple(with_rule(item) for item in instance_plan.population_mechanisms),
            temporal_processes=tuple(with_rule(item) for item in instance_plan.temporal_processes),
        )
        modified = DatabaseGenerator().materialize(schema=schema, plan=modified_plan)
        self.assertNotEqual(
            int(dict(materialization.plan.population_mechanisms[0].parameters)["realized_event_count"]),
            int(dict(modified.plan.population_mechanisms[0].parameters)["realized_event_count"]),
        )
