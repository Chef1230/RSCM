from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.compilation.compiler import (
    PhysicalCompilerConfig,
    PhysicalSchemaCompiler,
    RoleFeatureRule,
    TableCountFeatureRule,
)
from rdb_prior.compilation.model import (
    ColumnKind,
    CompilationResult,
    PhysicalDataType,
    PhysicalSchema,
)
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.semantics import (
    ColumnSemanticRole,
    SemanticNodePlan,
    SemanticSchemaPlan,
    TableSemanticPlan,
    TableSemanticRole,
)
from rdb_prior.schema.sampler import BlueprintSampler, BlueprintSamplerConfig
from rdb_prior.schema.spec import TableRole


class PhysicalSchemaCompilerTests(unittest.TestCase):
    def _compile(self, sample_id: str = "sample_0"):
        runtime = RuntimeContext(42).for_sample(sample_id)
        blueprint = BlueprintSampler().sample(sample_id, runtime)
        schema = PhysicalSchemaCompiler().compile(
            blueprint,
            sample_id,
            runtime,
        )
        return blueprint, schema

    def test_semantic_node_plan_controls_physical_columns(self) -> None:
        sample_id = "semantic_columns"
        runtime = RuntimeContext(77).for_sample(sample_id)
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
        tables = []
        nodes = []
        for item in blueprint.nodes:
            role = (
                TableSemanticRole.ACTOR
                if item.role is TableRole.ENTITY
                else (
                    TableSemanticRole.TRANSACTION
                    if item.role is TableRole.EVENT
                    else TableSemanticRole.OBJECT
                )
            )
            required = {
                TableSemanticRole.ACTOR: (
                    ColumnSemanticRole.STATIC_ATTRIBUTE,
                    ColumnSemanticRole.STATE,
                    ColumnSemanticRole.CATEGORY,
                ),
                TableSemanticRole.TRANSACTION: (
                    ColumnSemanticRole.AMOUNT,
                    ColumnSemanticRole.ACTION_TYPE,
                    ColumnSemanticRole.OUTCOME,
                ),
                TableSemanticRole.OBJECT: (
                    ColumnSemanticRole.STATIC_ATTRIBUTE,
                    ColumnSemanticRole.CATEGORY,
                ),
            }[role]
            tables.append(TableSemanticPlan(table_id=item.node_id, role=role))
            nodes.append(
                SemanticNodePlan(
                    node_id=item.node_id,
                    role=role,
                    required_column_roles=required,
                )
            )
        semantic = SemanticSchemaPlan(
            schema_id="schema_semantic_columns",
            prototype_id="test",
            seed=77,
            tables=tuple(tables),
            columns=(),
            nodes=tuple(nodes),
        )
        compiler = PhysicalSchemaCompiler(
            PhysicalCompilerConfig(min_feature_columns=0, max_feature_columns=0)
        )
        schema = compiler.compile(
            blueprint,
            sample_id,
            runtime,
            semantic_schema=semantic,
        )
        entity = next(item for item in schema.tables if item.role is TableRole.ENTITY)
        event = next(item for item in schema.tables if item.role is TableRole.EVENT)
        self.assertEqual(
            [PhysicalDataType.DOUBLE, PhysicalDataType.BOOLEAN, PhysicalDataType.TEXT],
            [item.data_type for item in entity.columns if item.kind is ColumnKind.FEATURE],
        )
        self.assertEqual(
            [PhysicalDataType.DOUBLE, PhysicalDataType.TEXT, PhysicalDataType.BOOLEAN],
            [item.data_type for item in event.columns if item.kind is ColumnKind.FEATURE],
        )

    def test_compilation_is_deterministic(self) -> None:
        first_blueprint, first_schema = self._compile()
        second_blueprint, second_schema = self._compile()

        self.assertEqual(first_blueprint, second_blueprint)
        self.assertEqual(first_schema, second_schema)

    def test_physical_schema_preserves_logical_identity_and_direction(self) -> None:
        blueprint, schema = self._compile()

        self.assertEqual(
            {node.node_id for node in blueprint.nodes},
            {table.table_id for table in schema.tables},
        )
        self.assertEqual(
            {edge.edge_id for edge in blueprint.edges},
            {fk.foreign_key_id for fk in schema.foreign_keys},
        )

        for foreign_key in schema.foreign_keys:
            logical = blueprint.edge(foreign_key.foreign_key_id)
            self.assertEqual(
                logical.parent_node_id,
                foreign_key.parent_table_id,
            )
            self.assertEqual(
                logical.child_node_id,
                foreign_key.child_table_id,
            )
            parent_column = schema.table(
                foreign_key.parent_table_id
            ).column(foreign_key.parent_column_id)
            child_column = schema.table(
                foreign_key.child_table_id
            ).column(foreign_key.child_column_id)
            self.assertIs(ColumnKind.PRIMARY_KEY, parent_column.kind)
            self.assertIs(ColumnKind.FOREIGN_KEY, child_column.kind)
            self.assertIs(parent_column.data_type, child_column.data_type)

    def test_event_tables_have_time_columns(self) -> None:
        _blueprint, schema = self._compile()

        for table in schema.tables:
            if table.role is not TableRole.EVENT:
                continue
            time_columns = tuple(
                column
                for column in table.columns
                if column.kind is ColumnKind.TIME
            )
            self.assertEqual(1, len(time_columns))
            self.assertIs(
                PhysicalDataType.TIMESTAMP,
                time_columns[0].data_type,
            )

    def test_json_round_trip_is_lossless(self) -> None:
        _blueprint, schema = self._compile()

        self.assertEqual(
            schema,
            PhysicalSchema.from_dict(schema.to_dict()),
        )

    def test_compilation_result_contains_identity_trace(self) -> None:
        sample_id = "trace"
        runtime = RuntimeContext(42).for_sample(sample_id)
        blueprint = BlueprintSampler().sample(sample_id, runtime)
        result = PhysicalSchemaCompiler().compile_result(
            blueprint,
            sample_id,
            runtime,
        )

        self.assertEqual(
            {node.node_id for node in blueprint.nodes},
            set(result.trace.node_tables),
        )
        self.assertEqual(
            {edge.edge_id for edge in blueprint.edges},
            set(result.trace.edge_foreign_keys),
        )
        self.assertEqual(
            result,
            CompilationResult.from_dict(result.to_dict()),
        )

    def test_role_specific_columns_keep_anonymous_names(self) -> None:
        _blueprint, schema = self._compile("anonymous_columns")
        forbidden = {"event_time", "code", "label", "position"}
        names = {
            column.name
            for table in schema.tables
            for column in table.columns
        }
        self.assertTrue(forbidden.isdisjoint(names))

    def test_role_feature_rule_precedes_table_count_rule(self) -> None:
        sample_id = "feature_rules"
        runtime = RuntimeContext(42).for_sample(sample_id)
        blueprint = BlueprintSampler(
            BlueprintSamplerConfig(min_tables=3, max_tables=3)
        ).sample(sample_id, runtime)
        compiler = PhysicalSchemaCompiler(
            PhysicalCompilerConfig(
                min_feature_columns=2,
                max_feature_columns=2,
                feature_columns_by_table_count=(
                    TableCountFeatureRule(
                        table_count_min=3,
                        table_count_max=3,
                        min_columns=4,
                        max_columns=4,
                    ),
                ),
                feature_columns_by_role=(
                    RoleFeatureRule(
                        role=TableRole.ENTITY,
                        min_columns=1,
                        max_columns=1,
                    ),
                ),
            )
        )

        schema = compiler.compile(blueprint, sample_id, runtime)
        for table in schema.tables:
            generated_features = tuple(
                column
                for column in table.columns
                if column.name.startswith("f_")
            )
            expected = 1 if table.role is TableRole.ENTITY else 4
            self.assertEqual(expected, len(generated_features))


if __name__ == "__main__":
    unittest.main()
