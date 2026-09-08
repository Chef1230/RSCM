# src/rdb_prior/compilation/compiler.py
# -*- coding: utf-8 -*-
"""Deterministic randomized compilation from Blueprint to PhysicalSchema."""

from __future__ import annotations

from dataclasses import dataclass
import re

from rdb_prior.compilation.model import (
    ColumnKind,
    CompilationResult,
    CompilationTrace,
    PhysicalColumn,
    PhysicalDataType,
    PhysicalForeignKey,
    PhysicalSchema,
    PhysicalTable,
)
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.blueprint import SchemaBlueprint
from rdb_prior.schema.roles import get_role_edge_rule
from rdb_prior.schema.semantics import (
    ColumnSemanticRole,
    SemanticNodePlan,
    SemanticSchemaPlan,
)
from rdb_prior.schema.spec import Optionality, TableRole
from rdb_prior.schema.validation import validate_blueprint


_SQL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_column_bounds(
    minimum: int,
    maximum: int,
    *,
    prefix: str,
) -> None:
    for name, value in (
        (f"{prefix}.min_columns", minimum),
        (f"{prefix}.max_columns", maximum),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if maximum < minimum:
        raise ValueError(
            f"{prefix}.max_columns must be at least min_columns"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TableCountFeatureRule:
    """Feature-column bounds selected by total schema table count."""

    table_count_min: int
    table_count_max: int
    min_columns: int
    max_columns: int

    def __post_init__(self) -> None:
        for name, value in (
            ("table_count_min", self.table_count_min),
            ("table_count_max", self.table_count_max),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.table_count_max < self.table_count_min:
            raise ValueError(
                "table_count_max must be at least table_count_min"
            )
        _validate_column_bounds(
            self.min_columns,
            self.max_columns,
            prefix="table-count feature rule",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RoleFeatureRule:
    """Feature-column bounds overriding defaults for one table role."""

    role: TableRole
    min_columns: int
    max_columns: int

    def __post_init__(self) -> None:
        if not isinstance(self.role, TableRole):
            raise TypeError("role must be TableRole")
        _validate_column_bounds(
            self.min_columns,
            self.max_columns,
            prefix=f"role feature rule {self.role.value}",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PhysicalCompilerConfig:
    min_feature_columns: int = 2
    max_feature_columns: int = 6
    feature_columns_by_table_count: tuple[
        TableCountFeatureRule, ...
    ] = ()
    feature_columns_by_role: tuple[RoleFeatureRule, ...] = ()
    feature_nullable_probability: float = 0.15
    primary_key_names: tuple[str, ...] = ("id", "pk_id", "row_id")
    schema_id_prefix: str = "schema"

    def __post_init__(self) -> None:
        for name, value in (
            ("min_feature_columns", self.min_feature_columns),
            ("max_feature_columns", self.max_feature_columns),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_feature_columns < self.min_feature_columns:
            raise ValueError(
                "max_feature_columns must be at least min_feature_columns"
            )

        if not isinstance(self.feature_columns_by_table_count, tuple):
            raise TypeError(
                "feature_columns_by_table_count must be a tuple"
            )
        if not all(
            isinstance(rule, TableCountFeatureRule)
            for rule in self.feature_columns_by_table_count
        ):
            raise TypeError(
                "feature_columns_by_table_count items must be "
                "TableCountFeatureRule"
            )
        ordered_rules = sorted(
            self.feature_columns_by_table_count,
            key=lambda rule: rule.table_count_min,
        )
        for previous, current in zip(ordered_rules, ordered_rules[1:]):
            if current.table_count_min <= previous.table_count_max:
                raise ValueError(
                    "feature_columns_by_table_count ranges must not overlap"
                )

        if not isinstance(self.feature_columns_by_role, tuple):
            raise TypeError("feature_columns_by_role must be a tuple")
        if not all(
            isinstance(rule, RoleFeatureRule)
            for rule in self.feature_columns_by_role
        ):
            raise TypeError(
                "feature_columns_by_role items must be RoleFeatureRule"
            )
        roles = tuple(rule.role for rule in self.feature_columns_by_role)
        if len(set(roles)) != len(roles):
            raise ValueError("feature_columns_by_role roles must be unique")

        if isinstance(self.feature_nullable_probability, bool) or not isinstance(
            self.feature_nullable_probability,
            (int, float),
        ):
            raise TypeError("feature_nullable_probability must be numeric")
        if not 0 <= self.feature_nullable_probability <= 1:
            raise ValueError(
                "feature_nullable_probability must be between zero and one"
            )

        if not isinstance(self.primary_key_names, tuple):
            raise TypeError("primary_key_names must be a tuple")
        if not self.primary_key_names:
            raise ValueError("primary_key_names must not be empty")
        if len(set(self.primary_key_names)) != len(self.primary_key_names):
            raise ValueError("primary_key_names must be unique")
        for name in self.primary_key_names:
            if not isinstance(name, str) or not _SQL_NAME.fullmatch(name):
                raise ValueError(
                    "primary_key_names must contain lowercase SQL-safe names"
                )
            if name in {"event_time", "code", "label", "position"} or (
                name.startswith("fk_") or name.startswith("f_")
            ):
                raise ValueError(
                    "primary_key_names must not overlap generated column "
                    "namespaces"
                )
        if (
            not isinstance(self.schema_id_prefix, str)
            or not self.schema_id_prefix.strip()
        ):
            raise ValueError("schema_id_prefix must not be empty")


class PhysicalSchemaCompiler:
    """Compile one valid Blueprint while sampling physical design choices."""

    __slots__ = ("config",)

    def __init__(
        self,
        config: PhysicalCompilerConfig | None = None,
    ) -> None:
        self.config = config or PhysicalCompilerConfig()

    def compile(
        self,
        blueprint: SchemaBlueprint,
        sample_id: str | int,
        runtime: RuntimeContext,
        *,
        semantic_schema: SemanticSchemaPlan | None = None,
    ) -> PhysicalSchema:
        if not isinstance(blueprint, SchemaBlueprint):
            raise TypeError("blueprint must be SchemaBlueprint")
        if isinstance(sample_id, bool) or not isinstance(
            sample_id,
            (str, int),
        ):
            raise TypeError("sample_id must be a string or integer")
        if isinstance(sample_id, str) and not sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if not isinstance(runtime, RuntimeContext):
            raise TypeError("runtime must be RuntimeContext")

        validate_blueprint(blueprint, raise_on_error=True)
        if semantic_schema is not None:
            self._validate_semantic_schema(blueprint, semantic_schema)
        table_names = self._table_names(blueprint, runtime)
        primary_key_names = self._primary_key_names(blueprint, runtime)
        incoming_edges = {
            node.node_id: blueprint.incoming_edges(node.node_id)
            for node in blueprint.nodes
        }

        tables: list[PhysicalTable] = []
        fk_column_ids: dict[str, str] = {}

        for node in blueprint.nodes:
            columns: list[PhysicalColumn] = []
            primary_key_id = f"{node.node_id}_C000"
            columns.append(
                PhysicalColumn(
                    column_id=primary_key_id,
                    name=primary_key_names[node.node_id],
                    data_type=PhysicalDataType.BIGINT,
                    kind=ColumnKind.PRIMARY_KEY,
                    ordinal=0,
                    nullable=False,
                    unique=True,
                )
            )

            for edge in incoming_edges[node.node_id]:
                ordinal = len(columns)
                column_id = f"{node.node_id}_C{ordinal:03d}"
                fk_column_ids[edge.edge_id] = column_id
                rule = get_role_edge_rule(
                    blueprint.node(edge.parent_node_id).role,
                    node.role,
                )
                columns.append(
                    PhysicalColumn(
                        column_id=column_id,
                        name=f"fk_{edge.edge_id.lower()}",
                        data_type=PhysicalDataType.BIGINT,
                        kind=ColumnKind.FOREIGN_KEY,
                        ordinal=ordinal,
                        nullable=(
                            rule.optionality is Optionality.OPTIONAL
                        ),
                    )
                )

            semantic_node = (
                semantic_schema.node_plan(node.node_id)
                if semantic_schema is not None
                else None
            )
            self._add_role_columns(
                node_id=node.node_id,
                role=node.role,
                columns=columns,
                runtime=runtime,
                semantic_node=semantic_node,
            )
            self._add_feature_columns(
                node_id=node.node_id,
                role=node.role,
                table_count=len(blueprint.nodes),
                columns=columns,
                runtime=runtime,
            )

            tables.append(
                PhysicalTable(
                    table_id=node.node_id,
                    name=table_names[node.node_id],
                    role=node.role,
                    rank=node.rank,
                    columns=tuple(columns),
                )
            )

        table_by_id = {table.table_id: table for table in tables}
        foreign_keys: list[PhysicalForeignKey] = []
        for edge in blueprint.edges:
            parent_node = blueprint.node(edge.parent_node_id)
            child_node = blueprint.node(edge.child_node_id)
            rule = get_role_edge_rule(parent_node.role, child_node.role)
            foreign_keys.append(
                PhysicalForeignKey(
                    foreign_key_id=edge.edge_id,
                    name=f"fk_{edge.edge_id.lower()}_constraint",
                    parent_table_id=parent_node.node_id,
                    parent_column_id=(
                        table_by_id[parent_node.node_id].primary_key.column_id
                    ),
                    child_table_id=child_node.node_id,
                    child_column_id=fk_column_ids[edge.edge_id],
                    cardinality=rule.cardinality,
                    optionality=rule.optionality,
                    identity_dependency=rule.identity_dependency,
                    relation_strategy=rule.relation_strategy,
                )
            )

        return PhysicalSchema(
            schema_id=f"{self.config.schema_id_prefix}_{sample_id}",
            blueprint_id=blueprint.blueprint_id,
            tables=tuple(tables),
            foreign_keys=tuple(foreign_keys),
        )

    def compile_result(
        self,
        blueprint: SchemaBlueprint,
        sample_id: str | int,
        runtime: RuntimeContext,
        *,
        semantic_schema: SemanticSchemaPlan | None = None,
    ) -> CompilationResult:
        """Compile with an explicit logical-to-physical trace."""
        schema = self.compile(
            blueprint,
            sample_id,
            runtime,
            semantic_schema=semantic_schema,
        )
        trace = CompilationTrace(
            blueprint_id=blueprint.blueprint_id,
            schema_id=schema.schema_id,
            node_to_tables=tuple(
                (node.node_id, (node.node_id,))
                for node in blueprint.nodes
            ),
            edge_to_foreign_keys=tuple(
                (edge.edge_id, (edge.edge_id,))
                for edge in blueprint.edges
            ),
        )
        return CompilationResult(schema=schema, trace=trace)

    @staticmethod
    def _table_names(
        blueprint: SchemaBlueprint,
        runtime: RuntimeContext,
    ) -> dict[str, str]:
        names: dict[str, str] = {}
        for index, node in enumerate(blueprint.nodes):
            token = runtime.uint32_seed(
                "schema",
                "physical",
                "table-name",
                node.node_id,
            ) & 0xFFFF
            names[node.node_id] = f"t_{index:03d}_{token:04x}"
        return names

    def _primary_key_names(
        self,
        blueprint: SchemaBlueprint,
        runtime: RuntimeContext,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for node in blueprint.nodes:
            rng = runtime.python_rng(
                "schema",
                "physical",
                "primary-key-name",
                node.node_id,
            )
            result[node.node_id] = self.config.primary_key_names[
                rng.randrange(len(self.config.primary_key_names))
            ]
        return result

    def _add_role_columns(
        self,
        *,
        node_id: str,
        role: TableRole,
        columns: list[PhysicalColumn],
        runtime: RuntimeContext,
        semantic_node: SemanticNodePlan | None = None,
    ) -> None:
        specifications: tuple[
            tuple[PhysicalDataType, ColumnKind, bool, bool], ...
        ]
        if role is TableRole.EVENT:
            specifications = (
                (
                    PhysicalDataType.TIMESTAMP,
                    ColumnKind.TIME,
                    False,
                    False,
                ),
            )
        elif role is TableRole.LOOKUP:
            specifications = (
                (
                    PhysicalDataType.TEXT,
                    ColumnKind.FEATURE,
                    False,
                    True,
                ),
                (
                    PhysicalDataType.TEXT,
                    ColumnKind.FEATURE,
                    False,
                    False,
                ),
            )
        elif role is TableRole.DETAIL:
            specifications = (
                (
                    PhysicalDataType.INTEGER,
                    ColumnKind.FEATURE,
                    False,
                    False,
                ),
                (
                    PhysicalDataType.TIMESTAMP,
                    ColumnKind.TIME,
                    False,
                    False,
                ),
            )
        else:
            specifications = ()

        existing_names = {column.name for column in columns}
        for data_type, kind, nullable, unique in specifications:
            ordinal = len(columns)
            token = runtime.uint32_seed(
                "schema",
                "physical",
                "role-column",
                node_id,
                ordinal,
            ) & 0xFFFF
            name = f"c_{ordinal:03d}_{token:04x}"
            if name in existing_names:
                raise RuntimeError("anonymous role-column name collision")
            columns.append(
                PhysicalColumn(
                    column_id=f"{node_id}_C{ordinal:03d}",
                    name=name,
                    data_type=data_type,
                    kind=kind,
                    ordinal=ordinal,
                    nullable=nullable,
                    unique=unique,
                )
            )
            existing_names.add(name)

        if semantic_node is None:
            return
        for semantic_role in (
            semantic_node.required_column_roles
            + semantic_node.optional_column_roles
        ):
            if semantic_role is ColumnSemanticRole.TIMESTAMP and any(
                column.kind is ColumnKind.TIME for column in columns
            ):
                continue
            data_type = self._semantic_data_type(semantic_role)
            ordinal = len(columns)
            token = runtime.uint32_seed(
                "schema",
                "semantic",
                "column",
                node_id,
                semantic_role.value,
                ordinal,
            ) & 0xFFFF
            name = f"c_{ordinal:03d}_{token:04x}"
            if name in existing_names:
                raise RuntimeError("anonymous semantic-column name collision")
            columns.append(
                PhysicalColumn(
                    column_id=f"{node_id}_C{ordinal:03d}",
                    name=name,
                    data_type=data_type,
                    kind=ColumnKind.FEATURE,
                    ordinal=ordinal,
                    nullable=False,
                    unique=False,
                )
            )
            existing_names.add(name)

    @staticmethod
    def _semantic_data_type(role: ColumnSemanticRole) -> PhysicalDataType:
        if role in {
            ColumnSemanticRole.STATIC_ATTRIBUTE,
            ColumnSemanticRole.MEASUREMENT,
            ColumnSemanticRole.AMOUNT,
        }:
            return PhysicalDataType.DOUBLE
        if role is ColumnSemanticRole.STATE or role is ColumnSemanticRole.OUTCOME:
            return PhysicalDataType.BOOLEAN
        if role in {
            ColumnSemanticRole.CATEGORY,
            ColumnSemanticRole.ACTION_TYPE,
        }:
            return PhysicalDataType.TEXT
        if role is ColumnSemanticRole.TIMESTAMP:
            return PhysicalDataType.TIMESTAMP
        raise ValueError(f"unsupported semantic column role: {role!r}")

    @staticmethod
    def _validate_semantic_schema(
        blueprint: SchemaBlueprint,
        semantic_schema: SemanticSchemaPlan,
    ) -> None:
        if not isinstance(semantic_schema, SemanticSchemaPlan):
            raise TypeError("semantic_schema must be SemanticSchemaPlan")
        expected = {node.node_id for node in blueprint.nodes}
        actual = {table.table_id for table in semantic_schema.tables}
        if actual != expected:
            raise ValueError("semantic schema table IDs do not match blueprint")
        node_ids = {node.node_id for node in semantic_schema.nodes}
        if node_ids and node_ids != expected:
            raise ValueError("semantic schema node IDs do not match blueprint")

    def _add_feature_columns(
        self,
        *,
        node_id: str,
        role: TableRole,
        table_count: int,
        columns: list[PhysicalColumn],
        runtime: RuntimeContext,
    ) -> None:
        rng = runtime.python_rng(
            "schema",
            "physical",
            "features",
            node_id,
        )
        minimum, maximum = self._feature_column_bounds(
            role=role,
            table_count=table_count,
        )
        feature_count = rng.randint(minimum, maximum)
        type_choices = self._feature_type_choices(role)
        existing_names = {column.name for column in columns}

        for feature_index in range(feature_count):
            ordinal = len(columns)
            token = rng.randrange(0x10000)
            name = f"f_{feature_index:03d}_{token:04x}"
            while name in existing_names:
                token = rng.randrange(0x10000)
                name = f"f_{feature_index:03d}_{token:04x}"
            data_type = type_choices[rng.randrange(len(type_choices))]
            columns.append(
                PhysicalColumn(
                    column_id=f"{node_id}_C{ordinal:03d}",
                    name=name,
                    data_type=data_type,
                    kind=ColumnKind.FEATURE,
                    ordinal=ordinal,
                    nullable=(
                        rng.random()
                        < self.config.feature_nullable_probability
                    ),
                )
            )
            existing_names.add(name)

    def _feature_column_bounds(
        self,
        *,
        role: TableRole,
        table_count: int,
    ) -> tuple[int, int]:
        for rule in self.config.feature_columns_by_role:
            if rule.role is role:
                return rule.min_columns, rule.max_columns

        for rule in self.config.feature_columns_by_table_count:
            if rule.table_count_min <= table_count <= rule.table_count_max:
                return rule.min_columns, rule.max_columns

        return (
            self.config.min_feature_columns,
            self.config.max_feature_columns,
        )

    @staticmethod
    def _feature_type_choices(
        role: TableRole,
    ) -> tuple[PhysicalDataType, ...]:
        if role is TableRole.LOOKUP:
            return (
                PhysicalDataType.TEXT,
                PhysicalDataType.TEXT,
                PhysicalDataType.BOOLEAN,
            )
        if role in {TableRole.EVENT, TableRole.DETAIL}:
            return (
                PhysicalDataType.DOUBLE,
                PhysicalDataType.DOUBLE,
                PhysicalDataType.INTEGER,
                PhysicalDataType.BOOLEAN,
                PhysicalDataType.TEXT,
            )
        return (
            PhysicalDataType.DOUBLE,
            PhysicalDataType.INTEGER,
            PhysicalDataType.BOOLEAN,
            PhysicalDataType.TEXT,
        )


__all__ = [
    "TableCountFeatureRule",
    "RoleFeatureRule",
    "PhysicalCompilerConfig",
    "PhysicalSchemaCompiler",
]
