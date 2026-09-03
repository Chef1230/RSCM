"""End-to-end materialization of one anonymous database instance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rdb_prior.compilation.model import ColumnKind, PhysicalSchema
from rdb_prior.generation.dispatch import finalize_plan, requires_finalization
from rdb_prior.generation.features import generate_table_features
from rdb_prior.generation.latent import generate_latent_registry
from rdb_prior.generation.model import DatabaseInstance, TableData
from rdb_prior.generation.relations import generate_relations
from rdb_prior.generation.temporal_processes import apply_temporal_event_processes
from rdb_prior.instance.plan import InstancePlan


@dataclass(frozen=True, slots=True, kw_only=True)
class DatabaseMaterialization:
    """A generated database paired with the resolved immutable plan."""

    plan: InstancePlan
    database: DatabaseInstance


class DatabaseGenerator:
    def generate(
        self,
        *,
        schema: PhysicalSchema,
        plan: InstancePlan,
    ) -> DatabaseInstance:
        return self._generate_once(schema=schema, plan=plan)

    def _generate_once(
        self,
        *,
        schema: PhysicalSchema,
        plan: InstancePlan,
    ) -> DatabaseInstance:
        if schema.schema_id != plan.schema_id:
            raise ValueError("instance plan does not belong to physical schema")
        latents = generate_latent_registry(plan)
        relations = generate_relations(plan, latents)
        foreign_keys = {
            foreign_key.child_column_id: foreign_key
            for foreign_key in schema.foreign_keys
        }
        generated: dict[str, TableData] = {}

        for table_id in plan.generation_order:
            table = schema.table(table_id)
            row_count = plan.table(table_id).population.row_count
            columns: dict[str, np.ndarray] = {}
            for column in table.columns:
                if column.kind is ColumnKind.PRIMARY_KEY:
                    columns[column.column_id] = np.arange(
                        row_count, dtype=np.int64
                    )
                elif column.kind is ColumnKind.FOREIGN_KEY:
                    foreign_key = foreign_keys[column.column_id]
                    columns[column.column_id] = relations[
                        foreign_key.foreign_key_id
                    ]

            columns.update(
                generate_table_features(
                    schema=schema,
                    table=table,
                    plan=plan,
                    latents=latents,
                    relations=relations,
                    generated_tables=generated,
                )
            )
            generated[table_id] = TableData(table_id=table_id, columns=columns)

        database = DatabaseInstance(
            instance_id=f"instance_{plan.sample_id}",
            schema_id=schema.schema_id,
            plan_id=plan.plan_id,
            tables=tuple(generated.values()),
        )
        return apply_temporal_event_processes(schema, plan, database)

    def materialize(
        self,
        *,
        schema: PhysicalSchema,
        plan: InstancePlan,
    ) -> DatabaseMaterialization:
        """Return the final plan and database, preserving ``generate`` API.

        Legacy plans materialize exactly once.  P1 temporal plans first use a
        draft to obtain Entity attributes, resolve state+attribute-conditioned
        Event counts, then materialize that final immutable plan.
        """
        if not requires_finalization(plan):
            return DatabaseMaterialization(
                plan=plan,
                database=self._generate_once(schema=schema, plan=plan),
            )
        draft = self._generate_once(schema=schema, plan=plan)
        final_plan = finalize_plan(schema, plan, draft)
        return DatabaseMaterialization(
            plan=final_plan,
            database=self._generate_once(schema=schema, plan=final_plan),
        )


__all__ = ["DatabaseMaterialization", "DatabaseGenerator"]
