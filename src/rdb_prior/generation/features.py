"""Role-aware feature and temporal column generation."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from rdb_prior.compilation.model import (
    ColumnKind,
    PhysicalColumn,
    PhysicalSchema,
    PhysicalTable,
)
from rdb_prior.generation.column_dag import execute_column_dag
from rdb_prior.generation.feature_strategies import generate_feature_signal
from rdb_prior.generation.latent import LatentRegistry
from rdb_prior.generation.trees.executor import evaluate_forest
from rdb_prior.generation.trees.model import ForestPlan
from rdb_prior.generation.model import TableData
from rdb_prior.instance.plan import (
    EventTemporalMechanism,
    ColumnMechanismPlan,
    InstancePlan,
    TableMechanismPlan,
    TemporalFamily,
)
from rdb_prior.generation.encoding import (
    apply_missing as _apply_missing,
    encode_feature_score,
    softmax_codes as _softmax_codes,
)
from rdb_prior.schema.spec import TableRole
from rdb_prior.time_bounds import assert_within_interval


def generate_table_features(
    *,
    schema: PhysicalSchema,
    table: PhysicalTable,
    plan: InstancePlan,
    latents: LatentRegistry,
    relations: Mapping[str, np.ndarray],
    generated_tables: Mapping[str, TableData],
) -> dict[str, np.ndarray]:
    table_plan = plan.table(table.table_id)
    rng = np.random.Generator(np.random.PCG64DXSM(table_plan.feature_seed))
    context = _causal_context(schema, table, latents, relations)
    db_start = plan.calendar_start_seconds
    db_end = plan.calendar_end_seconds
    if db_start is None or db_end is None:
        raise ValueError("instance plan lacks calendar interval")
    values: dict[str, np.ndarray] = {}
    mechanisms = {item.column_id: item for item in plan.column_mechanisms}
    dag_values = (
        execute_column_dag(
            schema=schema,
            table=table,
            mechanisms=mechanisms,
            latents=latents,
            relations=relations,
            generated_tables=generated_tables,
        )
        if plan.prior_family == "relational_scm"
        else {}
    )

    # Tree mechanisms may use time regardless of physical column order.
    for column in table.columns:
        if column.kind is ColumnKind.TIME:
            values[column.column_id] = _generate_time(
                schema=schema,
                table=table,
                column=column,
                table_plan=table_plan,
                relations=relations,
                generated_tables=generated_tables,
                db_start=db_start,
                db_end=db_end,
            )

    for column in table.columns:
        if column.kind in {
            ColumnKind.PRIMARY_KEY,
            ColumnKind.FOREIGN_KEY,
            ColumnKind.TIME,
        }:
            continue
        mechanism = mechanisms.get(column.column_id)
        if column.column_id in dag_values:
            signal = dag_values[column.column_id]
        elif (
            mechanism is not None
            and mechanism.family == "tree"
            and plan.prior_family != "temporal_event"
        ):
            signal = _tree_feature_signal(
                mechanism=mechanism,
                schema=schema,
                table=table,
                latents=latents,
                relations=relations,
                generated_tables=generated_tables,
                local_values=values,
            )
        elif column.column_id not in dag_values:
            signal = generate_feature_signal(
                table_plan.feature_family,
                context,
                rng,
                noise_scale=table_plan.parameter_map["noise_scale"],
                signal_scale=table_plan.parameter_map["signal_scale"],
                activation_scale=table_plan.parameter_map["activation_scale"],
                output_scale=table_plan.parameter_map["output_scale"],
                long_tail_enabled=bool(
                    table_plan.parameter_map["long_tail_enabled"]
                ),
                long_tail_alpha=table_plan.parameter_map["long_tail_alpha"],
                mlp_depth=int(table_plan.parameter_map.get("mlp_depth", 1)),
                mlp_hidden_factor=float(
                    table_plan.parameter_map.get("mlp_hidden_factor", 2.0)
                ),
                mlp_dropout_rate=float(
                    table_plan.parameter_map.get("mlp_dropout_rate", 0.0)
                ),
            )
        values[column.column_id] = encode_feature_score(
            signal,
            column,
            table.role,
            rng,
            cardinality=int(table_plan.parameter_map["categorical_cardinality"]),
            db_start=db_start,
            db_end=db_end,
            categorical_dirichlet_alpha=float(
                table_plan.parameter_map.get(
                    "categorical_dirichlet_alpha", 1.0
                )
            ),
            categorical_signal_strength=float(
                table_plan.parameter_map.get(
                    "categorical_signal_strength", 1.0
                )
            ),
            missing_rate=table_plan.parameter_map["missing_rate"],
            # Feature strategies already consumed this table's long-tail draw.
            long_tail_enabled=False,
        )
    return values


def _tree_feature_signal(
    *,
    mechanism: ColumnMechanismPlan,
    schema: PhysicalSchema,
    table: PhysicalTable,
    latents: LatentRegistry,
    relations: Mapping[str, np.ndarray],
    generated_tables: Mapping[str, TableData],
    local_values: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Evaluate a sampled forest against anonymous table/parent context."""
    payload = dict(mechanism.parameters).get("forest")
    if not isinstance(payload, Mapping):
        raise ValueError("tree column mechanism lacks serialized forest")
    forest = ForestPlan.from_dict(payload)
    rows = latents.table(table.table_id).values.shape[0]
    own = latents.table(table.table_id).values
    parent_latent = np.zeros(rows, dtype=np.float64)
    parent_feature = np.zeros(rows, dtype=np.float64)
    history = np.zeros(rows, dtype=np.float64)
    for foreign_key in schema.foreign_keys:
        if foreign_key.child_table_id != table.table_id:
            continue
        assignments = relations[foreign_key.foreign_key_id]
        valid = assignments >= 0
        parent = latents.table(foreign_key.parent_table_id).values
        parent_latent[valid] = parent[assignments[valid], 0]
        parent_data = generated_tables.get(foreign_key.parent_table_id)
        if parent_data is not None:
            feature = next(
                (
                    column for column in schema.table(foreign_key.parent_table_id).columns
                    if column.kind is ColumnKind.FEATURE
                ),
                None,
            )
            if feature is not None:
                numeric = _numeric_feature(parent_data.column(feature.column_id))
                parent_feature[valid] = numeric[assignments[valid]]
        for parent_index in np.unique(assignments[valid]):
            indices = np.flatnonzero(assignments == parent_index)
            history[indices] = np.arange(len(indices), dtype=np.float64)
        break
    time_value = np.zeros(rows, dtype=np.float64)
    for column in table.columns:
        if column.kind is ColumnKind.TIME and column.column_id in local_values:
            time_value = np.asarray(local_values[column.column_id], dtype=np.float64)
            break
    if np.any(time_value):
        time_value = (time_value - time_value.mean()) / max(
            float(time_value.std()), 1e-6
        )
    history /= max(float(history.max(initial=0.0)), 1.0)
    return evaluate_forest(
        forest,
        {
            "self_latent_0": own[:, 0],
            "parent_latent_0": parent_latent,
            "parent_feature_0": parent_feature,
            "time": time_value,
            "history": history,
        },
        row_count=rows,
    )


def _numeric_feature(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind in {"b", "i", "u", "f"}:
        numeric = raw.astype(np.float64)
    else:
        _categories, numeric = np.unique(raw.astype(str), return_inverse=True)
        numeric = numeric.astype(np.float64)
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=8.0, neginf=-8.0)
    return (numeric - numeric.mean()) / max(float(numeric.std()), 1e-6)


def _causal_context(
    schema: PhysicalSchema,
    table: PhysicalTable,
    latents: LatentRegistry,
    relations: Mapping[str, np.ndarray],
) -> np.ndarray:
    pieces = [latents.table(table.table_id).values]
    for foreign_key in schema.foreign_keys:
        if foreign_key.child_table_id != table.table_id:
            continue
        assignments = relations[foreign_key.foreign_key_id]
        parent = latents.table(foreign_key.parent_table_id).values
        selected = np.zeros((len(assignments), parent.shape[1]), dtype=np.float64)
        valid = assignments >= 0
        selected[valid] = parent[assignments[valid]]
        pieces.append(selected)
    return np.concatenate(pieces, axis=1)


def _generate_time(
    *,
    schema: PhysicalSchema,
    table: PhysicalTable,
    column: PhysicalColumn,
    table_plan: TableMechanismPlan,
    relations: Mapping[str, np.ndarray],
    generated_tables: Mapping[str, TableData],
    db_start: int,
    db_end: int,
) -> np.ndarray:
    seed = table_plan.temporal_seed + column.ordinal * 104_729
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    rows = table_plan.population.row_count
    scale = table_plan.parameter_map["time_scale_seconds"]
    # Reserve one second of headroom per TIME_LAGGED chain link so a child can
    # always land strictly after its parent while every value stays inside the
    # database calendar interval.
    depth = _max_time_lagged_depth(schema, table.table_id)
    eff_end = db_end - depth
    if eff_end - db_start < 2:
        eff_end = db_start + 2

    incoming = tuple(
        foreign_key
        for foreign_key in schema.foreign_keys
        if foreign_key.child_table_id == table.table_id
    )
    if table_plan.temporal_family is TemporalFamily.TIME_LAGGED:
        for foreign_key in incoming:
            parent_table = schema.table(foreign_key.parent_table_id)
            parent_time = next(
                (
                    item
                    for item in parent_table.columns
                    if item.kind is ColumnKind.TIME
                ),
                None,
            )
            if parent_time is None or parent_table.table_id not in generated_tables:
                continue
            assignments = relations[foreign_key.foreign_key_id]
            parent_values = generated_tables[parent_table.table_id].column(
                parent_time.column_id
            )
            values = np.full(rows, db_start, dtype=np.int64)
            valid = assignments >= 0
            lag = np.maximum(1, rng.lognormal(np.log(scale), 0.8, size=rows))
            raw = parent_values[assignments[valid]] + lag[valid].astype(np.int64)
            values[valid] = np.minimum(raw, eff_end)
            assert_within_interval(
                values,
                db_start,
                db_end,
                context=(
                    f"generated TIME {table.table_id}.{column.column_id}"
                ),
            )
            return values

    grouping = next(
        (
            relations[foreign_key.foreign_key_id]
            for foreign_key in incoming
            if foreign_key.relation_strategy != "lookup_assignment"
        ),
        np.zeros(rows, dtype=np.int64),
    )
    mechanism = table_plan.event_mechanism
    values = np.empty(rows, dtype=np.int64)
    for parent_index in np.unique(grouping):
        indices = np.flatnonzero(grouping == parent_index)
        group_base = db_start + int(rng.integers(0, max(1, eff_end - db_start)))
        span = eff_end - group_base
        ticks = _mechanism_ticks(rng, mechanism, table_plan, len(indices), span)
        values[indices] = (group_base + ticks).astype(np.int64)
    assert_within_interval(
        values,
        db_start,
        db_end,
        context=f"generated TIME {table.table_id}.{column.column_id}",
    )
    return values


def _max_time_lagged_depth(schema: PhysicalSchema, table_id: str) -> int:
    """Longest temporal-child chain starting at ``table_id``.

    Temporal children (EVENT/DETAIL tables carrying a TIME column) may be
    ``TIME_LAGGED``, so each chain link needs one second of headroom at the top
    of the calendar window to keep child times strictly after their parents.
    ``depth(parent) == 1 + max(depth(children))`` guarantees the parent's
    effective end stays at least one second below every child's.
    """

    children = [
        foreign_key.child_table_id
        for foreign_key in schema.foreign_keys
        if foreign_key.parent_table_id == table_id
        and any(
            column.kind is ColumnKind.TIME
            for column in schema.table(foreign_key.child_table_id).columns
        )
    ]
    if not children:
        return 1
    return 1 + max(_max_time_lagged_depth(schema, child) for child in children)


def _mechanism_ticks(
    rng: np.random.Generator,
    mechanism: EventTemporalMechanism,
    table_plan: TableMechanismPlan,
    count: int,
    span: int,
) -> np.ndarray:
    """Return ``count`` ascending float offsets inside ``[0, span]``."""
    if count == 0:
        return np.empty(0, dtype=np.float64)
    if mechanism is EventTemporalMechanism.STATIONARY:
        ticks = np.sort(rng.uniform(size=count))
        return ticks * span
    if mechanism is EventTemporalMechanism.BURST:
        return _burst_ticks(rng, table_plan, count, span)
    if mechanism is EventTemporalMechanism.CHURN:
        exponent = rng.uniform(
            table_plan.parameter_map["churn_exponent_min"],
            table_plan.parameter_map["churn_exponent_max"],
        )
        ticks = rng.uniform(size=count) ** exponent
        return ticks * span
    if mechanism is EventTemporalMechanism.SEASONAL:
        return _seasonal_ticks(rng, table_plan, count, span)
    raise ValueError(f"unsupported event temporal mechanism: {mechanism}")


def _burst_ticks(
    rng: np.random.Generator,
    table_plan: TableMechanismPlan,
    count: int,
    span: int,
) -> np.ndarray:
    """Cluster most arrivals into a few narrow bursts over a quiet background.

    The background pool keeps a sparse uniform floor, while each burst draws
    from a narrow sub-window, producing dense activity separated by long
    silence inside the group's calendar window.
    """
    max_clusters = int(table_plan.parameter_map["burst_max_clusters"])
    width_min = table_plan.parameter_map["burst_cluster_width_min"]
    width_max = table_plan.parameter_map["burst_cluster_width_max"]
    cluster_count = int(rng.integers(1, max_clusters + 1))
    weights = np.concatenate(
        ([0.3], np.full(cluster_count, 0.7 / cluster_count))
    )
    allocation = rng.multinomial(count, weights)
    points: list[np.ndarray] = []
    background = allocation[0]
    if background > 0:
        points.append(rng.uniform(size=background) * span)
    for index in range(1, cluster_count + 1):
        members = allocation[index]
        if members <= 0:
            continue
        center = rng.uniform(0.05, 0.95) * span
        half_width = rng.uniform(width_min, width_max) * span
        low = max(0.0, center - half_width)
        high = min(float(span), center + half_width)
        points.append(rng.uniform(low, high, size=members))
    ticks = np.concatenate(points) if points else np.empty(0)
    ticks = np.clip(ticks, 0.0, float(span))
    return np.sort(ticks)


def _seasonal_ticks(
    rng: np.random.Generator,
    table_plan: TableMechanismPlan,
    count: int,
    span: int,
) -> np.ndarray:
    """Arrivals modulated by a sinusoidal calendar intensity (rejection sampling).

    The period is sampled per group and the phase uniformly; a bounded candidate
    pool is topped up with uniform draws so exactly ``count`` points are emitted
    inside ``[0, span]``.
    """
    period_days = int(
        rng.integers(
            int(table_plan.parameter_map["seasonal_period_days_min"]),
            int(table_plan.parameter_map["seasonal_period_days_max"]) + 1,
        )
    )
    period = period_days * 86_400
    phase = rng.uniform(0.0, period)
    amplitude = rng.uniform(0.5, 0.9)

    def intensity(x: np.ndarray) -> np.ndarray:
        return 1.0 + amplitude * np.sin(2.0 * np.pi * (x + phase) / period)

    max_intensity = 1.0 + amplitude
    candidate_count = int(np.ceil(count * (1.0 + amplitude))) + 8
    candidates = rng.uniform(size=candidate_count) * span
    accepted = candidates[
        rng.uniform(size=candidate_count)
        < (intensity(candidates) / max_intensity)
    ]
    if accepted.size < count:
        topup = rng.uniform(size=count - accepted.size) * span
        accepted = np.concatenate([accepted, topup])
    ticks = np.sort(accepted[:count])
    return np.clip(ticks, 0.0, float(span))



__all__ = ["generate_table_features"]
