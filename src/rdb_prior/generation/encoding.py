"""Shared physical-column encoding for feature and temporal generators."""

from __future__ import annotations

import numpy as np

from rdb_prior.compilation.model import PhysicalColumn, PhysicalDataType
from rdb_prior.schema.spec import TableRole


def encode_feature_score(
    score: np.ndarray,
    column: PhysicalColumn,
    role: TableRole,
    rng: np.random.Generator,
    *,
    cardinality: int,
    db_start: int,
    db_end: int,
    categorical_dirichlet_alpha: float,
    categorical_signal_strength: float,
    missing_rate: float,
    long_tail_enabled: bool = False,
    long_tail_alpha: float = 1.0,
) -> np.ndarray:
    """Encode one causal score without exposing generator-private context."""
    transformed = apply_long_tail(
        score,
        enabled=long_tail_enabled,
        alpha=long_tail_alpha,
    )
    return apply_missing(
        encode_signal(
            transformed,
            column,
            role,
            rng,
            cardinality=cardinality,
            db_start=db_start,
            db_end=db_end,
            categorical_dirichlet_alpha=categorical_dirichlet_alpha,
            categorical_signal_strength=categorical_signal_strength,
        ),
        column,
        rng,
        missing_rate,
    )


def apply_long_tail(
    score: np.ndarray,
    *,
    enabled: bool,
    alpha: float,
) -> np.ndarray:
    """Apply a bounded signed power transform for planned long-tail columns."""
    values = np.nan_to_num(np.asarray(score, dtype=np.float64), nan=0.0)
    if not enabled:
        return values
    exponent = max(float(alpha), 1.0)
    return np.sign(values) * np.power(np.abs(values), exponent)


def encode_signal(
    signal: np.ndarray,
    column: PhysicalColumn,
    role: TableRole,
    rng: np.random.Generator,
    *,
    cardinality: int,
    db_start: int,
    db_end: int,
    categorical_dirichlet_alpha: float,
    categorical_signal_strength: float,
) -> np.ndarray:
    """Convert a numeric causal score to an anonymous physical column."""
    signal = np.nan_to_num(np.asarray(signal, dtype=np.float64), nan=0.0)
    if column.unique:
        order = np.argsort(signal, kind="stable")
        unique = np.empty(len(signal), dtype=np.int64)
        unique[order] = np.arange(len(signal), dtype=np.int64)
        if column.data_type is PhysicalDataType.TEXT:
            return np.char.add("v", unique.astype(str))
        return unique
    if column.data_type is PhysicalDataType.DOUBLE:
        return signal.astype(np.float64)
    if column.data_type is PhysicalDataType.INTEGER:
        if role is TableRole.LOOKUP:
            return softmax_codes(
                signal, min(cardinality, len(signal)), rng,
                dirichlet_alpha=categorical_dirichlet_alpha,
                signal_strength=categorical_signal_strength,
            )
        return np.rint(signal * float(rng.uniform(2.0, 20.0))).astype(np.int64)
    if column.data_type is PhysicalDataType.BOOLEAN:
        threshold = float(np.quantile(signal, rng.uniform(0.3, 0.7)))
        return (signal > threshold).astype(np.int8)
    if column.data_type is PhysicalDataType.TEXT:
        codes = softmax_codes(
            signal, min(cardinality, len(signal)), rng,
            dirichlet_alpha=categorical_dirichlet_alpha,
            signal_strength=categorical_signal_strength,
        )
        return np.char.add("v", codes.astype(str))
    if column.data_type is PhysicalDataType.TIMESTAMP:
        return (db_start + signal * 86_400).clip(db_start, db_end).astype(np.int64)
    raise ValueError(f"unsupported physical data type: {column.data_type}")


def softmax_codes(
    signal: np.ndarray,
    cardinality: int,
    rng: np.random.Generator,
    *,
    dirichlet_alpha: float,
    signal_strength: float,
) -> np.ndarray:
    """Sample categorical codes with a Dirichlet prior and Gumbel-max."""
    rows = len(signal)
    if rows == 0:
        return np.empty(0, dtype=np.int64)
    cardinality = max(1, min(cardinality, rows))
    if cardinality == 1:
        return np.zeros(rows, dtype=np.int64)
    class_prior = rng.dirichlet(np.full(cardinality, float(dirichlet_alpha)))
    log_prior = np.log(np.maximum(class_prior, 1e-12))
    projection = rng.normal(size=cardinality)
    output = np.empty(rows, dtype=np.int64)
    for start in range(0, rows, 2048):
        stop = min(rows, start + 2048)
        uniform = rng.random((stop - start, cardinality))
        gumbel = -np.log(-np.log(np.clip(uniform, 1e-12, 1.0 - 1e-12)))
        logits = signal_strength * signal[start:stop, None] * projection
        output[start:stop] = np.argmax(
            logits + log_prior + gumbel, axis=1
        ).astype(np.int64)
    return output


def apply_missing(
    values: np.ndarray,
    column: PhysicalColumn,
    rng: np.random.Generator,
    missing_rate: float,
) -> np.ndarray:
    """Apply planned nullable masking while preserving one observed value."""
    if not column.nullable or missing_rate <= 0:
        return values
    missing = rng.random(len(values)) < missing_rate
    if len(values) > 0 and missing.all():
        missing[rng.integers(len(values))] = False
    if values.dtype.kind in {"U", "S"}:
        width = max(1, values.dtype.itemsize // np.dtype("U1").itemsize)
        result = values.astype(f"<U{width}", copy=True)
        result[missing] = ""
        return result
    result = values.astype(np.float64, copy=True)
    result[missing] = np.nan
    return result


__all__ = [
    "apply_long_tail",
    "apply_missing",
    "encode_feature_score",
    "encode_signal",
    "softmax_codes",
]
