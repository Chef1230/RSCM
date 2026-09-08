"""Deterministic missingness overlay primitives."""
from __future__ import annotations
import numpy as np
from rdb_prior.compilation.model import PhysicalColumn
from rdb_prior.priors.model import MissingnessPlan

def _numeric(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"b", "i", "u", "f"}:
        return np.nan_to_num(values.astype(np.float64), nan=0.0)
    if values.dtype.kind in {"U", "S"}:
        _, codes = np.unique(values.astype(str), return_inverse=True)
        return codes.astype(np.float64)
    return np.zeros(len(values), dtype=np.float64)

def sample_missing_mask(plan: MissingnessPlan, values: np.ndarray, *, driver_values: tuple[np.ndarray, ...] = (), time_values: np.ndarray | None = None, latent_values: np.ndarray | None = None, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=bool)
    rate = float(np.clip(plan.rate, 0.0, 1.0))
    family = plan.family.lower()
    if rate <= 0.0 and family not in {"block", "block_missingness"}:
        return np.zeros(n, dtype=bool)
    if family == "mcar":
        probability = np.full(n, rate, dtype=np.float64)
    elif family == "mar":
        signal = np.zeros(n, dtype=np.float64)
        for item in driver_values:
            if len(item) == n:
                signal += _numeric(item)
        if driver_values:
            signal /= max(1, len(driver_values))
        signal = (signal - signal.mean()) / max(float(signal.std()), 1e-8)
        intercept = np.log(rate / max(1e-6, 1.0-rate))
        probability = 1.0 / (1.0 + np.exp(-(intercept + 0.8*signal)))
    elif family == "mnar":
        signal = _numeric(values)
        if latent_values is not None and len(latent_values) == n:
            signal = 0.7 * signal + 0.3 * _numeric(latent_values)
        signal = (signal - signal.mean()) / max(float(signal.std()), 1e-8)
        intercept = np.log(rate / max(1e-6, 1.0-rate))
        probability = 1.0 / (1.0 + np.exp(-(intercept + 0.9*signal)))
    elif family in {"block", "block_missingness"}:
        probability = np.zeros(n, dtype=np.float64)
        block = max(1, int(plan.block_size or round(n * max(rate, 0.05))))
        start = int(rng.integers(0, max(1, n)))
        probability[start:min(n, start+block)] = 1.0
    elif family in {"time", "time_dependent"}:
        if time_values is None or len(time_values) != n:
            raise ValueError("time-dependent missingness requires aligned time values")
        signal = _numeric(time_values)
        signal = (signal - signal.mean()) / max(float(signal.std()), 1e-8)
        intercept = np.log(rate / max(1e-6, 1.0-rate))
        probability = 1.0 / (1.0 + np.exp(-(intercept + 0.8*signal)))
    else:
        raise ValueError(f"unsupported nuisance missingness family: {plan.family}")
    mask = rng.random(n) < np.clip(probability, 0.0, 1.0)
    if mask.all():
        mask[int(rng.integers(0, n))] = False
    return mask

def apply_missing_mask(values: np.ndarray, column: PhysicalColumn, mask: np.ndarray) -> np.ndarray:
    if not column.nullable or not np.any(mask):
        return values
    raw = np.asarray(values)
    if raw.dtype.kind in {"U", "S"}:
        width = max(1, raw.dtype.itemsize // np.dtype("U1").itemsize)
        result = raw.astype(f"<U{width}", copy=True)
        result[mask] = ""
        return result
    result = raw.astype(np.float64, copy=True)
    result[mask] = np.nan
    return result

__all__ = ["sample_missing_mask", "apply_missing_mask"]
