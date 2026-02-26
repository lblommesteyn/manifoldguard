from __future__ import annotations

import math

import numpy as np

from manifoldguard._utils import quantile_higher as _quantile_higher


Array = np.ndarray


def split_conformal_quantile(abs_residuals: Array, alpha: float = 0.1) -> float:
    """
    Finite-sample split conformal quantile for target coverage (1 - alpha).
    """
    residuals = np.asarray(abs_residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        raise ValueError("abs_residuals must contain at least one finite value.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")

    n = residuals.size
    rank = math.ceil((n + 1) * (1.0 - alpha))
    quantile_level = min(max(rank / n, 0.0), 1.0)
    return float(_quantile_higher(residuals, quantile_level))


def interval_bounds(predictions: Array, q: float) -> tuple[Array, Array]:
    preds = np.asarray(predictions, dtype=float)
    return preds - q, preds + q


def empirical_coverage(predictions: Array, targets: Array, q: float) -> float:
    preds = np.asarray(predictions, dtype=float)
    y = np.asarray(targets, dtype=float)
    if preds.shape != y.shape:
        raise ValueError("predictions and targets must have the same shape.")
    low, high = interval_bounds(preds, q)
    covered = (y >= low) & (y <= high)
    return float(np.mean(covered))


