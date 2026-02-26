from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


Array = np.ndarray


@dataclass(frozen=True)
class LatentDistribution:
    mean: Array
    precision: Array


def fit_latent_distribution(U: Array) -> LatentDistribution:
    if U.ndim != 2:
        raise ValueError("U must be 2D.")
    estimator = LedoitWolf().fit(U)
    return LatentDistribution(mean=estimator.location_, precision=estimator.precision_)


def mahalanobis_distance(u: Array, distribution: LatentDistribution) -> float:
    if u.ndim != 1:
        raise ValueError("u must be 1D.")
    delta = u - distribution.mean
    distance_sq = float(delta.T @ distribution.precision @ delta)
    return float(np.sqrt(max(distance_sq, 0.0)))


def residual_features(observed_values: Array, observed_predictions: Array) -> tuple[float, float]:
    y = np.asarray(observed_values, dtype=float)
    y_hat = np.asarray(observed_predictions, dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError("observed_values and observed_predictions must have same shape.")
    residual = y - y_hat
    energy = float(np.mean(residual**2))
    max_abs = float(np.max(np.abs(residual)))
    return energy, max_abs


def summary_variance_features(variance_by_benchmark: Array, target_indices: Array) -> tuple[float, float]:
    variance = np.asarray(variance_by_benchmark, dtype=float)
    indices = np.asarray(target_indices, dtype=int)
    if variance.ndim != 1:
        raise ValueError("variance_by_benchmark must be 1D.")
    if indices.ndim != 1:
        raise ValueError("target_indices must be 1D.")

    if len(indices) == 0:
        values = variance
    else:
        values = variance[indices]
    return float(np.mean(values)), float(np.max(values))


def observation_coverage_features(V: Array, observed_indices: Array) -> tuple[float, float]:
    """Latent-space coverage of the observed benchmark subset.

    Computes the singular values of V_S = V[observed_indices] and returns:
      - ``min_sv``: smallest singular value relevant to the rank
        (0 when fewer observations than rank -> unidentifiable).
      - ``condition_number``: ratio max_sv / max(min_sv, eps).

    Low ``min_sv`` / high ``condition_number`` means the observed benchmarks fail
    to span the full latent space, so the inferred ``u`` is poorly determined and
    hidden-benchmark errors will be large. These are primary geometric OOD signals.
    """
    if V.ndim != 2:
        raise ValueError("V must be 2D.")
    indices = np.asarray(observed_indices, dtype=int)
    rank = V.shape[1]
    v_obs = V[indices]
    singular_values = np.linalg.svd(v_obs, compute_uv=False)

    if len(singular_values) >= rank:
        min_sv = float(singular_values[rank - 1])
    else:
        min_sv = 0.0

    max_sv = float(singular_values[0]) if singular_values.size > 0 else 1.0
    cond = max_sv / max(min_sv, 1e-10)
    return min_sv, cond
