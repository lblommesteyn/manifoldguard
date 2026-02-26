from __future__ import annotations

import numpy as np


Array = np.ndarray


def infer_latent_u_ridge(
    V: Array,
    observed_indices: Array,
    observed_values: Array,
    ridge: float = 1e-2,
) -> Array:
    """
    Infer latent vector u for a new model using closed-form ridge regression.

    Solves:
        u = argmin ||V_S u - y_S||^2 + ridge * ||u||^2
    """
    if V.ndim != 2:
        raise ValueError("V must be 2D.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")

    indices = np.asarray(observed_indices, dtype=int)
    values = np.asarray(observed_values, dtype=float)
    if indices.ndim != 1:
        raise ValueError("observed_indices must be 1D.")
    if values.ndim != 1:
        raise ValueError("observed_values must be 1D.")
    if len(indices) != len(values):
        raise ValueError("observed_indices and observed_values must have equal length.")
    if len(indices) == 0:
        raise ValueError("at least one observed entry is required.")

    v_obs = V[indices]
    rank = V.shape[1]
    a = v_obs.T @ v_obs + ridge * np.eye(rank, dtype=float)
    b = v_obs.T @ values
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, b, rcond=None)[0]


def predict_scores(u: Array, V: Array) -> Array:
    if u.ndim != 1:
        raise ValueError("u must be 1D.")
    if V.ndim != 2:
        raise ValueError("V must be 2D.")
    if u.shape[0] != V.shape[1]:
        raise ValueError("u dimension must match V latent dimension.")
    return V @ u


def loo_observed_residuals(
    V: Array,
    observed_indices: Array,
    observed_values: Array,
    ridge: float = 1e-2,
) -> Array:
    """
    Leave-one-out absolute prediction errors on the observed subset.

    For each j in S, infers u from S \\ {j} via ridge regression, then predicts
    the held-out entry and returns |pred - true|.  When |S| < 2, returns zeros
    (the LOO estimate is undefined with a single observation).

    These errors are a proxy for how well the latent representation generalises
    to unseen benchmarks: high LOO error on observed entries signals that the
    inferred u is unreliable, which correlates with high error on hidden entries.
    """
    if V.ndim != 2:
        raise ValueError("V must be 2D.")

    indices = np.asarray(observed_indices, dtype=int)
    values = np.asarray(observed_values, dtype=float)
    n = len(indices)

    if n < 2:
        return np.zeros(max(n, 0), dtype=float)

    errors = np.empty(n, dtype=float)
    keep = np.ones(n, dtype=bool)
    for k in range(n):
        keep[k] = False
        u_loo = infer_latent_u_ridge(V, indices[keep], values[keep], ridge)
        pred = float(V[indices[k]] @ u_loo)
        errors[k] = abs(pred - values[k])
        keep[k] = True

    return errors
