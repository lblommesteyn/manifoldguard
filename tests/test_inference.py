import numpy as np
import pytest

from manifoldguard.inference import infer_latent_u_ridge, loo_observed_residuals, predict_scores


def test_infer_latent_u_ridge_recovers_true_u() -> None:
    rng = np.random.default_rng(7)
    V = rng.normal(size=(10, 3))
    u_true = np.array([0.4, -1.1, 2.2], dtype=float)
    observed_idx = np.array([0, 2, 3, 5, 9], dtype=int)
    observed_values = V[observed_idx] @ u_true

    u_hat = infer_latent_u_ridge(V=V, observed_indices=observed_idx, observed_values=observed_values, ridge=1e-8)
    assert np.allclose(u_hat, u_true, atol=1e-5)


def test_predict_scores_shape() -> None:
    rng = np.random.default_rng(3)
    V = rng.normal(size=(6, 2))
    u = np.array([1.0, -0.5], dtype=float)
    pred = predict_scores(u=u, V=V)
    assert pred.shape == (6,)


def test_loo_observed_residuals_near_zero_on_exact_low_rank() -> None:
    """When data is exactly low-rank, LOO errors should be small."""
    rng = np.random.default_rng(11)
    V = rng.normal(size=(10, 3))
    u_true = np.array([1.0, -0.5, 0.8])
    observed_idx = np.arange(8, dtype=int)
    observed_values = V[observed_idx] @ u_true  # exact low-rank, no noise

    errors = loo_observed_residuals(V, observed_idx, observed_values, ridge=1e-8)
    assert errors.shape == (8,)
    assert float(np.max(errors)) < 0.1


def test_loo_observed_residuals_length_one_returns_zeros() -> None:
    rng = np.random.default_rng(0)
    V = rng.normal(size=(5, 2))
    errors = loo_observed_residuals(V, np.array([2]), np.array([0.5]), ridge=1e-2)
    assert errors.shape == (1,)
    assert float(errors[0]) == pytest.approx(0.0)


def test_loo_errors_higher_for_noisy_data() -> None:
    """LOO errors on noisy observations should exceed those on clean observations."""
    rng = np.random.default_rng(42)
    V = rng.normal(size=(12, 4))
    u_true = rng.normal(size=(4,))
    observed_idx = np.arange(10, dtype=int)

    clean_values = V[observed_idx] @ u_true
    noisy_values = clean_values + rng.normal(scale=2.0, size=len(observed_idx))

    clean_errors = loo_observed_residuals(V, observed_idx, clean_values, ridge=1e-2)
    noisy_errors = loo_observed_residuals(V, observed_idx, noisy_values, ridge=1e-2)
    assert float(np.mean(noisy_errors)) > float(np.mean(clean_errors))
