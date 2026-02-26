import numpy as np
import pytest

from manifoldguard.ood import (
    LatentDistribution,
    mahalanobis_distance,
    observation_coverage_features,
    residual_features,
    summary_variance_features,
)


def test_residual_features_values() -> None:
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    y_hat = np.array([1.5, 1.5, 2.0], dtype=float)

    energy, max_abs = residual_features(y, y_hat)
    assert energy == pytest.approx(0.5)
    assert max_abs == pytest.approx(1.0)


def test_mahalanobis_distance_identity_precision() -> None:
    dist = LatentDistribution(mean=np.zeros(2), precision=np.eye(2))
    value = mahalanobis_distance(np.array([3.0, 4.0]), dist)
    assert value == pytest.approx(5.0)


def test_summary_variance_features() -> None:
    variance = np.array([0.2, 0.1, 0.3, 0.05], dtype=float)
    mean_var, max_var = summary_variance_features(variance, np.array([1, 3], dtype=int))
    assert mean_var == pytest.approx(0.075)
    assert max_var == pytest.approx(0.1)


def test_observation_coverage_well_conditioned() -> None:
    """An orthonormal V_S should have condition number 1 and min_sv > 0."""
    rng = np.random.default_rng(0)
    # Build V with 10 benchmarks, rank 3; use the first 3 rows as an orthonormal basis.
    V = rng.normal(size=(10, 3))
    Q, _ = np.linalg.qr(V[:3].T)  # Q is 3x3 orthonormal
    V[:3] = Q.T  # first 3 rows are orthonormal
    min_sv, cond = observation_coverage_features(V, np.array([0, 1, 2]))
    assert min_sv == pytest.approx(1.0, abs=1e-6)
    assert cond == pytest.approx(1.0, abs=1e-6)


def test_observation_coverage_rank_deficient() -> None:
    """Duplicate rows yield min_sv = 0 and very high condition number."""
    V = np.ones((5, 2))  # all rows identical -> rank-1 V_S
    indices = np.array([0, 1, 2])
    min_sv, cond = observation_coverage_features(V, indices)
    assert min_sv == pytest.approx(0.0, abs=1e-6)
    assert cond > 1e8
