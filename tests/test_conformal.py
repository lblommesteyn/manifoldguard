import numpy as np
import pytest

from manifoldguard.conformal import empirical_coverage, interval_bounds, split_conformal_quantile


def test_split_conformal_quantile() -> None:
    residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    q = split_conformal_quantile(residuals, alpha=0.1)
    assert q == pytest.approx(5.0)


def test_interval_bounds() -> None:
    preds = np.array([0.0, 1.0, 2.0], dtype=float)
    low, high = interval_bounds(preds, q=0.2)
    assert np.allclose(low, [-0.2, 0.8, 1.8])
    assert np.allclose(high, [0.2, 1.2, 2.2])


def test_empirical_coverage() -> None:
    preds = np.array([1.0, 2.0, 3.0], dtype=float)
    y = np.array([1.1, 1.9, 3.5], dtype=float)
    cov = empirical_coverage(predictions=preds, targets=y, q=0.2)
    assert cov == pytest.approx(2.0 / 3.0)
