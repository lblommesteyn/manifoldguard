import numpy as np
import pytest

from manifoldguard.data import generate_synthetic_scores
from manifoldguard.evaluation import (
    _fit_failure_auc,
    _grouped_calibration_test_indices,
    _grouped_failure_probabilities,
    evaluate_experiment_split,
)


def test_grouped_calibration_split_has_no_group_overlap() -> None:
    groups = np.repeat(np.arange(8), 3)
    calibration_idx, test_idx = _grouped_calibration_test_indices(groups, seed=123)

    assert len(set(calibration_idx).intersection(set(test_idx))) == 0
    assert len(calibration_idx) + len(test_idx) == len(groups)

    calibration_groups = set(groups[calibration_idx].tolist())
    test_groups = set(groups[test_idx].tolist())
    assert calibration_groups.isdisjoint(test_groups)


def test_grouped_calibration_split_requires_two_groups() -> None:
    with pytest.raises(ValueError):
        _grouped_calibration_test_indices(np.zeros(5, dtype=int), seed=0)


def test_fit_failure_auc_grouped_cv_runs() -> None:
    rng = np.random.default_rng(0)
    groups_list = []
    labels_list = []
    features_list = []

    # 10 groups, each with 3 episodes. Half positive, half negative.
    for group in range(10):
        label = 1 if group < 5 else 0
        for _ in range(3):
            signal = 1.0 if label == 1 else -1.0
            features = np.array([signal, signal * 0.5, signal * 2.0]) + rng.normal(scale=0.2, size=3)
            groups_list.append(group)
            labels_list.append(label)
            features_list.append(features)

    group_arr = np.asarray(groups_list, dtype=int)
    label_arr = np.asarray(labels_list, dtype=int)
    feature_arr = np.vstack(features_list)

    auc, _, _ = _fit_failure_auc(feature_arr, label_arr, group_arr, seed=7)
    assert np.isfinite(auc)
    assert 0.5 <= auc <= 1.0


def test_grouped_failure_probabilities_return_out_of_fold_scores() -> None:
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(10), 3)
    labels = np.repeat([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 3)
    features = np.column_stack(
        [
            labels + rng.normal(scale=0.1, size=labels.size),
            (1 - labels) + rng.normal(scale=0.1, size=labels.size),
        ]
    )

    probabilities = _grouped_failure_probabilities(features, labels, groups, seed=9)

    assert probabilities.shape == labels.shape
    assert np.isfinite(probabilities).sum() >= labels.size // 2
    assert np.nanmin(probabilities) >= 0.0
    assert np.nanmax(probabilities) <= 1.0


def test_evaluate_experiment_accepts_feature_subsets() -> None:
    score_matrix = generate_synthetic_scores(
        num_models=20,
        num_benchmarks=6,
        rank=2,
        noise_std=0.05,
        missing_rate=0.1,
        seed=0,
    )

    from manifoldguard.evaluation import evaluate_experiment

    metrics = evaluate_experiment(
        matrix=score_matrix.values,
        rank=2,
        ensemble_size=1,
        epochs=10,
        episodes_per_model=3,
        observed_fraction=0.5,
        model_test_fraction=0.4,
        feature_columns=[0, 1, 7],
        seed=3,
    )

    assert np.isfinite(metrics.completion_mae)
    cov = metrics.conformal_coverage
    if isinstance(cov, dict):
        for val in cov.values():
            assert np.isfinite(val)
    else:
        assert np.isfinite(cov)
    assert metrics.num_episodes > 0


def test_evaluate_experiment_split_runs_with_explicit_indices() -> None:
    score_matrix = generate_synthetic_scores(
        num_models=20,
        num_benchmarks=6,
        rank=2,
        noise_std=0.05,
        missing_rate=0.1,
        seed=2,
    )

    metrics = evaluate_experiment_split(
        matrix=score_matrix.values,
        train_indices=np.arange(0, 10),
        test_indices=np.arange(10, 20),
        rank=2,
        ensemble_size=1,
        epochs=10,
        episodes_per_model=3,
        observed_fraction=0.5,
        seed=5,
    )

    assert metrics.num_train_models == 10
    assert metrics.num_test_models == 10
    assert metrics.num_episodes > 0


def test_evaluate_experiment_rejects_empty_feature_subsets() -> None:
    score_matrix = generate_synthetic_scores(
        num_models=20,
        num_benchmarks=6,
        rank=2,
        noise_std=0.05,
        missing_rate=0.1,
        seed=1,
    )

    from manifoldguard.evaluation import evaluate_experiment

    with pytest.raises(ValueError, match="feature_columns must contain at least one column index"):
        evaluate_experiment(
            matrix=score_matrix.values,
            rank=2,
            ensemble_size=1,
            epochs=10,
            episodes_per_model=3,
            observed_fraction=0.5,
            model_test_fraction=0.4,
            feature_columns=[],
            seed=4,
        )
