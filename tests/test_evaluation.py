import numpy as np
import pytest

from manifoldguard.evaluation import _fit_failure_auc, _grouped_calibration_test_indices


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

    auc = _fit_failure_auc(feature_arr, label_arr, group_arr, seed=7)
    assert np.isfinite(auc)
    assert 0.5 <= auc <= 1.0
