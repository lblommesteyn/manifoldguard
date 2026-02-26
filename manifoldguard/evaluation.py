from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from manifoldguard._utils import quantile_higher as _quantile_higher
from manifoldguard.conformal import empirical_coverage, split_conformal_quantile
from manifoldguard.ensemble import EnsembleMF, predict_new_model, train_ensemble
from manifoldguard.episodes import Episode, simulate_new_model_episodes
from manifoldguard.inference import loo_observed_residuals
from manifoldguard.ood import observation_coverage_features, residual_features, summary_variance_features


Array = np.ndarray

FEATURE_NAMES = (
    "residual_energy_obs",
    "max_abs_residual_obs",
    "mahalanobis_u",
    "mean_predictive_variance_hidden",
    "max_predictive_variance_hidden",
    "loo_mean_abs_error_obs",
    "loo_max_abs_error_obs",
    "min_singular_value_obs",  # large = good coverage -> low MAE (negative predictor)
    "condition_number_obs",  # large = poor coverage -> high MAE (positive predictor)
)


@dataclass(frozen=True)
class EpisodeResult:
    features: Array
    hidden_mae: float
    hidden_predictions: Array
    hidden_targets: Array
    hidden_abs_residuals: Array


@dataclass(frozen=True)
class ExperimentMetrics:
    completion_mae: float
    failure_auc: float
    conformal_coverage: float
    conformal_quantile: float
    failure_threshold: float
    num_episodes: int
    num_train_models: int
    num_test_models: int


def evaluate_experiment(
    matrix: Array,
    rank: int = 4,
    ensemble_size: int = 5,
    mf_reg: float = 1e-2,
    ridge: float = 1e-2,
    lr: float = 5e-2,
    epochs: int = 700,
    episodes_per_model: int = 3,
    observed_fraction: float = 0.5,
    model_test_fraction: float = 0.25,
    seed: int = 0,
    alpha: float = 0.1,
    device: str | None = None,
) -> ExperimentMetrics:
    """Run completion + failure-risk + conformal evaluation end-to-end.

    The matrix is split into train models (used to fit the ensemble) and test
    models (held out entirely during training). Episodes are simulated only on
    test models, so OOD features are evaluated on truly unseen rows.
    """
    n_models = matrix.shape[0]
    rng = np.random.default_rng(seed)
    model_order = rng.permutation(n_models)

    n_test = max(1, int(round(model_test_fraction * n_models)))
    n_train = n_models - n_test
    if n_train < 2:
        raise ValueError(
            f"model_test_fraction={model_test_fraction} leaves only {n_train} train "
            "model(s); need at least 2. Reduce model_test_fraction or provide more rows."
        )

    train_indices = model_order[:n_train]
    test_indices = model_order[n_train:]
    train_matrix = matrix[np.sort(train_indices)]

    ensemble = train_ensemble(
        matrix=train_matrix,
        ensemble_size=ensemble_size,
        rank=rank,
        reg=mf_reg,
        lr=lr,
        epochs=epochs,
        seed=seed,
        device=device,
    )

    test_matrix = matrix[np.sort(test_indices)]
    episodes = simulate_new_model_episodes(
        matrix=test_matrix,
        episodes_per_model=episodes_per_model,
        observed_fraction=observed_fraction,
        seed=seed + 1,
    )
    if not episodes:
        raise ValueError("No valid episodes could be constructed from the test models.")

    episode_results = [_evaluate_episode(ep, ensemble, ridge) for ep in episodes]
    episode_groups = np.asarray([ep.model_index for ep in episodes], dtype=int)
    hidden_maes = np.asarray([r.hidden_mae for r in episode_results], dtype=float)
    completion_mae = float(np.mean(hidden_maes))

    failure_threshold = float(_quantile_higher(hidden_maes, 0.8))
    failure_labels = (hidden_maes >= failure_threshold).astype(int)
    feature_matrix = np.vstack([r.features for r in episode_results])
    failure_auc = _fit_failure_auc(feature_matrix, failure_labels, episode_groups, seed=seed)

    conformal_coverage, conformal_quantile = _evaluate_conformal(
        results=episode_results,
        episode_groups=episode_groups,
        alpha=alpha,
        seed=seed,
    )

    return ExperimentMetrics(
        completion_mae=completion_mae,
        failure_auc=failure_auc,
        conformal_coverage=conformal_coverage,
        conformal_quantile=conformal_quantile,
        failure_threshold=failure_threshold,
        num_episodes=len(episode_results),
        num_train_models=int(n_train),
        num_test_models=int(n_test),
    )


def _evaluate_episode(episode: Episode, ensemble: EnsembleMF, ridge: float) -> EpisodeResult:
    prediction = predict_new_model(
        ensemble=ensemble,
        observed_indices=episode.observed_indices,
        observed_values=episode.observed_values,
        ridge=ridge,
    )

    observed_pred = prediction.mean_prediction[episode.observed_indices]
    hidden_pred = prediction.mean_prediction[episode.hidden_indices]
    hidden_true = episode.hidden_values

    residual_energy, max_abs_residual = residual_features(episode.observed_values, observed_pred)
    maha = float(np.mean(prediction.mahalanobis_distances))
    mean_var, max_var = summary_variance_features(prediction.predictive_variance, episode.hidden_indices)

    # LOO residuals: average across ensemble members for a stable estimate.
    loo_errors_per_member = [
        loo_observed_residuals(
            V=member.V,
            observed_indices=episode.observed_indices,
            observed_values=episode.observed_values,
            ridge=ridge,
        )
        for member in ensemble.members
    ]
    loo_errors = np.mean(np.vstack(loo_errors_per_member), axis=0)
    loo_mean = float(np.mean(loo_errors))
    loo_max = float(np.max(loo_errors)) if loo_errors.size > 0 else 0.0

    # Geometric coverage: average min singular value and condition number of V_S
    # across ensemble members. These are primary signals for ill-conditioning.
    coverage_features = [
        observation_coverage_features(member.V, episode.observed_indices)
        for member in ensemble.members
    ]
    min_sv = float(np.mean([f[0] for f in coverage_features]))
    cond_num = float(np.mean([f[1] for f in coverage_features]))

    feature_vector = np.asarray(
        [residual_energy, max_abs_residual, maha, mean_var, max_var, loo_mean, loo_max, min_sv, cond_num],
        dtype=float,
    )
    abs_hidden_residuals = np.abs(hidden_pred - hidden_true)
    hidden_mae = float(np.mean(abs_hidden_residuals))

    return EpisodeResult(
        features=feature_vector,
        hidden_mae=hidden_mae,
        hidden_predictions=hidden_pred,
        hidden_targets=hidden_true,
        hidden_abs_residuals=abs_hidden_residuals,
    )


def _fit_failure_auc(features: Array, labels: Array, groups: Array, seed: int) -> float:
    """Estimate failure-detection AUC via grouped cross-validation.

    Episodes from the same model are kept in the same fold to avoid leakage.
    """
    if np.unique(labels).size < 2:
        return float("nan")

    unique_groups = np.unique(groups)
    n_folds = min(5, unique_groups.size)
    if n_folds < 2:
        return float("nan")

    clf = LogisticRegression(max_iter=2000, random_state=seed + 17)

    # Stratify by label while keeping each model in one fold.
    try:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + 11)
        folds = list(splitter.split(features, labels, groups))
    except ValueError:
        # Fallback for edge cases where grouped stratification is infeasible.
        splitter = GroupKFold(n_splits=n_folds)
        folds = list(splitter.split(features, labels, groups))

    fold_aucs: list[float] = []
    for train_idx, test_idx in folds:
        y_train, y_test = labels[train_idx], labels[test_idx]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        clf.fit(features[train_idx], y_train)
        probs = clf.predict_proba(features[test_idx])[:, 1]
        fold_aucs.append(float(roc_auc_score(y_test, probs)))

    return float(np.mean(fold_aucs)) if fold_aucs else float("nan")


def _evaluate_conformal(
    results: list[EpisodeResult],
    episode_groups: Array,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    calibration_idx, test_idx = _grouped_calibration_test_indices(episode_groups, seed=seed + 29)

    calibration_residuals = np.concatenate([results[i].hidden_abs_residuals for i in calibration_idx])
    q = split_conformal_quantile(calibration_residuals, alpha=alpha)

    covered = 0.0
    total = 0
    for i in test_idx:
        result = results[i]
        coverage = empirical_coverage(result.hidden_predictions, result.hidden_targets, q=q)
        covered += coverage * len(result.hidden_targets)
        total += len(result.hidden_targets)

    return float(covered / max(total, 1)), q


def _grouped_calibration_test_indices(groups: Array, seed: int) -> tuple[Array, Array]:
    """Split episode indices into calibration/test sets with disjoint model groups."""
    group_ids = np.asarray(groups, dtype=int)
    unique_groups = np.unique(group_ids)
    if unique_groups.size < 2:
        raise ValueError("Need at least two unique models for grouped conformal split.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    n_cal_groups = int(np.floor(unique_groups.size * 0.5))
    n_cal_groups = int(np.clip(n_cal_groups, 1, unique_groups.size - 1))

    calibration_groups = shuffled[:n_cal_groups]
    calibration_mask = np.isin(group_ids, calibration_groups)

    calibration_idx = np.flatnonzero(calibration_mask)
    test_idx = np.flatnonzero(~calibration_mask)
    if calibration_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Grouped calibration/test split produced an empty partition.")

    return calibration_idx, test_idx
