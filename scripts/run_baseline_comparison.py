"""Compare ManifoldGuard against simple baselines on real data.

Baselines:
  mean_fill       -- predict column mean from training matrix
  nearest_neighbor -- predict hidden scores from the most similar training model
  mf_only         -- MF predictions, no failure detection, no conformal
  mf_no_conformal -- MF + failure detection, no conformal intervals
  manifoldguard   -- full pipeline (MF + failure detection + conformal)

All methods are evaluated on the same episodes for a fair comparison.
Results are written to results/baseline_comparison/.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manifoldguard._utils import quantile_higher as _quantile_higher
from manifoldguard.conformal import empirical_coverage, split_conformal_quantile
from manifoldguard.data import load_score_csv
from manifoldguard.ensemble import predict_new_model, train_ensemble
from manifoldguard.episodes import simulate_new_model_episodes
from manifoldguard.evaluation import (
    _evaluate_conformal,
    _fit_failure_auc,
    _evaluate_episode,
)

DATASET = REPO_ROOT / "datasets" / "lm_eval_real" / "scores_expanded.csv"
OUT_DIR = REPO_ROOT / "results" / "baseline_comparison"
SEEDS = [0, 1, 2, 3, 4]

RANK = 3
ENSEMBLE_SIZE = 5
OBSERVED_FRACTION = 0.5
MODEL_TEST_FRACTION = 0.25
ALPHA = 0.1


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

def _mean_fill_mae(train_matrix: np.ndarray, episodes) -> float:
    """Predict each hidden benchmark as the column mean of the training matrix."""
    col_means = np.nanmean(train_matrix, axis=0)
    maes = []
    for ep in episodes:
        preds = col_means[ep.hidden_indices]
        maes.append(float(np.mean(np.abs(preds - ep.hidden_values))))
    return float(np.mean(maes))


def _nearest_neighbor_mae(train_matrix: np.ndarray, episodes) -> float:
    """Predict hidden scores using the most similar training model.

    Similarity is L2 distance on the observed benchmarks (ignoring NaN in
    the training row for that benchmark).
    """
    maes = []
    for ep in episodes:
        obs_idx = ep.observed_indices
        obs_val = ep.observed_values

        # For each training model, compute L2 distance on the observed indices.
        dists = []
        for row in train_matrix:
            train_obs = row[obs_idx]
            valid = ~np.isnan(train_obs)
            if valid.sum() == 0:
                dists.append(np.inf)
            else:
                dists.append(float(np.sqrt(np.mean((obs_val[valid] - train_obs[valid]) ** 2))))

        nearest = int(np.argmin(dists))
        preds = train_matrix[nearest][ep.hidden_indices]

        # If the nearest neighbor is missing some hidden benchmarks, fall back
        # to the column mean for those entries.
        col_means = np.nanmean(train_matrix, axis=0)
        nan_mask = np.isnan(preds)
        preds = np.where(nan_mask, col_means[ep.hidden_indices], preds)

        maes.append(float(np.mean(np.abs(preds - ep.hidden_values))))
    return float(np.mean(maes))


# ---------------------------------------------------------------------------
# Per-seed evaluation
# ---------------------------------------------------------------------------

def run_seed(matrix: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_models = matrix.shape[0]
    model_order = rng.permutation(n_models)

    n_test = max(1, int(round(MODEL_TEST_FRACTION * n_models)))
    n_train = n_models - n_test
    train_indices = model_order[:n_train]
    test_indices = model_order[n_train:]
    train_matrix = matrix[np.sort(train_indices)]
    test_matrix = matrix[np.sort(test_indices)]

    episodes = simulate_new_model_episodes(
        matrix=test_matrix,
        episodes_per_model=3,
        observed_fraction=OBSERVED_FRACTION,
        seed=seed + 1,
    )

    ensemble = train_ensemble(
        matrix=train_matrix,
        ensemble_size=ENSEMBLE_SIZE,
        rank=RANK,
        seed=seed,
    )

    # --- simple baselines ---
    mean_mae = _mean_fill_mae(train_matrix, episodes)
    nn_mae = _nearest_neighbor_mae(train_matrix, episodes)

    # --- MF episode results (shared for the three MF variants) ---
    episode_results = [_evaluate_episode(ep, ensemble, ridge=1e-2) for ep in episodes]
    episode_groups = np.asarray([ep.model_index for ep in episodes], dtype=int)
    hidden_maes = np.asarray([r.hidden_mae for r in episode_results], dtype=float)
    mf_mae = float(np.mean(hidden_maes))

    # --- failure detection ---
    failure_threshold = float(_quantile_higher(hidden_maes, 0.8))
    failure_labels = (hidden_maes >= failure_threshold).astype(int)
    feature_matrix = np.vstack([r.features for r in episode_results])
    failure_auc = _fit_failure_auc(feature_matrix, failure_labels, episode_groups, seed=seed)

    # --- conformal ---
    conformal_coverage, conformal_quantile = _evaluate_conformal(
        results=episode_results,
        episode_groups=episode_groups,
        alpha=ALPHA,
        seed=seed,
    )

    return {
        "mean_fill_mae":          mean_mae,
        "nn_mae":                 nn_mae,
        "mf_only_mae":            mf_mae,
        "mf_no_conformal_mae":    mf_mae,
        "manifoldguard_mae":      mf_mae,
        "mf_no_conformal_auc":    failure_auc,
        "manifoldguard_auc":      failure_auc,
        "manifoldguard_coverage": conformal_coverage,
        "manifoldguard_quantile": conformal_quantile,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    score_matrix = load_score_csv(DATASET)
    matrix = score_matrix.values
    print(f"Dataset: {DATASET}")
    print(f"Matrix:  {matrix.shape[0]} models x {matrix.shape[1]} benchmarks")
    print(f"Seeds:   {SEEDS}\n")

    rows = []
    for seed in SEEDS:
        print(f"  seed={seed} ...", end=" ", flush=True)
        result = run_seed(matrix, seed)
        rows.append(result)
        print(f"done  MF MAE={result['mf_only_mae']:.4f}  AUC={result['manifoldguard_auc']:.4f}")

    def _ms(key):
        vals = [r[key] for r in rows]
        return np.mean(vals), np.std(vals, ddof=1)

    nan = float("nan")

    methods = [
        ("Mean fill",                 _ms("mean_fill_mae"),       (nan, nan),              (nan, nan)),
        ("Nearest neighbor",          _ms("nn_mae"),               (nan, nan),              (nan, nan)),
        ("MF only",                   _ms("mf_only_mae"),          (nan, nan),              (nan, nan)),
        ("MF + failure detection",    _ms("mf_no_conformal_mae"),  _ms("mf_no_conformal_auc"), (nan, nan)),
        ("ManifoldGuard (full)",      _ms("manifoldguard_mae"),    _ms("manifoldguard_auc"), _ms("manifoldguard_coverage")),
    ]

    # --- print table ---
    header = f"{'Method':<26}  {'MAE':>14}  {'Failure AUC':>17}  {'Coverage':>10}"
    print("\n" + header)
    print("-" * len(header))
    for name, (mae_m, mae_s), (auc_m, auc_s), (cov_m, cov_s) in methods:
        mae_str = f"{mae_m:.4f} +/- {mae_s:.4f}"
        auc_str = f"{auc_m:.4f} +/- {auc_s:.4f}" if not np.isnan(auc_m) else "N/A"
        cov_str = f"{cov_m:.4f} +/- {cov_s:.4f}" if not np.isnan(cov_m) else "N/A"
        print(f"{name:<26}  {mae_str:>14}  {auc_str:>17}  {cov_str:>10}")

    # --- save CSV ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # per-seed
    by_seed_keys = [
        "mean_fill_mae", "nn_mae", "mf_only_mae",
        "mf_no_conformal_mae", "mf_no_conformal_auc",
        "manifoldguard_mae", "manifoldguard_auc",
        "manifoldguard_coverage", "manifoldguard_quantile",
    ]
    with (OUT_DIR / "results_by_seed.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed"] + by_seed_keys)
        for seed, row in zip(SEEDS, rows):
            writer.writerow([seed] + [f"{row[k]:.6f}" for k in by_seed_keys])

    # summary
    with (OUT_DIR / "summary_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "completion_mae_mean", "completion_mae_std",
                         "failure_auc_mean", "failure_auc_std",
                         "conformal_coverage_mean", "conformal_coverage_std"])
        for name, (mae_m, mae_s), (auc_m, auc_s), (cov_m, cov_s) in methods:
            writer.writerow([
                name,
                f"{mae_m:.6f}", f"{mae_s:.6f}",
                "" if np.isnan(auc_m) else f"{auc_m:.6f}",
                "" if np.isnan(auc_s) else f"{auc_s:.6f}",
                "" if np.isnan(cov_m) else f"{cov_m:.6f}",
                "" if np.isnan(cov_s) else f"{cov_s:.6f}",
            ])

    print(f"\nResults written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
