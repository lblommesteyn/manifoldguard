"""Estimate evaluation cost savings from risk-based early stopping.

Usage:
    python scripts/run_cost_savings_analysis.py
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
from manifoldguard.data import load_score_csv
from manifoldguard.ensemble import train_ensemble
from manifoldguard.episodes import simulate_new_model_episodes
from manifoldguard.evaluation import _evaluate_episode, _grouped_failure_probabilities

DATASET = REPO_ROOT / "datasets" / "lm_eval_real" / "scores.csv"
OUT_DIR = REPO_ROOT / "results" / "cost_savings"
SEEDS = [0, 1, 2, 3, 4]
RISK_THRESHOLDS = [0.20, 0.30, 0.35, 0.40, 0.50]

RANK = 3
ENSEMBLE_SIZE = 5
OBSERVED_FRACTION = 0.5
MODEL_TEST_FRACTION = 0.25


def _random_split_indices(n_models: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_models)
    n_test = max(1, int(round(test_fraction * n_models)))
    n_train = n_models - n_test
    return np.sort(order[:n_train]), np.sort(order[n_train:])


def main() -> None:
    score_matrix = load_score_csv(DATASET)
    matrix = score_matrix.values
    print(f"Dataset: {DATASET}")
    print(f"Matrix:  {matrix.shape[0]} models x {matrix.shape[1]} benchmarks")
    print(f"Risk thresholds: {RISK_THRESHOLDS}\n")

    rows: list[dict[str, float | int]] = []
    for seed in SEEDS:
        train_idx, test_idx = _random_split_indices(matrix.shape[0], MODEL_TEST_FRACTION, seed=seed)
        train_matrix = matrix[train_idx]
        test_matrix = matrix[test_idx]

        ensemble = train_ensemble(
            matrix=train_matrix,
            ensemble_size=ENSEMBLE_SIZE,
            rank=RANK,
            seed=seed,
        )
        episodes = simulate_new_model_episodes(
            matrix=test_matrix,
            episodes_per_model=3,
            observed_fraction=OBSERVED_FRACTION,
            seed=seed + 1,
        )
        episode_results = [_evaluate_episode(ep, ensemble, ridge=1e-2) for ep in episodes]
        hidden_maes = np.asarray([r.hidden_mae for r in episode_results], dtype=float)
        hidden_counts = np.asarray([len(r.hidden_targets) for r in episode_results], dtype=int)
        feature_matrix = np.vstack([r.features for r in episode_results])
        episode_groups = np.asarray([ep.model_index for ep in episodes], dtype=int)

        failure_threshold = float(_quantile_higher(hidden_maes, 0.8))
        failure_labels = (hidden_maes >= failure_threshold).astype(int)
        risk_probabilities = _grouped_failure_probabilities(
            feature_matrix,
            failure_labels,
            episode_groups,
            seed=seed,
        )

        total_hidden_benchmarks = int(np.sum(hidden_counts))
        print(f"seed={seed}  threshold@20pct failure MAE={failure_threshold:.4f}")
        for risk_threshold in RISK_THRESHOLDS:
            accept_mask = np.isfinite(risk_probabilities) & (risk_probabilities <= risk_threshold)
            accepted_hidden = int(np.sum(hidden_counts[accept_mask]))
            accepted_rate = float(np.mean(accept_mask))
            hidden_saved_fraction = accepted_hidden / max(total_hidden_benchmarks, 1)
            accepted_mae = float(np.mean(hidden_maes[accept_mask])) if np.any(accept_mask) else float("nan")
            accepted_failure_rate = (
                float(np.mean(failure_labels[accept_mask])) if np.any(accept_mask) else float("nan")
            )
            rows.append(
                {
                    "seed": seed,
                    "risk_threshold": risk_threshold,
                    "episodes_accepted": accepted_rate,
                    "benchmarks_avoided_fraction": hidden_saved_fraction,
                    "accepted_mae": accepted_mae,
                    "accepted_failure_rate": accepted_failure_rate,
                }
            )
            print(
                f"  risk<={risk_threshold:.2f}  "
                f"accept={accepted_rate:.3f}  "
                f"benchmarks_avoided={hidden_saved_fraction:.3f}  "
                f"accepted_MAE={accepted_mae:.4f}"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    by_seed_path = OUT_DIR / "results_by_seed.csv"
    with by_seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "risk_threshold",
                "episodes_accepted",
                "benchmarks_avoided_fraction",
                "accepted_mae",
                "accepted_failure_rate",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["seed"],
                    f"{row['risk_threshold']:.2f}",
                    f"{row['episodes_accepted']:.6f}",
                    f"{row['benchmarks_avoided_fraction']:.6f}",
                    "" if np.isnan(row["accepted_mae"]) else f"{row['accepted_mae']:.6f}",
                    "" if np.isnan(row["accepted_failure_rate"]) else f"{row['accepted_failure_rate']:.6f}",
                ]
            )

    summary_path = OUT_DIR / "summary_table.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "risk_threshold",
                "episodes_accepted_mean",
                "episodes_accepted_std",
                "benchmarks_avoided_fraction_mean",
                "benchmarks_avoided_fraction_std",
                "accepted_mae_mean",
                "accepted_mae_std",
                "accepted_failure_rate_mean",
                "accepted_failure_rate_std",
            ]
        )
        for threshold in RISK_THRESHOLDS:
            threshold_rows = [row for row in rows if row["risk_threshold"] == threshold]
            writer.writerow(
                [
                    f"{threshold:.2f}",
                    *_mean_std(threshold_rows, "episodes_accepted"),
                    *_mean_std(threshold_rows, "benchmarks_avoided_fraction"),
                    *_mean_std(threshold_rows, "accepted_mae"),
                    *_mean_std(threshold_rows, "accepted_failure_rate"),
                ]
            )

    print(f"\nResults written to {OUT_DIR}/")


def _mean_std(rows: list[dict[str, float | int]], key: str) -> tuple[str, str]:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return "", ""
    return f"{np.mean(values):.6f}", f"{np.std(values, ddof=1):.6f}"


if __name__ == "__main__":
    main()
