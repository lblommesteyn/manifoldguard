"""Run the experiment pipeline on real data across multiple seeds and save results.

Usage:
    python scripts/run_real_data_benchmarks.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manifoldguard.data import load_score_csv
from manifoldguard.evaluation import evaluate_experiment

DATASET = REPO_ROOT / "datasets" / "lm_eval_real" / "scores.csv"
OUT_DIR = REPO_ROOT / "results" / "real_data_benchmark"
SEEDS = [0, 1, 2, 3, 4]

# Pipeline hyperparameters (match defaults, rank=3 suits 7 benchmarks).
RANK = 3
ENSEMBLE_SIZE = 5
OBSERVED_FRACTION = 0.5
MODEL_TEST_FRACTION = 0.25
ALPHA = 0.1

METRIC_KEYS = [
    "completion_mae",
    "failure_auc",
    "conformal_coverage",
    "conformal_quantile",
]


def run_seed(matrix, seed: int) -> dict[str, float]:
    metrics = evaluate_experiment(
        matrix=matrix,
        rank=RANK,
        ensemble_size=ENSEMBLE_SIZE,
        observed_fraction=OBSERVED_FRACTION,
        model_test_fraction=MODEL_TEST_FRACTION,
        alpha=ALPHA,
        seed=seed,
    )
    return {
        "completion_mae": metrics.completion_mae,
        "failure_auc": metrics.failure_auc,
        "conformal_coverage": metrics.conformal_coverage,
        "conformal_quantile": metrics.conformal_quantile,
    }


def main() -> None:
    score_matrix = load_score_csv(DATASET)
    print(f"Dataset: {DATASET}")
    print(f"Matrix:  {score_matrix.values.shape[0]} models x {score_matrix.values.shape[1]} benchmarks")
    print(f"Seeds:   {SEEDS}\n")

    rows: list[dict[str, float]] = []
    for seed in SEEDS:
        print(f"  seed={seed} ... ", end="", flush=True)
        result = run_seed(score_matrix.values, seed)
        rows.append(result)
        print(
            f"MAE={result['completion_mae']:.4f}  "
            f"AUC={result['failure_auc']:.4f}  "
            f"cov={result['conformal_coverage']:.4f}  "
            f"q={result['conformal_quantile']:.4f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # results_by_seed.csv
    by_seed_path = OUT_DIR / "results_by_seed.csv"
    with by_seed_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed"] + METRIC_KEYS)
        for seed, row in zip(SEEDS, rows):
            writer.writerow([seed] + [f"{row[k]:.6f}" for k in METRIC_KEYS])

    # summary_table.csv
    import numpy as np

    summary_path = OUT_DIR / "summary_table.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        for k in METRIC_KEYS:
            values = [row[k] for row in rows]
            writer.writerow([k, f"{np.mean(values):.6f}", f"{np.std(values, ddof=1):.6f}"])

    print(f"\nResults written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
