"""Run a stronger OOD benchmark by holding out entire model families.

Usage:
    python scripts/run_family_split_benchmark.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manifoldguard.data import load_score_csv
from manifoldguard.evaluation import evaluate_experiment_split
from manifoldguard.splits import group_indices_by_family, select_family_holdout_indices

DATASET = REPO_ROOT / "datasets" / "lm_eval_real" / "scores.csv"
OUT_DIR = REPO_ROOT / "results" / "family_split"
SEEDS = [0, 1, 2, 3, 4]

RANK = 3
ENSEMBLE_SIZE = 5
OBSERVED_FRACTION = 0.5
ALPHA = 0.1
TEST_FRACTION = 0.25
MIN_FAMILIES = 2


def main() -> None:
    score_matrix = load_score_csv(DATASET)
    family_to_indices = group_indices_by_family(score_matrix.model_names)

    print(f"Dataset:  {DATASET}")
    print(f"Matrix:   {score_matrix.values.shape[0]} models x {score_matrix.values.shape[1]} benchmarks")
    print(f"Families: {len(family_to_indices)}\n")

    family_counts_path = OUT_DIR / "family_counts.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with family_counts_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "num_models"])
        for family, indices in family_to_indices.items():
            writer.writerow([family, len(indices)])

    rows: list[dict[str, str | float | int]] = []
    for seed in SEEDS:
        train_idx, test_idx, held_out_families = select_family_holdout_indices(
            score_matrix.model_names,
            test_fraction=TEST_FRACTION,
            min_families=MIN_FAMILIES,
            seed=seed,
        )
        metrics = evaluate_experiment_split(
            matrix=score_matrix.values,
            train_indices=train_idx,
            test_indices=test_idx,
            rank=RANK,
            ensemble_size=ENSEMBLE_SIZE,
            observed_fraction=OBSERVED_FRACTION,
            alpha=ALPHA,
            seed=seed,
        )
        held_out_text = "|".join(held_out_families)
        rows.append(
            {
                "seed": seed,
                "held_out_families": held_out_text,
                "num_train_models": metrics.num_train_models,
                "num_test_models": metrics.num_test_models,
                "completion_mae": metrics.completion_mae,
                "failure_auc": metrics.failure_auc,
                "conformal_coverage": metrics.conformal_coverage,
                "conformal_quantile": metrics.conformal_quantile,
            }
        )
        print(
            f"seed={seed}  holdout={held_out_text}  "
            f"MAE={metrics.completion_mae:.4f}  "
            f"AUC={metrics.failure_auc:.4f}  "
            f"cov={metrics.conformal_coverage:.4f}"
        )

    by_seed_path = OUT_DIR / "results_by_seed.csv"
    with by_seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "held_out_families",
                "num_train_models",
                "num_test_models",
                "completion_mae",
                "failure_auc",
                "conformal_coverage",
                "conformal_quantile",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["seed"],
                    row["held_out_families"],
                    row["num_train_models"],
                    row["num_test_models"],
                    f"{row['completion_mae']:.6f}",
                    f"{row['failure_auc']:.6f}",
                    f"{row['conformal_coverage']:.6f}",
                    f"{row['conformal_quantile']:.6f}",
                ]
            )

    metric_keys = [
        "completion_mae",
        "failure_auc",
        "conformal_coverage",
        "conformal_quantile",
    ]
    summary_path = OUT_DIR / "summary_table.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std"])
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in rows], dtype=float)
            writer.writerow([key, f"{np.mean(values):.6f}", f"{np.std(values, ddof=1):.6f}"])

    print(f"\nResults written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
