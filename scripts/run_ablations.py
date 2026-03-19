"""Ablation study: rank, ensemble size, observed fraction, and OOD feature subsets.

Usage:
    python scripts/run_ablations.py
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
from manifoldguard.evaluation import FEATURE_NAMES, evaluate_experiment

DATASET = REPO_ROOT / "datasets" / "lm_eval_real" / "scores.csv"
OUT_DIR = REPO_ROOT / "results" / "ablations"
SEEDS = [0, 1, 2]

# Defaults (matching run_real_data_benchmarks.py).
DEFAULTS = dict(rank=3, ensemble_size=5, observed_fraction=0.5, model_test_fraction=0.25, alpha=0.1)

# Feature groups by index into FEATURE_NAMES.
GEOMETRY_COLS = [i for i, n in enumerate(FEATURE_NAMES) if n in ("min_singular_value_obs", "condition_number_obs")]
NON_GEOMETRY_COLS = [i for i in range(len(FEATURE_NAMES)) if i not in GEOMETRY_COLS]

FEATURE_SUBSETS = {
    "all_features": None,  # None = use all
    "geometry_only": GEOMETRY_COLS,
    "all_minus_geometry": NON_GEOMETRY_COLS,
}


def run_config(matrix, seeds, **kwargs) -> dict[str, float]:
    """Run evaluate_experiment across seeds, return mean metrics."""
    results = []
    for seed in seeds:
        m = evaluate_experiment(matrix=matrix, seed=seed, **kwargs)
        results.append({
            "completion_mae": m.completion_mae,
            "failure_auc": m.failure_auc,
            "conformal_coverage": m.conformal_coverage,
        })
    return {k: float(np.mean([r[k] for r in results])) for k in results[0]}


def main() -> None:
    sm = load_score_csv(DATASET)
    matrix = sm.values
    print(f"Dataset: {matrix.shape[0]} models x {matrix.shape[1]} benchmarks\n")

    rows: list[dict[str, str]] = []

    # --- 1. Rank ablation ---
    print("=== Rank ablation ===")
    for rank in [2, 3, 4, 5]:
        label = f"rank={rank}"
        print(f"  {label} ...", end="", flush=True)
        r = run_config(matrix, SEEDS, **{**DEFAULTS, "rank": rank})
        print(f"  MAE={r['completion_mae']:.4f}  AUC={r['failure_auc']:.4f}  cov={r['conformal_coverage']:.4f}")
        rows.append({"ablation": "rank", "setting": str(rank), **{k: f"{v:.4f}" for k, v in r.items()}})

    # --- 2. Ensemble size ablation ---
    print("\n=== Ensemble size ablation ===")
    for es in [1, 3, 5, 7]:
        label = f"ensemble_size={es}"
        print(f"  {label} ...", end="", flush=True)
        r = run_config(matrix, SEEDS, **{**DEFAULTS, "ensemble_size": es})
        print(f"  MAE={r['completion_mae']:.4f}  AUC={r['failure_auc']:.4f}  cov={r['conformal_coverage']:.4f}")
        rows.append({"ablation": "ensemble_size", "setting": str(es), **{k: f"{v:.4f}" for k, v in r.items()}})

    # --- 3. Observed fraction ablation ---
    print("\n=== Observed fraction ablation ===")
    for frac in [0.3, 0.5, 0.7]:
        label = f"observed_fraction={frac}"
        print(f"  {label} ...", end="", flush=True)
        r = run_config(matrix, SEEDS, **{**DEFAULTS, "observed_fraction": frac})
        print(f"  MAE={r['completion_mae']:.4f}  AUC={r['failure_auc']:.4f}  cov={r['conformal_coverage']:.4f}")
        rows.append({"ablation": "observed_fraction", "setting": str(frac), **{k: f"{v:.4f}" for k, v in r.items()}})

    # --- 4. OOD feature subset ablation ---
    print("\n=== OOD feature subset ablation ===")
    for name, cols in FEATURE_SUBSETS.items():
        print(f"  {name} ...", end="", flush=True)
        r = run_config(matrix, SEEDS, **{**DEFAULTS, "feature_columns": cols})
        print(f"  MAE={r['completion_mae']:.4f}  AUC={r['failure_auc']:.4f}  cov={r['conformal_coverage']:.4f}")
        rows.append({"ablation": "feature_subset", "setting": name, **{k: f"{v:.4f}" for k, v in r.items()}})

    # --- Write CSV ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ablation_table.csv"
    fieldnames = ["ablation", "setting", "completion_mae", "failure_auc", "conformal_coverage"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAblation table written to {out_path}")


if __name__ == "__main__":
    main()
