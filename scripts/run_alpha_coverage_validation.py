"""Validate conformal coverage across multiple alpha values on real data.

Usage:
    python scripts/run_alpha_coverage_validation.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional visualization dependency
    plt = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manifoldguard.data import load_score_csv
from manifoldguard.evaluation import evaluate_experiment

DATASET = REPO_ROOT / "datasets" / "lm_eval_real" / "scores.csv"
OUT_DIR = REPO_ROOT / "results" / "alpha_validation"
DEFAULT_ALPHAS = [0.05, 0.10, 0.20, 0.30]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

RANK = 3
ENSEMBLE_SIZE = 5
OBSERVED_FRACTION = 0.5
MODEL_TEST_FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate conformal coverage across multiple alpha values.")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Miscoverage levels to validate.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds to average over.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR, help="Directory for validation artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_matrix = load_score_csv(DATASET)
    matrix = score_matrix.values

    print(f"Dataset: {DATASET}")
    print(f"Matrix:  {matrix.shape[0]} models x {matrix.shape[1]} benchmarks")
    print(f"Alphas:  {args.alphas}")
    print(f"Seeds:   {args.seeds}\n")

    rows: list[dict[str, float | int]] = []
    for alpha in args.alphas:
        target = 1.0 - alpha
        print(f"alpha={alpha:.2f}  target coverage={target:.2f}")
        for seed in args.seeds:
            metrics = evaluate_experiment(
                matrix=matrix,
                rank=RANK,
                ensemble_size=ENSEMBLE_SIZE,
                observed_fraction=OBSERVED_FRACTION,
                model_test_fraction=MODEL_TEST_FRACTION,
                alpha=alpha,
                seed=seed,
            )
            deviation = metrics.conformal_coverage - target
            rows.append(
                {
                    "alpha": alpha,
                    "target_coverage": target,
                    "seed": seed,
                    "empirical_coverage": metrics.conformal_coverage,
                    "coverage_deviation": deviation,
                    "conformal_quantile": metrics.conformal_quantile,
                }
            )
            print(
                f"  seed={seed}  empirical={metrics.conformal_coverage:.4f}  "
                f"delta={deviation:+.4f}  q={metrics.conformal_quantile:.4f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_results_by_seed(args.output_dir / "results_by_seed.csv", rows)
    summary_rows = _write_summary(args.output_dir / "summary_table.csv", rows)
    plot_path = args.output_dir / "coverage_vs_target.png"
    _write_plot(plot_path, summary_rows)

    print(f"\nResults written to {args.output_dir}/")
    if plt is None:
        print("matplotlib not installed; skipped coverage plot.")


def _write_results_by_seed(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "alpha",
                "target_coverage",
                "seed",
                "empirical_coverage",
                "coverage_deviation",
                "conformal_quantile",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    f"{float(row['alpha']):.2f}",
                    f"{float(row['target_coverage']):.2f}",
                    int(row["seed"]),
                    f"{float(row['empirical_coverage']):.6f}",
                    f"{float(row['coverage_deviation']):+.6f}",
                    f"{float(row['conformal_quantile']):.6f}",
                ]
            )


def _write_summary(path: Path, rows: list[dict[str, float | int]]) -> list[dict[str, float]]:
    summary_rows: list[dict[str, float]] = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "alpha",
                "target_coverage",
                "empirical_coverage_mean",
                "empirical_coverage_std",
                "coverage_deviation_mean",
                "coverage_deviation_std",
                "conformal_quantile_mean",
                "conformal_quantile_std",
            ]
        )
        for alpha in sorted({float(row["alpha"]) for row in rows}):
            alpha_rows = [row for row in rows if float(row["alpha"]) == alpha]
            empirical = np.asarray([float(row["empirical_coverage"]) for row in alpha_rows], dtype=float)
            deviation = np.asarray([float(row["coverage_deviation"]) for row in alpha_rows], dtype=float)
            quantiles = np.asarray([float(row["conformal_quantile"]) for row in alpha_rows], dtype=float)
            target = float(alpha_rows[0]["target_coverage"])
            summary = {
                "alpha": alpha,
                "target_coverage": target,
                "empirical_coverage_mean": float(np.mean(empirical)),
                "empirical_coverage_std": float(np.std(empirical, ddof=1)),
                "coverage_deviation_mean": float(np.mean(deviation)),
                "coverage_deviation_std": float(np.std(deviation, ddof=1)),
                "conformal_quantile_mean": float(np.mean(quantiles)),
                "conformal_quantile_std": float(np.std(quantiles, ddof=1)),
            }
            writer.writerow(
                [
                    f"{summary['alpha']:.2f}",
                    f"{summary['target_coverage']:.2f}",
                    f"{summary['empirical_coverage_mean']:.6f}",
                    f"{summary['empirical_coverage_std']:.6f}",
                    f"{summary['coverage_deviation_mean']:+.6f}",
                    f"{summary['coverage_deviation_std']:.6f}",
                    f"{summary['conformal_quantile_mean']:.6f}",
                    f"{summary['conformal_quantile_std']:.6f}",
                ]
            )
            summary_rows.append(summary)
    return summary_rows


def _write_plot(path: Path, summary_rows: list[dict[str, float]]) -> None:
    if plt is None:
        return

    targets = np.asarray([row["target_coverage"] for row in summary_rows], dtype=float)
    empirical = np.asarray([row["empirical_coverage_mean"] for row in summary_rows], dtype=float)
    stds = np.asarray([row["empirical_coverage_std"] for row in summary_rows], dtype=float)
    alphas = np.asarray([row["alpha"] for row in summary_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(
        targets,
        empirical,
        yerr=stds,
        fmt="o-",
        color="#E86535",
        ecolor="#0D2233",
        elinewidth=1.2,
        capsize=4,
        markersize=7,
    )
    ax.plot([0.65, 1.0], [0.65, 1.0], linestyle="--", linewidth=1.2, color="#18A8B8")
    for x, y, alpha in zip(targets, empirical, alphas):
        ax.text(x + 0.006, y + 0.004, f"a={alpha:.2f}", fontsize=9.5, color="#0D2233")

    ax.set_xlabel("Target coverage (1 - alpha)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Conformal coverage across alpha values")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.65, 1.0)
    ax.set_ylim(0.65, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
