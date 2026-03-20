"""Generate chart-driven conference PNG figures from tracked result artifacts.

Usage:
    python scripts/generate_conference_figures.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - import guard for optional dependency
    raise SystemExit("matplotlib is required for figure generation. Install with `pip install -e \".[viz]\"`.") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manifoldguard._utils import quantile_higher as _quantile_higher
from manifoldguard.data import load_score_csv
from manifoldguard.ensemble import train_ensemble
from manifoldguard.episodes import simulate_new_model_episodes
from manifoldguard.evaluation import _evaluate_episode, _grouped_failure_probabilities


BENCHMARK_SUMMARY = REPO_ROOT / "results" / "real_data_benchmark" / "summary_table.csv"
BASELINE_SUMMARY = REPO_ROOT / "results" / "baseline_comparison" / "summary_table.csv"
ABLATION_TABLE = REPO_ROOT / "results" / "ablations" / "ablation_table.csv"
ALPHA_SUMMARY = REPO_ROOT / "results" / "alpha_validation" / "summary_table.csv"
FAMILY_SUMMARY = REPO_ROOT / "results" / "family_split" / "summary_table.csv"
FAMILY_COUNTS = REPO_ROOT / "results" / "family_split" / "family_counts.csv"
COST_SUMMARY = REPO_ROOT / "results" / "cost_savings" / "summary_table.csv"
DEMO_REPORT = REPO_ROOT / "results" / "demo" / "demo_report.json"
SCORE_CSV = REPO_ROOT / "datasets" / "lm_eval_real" / "scores.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "conference_figures"

RISK_SEEDS = [0, 1, 2, 3, 4]

COLORS = {
    "ink": "#1F2937",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "mae": "#E76F51",
    "auc": "#2A9D8F",
    "coverage": "#3B82F6",
    "quantile": "#A855F7",
    "gold": "#F4A261",
    "red": "#C0392B",
    "navy": "#264653",
    "gray": "#9CA3AF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate conference-ready chart figures.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for the PNG outputs.")
    parser.add_argument("--dpi", type=int, default=260, help="PNG DPI.")
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "axes.axisbelow": True,
        }
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    return float(text) if text else float("nan")


def ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, data_dir


def save_figure(fig: mpl.figure.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: mpl.axes.Axes, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(length=0)


def load_benchmark_metrics() -> dict[str, float]:
    return {row["metric"]: as_float(row["mean"]) for row in read_csv_rows(BENCHMARK_SUMMARY)}


def load_baseline_rows() -> list[dict[str, float | str]]:
    rows = []
    for row in read_csv_rows(BASELINE_SUMMARY):
        rows.append(
            {
                "method": row["method"],
                "completion_mae_mean": as_float(row["completion_mae_mean"]),
                "completion_mae_std": as_float(row["completion_mae_std"]),
                "failure_auc_mean": as_float(row["failure_auc_mean"]),
                "failure_auc_std": as_float(row["failure_auc_std"]),
                "conformal_coverage_mean": as_float(row["conformal_coverage_mean"]),
                "conformal_coverage_std": as_float(row["conformal_coverage_std"]),
            }
        )
    return rows


def load_observed_fraction_rows() -> list[dict[str, float]]:
    rows = []
    for row in read_csv_rows(ABLATION_TABLE):
        if row["ablation"] != "observed_fraction":
            continue
        rows.append(
            {
                "fraction": as_float(row["setting"]),
                "completion_mae": as_float(row["completion_mae"]),
                "failure_auc": as_float(row["failure_auc"]),
                "conformal_coverage": as_float(row["conformal_coverage"]),
            }
        )
    return sorted(rows, key=lambda entry: entry["fraction"])


def load_alpha_rows() -> list[dict[str, float]]:
    if not ALPHA_SUMMARY.exists():
        raise SystemExit(f"Missing {ALPHA_SUMMARY}. Run `python scripts/run_alpha_coverage_validation.py` first.")

    rows = []
    for row in read_csv_rows(ALPHA_SUMMARY):
        rows.append(
            {
                "alpha": as_float(row["alpha"]),
                "target_coverage": as_float(row["target_coverage"]),
                "empirical_coverage_mean": as_float(row["empirical_coverage_mean"]),
                "empirical_coverage_std": as_float(row["empirical_coverage_std"]),
            }
        )
    return sorted(rows, key=lambda entry: entry["alpha"])


def load_family_summary() -> dict[str, float]:
    return {row["metric"]: as_float(row["mean"]) for row in read_csv_rows(FAMILY_SUMMARY)}


def load_family_counts() -> list[tuple[str, int]]:
    rows = [(row["family"], int(as_float(row["num_models"]))) for row in read_csv_rows(FAMILY_COUNTS)]
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def load_cost_rows() -> list[dict[str, float]]:
    rows = []
    for row in read_csv_rows(COST_SUMMARY):
        rows.append(
            {
                "risk_threshold": as_float(row["risk_threshold"]),
                "episodes_accepted_mean": as_float(row["episodes_accepted_mean"]),
                "benchmarks_avoided_fraction_mean": as_float(row["benchmarks_avoided_fraction_mean"]),
                "accepted_mae_mean": as_float(row["accepted_mae_mean"]),
                "accepted_failure_rate_mean": as_float(row["accepted_failure_rate_mean"]),
            }
        )
    return sorted(rows, key=lambda entry: entry["risk_threshold"])


def load_demo_report() -> dict[str, object]:
    return json.loads(DEMO_REPORT.read_text(encoding="utf-8"))


def load_benchmark_coverage() -> list[tuple[str, int]]:
    score_matrix = load_score_csv(SCORE_CSV)
    rows = []
    for idx, name in enumerate(score_matrix.benchmark_names):
        rows.append((name, int(np.sum(~np.isnan(score_matrix.values[:, idx])))))
    return rows


def random_split_indices(n_models: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_models)
    n_test = max(1, int(round(test_fraction * n_models)))
    n_train = n_models - n_test
    return np.sort(order[:n_train]), np.sort(order[n_train:])


def build_risk_stratification_artifacts(data_dir: Path) -> list[dict[str, float]]:
    score_matrix = load_score_csv(SCORE_CSV)
    matrix = score_matrix.values
    episode_rows: list[dict[str, float]] = []

    for seed in RISK_SEEDS:
        train_idx, test_idx = random_split_indices(matrix.shape[0], test_fraction=0.25, seed=seed)
        train_matrix = matrix[train_idx]
        test_matrix = matrix[test_idx]
        ensemble = train_ensemble(matrix=train_matrix, ensemble_size=5, rank=3, seed=seed)
        episodes = simulate_new_model_episodes(
            matrix=test_matrix,
            episodes_per_model=3,
            observed_fraction=0.5,
            seed=seed + 1,
        )
        results = [_evaluate_episode(ep, ensemble, ridge=1e-2) for ep in episodes]
        hidden_maes = np.asarray([result.hidden_mae for result in results], dtype=float)
        groups = np.asarray([episode.model_index for episode in episodes], dtype=int)
        features = np.vstack([result.features for result in results])
        failure_threshold = float(_quantile_higher(hidden_maes, 0.8))
        labels = (hidden_maes >= failure_threshold).astype(int)
        probabilities = _grouped_failure_probabilities(features, labels, groups, seed=seed)

        for episode_index, (probability, hidden_mae, label) in enumerate(zip(probabilities, hidden_maes, labels)):
            if not np.isfinite(probability):
                continue
            episode_rows.append(
                {
                    "seed": float(seed),
                    "episode_index": float(episode_index),
                    "risk_probability": float(probability),
                    "hidden_mae": float(hidden_mae),
                    "failure_label": float(label),
                }
            )

    by_episode_path = data_dir / "risk_stratification_by_episode.csv"
    with by_episode_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "episode_index", "risk_probability", "hidden_mae", "failure_label"])
        for row in episode_rows:
            writer.writerow(
                [
                    int(row["seed"]),
                    int(row["episode_index"]),
                    f"{row['risk_probability']:.6f}",
                    f"{row['hidden_mae']:.6f}",
                    int(row["failure_label"]),
                ]
            )

    bins = summarize_risk_bins(episode_rows)
    bins_path = data_dir / "risk_stratification_bins.csv"
    with bins_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_left", "bin_right", "mean_probability", "mean_hidden_mae", "failure_rate", "count"])
        for row in bins:
            writer.writerow(
                [
                    f"{row['bin_left']:.2f}",
                    f"{row['bin_right']:.2f}",
                    f"{row['mean_probability']:.6f}",
                    f"{row['mean_hidden_mae']:.6f}",
                    f"{row['failure_rate']:.6f}",
                    int(row["count"]),
                ]
            )
    return bins


def summarize_risk_bins(episode_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    bin_edges = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.000001], dtype=float)
    bins: list[dict[str, float]] = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        members = [row for row in episode_rows if left <= row["risk_probability"] < right]
        if not members:
            continue
        probabilities = np.asarray([row["risk_probability"] for row in members], dtype=float)
        maes = np.asarray([row["hidden_mae"] for row in members], dtype=float)
        failures = np.asarray([row["failure_label"] for row in members], dtype=float)
        bins.append(
            {
                "bin_left": left,
                "bin_right": min(right, 1.0),
                "mean_probability": float(np.mean(probabilities)),
                "mean_hidden_mae": float(np.mean(maes)),
                "failure_rate": float(np.mean(failures)),
                "count": float(len(members)),
            }
        )
    return bins


def generate_overview_figure(
    path: Path,
    dpi: int,
    benchmark_metrics: dict[str, float],
    alpha_rows: list[dict[str, float]],
    benchmark_coverage: list[tuple[str, int]],
    family_counts: list[tuple[str, int]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    metric_keys = ["completion_mae", "failure_auc", "conformal_coverage", "conformal_quantile"]
    metric_labels = ["Completion MAE", "Failure AUC", "Coverage", "Quantile"]
    metric_colors = [COLORS["mae"], COLORS["auc"], COLORS["coverage"], COLORS["quantile"]]
    values = [benchmark_metrics[key] for key in metric_keys]
    ax = axes[0, 0]
    ax.bar(metric_labels, values, color=metric_colors)
    style_axis(ax, "Primary benchmark summary", ylabel="Value")
    for x, value in enumerate(values):
        ax.text(x, value + 0.015, f"{value:.3f}", ha="center", fontsize=9.5)

    ax = axes[0, 1]
    names = [name for name, _ in benchmark_coverage]
    counts = [count for _, count in benchmark_coverage]
    ax.bar(names, counts, color=COLORS["navy"])
    style_axis(ax, "Observed rows per benchmark", ylabel="Models with scores")
    ax.tick_params(axis="x", rotation=25)
    for x, count in enumerate(counts):
        ax.text(x, count + 0.6, str(count), ha="center", fontsize=9)

    ax = axes[1, 0]
    targets = np.asarray([row["target_coverage"] for row in alpha_rows], dtype=float)
    empirical = np.asarray([row["empirical_coverage_mean"] for row in alpha_rows], dtype=float)
    stds = np.asarray([row["empirical_coverage_std"] for row in alpha_rows], dtype=float)
    alphas = np.asarray([row["alpha"] for row in alpha_rows], dtype=float)
    ax.errorbar(targets, empirical, yerr=stds, fmt="o-", color=COLORS["coverage"], ecolor=COLORS["ink"], capsize=4)
    ax.plot([0.65, 1.0], [0.65, 1.0], linestyle="--", linewidth=1.2, color=COLORS["gray"])
    style_axis(ax, "Conformal coverage vs target", xlabel="Target coverage (1 - alpha)", ylabel="Empirical coverage")
    ax.set_xlim(0.65, 1.0)
    ax.set_ylim(0.65, 1.0)
    for x, y, alpha in zip(targets, empirical, alphas):
        ax.text(x + 0.005, y + 0.004, f"a={alpha:.2f}", fontsize=9)

    ax = axes[1, 1]
    top_families = family_counts[:8][::-1]
    family_names = [name for name, _ in top_families]
    family_sizes = [count for _, count in top_families]
    ax.barh(family_names, family_sizes, color=COLORS["gold"])
    style_axis(ax, "Largest model families", xlabel="Number of models")
    for y, count in enumerate(family_sizes):
        ax.text(count + 0.15, y, str(count), va="center", fontsize=9)

    fig.suptitle("Conference Figure 01: benchmark and dataset overview", fontsize=14, fontweight="bold")
    save_figure(fig, path, dpi)


def generate_baseline_comparison(path: Path, dpi: int, baseline_rows: list[dict[str, float | str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), constrained_layout=True)
    names = [str(row["method"]) for row in baseline_rows]
    y = np.arange(len(names))
    highlight = "ManifoldGuard (full)"

    metric_specs = [
        ("completion_mae_mean", "completion_mae_std", "Completion MAE", COLORS["mae"], True),
        ("failure_auc_mean", "failure_auc_std", "Failure AUC", COLORS["auc"], False),
        ("conformal_coverage_mean", "conformal_coverage_std", "Coverage", COLORS["coverage"], False),
    ]

    for ax, (metric_key, std_key, title, color, invert) in zip(axes, metric_specs):
        values = np.asarray([float(row[metric_key]) for row in baseline_rows], dtype=float)
        stds = np.asarray([float(row[std_key]) for row in baseline_rows], dtype=float)
        bar_colors = [color if name == highlight else COLORS["gray"] for name in names]
        finite = np.isfinite(values)
        ax.barh(y[finite], values[finite], color=np.asarray(bar_colors, dtype=object)[finite])
        if np.any(finite & np.isfinite(stds)):
            ax.errorbar(values[finite], y[finite], xerr=stds[finite], fmt="none", ecolor=COLORS["ink"], capsize=3)
        style_axis(ax, title)
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax is axes[0] else [])
        if invert:
            ax.invert_xaxis()
        for yy, value in zip(y[finite], values[finite]):
            ax.text(value, yy, f" {value:.3f}" if not invert else f"{value:.3f} ", va="center", ha="left" if not invert else "right", fontsize=9)
        ax.set_ylim(-0.6, len(names) - 0.4)

    fig.suptitle("Conference Figure 02: baseline comparison", fontsize=14, fontweight="bold")
    save_figure(fig, path, dpi)


def generate_observed_fraction_curve(path: Path, dpi: int, observed_rows: list[dict[str, float]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10), constrained_layout=True, sharex=True)
    fractions = np.asarray([row["fraction"] for row in observed_rows], dtype=float)
    series = [
        ("completion_mae", "Completion MAE", COLORS["mae"]),
        ("failure_auc", "Failure AUC", COLORS["auc"]),
        ("conformal_coverage", "Coverage", COLORS["coverage"]),
    ]
    for ax, (key, title, color) in zip(axes, series):
        values = np.asarray([row[key] for row in observed_rows], dtype=float)
        ax.plot(fractions, values, color=color, marker="o", linewidth=2.0)
        ax.fill_between(fractions, values, color=color, alpha=0.12)
        style_axis(ax, title, ylabel="Value")
        for x, y in zip(fractions, values):
            ax.text(x, y + 0.01, f"{y:.3f}", ha="center", fontsize=9)
        if key == "conformal_coverage":
            ax.axhline(0.90, linestyle="--", linewidth=1.2, color=COLORS["gray"])
    axes[-1].set_xlabel("Observed benchmark fraction")
    axes[-1].set_xticks(fractions)
    axes[-1].set_xticklabels([f"{int(value * 100)}%" for value in fractions])
    fig.suptitle("Conference Figure 03: performance vs observed fraction", fontsize=14, fontweight="bold")
    save_figure(fig, path, dpi)


def generate_family_split_figure(
    path: Path,
    dpi: int,
    benchmark_metrics: dict[str, float],
    family_metrics: dict[str, float],
    family_counts: list[tuple[str, int]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)

    ax = axes[0]
    metric_keys = ["completion_mae", "failure_auc", "conformal_coverage", "conformal_quantile"]
    labels = ["MAE", "AUC", "Coverage", "Quantile"]
    random_values = np.asarray([benchmark_metrics[key] for key in metric_keys], dtype=float)
    family_values = np.asarray([family_metrics[key] for key in metric_keys], dtype=float)
    x = np.arange(len(metric_keys))
    width = 0.36
    ax.bar(x - width / 2, random_values, width, color=COLORS["auc"], label="Random split")
    ax.bar(x + width / 2, family_values, width, color=COLORS["mae"], label="Family holdout")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axis(ax, "Random split vs family holdout", ylabel="Metric value")
    ax.legend(frameon=False)
    for xpos, random_value, family_value in zip(x, random_values, family_values):
        ax.text(xpos - width / 2, random_value + 0.01, f"{random_value:.3f}", ha="center", fontsize=9)
        ax.text(xpos + width / 2, family_value + 0.01, f"{family_value:.3f}", ha="center", fontsize=9)

    ax = axes[1]
    top_families = family_counts[:10][::-1]
    names = [name for name, _ in top_families]
    counts = [count for _, count in top_families]
    ax.barh(names, counts, color=COLORS["navy"])
    style_axis(ax, "Family sizes in the curated dataset", xlabel="Models")
    for y, count in enumerate(counts):
        ax.text(count + 0.15, y, str(count), va="center", fontsize=9)

    fig.suptitle("Conference Figure 04: tougher family-holdout split", fontsize=14, fontweight="bold")
    save_figure(fig, path, dpi)


def generate_cost_frontier(path: Path, dpi: int, cost_rows: list[dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    thresholds = np.asarray([row["risk_threshold"] for row in cost_rows], dtype=float)
    avoided = np.asarray([row["benchmarks_avoided_fraction_mean"] for row in cost_rows], dtype=float)
    accepted_mae = np.asarray([row["accepted_mae_mean"] for row in cost_rows], dtype=float)
    failure_rate = np.asarray([row["accepted_failure_rate_mean"] for row in cost_rows], dtype=float)
    accepted_share = np.asarray([row["episodes_accepted_mean"] for row in cost_rows], dtype=float)

    ax = axes[0]
    sizes = 900 * accepted_share + 40
    scatter = ax.scatter(avoided, accepted_mae, s=sizes, c=thresholds, cmap="viridis", edgecolors="white", linewidths=1.0)
    ax.plot(avoided, accepted_mae, color=COLORS["gray"], linewidth=1.2)
    style_axis(ax, "Benchmarks avoided vs accepted MAE", xlabel="Benchmarks avoided fraction", ylabel="Accepted MAE")
    for x, y, threshold in zip(avoided, accepted_mae, thresholds):
        ax.text(x, y + 0.0015, f"{threshold:.2f}", ha="center", fontsize=9)
    fig.colorbar(scatter, ax=ax, label="Risk threshold")

    ax = axes[1]
    ax.plot(thresholds, failure_rate, color=COLORS["red"], marker="o", linewidth=2.0, label="Accepted failure rate")
    ax.plot(thresholds, avoided, color=COLORS["auc"], marker="s", linewidth=2.0, label="Benchmarks avoided")
    style_axis(ax, "Threshold trade-off curve", xlabel="Risk threshold", ylabel="Value")
    ax.set_xticks(thresholds)
    ax.legend(frameon=False)

    fig.suptitle("Conference Figure 05: cost-savings frontier", fontsize=14, fontweight="bold")
    save_figure(fig, path, dpi)


def generate_risk_stratification(path: Path, dpi: int, bins: list[dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    labels = [f"{row['bin_left']:.1f}-{row['bin_right']:.1f}" for row in bins]
    failure_rates = np.asarray([row["failure_rate"] for row in bins], dtype=float)
    mean_maes = np.asarray([row["mean_hidden_mae"] for row in bins], dtype=float)
    counts = np.asarray([row["count"] for row in bins], dtype=float)
    x = np.arange(len(labels))

    bars = ax.bar(x, failure_rates, color=COLORS["gold"])
    style_axis(ax, "Failure rate by predicted-risk bin", xlabel="Predicted failure probability bin", ylabel="Empirical failure rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0.20, linestyle="--", linewidth=1.2, color=COLORS["gray"])

    ax2 = ax.twinx()
    ax2.plot(x, mean_maes, color=COLORS["auc"], marker="o", linewidth=2.0)
    ax2.set_ylabel("Mean hidden MAE", color=COLORS["auc"])
    ax2.tick_params(axis="y", colors=COLORS["auc"], length=0)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"n={int(count)}", ha="center", fontsize=9)

    fig.suptitle("Conference Figure 06: risk stratification", fontsize=14, fontweight="bold")
    save_figure(fig, path, dpi)


def generate_demo_case_study(path: Path, dpi: int, demo_report: dict[str, object]) -> None:
    observed_items = list(demo_report["observed_scores"].items())
    hidden_rows = list(demo_report["hidden_predictions"])
    names = [str(row["benchmark"]) for row in hidden_rows]
    preds = np.asarray([float(row["predicted_score"]) for row in hidden_rows], dtype=float)
    lowers = np.asarray([float(row["interval_lower"]) for row in hidden_rows], dtype=float)
    uppers = np.asarray([float(row["interval_upper"]) for row in hidden_rows], dtype=float)
    truths = np.asarray([float(row["true_score"]) for row in hidden_rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True, gridspec_kw={"width_ratios": [0.9, 2.1]})

    ax = axes[0]
    obs_names = [name for name, _ in observed_items]
    obs_values = [float(value) for _, value in observed_items]
    x = np.arange(len(obs_names))
    ax.bar(x, obs_values, color=COLORS["navy"])
    style_axis(ax, "Observed benchmark anchors", ylabel="Score")
    ax.set_xticks(x)
    ax.set_xticklabels(obs_names, rotation=20)
    for xpos, value in zip(x, obs_values):
        ax.text(xpos, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)

    ax = axes[1]
    x = np.arange(len(names))
    yerr = np.vstack([preds - lowers, uppers - preds])
    ax.errorbar(x, preds, yerr=yerr, fmt="o", color=COLORS["mae"], ecolor=COLORS["gray"], elinewidth=3, capsize=4, label="Prediction interval")
    ax.scatter(x, truths, color=COLORS["navy"], marker="D", s=42, label="Ground truth", zorder=5)
    style_axis(ax, "Hidden benchmark predictions", ylabel="Score")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend(frameon=False)
    for xpos, pred, truth in zip(x, preds, truths):
        ax.text(xpos, max(pred, truth) + 0.04, f"{abs(pred - truth):.3f}", ha="center", fontsize=8.5)

    fig.suptitle("Conference Figure 07: qualitative demo case", fontsize=14, fontweight="bold")
    save_figure(fig, path, dpi)


def generate_alpha_validation(path: Path, dpi: int, alpha_rows: list[dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
    targets = np.asarray([row["target_coverage"] for row in alpha_rows], dtype=float)
    empirical = np.asarray([row["empirical_coverage_mean"] for row in alpha_rows], dtype=float)
    stds = np.asarray([row["empirical_coverage_std"] for row in alpha_rows], dtype=float)
    alphas = np.asarray([row["alpha"] for row in alpha_rows], dtype=float)

    ax.errorbar(targets, empirical, yerr=stds, fmt="o-", color=COLORS["coverage"], ecolor=COLORS["ink"], capsize=4)
    ax.plot([0.65, 1.0], [0.65, 1.0], linestyle="--", linewidth=1.2, color=COLORS["gray"])
    style_axis(ax, "Empirical coverage versus target", xlabel="Target coverage (1 - alpha)", ylabel="Empirical coverage")
    ax.set_xlim(0.65, 1.0)
    ax.set_ylim(0.65, 1.0)
    for x, y, alpha in zip(targets, empirical, alphas):
        ax.text(x + 0.005, y + 0.004, f"a={alpha:.2f}", fontsize=9)

    fig.suptitle("Conference Figure 08: alpha validation", fontsize=14, fontweight="bold")
    save_figure(fig, path, dpi)


def main() -> None:
    args = parse_args()
    configure_style()
    out_dir, data_dir = ensure_dirs(args.output_dir)

    benchmark_metrics = load_benchmark_metrics()
    baseline_rows = load_baseline_rows()
    observed_rows = load_observed_fraction_rows()
    alpha_rows = load_alpha_rows()
    family_metrics = load_family_summary()
    family_counts = load_family_counts()
    benchmark_coverage = load_benchmark_coverage()
    cost_rows = load_cost_rows()
    demo_report = load_demo_report()
    risk_bins = build_risk_stratification_artifacts(data_dir)

    figures = [
        ("01_method_overview.png", lambda p: generate_overview_figure(p, args.dpi, benchmark_metrics, alpha_rows, benchmark_coverage, family_counts)),
        ("02_baseline_comparison.png", lambda p: generate_baseline_comparison(p, args.dpi, baseline_rows)),
        ("03_observed_fraction_curve.png", lambda p: generate_observed_fraction_curve(p, args.dpi, observed_rows)),
        ("04_family_holdout_vs_random.png", lambda p: generate_family_split_figure(p, args.dpi, benchmark_metrics, family_metrics, family_counts)),
        ("05_cost_savings_frontier.png", lambda p: generate_cost_frontier(p, args.dpi, cost_rows)),
        ("06_risk_stratification.png", lambda p: generate_risk_stratification(p, args.dpi, risk_bins)),
        ("07_demo_case_study.png", lambda p: generate_demo_case_study(p, args.dpi, demo_report)),
        ("08_alpha_validation.png", lambda p: generate_alpha_validation(p, args.dpi, alpha_rows)),
    ]

    print(f"Writing chart-driven conference figures to {out_dir}")
    for filename, builder in figures:
        target = out_dir / filename
        builder(target)
        print(f"  wrote {target.name}")


if __name__ == "__main__":
    main()
