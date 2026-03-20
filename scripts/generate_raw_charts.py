"""Generate polished charts-first PNG figures from the real-data artifacts.

Usage:
    python scripts/generate_raw_charts.py
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
    from matplotlib.colors import LinearSegmentedColormap
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for chart generation. Install with `pip install .[viz]`.") from exc

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
FAMILY_SUMMARY = REPO_ROOT / "results" / "family_split" / "summary_table.csv"
FAMILY_COUNTS = REPO_ROOT / "results" / "family_split" / "family_counts.csv"
COST_SUMMARY = REPO_ROOT / "results" / "cost_savings" / "summary_table.csv"
DEMO_REPORT = REPO_ROOT / "results" / "demo" / "demo_report.json"
SCORE_CSV = REPO_ROOT / "datasets" / "lm_eval_real" / "scores.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "raw_charts"
RISK_SEEDS = [0, 1, 2, 3, 4]

COLORS = {
    "bg": "#fcfbf7",
    "panel": "#ffffff",
    "ink": "#12253d",
    "muted": "#5c6f82",
    "grid": "#d9e2ec",
    "line": "#c7d1db",
    "accent": "#f26b38",
    "accent_dark": "#bf4e21",
    "teal": "#16817a",
    "teal_soft": "#8ed6cf",
    "gold": "#f0b44d",
    "navy": "#274690",
    "purple": "#7a5af8",
    "rose": "#d9485f",
    "slate": "#8a9aa9",
    "fill": "#eef3f7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate polished raw chart PNGs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for the output PNGs.")
    parser.add_argument("--dpi", type=int, default=320, help="PNG DPI.")
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["panel"],
            "savefig.facecolor": COLORS["bg"],
            "font.family": "DejaVu Sans",
            "axes.edgecolor": COLORS["line"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.75,
            "grid.linewidth": 0.8,
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


def polish_axes(ax: mpl.axes.Axes, title: str | None = None, subtitle: str | None = None) -> None:
    ax.spines["left"].set_color(COLORS["line"])
    ax.spines["bottom"].set_color(COLORS["line"])
    ax.tick_params(length=0)
    if title:
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10.5, color=COLORS["muted"], va="bottom")


def add_figure_header(fig: mpl.figure.Figure, label: str, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.965, label.upper(), fontsize=11, fontweight="bold", color=COLORS["accent_dark"])
    fig.text(0.055, 0.928, title, fontsize=23, fontweight="bold", color=COLORS["ink"])
    fig.text(0.055, 0.892, subtitle, fontsize=11.5, color=COLORS["muted"])


def save(fig: mpl.figure.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


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


def load_ablation_rows() -> list[dict[str, str]]:
    return read_csv_rows(ABLATION_TABLE)


def load_family_metrics() -> dict[str, float]:
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
    rows.sort(key=lambda item: item["risk_threshold"])
    return rows


def load_demo_report() -> dict[str, object]:
    return json.loads(DEMO_REPORT.read_text(encoding="utf-8"))


def random_split_indices(n_models: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_models)
    n_test = max(1, int(round(test_fraction * n_models)))
    n_train = n_models - n_test
    return np.sort(order[:n_train]), np.sort(order[n_train:])


def build_risk_stratification(data_dir: Path) -> list[dict[str, float]]:
    score_matrix = load_score_csv(SCORE_CSV)
    matrix = score_matrix.values
    episode_rows: list[dict[str, float]] = []

    for seed in RISK_SEEDS:
        train_idx, test_idx = random_split_indices(matrix.shape[0], test_fraction=0.25, seed=seed)
        ensemble = train_ensemble(matrix=matrix[train_idx], ensemble_size=5, rank=3, seed=seed)
        episodes = simulate_new_model_episodes(matrix=matrix[test_idx], episodes_per_model=3, observed_fraction=0.5, seed=seed + 1)
        results = [_evaluate_episode(ep, ensemble, ridge=1e-2) for ep in episodes]
        hidden_maes = np.asarray([result.hidden_mae for result in results], dtype=float)
        groups = np.asarray([episode.model_index for episode in episodes], dtype=int)
        features = np.vstack([result.features for result in results])
        labels = (hidden_maes >= float(_quantile_higher(hidden_maes, 0.8))).astype(int)
        probabilities = _grouped_failure_probabilities(features, labels, groups, seed=seed)

        for probability, hidden_mae, label in zip(probabilities, hidden_maes, labels):
            if np.isfinite(probability):
                episode_rows.append(
                    {
                        "risk_probability": float(probability),
                        "hidden_mae": float(hidden_mae),
                        "failure_label": float(label),
                    }
                )

    output = data_dir / "risk_by_episode.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["risk_probability", "hidden_mae", "failure_label"])
        for row in episode_rows:
            writer.writerow([f"{row['risk_probability']:.6f}", f"{row['hidden_mae']:.6f}", int(row["failure_label"])])
    return episode_rows


def chart_baselines(path: Path, dpi: int, baseline_rows: list[dict[str, float | str]]) -> None:
    fig = plt.figure(figsize=(15.2, 9.2))
    add_figure_header(
        fig,
        "Chart 01",
        "Baseline suite comparison",
        "Point completion stays tied with MF, while the full method adds failure ranking and calibrated intervals.",
    )
    gs = fig.add_gridspec(1, 3, left=0.08, right=0.97, bottom=0.12, top=0.82, wspace=0.16)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    methods = [str(row["method"]) for row in baseline_rows]
    y = np.arange(len(methods))[::-1]
    base_color = COLORS["slate"]
    highlight = COLORS["accent"]
    secondary = COLORS["teal"]
    colors = [base_color, base_color, base_color, secondary, highlight]

    specs = [
        ("completion_mae_mean", "completion_mae_std", "Completion MAE", "Lower is better", True, None),
        ("failure_auc_mean", "failure_auc_std", "Failure AUC", "Higher is better", False, None),
        ("conformal_coverage_mean", "conformal_coverage_std", "Conformal coverage", "Higher is better", False, 0.90),
    ]

    for ax, (metric_key, std_key, title, xlabel, invert, ref_line) in zip(axes, specs):
        polish_axes(ax, title)
        values = np.asarray([float(row[metric_key]) for row in baseline_rows], dtype=float)
        stds = np.asarray([float(row[std_key]) for row in baseline_rows], dtype=float)
        finite = values[np.isfinite(values)]
        max_val = float(np.max(finite)) if finite.size else 1.0
        min_val = float(np.min(finite)) if finite.size else 0.0
        pad = 0.15 * (max_val - min_val if max_val > min_val else max_val)
        left, right = (max_val + pad, max(0.0, min_val - pad * 0.5)) if invert else (max(0.0, min_val - pad * 0.25), max_val + pad)
        ax.set_xlim(left, right)
        ax.set_yticks(y)
        ax.set_yticklabels(methods if ax is axes[0] else [])
        ax.set_xlabel(xlabel, fontsize=10.5)

        for idx, (yy, val, std, color) in enumerate(zip(y, values, stds, colors)):
            if not np.isfinite(val):
                ax.hlines(yy, min_val, max_val, color=COLORS["grid"], linewidth=2.0, linestyles="dashed")
                ax.text(max_val, yy, "N/A", va="center", ha="left", color=COLORS["muted"], fontsize=10)
                continue
            ax.hlines(yy, min(left, right), val, color=COLORS["fill"], linewidth=6, zorder=1)
            ax.scatter([val], [yy], s=110 if idx == len(y) - 1 else 82, color=color, edgecolors="white", linewidths=1.5, zorder=3)
            if np.isfinite(std) and std > 0:
                ax.errorbar(val, yy, xerr=std, fmt="none", ecolor=COLORS["ink"], elinewidth=1.2, capsize=3, alpha=0.75, zorder=2)
            text_dx = -0.01 if invert else 0.01
            ax.text(val + text_dx * (left - right), yy, f"{val:.3f}", va="center", ha="right" if invert else "left", fontsize=10.5, fontweight="bold")

        if ref_line is not None:
            ax.axvline(ref_line, color=COLORS["navy"], linestyle="--", linewidth=1.5, alpha=0.9)
            ax.text(ref_line, y[0] + 0.85, "90% target", color=COLORS["navy"], fontsize=10.5, fontweight="bold", ha="left")

    fig.text(0.08, 0.085, "ManifoldGuard is highlighted in orange. MF + failure detection is shown in teal.", fontsize=10.5, color=COLORS["muted"])
    save(fig, path, dpi)


def chart_observed_fraction(path: Path, dpi: int, ablation_rows: list[dict[str, str]]) -> None:
    frac_rows = [
        {
            "fraction": as_float(row["setting"]),
            "completion_mae": as_float(row["completion_mae"]),
            "failure_auc": as_float(row["failure_auc"]),
            "conformal_coverage": as_float(row["conformal_coverage"]),
        }
        for row in ablation_rows
        if row["ablation"] == "observed_fraction"
    ]
    frac_rows.sort(key=lambda row: row["fraction"])
    x = np.asarray([row["fraction"] for row in frac_rows], dtype=float)
    mae = np.asarray([row["completion_mae"] for row in frac_rows], dtype=float)
    auc = np.asarray([row["failure_auc"] for row in frac_rows], dtype=float)
    cov = np.asarray([row["conformal_coverage"] for row in frac_rows], dtype=float)

    fig = plt.figure(figsize=(15.2, 9.4))
    add_figure_header(
        fig,
        "Chart 02",
        "Observed fraction curve",
        "As more benchmarks are revealed, completion error drops sharply and the failure detector gets stronger. Coverage stays near the nominal target.",
    )
    gs = fig.add_gridspec(3, 1, left=0.09, right=0.96, bottom=0.11, top=0.83, hspace=0.16)
    panels = [
        ("Completion MAE", mae, COLORS["accent"]),
        ("Failure AUC", auc, COLORS["teal"]),
        ("Conformal coverage", cov, COLORS["gold"]),
    ]
    for idx, (title, values, color) in enumerate(panels):
        ax = fig.add_subplot(gs[idx, 0])
        polish_axes(ax, title)
        ax.plot(x, values, color=color, linewidth=3.0, marker="o", markersize=7)
        ax.fill_between(x, values, color=color, alpha=0.10)
        ax.axvline(0.5, color=COLORS["line"], linestyle="--", linewidth=1.3)
        ax.text(0.505, 0.92, "default", transform=ax.get_xaxis_transform(), fontsize=9.5, color=COLORS["muted"], rotation=90, va="top")
        for xx, yy in zip(x, values):
            ax.scatter([xx], [yy], s=84, color=color, edgecolors="white", linewidths=1.4, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(xx * 100)}%" for xx in x])
        if title == "Conformal coverage":
            ax.axhline(0.90, color=COLORS["navy"], linestyle="--", linewidth=1.5)
            ax.text(x[0], 0.903, "90% target", fontsize=10, color=COLORS["navy"], fontweight="bold")
        if idx < 2:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Observed benchmark fraction", fontsize=11)
    save(fig, path, dpi)


def chart_component_ablations(path: Path, dpi: int, ablation_rows: list[dict[str, str]]) -> None:
    rank_rows = [
        {"rank": int(as_float(row["setting"])), "completion_mae": as_float(row["completion_mae"]), "failure_auc": as_float(row["failure_auc"])}
        for row in ablation_rows
        if row["ablation"] == "rank"
    ]
    feature_rows = [
        {"subset": row["setting"], "failure_auc": as_float(row["failure_auc"])}
        for row in ablation_rows
        if row["ablation"] == "feature_subset"
    ]
    rank_rows.sort(key=lambda row: row["rank"])
    feature_rows.sort(key=lambda row: row["failure_auc"])

    fig = plt.figure(figsize=(15.0, 8.8))
    add_figure_header(
        fig,
        "Chart 03",
        "Component ablations",
        "The geometry signals help the detector, and the low-rank model is strongest around rank 3 to 4 on the fixed real benchmark matrix.",
    )
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.97, bottom=0.12, top=0.82, wspace=0.18)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    polish_axes(ax_left, "Rank sweep")
    polish_axes(ax_right, "OOD feature subsets")

    ranks = np.asarray([row["rank"] for row in rank_rows], dtype=int)
    maes = np.asarray([row["completion_mae"] for row in rank_rows], dtype=float)
    aucs = np.asarray([row["failure_auc"] for row in rank_rows], dtype=float)
    ax_left.plot(ranks, maes, color=COLORS["accent"], linewidth=2.8, marker="o", markersize=7)
    ax_left.set_xlabel("Latent rank")
    ax_left.set_ylabel("Completion MAE", color=COLORS["accent_dark"])
    ax_left.tick_params(axis="y", colors=COLORS["accent_dark"])
    ax_left.set_xticks(ranks)

    ax_left_b = ax_left.twinx()
    ax_left_b.plot(ranks, aucs, color=COLORS["teal"], linewidth=2.8, marker="s", markersize=6)
    ax_left_b.set_ylabel("Failure AUC", color=COLORS["teal"])
    ax_left_b.tick_params(axis="y", colors=COLORS["teal"])
    ax_left_b.spines["right"].set_visible(False)
    ax_left_b.grid(False)

    names = [row["subset"].replace("_", " ") for row in feature_rows]
    vals = np.asarray([row["failure_auc"] for row in feature_rows], dtype=float)
    y = np.arange(len(names))
    ax_right.barh(y, vals, color=[COLORS["slate"], COLORS["accent"], COLORS["teal"]], alpha=0.9, height=0.58)
    ax_right.set_yticks(y)
    ax_right.set_yticklabels(names)
    ax_right.set_xlabel("Failure AUC")
    for yy, val in zip(y, vals):
        ax_right.text(val + 0.01, yy, f"{val:.3f}", va="center", fontsize=10.5, fontweight="bold")

    save(fig, path, dpi)


def chart_family_holdout(path: Path, dpi: int, benchmark_metrics: dict[str, float], family_metrics: dict[str, float], family_counts: list[tuple[str, int]]) -> None:
    fig = plt.figure(figsize=(15.4, 9.0))
    add_figure_header(
        fig,
        "Chart 04",
        "Random split versus family holdout",
        "Holding out entire model families is materially harder than a random row split, but the uncertainty layer still stays close to the intended operating regime.",
    )
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.97, bottom=0.12, top=0.82, width_ratios=[1.1, 0.9], wspace=0.18)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    polish_axes(ax_left, "Metric comparison")
    polish_axes(ax_right, "Largest model families")

    metric_keys = ["completion_mae", "failure_auc", "conformal_coverage", "conformal_quantile"]
    metric_labels = ["Completion MAE", "Failure AUC", "Coverage", "Quantile"]
    x = np.arange(len(metric_keys))
    random_values = np.asarray([benchmark_metrics[key] for key in metric_keys], dtype=float)
    family_values = np.asarray([family_metrics[key] for key in metric_keys], dtype=float)
    for idx, (xx, rnd, fam) in enumerate(zip(x, random_values, family_values)):
        ax_left.plot([xx, xx], [rnd, fam], color=COLORS["grid"], linewidth=3.0, zorder=1)
        ax_left.scatter([xx], [rnd], s=95, color=COLORS["teal"], edgecolors="white", linewidths=1.5, zorder=3)
        ax_left.scatter([xx], [fam], s=95, color=COLORS["accent"], edgecolors="white", linewidths=1.5, zorder=3)
        ax_left.text(xx - 0.07, rnd, f"{rnd:.3f}", ha="right", va="center", fontsize=10.5)
        ax_left.text(xx + 0.07, fam, f"{fam:.3f}", ha="left", va="center", fontsize=10.5, fontweight="bold")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(metric_labels)
    ax_left.set_ylabel("Metric value")
    ax_left.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["teal"], markeredgecolor="white", markersize=9, label="Random split"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["accent"], markeredgecolor="white", markersize=9, label="Family holdout"),
        ],
        frameon=False,
        loc="upper left",
    )

    top_families = family_counts[:8][::-1]
    names = [name for name, _ in top_families]
    counts = [count for _, count in top_families]
    ax_right.barh(names, counts, color=COLORS["navy"], alpha=0.9, height=0.58)
    ax_right.set_xlabel("Number of models")
    for yy, val in enumerate(counts):
        ax_right.text(val + 0.2, yy, str(val), va="center", fontsize=10.5)

    save(fig, path, dpi)


def chart_cost_frontier(path: Path, dpi: int, cost_rows: list[dict[str, float]]) -> None:
    thresholds = np.asarray([row["risk_threshold"] for row in cost_rows], dtype=float)
    avoided = np.asarray([row["benchmarks_avoided_fraction_mean"] for row in cost_rows], dtype=float)
    failure_rates = np.asarray([row["accepted_failure_rate_mean"] for row in cost_rows], dtype=float)
    maes = np.asarray([row["accepted_mae_mean"] for row in cost_rows], dtype=float)
    accepted_share = np.asarray([row["episodes_accepted_mean"] for row in cost_rows], dtype=float)

    fig = plt.figure(figsize=(15.0, 8.8))
    add_figure_header(
        fig,
        "Chart 05",
        "Cost-savings frontier",
        "Each point is a risk threshold. Moving right saves more benchmark executions, but it also admits riskier episodes. Point size shows acceptance rate.",
    )
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.97, bottom=0.12, top=0.82, width_ratios=[1.2, 0.8], wspace=0.16)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    polish_axes(ax_left, "Benchmarks avoided versus accepted failure rate")
    polish_axes(ax_right, "Accepted MAE by threshold")

    cmap = LinearSegmentedColormap.from_list("mg_thresholds", [COLORS["teal"], COLORS["gold"], COLORS["accent"]])
    sizes = 520 * accepted_share + 130
    scatter = ax_left.scatter(avoided, failure_rates, s=sizes, c=thresholds, cmap=cmap, edgecolors="white", linewidths=1.7, zorder=3)
    ax_left.plot(avoided, failure_rates, color=COLORS["line"], linewidth=2.0, zorder=1)
    ax_left.set_xlabel("Benchmarks avoided fraction")
    ax_left.set_ylabel("Accepted failure rate")
    for xx, yy, threshold in zip(avoided, failure_rates, thresholds):
        ax_left.text(xx, yy + 0.013, f"{threshold:.2f}", ha="center", fontsize=10.5, fontweight="bold")
    chosen = int(np.argmin(np.abs(thresholds - 0.35)))
    ax_left.scatter([avoided[chosen]], [failure_rates[chosen]], s=sizes[chosen] * 1.15, facecolors="none", edgecolors=COLORS["ink"], linewidths=1.6, zorder=4)
    ax_left.annotate(
        "Practical knee at 0.35",
        xy=(avoided[chosen], failure_rates[chosen]),
        xytext=(avoided[chosen] - 0.22, failure_rates[chosen] + 0.09),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["ink"], "linewidth": 1.3},
        fontsize=11,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.colorbar(scatter, ax=ax_left, fraction=0.04, pad=0.02, label="Risk threshold")

    ax_right.plot(thresholds, maes, color=COLORS["accent"], linewidth=3.0, marker="o", markersize=7)
    ax_right.fill_between(thresholds, maes, color=COLORS["accent"], alpha=0.12)
    ax_right.set_xlabel("Risk threshold")
    ax_right.set_ylabel("Accepted MAE")
    ax_right.set_xticks(thresholds)
    for xx, yy in zip(thresholds, maes):
        ax_right.text(xx, yy + 0.0015, f"{yy:.3f}", ha="center", fontsize=10)

    save(fig, path, dpi)


def chart_risk_stratification(path: Path, dpi: int, episode_rows: list[dict[str, float]]) -> None:
    bin_edges = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.000001], dtype=float)
    bins = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        members = [row for row in episode_rows if left <= row["risk_probability"] < right]
        if not members:
            continue
        probs = np.asarray([row["risk_probability"] for row in members], dtype=float)
        maes = np.asarray([row["hidden_mae"] for row in members], dtype=float)
        fails = np.asarray([row["failure_label"] for row in members], dtype=float)
        bins.append(
            {
                "label": f"{left:.1f}-{min(right, 1.0):.1f}",
                "mean_probability": float(np.mean(probs)),
                "mean_hidden_mae": float(np.mean(maes)),
                "failure_rate": float(np.mean(fails)),
                "count": len(members),
            }
        )

    fig = plt.figure(figsize=(15.0, 8.8))
    add_figure_header(
        fig,
        "Chart 06",
        "Risk stratification",
        "Grouped out-of-fold risk scores sort the test episodes into bins that line up with both higher failure rate and larger hidden-score error.",
    )
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.68])
    polish_axes(ax, "Empirical failure rate by predicted-risk bin")
    labels = [row["label"] for row in bins]
    failure_rates = np.asarray([row["failure_rate"] for row in bins], dtype=float)
    mean_maes = np.asarray([row["mean_hidden_mae"] for row in bins], dtype=float)
    counts = np.asarray([row["count"] for row in bins], dtype=int)
    x = np.arange(len(labels))
    bar_colors = [mpl.colors.to_rgba(COLORS["accent"], alpha) for alpha in np.linspace(0.4, 0.95, len(labels))]
    bars = ax.bar(x, failure_rates, color=bar_colors, width=0.62, edgecolor="white", linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Predicted failure probability bin")
    ax.set_ylabel("Empirical failure rate")
    ax.axhline(0.20, color=COLORS["navy"], linestyle="--", linewidth=1.5)
    ax.text(len(labels) - 0.25, 0.205, "20% failure prior", ha="right", fontsize=10.5, color=COLORS["navy"], fontweight="bold")

    ax2 = ax.twinx()
    ax2.plot(x, mean_maes, color=COLORS["teal"], linewidth=3.0, marker="o", markersize=7)
    ax2.fill_between(x, mean_maes, color=COLORS["teal"], alpha=0.12)
    ax2.set_ylabel("Mean hidden MAE", color=COLORS["teal"])
    ax2.tick_params(axis="y", colors=COLORS["teal"])
    ax2.spines["right"].set_visible(False)
    ax2.grid(False)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018, f"n={count}", ha="center", fontsize=10, color=COLORS["muted"])

    save(fig, path, dpi)


def chart_demo_case(path: Path, dpi: int, demo_report: dict[str, object]) -> None:
    observed = list(demo_report["observed_scores"].items())
    hidden = list(demo_report["hidden_predictions"])
    names = [str(row["benchmark"]) for row in hidden]
    preds = np.asarray([float(row["predicted_score"]) for row in hidden], dtype=float)
    lowers = np.asarray([float(row["interval_lower"]) for row in hidden], dtype=float)
    uppers = np.asarray([float(row["interval_upper"]) for row in hidden], dtype=float)
    truths = np.asarray([float(row["true_score"]) for row in hidden], dtype=float)

    fig = plt.figure(figsize=(15.5, 9.2))
    add_figure_header(
        fig,
        "Chart 07",
        "Demo episode: prediction intervals on the hidden benchmarks",
        "A single held-out partial evaluation for Meta-Llama-3-70B-Instruct. The left panel shows the revealed anchors, and the right panel shows interval predictions against ground truth.",
    )
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.97, bottom=0.14, top=0.82, width_ratios=[0.72, 1.45], wspace=0.18)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    polish_axes(ax_left, "Observed benchmark anchors")
    polish_axes(ax_right, "Hidden benchmark predictions")

    obs_names = [name for name, _ in observed]
    obs_vals = [float(value) for _, value in observed]
    obs_x = np.arange(len(obs_names))
    ax_left.bar(obs_x, obs_vals, color=[COLORS["navy"], COLORS["purple"], COLORS["gold"]], width=0.58)
    ax_left.set_xticks(obs_x)
    ax_left.set_xticklabels(obs_names, rotation=0)
    ax_left.set_ylabel("Observed score")
    ax_left.set_ylim(0.0, 1.0)
    for xx, yy in zip(obs_x, obs_vals):
        ax_left.text(xx, yy + 0.03, f"{yy:.3f}", ha="center", fontsize=11, fontweight="bold")

    x = np.arange(len(names))
    yerr = np.vstack([preds - lowers, uppers - preds])
    ax_right.errorbar(
        x,
        preds,
        yerr=yerr,
        fmt="o",
        color=COLORS["accent_dark"],
        ecolor=COLORS["gold"],
        elinewidth=4.2,
        capsize=5,
        markersize=7,
        markerfacecolor=COLORS["accent"],
        markeredgecolor="white",
        label="Predicted score + interval",
    )
    ax_right.scatter(x, truths, color=COLORS["navy"], marker="D", s=58, zorder=5, label="Ground truth")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(names, rotation=0)
    ax_right.set_ylabel("Score")
    ax_right.set_ylim(min(lowers.min(), truths.min()) - 0.05, max(uppers.max(), truths.max()) + 0.05)
    ax_right.legend(frameon=False, loc="upper left")
    for xx, pred, truth in zip(x, preds, truths):
        ax_right.text(xx, max(pred, truth) + 0.05, f"abs err {abs(pred - truth):.3f}", ha="center", fontsize=9.5, color=COLORS["muted"])

    risk_prob = float(demo_report["risk_probability"])
    decision = str(demo_report["decision"]).upper()
    next_benchmark = str(demo_report["recommended_next_benchmark"]["benchmark"])
    fig.text(0.08, 0.085, f"Risk {risk_prob:.0%}   |   Decision {decision}   |   Next benchmark {next_benchmark}", fontsize=11.2, color=COLORS["muted"])
    save(fig, path, dpi)


def main() -> None:
    args = parse_args()
    configure_style()
    out_dir, data_dir = ensure_dirs(args.output_dir)

    benchmark_metrics = load_benchmark_metrics()
    baseline_rows = load_baseline_rows()
    ablation_rows = load_ablation_rows()
    family_metrics = load_family_metrics()
    family_counts = load_family_counts()
    cost_rows = load_cost_rows()
    demo_report = load_demo_report()
    risk_rows = build_risk_stratification(data_dir)

    charts = [
        ("01_baseline_suite.png", lambda p: chart_baselines(p, args.dpi, baseline_rows)),
        ("02_observed_fraction_curve.png", lambda p: chart_observed_fraction(p, args.dpi, ablation_rows)),
        ("03_component_ablations.png", lambda p: chart_component_ablations(p, args.dpi, ablation_rows)),
        ("04_family_holdout.png", lambda p: chart_family_holdout(p, args.dpi, benchmark_metrics, family_metrics, family_counts)),
        ("05_cost_frontier.png", lambda p: chart_cost_frontier(p, args.dpi, cost_rows)),
        ("06_risk_stratification.png", lambda p: chart_risk_stratification(p, args.dpi, risk_rows)),
        ("07_demo_case.png", lambda p: chart_demo_case(p, args.dpi, demo_report)),
    ]

    print(f"Writing charts to {out_dir}")
    for filename, builder in charts:
        target = out_dir / filename
        builder(target)
        print(f"  wrote {target.name}")


if __name__ == "__main__":
    main()
