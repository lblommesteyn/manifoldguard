"""Generate slide-ready PNG figures from the real-data artifacts on main.

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
from textwrap import fill

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge
except ImportError as exc:  # pragma: no cover - import guard for optional dependency
    raise SystemExit("matplotlib is required for figure generation. Install with `pip install .[viz]`.") from exc

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
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "conference_figures"

RISK_SEEDS = [0, 1, 2, 3, 4]

PALETTE = {
    # Backgrounds — warm parchment with cool-mint counterpoint
    "bg0": "#F3EDE1",
    "bg1": "#FAF4EA",
    "bg2": "#E4F2EC",
    # Text
    "ink": "#0D2233",
    "muted": "#4A6375",
    # Structural
    "grid": "#DDE7EF",
    "panel": "#FEFCF8",
    "panel_edge": "#CEBFA0",
    # Orange accent family
    "accent": "#E86535",
    "accent_dark": "#C04C1E",
    "accent_soft": "#F5A880",
    # Teal family
    "teal": "#18A8B8",
    "teal_dark": "#0E6E79",
    "teal_light": "#B7E8EE",
    # Gold family
    "gold": "#F2A94E",
    "gold_dark": "#A86B12",
    "gold_light": "#FDEAC6",
    # Status / categorical
    "crimson": "#C93B48",
    "navy": "#1A3575",
    # UI helpers
    "lavender": "#D4E5FA",
    "smoke": "#EBF0F5",
    "white": "#FFFFFF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate polished PNG figures.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for the PNG outputs.")
    parser.add_argument("--dpi", type=int, default=320, help="PNG DPI.")
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.facecolor": PALETTE["bg0"],
            "axes.facecolor": PALETTE["panel"],
            "axes.edgecolor": PALETTE["panel_edge"],
            "axes.linewidth": 0.9,
            "axes.labelcolor": PALETTE["ink"],
            "axes.labelsize": 11,
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlesize": 13,
            "axes.titlepad": 12,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.major.pad": 5,
            "ytick.major.pad": 5,
            "text.color": PALETTE["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.65,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "lines.linewidth": 2.5,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.9,
            "legend.fontsize": 10.5,
            "legend.framealpha": 0.92,
            "legend.edgecolor": PALETTE["panel_edge"],
            "legend.borderpad": 0.7,
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
    figure_dir = out_dir
    data_dir = out_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir, data_dir


def add_gradient_background(fig: mpl.figure.Figure) -> None:
    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-50)
    bg_ax.set_axis_off()
    # Diagonal gradient: warm top-left → cool bottom-right
    g_v = np.linspace(1, 0, 900)
    g_h = np.linspace(0, 1, 900)
    gradient = np.outer(g_v, np.ones(900)) * 0.60 + np.outer(np.ones(900), g_h) * 0.40
    cmap = LinearSegmentedColormap.from_list("mg_bg", [PALETTE["bg1"], PALETTE["bg0"], PALETTE["bg2"]])
    bg_ax.imshow(gradient, aspect="auto", cmap=cmap, origin="lower", extent=[0, 1, 0, 1], interpolation="bicubic")

    # Thin header bar that frames the title area
    bg_ax.add_patch(Rectangle((0, 0.88), 1, 0.12, transform=bg_ax.transData, facecolor=PALETTE["bg1"], edgecolor="none", alpha=0.55, zorder=-18))
    # Thin accent rule below header
    bg_ax.add_patch(Rectangle((0.045, 0.875), 0.91, 0.0025, transform=bg_ax.transData, facecolor=PALETTE["panel_edge"], edgecolor="none", alpha=0.55, zorder=-17))

    fig.patches.extend(
        [
            # Top-left warm orb
            Ellipse(
                (0.08, 0.92),
                0.28,
                0.20,
                transform=fig.transFigure,
                facecolor=PALETTE["accent_soft"],
                edgecolor="none",
                alpha=0.13,
                zorder=-20,
            ),
            # Bottom-right cool orb
            Ellipse(
                (0.94, 0.12),
                0.36,
                0.26,
                transform=fig.transFigure,
                facecolor=PALETTE["teal"],
                edgecolor="none",
                alpha=0.08,
                zorder=-20,
            ),
            # Top-right gold accent
            Ellipse(
                (0.80, 0.88),
                0.18,
                0.13,
                transform=fig.transFigure,
                facecolor=PALETTE["gold"],
                edgecolor="none",
                alpha=0.10,
                zorder=-20,
            ),
            # Bottom-left lavender accent
            Ellipse(
                (0.06, 0.08),
                0.22,
                0.16,
                transform=fig.transFigure,
                facecolor=PALETTE["lavender"],
                edgecolor="none",
                alpha=0.22,
                zorder=-20,
            ),
        ]
    )


def add_figure_titles(fig: mpl.figure.Figure, kicker: str, title: str, subtitle: str) -> None:
    # Kicker chip — small coloured label
    kicker_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=5)
    kicker_ax.set_axis_off()
    kicker_ax.add_patch(
        FancyBboxPatch(
            (0.052, 0.945),
            0.072,
            0.028,
            boxstyle="round,pad=0.005,rounding_size=0.006",
            linewidth=0,
            facecolor=PALETTE["accent_dark"],
            edgecolor="none",
            alpha=0.88,
            transform=fig.transFigure,
        )
    )
    fig.text(0.058, 0.955, kicker.upper(), fontsize=8.5, fontweight="bold", color=PALETTE["white"], va="center")
    fig.text(0.055, 0.912, title, fontsize=22, fontweight="bold", color=PALETTE["ink"], linespacing=1.15)
    fig.text(0.055, 0.873, subtitle, fontsize=11, color=PALETTE["muted"], linespacing=1.35)


def style_axis(ax: mpl.axes.Axes, title: str | None = None, ygrid: bool = True) -> None:
    ax.set_facecolor(PALETTE["panel"])
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["panel_edge"])
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(length=0, pad=5)
    if title:
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=14, color=PALETTE["ink"])
    axis = "y" if ygrid else "x"
    ax.grid(axis=axis, color=PALETTE["grid"], linewidth=0.65, alpha=0.70, zorder=0)
    # Subtle inner top/right fill line for panel depth
    ax.set_axisbelow(True)


def add_panel_box(ax: mpl.axes.Axes, x: float, y: float, w: float, h: float, fc: str, ec: str, alpha: float = 1.0) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.03",
            linewidth=1.5,
            facecolor=fc,
            edgecolor=ec,
            alpha=alpha,
        )
    )


def draw_metric_card(ax: mpl.axes.Axes, x: float, y: float, w: float, h: float, label: str, value: str, accent: str) -> None:
    # Shadow layer
    add_panel_box(ax, x + 0.004, y - 0.006, w, h, fc=PALETTE["ink"], ec="none", alpha=0.06)
    # Card face
    add_panel_box(ax, x, y, w, h, fc=PALETTE["white"], ec=PALETTE["panel_edge"])
    # Bottom accent stripe (visual weight at base)
    ax.add_patch(Rectangle((x + 0.003, y), w - 0.006, 0.018, color=accent, lw=0, zorder=3))
    # Label
    ax.text(x + 0.03, y + h - 0.050, label.upper(), fontsize=8.8, color=PALETTE["muted"],
            fontweight="bold", va="top", linespacing=1.2)
    # Value — vertically centred in the card body
    ax.text(x + 0.03, y + h * 0.40, value, fontsize=22, fontweight="bold", color=PALETTE["ink"], va="center")


def save_figure(fig: mpl.figure.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def load_benchmark_metrics() -> dict[str, float]:
    rows = read_csv_rows(BENCHMARK_SUMMARY)
    return {row["metric"]: as_float(row["mean"]) for row in rows}


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
                "setting": as_float(row["setting"]),
                "completion_mae": as_float(row["completion_mae"]),
                "failure_auc": as_float(row["failure_auc"]),
                "conformal_coverage": as_float(row["conformal_coverage"]),
            }
        )
    return sorted(rows, key=lambda entry: entry["setting"])


def load_family_summary() -> dict[str, float]:
    rows = read_csv_rows(FAMILY_SUMMARY)
    return {row["metric"]: as_float(row["mean"]) for row in rows}


def load_family_counts() -> list[tuple[str, int]]:
    rows = []
    for row in read_csv_rows(FAMILY_COUNTS):
        rows.append((row["family"], int(as_float(row["num_models"]))))
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


def generate_method_overview(path: Path, dpi: int, benchmark_metrics: dict[str, float], demo_report: dict[str, object]) -> None:
    fig, (ax_diag, ax_metrics) = plt.subplots(1, 2, figsize=(14, 5),
                                               gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.12})
    fig.patch.set_facecolor("white")
    fig.suptitle("ManifoldGuard — method overview", fontsize=14, fontweight="bold", x=0.04, ha="left", y=1.01)

    # ── Pipeline diagram (left) ──────────────────────────────────────────────
    ax_diag.set_xlim(0, 1)
    ax_diag.set_ylim(0, 1)
    ax_diag.axis("off")

    steps = [
        ("1. Partial\nobservations", PALETTE["navy"]),
        ("2. Ensemble\nfactorization", PALETTE["accent"]),
        ("3. Reliability\nlayer", PALETTE["teal"]),
        ("4. Decision\noutput", PALETTE["gold_dark"]),
    ]
    bw, bh, by = 0.20, 0.32, 0.34
    xs = [0.02, 0.27, 0.52, 0.77]
    for (label, color), bx in zip(steps, xs):
        ax_diag.add_patch(Rectangle((bx, by), bw, bh, facecolor=color, edgecolor=color, alpha=0.12, linewidth=0))
        ax_diag.add_patch(Rectangle((bx, by + bh - 0.025), bw, 0.025, facecolor=color, edgecolor="none"))
        ax_diag.text(bx + bw / 2, by + bh / 2, label, ha="center", va="center",
                     fontsize=11, fontweight="bold", color=PALETTE["ink"], linespacing=1.4)
    for x0, x1 in zip([0.22, 0.47, 0.72], [0.27, 0.52, 0.77]):
        ax_diag.annotate("", xy=(x1, by + bh / 2), xytext=(x0, by + bh / 2),
                         arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.4))

    # Observed scores listed below the pipeline
    for idx, (name, value) in enumerate(list(demo_report["observed_scores"].items())):
        ax_diag.text(0.04 + idx * 0.32, 0.22, f"{name}  {float(value):.3f}",
                     fontsize=10, color=PALETTE["ink"],
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", linewidth=0.8))

    decision = str(demo_report["decision"]).upper()
    ax_diag.text(0.5, 0.06, f"Decision: {decision}", ha="center", fontsize=12, fontweight="bold",
                 color=PALETTE["crimson"] if demo_report["decision"] == "continue" else PALETTE["teal_dark"])

    # ── Key metrics bar chart (right) ─────────────────────────────────────────
    ax_metrics.set_facecolor("white")
    for spine in ax_metrics.spines.values():
        spine.set_visible(False)
    metric_labels = ["Completion\nMAE", "Failure\nAUC", "Conformal\ncoverage", "Quantile"]
    metric_keys = ["completion_mae", "failure_auc", "conformal_coverage", "conformal_quantile"]
    values = [benchmark_metrics[k] for k in metric_keys]
    colors = [PALETTE["accent"], PALETTE["teal"], PALETTE["gold_dark"], PALETTE["navy"]]
    yp = np.arange(len(metric_labels))
    bars = ax_metrics.barh(yp, values, color=colors, height=0.5, edgecolor="none")
    ax_metrics.set_yticks(yp)
    ax_metrics.set_yticklabels(metric_labels, fontsize=10.5)
    ax_metrics.tick_params(axis="both", length=0)
    ax_metrics.set_xlabel("Value", fontsize=10.5)
    ax_metrics.set_title("Key metrics", loc="left", fontsize=12, fontweight="bold", pad=10)
    ax_metrics.grid(axis="x", color="#DDDDDD", linewidth=0.7, alpha=0.8)
    ax_metrics.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax_metrics.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=10.5, fontweight="bold", color=PALETTE["ink"])

    save_figure(fig, path, dpi)


def generate_baseline_comparison(path: Path, dpi: int, baseline_rows: list[dict[str, float | str]]) -> None:
    fig = plt.figure(figsize=(14, 5.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Baseline comparison", fontsize=13, fontweight="bold", x=0.04, ha="left", y=1.02)
    gs = fig.add_gridspec(1, 3, left=0.04, right=0.98, bottom=0.12, top=0.94, wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    metric_specs = [
        ("completion_mae_mean", "completion_mae_std", "Completion MAE", "Lower is better", True),
        ("failure_auc_mean", "failure_auc_std", "Failure AUC", "Higher is better", False),
        ("conformal_coverage_mean", "conformal_coverage_std", "Coverage", "Higher is better", False),
    ]

    names = [str(row["method"]) for row in baseline_rows]
    y_positions = np.arange(len(names))[::-1]
    color_map = {
        "Mean fill": "#B0BECA",
        "Nearest neighbor": "#8FAABF",
        "MF only": "#6A8FAD",
        "MF + failure detection": PALETTE["teal"],
        "ManifoldGuard (full)": PALETTE["accent"],
    }

    for ax, (metric_key, std_key, title, note, invert) in zip(axes, metric_specs):
        style_axis(ax, title=title)
        values = [float(row[metric_key]) for row in baseline_rows]
        stds = [float(row[std_key]) for row in baseline_rows]
        finite_values = [val for val in values if np.isfinite(val)]
        upper = max(finite_values) * (1.28 if title != "Coverage" else 1.14)
        ax.set_xlim((upper, 0) if invert else (0, upper))
        ax.set_yticks(y_positions)
        ax.set_yticklabels(names, fontsize=10.5)
        ax.set_xlabel(note, fontsize=10.5, color=PALETTE["muted"], labelpad=10)

        for y, name, value, std in zip(y_positions, names, values, stds):
            is_best = name == "ManifoldGuard (full)"
            if np.isfinite(value):
                color = color_map[name]
                bar = ax.barh(y, value, color=color, alpha=0.93, height=0.58,
                              edgecolor="white" if is_best else "none", linewidth=1.5 if is_best else 0)
                if np.isfinite(std) and std > 0:
                    ax.errorbar(value, y, xerr=std, fmt="none", ecolor=PALETTE["ink"],
                                elinewidth=1.1, capsize=4, alpha=0.65)
                offset = -0.01 * upper if invert else 0.01 * upper
                ha = "right" if invert else "left"
                fw = "bold" if is_best else "normal"
                ax.text(value + offset, y, f"{value:.3f}", va="center", ha=ha,
                        fontsize=10.5, fontweight=fw, color=PALETTE["ink"])
            else:
                xloc = upper * (0.56 if invert else 0.08)
                ax.hlines(y, xloc - upper * 0.05, xloc + upper * 0.05,
                          color=PALETTE["grid"], lw=1.8, linestyles="dashed")
                ax.text(
                    xloc + (upper * 0.07 if not invert else -upper * 0.07),
                    y, "N/A", va="center",
                    ha="left" if not invert else "right",
                    fontsize=10, color=PALETTE["muted"],
                )

        if title == "Coverage":
            ax.axvline(0.90, color=PALETTE["navy"], linestyle=(0, (5, 3)), linewidth=1.6, alpha=0.85)
            ax.text(0.902, len(names) - 0.45, "90% target", color=PALETTE["navy"],
                    fontsize=10, fontweight="bold")

    save_figure(fig, path, dpi)


def generate_observed_fraction_curve(path: Path, dpi: int, observed_rows: list[dict[str, float]]) -> None:
    fractions = np.asarray([row["setting"] for row in observed_rows], dtype=float)
    maes = np.asarray([row["completion_mae"] for row in observed_rows], dtype=float)
    aucs = np.asarray([row["failure_auc"] for row in observed_rows], dtype=float)
    coverages = np.asarray([row["conformal_coverage"] for row in observed_rows], dtype=float)

    fig = plt.figure(figsize=(15.8, 8.8))
    add_gradient_background(fig)
    add_figure_titles(
        fig,
        "Figure 03",
        "More revealed benchmarks rapidly improve the completion quality",
        "The observed-fraction ablation traces a clear tradeoff: reveal a bit more of the evaluation grid and the hidden-score error drops while failure ranking improves.",
    )
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.96, bottom=0.13, top=0.78, wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    panels = [
        ("Completion MAE", maes, PALETTE["accent"]),
        ("Failure AUC", aucs, PALETTE["teal"]),
        ("Coverage", coverages, PALETTE["gold_dark"]),
    ]
    for ax, (title, values, color) in zip(axes, panels):
        style_axis(ax, title=title)
        # Gradient fill under the line
        ax.fill_between(fractions, values, alpha=0.13, color=color, zorder=1)
        # Second, slightly darker band near the line for depth
        ax.fill_between(fractions, values, [v * 0.97 + min(values) * 0.03 for v in values],
                        alpha=0.18, color=color, zorder=2)
        ax.plot(fractions, values, color=color, linewidth=2.8, zorder=3, solid_capstyle="round")
        # Default-region shading
        ax.axvspan(0.47, 0.53, color=PALETTE["smoke"], alpha=0.85, zorder=0)
        ax.text(0.5, 0.04 if title == "Completion MAE" else 0.97, "default",
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom" if title == "Completion MAE" else "top",
                fontsize=10, color=PALETTE["muted"], rotation=90)
        ax.set_xticks(fractions)
        ax.set_xticklabels([f"{int(frac * 100)}%" for frac in fractions], fontsize=10.5)
        ax.set_xlabel("Observed benchmark fraction", fontsize=10.5)
        if title == "Coverage":
            ax.axhline(0.90, color=PALETTE["navy"], linestyle=(0, (5, 3)), linewidth=1.5, alpha=0.85)
            ax.text(fractions[0], 0.9035, "90% target", fontsize=10, color=PALETTE["navy"], fontweight="bold")
        for frac, value in zip(fractions, values):
            # Markers: white ring + coloured dot
            ax.scatter([frac], [value], s=90, color=color, edgecolors="white", linewidths=2.0, zorder=6)
            delta = 0.012 if title != "Completion MAE" else 0.003
            # Small value label with subtle white backing
            ax.text(frac, value + delta, f"{value:.3f}", ha="center", fontsize=10,
                    color=PALETTE["ink"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.72))

    mae_gain = (maes[0] - maes[-1]) / maes[0]
    auc_gain = aucs[-1] - aucs[0]
    fig.text(0.06, 0.82, f"{mae_gain:.0%} lower MAE from 30% to 70% observed", fontsize=12.5, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.36, 0.82, f"+{auc_gain:.3f} AUC over the same range", fontsize=12.5, fontweight="bold", color=PALETTE["teal_dark"])
    save_figure(fig, path, dpi)


def generate_family_split_figure(
    path: Path,
    dpi: int,
    benchmark_metrics: dict[str, float],
    family_metrics: dict[str, float],
    family_counts: list[tuple[str, int]],
) -> None:
    fig = plt.figure(figsize=(15.5, 9.2))
    add_gradient_background(fig)
    add_figure_titles(
        fig,
        "Figure 04",
        "Family holdout is harder, but the system still holds together",
        "A stronger split by model family raises the completion error, yet the interval coverage remains near target and the failure ranking stays useful.",
    )
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.96, bottom=0.12, top=0.79, width_ratios=[1.3, 1.0], wspace=0.16)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    style_axis(ax_left, title="Random split vs family holdout")
    style_axis(ax_right, title="Model-family coverage in the dataset", ygrid=False)

    metrics = ["completion_mae", "failure_auc", "conformal_coverage", "conformal_quantile"]
    labels = ["MAE", "AUC", "Coverage", "Quantile"]
    random_values = [benchmark_metrics[key] for key in metrics]
    family_values = [family_metrics[key] for key in metrics]
    x = np.arange(len(metrics))
    width = 0.34
    ax_left.bar(x - width / 2, random_values, width, color=PALETTE["teal"], label="Random split",
                alpha=0.90, edgecolor="white", linewidth=1.2)
    ax_left.bar(x + width / 2, family_values, width, color=PALETTE["accent"], label="Family holdout",
                alpha=0.90, edgecolor="white", linewidth=1.2)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, fontsize=11)
    ax_left.legend(loc="upper left")
    for xpos, rnd, fam in zip(x, random_values, family_values):
        ax_left.text(xpos - width / 2, rnd + 0.010, f"{rnd:.3f}", ha="center", fontsize=10,
                     color=PALETTE["ink"], fontweight="bold")
        ax_left.text(xpos + width / 2, fam + 0.010, f"{fam:.3f}", ha="center", fontsize=10,
                     color=PALETTE["ink"], fontweight="bold")
        delta = fam - rnd
        delta_color = PALETTE["crimson"] if delta > 0 and xpos == 0 else PALETTE["muted"]
        ax_left.text(xpos, max(rnd, fam) + 0.055, f"Δ {delta:+.3f}", ha="center", fontsize=10.5,
                     fontweight="bold", color=delta_color,
                     bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.72))
    ax_left.set_ylim(0, max(max(random_values), max(family_values)) * 1.30)

    top_families = family_counts[:8]
    fam_names = [name for name, _ in top_families][::-1]
    counts = [count for _, count in top_families][::-1]
    bars = ax_right.barh(fam_names, counts, color=PALETTE["navy"], alpha=0.88, height=0.62)
    ax_right.set_xlabel("Number of models", fontsize=10.5)
    for bar, count in zip(bars, counts):
        ax_right.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=10.5)
    total_models = sum(count for _, count in family_counts)
    total_families = len(family_counts)
    fig.text(0.06, 0.82, f"{total_models} models across {total_families} inferred families", fontsize=12.5, fontweight="bold")
    fig.text(0.06, 0.795, "The hard split holds out entire families such as Gemma, Qwen, Falcon, or Yi in each seed.", fontsize=11.2, color=PALETTE["muted"])
    save_figure(fig, path, dpi)


def generate_cost_frontier(path: Path, dpi: int, cost_rows: list[dict[str, float]]) -> None:
    thresholds = np.asarray([row["risk_threshold"] for row in cost_rows], dtype=float)
    avoided = np.asarray([row["benchmarks_avoided_fraction_mean"] for row in cost_rows], dtype=float)
    failure_rates = np.asarray([row["accepted_failure_rate_mean"] for row in cost_rows], dtype=float)
    accepted_share = np.asarray([row["episodes_accepted_mean"] for row in cost_rows], dtype=float)

    fig = plt.figure(figsize=(15.8, 9.0))
    add_gradient_background(fig)
    add_figure_titles(
        fig,
        "Figure 05",
        "Risk gating can skip most of the hidden grid",
        "As the acceptance threshold loosens, more benchmark rows can be avoided. The 0.35 threshold is the practical knee: strong savings without an obvious quality collapse.",
    )
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.96, bottom=0.11, top=0.79, width_ratios=[1.35, 0.9], wspace=0.15)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    style_axis(ax_left, title="Savings frontier", ygrid=True)
    style_axis(ax_right, title="Threshold ladder", ygrid=False)

    sizes = 520 * accepted_share + 110
    frontier_cmap = LinearSegmentedColormap.from_list(
        "mg_frontier", [PALETTE["teal"], PALETTE["gold"], PALETTE["accent"]]
    )
    # Connecting line drawn first
    ax_left.plot(avoided, failure_rates, color=PALETTE["muted"], linewidth=1.8, alpha=0.55, zorder=2)
    scatter = ax_left.scatter(
        avoided,
        failure_rates,
        s=sizes,
        c=thresholds,
        cmap=frontier_cmap,
        edgecolors="white",
        linewidths=2.0,
        alpha=0.95,
        zorder=3,
    )
    for x, y, threshold in zip(avoided, failure_rates, thresholds):
        label = f"{threshold:.2f}"
        is_knee = math.isclose(threshold, 0.35, rel_tol=0.0, abs_tol=1e-9)
        fw = "bold" if is_knee else "normal"
        fc = PALETTE["accent_dark"] if is_knee else "white"
        ec = PALETTE["accent_dark"] if is_knee else PALETTE["panel_edge"]
        ax_left.text(x, y + 0.016, label, ha="center", fontsize=10.5, fontweight=fw, color=PALETTE["ink"],
                     bbox=dict(boxstyle="round,pad=0.18", fc=fc, ec=ec, alpha=0.88 if is_knee else 0.80, linewidth=0.8))
    ax_left.set_xlabel("Benchmarks avoided fraction", fontsize=11)
    ax_left.set_ylabel("Accepted failure rate", fontsize=11)
    ax_left.axvspan(0.60, 0.72, color=PALETTE["gold_light"], alpha=0.55, zorder=0)
    ax_left.axhspan(0.0, 0.28, color=PALETTE["teal_light"], alpha=0.30, zorder=0)
    ax_left.text(0.648, 0.31, "sweet spot", fontsize=11, fontweight="bold", color=PALETTE["gold_dark"])
    cbar = fig.colorbar(scatter, ax=ax_left, fraction=0.034, pad=0.02)
    cbar.set_label("Risk threshold", fontsize=10.5)
    cbar.ax.tick_params(labelsize=10, length=0)

    highlighted = next(row for row in cost_rows if math.isclose(row["risk_threshold"], 0.35, abs_tol=1e-12))
    draw_metric_card(ax_right, 0.04, 0.66, 0.90, 0.22, "Recommended threshold", "0.35", PALETTE["accent"])
    draw_metric_card(ax_right, 0.04, 0.38, 0.43, 0.20, "Benchmarks avoided", f"{highlighted['benchmarks_avoided_fraction_mean']:.1%}", PALETTE["teal"])
    draw_metric_card(ax_right, 0.51, 0.38, 0.43, 0.20, "Accepted MAE", f"{highlighted['accepted_mae_mean']:.3f}", PALETTE["gold_dark"])
    draw_metric_card(ax_right, 0.04, 0.10, 0.43, 0.20, "Episodes accepted", f"{highlighted['episodes_accepted_mean']:.1%}", PALETTE["navy"])
    draw_metric_card(ax_right, 0.51, 0.10, 0.43, 0.20, "Failure rate", f"{highlighted['accepted_failure_rate_mean']:.1%}", PALETTE["crimson"])
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)
    ax_right.axis("off")
    save_figure(fig, path, dpi)


def random_split_indices(n_models: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_models)
    n_test = max(1, int(round(test_fraction * n_models)))
    n_train = n_models - n_test
    return np.sort(order[:n_train]), np.sort(order[n_train:])


def build_risk_stratification_artifacts(data_dir: Path) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    score_matrix = load_score_csv(SCORE_CSV)
    matrix = score_matrix.values
    episode_rows: list[dict[str, float]] = []

    for seed in RISK_SEEDS:
        train_idx, test_idx = random_split_indices(matrix.shape[0], test_fraction=0.25, seed=seed)
        train_matrix = matrix[train_idx]
        test_matrix = matrix[test_idx]
        ensemble = train_ensemble(matrix=train_matrix, ensemble_size=5, rank=3, seed=seed)
        episodes = simulate_new_model_episodes(matrix=test_matrix, episodes_per_model=3, observed_fraction=0.5, seed=seed + 1)
        results = [_evaluate_episode(ep, ensemble, ridge=1e-2) for ep in episodes]
        hidden_maes = np.asarray([result.hidden_mae for result in results], dtype=float)
        groups = np.asarray([episode.model_index for episode in episodes], dtype=int)
        features = np.vstack([result.features for result in results])
        failure_threshold = float(_quantile_higher(hidden_maes, 0.8))
        labels = (hidden_maes >= failure_threshold).astype(int)
        probabilities = _grouped_failure_probabilities(features, labels, groups, seed=seed)
        for episode_index, (probability, hidden_mae, label, episode, result) in enumerate(
            zip(probabilities, hidden_maes, labels, episodes, results)
        ):
            if not np.isfinite(probability):
                continue
            episode_rows.append(
                {
                    "seed": float(seed),
                    "episode_index": float(episode_index),
                    "risk_probability": float(probability),
                    "hidden_mae": float(hidden_mae),
                    "failure_label": float(label),
                    "num_hidden": float(len(episode.hidden_indices)),
                    "mean_hidden_abs_residual": float(np.mean(result.hidden_abs_residuals)),
                }
            )

    episode_csv = data_dir / "risk_stratification_by_episode.csv"
    with episode_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "episode_index", "risk_probability", "hidden_mae", "failure_label", "num_hidden", "mean_hidden_abs_residual"])
        for row in episode_rows:
            writer.writerow(
                [
                    int(row["seed"]),
                    int(row["episode_index"]),
                    f"{row['risk_probability']:.6f}",
                    f"{row['hidden_mae']:.6f}",
                    int(row["failure_label"]),
                    int(row["num_hidden"]),
                    f"{row['mean_hidden_abs_residual']:.6f}",
                ]
            )

    bin_edges = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.000001], dtype=float)
    binned_rows: list[dict[str, float]] = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        members = [row for row in episode_rows if left <= row["risk_probability"] < right]
        if not members:
            continue
        probabilities = np.asarray([row["risk_probability"] for row in members], dtype=float)
        maes = np.asarray([row["hidden_mae"] for row in members], dtype=float)
        failures = np.asarray([row["failure_label"] for row in members], dtype=float)
        binned_rows.append(
            {
                "bin_left": left,
                "bin_right": min(right, 1.0),
                "mean_probability": float(np.mean(probabilities)),
                "mean_hidden_mae": float(np.mean(maes)),
                "failure_rate": float(np.mean(failures)),
                "count": float(len(members)),
            }
        )

    binned_csv = data_dir / "risk_stratification_bins.csv"
    with binned_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_left", "bin_right", "mean_probability", "mean_hidden_mae", "failure_rate", "count"])
        for row in binned_rows:
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
    return episode_rows, binned_rows


def generate_risk_stratification(path: Path, dpi: int, binned_rows: list[dict[str, float]]) -> None:
    labels = [f"{row['bin_left']:.1f}–{row['bin_right']:.1f}" for row in binned_rows]
    failure_rates = np.asarray([row["failure_rate"] for row in binned_rows], dtype=float)
    mean_maes = np.asarray([row["mean_hidden_mae"] for row in binned_rows], dtype=float)
    counts = np.asarray([row["count"] for row in binned_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    x = np.arange(len(labels))
    bar_colors = [PALETTE["smoke"], PALETTE["gold_light"], PALETTE["gold"], PALETTE["accent"], PALETTE["accent_dark"]]
    bars = ax.bar(x, failure_rates, color=bar_colors[:len(labels)], width=0.55, edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted failure probability bin", fontsize=11)
    ax.set_ylabel("Empirical failure rate", fontsize=11)
    ax.set_ylim(0.0, max(0.60, np.max(failure_rates) * 1.30))
    ax.axhline(0.20, color=PALETTE["navy"], linestyle="--", linewidth=1.4, alpha=0.80)
    ax.text(len(labels) - 0.5, 0.21, "20% prior", fontsize=10, color=PALETTE["navy"], ha="right")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Risk stratification: failure rate and MAE by risk bin", loc="left", fontsize=12, fontweight="bold", pad=10)

    ax2 = ax.twinx()
    ax2.plot(x, mean_maes, color=PALETTE["teal_dark"], linewidth=2.2, marker="o",
             markersize=7, markerfacecolor=PALETTE["teal_dark"], markeredgecolor="white", markeredgewidth=1.5)
    ax2.set_ylabel("Mean hidden MAE", fontsize=11, color=PALETTE["teal_dark"])
    ax2.tick_params(axis="y", colors=PALETTE["teal_dark"], length=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#CCCCCC")
    ax2.grid(False)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"n={int(count)}", ha="center", fontsize=9.5, color="#555555")

    fig.tight_layout()
    save_figure(fig, path, dpi)


def generate_demo_case_study(path: Path, dpi: int, demo_report: dict[str, object]) -> None:
    observed_items = list(demo_report["observed_scores"].items())
    hidden_rows = list(demo_report["hidden_predictions"])
    names = [str(row["benchmark"]) for row in hidden_rows]
    preds = np.asarray([float(row["predicted_score"]) for row in hidden_rows], dtype=float)
    lowers = np.asarray([float(row["interval_lower"]) for row in hidden_rows], dtype=float)
    uppers = np.asarray([float(row["interval_upper"]) for row in hidden_rows], dtype=float)
    truths = np.asarray([float(row["true_score"]) for row in hidden_rows], dtype=float)

    fig, (ax_obs, ax_main) = plt.subplots(1, 2, figsize=(13, 5),
                                           gridspec_kw={"width_ratios": [0.8, 2.2], "wspace": 0.30})
    fig.patch.set_facecolor("white")
    fig.suptitle("Demo case study — conformal predictions", fontsize=13, fontweight="bold",
                 x=0.04, ha="left", y=1.02)

    # ── Observed scores (left) ────────────────────────────────────────────────
    ax_obs.set_facecolor("white")
    for spine in ax_obs.spines.values():
        spine.set_visible(False)
    ax_obs.set_xlim(0, 1)
    ax_obs.set_ylim(-0.5, len(observed_items) - 0.5)
    ax_obs.set_xticks([])
    ax_obs.set_yticks([])
    ax_obs.set_title("Observed scores", loc="left", fontsize=12, fontweight="bold", pad=10)
    for idx, (name, value) in enumerate(reversed(observed_items)):
        y = idx
        ax_obs.axhline(y, color="#EEEEEE", linewidth=0.8)
        ax_obs.text(0.04, y, name, va="center", fontsize=11, color=PALETTE["ink"])
        ax_obs.text(0.96, y, f"{float(value):.3f}", va="center", ha="right",
                    fontsize=11, fontweight="bold", color=PALETTE["accent_dark"])

    # ── Predictions + conformal intervals (right) ─────────────────────────────
    ax_main.set_facecolor("white")
    for spine in ("top", "right"):
        ax_main.spines[spine].set_visible(False)
    ax_main.grid(axis="y", color="#EEEEEE", linewidth=0.8)
    ax_main.set_axisbelow(True)

    x = np.arange(len(names))
    ax_main.errorbar(x, preds,
                     yerr=np.vstack([preds - lowers, uppers - preds]),
                     fmt="o", color=PALETTE["accent"], ecolor="#AAAAAA",
                     elinewidth=6, capsize=0, markersize=7,
                     markeredgecolor="white", markeredgewidth=1.5,
                     label="Prediction ± conformal interval")
    ax_main.scatter(x, truths, color=PALETTE["navy"], marker="D", s=50,
                    edgecolors="white", linewidths=1.5, zorder=5, label="Ground truth")
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(names, fontsize=10, rotation=20, ha="right")
    ax_main.set_ylabel("Score", fontsize=11)
    ax_main.set_ylim(min(lowers.min(), truths.min()) - 0.06, max(uppers.max(), truths.max()) + 0.10)
    ax_main.legend(fontsize=10, frameon=True, loc="upper left")
    ax_main.set_title(
        f"Hidden benchmark predictions  |  Risk: {float(demo_report['risk_probability']):.0%}  |  "
        f"Decision: {str(demo_report['decision']).upper()}",
        loc="left", fontsize=11, fontweight="bold", pad=10,
    )
    for xi, pred, truth in zip(x, preds, truths):
        ax_main.text(xi, uppers[xi] + 0.012, f"Δ{abs(pred - truth):.3f}",
                     ha="center", fontsize=9, color="#888888")

    fig.tight_layout()
    save_figure(fig, path, dpi)


def main() -> None:
    args = parse_args()
    configure_style()
    out_dir, data_dir = ensure_dirs(args.output_dir)

    benchmark_metrics = load_benchmark_metrics()
    baseline_rows = load_baseline_rows()
    observed_rows = load_observed_fraction_rows()
    family_metrics = load_family_summary()
    family_counts = load_family_counts()
    cost_rows = load_cost_rows()
    demo_report = load_demo_report()
    _, risk_bins = build_risk_stratification_artifacts(data_dir)

    manifest = [
        ("01_method_overview.png", lambda p: generate_method_overview(p, args.dpi, benchmark_metrics, demo_report)),
        ("02_baseline_comparison.png", lambda p: generate_baseline_comparison(p, args.dpi, baseline_rows)),
        ("03_observed_fraction_curve.png", lambda p: generate_observed_fraction_curve(p, args.dpi, observed_rows)),
        ("04_family_holdout_vs_random.png", lambda p: generate_family_split_figure(p, args.dpi, benchmark_metrics, family_metrics, family_counts)),
        ("05_cost_savings_frontier.png", lambda p: generate_cost_frontier(p, args.dpi, cost_rows)),
        ("06_risk_stratification.png", lambda p: generate_risk_stratification(p, args.dpi, risk_bins)),
        ("07_demo_case_study.png", lambda p: generate_demo_case_study(p, args.dpi, demo_report)),
    ]

    print(f"Writing figures to {out_dir}")
    for filename, builder in manifest:
        target = out_dir / filename
        builder(target)
        print(f"  wrote {target.name}")


if __name__ == "__main__":
    main()
