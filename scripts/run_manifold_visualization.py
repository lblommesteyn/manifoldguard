"""Project the learned model manifold and show where a partially observed model lands.

Usage:
    python scripts/run_manifold_visualization.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional visualization dependency
    raise SystemExit("matplotlib is required for manifold visualization. Install with `pip install -e \".[viz]\"`.") from exc

from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manifoldguard.data import load_score_csv
from manifoldguard.ensemble import train_ensemble
from manifoldguard.inference import infer_latent_u_ridge
from manifoldguard.splits import infer_model_family

DATASET = REPO_ROOT / "datasets" / "lm_eval_real" / "scores.csv"
DEFAULT_REQUEST = REPO_ROOT / "examples" / "demo_request.json"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "manifold_visualization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the learned model manifold.")
    parser.add_argument("--csv", type=Path, default=DATASET, help="Score matrix CSV to use.")
    parser.add_argument("--request", type=Path, default=None, help="Optional partial-observation request JSON.")
    parser.add_argument("--model-name", type=str, default=None, help="Model to auto-sample from the CSV.")
    parser.add_argument(
        "--observed-fraction",
        type=float,
        default=0.35,
        help="Observed fraction for auto-sampled target requests.",
    )
    parser.add_argument("--rank", type=int, default=3, help="MF latent rank.")
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument(
        "--label-neighbors",
        type=int,
        default=5,
        help="Number of nearest training models to annotate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_matrix = load_score_csv(args.csv)
    request = _resolve_request(args, score_matrix)

    benchmark_to_idx = {
        _normalize_benchmark_name(name): idx
        for idx, name in enumerate(score_matrix.benchmark_names)
    }
    observed_items = list(request["observed_scores"].items())
    observed_names = [_normalize_benchmark_name(name) for name, _ in observed_items]
    observed_indices = np.asarray([benchmark_to_idx[name] for name in observed_names], dtype=int)
    observed_values = np.asarray([float(value) for _, value in observed_items], dtype=float)

    target_idx = score_matrix.model_names.index(request["model_name"])
    train_mask = np.ones(len(score_matrix.model_names), dtype=bool)
    train_mask[target_idx] = False
    train_matrix = score_matrix.values[train_mask]
    train_names = [name for idx, name in enumerate(score_matrix.model_names) if train_mask[idx]]

    ensemble = train_ensemble(
        matrix=train_matrix,
        ensemble_size=5,
        rank=args.rank,
        epochs=args.epochs,
        seed=args.seed,
    )
    reference_member = ensemble.members[0]
    target_latent = infer_latent_u_ridge(
        V=reference_member.V,
        observed_indices=observed_indices,
        observed_values=observed_values,
        ridge=1e-2,
    )

    all_latents = np.vstack([reference_member.U, target_latent.reshape(1, -1)])
    model_names = train_names + [request["model_name"]]
    families = [infer_model_family(name) for name in model_names]
    point_types = ["training"] * len(train_names) + ["target"]

    pca = PCA(n_components=2)
    projection = pca.fit_transform(all_latents)
    distances = np.linalg.norm(projection[:-1] - projection[-1], axis=1)
    nearest_idx = np.argsort(distances)[: max(args.label_neighbors, 0)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    projection_path = args.output_dir / "projection.csv"
    _write_projection_csv(projection_path, model_names, families, point_types, projection)
    figure_path = args.output_dir / "manifold_plot.png"
    _write_plot(
        figure_path=figure_path,
        model_names=model_names,
        families=families,
        point_types=point_types,
        projection=projection,
        target_name=request["model_name"],
        nearest_idx=nearest_idx,
    )

    print(f"Projection written to {projection_path}")
    print(f"Figure written to {figure_path}")
    if nearest_idx.size > 0:
        print("Nearest training neighbors:")
        for idx in nearest_idx.tolist():
            print(f"  {model_names[idx]}  family={families[idx]}  distance={distances[idx]:.4f}")


def _resolve_request(args: argparse.Namespace, score_matrix) -> dict[str, object]:
    if args.model_name is not None:
        return _sample_request_from_model(
            score_matrix=score_matrix,
            model_name=args.model_name,
            observed_fraction=args.observed_fraction,
            seed=args.seed,
        )

    request_path = args.request or DEFAULT_REQUEST
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Demo request must be a JSON object.")
    return payload


def _sample_request_from_model(score_matrix, model_name: str, observed_fraction: float, seed: int) -> dict[str, object]:
    if not 0.0 < observed_fraction < 1.0:
        raise ValueError("observed_fraction must be in (0, 1).")

    target_idx = score_matrix.model_names.index(model_name)
    target_row = score_matrix.values[target_idx]
    observed_full = np.flatnonzero(~np.isnan(target_row))
    if observed_full.size < 3:
        raise ValueError("Auto-sampled manifold view requires at least 3 observed benchmarks.")

    max_observed = observed_full.size - 1
    desired_observed = int(np.floor(observed_fraction * observed_full.size))
    observed_count = int(np.clip(desired_observed, 2, max_observed))
    rng = np.random.default_rng(seed)
    observed_indices = np.sort(rng.choice(observed_full, size=observed_count, replace=False))
    observed_scores = {
        score_matrix.benchmark_names[idx]: float(target_row[idx])
        for idx in observed_indices.tolist()
    }
    return {"model_name": model_name, "observed_scores": observed_scores}


def _normalize_benchmark_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _write_projection_csv(
    path: Path,
    model_names: list[str],
    families: list[str],
    point_types: list[str],
    projection: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_name", "family", "point_type", "pc1", "pc2"])
        for name, family, point_type, coords in zip(model_names, families, point_types, projection):
            writer.writerow([name, family, point_type, f"{coords[0]:.6f}", f"{coords[1]:.6f}"])


def _write_plot(
    figure_path: Path,
    model_names: list[str],
    families: list[str],
    point_types: list[str],
    projection: np.ndarray,
    target_name: str,
    nearest_idx: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    unique_families = sorted({family for family, point_type in zip(families, point_types) if point_type == "training"})
    cmap = plt.get_cmap("tab20")
    family_to_color = {family: cmap(idx % 20) for idx, family in enumerate(unique_families)}

    for family in unique_families:
        indices = [
            idx for idx, (fam, point_type) in enumerate(zip(families, point_types))
            if fam == family and point_type == "training"
        ]
        ax.scatter(
            projection[indices, 0],
            projection[indices, 1],
            s=54,
            alpha=0.82,
            color=family_to_color[family],
            label=family,
            edgecolors="none",
        )

    target_idx = len(model_names) - 1
    ax.scatter(
        projection[target_idx, 0],
        projection[target_idx, 1],
        s=280,
        marker="*",
        color="#C93B48",
        edgecolors="black",
        linewidths=1.0,
        zorder=10,
        label="target model",
    )
    ax.annotate(
        target_name,
        (projection[target_idx, 0], projection[target_idx, 1]),
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=10.5,
        fontweight="bold",
        color="#0D2233",
    )

    for idx in nearest_idx.tolist():
        ax.annotate(
            model_names[idx],
            (projection[idx, 0], projection[idx, 1]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.5,
            color="#4A6375",
        )

    ax.set_title("Learned model manifold with inferred target position")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="training families")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
