"""Run a cold-start demo on a fixed or auto-sampled partial benchmark request.

Usage:
    python scripts/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manifoldguard._utils import quantile_higher as _quantile_higher
from manifoldguard.conformal import split_conformal_quantile
from manifoldguard.data import ScoreMatrix, load_score_csv
from manifoldguard.ensemble import EnsembleMF, predict_new_model, train_ensemble
from manifoldguard.episodes import simulate_new_model_episodes
from manifoldguard.evaluation import _evaluate_episode, _grouped_calibration_test_indices
from manifoldguard.ood import observation_coverage_features, residual_features, summary_variance_features
from manifoldguard.inference import loo_observed_residuals
from manifoldguard.lm_eval import load_lm_eval_results_dir
from manifoldguard.splits import infer_model_family

DATASET = REPO_ROOT / "datasets" / "lm_eval_real" / "scores.csv"
DEFAULT_REQUEST = REPO_ROOT / "examples" / "demo_request.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "demo" / "demo_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ManifoldGuard demo path.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, default=None, help="Optional score matrix CSV to demo on.")
    source.add_argument(
        "--lm-eval-dir",
        type=Path,
        default=None,
        help="Optional lm-eval-harness JSON directory to demo on.",
    )
    parser.add_argument("--request", type=Path, default=None, help="Optional demo request JSON.")
    parser.add_argument("--model-name", type=str, default=None, help="Model to sample a demo episode from.")
    parser.add_argument(
        "--observed-fraction",
        type=float,
        default=0.35,
        help="Observed benchmark fraction for auto-sampled demo episodes.",
    )
    parser.add_argument("--show-all", action="store_true", help="Print all hidden-benchmark predictions.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT, help="Path to write demo report JSON.")
    parser.add_argument("--rank", type=int, default=3, help="MF latent rank.")
    parser.add_argument("--ensemble-size", type=int, default=5, help="Ensemble size.")
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs for the demo run.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Conformal miscoverage level.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_matrix = _load_score_matrix(args)
    request = _resolve_request(args, score_matrix)

    benchmark_to_idx = {name: idx for idx, name in enumerate(score_matrix.benchmark_names)}
    try:
        target_idx = score_matrix.model_names.index(request["model_name"])
    except ValueError as exc:
        raise ValueError(f"Unknown model in demo request: {request['model_name']}") from exc

    target_row = score_matrix.values[target_idx]
    observed_scores = request["observed_scores"]
    observed_names = list(observed_scores)
    observed_indices = np.asarray([benchmark_to_idx[name] for name in observed_names], dtype=int)
    observed_values = np.asarray([float(observed_scores[name]) for name in observed_names], dtype=float)

    missing_from_target = [name for name in observed_names if np.isnan(target_row[benchmark_to_idx[name]])]
    if missing_from_target:
        raise ValueError(f"Demo request includes benchmarks with missing ground truth: {missing_from_target}")

    hidden_indices = np.asarray(
        [idx for idx in np.flatnonzero(~np.isnan(target_row)) if idx not in set(observed_indices.tolist())],
        dtype=int,
    )
    if hidden_indices.size == 0:
        raise ValueError("Demo request must leave at least one hidden benchmark to predict.")
    hidden_names = [score_matrix.benchmark_names[idx] for idx in hidden_indices.tolist()]
    hidden_truth = target_row[hidden_indices]

    train_mask = np.ones(len(score_matrix.model_names), dtype=bool)
    train_mask[target_idx] = False
    train_matrix = score_matrix.values[train_mask]

    observed_fraction = len(observed_indices) / max(int(np.sum(~np.isnan(target_row))), 1)
    if not 0.0 < observed_fraction < 1.0:
        raise ValueError("Demo request must reveal a strict subset of the available benchmarks.")
    ensemble = train_ensemble(
        matrix=train_matrix,
        ensemble_size=args.ensemble_size,
        rank=args.rank,
        epochs=args.epochs,
        seed=args.seed,
    )

    failure_model, failure_threshold, conformal_q, feature_reference = _fit_demo_calibrators(
        train_matrix=train_matrix,
        ensemble=ensemble,
        observed_fraction=observed_fraction,
        alpha=args.alpha,
        seed=args.seed,
    )

    prediction = predict_new_model(
        ensemble=ensemble,
        observed_indices=observed_indices,
        observed_values=observed_values,
        ridge=1e-2,
    )
    feature_vector = _build_feature_vector(
        ensemble=ensemble,
        observed_indices=observed_indices,
        observed_values=observed_values,
        hidden_indices=hidden_indices,
        prediction_mean=prediction.mean_prediction,
        prediction_variance=prediction.predictive_variance,
        mahalanobis_distances=prediction.mahalanobis_distances,
        ridge=1e-2,
    )

    if failure_model is None:
        risk_probability = float("nan")
        risk_label = "unknown"
    else:
        risk_probability = float(failure_model.predict_proba(feature_vector.reshape(1, -1))[0, 1])
        risk_label = _risk_label(risk_probability)

    risk_reasons = _rank_risk_reasons(feature_vector, feature_reference)
    recommended_index, recommendation_score = _recommend_next_benchmark(
        ensemble=ensemble,
        observed_indices=observed_indices,
        hidden_indices=hidden_indices,
        predictive_variance=prediction.predictive_variance,
    )
    decision, rationale = _stop_or_continue(
        risk_probability=risk_probability,
        feature_vector=feature_vector,
        feature_reference=feature_reference,
    )

    prediction_rows = []
    for hidden_name, hidden_idx, truth in zip(hidden_names, hidden_indices.tolist(), hidden_truth.tolist()):
        pred_value = float(prediction.mean_prediction[hidden_idx])
        lower = pred_value - conformal_q
        upper = pred_value + conformal_q
        prediction_rows.append(
            {
                "benchmark": hidden_name,
                "predicted_score": pred_value,
                "interval_lower": lower,
                "interval_upper": upper,
                "true_score": float(truth),
                "abs_error": abs(pred_value - float(truth)),
            }
        )

    report = {
        "model_name": request["model_name"],
        "model_family": infer_model_family(request["model_name"]),
        "observed_scores": observed_scores,
        "hidden_predictions": prediction_rows,
        "risk_probability": risk_probability,
        "risk_label": risk_label,
        "failure_threshold_mae": failure_threshold,
        "conformal_quantile": conformal_q,
        "risk_reasons": risk_reasons,
        "recommended_next_benchmark": {
            "benchmark": score_matrix.benchmark_names[recommended_index],
            "score": recommendation_score,
        },
        "decision": decision,
        "decision_rationale": rationale,
    }

    _print_report(report, show_all=args.show_all)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {args.output_json}")


def _load_score_matrix(args: argparse.Namespace) -> ScoreMatrix:
    if args.csv is not None:
        return load_score_csv(args.csv)
    if args.lm_eval_dir is not None:
        return load_lm_eval_results_dir(args.lm_eval_dir)
    return load_score_csv(DATASET)


def _resolve_request(args: argparse.Namespace, score_matrix: ScoreMatrix) -> dict[str, object]:
    if args.model_name is not None:
        return _sample_request_from_model(
            score_matrix=score_matrix,
            model_name=args.model_name,
            observed_fraction=args.observed_fraction,
            seed=args.seed,
        )

    request_path = args.request or DEFAULT_REQUEST
    return _load_request(request_path)


def _load_request(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Demo request must be a JSON object.")
    model_name = payload.get("model_name")
    observed_scores = payload.get("observed_scores")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Demo request must include a non-empty model_name.")
    if not isinstance(observed_scores, dict) or not observed_scores:
        raise ValueError("Demo request must include observed_scores.")
    return {"model_name": model_name, "observed_scores": observed_scores}


def _sample_request_from_model(
    score_matrix: ScoreMatrix,
    model_name: str,
    observed_fraction: float,
    seed: int,
) -> dict[str, object]:
    if not 0.0 < observed_fraction < 1.0:
        raise ValueError("observed_fraction must be in (0, 1).")

    try:
        target_idx = score_matrix.model_names.index(model_name)
    except ValueError as exc:
        raise ValueError(f"Unknown model for demo sampling: {model_name}") from exc

    target_row = score_matrix.values[target_idx]
    observed_full = np.flatnonzero(~np.isnan(target_row))
    if observed_full.size < 3:
        raise ValueError("Auto-sampled demo requires at least 3 observed benchmarks for the target model.")

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


def _fit_demo_calibrators(
    train_matrix: np.ndarray,
    ensemble: EnsembleMF,
    observed_fraction: float,
    alpha: float,
    seed: int,
) -> tuple[LogisticRegression | None, float, float, np.ndarray]:
    episodes = simulate_new_model_episodes(
        matrix=train_matrix,
        episodes_per_model=3,
        observed_fraction=observed_fraction,
        seed=seed + 1,
    )
    if not episodes:
        raise ValueError("Could not build calibration episodes for the demo request.")
    episode_results = [_evaluate_episode(ep, ensemble, ridge=1e-2) for ep in episodes]
    hidden_maes = np.asarray([r.hidden_mae for r in episode_results], dtype=float)
    failure_threshold = float(_quantile_higher(hidden_maes, 0.8))
    failure_labels = (hidden_maes >= failure_threshold).astype(int)
    feature_matrix = np.vstack([r.features for r in episode_results])

    failure_model: LogisticRegression | None = None
    if np.unique(failure_labels).size >= 2:
        failure_model = LogisticRegression(max_iter=2000, random_state=seed + 17)
        failure_model.fit(feature_matrix, failure_labels)

    episode_groups = np.asarray([ep.model_index for ep in episodes], dtype=int)
    calibration_idx, _ = _grouped_calibration_test_indices(episode_groups, seed=seed + 29)
    calibration_residuals = np.concatenate([episode_results[idx].hidden_abs_residuals for idx in calibration_idx])
    conformal_q = split_conformal_quantile(calibration_residuals, alpha=alpha)
    return failure_model, failure_threshold, conformal_q, feature_matrix


def _build_feature_vector(
    ensemble: EnsembleMF,
    observed_indices: np.ndarray,
    observed_values: np.ndarray,
    hidden_indices: np.ndarray,
    prediction_mean: np.ndarray,
    prediction_variance: np.ndarray,
    mahalanobis_distances: np.ndarray,
    ridge: float,
) -> np.ndarray:
    observed_pred = prediction_mean[observed_indices]
    residual_energy, max_abs_residual = residual_features(observed_values, observed_pred)
    maha = float(np.mean(mahalanobis_distances))
    mean_var, max_var = summary_variance_features(prediction_variance, hidden_indices)

    loo_errors = np.mean(
        np.vstack(
            [
                loo_observed_residuals(
                    V=member.V,
                    observed_indices=observed_indices,
                    observed_values=observed_values,
                    ridge=ridge,
                )
                for member in ensemble.members
            ]
        ),
        axis=0,
    )
    loo_mean = float(np.mean(loo_errors))
    loo_max = float(np.max(loo_errors)) if loo_errors.size > 0 else 0.0

    geometry = [observation_coverage_features(member.V, observed_indices) for member in ensemble.members]
    min_sv = float(np.mean([values[0] for values in geometry]))
    cond_num = float(np.mean([values[1] for values in geometry]))
    return np.asarray(
        [residual_energy, max_abs_residual, maha, mean_var, max_var, loo_mean, loo_max, min_sv, cond_num],
        dtype=float,
    )


def _rank_risk_reasons(feature_vector: np.ndarray, reference_matrix: np.ndarray) -> list[dict[str, float | str]]:
    category_map = {
        "residual mismatch": [0, 1, 5, 6],
        "latent shift": [2],
        "predictive variance": [3, 4],
        "observation geometry": [7, 8],
    }

    reasons = []
    means = np.mean(reference_matrix, axis=0)
    stds = np.std(reference_matrix, axis=0)
    stds = np.where(stds <= 1e-8, 1.0, stds)

    for category, indices in category_map.items():
        badness_scores = []
        for idx in indices:
            if idx == 7:
                badness = (means[idx] - feature_vector[idx]) / stds[idx]
            else:
                badness = (feature_vector[idx] - means[idx]) / stds[idx]
            badness_scores.append(max(float(badness), 0.0))
        reasons.append({"category": category, "score": float(np.mean(badness_scores))})

    ranked = sorted(reasons, key=lambda row: row["score"], reverse=True)
    return [row for row in ranked if row["score"] > 0.15][:3]


def _recommend_next_benchmark(
    ensemble: EnsembleMF,
    observed_indices: np.ndarray,
    hidden_indices: np.ndarray,
    predictive_variance: np.ndarray,
) -> tuple[int, float]:
    if hidden_indices.size == 0:
        raise ValueError("At least one hidden benchmark is required for recommendation.")
    if hidden_indices.size == 1:
        idx = int(hidden_indices[0])
        return idx, 1.0

    variance_scores = predictive_variance[hidden_indices].astype(float)
    geometry_gains = []
    observed_set = np.asarray(observed_indices, dtype=int)
    for hidden_idx in hidden_indices.tolist():
        extended = np.sort(np.append(observed_set, hidden_idx))
        per_member_gain = []
        for member in ensemble.members:
            current_min_sv, current_cond = observation_coverage_features(member.V, observed_set)
            next_min_sv, next_cond = observation_coverage_features(member.V, extended)
            cond_gain = (current_cond - next_cond) / max(current_cond, 1.0)
            per_member_gain.append((next_min_sv - current_min_sv) + 0.05 * cond_gain)
        geometry_gains.append(float(np.mean(per_member_gain)))

    variance_norm = _normalize_scores(variance_scores)
    geometry_norm = _normalize_scores(np.asarray(geometry_gains, dtype=float))
    combined = 0.6 * variance_norm + 0.4 * geometry_norm
    best_local = int(np.argmax(combined))
    return int(hidden_indices[best_local]), float(combined[best_local])


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value <= 1e-10:
        return np.ones_like(values)
    return (values - min_value) / (max_value - min_value)


def _stop_or_continue(
    risk_probability: float,
    feature_vector: np.ndarray,
    feature_reference: np.ndarray,
) -> tuple[str, str]:
    if np.isnan(risk_probability):
        return "continue", "Risk model could not be calibrated cleanly, so the safe fallback is to run more benchmarks."

    mean_var_cap = float(np.quantile(feature_reference[:, 3], 0.75))
    min_sv_floor = float(np.quantile(feature_reference[:, 7], 0.25))

    if risk_probability > 0.35:
        return "continue", "Predicted failure risk is still above the demo stop threshold."
    if feature_vector[7] < min_sv_floor:
        return "continue", "Observed benchmarks do not yet span the latent space well enough."
    if feature_vector[3] > mean_var_cap:
        return "continue", "Predictive variance is still high on the hidden benchmarks."
    return "stop", "Risk, geometry, and predictive variance are all within the training regime."


def _risk_label(risk_probability: float) -> str:
    if np.isnan(risk_probability):
        return "unknown"
    if risk_probability <= 0.20:
        return "low"
    if risk_probability <= 0.45:
        return "medium"
    return "high"


def _print_report(report: dict[str, object], show_all: bool = False, print_limit: int = 10) -> None:
    print(f"Model:     {report['model_name']}")
    print(f"Family:    {report['model_family']}")
    print("Observed:")
    for benchmark, value in report["observed_scores"].items():
        print(f"  {benchmark:<14} {float(value):.4f}")

    print("\nPredicted hidden benchmarks:")
    hidden_predictions = list(report["hidden_predictions"])
    for idx, row in enumerate(hidden_predictions):
        if not show_all and idx >= print_limit:
            print(f"  ... and {len(hidden_predictions) - print_limit} more")
            break
        print(
            f"  {row['benchmark']:<14} pred={row['predicted_score']:.4f}  "
            f"interval=[{row['interval_lower']:.4f}, {row['interval_upper']:.4f}]  "
            f"true={row['true_score']:.4f}"
        )

    print(f"\nRisk:      {report['risk_label']} ({report['risk_probability']:.4f})")
    print("Reasons:")
    if report["risk_reasons"]:
        for reason in report["risk_reasons"]:
            print(f"  {reason['category']:<22} score={reason['score']:.3f}")
    else:
        print("  none")

    recommendation = report["recommended_next_benchmark"]
    print(
        f"\nNext benchmark: {recommendation['benchmark']} "
        f"(priority score {recommendation['score']:.3f})"
    )
    print(f"Decision:       {report['decision']}")
    print(f"Rationale:      {report['decision_rationale']}")


if __name__ == "__main__":
    main()
