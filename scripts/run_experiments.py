from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manifoldguard.data import generate_synthetic_scores, load_score_csv
from manifoldguard.evaluation import evaluate_experiment
from manifoldguard.lm_eval import load_lm_eval_results_dir
from scripts.plot_calibration import generate_calibration_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manifoldguard experiments.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, default=None, help="Path to score matrix CSV.")
    source.add_argument(
        "--lm-eval-dir",
        type=Path,
        default=None,
        help="Directory with lm-eval-harness JSON outputs (recursive).",
    )
    parser.add_argument("--rank", type=int, default=4, help="MF latent rank.")
    parser.add_argument("--ensemble-size", type=int, default=5, help="Number of MF models in ensemble.")
    parser.add_argument("--epochs", type=int, default=700, help="Training epochs per MF model.")
    parser.add_argument("--lr", type=float, default=5e-2, help="Learning rate for MF training.")
    parser.add_argument("--mf-reg", type=float, default=1e-2, help="L2 regularization for MF training.")
    parser.add_argument("--ridge", type=float, default=1e-2, help="Ridge penalty for new-model inference.")
    parser.add_argument("--episodes-per-model", type=int, default=5, help="Simulation episodes per model.")
    parser.add_argument(
        "--observed-fraction",
        type=float,
        default=0.35,
        help="Fraction of a model's observed benchmarks given at inference time.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20],
        help="List of conformal miscoverage levels.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--model-test-fraction",
        type=float,
        default=0.25,
        help="Fraction of models held out from MF training for evaluation (no leakage).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device ('cpu', 'cuda', 'mps', ...). Defaults to auto-detect.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write metrics JSON.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path to write metrics CSV.")

    parser.add_argument("--synthetic-models", type=int, default=80, help="Synthetic data: number of models.")
    parser.add_argument(
        "--synthetic-benchmarks",
        type=int,
        default=20,
        help="Synthetic data: number of benchmarks.",
    )
    parser.add_argument("--synthetic-rank", type=int, default=4, help="Synthetic data: latent rank.")
    parser.add_argument("--synthetic-noise", type=float, default=0.15, help="Synthetic data: noise std.")
    parser.add_argument(
        "--synthetic-missing-rate",
        type=float,
        default=0.25,
        help="Synthetic data: missing entry fraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.csv is not None:
        score_matrix = load_score_csv(args.csv)
    elif args.lm_eval_dir is not None:
        score_matrix = load_lm_eval_results_dir(args.lm_eval_dir)
    else:
        score_matrix = generate_synthetic_scores(
            num_models=args.synthetic_models,
            num_benchmarks=args.synthetic_benchmarks,
            rank=args.synthetic_rank,
            noise_std=args.synthetic_noise,
            missing_rate=args.synthetic_missing_rate,
            seed=args.seed,
        )

    metrics = evaluate_experiment(
        matrix=score_matrix.values,
        rank=args.rank,
        ensemble_size=args.ensemble_size,
        mf_reg=args.mf_reg,
        ridge=args.ridge,
        lr=args.lr,
        epochs=args.epochs,
        episodes_per_model=args.episodes_per_model,
        observed_fraction=args.observed_fraction,
        model_test_fraction=args.model_test_fraction,
        seed=args.seed,
        alpha=args.alphas,
        device=args.device,
    )

    metrics_row = {
        "train_models": metrics.num_train_models,
        "test_models": metrics.num_test_models,
        "episodes": metrics.num_episodes,
        "completion_mae": metrics.completion_mae,
        "failure_auc": metrics.failure_auc,
        "conformal_coverage": metrics.conformal_coverage,
        "conformal_quantile": metrics.conformal_quantile,
    }

    print(f"train models:       {metrics.num_train_models}")
    print(f"test models:        {metrics.num_test_models}")
    print(f"episodes:           {metrics.num_episodes}")
    print(f"completion MAE:     {metrics.completion_mae:.4f}")
    print(f"failure AUC:        {metrics.failure_auc:.4f}")
    
    print("\nConformal Coverage Results:")
    print(f"{'alpha':^7} | {'target (1-a)':^14} | {'empirical':^11} | {'difference':^10}")
    print("-" * 52)

    coverages = (
        metrics.conformal_coverage
        if isinstance(metrics.conformal_coverage, dict)
        else {args.alphas[0]: metrics.conformal_coverage}
    )
    for a in args.alphas:
        cov = coverages[a]
        target = 1.0 - a
        diff = cov - target
        print(f"{a:^7.2f} | {target:^14.2f} | {cov:^11.4f} | {diff:^+10.4f}")

    if len(metrics.failure_oof_probs) > 0 and len(metrics.failure_oof_labels) > 0:
        generate_calibration_plot(metrics.failure_oof_probs, metrics.failure_oof_labels)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics_row, indent=2), encoding="utf-8")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics_row))
            writer.writeheader()
            writer.writerow(metrics_row)

if __name__ == "__main__":
    main()
