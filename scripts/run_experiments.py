from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manifoldguard.data import generate_synthetic_scores, load_score_csv
from manifoldguard.evaluation import evaluate_experiment
from manifoldguard.lm_eval import load_lm_eval_results_dir


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
    parser.add_argument("--alpha", type=float, default=0.1, help="Conformal miscoverage level.")
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
        alpha=args.alpha,
        device=args.device,
    )

    print(f"train models:       {metrics.num_train_models}")
    print(f"test models:        {metrics.num_test_models}")
    print(f"episodes:           {metrics.num_episodes}")
    print(f"completion MAE:     {metrics.completion_mae:.4f}")
    print(f"failure AUC:        {metrics.failure_auc:.4f}")
    print(f"conformal coverage: {metrics.conformal_coverage:.4f}")
    print(f"conformal quantile: {metrics.conformal_quantile:.4f}")


if __name__ == "__main__":
    main()
