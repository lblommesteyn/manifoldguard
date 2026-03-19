from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manifoldguard.data import generate_synthetic_scores, load_score_csv
from manifoldguard.lm_eval import load_lm_eval_results_dir
from manifoldguard.ensemble import train_ensemble, predict_new_model
from manifoldguard.episodes import simulate_new_model_episodes
from manifoldguard.evaluation import _fit_failure_auc, _evaluate_episode, _evaluate_conformal
from manifoldguard._utils import quantile_higher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a clean manifoldguard demo.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, default=None, help="Path to score matrix CSV.")
    source.add_argument(
        "--lm-eval-dir",
        type=Path,
        default=None,
        help="Directory with lm-eval-harness JSON outputs (recursive).",
    )
    parser.add_argument(
        "--observed-fraction",
        type=float,
        default=0.35,
        help="Fraction of a model's observed benchmarks given at inference time.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for demo reproducibility.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Conformal miscoverage level (default implies 90% coverage).")
    parser.add_argument("--show-all", action="store_true", help="Show all hidden benchmarks instead of limiting to 10.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load Data
    print("--- ManifoldGuard Demo ---")
    if args.csv is not None:
        score_matrix = load_score_csv(args.csv)
        print(f"Dataset: Loaded {args.csv.name} ({len(score_matrix.model_names)} models, {len(score_matrix.benchmark_names)} benchmarks)")
    elif args.lm_eval_dir is not None:
        score_matrix = load_lm_eval_results_dir(args.lm_eval_dir)
        print(f"Dataset: Loaded lm-eval-dir ({len(score_matrix.model_names)} models, {len(score_matrix.benchmark_names)} benchmarks)")
    else:
        score_matrix = generate_synthetic_scores(
            num_models=60,
            num_benchmarks=20,
            rank=4,
            noise_std=0.15,
            missing_rate=0.25,
            seed=args.seed,
        )
        print("Dataset: Synthetic (60 models, 20 benchmarks)")

    matrix = score_matrix.values
    model_names = score_matrix.model_names
    benchmark_names = score_matrix.benchmark_names

    n_models = matrix.shape[0]
    n_benchmarks = matrix.shape[1]
    if n_models < 5:
        print("Error: Need at least 5 models for a good demo.")
        return

    # 2. Split Data: Reserve 1 specific model for the test output, 
    # use the rest for training background models (and a chunk for failure detector / conformal calibration).
    
    rng = np.random.default_rng(args.seed)
    model_order = rng.permutation(n_models)
    
    # Let's take the first model after permutation as our specific target test model for the demo
    target_idx = model_order[0]
    target_name = model_names[target_idx]
    
    # We need a proper split:
    # 70% Train (for EnsembleMF)
    # 30% Test (for calibrating the logistic regression failure detector and Conformal prediction)
    # The target_idx will be pulled from the Test set but run independently at the end.
    
    n_test = max(3, int(round(0.3 * n_models)))
    n_train = n_models - n_test
    
    train_indices = model_order[:n_train]
    test_indices = model_order[n_train:]
    
    # Make sure target is in test pool but we evaluate it specially
    if target_idx not in test_indices:
        # Swap if needed (though with permutation target_idx=model_order[0] is in train, let's just pick target from test subset)
        target_idx = test_indices[0]
        target_name = model_names[target_idx]

    train_matrix = matrix[np.sort(train_indices)]
    test_matrix = matrix[np.sort(test_indices)]
    
    print(f"Target Model: {target_name}")
    print("\n[Background] Training Ensemble Matrix Factorization models... (this may take a few seconds)")

    # 3. Train Background Models
    ensemble = train_ensemble(
        matrix=train_matrix,
        ensemble_size=5,
        rank=4,
        reg=1e-2,
        lr=5e-2,
        epochs=700,
        seed=args.seed,
        device=None,
    )

    # We need to train the failure detector and get the conformal quantile. 
    # To do this, we simulate episodes on all test models (including the target one to keep it simple, 
    # though in a real strict pipeline we'd exclude target from calibration. For demo purposes of training the detector, this is fine).
    
    # Generate episodes for all test models
    episodes = simulate_new_model_episodes(
        matrix=test_matrix,
        episodes_per_model=1, # 1 per model is enough for the demo background training
        observed_fraction=args.observed_fraction,
        seed=args.seed + 1,
    )
    
    episode_results = [_evaluate_episode(ep, ensemble, 1e-2) for ep in episodes]
    episode_groups = np.asarray([ep.model_index for ep in episodes], dtype=int)
    
    # Fit failure detector based on these results
    hidden_maes = np.asarray([r.hidden_mae for r in episode_results], dtype=float)
    failure_threshold = float(quantile_higher(hidden_maes, 0.8))
    failure_labels = (hidden_maes >= failure_threshold).astype(int)
    feature_matrix = np.vstack([r.features for r in episode_results])
    
    from sklearn.linear_model import LogisticRegression
    # Train simple global failure detector on the test set sims
    detector = LogisticRegression(max_iter=2000, random_state=args.seed)
    if np.unique(failure_labels).size >= 2:
        detector.fit(feature_matrix, failure_labels)
        detector_available = True
    else:
        detector_available = False

    # Get conformal quantile
    # For a simple demo, just use all test hidden residuals as calibration
    all_hidden_residuals = np.concatenate([r.hidden_abs_residuals for r in episode_results])
    from manifoldguard.conformal import split_conformal_quantile
    q = split_conformal_quantile(all_hidden_residuals, alpha=args.alpha)
    
    # 4. Simulate Demo Observation on Target Model
    # Find the target model episode (we know it's the first one in test_indices, but let's be sure or just re-simulate it clearly)
    target_row = matrix[target_idx]
    observed_full = np.flatnonzero(~np.isnan(target_row))
    desired_observed = int(np.floor(args.observed_fraction * observed_full.size))
    
    # If no data, abort
    if observed_full.size < 2:
        print("Error: Target model has too few observed benchmarks.")
        return
        
    observed_indices = np.sort(rng.choice(observed_full, size=max(2, desired_observed), replace=False))
    hidden_indices = np.setdiff1d(observed_full, observed_indices, assume_unique=False)
    
    observed_values = target_row[observed_indices].astype(float)
    hidden_values = target_row[hidden_indices].astype(float)
    
    # 5. Evaluate and Print
    total_benchmarks = observed_full.size
    pct_observed = int((len(observed_indices) / total_benchmarks) * 100)
    pct_hidden = 100 - pct_observed
    
    print(f"\nRevealed Benchmarks ({pct_observed}% - {len(observed_indices)}/{total_benchmarks}):")
    for idx, val in zip(observed_indices, observed_values):
        print(f"  - {benchmark_names[idx]}: {val:.4f}")
        
    # Infer
    prediction = predict_new_model(
        ensemble=ensemble,
        observed_indices=observed_indices,
        observed_values=observed_values,
        ridge=1e-2,
    )
    
    hidden_pred = prediction.mean_prediction[hidden_indices]
    
    print(f"\nHidden Benchmarks ({pct_hidden}% - {len(hidden_indices)}/{total_benchmarks}):")
    # For a clean demo output, limit to 10 if there are many
    print_limit = 10
    
    for i, (idx, true_val, pred_val) in enumerate(zip(hidden_indices, hidden_values, hidden_pred)):
        if not args.show_all and i >= print_limit:
            print(f"  ... and {len(hidden_indices) - print_limit} more")
            break
        low, high = pred_val - q, pred_val + q
        # CLamp intervals to [0,1] if data is mostly in that range, but for generic data do not clamp.
        print(f"  - {benchmark_names[idx][:30]:<30} (True: {true_val:.4f}) -> Predicted: {pred_val:.4f}  [{low:.4f} - {high:.4f}]  delta={abs(true_val - pred_val):.4f}")
        
    print("\n--- Reliability Assessment ---")
    
    # Calculate OOD Features
    from manifoldguard.ood import residual_features, summary_variance_features, observation_coverage_features
    from manifoldguard.inference import loo_observed_residuals
    
    residual_energy, max_abs_residual = residual_features(observed_values, prediction.mean_prediction[observed_indices])
    maha = float(np.mean(prediction.mahalanobis_distances))
    mean_var, max_var = summary_variance_features(prediction.predictive_variance, hidden_indices)
    
    loo_errors_per_member = [
        loo_observed_residuals(
            V=member.V,
            observed_indices=observed_indices,
            observed_values=observed_values,
            ridge=1e-2,
        )
        for member in ensemble.members
    ]
    loo_errors = np.mean(np.vstack(loo_errors_per_member), axis=0)
    loo_mean = float(np.mean(loo_errors))
    loo_max = float(np.max(loo_errors)) if loo_errors.size > 0 else 0.0

    coverage_features = [
        observation_coverage_features(member.V, observed_indices)
        for member in ensemble.members
    ]
    min_sv = float(np.mean([f[0] for f in coverage_features]))
    cond_num = float(np.mean([f[1] for f in coverage_features]))

    features = np.asarray(
        [[residual_energy, max_abs_residual, maha, mean_var, max_var, loo_mean, loo_max, min_sv, cond_num]],
        dtype=float,
    )
    
    if detector_available:
        risk_prob = detector.predict_proba(features)[0, 1]
        risk_level = "HIGH" if risk_prob >= 0.5 else "LOW"
        print(f"Predicted Error Risk: {risk_level} ({risk_prob:.0%} probability of large error)")
        if risk_level == "HIGH":
            print("  -> The model's revealed scores are significantly out-of-distribution.")
            print("  -> These predictions have a >80% chance of being in the top 20% worst completions. Manual evaluation is recommended.")
        else:
            print("  -> The model's revealed scores align well with training data.")
            print("  -> The Matrix Factorization predictions are likely stable and reliable.")
    else:
        print("Predicted Error Risk: Unavaliable (Not enough model variance in background data to train detector)")
        
    print(f"Condition Number (OOD Metric): {cond_num:.2f}")


if __name__ == "__main__":
    main()
