"""manifoldguard package."""

from manifoldguard.conformal import empirical_coverage, interval_bounds, split_conformal_quantile
from manifoldguard.data import ScoreMatrix, generate_synthetic_scores, load_score_csv, observed_mask
from manifoldguard.ensemble import EnsembleMF, NewModelPrediction, predict_new_model, train_ensemble
from manifoldguard.episodes import Episode, simulate_new_model_episodes
from manifoldguard.evaluation import EpisodeResult, ExperimentMetrics, evaluate_experiment
from manifoldguard.inference import infer_latent_u_ridge, loo_observed_residuals, predict_scores
from manifoldguard.lm_eval import load_lm_eval_results_dir
from manifoldguard.mf import MFModel, reconstruct, train_matrix_factorization
from manifoldguard.ood import (
    LatentDistribution,
    fit_latent_distribution,
    mahalanobis_distance,
    observation_coverage_features,
    residual_features,
    summary_variance_features,
)

__all__ = [
    "EnsembleMF",
    "Episode",
    "EpisodeResult",
    "ExperimentMetrics",
    "LatentDistribution",
    "MFModel",
    "NewModelPrediction",
    "ScoreMatrix",
    "empirical_coverage",
    "evaluate_experiment",
    "fit_latent_distribution",
    "generate_synthetic_scores",
    "infer_latent_u_ridge",
    "interval_bounds",
    "load_score_csv",
    "load_lm_eval_results_dir",
    "loo_observed_residuals",
    "mahalanobis_distance",
    "observation_coverage_features",
    "observed_mask",
    "predict_new_model",
    "predict_scores",
    "reconstruct",
    "residual_features",
    "simulate_new_model_episodes",
    "split_conformal_quantile",
    "summary_variance_features",
    "train_ensemble",
    "train_matrix_factorization",
]
