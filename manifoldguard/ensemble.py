from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifoldguard.inference import infer_latent_u_ridge, predict_scores
from manifoldguard.mf import MFModel, train_matrix_factorization
from manifoldguard.ood import LatentDistribution, fit_latent_distribution, mahalanobis_distance


Array = np.ndarray


@dataclass(frozen=True)
class EnsembleMF:
    members: list[MFModel]
    latent_distributions: list[LatentDistribution]


@dataclass(frozen=True)
class NewModelPrediction:
    per_member_predictions: Array
    mean_prediction: Array
    predictive_variance: Array
    inferred_latents: Array
    mahalanobis_distances: Array


def train_ensemble(
    matrix: Array,
    ensemble_size: int,
    rank: int,
    reg: float = 1e-2,
    lr: float = 5e-2,
    epochs: int = 700,
    seed: int = 0,
    device: str | None = None,
) -> EnsembleMF:
    if ensemble_size <= 0:
        raise ValueError("ensemble_size must be positive.")

    members: list[MFModel] = []
    dists: list[LatentDistribution] = []
    for i in range(ensemble_size):
        member_seed = seed + i
        model = train_matrix_factorization(
            matrix=matrix,
            rank=rank,
            reg=reg,
            lr=lr,
            epochs=epochs,
            seed=member_seed,
            device=device,
        )
        members.append(model)
        dists.append(fit_latent_distribution(model.U))
    return EnsembleMF(members=members, latent_distributions=dists)


def predict_new_model(
    ensemble: EnsembleMF,
    observed_indices: Array,
    observed_values: Array,
    ridge: float = 1e-2,
) -> NewModelPrediction:
    if not ensemble.members:
        raise ValueError("ensemble is empty.")

    predictions = []
    latents = []
    distances = []
    for member, dist in zip(ensemble.members, ensemble.latent_distributions):
        u = infer_latent_u_ridge(
            V=member.V,
            observed_indices=observed_indices,
            observed_values=observed_values,
            ridge=ridge,
        )
        pred = predict_scores(u=u, V=member.V)

        predictions.append(pred)
        latents.append(u)
        distances.append(mahalanobis_distance(u, dist))

    pred_stack = np.vstack(predictions)
    latent_stack = np.vstack(latents)
    ddof = 1 if pred_stack.shape[0] > 1 else 0
    pred_var = np.var(pred_stack, axis=0, ddof=ddof)
    pred_mean = np.mean(pred_stack, axis=0)

    return NewModelPrediction(
        per_member_predictions=pred_stack,
        mean_prediction=pred_mean,
        predictive_variance=pred_var,
        inferred_latents=latent_stack,
        mahalanobis_distances=np.asarray(distances, dtype=float),
    )
