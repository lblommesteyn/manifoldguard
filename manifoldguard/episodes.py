from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Episode:
    model_index: int
    observed_indices: Array
    hidden_indices: Array
    observed_values: Array
    hidden_values: Array


def simulate_new_model_episodes(
    matrix: Array,
    episodes_per_model: int = 3,
    observed_fraction: float = 0.5,
    min_observed: int = 2,
    min_hidden: int = 1,
    seed: int = 0,
) -> list[Episode]:
    """
    Simulate new-model episodes by selecting observed subset S and hiding the rest.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if episodes_per_model <= 0:
        raise ValueError("episodes_per_model must be positive.")
    if not 0.0 < observed_fraction < 1.0:
        raise ValueError("observed_fraction must be in (0, 1).")

    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []

    for model_idx in range(matrix.shape[0]):
        row = matrix[model_idx]
        observed_full = np.flatnonzero(~np.isnan(row))
        if observed_full.size < (min_observed + min_hidden):
            continue

        max_observed = observed_full.size - min_hidden
        desired_observed = int(np.floor(observed_fraction * observed_full.size))
        observed_count = int(np.clip(desired_observed, min_observed, max_observed))
        if observed_count < min_observed:
            continue

        for _ in range(episodes_per_model):
            observed_indices = np.sort(rng.choice(observed_full, size=observed_count, replace=False))
            hidden_indices = np.setdiff1d(observed_full, observed_indices, assume_unique=False)

            if hidden_indices.size < min_hidden:
                continue

            episodes.append(
                Episode(
                    model_index=model_idx,
                    observed_indices=observed_indices,
                    hidden_indices=hidden_indices,
                    observed_values=row[observed_indices].astype(float),
                    hidden_values=row[hidden_indices].astype(float),
                )
            )

    return episodes
