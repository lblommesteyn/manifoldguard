import numpy as np
import pytest

from manifoldguard.data import generate_synthetic_scores
from manifoldguard.episodes import Episode, simulate_new_model_episodes


def _full_matrix(n: int = 10, m: int = 8, seed: int = 0) -> np.ndarray:
    """Fully observed matrix (no NaNs)."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, m))


def test_returns_list_of_episodes() -> None:
    mat = _full_matrix()
    episodes = simulate_new_model_episodes(mat, episodes_per_model=2, seed=0)
    assert all(isinstance(ep, Episode) for ep in episodes)


def test_each_episode_has_disjoint_observed_and_hidden() -> None:
    mat = _full_matrix()
    for ep in simulate_new_model_episodes(mat, episodes_per_model=1, seed=7):
        assert len(np.intersect1d(ep.observed_indices, ep.hidden_indices)) == 0


def test_each_episode_covers_all_row_indices() -> None:
    mat = _full_matrix()
    for ep in simulate_new_model_episodes(mat, episodes_per_model=1, seed=3):
        row = mat[ep.model_index]
        all_observed = np.flatnonzero(~np.isnan(row))
        union = np.union1d(ep.observed_indices, ep.hidden_indices)
        assert np.array_equal(np.sort(union), np.sort(all_observed))


def test_observed_values_match_matrix() -> None:
    mat = _full_matrix()
    for ep in simulate_new_model_episodes(mat, episodes_per_model=1, seed=5):
        assert np.allclose(ep.observed_values, mat[ep.model_index, ep.observed_indices])
        assert np.allclose(ep.hidden_values, mat[ep.model_index, ep.hidden_indices])


def test_min_hidden_respected() -> None:
    mat = _full_matrix()
    for ep in simulate_new_model_episodes(mat, min_hidden=2, seed=0):
        assert len(ep.hidden_indices) >= 2


def test_episodes_per_model_count() -> None:
    mat = _full_matrix(n=5, m=10)
    episodes = simulate_new_model_episodes(mat, episodes_per_model=3, seed=0)
    # At most 3 episodes per model (exact count when all models have enough observations)
    counts = {}
    for ep in episodes:
        counts[ep.model_index] = counts.get(ep.model_index, 0) + 1
    assert all(c <= 3 for c in counts.values())


def test_empty_matrix_raises() -> None:
    with pytest.raises(ValueError, match="2D"):
        simulate_new_model_episodes(np.array([1.0, 2.0]))


def test_no_valid_episodes_with_tiny_matrix() -> None:
    """A 2x2 all-observed matrix with observed_fraction=0.9 forces hidden<min_hidden."""
    mat = np.ones((2, 2))
    # Each row has 2 observed; with min_observed=2 and min_hidden=1, observed_count
    # clips to max_observed = 2-1 = 1, but min_observed=2 > 1, so no valid episodes.
    episodes = simulate_new_model_episodes(
        mat, episodes_per_model=1, min_observed=2, min_hidden=1, observed_fraction=0.9
    )
    assert episodes == []
