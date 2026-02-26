import numpy as np
import pytest

from manifoldguard.mf import MFModel, reconstruct, train_matrix_factorization


def _low_rank_matrix(n: int = 12, m: int = 8, rank: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, rank))
    V = rng.normal(size=(m, rank))
    return U @ V.T


def test_train_returns_mfmodel() -> None:
    mat = _low_rank_matrix()
    model = train_matrix_factorization(mat, rank=2, epochs=5, seed=0)
    assert isinstance(model, MFModel)
    assert model.U.shape == (12, 2)
    assert model.V.shape == (8, 2)


def test_train_loss_decreases() -> None:
    """Final loss after 500 epochs should be lower than after 5 epochs."""
    mat = _low_rank_matrix()
    short = train_matrix_factorization(mat, rank=2, epochs=5, seed=42)
    long = train_matrix_factorization(mat, rank=2, epochs=500, seed=42)
    assert long.final_loss < short.final_loss


def test_reconstruct_shape() -> None:
    mat = _low_rank_matrix()
    model = train_matrix_factorization(mat, rank=2, epochs=50, seed=0)
    recon = reconstruct(model)
    assert recon.shape == mat.shape


def test_train_with_nans() -> None:
    mat = _low_rank_matrix()
    mat_nan = mat.copy()
    mat_nan[0, 0] = np.nan
    mat_nan[3, 5] = np.nan
    model = train_matrix_factorization(mat_nan, rank=2, epochs=50, seed=0)
    assert model.U.shape[0] == mat.shape[0]


def test_train_raises_on_all_nan() -> None:
    mat = np.full((4, 4), np.nan)
    with pytest.raises(ValueError, match="no observed entries"):
        train_matrix_factorization(mat, rank=2)


def test_train_raises_on_bad_rank() -> None:
    mat = _low_rank_matrix()
    with pytest.raises(ValueError, match="rank must be positive"):
        train_matrix_factorization(mat, rank=0)


def test_different_seeds_give_different_models() -> None:
    mat = _low_rank_matrix()
    m0 = train_matrix_factorization(mat, rank=2, epochs=10, seed=0)
    m1 = train_matrix_factorization(mat, rank=2, epochs=10, seed=1)
    assert not np.allclose(m0.U, m1.U)
