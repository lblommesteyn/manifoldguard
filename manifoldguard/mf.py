from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


Array = np.ndarray


@dataclass(frozen=True)
class MFModel:
    U: Array
    V: Array
    rank: int
    reg: float
    seed: int
    final_loss: float


def train_matrix_factorization(
    matrix: Array,
    rank: int,
    reg: float = 1e-2,
    lr: float = 5e-2,
    epochs: int = 700,
    seed: int = 0,
    device: str | None = None,
) -> MFModel:
    """Train low-rank MF with squared-error over observed entries only."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if rank <= 0:
        raise ValueError("rank must be positive.")

    observed = ~np.isnan(matrix)
    if observed.sum() == 0:
        raise ValueError("matrix has no observed entries.")

    n_models, n_benchmarks = matrix.shape
    device_name = _resolve_device(device)
    torch.manual_seed(seed)

    x = torch.tensor(np.nan_to_num(matrix, nan=0.0), dtype=torch.float32, device=device_name)
    mask = torch.tensor(observed.astype(np.float32), dtype=torch.float32, device=device_name)

    u = torch.nn.Parameter(torch.randn(n_models, rank, device=device_name) * 0.1)
    v = torch.nn.Parameter(torch.randn(n_benchmarks, rank, device=device_name) * 0.1)
    optimizer = torch.optim.Adam([u, v], lr=lr)

    denom = mask.sum().clamp(min=1.0)
    final_loss = float("nan")
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)

        pred = u @ v.T
        residual = (pred - x) * mask
        data_loss = residual.pow(2).sum() / denom
        reg_loss = reg * (u.pow(2).mean() + v.pow(2).mean())
        loss = data_loss + reg_loss

        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    return MFModel(
        U=u.detach().cpu().numpy(),
        V=v.detach().cpu().numpy(),
        rank=rank,
        reg=reg,
        seed=seed,
        final_loss=final_loss,
    )


def reconstruct(model: MFModel) -> Array:
    return model.U @ model.V.T


def _resolve_device(device: str | None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device
