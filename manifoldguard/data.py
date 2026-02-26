from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


Array = np.ndarray
_MISSING_TOKENS = {"", "nan", "na", "null", "none"}


@dataclass(frozen=True)
class ScoreMatrix:
    values: Array
    model_names: list[str]
    benchmark_names: list[str]


def load_score_csv(path: str | Path) -> ScoreMatrix:
    """Load a model x benchmark score matrix from CSV with NaN support."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]

    if not rows:
        raise ValueError(f"CSV is empty: {file_path}")

    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]

    has_header = any(not _is_float_like(cell) for cell in normalized_rows[0])
    if has_header:
        header = normalized_rows[0]
        data_rows = normalized_rows[1:]
    else:
        header = []
        data_rows = normalized_rows

    if not data_rows:
        raise ValueError("CSV contains header only and no data rows.")

    has_model_names = any(not _is_float_like(row[0]) for row in data_rows)
    if has_model_names:
        model_names = [row[0].strip() or f"model_{idx}" for idx, row in enumerate(data_rows)]
        data_cells = [row[1:] for row in data_rows]
    else:
        model_names = [f"model_{idx}" for idx in range(len(data_rows))]
        data_cells = data_rows

    values = np.asarray([[_parse_float(cell) for cell in row] for row in data_cells], dtype=float)
    benchmark_names = _resolve_benchmark_names(
        header=header,
        has_model_names=has_model_names,
        n_benchmarks=values.shape[1],
    )
    return ScoreMatrix(values=values, model_names=model_names, benchmark_names=benchmark_names)


def observed_mask(values: Array) -> Array:
    return ~np.isnan(values)


def generate_synthetic_scores(
    num_models: int = 48,
    num_benchmarks: int = 20,
    rank: int = 4,
    noise_std: float = 0.08,
    missing_rate: float = 0.25,
    seed: int = 0,
) -> ScoreMatrix:
    """Generate a synthetic low-rank score matrix with missing values."""
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1).")

    rng = np.random.default_rng(seed)
    u = rng.normal(loc=0.0, scale=1.0, size=(num_models, rank))
    v = rng.normal(loc=0.0, scale=1.0, size=(num_benchmarks, rank))
    values = (u @ v.T) + rng.normal(loc=0.0, scale=noise_std, size=(num_models, num_benchmarks))

    missing_mask = rng.random(size=values.shape) < missing_rate
    values = values.astype(float)
    values[missing_mask] = np.nan

    model_names = [f"model_{idx}" for idx in range(num_models)]
    benchmark_names = [f"benchmark_{idx}" for idx in range(num_benchmarks)]
    return ScoreMatrix(values=values, model_names=model_names, benchmark_names=benchmark_names)


def _is_float_like(cell: str) -> bool:
    text = cell.strip().lower()
    if text in _MISSING_TOKENS:
        return True
    try:
        float(text)
        return True
    except ValueError:
        return False


def _parse_float(cell: str) -> float:
    text = cell.strip().lower()
    if text in _MISSING_TOKENS:
        return float("nan")
    return float(text)


def _resolve_benchmark_names(header: Iterable[str], has_model_names: bool, n_benchmarks: int) -> list[str]:
    header_list = list(header)
    candidate = header_list[1:] if has_model_names else header_list
    cleaned = [name.strip() for name in candidate if name.strip()]

    if not cleaned:
        return [f"benchmark_{idx}" for idx in range(n_benchmarks)]

    if len(cleaned) < n_benchmarks:
        extension = [f"benchmark_{idx}" for idx in range(len(cleaned), n_benchmarks)]
        return cleaned + extension

    return cleaned[:n_benchmarks]
