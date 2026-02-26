import io
import textwrap
from pathlib import Path

import numpy as np
import pytest

from manifoldguard.data import (
    ScoreMatrix,
    generate_synthetic_scores,
    load_score_csv,
    observed_mask,
)


# ---------------------------------------------------------------------------
# load_score_csv
# ---------------------------------------------------------------------------

def _write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "scores.csv"
    p.write_text(textwrap.dedent(content).strip())
    return p


def test_load_csv_with_header_and_model_names(tmp_path: Path) -> None:
    p = _write_csv(tmp_path, """\
        model,bench_A,bench_B,bench_C
        gpt4,0.9,0.8,0.7
        llama,0.6,,0.5
    """)
    sm = load_score_csv(p)
    assert sm.model_names == ["gpt4", "llama"]
    assert sm.benchmark_names == ["bench_A", "bench_B", "bench_C"]
    assert sm.values.shape == (2, 3)
    assert np.isnan(sm.values[1, 1])


def test_load_csv_no_header_no_model_names(tmp_path: Path) -> None:
    p = _write_csv(tmp_path, """\
        0.1,0.2,0.3
        0.4,0.5,nan
    """)
    sm = load_score_csv(p)
    assert sm.model_names == ["model_0", "model_1"]
    assert sm.benchmark_names == ["benchmark_0", "benchmark_1", "benchmark_2"]
    assert np.isnan(sm.values[1, 2])


def test_load_csv_empty_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_score_csv(p)


def test_load_csv_header_only_raises(tmp_path: Path) -> None:
    p = _write_csv(tmp_path, "model,bench_A,bench_B\n")
    with pytest.raises(ValueError, match="header only"):
        load_score_csv(p)


# ---------------------------------------------------------------------------
# generate_synthetic_scores
# ---------------------------------------------------------------------------

def test_generate_synthetic_scores_shape() -> None:
    sm = generate_synthetic_scores(num_models=10, num_benchmarks=6, seed=0)
    assert sm.values.shape == (10, 6)


def test_generate_synthetic_scores_missing_rate() -> None:
    sm = generate_synthetic_scores(num_models=100, num_benchmarks=20, missing_rate=0.3, seed=1)
    actual_rate = np.mean(np.isnan(sm.values))
    assert abs(actual_rate - 0.3) < 0.05


def test_generate_synthetic_scores_no_missing() -> None:
    sm = generate_synthetic_scores(missing_rate=0.0, seed=0)
    assert not np.any(np.isnan(sm.values))


def test_generate_synthetic_scores_bad_missing_rate() -> None:
    with pytest.raises(ValueError, match="missing_rate"):
        generate_synthetic_scores(missing_rate=1.0)


# ---------------------------------------------------------------------------
# observed_mask
# ---------------------------------------------------------------------------

def test_observed_mask() -> None:
    vals = np.array([[1.0, np.nan], [np.nan, 2.0]])
    mask = observed_mask(vals)
    expected = np.array([[True, False], [False, True]])
    assert np.array_equal(mask, expected)
