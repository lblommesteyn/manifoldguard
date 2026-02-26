from __future__ import annotations

import json

import numpy as np
import pytest

from manifoldguard.lm_eval import load_lm_eval_results_dir


def test_load_lm_eval_results_dir_basic(tmp_path) -> None:
    model_a_dir = tmp_path / "model-a"
    model_a_dir.mkdir()
    with (model_a_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {"model": "hf", "model_args": "pretrained=model-a"},
                "results": {
                    "arc_easy": {"acc,none": 0.7, "acc_stderr,none": 0.01},
                    "hellaswag": {"acc_norm,none": 0.45, "acc,none": 0.4},
                },
            },
            handle,
        )

    model_b_dir = tmp_path / "model-b"
    model_b_dir.mkdir()
    with (model_b_dir / "eval_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": "custom-b",
                "results": {
                    "arc_easy": {"acc,none": 0.66},
                    "truthfulqa": {"mc2,none": 0.39},
                },
            },
            handle,
        )

    matrix = load_lm_eval_results_dir(tmp_path)

    assert matrix.values.shape == (2, 3)
    assert matrix.model_names == ["hf:pretrained=model-a", "custom-b"]
    assert matrix.benchmark_names == ["arc_easy", "hellaswag", "truthfulqa"]
    assert matrix.values[0, 0] == pytest.approx(0.7)
    assert matrix.values[0, 1] == pytest.approx(0.45)
    assert np.isnan(matrix.values[0, 2])
    assert matrix.values[1, 0] == pytest.approx(0.66)
    assert np.isnan(matrix.values[1, 1])
    assert matrix.values[1, 2] == pytest.approx(0.39)


def test_load_lm_eval_results_uses_numeric_fallback(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "results": {
                    "custom_task": {
                        "custom_score": 0.12,
                        "custom_score_stderr": 0.01,
                    }
                }
            },
            handle,
        )

    matrix = load_lm_eval_results_dir(tmp_path)
    assert matrix.values.shape == (1, 1)
    assert matrix.values[0, 0] == pytest.approx(0.12)


def test_load_lm_eval_results_dir_raises_when_missing(tmp_path) -> None:
    with (tmp_path / "ignore.json").open("w", encoding="utf-8") as handle:
        json.dump({"not_results": {}}, handle)

    with pytest.raises(ValueError):
        load_lm_eval_results_dir(tmp_path)
