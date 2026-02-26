from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from manifoldguard.data import ScoreMatrix


Array = np.ndarray
DEFAULT_METRIC_PREFERENCE = (
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "f1,none",
    "mc2,none",
    "mc1,none",
)


def load_lm_eval_results_dir(path: str | Path) -> ScoreMatrix:
    """
    Load a model x task score matrix from lm-eval-harness JSON outputs.

    The directory can contain nested model folders. Each JSON file with a
    top-level "results" object is treated as one model run.
    """
    root = Path(path)
    if not root.exists():
        raise ValueError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Path must be a directory: {root}")

    run_rows: list[dict[str, Any]] = []
    for json_path in sorted(root.rglob("*.json")):
        payload = _read_json(json_path)
        if payload is None or not isinstance(payload, dict):
            continue
        results = payload.get("results")
        if not isinstance(results, dict) or not results:
            continue

        model_name = _infer_model_name(payload, json_path)
        scores: dict[str, float] = {}
        for task_name, metrics in results.items():
            if not isinstance(task_name, str) or not isinstance(metrics, dict):
                continue
            value = _select_metric_value(metrics)
            if value is None:
                continue
            scores[task_name] = value

        if scores:
            run_rows.append({"model_name": model_name, "scores": scores})

    if not run_rows:
        raise ValueError(f"No lm-eval result JSON files found under: {root}")

    benchmark_names = sorted({bench for row in run_rows for bench in row["scores"].keys()})
    values = np.full((len(run_rows), len(benchmark_names)), np.nan, dtype=float)
    model_names: list[str] = []
    bench_to_idx = {name: idx for idx, name in enumerate(benchmark_names)}

    for row_idx, row in enumerate(run_rows):
        model_names.append(str(row["model_name"]))
        for bench, score in row["scores"].items():
            values[row_idx, bench_to_idx[bench]] = float(score)

    return ScoreMatrix(values=values, model_names=model_names, benchmark_names=benchmark_names)


def _read_json(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _infer_model_name(payload: dict[str, Any], json_path: Path) -> str:
    if isinstance(payload.get("model_name"), str) and payload["model_name"].strip():
        return payload["model_name"].strip()

    config = payload.get("config")
    if isinstance(config, dict):
        model = config.get("model")
        model_args = config.get("model_args")
        if isinstance(model, str) and model.strip():
            if isinstance(model_args, str) and model_args.strip():
                return f"{model}:{model_args}"
            return model.strip()

    stem = json_path.stem.lower()
    if stem in {"results", "result", "eval_results"} and json_path.parent.name:
        return json_path.parent.name
    return json_path.stem


def _select_metric_value(metrics: dict[str, Any]) -> float | None:
    for key in DEFAULT_METRIC_PREFERENCE:
        candidate = metrics.get(key)
        if _is_numeric(candidate):
            return float(candidate)

    for key, value in metrics.items():
        if _looks_like_stderr_key(key):
            continue
        if _is_numeric(value):
            return float(value)

    return None


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def _looks_like_stderr_key(key: str) -> bool:
    lowered = key.lower()
    return "stderr" in lowered or lowered.endswith("_se") or lowered.endswith("_std")
