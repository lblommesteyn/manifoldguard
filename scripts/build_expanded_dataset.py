"""Fetch Open LLM Leaderboard v2 results and build scores_expanded.csv.

Requires: pip install huggingface_hub
This is NOT a runtime dependency - run once to regenerate the dataset.

Usage:
    python scripts/build_expanded_dataset.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manifoldguard.leaderboard_dataset import (
    CORE_BENCHMARKS,
    CURATED_MODEL_IDS,
    discover_expanded_benchmarks,
    expanded_column_names,
    extract_benchmark_scores,
)

REPO_ID = "open-llm-leaderboard/results"
REPO_TYPE = "dataset"
OUT_DIR = REPO_ROOT / "datasets" / "lm_eval_real"


def main() -> None:
    api = HfApi()
    repo_files = set(api.list_repo_files(REPO_ID, repo_type=REPO_TYPE))

    payload_rows: list[tuple[str, dict]] = []
    for model in CURATED_MODEL_IDS:
        payload = _load_latest_payload_for_model(model, repo_files)
        if payload is None:
            print(f"  SKIP (no results): {model}", file=sys.stderr)
            continue
        payload_rows.append((model, payload))

    if not payload_rows:
        print("ERROR: no model payloads were found.", file=sys.stderr)
        sys.exit(1)

    benchmarks = discover_expanded_benchmarks(
        payloads=[payload for _, payload in payload_rows],
        core_benchmarks=CORE_BENCHMARKS,
        extra_limit=50,
    )
    benchmark_keys = list(benchmarks)
    column_names = expanded_column_names(benchmark_keys)
    required_valid = max(4, math.ceil(len(benchmark_keys) * 0.1))

    rows: list[dict[str, float | None] | dict[str, str | float | None]] = []
    for model, payload in payload_rows:
        scores = extract_benchmark_scores(payload, benchmarks)
        n_valid = sum(value is not None for value in scores.values())
        if n_valid < required_valid:
            print(f"  SKIP (<{required_valid} scores): {model} ({n_valid}/{len(benchmark_keys)})", file=sys.stderr)
            continue

        rows.append({"model": model, **scores})
        print(f"  OK: {model} ({n_valid}/{len(benchmark_keys)} benchmarks)", file=sys.stderr)

    if not rows:
        print("ERROR: no models passed filtering.", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "scores_expanded.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model"] + column_names)
        for row in rows:
            cells = [row["model"]]
            for bench_key in benchmark_keys:
                value = row[bench_key]
                cells.append(f"{value:.6f}" if value is not None else "")
            writer.writerow(cells)

    print(f"\nWrote {len(rows)} models x {len(column_names)} benchmarks to {out_path}", file=sys.stderr)


def _load_latest_payload_for_model(model: str, repo_files: set[str]) -> dict | None:
    prefix = f"{model}/"
    model_files = sorted(path for path in repo_files if path.startswith(prefix) and path.endswith(".json"))
    if not model_files:
        return None

    local_path = hf_hub_download(REPO_ID, model_files[-1], repo_type=REPO_TYPE)
    with open(local_path, encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
