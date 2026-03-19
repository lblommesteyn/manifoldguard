"""Fetch Open LLM Leaderboard v2 results and build scores.csv.

Requires: pip install huggingface_hub
This is NOT a runtime dependency — run once to regenerate the dataset.

Usage:
    python scripts/build_dataset.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, HfApi

REPO_ID = "open-llm-leaderboard/results"
REPO_TYPE = "dataset"
OUT_DIR = Path(__file__).resolve().parents[1] / "datasets" / "lm_eval_real"

# Top-level aggregate benchmarks and their preferred metric key.
BENCHMARKS: dict[str, str] = {
    "leaderboard_arc_challenge": "acc_norm,none",
    "leaderboard_bbh": "acc_norm,none",
    "leaderboard_gpqa": "acc_norm,none",
    "leaderboard_ifeval": "inst_level_strict_acc,none",
    "leaderboard_math_hard": "exact_match,none",
    "leaderboard_mmlu_pro": "acc,none",
    "leaderboard_musr": "acc_norm,none",
}

COLUMN_NAMES = [
    "ARC-Challenge",
    "BBH",
    "GPQA",
    "IFEval",
    "MATH-Hard",
    "MMLU-Pro",
    "MUSR",
]

# Curated model list: diverse families, sizes, base + instruct variants.
MODELS = [
    # Llama family
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.3-70B-Instruct",
    # Mistral family
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    "mistralai/Mistral-Nemo-Base-2407",
    "mistralai/Mistral-Small-24B-Base-2501",
    # Qwen family
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-72B",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/QwQ-32B",
    # Google Gemma
    "google/gemma-7b",
    "google/gemma-2-9b",
    "google/gemma-2-27b",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    # Microsoft Phi
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/phi-4",
    # Yi
    "01-ai/Yi-1.5-9B",
    "01-ai/Yi-1.5-34B",
    "01-ai/Yi-1.5-34B-Chat",
    # Falcon
    "tiiuae/falcon-7b",
    "tiiuae/falcon-40b",
    "tiiuae/Falcon3-7B-Base",
    "tiiuae/Falcon3-10B-Base",
    # DeepSeek
    "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    # EleutherAI
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-12b",
    # Stability / DBRX / Others
    "stabilityai/stablelm-2-12b",
    "databricks/dbrx-base",
    "HuggingFaceH4/zephyr-7b-beta",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "allenai/OLMo-7B-hf",
    "allenai/Llama-3.1-Tulu-3-8B",
]


def fetch_latest_result(api: HfApi, model: str) -> dict | None:
    """Download the most recent results JSON for a model."""
    prefix = f"{model}/"
    files = [
        f for f in api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
        if f.startswith(prefix) and f.endswith(".json")
    ]
    if not files:
        return None

    # Take the last file (lexicographic = most recent timestamp).
    target = sorted(files)[-1]
    local = hf_hub_download(REPO_ID, target, repo_type=REPO_TYPE)
    with open(local, encoding="utf-8") as f:
        return json.load(f)


def extract_scores(payload: dict) -> dict[str, float | None]:
    """Extract benchmark scores from a result payload."""
    results = payload.get("results", {})
    scores: dict[str, float | None] = {}
    for bench_key, metric_key in BENCHMARKS.items():
        task_data = results.get(bench_key, {})
        value = task_data.get(metric_key)
        if isinstance(value, (int, float)):
            scores[bench_key] = round(float(value), 6)
        else:
            scores[bench_key] = None
    return scores


def main() -> None:
    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    file_set = set(all_files)

    rows: list[dict] = []
    for model in MODELS:
        prefix = f"{model}/"
        model_files = sorted(f for f in file_set if f.startswith(prefix) and f.endswith(".json"))
        if not model_files:
            print(f"  SKIP (no results): {model}", file=sys.stderr)
            continue

        target = model_files[-1]
        local = hf_hub_download(REPO_ID, target, repo_type=REPO_TYPE)
        with open(local, encoding="utf-8") as f:
            payload = json.load(f)

        scores = extract_scores(payload)
        n_valid = sum(1 for v in scores.values() if v is not None)
        if n_valid < 4:
            print(f"  SKIP (<4 scores): {model} ({n_valid}/7)", file=sys.stderr)
            continue

        rows.append({"model": model, **scores})
        print(f"  OK: {model} ({n_valid}/7 benchmarks)", file=sys.stderr)

    if not rows:
        print("ERROR: no models passed filtering.", file=sys.stderr)
        sys.exit(1)

    # Write CSV.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "scores.csv"
    bench_keys = list(BENCHMARKS.keys())

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + COLUMN_NAMES)
        for row in rows:
            cells = [row["model"]]
            for key in bench_keys:
                val = row[key]
                cells.append(f"{val:.6f}" if val is not None else "")
            writer.writerow(cells)

    print(f"\nWrote {len(rows)} models x {len(COLUMN_NAMES)} benchmarks to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
