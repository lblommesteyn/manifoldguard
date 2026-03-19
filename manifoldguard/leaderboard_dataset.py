from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


CORE_BENCHMARKS: dict[str, str] = {
    "leaderboard_arc_challenge": "acc_norm,none",
    "leaderboard_bbh": "acc_norm,none",
    "leaderboard_gpqa": "acc_norm,none",
    "leaderboard_ifeval": "inst_level_strict_acc,none",
    "leaderboard_math_hard": "exact_match,none",
    "leaderboard_mmlu_pro": "acc,none",
    "leaderboard_musr": "acc_norm,none",
}

PREFERRED_METRIC_KEYS = (
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "inst_level_strict_acc,none",
)

CURATED_MODEL_IDS = [
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
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    "mistralai/Mistral-Nemo-Base-2407",
    "mistralai/Mistral-Small-24B-Base-2501",
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
    "google/gemma-7b",
    "google/gemma-2-9b",
    "google/gemma-2-27b",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/phi-4",
    "01-ai/Yi-1.5-9B",
    "01-ai/Yi-1.5-34B",
    "01-ai/Yi-1.5-34B-Chat",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-40b",
    "tiiuae/Falcon3-7B-Base",
    "tiiuae/Falcon3-10B-Base",
    "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-12b",
    "stabilityai/stablelm-2-12b",
    "databricks/dbrx-base",
    "HuggingFaceH4/zephyr-7b-beta",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "allenai/OLMo-7B-hf",
    "allenai/Llama-3.1-Tulu-3-8B",
]


def select_metric_key(metrics: Mapping[str, Any]) -> str | None:
    for key in PREFERRED_METRIC_KEYS:
        if isinstance(metrics.get(key), (int, float)):
            return key
    return None


def discover_expanded_benchmarks(
    payloads: Sequence[Mapping[str, Any]],
    core_benchmarks: Mapping[str, str] | None = None,
    extra_limit: int = 50,
) -> dict[str, str]:
    """Build a deterministic expanded benchmark schema from multiple payloads."""
    if extra_limit < 0:
        raise ValueError("extra_limit must be non-negative.")

    benchmarks = dict(core_benchmarks or CORE_BENCHMARKS)
    occurrence_counts: Counter[str] = Counter()
    metric_candidates: dict[str, list[Mapping[str, Any]]] = {}

    for payload in payloads:
        results = payload.get("results", {})
        if not isinstance(results, Mapping):
            continue
        for bench_key, metrics in results.items():
            if not isinstance(bench_key, str) or not bench_key.startswith("leaderboard_"):
                continue
            if bench_key in benchmarks:
                continue
            if not isinstance(metrics, Mapping):
                continue
            occurrence_counts[bench_key] += 1
            metric_candidates.setdefault(bench_key, []).append(metrics)

    ranked_candidates = sorted(
        metric_candidates,
        key=lambda bench_key: (-occurrence_counts[bench_key], bench_key),
    )

    for bench_key in ranked_candidates[:extra_limit]:
        selected_key = None
        for metrics in metric_candidates[bench_key]:
            selected_key = select_metric_key(metrics)
            if selected_key is not None:
                break
        if selected_key is not None:
            benchmarks[bench_key] = selected_key

    return benchmarks


def expanded_column_names(benchmark_keys: Iterable[str]) -> list[str]:
    return [bench_key.removeprefix("leaderboard_") for bench_key in benchmark_keys]


def extract_benchmark_scores(
    payload: Mapping[str, Any],
    benchmarks: Mapping[str, str],
) -> dict[str, float | None]:
    results = payload.get("results", {})
    if not isinstance(results, Mapping):
        return {bench_key: None for bench_key in benchmarks}

    scores: dict[str, float | None] = {}
    for bench_key, metric_key in benchmarks.items():
        metrics = results.get(bench_key, {})
        if not isinstance(metrics, Mapping):
            scores[bench_key] = None
            continue
        value = metrics.get(metric_key)
        scores[bench_key] = round(float(value), 6) if isinstance(value, (int, float)) else None
    return scores
