# Dataset: Open LLM Leaderboard v2 Scores

## Source

Scores are from the [Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) hosted by HuggingFace.
Raw results are stored in the `open-llm-leaderboard/results` dataset repository.
Each model's most recent result JSON was used.

## Benchmarks (7 tasks)

| Column | Leaderboard Key | Metric |
|--------|----------------|--------|
| ARC-Challenge | `leaderboard_arc_challenge` | `acc_norm` |
| BBH | `leaderboard_bbh` | `acc_norm` |
| GPQA | `leaderboard_gpqa` | `acc_norm` |
| IFEval | `leaderboard_ifeval` | `inst_level_strict_acc` |
| MATH-Hard | `leaderboard_math_hard` | `exact_match` |
| MMLU-Pro | `leaderboard_mmlu_pro` | `acc` |
| MUSR | `leaderboard_musr` | `acc_norm` |

All scores are in [0, 1].

## Model selection

59 models were selected from major open-weight families to provide diversity in:
- **Architecture**: decoder-only transformers, MoE (Mixtral, DBRX)
- **Scale**: 1B to 72B parameters
- **Training**: base and instruction-tuned variants
- **Organizations**: Meta, Mistral, Qwen, Google, Microsoft, DeepSeek, 01.ai, TII, EleutherAI, Stability, Databricks, HuggingFace, NousResearch, Allen AI

## Filtering

- Models must have scores on at least 4 of 7 benchmarks.
- Two models were excluded for having only 2/7 scores (Mistral-7B-v0.1, Mistral-7B-v0.3).
- No deduplication was needed; each HuggingFace model ID appears once.

## Missing values

- 33 of 413 cells are missing (8.0%).
- 26 of 59 models have complete data across all 7 benchmarks.
- Missingness is concentrated in `ARC-Challenge`, which is absent for many newer models (Llama 3.1+, Qwen 2.5, Gemma 2) — likely due to leaderboard evaluation timing or task availability changes.
- Empty cells in the CSV are loaded as NaN by `manifoldguard.data.load_score_csv()`.

## Regeneration

```bash
pip install huggingface_hub
python scripts/build_dataset.py
```

## Date

Data fetched: 2026-03-16.
