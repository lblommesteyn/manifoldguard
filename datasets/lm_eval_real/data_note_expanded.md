# Dataset: Expanded Open LLM Leaderboard v2 Scores

## Source

Scores are fetched from the Hugging Face `open-llm-leaderboard/results` dataset, using the latest available JSON result for each curated model in the repo's model list.

## Shape

- 61 models
- 44 benchmark columns
- 7 core aggregate benchmarks
- 37 additional leaderboard subtasks discovered across the full payload set

## Schema

Columns use raw leaderboard-style task names with the `leaderboard_` prefix removed, for example:

- `arc_challenge`
- `bbh`
- `gpqa`
- `ifeval`
- `math_hard`
- `mmlu_pro`
- `musr`
- `bbh_boolean_expressions`
- `gpqa_extended`
- `musr_object_placements`

## Metric selection

For each task, the builder prefers the first available metric in this order:

1. `acc_norm,none`
2. `acc,none`
3. `exact_match,none`
4. `inst_level_strict_acc,none`

## Filtering

- The schema is discovered from the union of all downloaded payloads, not just the first model.
- A model is kept if it has at least `max(4, ceil(10% of columns))` non-missing scores.

## Missingness

- 55 of 2684 cells are missing (2.0%).
- 26 of 61 models have complete data across all 44 columns.

## Regeneration

```bash
pip install huggingface_hub
python scripts/build_expanded_dataset.py
```
