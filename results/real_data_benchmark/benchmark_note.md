# Real-Data Benchmark Results

## Dataset

- **Source:** `datasets/lm_eval_real/scores.csv`
- **Origin:** Open LLM Leaderboard v2 (HuggingFace)
- **Matrix:** 59 models x 7 benchmarks
- **Benchmarks:** ARC-Challenge, BBH, GPQA, IFEval, MATH-Hard, MMLU-Pro, MUSR
- **Missingness:** 8.0% (33/413 cells), naturally occurring

## Experiment Setup

- **Seeds:** 0, 1, 2, 3, 4 (5 independent runs)
- **Rank:** 3 (latent dimension for matrix factorization)
- **Ensemble size:** 5
- **Observed fraction:** 0.5 (half of each test model's benchmarks revealed)
- **Model test fraction:** 0.25 (15 models held out, 44 used for training)
- **Conformal alpha:** 0.1 (target 90% coverage)

## Results (mean ± std across 5 seeds)

| Metric | Mean | Std |
|--------|------|-----|
| Completion MAE | 0.0689 | 0.0100 |
| Failure AUC | 0.6555 | 0.1158 |
| Conformal Coverage | 0.9070 | 0.0820 |
| Conformal Quantile | 0.2039 | 0.1155 |

## Reproduction

```bash
python scripts/run_real_data_benchmarks.py
```

This runs all 5 seeds and writes `results_by_seed.csv` and `summary_table.csv` to `results/real_data_benchmark/`.

## Notes

- No preprocessing was applied to the dataset; `load_score_csv()` handles NaN parsing.
- Each seed controls the train/test model split, episode simulation, and cross-validation folds.
- Variance across seeds is driven primarily by which models land in the test set (only 15 models).
