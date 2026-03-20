# manifoldguard

`manifoldguard` predicts missing benchmark scores for a partially evaluated model and estimates when those predictions should not be trusted yet.

The project is aimed at the practical question behind expensive LLM evaluation: if you have only run a few benchmarks, can you forecast the rest of the suite, estimate the risk of being wrong, and decide whether to stop or keep evaluating?

## Why this matters

Running a full benchmark suite across many models is slow and expensive. `manifoldguard` tries to reduce that burden by:

- completing missing benchmark scores with low-rank matrix factorization,
- flagging high-risk predictions before you trust them,
- wrapping predictions in conformal intervals, and
- giving a simple recommendation about which benchmark to run next.

## What it does

Given a model x benchmark score matrix with NaNs allowed:

1. Matrix completion: train a low-rank factorization `U V^T` on observed entries only.
2. New-model inference: infer a latent vector for a partially observed model with ridge regression.
3. Failure detection: use OOD features to estimate when hidden-score predictions are likely to fail.
4. Conformal intervals: attach distribution-free uncertainty intervals to each hidden prediction.
5. Decision support: recommend whether to stop, continue, or run a specific next benchmark.

## Quickstart

Install the package and run the tests:

```bash
pip install -e ".[dev]"
pytest
```

For figures and other visualization artifacts, install the optional plotting extra:

```bash
pip install -e ".[dev,viz]"
```

Run the cold-start demo:

```bash
python scripts/run_demo.py
```

That command reads [examples/demo_request.json](examples/demo_request.json) and writes a human-readable report plus JSON output to [results/demo/demo_report.json](results/demo/demo_report.json).

You can also auto-sample a partial evaluation episode from a model in a CSV or lm-eval directory:

```bash
python scripts/run_demo.py \
  --csv datasets/lm_eval_real/scores.csv \
  --model-name "meta-llama/Meta-Llama-3-70B-Instruct" \
  --observed-fraction 0.5 \
  --show-all
```

## Real-data workflow

The repo includes a fixed real dataset at [datasets/lm_eval_real/scores.csv](datasets/lm_eval_real/scores.csv). The main experiment scripts are:

```bash
python scripts/run_real_data_benchmarks.py
python scripts/run_baseline_comparison.py
python scripts/run_ablations.py
python scripts/run_alpha_coverage_validation.py
python scripts/run_family_split_benchmark.py
python scripts/run_cost_savings_analysis.py
python scripts/generate_raw_charts.py
python scripts/generate_conference_figures.py
python scripts/run_manifold_visualization.py
```

These produce reusable CSV artifacts under `results/`:

- `results/real_data_benchmark/` for the main random-split benchmark
- `results/baseline_comparison/` for mean fill / nearest neighbor / MF comparisons
- `results/ablations/` for rank, ensemble size, observed fraction, and feature subset ablations
- `results/alpha_validation/` for empirical coverage versus target coverage across multiple alpha values
- `results/family_split/` for the tougher family-holdout OOD setting
- `results/cost_savings/` for risk-threshold versus benchmarks-avoided tradeoffs
- `results/demo/` for the sample partial-score demo report
- `results/raw_charts/` and `results/conference_figures/` for slide-ready visuals
- `results/manifold_visualization/` for the learned manifold projection and target-model placement

For a wider real matrix with subtask-level columns, build [datasets/lm_eval_real/scores_expanded.csv](datasets/lm_eval_real/scores_expanded.csv):

```bash
pip install huggingface_hub
python scripts/build_expanded_dataset.py
```

That expanded dataset currently contains 61 models x 44 benchmark columns and is documented in [data_note_expanded.md](datasets/lm_eval_real/data_note_expanded.md).

## Generic experiment runner

For synthetic data, CSV matrices, or lm-eval JSON directories:

```bash
python scripts/run_experiments.py
python scripts/run_experiments.py --csv datasets/lm_eval_real/scores.csv
python scripts/run_experiments.py --lm-eval-dir path/to/lm_eval_outputs
```

The runner can also export reusable metrics directly:

```bash
python scripts/run_experiments.py \
  --csv datasets/lm_eval_real/scores.csv \
  --output-json results/exports/example_metrics.json \
  --output-csv results/exports/example_metrics.csv
```

## Current real-data artifacts

The repo currently ships:

- a curated real lm-eval style dataset with model and task coverage notes,
- random-split benchmark results across multiple seeds,
- baseline comparison tables,
- ablation tables,
- conformal coverage validation across multiple alpha values,
- a stronger family-holdout benchmark,
- a risk-threshold cost-savings table, and
- a one-command demo report for a partially observed model,
- conference figures and raw chart exports, and
- a latent-space plot showing where a partially observed model lands.

## Repository layout

```text
manifoldguard/
  data.py         CSV loading and synthetic data generation
  mf.py           PyTorch matrix factorization
  inference.py    closed-form new-model latent inference
  ensemble.py     ensemble training and predictive variance
  episodes.py     simulated partial-observation episodes
  evaluation.py   benchmark metrics and grouped validation utilities
  ood.py          residual, variance, and geometry OOD features
  lm_eval.py      lm-eval JSON ingestion
  splits.py       family inference and stronger OOD split helpers

scripts/
  run_demo.py
  run_experiments.py
  run_real_data_benchmarks.py
  run_baseline_comparison.py
  run_ablations.py
  run_family_split_benchmark.py
  run_cost_savings_analysis.py
```

## OOD features

| Feature | Description |
|---|---|
| `residual_energy_obs` | mean squared error on observed entries |
| `max_abs_residual_obs` | largest absolute observed-entry error |
| `mahalanobis_u` | latent shift relative to the training model manifold |
| `mean/max_predictive_variance_hidden` | ensemble uncertainty on hidden benchmarks |
| `loo_mean/max_abs_error_obs` | self-consistency of the observed subset |
| `min_singular_value_obs` | how well the observed benchmarks span the latent space |
| `condition_number_obs` | how ill-conditioned the observed subset is |

The geometry features (`min_singular_value_obs`, `condition_number_obs`) are especially important because they tell you whether the revealed benchmarks actually identify the model's latent position well enough to trust completion.

## Status

The repo is in active research-prototype mode. It is strong on reproducible experiment scripts, demo flows, and figure generation, though it still prioritizes research clarity over packaging polish.
