# manifoldguard

`manifoldguard` predicts missing benchmark scores for new LLMs/models and flags when those predictions are likely to be unreliable before you pay to run the full evaluation suite.

## What it does

Given a partially observed model x benchmark score matrix (rows = models, cols = benchmarks, cells = scores with NaNs allowed):

1. **Matrix completion** - trains a low-rank factorization `U V^T` (PyTorch, observed entries only) to fill missing cells.
2. **New-model inference** - given a handful of observed scores for a brand-new model, infers its latent vector `u` via closed-form ridge regression and predicts all missing benchmarks.
3. **Failure detection** - computes OOD risk features and fits a logistic regression to predict whether completion error will be high (top-20% MAE), before running the hidden benchmarks.
4. **Conformal intervals** - wraps every prediction in a distribution-free `[pred +/- q]` interval guaranteed to cover the true score with >= 90% probability.

## Architecture

```
manifoldguard/
  _utils.py       shared numeric helpers (quantile_higher)
  data.py         CSV loader + synthetic data generator
  mf.py           PyTorch low-rank MF training (MFModel)
  inference.py    ridge-regression latent inference + LOO residuals
  ood.py          OOD features: residual energy, Mahalanobis, variance summary
  ensemble.py     ensemble of E MF models -> predictive variance
  episodes.py     simulate "new model" evaluation episodes
  evaluation.py   end-to-end harness: failure AUC + conformal coverage
  conformal.py    split conformal quantile + interval helpers

scripts/
  run_experiments.py   CLI entry-point
tests/                 unit tests (pytest)
```

## OOD features

| Feature | Description |
|---|---|
| `residual_energy_obs` | MSE of ensemble mean on observed entries |
| `max_abs_residual_obs` | largest absolute error on observed entries |
| `mahalanobis_u` | Mahalanobis distance of inferred `u` from training distribution (LedoitWolf covariance) |
| `mean/max_predictive_variance_hidden` | ensemble variance on hidden benchmarks |
| `loo_mean/max_abs_error_obs` | leave-one-out residuals on observed entries - measures self-consistency of the observed subset |
| `min_singular_value_obs` | min singular value of `V_S = V[observed_indices]` - low value means the observations do not span the latent space, so `u` is poorly identified (negative predictor of MAE) |
| `condition_number_obs` | max / min singular value of `V_S` - high value signals ill-conditioning and predicts large hidden-benchmark error (positive predictor of MAE) |

The geometric coverage features (`min_singular_value_obs`, `condition_number_obs`) are the most predictive: they directly measure whether the revealed benchmarks can uniquely determine the model's latent vector.

## Quickstart

```bash
pip install -e ".[dev]"
python scripts/run_experiments.py
```

Expected output (synthetic 80-model dataset, 35% observed fraction, noise=0.15):

```
train models:       60
test models:        20
episodes:           100
completion MAE:     ~0.30
failure AUC:        ~0.69
conformal coverage: ~0.97
conformal quantile: ~0.76
```

To run on your own CSV (rows = models, cols = benchmarks, first row/col may be names):

```bash
python scripts/run_experiments.py --csv path/to/scores.csv
```

To run directly from lm-eval-harness JSON outputs (no CSV conversion):

```bash
python scripts/run_experiments.py --lm-eval-dir path/to/lm_eval_outputs
```

## Key options

```
--rank INT               MF latent rank (default 4)
--ensemble-size INT      ensemble members (default 5)
--csv PATH               wide score matrix in CSV format
--lm-eval-dir PATH       lm-eval JSON directory (recursive)
--observed-fraction F    fraction of benchmarks revealed at inference (default 0.35)
--model-test-fraction F  fraction of models held out from MF training (default 0.25)
--alpha F                conformal miscoverage level, 1-alpha = target coverage (default 0.1)
--device STR             pytorch device: cpu / cuda / mps (default: auto)
--seed INT               random seed (default 0)
```

## Running tests

```bash
pytest
```
