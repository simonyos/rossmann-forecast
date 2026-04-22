# Rossmann Store Sales — Entity Embeddings vs. Gradient Boosting

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simonyos/rossmann-forecast/blob/main/notebooks/rossmann_forecast_colab.ipynb)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible reimplementation of

> **Guo, C. & Berkhahn, F. (2016). Entity Embeddings of Categorical Variables.** arXiv preprint arXiv:1604.06737. [arxiv.org/abs/1604.06737](https://arxiv.org/abs/1604.06737)

benchmarked on the **Kaggle Rossmann Store Sales** competition (Jun 2015) against
seasonal-naive baselines and a gradient-boosted tree (LightGBM). The paper showed that
learning per-category embedding tables jointly with a two-layer MLP can push a
previously unremarkable neural network into the top-ranks of a retail-forecasting
competition dominated by tree ensembles. Here we reproduce the core recipe — one
embedding table per categorical feature, dimension `min(50, ⌈cardinality/2⌉ + 1)`,
concatenated with standardized continuous features, two hidden layers of `1000 → 500`
ReLU units — and measure how far a disciplined LightGBM implementation can close the
gap under matched features and matched validation protocol.

---

## Abstract

Retail demand forecasting has historically been dominated by gradient-boosted decision
trees; most top Kaggle solutions for the 2015 Rossmann Store Sales competition were
GBM variants. Guo & Berkhahn (2016) showed that a plain feed-forward network can
approach GBM performance when each categorical input is embedded into a dense
low-dimensional vector learned end-to-end, rather than one-hot encoded. We reproduce
that claim on the same 1,017,209-row, 1,115-store Kaggle dataset under a common feature
schema and a chronological 6-week held-out validation split. Under matched features,
the entity-embedding MLP attains an **RMSPE of 0.1238** on the held-out validation
window, narrowly beating our LightGBM baseline's **0.1279** — replicating the paper's
central finding that a carefully-designed shallow MLP can meet or exceed gradient
boosting on a large tabular retail problem. LightGBM in turn is better on absolute-
scale metrics (RMSE **921** vs. **958**, MAE **635** vs. **647**), indicating the two
models are optimising subtly different objectives: the MLP, trained on standardised
`log1p(Sales)`, minimises log-scale error — closest to the percentage-based competition
metric — while LightGBM's tree structure tracks absolute magnitudes better at the
distribution tails. Both learned models roughly halve the seasonal-naive baseline's
RMSPE of 0.2563, confirming the features themselves carry most of the signal.

> **Reproducibility status.** All numbers in this README are produced by the commands
> in §8, without any manual curation. Full pipeline runs in ≈ 2 minutes on an Apple
> Silicon laptop (download + features + three models + evaluation).

---

## 1. Background

Rossmann is a German pharmacy chain operating ~1,115 stores across Germany. Their
Kaggle competition in 2015 asked participants to forecast daily store-level `Sales` for
the 6-week window 2015-08-01 → 2015-09-17 using a ~2.5-year history of per-store daily
observations. The evaluation metric is Root-Mean-Square-Percentage-Error (RMSPE), which
treats all stores as equally important regardless of their sales volume:

$$
\text{RMSPE}(y, \hat{y}) = \sqrt{\frac{1}{N} \sum_{i : y_i > 0} \left(\frac{y_i - \hat{y}_i}{y_i}\right)^2}.
$$

The three systems studied here are:

1. **Seasonal-naive** — for each `(Store, DayOfWeek)` pair, predict the mean of the last
   *N* observations in the training window. This is the canonical weak baseline for
   any seasonal retail series and anchors the "did we learn anything?" question.
2. **LightGBM** — a gradient-boosted decision-tree ensemble trained on
   `log1p(Sales)` with categorical columns passed as native `category` features.
3. **Entity-embeddings MLP** (Guo & Berkhahn, 2016) — a feed-forward network where
   each categorical input has its own learned embedding table; all embeddings are
   concatenated with the standardized continuous features and fed through an MLP.

Feature engineering is identical across the three learned models so that the
comparison isolates the model family, not the features.

## 2. What is reproduced, what is different

| Aspect | Original paper (Rossmann, 3rd place) | This work |
|---|---|---|
| Dataset | Kaggle Rossmann Store Sales (2015) | Same |
| Validation | Last 6 weeks of train (2015-06-19 → 2015-07-31) | Same chronological split |
| Metric | RMSPE on held-out window | Same, plus RMSE and MAE |
| Categorical features | Store, DayOfWeek, Day, Month, Year, Promo, StateHoliday, SchoolHoliday, StoreType, Assortment, Promo2, PromoIntervalCode | Same 12 + a derived `InPromoIntervalMonth` boolean |
| Continuous features | None (paper discards them) | `CompetitionDistance`, `CompetitionOpenMonths` |
| Embedding dim | `min(50, cardinality//2 + 1)` | Same |
| Hidden layers | `1000 → 500` ReLU + linear out | Same, with 0.1 dropout |
| Target | `log1p(Sales)`, standardized | Same |
| Optimizer | Adam, default LR | Adam, `lr = 1e-3`, `weight_decay = 1e-5`, early stopping on val RMSPE |
| GBM baseline | — | LightGBM with matched features, early stopping on val RMSE of log target |
| Tracking | None | MLflow per run, `artifacts/*_summary.json` |
| Reproducibility | — | Single command per stage; Colab notebook; Docker image; CI |

## 3. Dataset

Downloaded via the Kaggle API (`kaggle competitions download -c rossmann-store-sales`).
Three files: `train.csv` (1,017,209 rows), `test.csv` (41,088 rows, no `Sales`),
`store.csv` (1,115 rows).

Rows with `Open == 0` or `Sales == 0` are dropped prior to modelling — the Kaggle
metric does not score them and including them distorts the loss.

### 3.1 Splits

| Split | Date range | Rows (after drop) |
|---|---|---|
| train | 2013-01-01 → 2015-06-18 | ≈ 780,000 |
| valid | 2015-06-19 → 2015-07-31 (held-out 6 weeks) | ≈ 44,000 |

The Kaggle `test.csv` is the private leaderboard window (2015-08-01 onward) and has no
labels; we do not use it for model selection here.

## 4. Feature engineering

Twelve categorical inputs, each encoded to a small contiguous integer code, plus two
continuous inputs:

| Feature | Type | Cardinality / range | Notes |
|---|---|---|---|
| `Store` | cat | 1,115 | store id |
| `DayOfWeek` | cat | 7 | 1 = Monday |
| `Day`, `Month`, `Year` | cat | 31 / 12 / 3 | decomposed date |
| `StateHoliday` | cat | 4 | `0`, `a`, `b`, `c` |
| `SchoolHoliday` | cat | 2 | |
| `Promo` | cat | 2 | daily store promotion |
| `StoreType`, `Assortment` | cat | 4 / 3 | |
| `Promo2`, `PromoIntervalCode` | cat | 2 / 4 | consecutive-promo programme |
| `InPromoIntervalMonth` | cat | 2 | derived: is the current month listed in `PromoInterval`? |
| `CompetitionDistance` | cont | metres, median-imputed | |
| `CompetitionOpenMonths` | cont | 0–120 | months since competitor store opened |

## 5. Models

### 5.1 Seasonal-naive baselines

For each `(Store, DayOfWeek)` pair, predict the mean of the last *N* observed `Sales`
values in the training window. We report `N = 4` and `N = 8`. A per-store median is
reported for reference.

### 5.2 LightGBM

Trained on `log1p(Sales)`. Native `category` dtype passed to LightGBM for all
categorical columns. Early stopping on validation RMSE with a patience of 50 rounds;
maximum 3,000 rounds. Key hyperparameters:

- `learning_rate = 0.05`
- `num_leaves = 127`
- `min_data_in_leaf = 200`
- `feature_fraction = 0.9`
- `bagging_fraction = 0.8` every 5 rounds

### 5.3 Entity-embeddings MLP (paper reproduction)

For each of the 12 categorical columns, a `nn.Embedding(cardinality, d)` with
$d = \min(50, \lfloor c/2 \rfloor + 1)$. All 12 embedded vectors are concatenated with
the standardized continuous features and passed through a `Linear(1000) → ReLU →
Dropout(0.1) → Linear(500) → ReLU → Dropout(0.1) → Linear(1)` head. Loss is MSE on a
standardized `log1p(Sales)` target; predictions are undone at inference. Trained with
Adam (`lr = 1e-3`, `weight_decay = 1e-5`) for up to 20 epochs, batch size 1,024, early
stopping on validation RMSPE with patience 3.

## 6. Results

All numbers are on the held-out 6-week validation window (2015-06-19 → 2015-07-31,
n = 41,396 scored store-days, 1,115 stores). Ranked by RMSPE (Kaggle metric).

| Model                  | **Val RMSPE** |   RMSE |   MAE | Train seconds | Best iter / epoch |
|:-----------------------|--------------:|-------:|------:|--------------:|------------------:|
| **Entity-embeddings MLP** | **0.1238**    |  957.9 | 647.1 |          71.6 | epoch 7 of 20     |
| LightGBM               |        0.1279 |  921.2 | 634.7 |          42.0 | iter 1,163        |
| Seasonal-naive (N = 8) |        0.2563 | 1646.7 | 1267.5|           0.07 | —                 |
| Seasonal-naive (N = 4) |        0.2625 | 1680.3 | 1256.1|           0.05 | —                 |
| Per-store median       |        0.3096 | 1944.3 | 1424.0|           0.02 | —                 |

![Validation RMSPE comparison](reports/figures/rmspe_comparison.png)

![Residual distribution per model](reports/figures/residual_distribution.png)

For reference, the Rossmann Kaggle private-leaderboard winner hit RMSPE **0.10021**
(3rd-place was Guo & Berkhahn themselves at **0.10557**). Our models reach **0.12–0.13**
with roughly a dozen features, no store-level history features, no pseudo-labelling,
and no ensembling — adding those would close most of the remaining gap.

Full machine-readable summary: [`reports/summary.md`](reports/summary.md). Per-run
metrics are logged to MLflow (`mlruns/`). Raw predictions per model are saved as
`.npy` arrays under `artifacts/` so downstream residual / error-bucket analysis doesn't
need re-training.

## 7. Discussion

### 7.1 The paper's claim replicates — with nuance

Guo & Berkhahn argued that a plain two-hidden-layer MLP, given embedding tables per
categorical input, can match or beat gradient boosting on a large retail tabular
problem. Under our matched-feature protocol:

- **On the competition metric (RMSPE) the MLP wins** (0.1238 vs. 0.1279, ∆ ≈ 3.2 %
  relative reduction). This is the direction the paper predicts.
- **On absolute error (RMSE, MAE) LightGBM wins** (921 vs. 958, 635 vs. 647). This is
  consistent with a simple story: both models are trained on `log1p(Sales)`, but the MLP
  additionally standardises the log target, so its loss is *closer to relative error*
  and it is rewarded more for getting small-sales stores right. LightGBM, optimising
  plain MSE on the log scale, tolerates a heavier tail of large-sales mispredictions.

In other words, the two models are not identically ranked — they are optimising subtly
different objectives and trade off accordingly. This is a more honest statement than
"entity embeddings beat LightGBM".

### 7.2 Where each model visibly fails

The residual histogram (above) makes the trade-off concrete. LightGBM has a tighter
peak near zero error and longer tails; the MLP has a slightly wider central peak but
keeps percentage errors below 20 % for more of the distribution — exactly the regime
RMSPE cares about.

Error breakdown by `StateHoliday` and `Promo` (not plotted here, but available in
`artifacts/*_predictions.npy`):

- Both models over-predict on the first day after a holiday — a dynamic feature
  (days-since-holiday, sales-last-N-days) would likely fix this.
- Small stores (`CompetitionDistance > 20 km` rural locations) are where the MLP
  benefits most from the `Store` embedding learning store-specific baselines.

### 7.3 What's missing versus the Kaggle top-3

Our 0.12–0.13 RMSPE lands about 2 points behind Guo & Berkhahn's 3rd-place 0.10557.
The main missing ingredients, in descending order of expected impact:

1. **Lag and rolling-mean features** (sales-1-day, sales-7-days, sales-28-days, mean
   of last 4 same-DayOfWeek values, etc.) — the paper and all top finishers used
   rolling features; we deliberately skipped them to keep the pipeline fast.
2. **Store-level history embedding.** Averaging embeddings of the *same store's*
   embeddings across time (Guo-Berkhahn §4.3) was part of their final submission.
3. **Ensembling.** Every top-3 submission averaged at least 5 models.
4. **External data.** Google Trends, weather, school-vacation schedules.

Adding (1) alone typically drops RMSPE by 2–3 points on this dataset.

## 8. Inference API

```bash
make serve
curl -X POST -H "Content-Type: application/json" http://localhost:8000/predict -d '{
  "Store": 1,
  "Date": "2015-08-01",
  "Promo": 1,
  "StateHoliday": "0",
  "SchoolHoliday": 0,
  "StoreType": "c",
  "Assortment": "a",
  "CompetitionDistance": 1270.0,
  "CompetitionOpenSinceMonth": 9,
  "CompetitionOpenSinceYear": 2008,
  "Promo2": 0,
  "PromoInterval": null
}'
```

Response:

```json
{ "sales": 4812.3, "model_used": "lightgbm" }
```

Swap the model served via `MODEL_CHOICE=emb` once the entity-embedding hot path is
wired in (currently stubbed — PR tracking this in the issues).

## 9. Reproducibility

```bash
make setup                 # uv venv + editable install
forecast download          # Kaggle CLI; needs ~/.kaggle/kaggle.json + accepted rules
forecast build-features    # engineer features + chronological split → data/processed/*.parquet
forecast train-baseline    # seasonal naive + per-store median
forecast train-gbm         # LightGBM with early stopping
forecast train-emb         # entity-embeddings MLP (≈ 5 min on CPU, ≈ 1 min on T4)
forecast evaluate          # comparison table + figures
make serve                 # FastAPI at http://localhost:8000/docs
```

or, on a free Colab runtime, via the [notebook](https://colab.research.google.com/github/simonyos/rossmann-forecast/blob/main/notebooks/rossmann_forecast_colab.ipynb).

Environment and random seeds are controlled through `rossmann_forecast.config.Settings`.
Package versions are pinned in [`pyproject.toml`](pyproject.toml). CI runs `ruff` +
`pytest` on every push.

## 10. Repository layout

```
src/rossmann_forecast/
  config.py                 env-driven settings
  cli.py                    `forecast` Typer CLI
  data/download.py          Kaggle CLI wrapper
  data/load.py              CSV → typed pandas (with known Rossmann quirks handled)
  features/engineer.py      categorical encoding + chronological split + parquet persist
  metrics.py                RMSPE, RMSE, MAE
  models/baseline.py        seasonal-naive + per-store-median
  models/gbm.py             LightGBM
  models/entity_embeddings.py  Guo-Berkhahn reproduction
  evaluate.py               comparison table + figures
  serve/api.py              FastAPI /health, /predict
tests/                      pytest unit tests (metrics, features, config, CLI, API)
.github/workflows/ci.yml    ruff + pytest on push and PR
Dockerfile                  slim image that serves the API
```

## 11. Citation

Original paper:

```bibtex
@misc{guo2016entity,
  author       = {Cheng Guo and Felix Berkhahn},
  title        = {Entity Embeddings of Categorical Variables},
  howpublished = {arXiv preprint arXiv:1604.06737},
  year         = {2016},
  url          = {https://arxiv.org/abs/1604.06737}
}
```

This reimplementation:

```bibtex
@misc{yosboon2026rossmannrepro,
  author       = {Yosboon, Simon},
  title        = {Reproducing Entity Embeddings of Categorical Variables on Kaggle Rossmann Store Sales},
  year         = {2026},
  howpublished = {\url{https://github.com/simonyos/rossmann-forecast}}
}
```

## License

MIT — see [LICENSE](LICENSE).
