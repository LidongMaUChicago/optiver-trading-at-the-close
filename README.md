# Optiver Trading at the Close

Lidong Ma · Closing-auction forecasting · 2023 work, rebuilt November 2026

Predict a Nasdaq stock's next 60-second WAP movement relative to Optiver's synthetic
market index, measured in basis points, from order-book and closing-auction data.
This repository turns the original model notebooks into a shared, testable research
pipeline. It covers Ridge, Random Forest, XGBoost and an LSTM with jointly learned
stock embeddings, chronological evaluation, saved models and streaming inference.

**Historical best public MAE: 4.7 (4.7191, XGBoost version 4, January 16, 2024).**
This is the original project's historical public result. The same saved notebook
page displays private MAE 5.5610; the January rerun entry currently has an Error
status. It is not a verified final-private 4.7 score. The 2026 rewrite has separate
results and must not inherit the old score or a leaderboard rank.

**Current status (November 14, 2026).** 29 unit tests pass. The full-data run (full-v3) is complete; its results are below and in `reports/full_run/`. A follow-up research study is also complete: a frozen protocol, nested walk-forward evaluation, 440 fits, ablations, embedding tests, ensembles and statistics. It is summarised in the next section and written up in `docs/STUDY_REPORT.md`. All 2026 numbers come from local evaluation, not the Kaggle leaderboard.

### Research extension study (local-100d, November 2026)

This study ran on an Apple M4 laptop. Each model is trained on the 100 days before its validation window. The protocol (`configs/study.json`) was frozen before any result. Configurations were chosen on earlier windows only and scored on W3–W5 (days 400–459).

| Nested outer MAE (bp) | XGBoost | Random Forest | MLP | LSTM | Ridge | Zero |
|---|---|---|---|---|---|---|
| W3–W5 mean | **5.933** | 5.957 | 5.969 | 5.976 | 6.000 | 6.066 |

- **XGBoost vs the others.** It beats zero by 0.133 bp (block-bootstrap 95% CI [0.118, 0.147]). It beats Ridge, LSTM and RF by 0.067, 0.043 and 0.024 bp. All four comparisons are significant after Holm correction, using Newey-West and block bootstrap on daily loss differences.
- **Reproduction.** Ridge reproduces full-v3 exactly; CPU XGBoost is within 0.0006 bp of the GPU run.
- **Negative results:**
  - stock embeddings did not help (5 seeds, −0.0006 bp, p = 0.49);
  - LSTM sequence length 1 was best;
  - blends did not beat XGBoost alone.
- **Feature groups.** Timing and lag features matter; cross-sectional features are redundant.
- **Final model.** XGBoost plus cross-sectional centring, selected on development data. Evaluated once on the REUSED days 461–480: **MAE 5.660 vs 5.790 for zero**, better on 20/20 days. These days were already used by full-v3, so this is not a fresh test.
- **Inference checks.** Save/reload and stream-vs-batch parity are exact, and a local Kaggle API protocol test passes. A private Kaggle notebook (no submission) ran the frozen model on Kaggle's current image (Python 3.13) and matched the Mac predictions exactly (`reports/verification.json`).

| Topic | File |
|---|---|
| Full write-up | `docs/STUDY_REPORT.md` |
| Every fit | `reports/study/experiment_index.csv` |
| All aggregates | `reports/study/results.json` |
| Interview material | `docs/PROJECT_EXPLANATION_ZH.md`, `docs/INTERVIEW_NARRATIVE_EN.md` |
| Proposed resume bullets | `docs/RESUME_BULLETS_PROPOSED.md` |

### Full-run results (2026 rebuild, local evaluation)

Development MAE (basis points) on three expanding folds, 20 validation days each:

| Model | Fold 0 (days 400–419) | Fold 1 (420–439) | Fold 2 (440–459) | Pooled | Fit time / fold |
|---|---|---|---|---|---|
| Ridge | 6.458 | 5.590 | 5.970 | 6.006 | ~25 s |
| Random Forest | 6.415 | 5.543 | 5.944 | 5.967 | ~15 min |
| LSTM | 6.413 | 5.536 | 5.938 | 5.962 | ~10 min (GPU) |
| **XGBoost** | **6.367** | **5.508** | **5.910** | **5.928** | ~2.7 min (GPU) |

XGBoost has the lowest MAE in every fold, so it was selected by the pre-declared rule
(lowest pooled development MAE). The ranking XGBoost < LSTM ≈ Random Forest < Ridge is
stable across folds; the gaps are small (0.04–0.08 bp) next to the fold-to-fold swings
(about 0.9 bp), which come from market conditions, not from the models.

Terminal holdout (days 461–480, opened once after selection):

| | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost (selected) | 5.669 | 8.561 | 0.036 |
| Predict zero | 5.790 | 8.720 | 0.000 |

XGBoost beats the zero baseline by 0.12 bp (about 2.1%); the paired day-bootstrap 95%
interval for the improvement is [0.109, 0.135] bp, and it is positive on every one of the
20 holdout days. The most important feature by far is WAP vs mid price, followed by
cross-sectional book imbalance and WAP vs reference price
(`reports/full_run/feature_importance.csv`). These numbers are not comparable to the
historical 4.7191 public score: different period, different evaluation, no leaderboard.

An earlier real-data LSTM crash (SIGSEGV in `AdamW.step`) was a native-library
conflict, not a data problem: scikit-learn, xgboost and torch each load their own
OpenMP runtime on macOS, and initialising xgboost's before torch's made torch's CPU
kernels crash. The package now imports torch first (`src/optiver_close/__init__.py`),
and `tests/test_import_order.py` guards the order. NumPy matmul warnings emitted by
Ridge on macOS Accelerate are spurious: inputs and coefficients are finite, and the
product matches a long-double recomputation to 1e-14.

## Research workflow

```text
Official auction CSV
    → validated stock / day / timestamp keys
    → causal order-book, auction, temporal and cross-sectional features
    → expanding development folds with day gaps
    → compare fixed model candidates by pooled validation MAE
    → select once and refit on development dates
    → evaluate terminal holdout once
    → persist model, feature schema, metrics and data fingerprint
    → same features in row-aligned streaming inference
```

## What changed from the notebooks

| Original implementation | Rebuilt implementation |
|---|---|
| Several independent feature functions | One shared feature builder for training and inference |
| Some global shifts / rolling windows | Stock-and-day grouping, causal windows and boundary tests |
| Some whole-auction sums and means | Current timestamp cross-section and trailing history only |
| Validation dates excluded, but later dates allowed in training | Expanding chronological folds, one-day gaps, terminal holdout |
| Target-trained embeddings before cross-validation | Joint LSTM embeddings learned inside each training fold |
| float16 downcasting, overflow in R² diagnostics | float32 storage and float64 feature arithmetic / metrics |
| Batch caches could reorder outputs or mix features and raw data | Bounded raw current-day history and explicit row-ID alignment |
| Submission call could precede prediction assignment | Exactly one API predict call after predictions are assigned |

The linear baseline uses regularized Ridge instead of unstable unregularized OLS.
XGBoost optimizes absolute error, matching the competition metric. LSTM uses MAE
loss, gradient clipping, an internal chronological validation period and early
stopping; its scaler and imputer see only that internal training period.
There is no claim that these changes have already improved leaderboard MAE.

## Setup

Python 3.11+ (tested locally with the versions recorded in `requirements-lock.txt`):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,neural]'
pytest -q
```

`neural` is optional for tabular-only experiments, but both supplied configurations
include LSTM. CPU is the default, tree models use the `threads` value in the config,
and the package defaults native OpenMP pools to one thread. Full experiments
may require 16–32+ GB RAM and substantial time; use a dedicated machine or Kaggle
for larger runs. No training runs on import.

## Data

Download the official competition archive while signed in to Kaggle and extract
it under `data/raw/`. The local working copy already has the downloaded data.
On another machine use the competition's **Data → Download All → Download as zip**,
or an already configured official Kaggle CLI:

```bash
kaggle competitions download -c optiver-trading-at-the-close -p data/raw
unzip data/raw/optiver-trading-at-the-close.zip -d data/raw
```

Data access is subject to [competition rules](https://www.kaggle.com/competitions/optiver-trading-at-the-close/rules).
Raw data, predictions containing competition observations, credentials and binary
model artifacts are excluded from Git. Do not upload them to this repository.

## Small functional exercise

```bash
optiver-research --data data/raw/train.csv --config configs/quick.json --output runs/quick
```

This selects eight stocks and the last 20 training days, with small models and a
one-epoch LSTM. It checks the real-data path; it is **not** a competition-quality
benchmark. Cross-sectional features describe only those selected stocks.
Choose a fresh output directory each time: runs cannot silently overwrite results.

## Full research run

```bash
optiver-research --data data/raw/train.csv --config configs/full.json \
  --output runs/full-v3 --allow-full-run
```

The last 20 days are a terminal holdout; a one-day gap separates them from the
development data. Three expanding development folds each validate on 20 days,
also separated by one-day gaps. Hyperparameters in the configuration are fixed
candidates, not the result of a completed search. Tune only on development folds;
do not keep inspecting or optimizing the terminal holdout.

Every run saves config, selection metadata, per-fold MAE / RMSE / R², per-day
holdout metrics, zero-prediction baseline, paired day-bootstrap uncertainty,
diagnostic plots, source-data SHA-256, package versions and the evaluated model.
The model saved is the model actually evaluated, without a silent holdout refit.
The bootstrap assumes independent sampled days; it is not a profitability test.

## Streaming inference

```python
import joblib
from optiver_close.inference import StreamingPredictor

# Load only your own trusted artifacts; joblib can execute code while loading.
artifact = joblib.load('runs/full-v3/model.joblib')
predictor = StreamingPredictor(artifact)
prediction = predictor.predict(complete_current_timestamp_dataframe)
```

Each call must contain all available stocks for one timestamp in chronological
order. History is reset on a new day and kept for the current day only. Missing
stocks are allowed. Predictions retain the incoming row order. Kaggle integration
is in `scripts/kaggle_predict.py`; no submissions are made automatically.

## Results and limitations

- `reports/verification.json`: actual short test / real-data exercise status
- `docs/STUDY_REPORT.md`, `reports/study/`: research extension study (protocol, searches, ablations, identity, ensembles, statistics, final REUSED-holdout evaluation)
- `reports/full_run/`: full-run metrics, per-fold table, manifest, diagnostics, feature importance
- `reports/quick_metrics.json`: small-sample functional exercise only (not a benchmark)
- `docs/HISTORICAL_RESULTS.md`: historical public vs private score provenance
- `legacy/`: code-only exports of the original notebooks for comparison, not executable production code

MAE is prediction error, not trading profit. The project does not claim a profitable
strategy, executable alpha, a top rank, or new outperformance without evaluation.
Full-day offline transforms assume all stocks in each timestamp are available
together, consistent with Kaggle's batching. Prediction horizon and transaction
execution would require a separate trading study. Model choices can still overfit
development folds; regime and unseen-stock generalization need separate assessment.
The LSTM keeps its internal validation tail out of weight fitting, so it uses fewer
fitting days than the tabular models. Document this when comparing compute budgets.

## Sources

- [Competition and target definition](https://www.kaggle.com/competitions/optiver-trading-at-the-close/data)
- [Original XGBoost v4](https://www.kaggle.com/code/lidongmausa/xgboost-model?scriptVersionId=155544727)
- [Original XGBoost experiments](https://www.kaggle.com/code/lidongmausa/xgboost-2)
- [Original LSTM](https://www.kaggle.com/code/lidongmausa/lstm-model)
- [Original Random Forest](https://www.kaggle.com/code/lidongmausa/random-forest)
- [Original linear regression](https://www.kaggle.com/code/lidongmausa/linear-regression)
- [scikit-learn temporal validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)

The archived notebooks contain standard library / Kaggle environment boilerplate
and may incorporate community patterns. No claim of inventing the underlying
algorithms or exclusive ownership of third-party snippets is made. This repository
does not relicense competition data or third-party content.

