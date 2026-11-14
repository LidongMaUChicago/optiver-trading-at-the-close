# Research extension study (local-100d), November 2026

All numbers here are measured. Each one points to a file under `reports/study/`. MAE is in basis points (bp). This is a **local, scaled-down** study on an Apple M4 laptop (10 cores, 16 GB, CPU only). Every model is trained on the 100 development days just before its validation window. It is not a full-history study and not a Kaggle leaderboard result. Keep it separate from:

- the historical 2023/2024 public MAE 4.7191 (`docs/HISTORICAL_RESULTS.md`);
- the 2026 full-v3 run (`reports/full_run/`).

## 1. Problem and data

- **Target.** Kaggle "Optiver – Trading at the Close". For each stock and each 10-second step of the 10-minute closing auction, predict the next-60-second WAP move relative to Optiver's synthetic index, in bp.
- **Data.** `train.csv`: 200 stocks, days 0–480, 5,237,980 rows. sha256 `d7ca1efc…8f14`, recorded in `configs/study.json`.
- **Features.** 113 engineered columns (feature version 1). Two of them are keys (`stock_id`, `date_id`), so 111 are predictors. Every feature uses only the current row, earlier rows of the same stock and day, or the same timestamp's cross-section. All of these are available at prediction time under Kaggle's per-timestamp batching. Nothing crosses a day boundary.
- **Feature cache.** Built once in `src/optiver_close/study/cache.py`.
  - Development rows: days 0–459. 5,006,892 rows (88 rows with a missing target were dropped).
  - Terminal-holdout rows: days 461–480, 220,000 rows. Their labels are stored in a separate `*_SEALED.npy` file that the development code never opens. `Panel` asserts that no row is later than day 459.

## 2. Protocol, frozen before any result

`configs/study.json` (declared 2026-11-13, before any study result) fixes:

- **Windows.** Five 20-day validation windows: W1 360–379, W2 380–399, W3 400–419, W4 420–439, W5 440–459.
- **Training data.** Each fit uses the 100 days `[start−101, start−2]`, leaving a one-day gap. Learned transforms are refit on those dates only.
- **Nested evaluation.**
  - For outer window Wk (k = 3, 4, 5), each model family picks its configuration by lowest mean MAE on W1…W(k−1). The pick is then scored on Wk.
  - "Outer MAE" is the mean over W3–W5. It is the honest performance estimate.
  - Means over W1–W5 are also reported. They are used only by the final selection rule (§9).
- **Terminal holdout 461–480 (gap day 460).** These days were already opened once by full-v3, so they are labelled **REUSED**. They were evaluated once more, for the frozen final model only, and never used for any choice.
- **Also declared in advance:** seeds (42, plus 1–4 for repeats), the feature-group registry, the search spaces, the stock holdout (40 stocks, `default_rng(2026)`), the statistical tests, the regime definitions and the final rule.
- **Pre-freeze runs.** Four timing benchmarks were fitted on W1 before the protocol was frozen. They were deleted and not used.

**Why a scaled-down version?** A full expanding-history study needs a paid GPU instance, and the user chose local compute. The 100-day rolling window keeps all 200 stocks and complete cross-sections. Section 3 checks that the scaling does not distort the reference numbers.

**Compute.**
- 440 fits, 0 failures, 12.5 compute-hours. Peak RSS 6.2 GB.
- Two queues ran at once: trees with 6 threads, neural with 4 torch threads.
- The LSTM ran on CPU, because MPS was 2.6× slower for this model.
- Every fit (model, params, window, seed, rows, runtime, memory, status, prediction path) is listed in `reports/study/experiment_index.csv`.

## 3. Reproduction of full-v3

Full expanding history, same folds as full-v3 (`results.json → repro_full_history`):

| | W3 (400–419) | W4 (420–439) | W5 (440–459) |
|---|---|---|---|
| Ridge α=100, this study | 6.4579 | 5.5898 | 5.9703 |
| Ridge, full-v3 | 6.458 | 5.590 | 5.970 |
| XGBoost v3, CPU (this study) | 6.3676 | 5.5084 | 5.9104 |
| XGBoost v3, GPU (full-v3) | 6.3672 | 5.5084 | 5.9098 |

Ridge reproduces exactly. CPU XGBoost differs from the GPU run by at most 0.0006 bp, which is the expected size of GPU/CPU histogram differences. The pipeline is reproducible.

## 4. Baselines and model families

**Baselines** (mean W1–W5, from `baselines`):

| Baseline | MAE |
|---|---|
| Zero | 6.2297 |
| Training global median | 6.2297 |
| Training stock median | 6.2336 |

The per-stock median is slightly worse than zero, so a stock's level carries no signal.

**Searches.** 64 configurations (`search_results.csv`), each on all five windows:

| Family | Configurations | Range of mean W1–W5 MAE | Best |
|---|---|---|---|
| Ridge, α = 1 … 1e6 | 7 | 6.1649 – 6.1712 | α = 1e4 |
| Random Forest: depth, leaf, features, bootstrap fraction | 6 | 6.1224 – 6.1414 | rf5_d16 |
| XGBoost: depth/leaves, learning rate/trees, min_child_weight, sampling, L1/L2, loss, early stopping | 14 | 6.0964 – 6.1167 | x04_slow (lr 0.02, 1200 trees) |
| LSTM: hidden size, learning rate, batch size, weight decay/dropout, sequence length | 8 | 6.1391 – 6.1454 | l6_reg |
| MLP, last step only | 2 | 6.1313 – 6.1371 | m2_h128 |

- **XGBoost tuning barely matters.** 11 of the 14 configurations lie within 0.005 bp of each other. The exception is the loss: squared error (x12_l2) is 0.020 bp worse than the MAE objective, which matches the evaluation metric. Pseudo-Huber is close to MAE.
- **Early stopping.** Early stopping on the last 15% of the fitting days (x13/x14) is about 0.007 bp worse than the same settings with fixed rounds. The held-out tail costs more training data than early stopping saves.
- **Seed noise.** Five seeds of x01_v3 have SD 0.00014 bp. Refitting with the same seed gives a maximum prediction difference of 0.0.
- **RF bootstrap fraction** (full-v3 used max_samples 0.1 as a compute choice):
  - 0.1 vs 0.3, all windows: 6.1317 vs 6.1321;
  - 0.6, on W3 only: 6.4148 vs 6.4110.
  - Larger bootstrap samples do not help, so the 0.1 setting was not hurting RF.
- **Neural training.** All neural models used:
  - early stopping on the last 15% of fitting days (never on outer windows);
  - gradient clipping at 1.0;
  - a non-finite-step counter. It recorded 0 non-finite steps in every fit.

**Nested outer evaluation** (W3–W5, configuration chosen on earlier windows only; `nested`):

| Family | Outer MAE | Gain vs zero (6.0658) |
|---|---|---|
| **XGBoost** | **5.9332** | **0.1326 (2.2%)** |
| Random Forest | 5.9568 | 0.1090 |
| MLP | 5.9689 | 0.0968 |
| LSTM | 5.9759 | 0.0899 |
| Ridge | 6.0000 | 0.0658 |

The ranking XGBoost < RF < LSTM < Ridge is the same as in full-v3.

## 5. Feature-group ablation

**Groups.** The eight groups partition the 111 predictors exactly; a unit test checks this. They are:

| Group | Columns |
|---|---|
| raw_book + raw_auction ("raw") | 11 |
| timing | seconds_in_bucket and phase |
| liquidity_imbalance | — |
| auction_derived | — |
| lags_diffs | `_lagN` / `_diffN` |
| rolling | `_meanN` / `_stdN` |
| cross_section | `market_*`, `cross_section_relative_*` |

**Setup.** XGBoost v3 configuration, identical rows and windows (`ablation_xgboost.csv`). Values are the MAE change vs all features (6.0979, mean W1–W5); positive means worse.

| Step | Additive (features up to this group) | Leave this group out |
|---|---|---|
| raw only | +0.0573 | — |
| + timing | +0.0462 | +0.0123 |
| + liquidity/imbalance | +0.0374 | +0.0007 |
| + auction-derived | +0.0213 | +0.0017 |
| + lags/diffs | +0.0006 | +0.0093 |
| + rolling | −0.0003 | +0.0018 |
| + cross-section (= all) | 0 | −0.0003 |

**What the ablation shows:**
- Most of the gain comes from timing, liquidity/imbalance, auction-derived and lag features.
- **Timing** is the most important group to keep. Removing it costs 0.0123 bp, and all features beat the no-timing model on 90% of the 100 days (Holm p < 1e-33).
- **The cross-section group adds nothing** once the other groups are present.

**Permutation importance** (each group permuted within each day on the validation window; MAE increase):

| Group | MAE increase |
|---|---|
| lags | 0.078 |
| cross_section | 0.076 |
| liquidity | 0.060 |
| timing | 0.044 |
| auction | 0.035 |
| rolling | 0.032 |
| raw | 0.005 |

- **Reading the two together.** The fitted model *uses* the cross-section features heavily, but its information is redundant: other groups carry the same signal. This is why gain or permutation importance cannot be read as incremental value.
- **Limitation.** Permuting within a day breaks the link between correlated groups and creates unrealistic rows. That inflates importance for redundant groups.

**Ridge ablation** (`ablation_ridge.csv`). Only lags matter noticeably (leave-one-out +0.0049). This suggests most of XGBoost's advantage comes from nonlinear interactions, not from extra linear signal.

**`wap_vs_mid_price` audit** (the top full-v3 gain feature; `feature_audit_wap_vs_mid_price`):
- Formula: (wap − mid)/mid·1e4.
- Here WAP is the size-weighted microprice (median relative difference 3e-7).
- Algebraically, the feature equals half-spread × book imbalance: correlation 0.999996, median difference 0.003 bp.
- It uses same-row inputs only, so it is causal.
- 1e-5 of values are non-finite (zero sizes); the models handle these as missing.
- Its correlation with the target is −0.153. The sign is plausible: when the book leans to one side, the WAP is already pulled that way, and the index-relative move partly reverts.

## 6. Stock identity, sequences and the MLP

**Controlled comparison.** Same architecture, features, fitting dates, early-stopping period and rows; only the identity mechanism changes. Results on W3–W5 (`identity`):

| LSTM variant (l1_base, seq 12) | Outer MAE |
|---|---|
| No identity | 5.9764 |
| Embedding dim 2 / 4 / 8 / 16 | 5.9760 / 5.9740 / 5.9780 / 5.9819 |
| One-hot input | 5.9849 |

**Five seeds.** Embedding (dim 8) vs no identity, across 5 seeds and 60 days:
- No identity is better by 0.0006 bp on average.
- Newey-West p = 0.49; block-bootstrap 95% CI [−0.0024, 0.0009].
- The per-seed sign flips 3 to 2.
- **Conclusion: learned stock embeddings do not help.** This is a negative result.

**Sequence length** (W3–W5):

| Length | 1 | 3 | 6 | 12 | 24 |
|---|---|---|---|---|---|
| Outer MAE | **5.9681** | 5.9759 | 5.9760 | 5.9780 | 5.9752 |

- A one-step "LSTM" is best.
- The MLP (no identity) gets 5.9678 and beats the sequence LSTM.
- The engineered lag, diff and rolling features already summarize the recent path, so recurrence adds nothing. The useful part of the neural model is a per-row nonlinear map.

**Unseen stocks.** 40 stocks are removed from training and scored only on those stocks, on W3–W5 (`unseen_stocks`). Unknown ids are mapped to a reserved embedding index.

| Model | Unseen MAE | Same rows, stocks seen in training |
|---|---|---|
| Zero | 5.6562 | — |
| Stock median (falls back to global) | 5.6572 | — |
| Ridge | 5.6097 | 5.6000 |
| LSTM, no identity | 5.5733 | — |
| LSTM, embedding | 5.5674 | 5.5741 |
| XGBoost | 5.5357 | 5.5307 |

The models transfer to new stocks almost without loss, for example XGBoost +0.005 bp. The signal is market microstructure, not stock-specific. These are single-seed figures.

## 7. Ensembles and calibration

**Setup.** Member configurations, weights and calibration are chosen on earlier windows only, then frozen for the outer window (`ensembles_nested`):

| Variant | Outer MAE (W3–W5) |
|---|---|
| XGBoost alone | 5.9332 |
| Equal weight, 4 families | 5.9423 |
| Equal weight, XGB + LSTM | 5.9396 |
| Non-negative learned weights (≈0.8–0.9 XGB) | 5.9316 |
| XGB + learned bias | 5.9332 |
| **XGB, cross-sectionally centred** | **5.9281** |

- **Equal-weight blends are worse** than XGBoost alone. The weaker members dilute it.
- **Learned weights** gain 0.0016 bp, which does not justify running four models at inference.
- **Cross-sectional centring** subtracts each timestamp's mean prediction from every stock. It has no parameters.
  - It is better in each of W3, W4 and W5.
  - Its gain on the earlier windows was positive at every step.
  - It is also better on all five windows (6.0924 vs 6.0964).
- **Why centring helps.** The target is measured relative to an index of the same stocks, so its weighted cross-sectional mean is close to zero. The index weights are not public, so equal-weight centring only approximates this. It is not assumed to be exact.

## 8. Statistics

**Method.** The unit is the paired daily mean loss difference, so whole cross-sections stay together. On each comparison we report:
- the mean difference;
- the share of days that improve;
- a Newey-West HAC standard error (lag 5, because targets overlap and days are dependent);
- an iid-day bootstrap and a moving-block bootstrap (block lengths 2, 5 and 10).

Holm correction covers the declared confirmatory family; everything else is exploratory. Rows are never treated as independent.

**Confirmatory family** (`confirmatory`; outer W3–W5, 60 days, except the timing ablation, 100 days):

| Comparison | Mean gain (bp) | Block-bootstrap(5) 95% CI | Days better | Holm p |
|---|---|---|---|---|
| XGBoost vs zero | 0.1326 | [0.118, 0.147] | 100% | 2e-67 |
| XGBoost vs Ridge | 0.0668 | [0.058, 0.076] | 98% | 4e-46 |
| XGBoost vs LSTM | 0.0427 | [0.039, 0.047] | 100% | 1e-83 |
| XGBoost vs RF | 0.0236 | [0.018, 0.028] | 87% | 6e-20 |
| All features vs no timing | 0.0123 | [0.010, 0.014] | 90% | 7e-34 |
| LSTM embedding vs no identity | −0.0006 | [−0.0024, 0.0009] | 53% | 0.49 |

**How to read these.** The differences are real but small: 0.13 bp is about 2% of MAE. Searches across many configurations mean the selection-stage W1–W5 numbers are slightly optimistic; the nested outer numbers are not. **No profitability claim is made.** Turning MAE gains into P&L would need a separate execution study.

## 9. Stability and error analysis

**Scope.** These slices are descriptive and post hoc, except the regime and period definitions, which were declared in advance. Final-candidate configuration x04_slow, W1–W5, 1.1M rows (`error_analysis`, `error_analysis.png`). Gains are vs zero.

- **Auction phase:**

  | Phase | Gain (bp) |
  |---|---|
  | 0–290 s | 0.144 |
  | 300–470 s | 0.119 |
  | 480–540 s | 0.127 |

- **Spread quintile:** the gain rises from 0.032 bp (tightest) to 0.358 bp (widest). The model earns most when the book is wide and noisy.
- **Imbalance-pressure quintiles:** gains of 0.115–0.129 bp in the first four, 0.183 bp in the top quintile.
- **Far price missing:** 0.144 bp when missing vs 0.121 when present.
- **Volatility regime** (declared rule): 0.165 bp on 26 high-vol days vs 0.122 on 74 low-vol days.
- **Period:** 0.134 bp in W1–W2 vs 0.119 in W4–W5. A slight decay, but the gain is positive in every window (0.113–0.160).
- **Concentration:** the model beats zero on **100/100 days** and **198/200 stocks**. The best 5 days give 8% of the total gain. The worst stocks are about −0.002 bp, which is effectively zero.
- **Residuals:**
  - mean bias 0.02 bp;
  - prediction std 1.69 vs target std 9.24 (strong shrinkage, as expected under MAE loss);
  - the top 1% of rows by |error| hold 7% of total absolute error;
  - tail rows (|target| > p99) still gain 0.98 bp.

## 10. Final model and the REUSED holdout

**Selection rule** (declared): lowest mean W1–W5 MAE among single configurations and equal-weight blends. A learned or calibrated variant is allowed only if it beat equal weighting in nested outer evaluation.

**Result.** The winner is XGBoost x04_slow (6.0964). Cross-sectional centring passed the calibration gate (`final_selection`). The final model, **x04_slow + centring**, was trained on days 360–459 and evaluated **once** on days 461–480 (`final_holdout_REUSED.json`):

| | MAE | RMSE | R² |
|---|---|---|---|
| **Final model (100 training days)** | **5.6602** | 8.550 | 0.038 |
| Zero | 5.7904 | 8.720 | 0.000 |
| full-v3 XGBoost (all history, GPU), for reference | 5.6687 | 8.561 | 0.036 |

- **Gain vs zero:** 0.1302 bp. Block-bootstrap(5) 95% CI [0.119, 0.142]. Better on 20/20 days.
- **Caveat.** These days were seen before (by full-v3), so this is not a pristine test. No truly unused later data exist in `train.csv`, and **there is no fresh external test**.
- **Comparison with full-v3** (5.6602 vs 5.6687) is descriptive. It suggests that 100 recent days plus centring is at least as good as all history. It is not a confirmed improvement.

**Artifact and inference checks:**
- **Artifact.** `runs/study/final/model.joblib` (private, git-ignored), 5.5 MB, sha256 `0cf84a4a…d659`. It stores the schema, feature version, training-day boundary and a `requirements-freeze.txt` environment lock.
- **Save → reload:** maximum prediction difference 0.0.
- **Streaming vs batch:** on raw days 461–462 (110 timestamps, day transition, warmup), fed one timestamp at a time, maximum difference 0.0.
- **Latency:** with feature engineering, per 200-stock timestamp, mean 115 ms, p95 172 ms. Peak RSS 2.5 GB.
- **Local Kaggle API protocol test** on Kaggle's `example_test_files`:
  - 165 batches, 33,000 rows;
  - row_ids match `sample_submission` exactly;
  - all predictions finite;
  - `predict` called exactly once per batch.
- **The notebook run on Kaggle itself has not been done yet** (see `CODEX_HANDOFF.md`).

## 11. Tests and audit

`pytest`: 28 passed. The tests include:
- `tests/test_study.py`: group partition, split gaps, holdout exclusion, timing ablation, stock-median fallback, early stopping inside the fitting period, LSTM identity modes with unseen ids, MLP, stats helpers, cache guard and centring;
- earlier tests: OpenMP import order, chunked-Ridge exactness, future/target cannot change past features, no cross-stock or cross-day leakage, missing stocks, duplicate or out-of-order batches rejected, zero depth treated as missing, purge gaps, Kaggle row alignment, and save/stream parity.

**Approximations to document:**
- XGBoost runs on CPU here and on GPU in full-v3 (difference ≤ 0.0006 bp).
- All LSTM/MLP fits use CPU.
- The training window was scaled from full history to 100 days.
- Accelerate matmul warnings on macOS are spurious.

## 12. Conclusions

1. On rolling 100-day training, gradient-boosted trees (XGBoost) beat RF, LSTM, MLP and Ridge in nested, out-of-sample evaluation. The confirmatory statistics hold up under HAC and block bootstrap. The ranking reproduces full-v3.
2. The edge over zero is about 2% of MAE. It is broad-based: every day, 198/200 stocks and every phase. It is largest when spreads are wide.
3. Hyperparameter tuning matters far less than the loss function and the feature groups. Timing and lag features carry incremental value; cross-sectional features are redundant.
4. Negative results:
   - stock embeddings do not help, under controlled tests over 5 seeds;
   - sequence modelling adds nothing beyond the engineered lags;
   - blends do not justify their cost.
5. One parameter-free post-processing step, cross-sectional centring, gives a small, consistent gain and is part of the final model.
6. Limitations:
   - scaled-down training windows;
   - the holdout is REUSED;
   - single-seed unseen-stock tests;
   - MAE is not profit.
