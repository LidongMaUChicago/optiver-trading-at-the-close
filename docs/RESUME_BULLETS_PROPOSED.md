# Proposed resume bullets (not yet applied to the resume/PDFs)

**Optiver – Trading at the Close (closing-auction return forecasting)**

1. Rebuilt a Kaggle closing-auction forecasting project (200 Nasdaq stocks, 5.2M order-book snapshots) into a tested, reproducible pipeline. Reproduced the prior full-history results within 0.0006 bp.
   - Evidence: `reports/study/results.json → repro_full_history`; `reports/full_run/`.

2. Designed a pre-registered, nested walk-forward protocol with day-gapped windows, frozen search spaces and a sealed holdout. Ran 440 fits across Ridge, random forest, XGBoost, LSTM and MLP, with 0 failures.
   - Evidence: `configs/study.json`; `reports/study/experiment_index.csv`.

3. XGBoost reduced out-of-sample MAE by 0.133 bp (2.2%) vs a zero baseline, and beat RF, LSTM and Ridge by 0.024–0.067 bp. The gain held on 100/100 validation days and 198/200 stocks. Significance was tested with Newey-West HAC, block bootstrap and Holm correction.
   - Evidence: `results.json → nested_vs_zero, confirmatory, error_analysis.concentration`.

4. Ran controlled ablations showing:
   - timing and lag features add value; cross-sectional features are redundant;
   - learned stock embeddings and LSTM sequence structure add nothing (5 seeds, p = 0.49);
   - models transfer to 40 unseen stocks with no meaningful loss.
   - Evidence: `ablation_xgboost.csv`; `results.json → identity, unseen_stocks`.

5. Found a parameter-free cross-sectional centring step that improved every validation window. The frozen model scored MAE 5.660 vs 5.790 for the zero baseline on a reused 20-day holdout. Verified exact batch/stream parity and the Kaggle API row alignment.
   - Evidence: `reports/study/final_holdout_REUSED.json`.

Keep separate: the historical 2023/2024 public MAE 4.7191 is a different evaluation and must not be combined with these numbers.
