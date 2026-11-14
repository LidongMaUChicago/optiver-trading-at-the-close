# Interview narrative (English, spoken style)

## Two-minute version

"This is the Optiver closing-auction problem from Kaggle. For each Nasdaq stock, every ten seconds in the last ten minutes before the close, you predict how its price moves over the next sixty seconds relative to an index. The error is measured in basis points.

I first did it in 2023 as notebooks. In 2026 I rebuilt it as a proper research project.

The first thing I did was freeze the rules before running anything: the validation windows, a one-day gap, the search spaces, the seeds, the statistical tests and how the final model gets picked. Then I ran a nested time-series evaluation. Parameters are picked on earlier windows and scored on a later window. That gives an honest out-of-sample number.

I compared Ridge, random forest, XGBoost, an LSTM and an MLP, about 440 fits in total. XGBoost won. It beats predicting zero by about 0.13 basis points, roughly two percent of the error, and it beats the other models with tight confidence intervals. The gain is very broad: it's better on every one of the 100 validation days and on 198 of 200 stocks, and it's biggest when spreads are wide.

The more interesting part is what didn't work. Stock embeddings didn't help; I checked that with five seeds and it's basically zero. The LSTM's sequence structure didn't help either, because the lag features already capture the recent path. Blending models didn't pay off. What did work was a tiny trick: subtract the average prediction across stocks at each timestamp. It has no parameters and it was better in every window, so it went into the final model.

On the final 20 days the error was 5.66 versus 5.79 for predicting zero. Those days had been used once before, so I label that result as reused, not a clean test."

## Follow-up questions

**Q: Why not just use a random train/test split?**
A: The data is a time series, and stocks on the same day move together. A random split leaks the future and makes the model look better than it is. So I split by whole days, in time order, with a one-day gap so the targets don't overlap.

**Q: How did you make sure tuning didn't leak into your scores?**
A: Nested evaluation. For each outer window, the configuration is chosen only on earlier windows. The selection rule was written down before I saw any result. The last 20 days were never used for any choice.

**Q: Your improvement is 0.13 basis points. Is that real?**
A: Yes, but it's small. I didn't treat 5 million rows as independent; the unit is the day. I took the daily loss difference and used a Newey-West standard error and a block bootstrap, and I corrected for multiple tests with Holm. The model is better on all 60 outer days, and the confidence interval is about 0.12 to 0.15. But I don't claim it makes money. That would need an execution study with costs.

**Q: Why did XGBoost beat the LSTM?**
A: Most of the signal is in the current order book and in engineered changes over the last few steps. Once those lags are features, the sequence adds nothing. A one-step LSTM was actually the best LSTM, and an MLP did just as well. Trees handle the nonlinear interactions between spread, imbalance and auction phase very efficiently.

**Q: What was the most important feature, and did you check it?**
A: WAP versus mid price. I checked the formula, and it's exactly half the spread times the book imbalance. It only uses the current row, so it can't see the future. Its correlation with the target is negative, which makes sense: when the book leans one way, the price has already moved that way and partly reverts.

**Q: Feature importance said cross-sectional features matter. Your ablation says they don't. Which is right?**
A: Both, they answer different questions. Permutation importance says the fitted model uses them. Leaving the group out and refitting says the information is redundant, because other features carry the same signal. For "does this group add value", the refit ablation is the right test.

**Q: Did stock embeddings help?**
A: No. I kept everything else fixed: same features, same days, same early stopping. I compared no identity, one-hot, and embeddings of size 2 to 16, with five seeds on the key comparison. The difference was minus 0.0006, with a p-value around 0.5, and the sign flipped between seeds. Also, when I held out 40 stocks completely, the models did almost as well on those stocks. So the signal is market structure, not the stock's identity.

**Q: Why does subtracting the cross-sectional mean help?**
A: The target is relative to an index made of these same stocks, so across stocks it should average out near zero. Equal-weight centring is only an approximation, because the real index weights aren't public. I didn't assume it works. I tested it on earlier windows first, then confirmed it on later windows.

**Q: What would you do with more compute?**
A: Rerun the study with full history instead of 100 days, add more seeds for the unseen-stock test, and get a truly fresh test period. Then do a trading study: what this prediction is worth after spread and impact.

**Q: How do you know the production code matches the research code?**
A: I saved the model, reloaded it, and fed it one timestamp at a time like Kaggle does. The predictions matched the batch predictions exactly. I also ran a local copy of Kaggle's API loop on their example files and checked that every row ID matched. There are 28 tests, including no-lookahead and no-cross-day leakage.

**Q: What's the biggest limitation?**
A: Two things. The training window is 100 days because I ran it on a laptop. And the last 20 days were already used once, so there is no truly fresh test. I say that clearly in the report.
