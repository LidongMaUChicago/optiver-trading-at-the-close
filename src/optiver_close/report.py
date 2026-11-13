"""Diagnostic plots; no fabricated returns or leaderboard claims."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_report(predictions, output):
    d = predictions.assign(error=lambda x: (x.target - x.prediction).abs(), baseline=lambda x: x.target.abs())
    daily = d.groupby("date_id")[["error", "baseline"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    daily.rename(columns={"error": "Model", "baseline": "Zero baseline"}).plot(ax=axes[0])
    axes[0].set(title="Holdout MAE by day", ylabel="MAE (basis points)")
    (d.target - d.prediction).clip(-30, 30).hist(bins=50, ax=axes[1])
    axes[1].set(title="Residuals (display clipped at ±30 bp)", xlabel="Target minus prediction (bp)")
    fig.tight_layout(); fig.savefig(Path(output) / "diagnostics.png", dpi=160); plt.close(fig)

