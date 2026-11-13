"""Run fixed candidates on development folds, then open the terminal holdout once."""
from __future__ import annotations
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .data import load_data, engineer_panel, sha256
from .models import build_model
from .validation import split_days


def metrics(y, p):
    # float64 prevents the old float16 overflow in squared-error diagnostics.
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require nonempty, finite labels and predictions")
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))), "r2": float(r2_score(y, p)), "rows": len(y)}


def bootstrap_improvement(predictions, draws=2000, seed=42):
    """Paired resampling of whole holdout days; descriptive interval vs zero.

    This preserves within-day dependence but assumes independent sampled days.
    Do not interpret it as evidence of trading profitability.
    """
    d = predictions.copy()
    d["gain"] = d.target.abs() - (d.target - d.prediction).abs()
    days = d.groupby("date_id").gain.agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    sampled = days[rng.integers(0, len(days), size=(draws, len(days)))]
    gains = sampled[:, :, 0].sum(axis=1) / sampled[:, :, 1].sum(axis=1)
    return {"meaning": "MAE(zero) minus MAE(model), basis points",
            "daily_resampling_95pct_interval": np.quantile(gains, [0.025, 0.975]).tolist(),
            "days": len(days), "draws": draws}


def _plain(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=_plain) + "\n")


def run_experiment(csv_path, config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new run directory")
    output.mkdir(parents=True)
    save_json(output / "config.json", config)
    started = time.monotonic()
    raw = load_data(csv_path, **config.get("subset", {}))
    X = engineer_panel(raw)
    # Missing targets are removed only AFTER history features are built.
    ok = np.isfinite(raw.target)
    X, y = X.loc[ok].reset_index(drop=True), raw.loc[ok, "target"].to_numpy(dtype=np.float64)
    dev, holdout, folds = split_days(X.date_id, **config["split"])
    seed, threads = config.get("seed", 42), config.get("threads", 2)
    rows, models = [], config["models"]
    cv_predictions = []
    for fold_id, fold in enumerate(folds):
        train = X.date_id.isin(fold.train_days).to_numpy()
        val = X.date_id.isin(fold.validation_days).to_numpy()
        for candidate in models:
            name = candidate["name"]
            model = build_model(name, X.columns, candidate.get("params"), seed, threads)
            before = time.monotonic()
            print(f"fold={fold_id} model={name} train={train.sum()} validation={val.sum()}", flush=True)
            model.fit(X.loc[train], y[train])
            prediction = model.predict(X.loc[val])
            result = {"fold": fold_id, "model": name, **metrics(y[val], prediction),
                      "seconds": time.monotonic() - before, "train_last_day": int(max(fold.train_days)),
                      "validation_first_day": int(min(fold.validation_days)),
                      "validation_last_day": int(max(fold.validation_days))}
            rows.append(result)
            cv_predictions.append(pd.DataFrame({"model": name, "fold": fold_id,
                "date_id": X.loc[val, "date_id"].to_numpy(), "target": y[val], "prediction": prediction}))
            print(result, flush=True)
    cv = pd.concat(cv_predictions, ignore_index=True)
    scores = {name: metrics(g.target, g.prediction) for name, g in cv.groupby("model")}
    winner = min(scores, key=lambda name: scores[name]["mae"])
    selected = next(m for m in models if m["name"] == winner)
    # Seal selection metadata before reading holdout labels for evaluation.
    save_json(output / "selection.json", {"winner": winner, "development_scores": scores,
              "selection_rule": "lowest pooled development-fold MAE", "folds": [f.__dict__ for f in folds]})
    train, test = X.date_id.isin(dev).to_numpy(), X.date_id.isin(holdout).to_numpy()
    model = build_model(winner, X.columns, selected.get("params"), seed, threads)
    model.fit(X.loc[train], y[train])
    p = model.predict(X.loc[test])
    held = X.loc[test, ["stock_id", "date_id", "seconds_in_bucket"]].copy()
    held["target"], held["prediction"] = y[test], p
    held.to_csv(output / "holdout_predictions.csv", index=False)
    cv.to_csv(output / "development_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "fold_metrics.csv", index=False)
    # Persist the actually evaluated model, not a silent post-holdout refit.
    artifact = {"model": model, "name": winner, "feature_columns": list(X.columns),
                "feature_version": 1, "training_last_day": int(max(dev)), "config": config}
    joblib.dump(artifact, output / "model.joblib")
    by_day = [{"date_id": int(day), **metrics(g.target, g.prediction)} for day, g in held.groupby("date_id")]
    report = {"run_type": config["run_type"], "winner": winner, "development": scores,
              "holdout": metrics(y[test], p), "zero_baseline_holdout": metrics(y[test], np.zeros(test.sum())),
              "holdout_days": list(map(int, holdout)), "holdout_by_day": by_day,
              "paired_daily_bootstrap": bootstrap_improvement(held, seed=seed),
              "total_seconds": time.monotonic() - started,
              "historical_result": {"public_mae_rounded": 4.7, "public_mae_exact": 4.7191,
                  "date": "2024-01-16", "model": "original XGBoost v4", "new_run_result": False}}
    save_json(output / "metrics.json", report)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        revision = None
    save_json(output / "manifest.json", {"data_file": Path(csv_path).name, "sha256": sha256(csv_path),
        "source_rows": len(raw), "scored_rows": len(y), "features": len(X.columns), "git_revision": revision,
        "python": platform.python_version(), "packages": {n: version(n) for n in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]}})
    if winner != "lstm":
        est = model.named_steps["model"]
        if hasattr(est, "feature_importances_"):
            names = model.named_steps["preprocess"].get_feature_names_out()
            pd.DataFrame({"feature": names, "importance": est.feature_importances_}).sort_values(
                "importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    from .report import plot_report
    plot_report(held, output)
    return report

