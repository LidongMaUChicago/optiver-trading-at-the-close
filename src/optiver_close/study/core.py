"""One logged fit: train on a window's fitting days, predict that window, save.

Every fit appends one JSON line to runs/study/fits.jsonl (success OR failure)
and saves float32 predictions aligned with the window's evaluation index.
Existing predictions are reused, so an interrupted experiment can resume.
"""
from __future__ import annotations
import json
import platform
import resource
import time
import traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from ..models import build_model
from .protocol import CACHE, RUNS, WINDOWS, DEV_LAST_DAY, train_range


class Panel:
    """Memory-mapped development cache. Never contains terminal-holdout rows."""
    def __init__(self):
        meta = json.loads((CACHE / "meta.json").read_text())
        self.columns = meta["columns"]
        self.col_index = {c: i for i, c in enumerate(self.columns)}
        self.values = np.load(CACHE / "dev_features.npy", mmap_mode="r")
        self.keys = np.load(CACHE / "dev_keys.npy")
        self.target = np.load(CACHE / "dev_target.npy").astype(np.float64)
        if self.keys[:, 1].max() > DEV_LAST_DAY:
            raise RuntimeError("Development cache contains days after the development period")
        self.stock, self.day, self.second = self.keys[:, 0], self.keys[:, 1], self.keys[:, 2]

    def rows(self, first, last, include=None, exclude=None):
        mask = (self.day >= first) & (self.day <= last)
        if include is not None:
            mask &= np.isin(self.stock, include)
        if exclude is not None:
            mask &= ~np.isin(self.stock, exclude)
        return np.flatnonzero(mask)

    def frame(self, rows, columns=None):
        """Keys plus requested predictor columns, in cache (stock/day/second) order."""
        requested = columns or self.columns
        timing = "seconds_in_bucket" in requested     # dropped as a predictor in timing ablations
        columns = [c for c in requested if c not in ("stock_id", "date_id", "seconds_in_bucket")]
        idx = [self.col_index[c] for c in columns]
        data = np.asarray(self.values[rows][:, idx])
        frame = pd.DataFrame(data, columns=columns)
        frame.insert(0, "stock_id", self.stock[rows])
        frame.insert(1, "date_id", self.day[rows])
        if timing:
            frame.insert(2, "seconds_in_bucket", self.second[rows].astype(np.float32))
        return frame


def eval_rows(panel, window, stocks=None, tag="all"):
    """Saved once per window/evaluation subset so analysis can align predictions."""
    path = RUNS / "index" / f"{window}_{tag}.npy"
    if path.exists():
        return np.load(path)
    rows = panel.rows(*WINDOWS[window], include=stocks)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, rows)
    return rows


def _log(record):
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / "fits.jsonl", "a") as handle:
        handle.write(json.dumps(record, default=float) + "\n")


def _extras(model):
    out = {}
    inner = getattr(model, "best_iteration_", None)
    if inner is not None:
        out["best_iteration"] = inner
    if hasattr(model, "history"):
        out["epochs_run"] = len(model.history)
        out["inner_validation_mae"] = [round(h["inner_validation_mae"], 5) for h in model.history]
        out["nonfinite_steps"] = getattr(model, "nonfinite_steps", 0)
    return out


def run_fit(panel, experiment, candidate, model, params, window, *, seed=42, columns=None,
            exclude_stocks=None, eval_stocks=None, eval_tag="all", expanding=False, threads=10,
            keep_model=False):
    """Returns (predictions or None, fitted model if keep_model)."""
    pred_path = RUNS / "preds" / experiment / f"{candidate}__{window}__s{seed}.npy"
    if pred_path.exists() and not keep_model:
        return np.load(pred_path), None
    first, last = train_range(window, expanding)
    train = panel.rows(first, last, exclude=exclude_stocks)
    test = eval_rows(panel, window, eval_stocks, eval_tag)
    record = {"experiment": experiment, "candidate": candidate, "model": model, "params": params,
              "window": window, "seed": seed, "train_days": [first, last], "train_rows": len(train),
              "eval_rows": len(test), "eval_tag": eval_tag, "n_predictors": None,
              "excluded_stocks": len(exclude_stocks) if exclude_stocks is not None else 0,
              "host": platform.machine(), "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    t = time.monotonic()
    try:
        X = panel.frame(train, columns)
        record["n_predictors"] = X.shape[1] - 2
        estimator = build_model(model, X.columns, params, seed, threads)
        estimator.fit(X, panel.target[train])
        del X
        fit_seconds = time.monotonic() - t
        prediction = np.asarray(estimator.predict(panel.frame(test, columns)), dtype=np.float64)
        if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
            raise FloatingPointError("Non-finite or misaligned predictions")
        y = panel.target[test]
        record.update(status="ok", mae=float(np.abs(y - prediction).mean()),
                      zero_mae=float(np.abs(y).mean()), fit_seconds=fit_seconds,
                      seconds=time.monotonic() - t, **_extras(estimator))
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, prediction.astype(np.float32))
    except Exception as error:  # failures are retained in the log, not hidden
        record.update(status="failed", error=f"{type(error).__name__}: {error}",
                      trace=traceback.format_exc()[-1500:], seconds=time.monotonic() - t)
        prediction, estimator = None, None
    record["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    _log(record)
    print(f"{experiment} {candidate} {window} s{seed}: {record['status']} "
          f"mae={record.get('mae', float('nan')):.4f} {record['seconds']:.0f}s", flush=True)
    return prediction, (estimator if keep_model else None)
