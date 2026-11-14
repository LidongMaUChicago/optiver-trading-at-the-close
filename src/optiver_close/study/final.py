"""Freeze, retrain and evaluate the final candidate ONCE on the REUSED holdout.

    python -m optiver_close.study.final --data data/raw/train.csv

Reads the selection from reports/study/results.json (decided on development
windows only). Refuses to run twice. Then: save/reload parity, batch-vs-stream
parity on raw holdout days (stock/day transitions, unscored warmup), latency /
memory benchmark, and a local run of the Kaggle API adapter on Kaggle's example
test files. Nothing is uploaded or submitted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import resource
import subprocess
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ..data import load_data
from ..experiment import metrics
from ..inference import StreamingPredictor, run_kaggle
from ..models import build_model, Blend, CrossSectionCentered
from .core import Panel
from .protocol import CACHE, CONFIG, ROOT, RUNS, HOLDOUT, train_range
from .stats import compare

OUT = ROOT / "reports" / "study"
FINAL = RUNS / "final"


def member_spec(key):
    family, cand = key.split(":")
    if family == "blend":
        raise ValueError
    s = CONFIG["search"]
    if family == "ridge":
        return "ridge", {"alpha": float(cand.split("_a")[1])}
    params = next(p for p in s[family] if p["id"] == cand)
    params = {k: v for k, v in params.items() if k != "id"}
    if family in ("lstm", "mlp"):
        common = {k: v for k, v in s["neural_common"].items() if k != "note"}
        params = {**common, "device": "cpu", **params}
    return family, params


def build_final(results):
    winner = results["final_selection"]["winner"]
    if winner.startswith("blend:"):
        name = winner.split(":")[1]
        members = results["ensembles_all5"]["members"]
        use = {"equal_4": ["ridge", "random_forest", "xgboost", "lstm"], "equal_xgb_lstm": ["xgboost", "lstm"],
               "equal_xgb_rf_lstm": ["xgboost", "random_forest", "lstm"]}[name]
        specs = [member_spec(f"{f}:{members[f]}") for f in use]
    else:
        specs = [member_spec(winner)]
    return winner, specs


def env_lock():
    return subprocess.check_output([str(ROOT / ".venv" / "bin" / "python"), "-m", "pip", "freeze"], text=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--threads", type=int, default=8)
    args = a.parse_args()
    report_path = OUT / "final_holdout_REUSED.json"
    if report_path.exists():
        raise SystemExit("Final holdout evaluation already done; refusing to evaluate the holdout again.")
    results = json.loads((OUT / "results.json").read_text())
    winner, specs = build_final(results)
    FINAL.mkdir(parents=True, exist_ok=True)
    p = Panel()
    first, last = train_range(HOLDOUT[0])
    rows = p.rows(first, last)
    X = p.frame(rows)
    t = time.monotonic()
    models = [build_model(name, X.columns, params, 42, args.threads) for name, params in specs]
    model = models[0] if len(models) == 1 else Blend(models)
    calibration = results["final_selection"].get("calibration")
    if calibration == "cross_sectional_centering":
        model = CrossSectionCentered(model)
        winner = f"{winner}+centered"
    model.fit(X, p.target[rows])
    fit_seconds = time.monotonic() - t
    needs_history = any(name in ("lstm", "mlp") for name, _ in specs)
    artifact = {"model": model, "name": winner, "feature_columns": list(X.columns), "feature_version": 1,
                "needs_history": needs_history, "training_days": [first, last], "specs": specs,
                "protocol": "configs/study.json", "selected_on": "development windows W1-W5 only"}
    path = FINAL / "model.joblib"
    joblib.dump(artifact, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (FINAL / "requirements-freeze.txt").write_text(env_lock())
    # ---- the single reused-holdout evaluation ----
    hk = np.load(CACHE / "holdout_keys.npy")
    hv = np.load(CACHE / "holdout_features.npy")
    hold = pd.DataFrame(hv, columns=p.columns)
    hold[["stock_id", "date_id", "seconds_in_bucket"]] = hk
    hold = hold.astype({"stock_id": "int32", "date_id": "int32"})
    batch_pred = np.asarray(model.predict(hold[artifact["feature_columns"]]), dtype=np.float64)
    y = np.load(CACHE / "holdout_target_SEALED.npy").astype(np.float64)
    ok = np.isfinite(y)
    days = hk[:, 1]
    by_day = pd.DataFrame({"day": days[ok], "m": np.abs(y[ok] - batch_pred[ok]), "z": np.abs(y[ok])}).groupby("day").mean()
    report = {"status": "REUSED terminal holdout (days 461-480 were already evaluated in full-v3); evaluated once here for the frozen candidate; not a pristine test",
              "winner": winner, "specs": specs, "training_days": [first, last], "fit_seconds": fit_seconds,
              "holdout": metrics(y[ok], batch_pred[ok]), "zero_baseline": metrics(y[ok], np.zeros(ok.sum())),
              "paired_vs_zero": compare(by_day.z, by_day.m),
              "full_v3_reference": {"model": "xgboost full history (GPU)", "holdout_mae": 5.6687, "zero_mae": 5.7904},
              "artifact": {"path": str(path.relative_to(ROOT)), "sha256": digest, "bytes": path.stat().st_size}}
    # ---- save / reload parity ----
    reloaded = joblib.load(path)
    rp = np.asarray(reloaded["model"].predict(hold[artifact["feature_columns"]]), dtype=np.float64)
    report["reload_max_abs_diff"] = float(np.abs(rp - batch_pred).max())
    # ---- streaming parity, latency, memory on raw holdout days ----
    raw = load_data(args.data, last_days=None)
    raw = raw[raw.date_id.isin([HOLDOUT[0], HOLDOUT[0] + 1])].reset_index(drop=True)
    predictor = StreamingPredictor(reloaded)
    stream, lat = [], []
    for (d, s), g in raw.groupby(["date_id", "seconds_in_bucket"], sort=True):
        t0 = time.perf_counter()
        out = predictor.predict(g.drop(columns="target"))
        lat.append(time.perf_counter() - t0)
        stream.append(pd.Series(out, index=g.index))
    stream = pd.concat(stream).sort_index().to_numpy()
    ref = pd.Series(batch_pred, index=pd.MultiIndex.from_arrays([hk[:, 0], hk[:, 1], hk[:, 2]]))
    expected = ref.reindex(pd.MultiIndex.from_arrays([raw.stock_id, raw.date_id, raw.seconds_in_bucket])).to_numpy()
    report["stream_vs_batch"] = {"days": [HOLDOUT[0], HOLDOUT[0] + 1], "rows": len(raw),
                                 "max_abs_diff": float(np.nanmax(np.abs(stream - expected))),
                                 "timestamps": len(lat)}
    lat = np.array(lat) * 1000
    report["latency_ms_per_timestamp_batch"] = {"mean": float(lat.mean()), "p95": float(np.quantile(lat, 0.95)),
                                                "max": float(lat.max()), "stocks_per_batch": int(raw.groupby(["date_id", "seconds_in_bucket"]).size().median()),
                                                "note": "includes feature engineering on the current day's history; Kaggle limits are per full test run"}
    report["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    # ---- Kaggle API semantics on the example test files (local, no submission) ----
    report["kaggle_local_api"] = kaggle_local(reloaded)
    report["host"] = {"machine": platform.machine(), "python": platform.python_version()}
    report_path.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float)[:4000])


class LocalEnv:
    """Mirrors optiver2023 iter_test/predict: (test, revealed_targets, sample_prediction) per time_id."""
    def __init__(self, folder):
        self.test = pd.read_csv(folder / "test.csv")
        self.revealed = pd.read_csv(folder / "revealed_targets.csv")
        self.sample = pd.read_csv(folder / "sample_submission.csv")
        self.status, self.calls = "ready", []

    def iter_test(self):
        for time_id in self.test.time_id.drop_duplicates():
            self.status = "prediction_needed"
            yield (self.test[self.test.time_id == time_id].reset_index(drop=True),
                   self.revealed[self.revealed.time_id == time_id].reset_index(drop=True),
                   self.sample[self.sample.time_id == time_id].reset_index(drop=True))
            if self.status != "prediction_received":
                raise RuntimeError("predict() not called exactly once for this batch")

    def predict(self, frame):
        if self.status != "prediction_needed":
            raise RuntimeError("predict() called twice or out of order")
        self.calls.append(frame); self.status = "prediction_received"


def kaggle_local(artifact):
    env = LocalEnv(ROOT / "data" / "raw" / "example_test_files")
    t = time.monotonic()
    run_kaggle(artifact, env)
    sub = pd.concat(env.calls)
    return {"batches": len(env.calls), "rows": len(sub), "unique_row_ids": int(sub.row_id.nunique()),
            "row_ids_match_sample": bool(set(sub.row_id) == set(env.sample.row_id)),
            "finite": bool(np.isfinite(sub.target).all()), "seconds": time.monotonic() - t,
            "scored_rows": int(env.test.currently_scored.sum()),
            "note": "local re-implementation of the API protocol on Kaggle's example files; the real Kaggle notebook run was not performed"}


if __name__ == "__main__":
    main()
