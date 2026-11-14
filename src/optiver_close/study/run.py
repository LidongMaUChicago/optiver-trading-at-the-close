"""Development-only experiment runner. NEVER reads terminal-holdout data.

    python -m optiver_close.study.run <queue> [--threads N]

Queues: trees (baselines, ridge/xgboost/RF searches, ablations, permutation,
unseen-stock tabular models), neural (LSTM/MLP search, identity, sequence
length, seeds, unseen-stock neural), repro (full-history check of full-v3).
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
from .core import Panel, run_fit, eval_rows
from .protocol import CONFIG, RUNS, OUTER, WINDOWS, feature_sets, feature_groups, holdout_stocks

ALL = list(WINDOWS)
S = CONFIG["search"]


def _neural(params):
    common = {k: v for k, v in S["neural_common"].items() if k != "note"}
    return {**common, "device": "cpu", **{k: v for k, v in params.items() if k != "id"}}


def _xgb(params):
    return {k: v for k, v in params.items() if k != "id"}


def permutation(panel, model, columns, window, repeats=3, seed=42):
    """Group permutation within each date_id on the validation window."""
    out = RUNS / "permutation.jsonl"
    rows = eval_rows(panel, window)
    X = panel.frame(rows, columns)
    y = panel.target[rows]
    base = np.abs(y - model.predict(X)).mean()
    days = X.date_id.to_numpy()
    groups = feature_groups(panel.columns)
    rng = np.random.default_rng(seed)
    for name, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        increases = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = np.arange(len(X))
            for d in np.unique(days):
                at = np.flatnonzero(days == d)
                perm[at] = rng.permutation(at)
            Xp[cols] = X[cols].to_numpy()[perm]
            increases.append(float(np.abs(y - model.predict(Xp)).mean() - base))
        with open(out, "a") as handle:
            handle.write(json.dumps({"window": window, "group": name, "n_columns": len(cols),
                                     "base_mae": float(base), "mae_increase": increases}) + "\n")


def trees(panel, threads):
    for w in ALL:
        for name in ["zero", "global_median", "stock_median"]:
            run_fit(panel, "baselines", name, name, {}, w, threads=threads)
    for p in S["ridge"]:
        for w in ALL:
            run_fit(panel, "search_ridge", f"ridge_a{p['alpha']:g}", "ridge", p, w, threads=threads)
    sets = feature_sets(panel.columns)
    for set_name, cols in sets.items():
        for w in ALL:
            run_fit(panel, "ablation_ridge", set_name, "ridge", {"alpha": 100}, w, columns=cols, threads=threads)
    for p in S["xgboost"]:
        for w in ALL:
            run_fit(panel, "search_xgboost", p["id"], "xgboost", _xgb(p), w, threads=threads)
    # Ablation: full-v3 XGBoost configuration; "all" equals search x01_v3 (same fit), permutation on it.
    for w in ALL:
        if not (RUNS / "permutation_done" / w).exists():
            _, model = run_fit(panel, "ablation_xgboost", "all", "xgboost", {}, w, threads=threads, keep_model=True)
            if model is not None:
                permutation(panel, model, None, w)
                (RUNS / "permutation_done").mkdir(exist_ok=True); (RUNS / "permutation_done" / w).touch()
    for set_name, cols in sets.items():
        if set_name == "all":
            continue
        for w in ALL:
            run_fit(panel, "ablation_xgboost", set_name, "xgboost", {}, w, columns=cols, threads=threads)
    for seed in CONFIG["seeds"]["repeat"]:
        for w in ALL:
            run_fit(panel, "seeds_xgboost", "x01_v3", "xgboost", {}, w, seed=seed, threads=threads)
    unseen = holdout_stocks()
    for w in OUTER:
        for name, model, params in [("stock_median", "stock_median", {}), ("ridge_a100", "ridge", {"alpha": 100}),
                                    ("x01_v3", "xgboost", {})]:
            run_fit(panel, "unseen", name, model, params, w, exclude_stocks=unseen, eval_stocks=unseen,
                    eval_tag="unseen40", threads=threads)
    for p in S["random_forest"]:
        for w in ALL:
            run_fit(panel, "search_random_forest", p["id"], "random_forest",
                    {k: v for k, v in p.items() if k != "id"}, w, threads=threads)
    sens = S["random_forest_sensitivity"]
    for w in sens["windows"]:
        run_fit(panel, "rf_sensitivity", sens["id"], "random_forest",
                {k: v for k, v in sens.items() if k not in ("id", "windows", "note")}, w, threads=threads)


def neural(panel, threads):
    for p in S["lstm"]:
        for w in ALL:
            run_fit(panel, "search_lstm", p["id"], "lstm", _neural(p), w, threads=threads)
    for p in S["mlp"]:
        for w in ALL:
            run_fit(panel, "search_mlp", p["id"], "mlp", _neural(p), w, threads=threads)
    base = next(p for p in S["lstm"] if p["id"] == "l1_base")
    variants = {"id_none": {"identity": "none"}, "emb2": {"embedding_dim": 2}, "emb4": {"embedding_dim": 4},
                "emb16": {"embedding_dim": 16}, "onehot": {"identity": "onehot_input"},
                "seq1": {"sequence_length": 1}, "seq3": {"sequence_length": 3}}
    for name, change in variants.items():
        for w in OUTER:
            run_fit(panel, "identity", name, "lstm", _neural({**base, **change}), w, threads=threads)
    for w in OUTER:
        run_fit(panel, "identity", "mlp_id_none", "mlp",
                _neural({**next(p for p in S["mlp"] if p["id"] == "m1_h64"), "identity": "none"}), w, threads=threads)
    for seed in CONFIG["seeds"]["repeat"]:
        for w in OUTER:
            run_fit(panel, "identity_seeds", "emb8", "lstm", _neural(base), w, seed=seed, threads=threads)
            run_fit(panel, "identity_seeds", "id_none", "lstm", _neural({**base, "identity": "none"}), w,
                    seed=seed, threads=threads)
    unseen = holdout_stocks()
    for w in OUTER:
        for name, change in [("lstm_emb8", {}), ("lstm_id_none", {"identity": "none"})]:
            run_fit(panel, "unseen", name, "lstm", _neural({**base, **change}), w, exclude_stocks=unseen,
                    eval_stocks=unseen, eval_tag="unseen40", threads=threads)


def repro(panel, threads):
    """Full expanding history, same folds as full-v3 (CPU instead of GPU XGBoost)."""
    for w in OUTER:
        run_fit(panel, "repro_full_history", "ridge_a100", "ridge", {"alpha": 100}, w, expanding=True, threads=threads)
        run_fit(panel, "repro_full_history", "x01_v3", "xgboost", {}, w, expanding=True, threads=threads)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("queue", choices=["trees", "neural", "repro"])
    a.add_argument("--threads", type=int, default=6)
    args = a.parse_args()
    if args.queue == "neural":
        import torch
        torch.set_num_threads(args.threads)
    t = time.monotonic()
    panel = Panel()
    {"trees": trees, "neural": neural, "repro": repro}[args.queue](panel, args.threads)
    print(f"queue {args.queue} finished in {time.monotonic() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
