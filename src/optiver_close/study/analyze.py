"""Aggregate every logged fit into report tables. Development data only.

    python -m optiver_close.study.analyze

Writes reports/study/*.json|csv|png. Reads predictions under runs/study/preds.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import Panel
from .protocol import CONFIG, RUNS, ROOT, WINDOWS, OUTER, feature_groups
from .stats import compare, holm

OUT = ROOT / "reports" / "study"
ALL = list(WINDOWS)
FAMILIES = {"ridge": "search_ridge", "random_forest": "search_random_forest",
            "xgboost": "search_xgboost", "lstm": "search_lstm", "mlp": "search_mlp"}


class Store:
    def __init__(self, panel):
        self.p = panel
        self.idx = {}

    def rows(self, w, tag="all"):
        if (w, tag) not in self.idx:
            self.idx[w, tag] = np.load(RUNS / "index" / f"{w}_{tag}.npy")
        return self.idx[w, tag]

    def pred(self, exp, cand, w, seed=42):
        path = RUNS / "preds" / exp / f"{cand}__{w}__s{seed}.npy"
        return np.load(path).astype(np.float64) if path.exists() else None

    def has(self, exp, cand, windows, seed=42):
        return all(self.pred(exp, cand, w, seed) is not None for w in windows)

    def frame(self, exp, cand, windows, seed=42, tag="all"):
        parts = []
        for w in windows:
            r = self.rows(w, tag)
            parts.append(pd.DataFrame({"row": r, "window": w, "day": self.p.day[r], "y": self.p.target[r],
                                       "pred": self.pred(exp, cand, w, seed)}))
        return pd.concat(parts, ignore_index=True)

    def daily(self, exp, cand, windows, seed=42, tag="all", pred=None):
        f = self.frame(exp, cand, windows, seed, tag) if pred is None else pred
        return (f.y - f.pred).abs().groupby(f.day).mean()


def mae(f):
    return float((f.y - f.pred).abs().mean())


def candidates(exp):
    d = RUNS / "preds" / exp
    return sorted({p.name.split("__")[0] for p in d.glob("*.npy")}) if d.exists() else []


def search_tables(s):
    rows, complete = [], {}
    for family, exp in FAMILIES.items():
        for cand in candidates(exp):
            rec = {"family": family, "candidate": cand}
            for w in ALL:
                p = s.pred(exp, cand, w)
                rec[w] = None if p is None else float(np.abs(s.p.target[s.rows(w)] - p).mean())
            ok = [rec[w] for w in ALL if rec[w] is not None]
            rec["windows_done"] = len(ok)
            rec["mean_W1_W5"] = float(np.mean(ok)) if len(ok) == len(ALL) else None
            rows.append(rec)
            if len(ok) == len(ALL):
                complete.setdefault(family, []).append(cand)
    return pd.DataFrame(rows), complete


def nested(s, table, complete):
    """Select on W1..W(k-1), score on Wk, for each outer window and family."""
    out = {}
    pooled = lambda fam, cand, ws: float(np.mean([table.set_index(["family", "candidate"]).loc[(fam, cand), w] for w in ws]))
    for family, cands in complete.items():
        picks = []
        for k, w in enumerate(ALL):
            if w not in OUTER:
                continue
            prior = ALL[:k]
            best = min(cands, key=lambda c: (pooled(family, c, prior), cands.index(c)))
            picks.append({"outer": w, "selected": best, "selection_mae": pooled(family, best, prior),
                          "outer_mae": pooled(family, best, [w])})
        out[family] = {"picks": picks, "outer_mean_mae": float(np.mean([p["outer_mae"] for p in picks])),
                       "final_choice_all5": min(cands, key=lambda c: (pooled(family, c, ALL), cands.index(c)))}
    return out


def nested_daily(s, nest, family):
    parts = []
    for pick in nest[family]["picks"]:
        parts.append(s.frame(FAMILIES[family], pick["selected"], [pick["outer"]]))
    f = pd.concat(parts, ignore_index=True)
    return f


def simplex_grid(k, step=0.05):
    n = int(round(1 / step))
    for combo in itertools.product(range(n + 1), repeat=k - 1):
        if sum(combo) <= n:
            yield np.array(list(combo) + [n - sum(combo)]) / n


def ensembles(s, nest):
    fams = [f for f in ["ridge", "random_forest", "xgboost", "lstm"] if f in nest]
    if len(fams) < 4:
        return None
    res = {"outer": [], "note": "member configurations, weights and calibration chosen on earlier windows only"}
    for k, w in enumerate(ALL):
        if w not in OUTER:
            continue
        prior = ALL[:k]
        members = {f: next(p["selected"] for p in nest[f]["picks"] if p["outer"] == w) for f in fams}
        P = lambda ws: np.column_stack([np.concatenate([s.pred(FAMILIES[f], members[f], x) for x in ws]) for f in fams])
        y = lambda ws: np.concatenate([s.p.target[s.rows(x)] for x in ws])
        Pp, yp, Po, yo = P(prior), y(prior), P([w]), y([w])
        best_w, best = None, np.inf
        for wt in simplex_grid(len(fams)):
            m = np.abs(yp - Pp @ wt).mean()
            if m < best:
                best, best_w = m, wt
        xi = fams.index("xgboost"); li = fams.index("lstm")
        bias = float(np.median(yp - Pp[:, xi]))
        rows_o = s.rows(w)
        key = pd.Series(s.p.day[rows_o].astype(np.int64) * 1000 + s.p.second[rows_o])
        xo = Po[:, xi]
        centred = xo - pd.Series(xo).groupby(key.to_numpy()).transform("mean").to_numpy()
        rows_p = np.concatenate([s.rows(x) for x in prior])
        keyp = s.p.day[rows_p].astype(np.int64) * 1000 + s.p.second[rows_p]
        xp = Pp[:, xi]
        centred_prior_gain = float(np.abs(yp - xp).mean() - np.abs(yp - (xp - pd.Series(xp).groupby(keyp).transform("mean").to_numpy())).mean())
        cand = {"xgboost_single": xo, "equal_4": Po.mean(axis=1), "equal_xgb_lstm": (Po[:, xi] + Po[:, li]) / 2,
                "nonneg_weights": Po @ best_w, "xgb_plus_bias": xo + bias, "xgb_centred": centred}
        res["outer"].append({"outer": w, "members": members, "weights": dict(zip(fams, best_w.round(3).tolist())),
                             "bias": bias, "centering_gain_on_prior_windows": centred_prior_gain,
                             "mae": {n: float(np.abs(yo - v).mean()) for n, v in cand.items()}})
        for n, v in cand.items():
            np.save(RUNS / "preds" / "_ensemble" / f"{n}__{w}__s42.npy", v.astype(np.float32)) if (RUNS / "preds" / "_ensemble").exists() else None
    res["outer_mean"] = {n: float(np.mean([o["mae"][n] for o in res["outer"]])) for n in res["outer"][0]["mae"]}
    return res


def ensemble_all5(s, nest):
    """Equal-weight blends of the fixed all-window family-best configs (final rule)."""
    fams = [f for f in ["ridge", "random_forest", "xgboost", "lstm"] if f in nest]
    if len(fams) < 4:
        return None
    members = {f: nest[f]["final_choice_all5"] for f in fams}
    out = {"members": members}
    for name, use in [("equal_4", fams), ("equal_xgb_lstm", ["xgboost", "lstm"]),
                      ("equal_xgb_rf_lstm", ["xgboost", "random_forest", "lstm"])]:
        per = []
        for w in ALL:
            p = np.mean([s.pred(FAMILIES[f], members[f], w) for f in use], axis=0)
            per.append(float(np.abs(s.p.target[s.rows(w)] - p).mean()))
        out[name] = {"per_window": dict(zip(ALL, per)), "mean_W1_W5": float(np.mean(per))}
    return out


def experiment_table(s, exp, windows, tag="all", seeds=(42,)):
    rows = []
    for cand in candidates(exp):
        for seed in seeds:
            if not s.has(exp, cand, windows, seed):
                continue
            f = s.frame(exp, cand, windows, seed, tag)
            rows.append({"candidate": cand, "seed": seed, "mae": mae(f),
                         **{w: mae(f[f.window == w]) for w in windows}})
    return pd.DataFrame(rows)


def error_analysis(s, exp, cand):
    """Descriptive slices of the selected configuration's out-of-window predictions."""
    f = s.frame(exp, cand, ALL)
    p = s.p
    col = lambda c: np.asarray(p.values[f.row.to_numpy(), p.col_index[c]], dtype=np.float64)
    f["abs_err"], f["abs_zero"] = (f.y - f.pred).abs(), f.y.abs()
    f["second"] = p.second[f.row]
    f["stock"] = p.stock[f.row]
    f["phase"] = pd.cut(f.second, [-1, 290, 470, 540], labels=["0-290s", "300-470s", "480-540s"])
    f["spread_q"] = pd.qcut(col("relative_spread"), 5, labels=False, duplicates="drop")
    f["pressure_q"] = pd.qcut(np.abs(col("auction_pressure")), 5, labels=False, duplicates="drop")
    f["far_missing"] = np.isnan(col("far_price"))
    q99 = f.y.abs().quantile(0.99)
    f["tail"] = np.where(f.y.abs() > q99, "|target| > p99", "rest")
    out = {}
    for name in ["phase", "spread_q", "pressure_q", "far_missing", "tail", "window"]:
        g = f.groupby(name, observed=True).agg(rows=("y", "size"), mae=("abs_err", "mean"), zero_mae=("abs_zero", "mean"),
                                               bias=("pred", "mean"), target_mean=("y", "mean"))
        g["gain_vs_zero"] = g.zero_mae - g.mae
        out[name] = g.reset_index().to_dict("records")
    daily = f.groupby("day").agg(mae=("abs_err", "mean"), zero=("abs_zero", "mean"), rows=("y", "size"))
    daily["gain"] = (daily.zero - daily.mae) * daily.rows
    top5 = daily.gain.sort_values(ascending=False).head(5).sum() / daily.gain.sum()
    stock = f.groupby("stock").agg(mae=("abs_err", "mean"), zero=("abs_zero", "mean"))
    stock["gain"] = stock.zero - stock.mae
    resid = f.y - f.pred
    out["concentration"] = {"days": len(daily), "days_model_better": int((daily.zero > daily.mae).sum()),
                            "share_of_total_gain_from_best_5_days": float(top5),
                            "stocks": len(stock), "stocks_model_better": int((stock.gain > 0).sum()),
                            "worst_5_stocks_gain": stock.gain.nsmallest(5).round(3).to_dict(),
                            "best_5_stocks_gain": stock.gain.nlargest(5).round(3).to_dict()}
    out["residuals"] = {"mean": float(resid.mean()), "median": float(resid.median()),
                        "p01": float(resid.quantile(0.01)), "p99": float(resid.quantile(0.99)),
                        "largest_abs_error": float(f.abs_err.max()),
                        "share_of_abs_error_from_top1pct_rows": float(f.abs_err.nlargest(len(f) // 100).sum() / f.abs_err.sum()),
                        "pred_std": float(f.pred.std()), "target_std": float(f.y.std())}
    # Declared regime: cross-sectional mean |wap_diff1| over seconds <= 300, median split over dev days.
    early = p.second <= 300
    vol = pd.Series(np.abs(np.asarray(p.values[np.flatnonzero(early), p.col_index["wap_diff1"]], dtype=np.float64))).groupby(
        p.day[early]).mean()
    high = vol > vol.median()
    f["regime"] = np.where(high.reindex(f.day).to_numpy(), "high_vol", "low_vol")
    g = f.groupby("regime").agg(days=("day", "nunique"), rows=("y", "size"), mae=("abs_err", "mean"), zero_mae=("abs_zero", "mean"))
    g["gain_vs_zero"] = g.zero_mae - g.mae
    out["regime"] = g.reset_index().to_dict("records")
    period = np.where(f.window.isin(["W1", "W2"]), "earlier W1-W2", np.where(f.window.isin(["W4", "W5"]), "later W4-W5", "W3"))
    g = f.groupby(period).agg(rows=("y", "size"), mae=("abs_err", "mean"), zero_mae=("abs_zero", "mean"))
    g["gain_vs_zero"] = g.zero_mae - g.mae
    out["period"] = g.reset_index(names="period").to_dict("records")
    return out, f


def feature_audit(p):
    """wap_vs_mid_price: formula, identity with half-spread x book imbalance, stability."""
    rng = np.random.default_rng(0)
    r = np.sort(rng.choice(len(p.target), 200_000, replace=False))
    v = lambda c: np.asarray(p.values[r, p.col_index[c]], dtype=np.float64)
    wvm, rs, bi = v("wap_vs_mid_price"), v("relative_spread"), v("book_imbalance")
    bid, ask, bs, as_, wap = v("bid_price"), v("ask_price"), v("bid_size"), v("ask_size"), v("wap")
    micro = (bid * as_ + ask * bs) / (bs + as_)
    # If wap is the size-weighted microprice, wap - mid = (spread / 2) * book_imbalance exactly.
    identity = rs / 2 * bi * 1e4
    ok = np.isfinite(wvm) & np.isfinite(identity)
    y = p.target[r]
    return {"definition": "(wap - mid) / mid * 1e4, mid = (bid + ask) / 2; same-row inputs only (causal)",
            "rows_sampled": len(r), "nonfinite_share": float((~np.isfinite(wvm)).mean()),
            "wap_vs_size_weighted_microprice_median_abs_rel_diff": float(np.nanmedian(np.abs(wap - micro) / wap)),
            "wap_vs_size_weighted_microprice_p99_abs_rel_diff": float(np.nanquantile(np.abs(wap - micro) / wap, 0.99)),
            "corr_with_half_spread_times_imbalance": float(np.corrcoef(wvm[ok], identity[ok])[0, 1]),
            "median_abs_diff_vs_half_spread_times_imbalance_bp": float(np.median(np.abs(wvm - identity)[ok])),
            "quantiles_bp": dict(zip(["p01", "p50", "p99"], np.nanquantile(wvm, [0.01, 0.5, 0.99]).round(3).tolist())),
            "corr_with_target": float(np.corrcoef(wvm[ok], y[ok])[0, 1]),
            "corr_with_book_imbalance": float(np.corrcoef(wvm[ok], bi[ok])[0, 1])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (RUNS / "preds" / "_ensemble").mkdir(parents=True, exist_ok=True)
    p = Panel()
    s = Store(p)
    R = {"protocol": "configs/study.json", "scope": CONFIG["scope"]}
    fits = pd.read_json(RUNS / "fits.jsonl", lines=True)
    if "error" not in fits:
        fits["error"] = None
    R["fits"] = {"total": len(fits), "ok": int((fits.status == "ok").sum()), "failed": int((fits.status != "ok").sum()),
                 "failures": fits.loc[fits.status != "ok", ["experiment", "candidate", "window", "error"]].to_dict("records"),
                 "compute_hours": float(fits.seconds.sum() / 3600),
                 "by_experiment": fits.groupby("experiment").agg(fits=("status", "size"), hours=("seconds", lambda x: x.sum() / 3600)).round(2).reset_index().to_dict("records"),
                 "max_peak_rss_gb": float(fits.peak_rss_gb.max())}
    index = fits.drop(columns=["trace"], errors="ignore").copy()
    index["eval_rows_path"] = "runs/study/index/" + index.window + "_" + index.eval_tag + ".npy"
    index["predictions_path"] = ("runs/study/preds/" + index.experiment + "/" + index.candidate + "__"
                                 + index.window + "__s" + index.seed.astype(str) + ".npy")
    index["artifact_note"] = "private, git-ignored"
    index.to_csv(OUT / "experiment_index.csv", index=False)
    # Baselines
    base = experiment_table(s, "baselines", ALL)
    R["baselines"] = base.to_dict("records")
    # Searches + nested
    table, complete = search_tables(s)
    table.to_csv(OUT / "search_results.csv", index=False)
    nest = nested(s, table, complete)
    R["nested"] = nest
    zero = s.frame("baselines", "zero", OUTER)
    R["nested_vs_zero"] = {fam: float(mae(zero) - mae(nested_daily(s, nest, fam))) for fam in nest}
    # Confirmatory family comparisons on outer windows (nested selections), daily paired.
    conf = {}
    if all(f in nest for f in ["ridge", "random_forest", "xgboost", "lstm"]):
        dz = s.daily("baselines", "zero", OUTER)
        d = {f: s.daily(None, None, None, pred=nested_daily(s, nest, f)) for f in nest}
        conf["xgboost_vs_ridge"] = compare(d["ridge"], d["xgboost"])
        conf["xgboost_vs_lstm"] = compare(d["lstm"], d["xgboost"])
        conf["xgboost_vs_random_forest"] = compare(d["random_forest"], d["xgboost"])
        conf["xgboost_vs_zero"] = compare(dz, d["xgboost"])
    # Ensembles
    R["ensembles_nested"] = ensembles(s, nest)
    R["ensembles_all5"] = ensemble_all5(s, nest)
    # Ablations
    for model in ["xgboost", "ridge"]:
        exp = f"ablation_{model}"
        t = experiment_table(s, exp, ALL)
        if len(t) and "all" in set(t.candidate):
            ref = s.daily(exp, "all", ALL)
            comps = {}
            for cand in t.candidate:
                if cand != "all":
                    comps[cand] = compare(s.daily(exp, cand, ALL), ref)
            t["diff_vs_all"] = t.candidate.map(lambda c: comps[c]["mean_diff"] if c in comps else 0.0)
            R[exp] = {"table": t.to_dict("records"), "paired_vs_all": comps}
            t.to_csv(OUT / f"{exp}.csv", index=False)
    if "ablation_xgboost" in R:
        loo = {k: v for k, v in R["ablation_xgboost"]["paired_vs_all"].items() if k.startswith("loo_")}
        if loo:
            top = max(loo, key=lambda k: loo[k]["mean_diff"])
            conf[f"all_vs_{top}"] = loo[top]
    # Determinism: ablation 'all' vs search x01_v3 are independent fits of one config.
    det = [float(np.abs(s.pred("ablation_xgboost", "all", w) - s.pred("search_xgboost", "x01_v3", w)).max())
           for w in ALL if s.pred("ablation_xgboost", "all", w) is not None and s.pred("search_xgboost", "x01_v3", w) is not None]
    R["xgboost_refit_max_abs_pred_diff"] = max(det) if det else None
    # Permutation
    perm = RUNS / "permutation.jsonl"
    if perm.exists():
        pm = pd.read_json(perm, lines=True)
        pm["mean_increase"] = pm.mae_increase.map(np.mean)
        R["permutation"] = pm.groupby("group").agg(windows=("window", "nunique"), mae_increase=("mean_increase", "mean"),
            min_window=("mean_increase", "min"), max_window=("mean_increase", "max")).sort_values("mae_increase", ascending=False).reset_index().to_dict("records")
    # Seeds (XGBoost) and identity / embeddings / sequence length
    seeds = [42] + CONFIG["seeds"]["repeat"]
    xs = [mae(s.frame("search_xgboost" if sd == 42 else "seeds_xgboost", "x01_v3", ALL, sd))
          for sd in seeds if s.has("search_xgboost" if sd == 42 else "seeds_xgboost", "x01_v3", ALL, sd)]
    R["xgboost_seed_mae"] = {"values": xs, "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else None}
    ident = {}
    l1 = lambda exp, cand: mae(s.frame(exp, cand, OUTER)) if s.has(exp, cand, OUTER) else None
    ident["lstm_variants_W3_W5"] = {"emb8 (l1_base)": l1("search_lstm", "l1_base"),
                                    **{c: l1("identity", c) for c in candidates("identity")}}
    ident["sequence_length_W3_W5"] = {"1": l1("identity", "seq1"), "3": l1("identity", "seq3"), "6": l1("search_lstm", "l7_seq6"),
                                      "12": l1("search_lstm", "l1_base"), "24": l1("search_lstm", "l8_seq24")}
    ident["mlp_W3_W5"] = {"m1_h64 (embedding)": l1("search_mlp", "m1_h64"), "m1_h64 identity none": l1("identity", "mlp_id_none"),
                          "lstm l1_base": l1("search_lstm", "l1_base")}
    pairs = []
    for sd in seeds:
        e_exp, n_exp = ("search_lstm", "identity") if sd == 42 else ("identity_seeds", "identity_seeds")
        e_c, n_c = ("l1_base", "id_none") if sd == 42 else ("emb8", "id_none")
        if s.has(e_exp, e_c, OUTER, sd) and s.has(n_exp, n_c, OUTER, sd):
            pairs.append((sd, s.daily(n_exp, n_c, OUTER, sd), s.daily(e_exp, e_c, OUTER, sd)))
    if pairs:
        dn = sum(p[1] for p in pairs) / len(pairs); de = sum(p[2] for p in pairs) / len(pairs)
        ident["embedding_vs_none_seed_averaged"] = compare(dn, de)
        ident["embedding_vs_none_per_seed"] = {int(sd): float(n.mean() - e.mean()) for sd, n, e in pairs}
        conf["lstm_embedding_vs_no_identity"] = ident["embedding_vs_none_seed_averaged"]
    R["identity"] = ident
    # Unseen stocks
    un = experiment_table(s, "unseen", OUTER, tag="unseen40")
    if len(un):
        zero_unseen = float(np.abs(np.concatenate([p.target[s.rows(w, "unseen40")] for w in OUTER])).mean())
        R["unseen_stocks"] = {"zero_mae": zero_unseen, "table": un.to_dict("records"),
                              "seen_reference_same_rows": {
                                  c: float(np.mean(np.concatenate([np.abs(p.target[s.rows(w)] - s.pred(e, c, w))[np.isin(s.rows(w), s.rows(w, "unseen40"))] for w in OUTER])))
                                  for e, c in [("search_xgboost", "x01_v3"), ("search_lstm", "l1_base"), ("search_ridge", "ridge_a100")]
                                  if s.has(e, c, OUTER)}}
    # Reproduction vs full-v3
    rp = experiment_table(s, "repro_full_history", OUTER)
    if len(rp):
        R["repro_full_history"] = rp.to_dict("records")
    # Holm over the confirmatory family
    if conf:
        adj = holm({k: v["nw_p_two_sided"] for k, v in conf.items()})
        for k in conf:
            conf[k]["holm_p"] = adj[k]
        R["confirmatory"] = conf
    # Final selection by the declared rule
    pool = {f"{fam}:{c}": float(table.set_index(["family", "candidate"]).loc[(fam, c), "mean_W1_W5"])
            for fam, cs in complete.items() for c in cs}
    if R.get("ensembles_all5"):
        for n in ["equal_4", "equal_xgb_lstm", "equal_xgb_rf_lstm"]:
            pool[f"blend:{n}"] = R["ensembles_all5"][n]["mean_W1_W5"]
    if pool:
        winner = min(pool, key=pool.get)
        sel = {"rule": CONFIG["final"]["rule"], "winner": winner,
               "top10": dict(sorted(pool.items(), key=lambda kv: kv[1])[:10]), "calibration": None}
        en = R.get("ensembles_nested")
        if en and winner.startswith("xgboost:"):
            om = en["outer_mean"]
            equal_best = min(om["equal_4"], om["equal_xgb_lstm"])
            eligible = {k: om[k] for k in ["nonneg_weights", "xgb_plus_bias", "xgb_centred"] if om[k] < equal_best}
            best = min(eligible, key=eligible.get) if eligible else None
            sel["calibration_step"] = {"equal_weight_best_outer": equal_best, "eligible": eligible,
                                       "xgboost_single_outer": om["xgboost_single"]}
            # Step 2 of the rule: an eligible calibrated variant replaces the single model only if it
            # also beats it in the nested outer evaluation. Only parameter-free centering is supported.
            if best == "xgb_centred" and om[best] < om["xgboost_single"]:
                sel["calibration"] = "cross_sectional_centering"
        R["final_selection"] = sel
        if "xgboost" in nest:
            ea, frame = error_analysis(s, "search_xgboost", nest["xgboost"]["final_choice_all5"])
            R["error_analysis"] = {"model": "search_xgboost:" + nest["xgboost"]["final_choice_all5"], **ea}
            plot(frame, s)
    R["feature_audit_wap_vs_mid_price"] = feature_audit(p)
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: R[k] for k in ["fits", "final_selection"] if k in R}, indent=1, default=str)[:3000])


def plot(f, s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    d = f.groupby("day").agg(model=("abs_err", "mean"), zero=("abs_zero", "mean"))
    (d.zero - d.model).plot(ax=ax[0], marker=".", lw=0.8)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set(title="Daily MAE gain vs zero (W1-W5)", xlabel="date_id", ylabel="bp")
    g = f.groupby(f.second // 60 * 60).agg(model=("abs_err", "mean"), zero=("abs_zero", "mean"))
    g.plot(ax=ax[1], marker=".")
    ax[1].set(title="MAE by seconds_in_bucket", xlabel="seconds (60 s bins)", ylabel="MAE bp")
    st = f.groupby("stock").agg(model=("abs_err", "mean"), zero=("abs_zero", "mean"))
    (st.zero - st.model).sort_values().reset_index(drop=True).plot(ax=ax[2])
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set(title="Per-stock gain vs zero (sorted)", xlabel="stock rank", ylabel="bp")
    fig.tight_layout(); fig.savefig(OUT / "error_analysis.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
