"""Paired forecast-loss comparisons on DAILY loss differences.

Rows within a day share the market (and targets overlap in time), so the
independent unit is the day: every resample keeps whole cross-sections together.
"""
from __future__ import annotations
import math
import numpy as np


def newey_west_se(d, lag=5):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    e = d - d.mean()
    gamma0 = e @ e / n
    s = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * (e[k:] @ e[:-k]) / n
    return math.sqrt(max(s, 0.0) / n)


def _normal_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


def iid_bootstrap(d, draws=4000, seed=42):
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(draws, len(d)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def block_bootstrap(d, block, draws=4000, seed=42):
    """Moving-block bootstrap of the mean of daily differences (days in time order)."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    nblocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(draws, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(draws, -1)[:, :n]
    return np.quantile(d[idx].mean(axis=1), [0.025, 0.975]).tolist()


def compare(daily_a, daily_b, lag=5):
    """Mean of MAE_a - MAE_b over paired days (positive: b is better)."""
    d = np.asarray(daily_a, dtype=np.float64) - np.asarray(daily_b, dtype=np.float64)
    se = newey_west_se(d, lag)
    z = d.mean() / se if se > 0 else float("inf")
    return {"days": len(d), "mean_diff": float(d.mean()), "share_days_b_better": float((d > 0).mean()),
            "nw_se": se, "nw_z": float(z), "nw_p_two_sided": _normal_p(z),
            "iid_day_bootstrap_95": iid_bootstrap(d),
            "block_bootstrap_95": {str(b): block_bootstrap(d, b) for b in (2, 5, 10)}}


def holm(pvalues):
    """Holm step-down adjusted p-values, dict in -> dict out."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out
