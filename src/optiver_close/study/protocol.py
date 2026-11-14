"""Frozen protocol: windows, feature groups, stock holdout, cache layout."""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CONFIG = json.loads((ROOT / "configs" / "study.json").read_text())
CACHE = ROOT / "data" / "cache"
RUNS = ROOT / "runs" / "study"
DEV_LAST_DAY = CONFIG["days"]["development_last_day"]
HOLDOUT = tuple(CONFIG["days"]["terminal_holdout"])
WINDOWS = {k: tuple(v) for k, v in CONFIG["windows"].items() if re.fullmatch(r"W\d", k)}
TRAIN_DAYS = CONFIG["windows"]["train_days"]
OUTER = CONFIG["windows"]["outer_windows"]


def train_range(window, expanding=False):
    """Days [start-101, start-2]: 100 fitting days, one gap day, then the window."""
    start = WINDOWS[window][0] if window in WINDOWS else window
    last = start - 2
    return (0 if expanding else last - TRAIN_DAYS + 1, last)


def holdout_stocks():
    return np.sort(np.random.default_rng(2026).choice(200, CONFIG["stock_holdout"]["count"], replace=False))


def feature_groups(columns):
    """Map each of the 111 predictor columns to exactly one declared group."""
    g = CONFIG["feature_groups"]
    explicit = {name: g[name] for name in ["raw_book", "raw_auction", "timing", "liquidity_imbalance", "auction_derived"]}
    groups = {k: list(v) for k, v in explicit.items()}
    groups["lags_diffs"] = [c for c in columns if re.search(r"_(lag|diff)\d+$", c)]
    groups["rolling"] = [c for c in columns if re.search(r"_(mean|std)\d+$", c)]
    groups["cross_section"] = [c for c in columns if c.startswith(("market_", "cross_section_relative_"))]
    predictors = [c for c in columns if c not in ("stock_id", "date_id")]
    flat = [c for v in groups.values() for c in v]
    if sorted(flat) != sorted(predictors) or len(flat) != len(set(flat)):
        missing, extra = set(predictors) - set(flat), set(flat) - set(predictors)
        raise ValueError(f"Feature groups do not partition predictors: missing={missing} extra={extra}")
    return groups


def feature_sets(columns):
    """Additive and leave-one-group-out sets, exactly as declared."""
    g = feature_groups(columns)
    order = ["raw_book", "raw_auction", "timing", "liquidity_imbalance", "auction_derived", "lags_diffs", "rolling", "cross_section"]
    sets, current = {}, []
    for i, name in enumerate(order):
        current = current + g[name]
        if i >= 1:
            sets[f"add{i:02d}_{'raw' if i == 1 else name}"] = list(current)
    sets["all"] = sets.pop(f"add{len(order) - 1:02d}_cross_section")
    for name in order:
        sets[f"loo_{name}"] = [c for c in sets["all"] if c not in g[name]]
    return sets
