"""Deterministic, target-free features shared by batch and streaming inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["stock_id", "date_id", "seconds_in_bucket"]
RAW = ["imbalance_size", "imbalance_buy_sell_flag", "reference_price", "matched_size",
       "far_price", "near_price", "bid_price", "bid_size", "ask_price", "ask_size", "wap"]
REQUIRED = KEYS + [c for c in RAW if c not in ("far_price", "near_price")]


def validate(frame: pd.DataFrame) -> None:
    missing = set(REQUIRED) - set(frame)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Observation keys must be non-null and unique")
    keys = frame[KEYS].to_numpy(dtype=float)
    if not np.isfinite(keys).all() or (keys < 0).any() or (keys != np.floor(keys)).any():
        raise ValueError("Observation keys must be finite nonnegative integers")
    if (frame.seconds_in_bucket > 540).any():
        raise ValueError("Expected closing-auction observations from 0 to 540 seconds")


def ratio(a, b):
    return a / b.where(b.abs() > 1e-12)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per input row in ORIGINAL order; never read target.

    Cross-sectional features require a complete timestamp batch. Temporal windows
    are observation counts, not elapsed minutes, and never cross stock/day groups.
    """
    validate(frame)
    d = frame.reindex(columns=KEYS + RAW).copy().reset_index(drop=True)
    d["_order"] = np.arange(len(d))
    d = d.sort_values(KEYS, kind="stable").reset_index(drop=True)
    for c in RAW:
        d[c] = pd.to_numeric(d[c], errors="raise").astype("float64")
    f = d[KEYS + RAW].copy()
    f["spread"] = d.ask_price - d.bid_price
    f["mid_price"] = (d.ask_price + d.bid_price) / 2
    f["relative_spread"] = ratio(f.spread, f.mid_price)
    f["depth"] = d.bid_size + d.ask_size
    f["book_imbalance"] = ratio(d.bid_size - d.ask_size, f.depth)
    f["signed_auction_imbalance"] = d.imbalance_size * d.imbalance_buy_sell_flag
    f["auction_pressure"] = ratio(f.signed_auction_imbalance, d.matched_size + d.imbalance_size)
    f["auction_to_depth"] = ratio(d.imbalance_size, f.depth)
    for a, b in [("wap", "reference_price"), ("near_price", "far_price"),
                 ("reference_price", "mid_price"), ("wap", "mid_price")]:
        f[f"{a}_vs_{b}"] = ratio(f[a] - f[b], f[b]) * 1e4
    for c in ["bid_size", "ask_size", "matched_size", "imbalance_size", "depth"]:
        f[f"log_{c}"] = np.log1p(f[c].clip(lower=0))
    f["seconds_to_close"] = 600 - d.seconds_in_bucket
    f["auction_phase"] = np.select([d.seconds_in_bucket < 300, d.seconds_in_bucket < 480], [0, 1], default=2)
    temporal = ["wap", "reference_price", "spread", "book_imbalance", "signed_auction_imbalance", "matched_size"]
    g = f.groupby([d.stock_id, d.date_id], sort=False)
    extra = {}
    for c in temporal:
        for lag in (1, 3, 6):
            old = g[c].shift(lag)
            extra[f"{c}_lag{lag}"] = old
            extra[f"{c}_diff{lag}"] = f[c] - old
        for window in (3, 6, 12):
            rolled = g[c].rolling(window, min_periods=1)
            extra[f"{c}_mean{window}"] = rolled.mean().reset_index(level=[0, 1], drop=True)
            extra[f"{c}_std{window}"] = rolled.std(ddof=0).reset_index(level=[0, 1], drop=True)
    f = pd.concat([f, pd.DataFrame(extra, index=f.index)], axis=1)
    # All members of a current timestamp are observable together in the API.
    cs = f.groupby([d.date_id, d.seconds_in_bucket], sort=False)
    extra = {}
    for c in ["book_imbalance", "auction_pressure", "spread", "wap_vs_reference_price"]:
        mean = cs[c].transform("mean")
        extra[f"market_{c}"] = mean
        extra[f"cross_section_relative_{c}"] = f[c] - mean
    f = pd.concat([f, pd.DataFrame(extra, index=f.index)], axis=1)
    f = f.replace([np.inf, -np.inf], np.nan)
    cols = [c for c in f if c not in KEYS]
    f = f.astype({c: "float32" for c in cols}).copy()
    f = f.iloc[np.argsort(d._order.to_numpy())]
    f.index = frame.index
    return f


class FeatureStream:
    """Bounded current-day history; one complete timestamp per call.

    Store raw rows, not engineered rows, to avoid duplicate transformations.
    Reject repeated/out-of-order batches rather than silently misalign predictions.
    """
    def __init__(self):
        self.history = pd.DataFrame()
        self.last_key = None

    def update(self, batch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        validate(batch)
        stamps = batch[["date_id", "seconds_in_bucket"]].drop_duplicates()
        if len(stamps) != 1:
            raise ValueError("Each streaming batch must contain exactly one complete timestamp")
        key = tuple(map(int, stamps.iloc[0]))
        if self.last_key is not None and key <= self.last_key:
            raise ValueError("Streaming timestamps must be strictly increasing")
        history = self.history if self.last_key and key[0] == self.last_key[0] else pd.DataFrame()
        raw = batch.reindex(columns=KEYS + RAW).copy()
        combined = pd.concat([history, raw], ignore_index=True) if len(history) else raw.reset_index(drop=True)
        features = make_features(combined)
        current = np.arange(len(combined) - len(batch), len(combined))
        self.history, self.last_key = combined, key
        return features.iloc[current].set_axis(batch.index), features, current

