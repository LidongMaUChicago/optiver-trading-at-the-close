"""Memory-conscious CSV ingestion and deterministic small development subsets."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .features import KEYS, RAW, make_features, validate


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_data(path, last_days=None, stocks=None):
    """Keep complete observations for chosen stock/day groups; no random row split.

    The small preset selects the last N chronological days and first K stock IDs.
    Its cross-sectional features describe that SUBSET, not the complete market.
    """
    usecols = KEYS + RAW + ["target"]
    dtype = {c: "float32" for c in RAW + ["target"]}
    dtype.update({c: "int32" for c in KEYS})
    pieces = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=250_000):
        if stocks is not None:
            chunk = chunk[chunk.stock_id.isin(stocks)]
        pieces.append(chunk)
    raw = pd.concat(pieces, ignore_index=True)
    if last_days:
        days = np.sort(raw.date_id.unique())[-last_days:]
        raw = raw[raw.date_id.isin(days)].reset_index(drop=True)
    if raw.empty:
        raise ValueError("No observations selected")
    validate(raw)
    return raw


def engineer_panel(raw):
    # Each auction day is independent. Avoid constructing every float64 temporal
    # intermediate for all 5M+ observations at once.
    parts = [make_features(day) for _, day in raw.groupby("date_id", sort=True)]
    return pd.concat(parts).reindex(raw.index)

