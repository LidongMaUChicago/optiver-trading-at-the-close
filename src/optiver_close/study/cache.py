"""Build the feature cache once: complete cross-sections of all 200 stocks.

Development rows (days <= 459, finite target) and terminal-holdout rows are
written to separate files; holdout labels go to their own file that only
study/final.py reads.
"""
import json
import sys
import time
import numpy as np
from ..data import load_data, engineer_panel, sha256
from .protocol import CACHE, CONFIG, DEV_LAST_DAY, HOLDOUT


def build(csv_path):
    CACHE.mkdir(parents=True, exist_ok=True)
    digest = sha256(csv_path)
    if digest != CONFIG["data"]["sha256"]:
        raise ValueError(f"Unexpected data hash {digest}")
    t = time.monotonic()
    raw = load_data(csv_path)
    X = engineer_panel(raw)
    columns = list(X.columns)
    keys = raw[["stock_id", "date_id", "seconds_in_bucket"]].to_numpy(np.int32)
    target = raw.target.to_numpy(np.float32)
    dev = (keys[:, 1] <= DEV_LAST_DAY) & np.isfinite(target)
    hold = (keys[:, 1] >= HOLDOUT[0]) & (keys[:, 1] <= HOLDOUT[1])
    values = X.to_numpy(np.float32)
    del X
    for name, mask in [("dev", dev), ("holdout", hold)]:
        np.save(CACHE / f"{name}_features.npy", values[mask])
        np.save(CACHE / f"{name}_keys.npy", keys[mask])
    np.save(CACHE / "dev_target.npy", target[dev])
    np.save(CACHE / "holdout_target_SEALED.npy", target[hold])
    meta = {"columns": columns, "sha256": digest, "dev_rows": int(dev.sum()), "holdout_rows": int(hold.sum()),
            "dropped_dev_rows_missing_target": int(((keys[:, 1] <= DEV_LAST_DAY) & ~np.isfinite(target)).sum()),
            "seconds": time.monotonic() - t}
    (CACHE / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "columns"}))


if __name__ == "__main__":
    build(sys.argv[1])
