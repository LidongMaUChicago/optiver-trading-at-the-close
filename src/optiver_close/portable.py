"""Version-portable export of the final XGBoost + centring model.

A joblib pickle ties the artifact to exact sklearn/xgboost versions. This module
stores the same model as XGBoost's native JSON plus the training medians, so it
can be loaded on another runtime (e.g. a Kaggle image) and checked for parity.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def export(artifact, folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    centred = type(artifact["model"]).__name__ == "CrossSectionCentered"
    pipe = artifact["model"].model if centred else artifact["model"]
    pre, xgb = pipe.named_steps["preprocess"], pipe.named_steps["model"]
    name, imputer, columns = pre.transformers_[0]
    if len(pre.transformers_) > 1 and pre.transformers_[1][1] != "drop":
        raise ValueError("unexpected passthrough columns")
    xgb.get_booster().save_model(str(folder / "booster.json"))
    meta = {"name": artifact["name"], "centred": centred, "columns": list(columns),
            "medians": [float(v) for v in imputer.statistics_], "feature_columns": artifact["feature_columns"],
            "feature_version": artifact["feature_version"], "training_days": artifact["training_days"]}
    (folder / "portable.json").write_text(json.dumps(meta, indent=1))
    return meta


class PortableModel:
    def __init__(self, folder):
        import xgboost
        folder = Path(folder)
        self.meta = json.loads((folder / "portable.json").read_text())
        self.booster = xgboost.Booster()
        self.booster.load_model(str(folder / "booster.json"))
        self.medians = np.asarray(self.meta["medians"], dtype=np.float64)

    def predict(self, X):
        import xgboost
        A = X[self.meta["columns"]].to_numpy(dtype=np.float64)
        A = np.where(np.isnan(A), self.medians, A)
        p = self.booster.predict(xgboost.DMatrix(A)).astype(np.float64)
        if self.meta["centred"]:
            key = X.date_id.to_numpy().astype(np.int64) * 1000 + X.seconds_in_bucket.to_numpy().astype(np.int64)
            p = p - pd.Series(p).groupby(key).transform("mean").to_numpy()
        return p

    def artifact(self):
        return {"model": self, "name": self.meta["name"] + " (portable)", "feature_columns": self.meta["feature_columns"],
                "feature_version": self.meta["feature_version"], "needs_history": False}
