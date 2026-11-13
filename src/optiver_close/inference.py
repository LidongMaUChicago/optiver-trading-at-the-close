"""Row-aligned streaming prediction, independent of Kaggle's runtime package."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import FeatureStream


class StreamingPredictor:
    def __init__(self, artifact):
        if artifact.get("feature_version") != 1:
            raise ValueError("Unknown feature version")
        self.artifact, self.stream = artifact, FeatureStream()

    def predict(self, batch):
        current, history, positions = self.stream.update(batch)
        cols = self.artifact["feature_columns"]
        model = self.artifact["model"]
        # Neural sequences need complete current-day history; tabular models do not.
        if self.artifact["name"] == "lstm":
            values = model.predict(history[cols])[positions]
        else:
            values = model.predict(current[cols])
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(batch),) or not np.isfinite(values).all():
            raise ValueError("Non-finite or misaligned predictions")
        return values


def run_kaggle(artifact, env):
    predictor = StreamingPredictor(artifact)
    for test, revealed_targets, sample_submission in env.iter_test():
        # Process unscored warmup batches too: they provide temporal context.
        if "row_id" not in test or "row_id" not in sample_submission:
            raise ValueError("Kaggle adapter requires row_id")
        if test.row_id.duplicated().any() or sample_submission.row_id.duplicated().any():
            raise ValueError("Duplicate submission row IDs")
        p = predictor.predict(test)
        aligned = pd.Series(p, index=test.row_id).reindex(sample_submission.row_id)
        if aligned.isna().any() or set(test.row_id) != set(sample_submission.row_id):
            raise ValueError("Test and submission row IDs differ")
        result = sample_submission.copy()
        result["target"] = aligned.to_numpy()
        env.predict(result)  # Exactly once, AFTER assigning this batch's predictions.

