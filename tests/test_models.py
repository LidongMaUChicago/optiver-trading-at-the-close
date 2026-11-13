import joblib
import numpy as np
import pandas as pd
import pytest
from optiver_close.features import make_features
from optiver_close.models import build_model
from optiver_close.inference import StreamingPredictor, run_kaggle


@pytest.mark.parametrize("name", ["ridge", "random_forest", "xgboost"])
def test_tabular_fit_persist_stream_parity(panel, tmp_path, name):
    f = make_features(panel)
    train = panel.date_id < 7
    params = {"n_estimators": 3, "max_depth": 2} if name != "ridge" else {}
    model = build_model(name, f.columns, params)
    model.fit(f.loc[train], panel.loc[train, "target"])
    artifact = dict(model=model, name=name, feature_columns=list(f), feature_version=1)
    joblib.dump(artifact, tmp_path / "trusted.joblib")
    predictor = StreamingPredictor(joblib.load(tmp_path / "trusted.joblib"))
    test = panel.loc[~train]
    predictions = []
    for _, group in test.groupby(["date_id", "seconds_in_bucket"]):
        predictions.extend(predictor.predict(group))
    np.testing.assert_allclose(predictions, model.predict(f.loc[~train]), atol=1e-5)


def test_kaggle_row_alignment_and_one_predict_per_batch(panel):
    f = make_features(panel)
    model = build_model("ridge", f.columns).fit(f, panel.target)
    artifact = dict(model=model, name="ridge", feature_columns=list(f), feature_version=1)
    class Env:
        def __init__(self): self.calls = []
        def iter_test(self):
            for _, group in panel.query("date_id == 9").groupby("seconds_in_bucket"):
                template = group[["row_id"]].iloc[::-1].copy(); template["target"] = 0
                yield group.drop(columns="target"), pd.DataFrame(), template
        def predict(self, result): self.calls.append(result)
    env = Env(); run_kaggle(artifact, env)
    assert len(env.calls) == 8
    for group in env.calls:
        assert group.row_id.iloc[0].endswith("_2")
        assert np.isfinite(group.target).all()


def test_lstm_sequences_and_stream_parity(panel, tmp_path):
    pytest.importorskip("torch")
    from optiver_close.neural import Sequences
    f = make_features(panel)
    train = panel.date_id < 7
    model = build_model("lstm", f.columns, dict(epochs=1, hidden=4, embedding_dim=2,
                                                batch_size=128, sequence_length=3))
    model.fit(f.loc[train], panel.loc[train, "target"])
    artifact = dict(model=model, name="lstm", feature_columns=list(f), feature_version=1)
    joblib.dump(artifact, tmp_path / "lstm.joblib")
    predictor = StreamingPredictor(joblib.load(tmp_path / "lstm.joblib"))
    expected = model.predict(f.loc[~train])
    actual = []
    for _, group in panel.loc[~train].groupby(["date_id", "seconds_in_bucket"]):
        actual.extend(predictor.predict(group))
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    assert model.history
    # Validation-day preprocessing is not used in fitting the scaler.
    assert int(model.scaler.n_samples_seen_) < train.sum()

