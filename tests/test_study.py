import json
import numpy as np
import pandas as pd
import pytest
from optiver_close.features import make_features
from optiver_close.models import build_model, MedianBaseline
from optiver_close.study import protocol
from optiver_close.study.stats import compare, holm, block_bootstrap, newey_west_se


def test_feature_groups_partition_all_predictors(panel):
    cols = list(make_features(panel).columns)
    groups = protocol.feature_groups(cols)
    flat = [c for v in groups.values() for c in v]
    assert len(flat) == len(set(flat)) == len(cols) - 2      # everything except stock_id, date_id
    sets = protocol.feature_sets(cols)
    assert sorted(sets["all"]) == sorted(flat)
    sizes = [len(v) for k, v in sets.items() if k.startswith("add")]
    assert sizes == sorted(sizes) and len(sets["add01_raw"]) == 11
    for name, members in groups.items():
        assert not set(members) & set(sets[f"loo_{name}"])


def test_windows_have_gap_and_never_touch_holdout():
    for w, (first, last) in protocol.WINDOWS.items():
        t0, t1 = protocol.train_range(w)
        assert t1 == first - 2 and t1 - t0 + 1 == protocol.TRAIN_DAYS
        assert last <= protocol.DEV_LAST_DAY < protocol.HOLDOUT[0] - 1
    assert protocol.train_range("W3", expanding=True) == (0, 398)
    assert len(protocol.holdout_stocks()) == 40


def test_timing_ablation_removes_seconds_from_predictors(panel):
    cols = list(make_features(panel).columns)
    assert "seconds_in_bucket" not in protocol.feature_sets(cols)["loo_timing"]


def test_stock_median_falls_back_for_unseen_stock():
    X = pd.DataFrame({"stock_id": [0, 0, 1, 1]})
    m = MedianBaseline(by_stock=True).fit(X, [1.0, 3.0, 10.0, 10.0])
    assert m.predict(pd.DataFrame({"stock_id": [0, 1, 7]})).tolist() == [2.0, 10.0, m.global_]


def test_early_stopped_xgb_uses_only_last_fitting_days(panel):
    f = make_features(panel)
    m = build_model("xgboost", f.columns, {"n_estimators": 20, "early_stopping_rounds": 3, "max_depth": 2})
    m.fit(f, panel.target)
    assert 0 <= m.best_iteration_ < 20 and np.isfinite(m.predict(f)).all()


@pytest.mark.parametrize("identity", ["none", "onehot_input", "embedding"])
def test_lstm_identity_modes(panel, identity):
    pytest.importorskip("torch")
    f = make_features(panel)
    train = panel.date_id < 7
    m = build_model("lstm", f.columns, dict(epochs=1, hidden=4, embedding_dim=2, batch_size=64,
                                             sequence_length=3, identity=identity, device="cpu"))
    m.fit(f.loc[train], panel.loc[train, "target"])
    test = f.loc[~train].copy()
    base = m.predict(test)
    unknown = test.assign(stock_id=test.stock_id + 100)  # unseen ids, same per-stock grouping
    out = m.predict(unknown)
    assert np.isfinite(out).all() and m.nonfinite_steps == 0
    if identity == "none":
        np.testing.assert_allclose(out, base, atol=1e-6)  # identity carries no information
    else:
        assert not np.allclose(out, base)


def test_mlp_is_per_row(panel):
    pytest.importorskip("torch")
    f = make_features(panel)
    m = build_model("mlp", f.columns, dict(epochs=1, hidden=4, embedding_dim=2, batch_size=64, device="cpu"))
    assert m.sequence_length == 1
    m.fit(f.loc[panel.date_id < 7], panel.loc[panel.date_id < 7, "target"])
    assert np.isfinite(m.predict(f.loc[panel.date_id >= 7])).all()


def test_stats_helpers():
    d = np.r_[np.full(30, 0.1), np.full(30, 0.3)]
    r = compare(d + 1.0, np.ones(60))
    assert r["mean_diff"] == pytest.approx(0.2) and r["share_days_b_better"] == 1.0
    lo, hi = block_bootstrap(d, 5)
    assert lo <= d.mean() <= hi
    assert newey_west_se(np.ones(10)) == 0.0
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_dev_cache_excludes_holdout_if_present():
    meta = protocol.CACHE / "meta.json"
    if not meta.exists():
        pytest.skip("feature cache not built")
    keys = np.load(protocol.CACHE / "dev_keys.npy", mmap_mode="r")
    assert keys[:, 1].max() <= protocol.DEV_LAST_DAY
    assert json.loads(meta.read_text())["sha256"] == protocol.CONFIG["data"]["sha256"]


def test_cross_section_centering_per_timestamp(panel):
    from optiver_close.models import CrossSectionCentered
    f = make_features(panel)
    m = CrossSectionCentered(build_model("ridge", f.columns)).fit(f, panel.target)
    out = pd.Series(m.predict(f)).groupby([f.date_id.to_numpy(), f.seconds_in_bucket.to_numpy()]).sum()
    np.testing.assert_allclose(out, 0, atol=1e-9)
