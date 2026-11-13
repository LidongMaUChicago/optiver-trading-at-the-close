import numpy as np
import pandas as pd
import pytest
from optiver_close.features import make_features, FeatureStream


def test_future_and_target_cannot_change_past(panel):
    base = make_features(panel)
    changed = panel.copy()
    changed.loc[changed.seconds_in_bucket > 30, ["wap", "bid_size", "ask_price"]] *= 10
    changed["target"] = 9999
    out = make_features(changed)
    pd.testing.assert_frame_equal(base.loc[panel.seconds_in_bucket <= 30], out.loc[panel.seconds_in_bucket <= 30])
    assert "target" not in base


def test_no_temporal_cross_stock_or_day(panel):
    f = make_features(panel)
    first = panel.seconds_in_bucket == 0
    assert f.loc[first, "wap_lag1"].isna().all()
    row = panel.query("stock_id == 2 and date_id == 1 and seconds_in_bucket == 20").index[0]
    previous = panel.query("stock_id == 2 and date_id == 1 and seconds_in_bucket == 10").wap.iloc[0]
    assert f.loc[row, "wap_lag1"] == pytest.approx(previous)


def test_input_permutation_does_not_change_results(panel):
    shuffled = panel.sample(frac=1, random_state=22)
    pd.testing.assert_frame_equal(make_features(panel).sort_index(), make_features(shuffled).sort_index())


def test_stream_equals_batch_with_missing_stocks(panel):
    panel = panel.drop(panel.query("stock_id == 1 and seconds_in_bucket == 20").index)
    batch = make_features(panel)
    stream = FeatureStream(); parts = []
    for _, group in panel.groupby(["date_id", "seconds_in_bucket"], sort=True):
        current, history, positions = stream.update(group)
        parts.append(current)
        assert history.date_id.nunique() == 1
        assert len(stream.history) <= 24
    pd.testing.assert_frame_equal(batch.sort_index(), pd.concat(parts).sort_index())


def test_duplicate_or_out_of_order_rejected(panel):
    with pytest.raises(ValueError):
        make_features(pd.concat([panel, panel.iloc[:1]]))
    stream = FeatureStream(); group = panel.query("date_id == 0 and seconds_in_bucket == 0")
    stream.update(group)
    with pytest.raises(ValueError):
        stream.update(group)
    with pytest.raises(ValueError):
        FeatureStream().update(panel)


def test_zero_depth_is_missing_not_infinite(panel):
    panel[["bid_size", "ask_size"]] = 0
    f = make_features(panel)
    assert f.book_imbalance.isna().all()
    assert not np.isinf(f.to_numpy()).any()

