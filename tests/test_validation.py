import numpy as np
import pytest
from optiver_close.validation import split_days
from optiver_close.experiment import metrics


def test_chronological_purge_and_terminal_holdout():
    dev, holdout, folds = split_days(np.arange(30), holdout_days=4, validation_days=3,
                                     folds=3, gap_days=1, min_train_days=5)
    assert max(dev) + 1 < min(holdout)
    for fold in folds:
        assert max(fold.train_days) + 1 < min(fold.validation_days)
        assert set(fold.validation_days).isdisjoint(holdout)
    assert set(folds[0].train_days) < set(folds[1].train_days)


def test_invalid_split_fails():
    with pytest.raises(ValueError):
        split_days(np.arange(3))


def test_metrics_promote_float16():
    y = np.array([500, -500, 0], dtype=np.float16)
    m = metrics(y, np.zeros(3))
    assert m["r2"] == pytest.approx(0)
    assert np.isfinite(m["rmse"])
    with pytest.raises(ValueError):
        metrics([1, 2], [1, np.nan])

