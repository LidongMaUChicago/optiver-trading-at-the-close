"""Day-level expanding windows, with a terminal holdout excluded from selection."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Fold:
    train_days: tuple
    validation_days: tuple


def split_days(days, *, holdout_days=20, validation_days=20, folds=3, gap_days=1, min_train_days=60):
    days = np.sort(np.unique(days)).astype(int)
    if min(holdout_days, validation_days, folds, min_train_days) < 1 or gap_days < 0:
        raise ValueError("Invalid split configuration")
    if len(days) <= holdout_days + gap_days:
        raise ValueError("Not enough days for holdout")
    holdout = days[-holdout_days:]
    dev = days[:len(days) - holdout_days - gap_days]
    result = []
    for i in range(folds):
        start = len(dev) - (folds - i) * validation_days
        train_end = start - gap_days
        if train_end < min_train_days:
            raise ValueError("Not enough training days; reduce folds/windows or use more data")
        result.append(Fold(tuple(dev[:train_end]), tuple(dev[start:start + validation_days])))
    return dev, holdout, result

