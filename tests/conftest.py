import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def panel():
    rng = np.random.default_rng(7)
    rows = []
    for day in range(10):
        for seconds in range(0, 80, 10):
            for stock in range(3):
                mid = 1 + 0.001 * stock + 0.0001 * seconds + rng.normal(0, 1e-5)
                rows.append(dict(stock_id=stock, date_id=day, seconds_in_bucket=seconds,
                    imbalance_size=100 + seconds, imbalance_buy_sell_flag=(-1) ** stock,
                    reference_price=mid, matched_size=1000 + seconds, far_price=np.nan,
                    near_price=mid, bid_price=mid - 0.0001, ask_price=mid + 0.0001,
                    bid_size=100 + stock, ask_size=200 - stock, wap=mid,
                    target=stock + 0.01 * seconds + rng.normal(0, 0.1),
                    row_id=f"{day}_{seconds}_{stock}"))
    return pd.DataFrame(rows)

