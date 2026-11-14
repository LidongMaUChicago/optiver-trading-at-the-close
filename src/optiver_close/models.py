"""Four model families from the original work, with fold-local preprocessing."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


class ChunkedRidge:
    """Ridge on median-imputed, standardised numeric features plus one-hot stock IDs.

    Same objective as scikit-learn's Ridge(alpha) on that design (unpenalised
    intercept), solved exactly via a Gram matrix accumulated over row chunks, so the
    one-hot design matrix is never materialised. sklearn's ColumnTransformer + lsqr
    version needed >30 GB on the 4.3M-row training folds and was OOM-killed.
    """
    def __init__(self, columns, alpha=100.0, chunk=250_000):
        self.numeric = [c for c in columns if c not in ("stock_id", "date_id")]
        self.alpha, self.chunk = float(alpha), chunk

    def _design(self, X):
        z = X[self.numeric].to_numpy(dtype=np.float64)
        z = np.where(np.isnan(z), self.median_, z)
        z = (z - self.mean_) / self.scale_
        onehot = np.zeros((len(X), len(self.stocks_)))
        pos = np.searchsorted(self.stocks_, X.stock_id.to_numpy())
        known = (pos < len(self.stocks_)) & (self.stocks_[np.minimum(pos, len(self.stocks_) - 1)] == X.stock_id.to_numpy())
        onehot[np.flatnonzero(known), pos[known]] = 1.0      # unseen stocks: all zeros, as handle_unknown="ignore"
        return np.hstack([np.ones((len(X), 1)), z, onehot])

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float64)
        values = X[self.numeric].to_numpy(dtype=np.float32)
        self.median_ = np.nanmedian(values, axis=0).astype(np.float64)
        self.median_ = np.where(np.isnan(self.median_), 0.0, self.median_)
        filled = np.where(np.isnan(values), self.median_.astype(np.float32), values)
        del values
        self.mean_ = filled.mean(axis=0, dtype=np.float64)
        self.scale_ = filled.std(axis=0, dtype=np.float64)
        self.scale_[self.scale_ == 0] = 1.0                 # as StandardScaler for constant columns
        del filled
        self.stocks_ = np.sort(X.stock_id.unique())
        p = 1 + len(self.numeric) + len(self.stocks_)
        G, b = np.zeros((p, p)), np.zeros(p)
        for i in range(0, len(X), self.chunk):
            D = self._design(X.iloc[i:i + self.chunk])
            G += D.T @ D; b += D.T @ y[i:i + self.chunk]
        penalty = np.full(p, self.alpha); penalty[0] = 0.0     # intercept is not penalised
        self.coef_ = np.linalg.solve(G + np.diag(penalty), b)
        return self

    def predict(self, X):
        return np.concatenate([self._design(X.iloc[i:i + self.chunk]) @ self.coef_
                               for i in range(0, len(X), self.chunk)])


class MedianBaseline:
    """Training-only median of the target: global, or per stock with a global fallback."""
    def __init__(self, by_stock=False):
        self.by_stock = by_stock

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float64)
        self.global_ = float(np.median(y))
        if self.by_stock:
            self.stock_ = pd.Series(y).groupby(X.stock_id.to_numpy()).median()
        return self

    def predict(self, X):
        if not self.by_stock:
            return np.full(len(X), self.global_)
        return X.stock_id.map(self.stock_).fillna(self.global_).to_numpy(dtype=np.float64)


class ZeroBaseline:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class EarlyStoppedXGB:
    """XGBoost whose tree count is chosen on the LAST training days only.

    The final `inner_fraction` of the fitting dates is an internal validation set;
    the model is fitted on the earlier dates. Outer/holdout rows are never seen.
    """
    def __init__(self, columns, params, inner_fraction=0.15, rounds=100):
        self.numeric = [c for c in columns if c not in ("stock_id", "date_id")]
        self.params, self.inner_fraction, self.rounds = params, inner_fraction, rounds

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float64)
        days = np.sort(X.date_id.unique())
        inner = days[-max(1, int(len(days) * self.inner_fraction)):]
        val = X.date_id.isin(inner).to_numpy()
        self.median_ = X.loc[~val, self.numeric].median().fillna(0.0)
        fill = lambda part: part[self.numeric].fillna(self.median_)
        self.model = XGBRegressor(early_stopping_rounds=self.rounds, **self.params)
        self.model.fit(fill(X.loc[~val]), y[~val], eval_set=[(fill(X.loc[val]), y[val])], verbose=False)
        self.best_iteration_ = int(self.model.best_iteration)
        return self

    def predict(self, X):
        return self.model.predict(X[self.numeric].fillna(self.median_))


class Blend:
    """Equal-weight average of fitted member models (weights fixed in advance)."""
    def __init__(self, members, weights=None):
        self.members = members
        self.weights = np.full(len(members), 1 / len(members)) if weights is None else np.asarray(weights)

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        return self

    def predict(self, X):
        return sum(w * np.asarray(m.predict(X), dtype=np.float64) for w, m in zip(self.weights, self.members))


class CrossSectionCentered:
    """Subtract each timestamp's cross-sectional mean prediction (no fitted parameters).

    Requires complete timestamp batches, as in Kaggle's API and the batch evaluation.
    """
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        p = np.asarray(self.model.predict(X), dtype=np.float64)
        key = X.date_id.to_numpy().astype(np.int64) * 1000 + X.seconds_in_bucket.to_numpy().astype(np.int64)
        return p - pd.Series(p).groupby(key).transform("mean").to_numpy()


def build_model(name, columns, params=None, seed=42, threads=2):
    p = dict(params or {})
    if name == "zero":
        return ZeroBaseline()
    if name in ("global_median", "stock_median"):
        return MedianBaseline(by_stock=name == "stock_median")
    if name in ("lstm", "mlp"):
        if name == "mlp":
            p["arch"] = "mlp"
        from .neural import LSTMRegressor
        return LSTMRegressor(seed=seed, threads=threads, **p)
    if name == "ridge":
        return ChunkedRidge(columns, alpha=p.pop("alpha", 100.0), **p)
    numeric = [c for c in columns if c not in ("stock_id", "date_id")]
    impute = SimpleImputer(strategy="median", keep_empty_features=True)
    # stock ID is a category indicator for the neural/ridge models; omitted
    # for trees rather than imposing an artificial numeric order.
    prep = ColumnTransformer([("numeric", impute, numeric)])
    if name == "random_forest":
        defaults = dict(n_estimators=150, max_depth=12, min_samples_leaf=30,
                        max_features=0.7, n_jobs=threads, random_state=seed)
        defaults.update(p)
        estimator = RandomForestRegressor(**defaults)
    elif name == "xgboost":
        defaults = dict(n_estimators=500, max_depth=6, learning_rate=0.04,
                        min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
                        reg_lambda=10, reg_alpha=0.1, tree_method="hist",
                        objective="reg:absoluteerror", eval_metric="mae",
                        n_jobs=threads, random_state=seed)
        early = p.pop("early_stopping_rounds", None)
        defaults.update(p)
        if early:
            return EarlyStoppedXGB(columns, defaults, rounds=early)
        estimator = XGBRegressor(**defaults)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocess", prep), ("model", estimator)])

