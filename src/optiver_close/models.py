"""Four model families from the original work, with fold-local preprocessing."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


def build_model(name, columns, params=None, seed=42, threads=2):
    p = dict(params or {})
    if name == "lstm":
        from .neural import LSTMRegressor
        return LSTMRegressor(seed=seed, threads=threads, **p)
    numeric = [c for c in columns if c not in ("stock_id", "date_id")]
    impute = SimpleImputer(strategy="median", keep_empty_features=True)
    if name == "ridge":
        prep = ColumnTransformer([
            ("numeric", Pipeline([("impute", impute), ("scale", StandardScaler())]), numeric),
            ("stock", OneHotEncoder(handle_unknown="ignore"), ["stock_id"]),
        ])
        estimator = Ridge(alpha=p.pop("alpha", 100.0), solver="lsqr", **p)
    else:
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
            defaults.update(p)
            estimator = XGBRegressor(**defaults)
        else:
            raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocess", prep), ("model", estimator)])

