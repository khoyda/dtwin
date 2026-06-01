"""Scikit-learn yield-prediction model wrapper.

Trains on season-level features (see :mod:`canola_dt.features`) to predict
end-of-season yield in kg/ha. Kept deliberately simple and swappable via config.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_estimator(model_cfg: dict):
    """Instantiate a regressor from the ``model`` section of the config."""
    kind = model_cfg.get("type", "random_forest")
    rs = model_cfg.get("random_state", 42)
    if kind == "random_forest":
        params = model_cfg.get("random_forest", {})
        return RandomForestRegressor(random_state=rs, **params)
    if kind == "gradient_boosting":
        params = model_cfg.get("gradient_boosting", {})
        return GradientBoostingRegressor(random_state=rs, **params)
    if kind == "linear":
        return LinearRegression()
    raise ValueError(f"unknown model type: {kind!r}")


class YieldModel:
    """Fit/predict/persist wrapper around a scikit-learn pipeline."""

    def __init__(self, model_cfg: dict):
        self.cfg = model_cfg
        self.target = model_cfg.get("target", "yield_kg_ha")
        self.feature_names: list[str] | None = None
        self.pipeline = Pipeline(
            [("scale", StandardScaler()), ("est", build_estimator(model_cfg))]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Train with a held-out split; returns validation metrics."""
        self.feature_names = list(X.columns)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=self.cfg.get("test_size", 0.2),
            random_state=self.cfg.get("random_state", 42),
        )
        self.pipeline.fit(X_tr, y_tr)
        pred = self.pipeline.predict(X_te)
        return {
            "r2": float(r2_score(y_te, pred)),
            "mae_kg_ha": float(mean_absolute_error(y_te, pred)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
        }

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict[str, float]:
        """K-fold CV metrics — more honest than a single split on a small dataset.

        Uses a fresh pipeline per fold so scaler/estimator never see held-out rows.
        """
        n_splits = min(n_splits, len(X))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.cfg.get("random_state", 42))
        r2 = cross_val_score(self.pipeline, X, y, cv=kf, scoring="r2")
        mae = -cross_val_score(self.pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error")
        return {
            "cv_r2_mean": float(r2.mean()),
            "cv_r2_std": float(r2.std()),
            "cv_mae_mean": float(mae.mean()),
            "n_splits": int(n_splits),
        }

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.feature_names is not None:
            X = X[self.feature_names]
        return pd.Series(self.pipeline.predict(X), index=X.index, name=self.target)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "features": self.feature_names}, path)

    @classmethod
    def load(cls, path: str | Path, model_cfg: dict) -> "YieldModel":
        obj = cls(model_cfg)
        blob = joblib.load(path)
        obj.pipeline = blob["pipeline"]
        obj.feature_names = blob["features"]
        return obj
