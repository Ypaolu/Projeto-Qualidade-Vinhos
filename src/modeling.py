"""Regressores comparados pela MAE de validação."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import get_config


def build_models() -> dict:
    cfg = get_config()
    return {
        "Ridge": Pipeline([
            ("standard_scaler", StandardScaler()),
            ("ridge", Ridge(alpha=cfg["ridge_alpha"])),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=cfg["rf_n_estimators"],
            random_state=cfg["random_state"], n_jobs=-1,
        ),
    }


def evaluate(estimator, X, y) -> dict:
    pred = estimator.predict(X)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
    }
