"""Seleção independente por três métodos e voto mínimo 2/3."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.common import get_config


def vote_features(X: pd.DataFrame, y: pd.Series) -> tuple[list[str], pd.DataFrame]:
    cfg = get_config()
    cols = list(X.columns)
    k = min(cfg["k_features_per_method"], len(cols))
    if not cols or len(X) != len(y):
        raise ValueError("Treino inválido ou sem atributos.")

    kbest = SelectKBest(score_func=f_regression, k=k).fit(X, y)
    selected_kbest = set(X.columns[kbest.get_support()])

    forest = RandomForestRegressor(
        n_estimators=cfg["rf_n_estimators"],
        random_state=cfg["random_state"], n_jobs=-1,
    ).fit(X, y)
    importances = pd.Series(forest.feature_importances_, index=cols)
    selected_forest = set(importances.sort_values(ascending=False).head(k).index)

    train_scaled = StandardScaler().fit_transform(X)
    rfe = RFE(Ridge(alpha=cfg["ridge_alpha"]), n_features_to_select=k).fit(train_scaled, y)
    selected_rfe = set(X.columns[rfe.get_support()])

    rows = []
    for col in cols:
        a, b, c = (int(col in selected_kbest), int(col in selected_forest), int(col in selected_rfe))
        rows.append({"feature": col, "SelectKBest": a, "RandomForest": b,
                     "RFE": c, "votes": a + b + c, "selected": (a + b + c) >= cfg["minimum_votes"]})
    votes = pd.DataFrame(rows).sort_values(
        ["votes", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    chosen = votes.loc[votes["selected"], "feature"].tolist()
    if not chosen:
        raise ValueError("Nenhum atributo aprovado por pelo menos 2 métodos.")
    return chosen, votes
