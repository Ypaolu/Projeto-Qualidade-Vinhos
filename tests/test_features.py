"""Testes de lógica independentes de conexão com a UCI."""
import numpy as np
import pandas as pd

from src.features import vote_features
from src.modeling import build_models, evaluate


def test_votacao_2_em_3_e_treinamento():
    gen = np.random.default_rng(42)
    X = pd.DataFrame(gen.normal(size=(150, 11)), columns=[f"x_{i}" for i in range(11)])
    y = 2 * X["x_0"] - X["x_1"] + gen.normal(0, 0.2, 150)
    chosen, table = vote_features(X, y)
    assert len(chosen) >= 1
    assert set(chosen) == set(table.loc[table["votes"] >= 2, "feature"])
    assert (table[["SelectKBest", "RandomForest", "RFE"]].sum(axis=1) == table["votes"]).all()
    assert table["SelectKBest"].sum() == 6
    assert table["RandomForest"].sum() == 6
    assert table["RFE"].sum() == 6
    for name, model in build_models().items():
        model.fit(X[chosen], y)
        score = evaluate(model, X[chosen], y)
        assert score["mae"] >= 0 and np.isfinite(score["mae"]), name


def test_models_tem_os_dois_algoritmos():
    assert set(build_models()) == {"Ridge", "RandomForest"}
