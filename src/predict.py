"""Predição única usando os atributos exigidos pelo modelo promovido."""
from __future__ import annotations

import json

import joblib
import pandas as pd

from src.common import MODELS


def predict_one(measurements: dict) -> float:
    path = MODELS / "current.joblib"
    if not path.exists():
        raise FileNotFoundError("Execute ao menos o ciclo 1 antes de prever.")
    package = joblib.load(path)  # Somente artefatos locais confiáveis.
    features = package["features"]
    missing = sorted(set(features) - set(measurements))
    if missing:
        raise ValueError(f"Medições ausentes: {missing}")
    data = pd.DataFrame([{feature: float(measurements[feature]) for feature in features}])
    return float(package["estimator"].predict(data)[0])


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit('Uso: python -m src.predict \'{"alcohol": 12.5, "pH": 3.2, ...}\'')
    print(f"Nota prevista: {predict_one(json.loads(sys.argv[1])):.3f}")
