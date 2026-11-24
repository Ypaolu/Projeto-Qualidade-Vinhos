"""Aquisição UCI, limpeza e partições determinísticas sem vazamento."""
from __future__ import annotations

import hashlib
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import PROCESSED, RAW, REPORTS, ensure_dirs, get_config, write_json

TARGET = "quality"
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]


def prepare_data() -> dict:
    """Separa treino em 2 lotes e fixa validação/teste para ambos os ciclos."""
    ensure_dirs()
    config = get_config()
    source = RAW / "winequality-red.csv"
    if not source.exists():
        try:
            urlretrieve(config["dataset_url"], source)
        except Exception as error:
            source.unlink(missing_ok=True)
            raise RuntimeError(
                "Falha ao baixar a base UCI. Baixe 'winequality-red.csv' da UCI "
                "e salve em data/raw/winequality-red.csv antes de executar novamente."
            ) from error

    # A origem usa ';'. Não substituir silenciosamente outra base.
    df = pd.read_csv(source, sep=";")
    expected = FEATURES + [TARGET]
    if sorted(df.columns) != sorted(expected):
        raise ValueError(
            f"Colunas inesperadas. Esperado: {expected}. Encontrado: {list(df.columns)}"
        )
    total_original = len(df)
    df = df[expected].apply(pd.to_numeric, errors="coerce")
    missing_count = int(df.isna().any(axis=1).sum())
    df = df.dropna()
    repeated_count = int(df.duplicated().sum())
    df = df.drop_duplicates()
    invalid_target = int((~df[TARGET].between(0, 10)).sum())
    df = df.loc[df[TARGET].between(0, 10)].reset_index(drop=True)
    if len(df) < 50:
        raise ValueError("Dados insuficientes para treino, validação e teste.")

    # Divisão total: treino 60%, validação 20%, teste 20%.
    dev, test = train_test_split(
        df, test_size=config["test_size"], random_state=config["random_state"]
    )
    train, validation = train_test_split(
        dev,
        test_size=config["validation_size_from_dev"],
        random_state=config["random_state"],
    )
    train = train.sample(frac=1, random_state=config["random_state"])
    half = len(train) // 2
    parts = {
        "batch_01": train.iloc[:half],
        "batch_02": train.iloc[half:],
        "validation": validation,
        "test": test,
    }
    # Checagem por índice antes de remover o índice na gravação.
    idx_sets = [set(part.index) for part in parts.values()]
    assert len(set.union(*idx_sets)) == len(df)
    assert sum(map(len, idx_sets)) == len(df)

    for name, rows in parts.items():
        rows.to_csv(PROCESSED / f"{name}.csv", index=False)

    manifest = {
        "dataset": "UCI Wine Quality — red wine",
        "source_url": config["dataset_url"],
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "original_rows": total_original,
        "removed_rows_with_missing": missing_count,
        "removed_duplicate_rows": repeated_count,
        "removed_invalid_quality": invalid_target,
        "clean_rows": len(df),
        "feature_columns": FEATURES,
        "target": TARGET,
        "parts": {name: len(part) for name, part in parts.items()},
        "seed": config["random_state"],
        "protocol": "lote1 -> lote1+lote2; validacao fixa; teste isolado ate ciclo2",
    }
    write_json(REPORTS / "dataset_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(prepare_data())
