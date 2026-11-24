"""Testa a divisão sem necessidade de dados externos."""
import pandas as pd
from sklearn.model_selection import train_test_split


def test_split_sem_sobreposicao():
    df = pd.DataFrame({"id": range(100)})
    dev, test = train_test_split(df, test_size=0.2, random_state=42)
    train, valid = train_test_split(dev, test_size=0.25, random_state=42)
    groups = [set(p["id"]) for p in [train, valid, test]]
    assert [len(s) for s in groups] == [60, 20, 20]
    assert not (groups[0] & groups[1])
    assert not (groups[0] & groups[2])
    assert not (groups[1] & groups[2])
