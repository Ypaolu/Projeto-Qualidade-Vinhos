"""Painel Streamlit do experimento e previsão do modelo vigente."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.common import MODELS, PROCESSED, REPORTS
from src.predict import predict_one

st.set_page_config(page_title="Wine Quality | MLOps N3", page_icon="🍷", layout="wide")
st.title("Qualidade dos Vinhos — MLOps N3")
st.caption("UCI Wine Quality (tintos) • Regressão de quality • MAE menor significa melhor precisão")

history_file = REPORTS / "history.csv"
current_file = MODELS / "current.json"
if not history_file.exists() or not current_file.exists():
    st.warning("Execute primeiro: python -m src.pipeline --cycle 1")
    st.stop()

history = pd.read_csv(history_file)
current = json.loads(current_file.read_text(encoding="utf-8"))

m1, m2, m3 = st.columns(3)
m1.metric("Modelo vigente", current["algorithm"])
m2.metric("MAE validação (vigente)", f"{current['validation_mae']:.4f}")
m3.metric("Ciclo de promoção", str(current["cycle"]))

st.subheader("Comparação de erros na validação")
wide = history.pivot(index="cycle", columns="model", values="validation_mae")
st.bar_chart(wide)
st.dataframe(
    history[["cycle", "model", "train_rows", "validation_mae", "validation_rmse", "winner_of_cycle", "promoted"]],
    use_container_width=True,
)

st.subheader("Votação e atributos selecionados")
cycle = int(history["cycle"].max())
cycle_label = st.selectbox("Visualizar votação do ciclo", sorted(history["cycle"].unique(), reverse=True))
votes_file = REPORTS / f"votes_cycle_{int(cycle_label)}.csv"
if votes_file.exists():
    votes = pd.read_csv(votes_file)
    st.dataframe(votes, use_container_width=True, hide_index=True)
    st.bar_chart(votes.set_index("feature")[["SelectKBest", "RandomForest", "RFE"]])
st.write("**Atributos do modelo vigente:**", ", ".join(current["features"]))

st.subheader("Estimar a qualidade de um vinho")
batch_file = PROCESSED / "batch_01.csv"
if batch_file.exists():
    medians = pd.read_csv(batch_file).median(numeric_only=True)
    with st.form("predict"):
        measurements = {}
        columns = st.columns(2)
        for i, feature in enumerate(current["features"]):
            with columns[i % 2]:
                measurements[feature] = st.number_input(
                    feature, value=float(medians[feature]), format="%.4f"
                )
        submitted = st.form_submit_button("Estimar nota")
    if submitted:
        st.success(f"Nota estimada: {predict_one(measurements):.3f} (escala do alvo quality)")
        st.caption("É uma estimativa estatística; não substitui degustação ou avaliação profissional.")

final_file = REPORTS / "final_test.json"
pred_file = REPORTS / "test_predictions.csv"
if final_file.exists():
    final = json.loads(final_file.read_text(encoding="utf-8"))
    st.subheader("Avaliação final: teste isolado")
    st.metric("MAE final no teste", f"{final['mae']:.4f}")
    st.caption("Teste reservado e utilizado somente após a conclusão do segundo ciclo.")
    if pred_file.exists():
        st.dataframe(pd.read_csv(pred_file).head(30), use_container_width=True, hide_index=True)
