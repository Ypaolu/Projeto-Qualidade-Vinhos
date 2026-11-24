# Qualidade de Vinhos — Regressão com MLOps (N3)

Projeto acadêmico com base **Wine Quality (vinhos tintos)** da [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality). O alvo `quality` é uma **nota numérica**: o trabalho é de **regressão**, não de classificação.

## O que foi implementado

- ETL do CSV UCI (`sep=";"`), validação de 11 atributos, conversão numérica, remoção de nulos/duplicatas, SHA-256 e manifesto;
- Partição determinística **60% treino, 20% validação e 20% teste**, com dois lotes de treinamento, sem vazamento;
- Votação de atributos com **SelectKBest (f_regression)**, **Random Forest (importância)** e **RFE (Ridge com padronização)**; cada método indica 6 atributos; entram os que recebem pelo menos 2 votos;
- Treinamento de **Ridge** e **RandomForestRegressor**, com seleção refita em cada ciclo;
- Comparação do **MAE** na validação e RMSE como métrica complementar;
- **MLflow** com quatro runs (dois modelos × dois ciclos), parâmetros, métricas e artefatos;
- **Promoção apenas quando o candidato melhora o MAE** na validação fixa, sem trocar o vigente por pior desempenho;
- Avaliação única no teste isolado ao fim do ciclo 2, previsão com o modelo vigente, relatório e figuras;
- Dashboard **Streamlit**; **GitHub Actions** com execução manual e semanal; testes **pytest**;
- Notebook didático executável `notebooks/00_Wine_Quality_N3_Completo.ipynb` e cinco roteiros complementares.

> **Importante:** o dataset é estático. Os dois ciclos são lotes progressivos simulados a partir do mesmo arquivo, não uma entrada de vinhos em tempo real.

## Ambiente

Recomendado **Python 3.11**. No PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Executar do zero

1. Abra a pasta raiz do projeto no VS Code ou terminal (não rode dentro de `src/`).
2. Conecte-se à internet para obter `winequality-red.csv` na primeira execução; se não for possível, baixe manualmente pela [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality) e salve em `data/raw/winequality-red.csv`.
3. Execute:

```bash
python -m src.etl
python -m src.pipeline --cycle 1
python -m src.pipeline --cycle 2
pytest -q
mlflow ui --backend-store-uri ./mlruns
```

Abra a UI do MLflow em `http://127.0.0.1:5000`. Em outro terminal:

```bash
streamlit run dashboard.py
```

Para trabalhar por interface visual: abra `notebooks/00_Wine_Quality_N3_Completo.ipynb` no Jupyter/VS Code e execute as células de cima para baixo. O notebook também roda sozinho em uma pasta vazia, criando seus próprios artefatos, quando as dependências e o CSV da UCI estão disponíveis. **Não execute o notebook e o pipeline do projeto na mesma pasta como se fossem o mesmo estado sem limpar/organizar os relatórios**; use cada fluxo como demonstração alternativa.

## Arquivos de resultado

| Local | Evidência |
|---|---|
| `reports/dataset_manifest.json` | origem, hash, contagens, partições |
| `reports/votes_cycle_1.csv` / `votes_cycle_2.csv` | votos de cada seletor |
| `reports/history.csv` | MAE por modelo e ciclo, promoção |
| `models/current.joblib` + `models/current.json` | estimador vigente e proveniência |
| `reports/summary_cycle_*.json` | decisão de cada ciclo |
| `reports/final_test.json`, `test_predictions.csv` | teste isolado |
| `mlruns/` | experimentos e quatro execuções |
| `evidencias/` | capturas de execução e figuras |

`data/raw`, `data/processed`, `reports`, `models`, `mlruns` são preenchidos na **execução real**. Não há métricas ou capturas fictícias no pacote.

## GitHub Actions

O arquivo `.github/workflows/pipeline_mlops.yml` agenda execução às segundas (12h13 UTC, 09h13 de Brasília) e oferece `workflow_dispatch` manual. Cada execução `--cycle auto` usa o próximo lote. Após completar os dois lotes, o status é `no_new_batch`, evitando fingir novas amostras. Deve-se permitir `contents: write` para persistir artefatos na branch padrão. Históricos MLflow locais versionados são aceitos aqui **apenas como demonstração acadêmica**; produção exige servidor/armazenamento persistentes. O GitHub Actions **não foi executado nem configurado em repositório real** ao gerar este pacote.

## Metodologia e limites

A nota `quality` deriva de avaliação sensorial no conjunto; o modelo aproxima esta nota usando medições físico-químicas. Os atributos são selecionados **sem consultar validação e teste**; o scaler Ridge é ajustado sobre dados de treino. Comparar em validação repetidas vezes pode induzir algum ajuste às características dessa partição; por isso o teste fica isolado até o encerramento. O resultado do modelo não substitui degustação profissional. Teste da qualidade preditiva requer executar com dados UCI reais.

**Fonte:** Cortez et al. (2009), Wine Quality — [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality).
