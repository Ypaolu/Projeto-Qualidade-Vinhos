"""Ciclo MLOps: ETL -> votação -> modelos -> MLflow -> promoção -> teste."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from src.common import MLRUNS, MODELS, PROCESSED, REPORTS, ensure_dirs, get_config, write_json
from src.etl import TARGET, prepare_data
from src.features import vote_features
from src.modeling import build_models, evaluate


def load_partition(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / f"{name}.csv")


def run_cycle(requested: str = "auto") -> dict:
    ensure_dirs()
    state_path = REPORTS / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"last_cycle": 0}
    last = int(state["last_cycle"])
    cycle = last + 1 if requested == "auto" else int(requested)
    if cycle <= last:
        return {"status": "already_completed", "cycle": cycle}
    if cycle > 2:
        return {"status": "no_new_batch", "cycle": cycle}
    if cycle != last + 1:
        raise ValueError(f"Ciclo inválido. Próximo ciclo permitido: {last + 1}")

    config = get_config()
    manifest_path = REPORTS / "dataset_manifest.json"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    manifest = prepare_data()
    if last > 0 and old_manifest and manifest["source_sha256"] != old_manifest["source_sha256"]:
        raise RuntimeError("O arquivo bruto mudou entre os ciclos. Não compare conjuntos de dados distintos.")

    batches = [load_partition("batch_01")]
    if cycle == 2:
        batches.append(load_partition("batch_02"))
    train = pd.concat(batches, ignore_index=True)
    valid = load_partition("validation")
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_valid, y_valid = valid.drop(columns=[TARGET]), valid[TARGET]

    selected, votes = vote_features(X_train, y_train)
    votes_path = REPORTS / f"votes_cycle_{cycle}.csv"
    votes.to_csv(votes_path, index=False)

    mlflow.set_tracking_uri(MLRUNS.resolve().as_uri())
    mlflow.set_experiment("WineQuality_N3")
    fitted = {}
    run_results = []
    for label, estimator in build_models().items():
        with mlflow.start_run(run_name=f"ciclo_{cycle}_{label}") as run:
            estimator.fit(X_train[selected], y_train)
            metrics = evaluate(estimator, X_valid[selected], y_valid)
            mlflow.log_params({
                "cycle": cycle,
                "algorithm": label,
                "train_rows": len(train),
                "feature_count": len(selected),
                "k_features_per_method": config["k_features_per_method"],
                "minimum_votes": config["minimum_votes"],
                "random_state": config["random_state"],
                "rf_n_estimators": config["rf_n_estimators"],
                "ridge_alpha": config["ridge_alpha"],
                "dataset_sha256": manifest["source_sha256"],
            })
            mlflow.log_metrics({
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
            })
            mlflow.set_tags({"dataset": "UCI winequality-red.csv", "evaluation_set": "fixed_validation"})
            mlflow.log_artifact(str(votes_path), artifact_path="feature_selection")
            mlflow.sklearn.log_model(estimator, artifact_path="model")
            fitted[label] = estimator
            run_results.append({
                "cycle": cycle,
                "model": label,
                "train_rows": len(train),
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "selected_features": ", ".join(selected),
                "run_id": run.info.run_id,
            })

    winner = min(run_results, key=lambda record: record["validation_mae"])
    incumbent_file = MODELS / "current.joblib"
    old_mae = None
    if incumbent_file.exists():
        incumbent = joblib.load(incumbent_file)  # Apenas arquivo confiável do projeto.
        old_mae = evaluate(
            incumbent["estimator"], X_valid[incumbent["features"]], y_valid
        )["mae"]
    promoted = old_mae is None or (
        winner["validation_mae"] < old_mae - config["min_mae_improvement"]
    )
    if promoted:
        joblib.dump({"estimator": fitted[winner["model"]], "features": selected}, incumbent_file)
        write_json(MODELS / "current.json", {
            "algorithm": winner["model"],
            "cycle": cycle,
            "validation_mae": winner["validation_mae"],
            "selected_features": selected,
            "features": selected,
            "mlflow_run_id": winner["run_id"],
            "dataset_sha256": manifest["source_sha256"],
            "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    for record in run_results:
        record["winner_of_cycle"] = record["model"] == winner["model"]
        record["promoted"] = bool(promoted and record["winner_of_cycle"])
        record["incumbent_mae_before"] = old_mae
    history_file = REPORTS / "history.csv"
    new_history = pd.DataFrame(run_results)
    history = pd.concat([pd.read_csv(history_file), new_history], ignore_index=True) if history_file.exists() else new_history
    history.to_csv(history_file, index=False)

    summary = {
        "status": "completed",
        "cycle": cycle,
        "training_rows": len(train),
        "selected_features": selected,
        "winner": winner["model"],
        "winner_validation_mae": winner["validation_mae"],
        "incumbent_validation_mae_before": old_mae,
        "promoted": bool(promoted),
        "results": run_results,
    }
    if cycle == 2:
        # Uso único do teste para o modelo efetivamente vigente, depois da decisão.
        test = load_partition("test")
        package = joblib.load(incumbent_file)
        estimate = package["estimator"].predict(test[package["features"]])
        metrics = evaluate(package["estimator"], test[package["features"]], test[TARGET])
        pd.DataFrame({
            "actual_quality": test[TARGET].to_numpy(),
            "predicted_quality": estimate,
            "abs_error": abs(test[TARGET].to_numpy() - estimate),
            "residual": test[TARGET].to_numpy() - estimate,
        }).to_csv(REPORTS / "test_predictions.csv", index=False)
        active_meta = json.loads((MODELS / "current.json").read_text(encoding="utf-8"))
        write_json(REPORTS / "final_test.json", {
            "algorithm": active_meta["algorithm"],
            "model_promoted_in_cycle": active_meta["cycle"],
            "test_rows": len(test),
            **metrics,
        })
        summary["final_test_mae"] = metrics["mae"]

    write_json(REPORTS / f"summary_cycle_{cycle}.json", summary)
    write_json(state_path, {
        "last_cycle": cycle,
        "last_execution_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": manifest["source_sha256"],
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa os ciclos 1 e 2 da demonstração MLOps")
    parser.add_argument("--cycle", choices=["auto", "1", "2"], default="auto")
    args = parser.parse_args()
    print(json.dumps(run_cycle(args.cycle), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
