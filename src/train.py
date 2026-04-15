from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd

from src.config import FAST_RUN, LABEL_MAP, SEED
from src.data import prepare_splits
from src.evaluation import build_results_table, compute_metrics
from src.features import add_feature_views
from src.models import train_cnn1d, train_dummy, train_logreg, train_mlp, train_random_forest


def compare_models(seed: int = SEED, fast_run: bool = FAST_RUN) -> tuple[dict, pd.DataFrame]:
    data = add_feature_views(prepare_splits(random_state=seed, fast_run=fast_run))
    results = []
    trained_models = {}

    y_pred_dummy = train_dummy(data, seed)
    results.append({"modelo": "Dummy - clase mayoritaria", "split": "val", **compute_metrics(data["y_val"], y_pred_dummy)})

    y_pred_logreg = train_logreg(data, seed)
    results.append({"modelo": "LogReg multinomial", "split": "val", **compute_metrics(data["y_val"], y_pred_logreg)})

    rf_model, y_pred_rf = train_random_forest(data, seed, fast_run=fast_run)
    trained_models["RandomForest + features"] = rf_model
    results.append({"modelo": "RandomForest + features", "split": "val", **compute_metrics(data["y_val"], y_pred_rf)})

    mlp_model, y_pred_mlp = train_mlp(data, seed, fast_run=fast_run)
    trained_models["MLP"] = mlp_model
    results.append({"modelo": "MLP", "split": "val", **compute_metrics(data["y_val"], y_pred_mlp)})

    cnn_model, y_pred_cnn = train_cnn1d(data, seed, fast_run=fast_run)
    trained_models["CNN 1D"] = cnn_model
    results.append({"modelo": "CNN 1D", "split": "val", **compute_metrics(data["y_val"], y_pred_cnn)})

    summary = build_results_table(results, split="val")

    return {
        "data": data,
        "results": results,
        "trained_models": trained_models,
    }, summary


def predict_with_model_name(model_name: str, bundle: dict, split: str = "test") -> np.ndarray:
    data = bundle["data"]
    trained_models = bundle["trained_models"]

    if model_name == "RandomForest + features":
        return trained_models[model_name].predict(data[f"X_{split}_feat"])
    if model_name == "MLP":
        return np.argmax(trained_models[model_name].predict(data[f"X_{split}_scaled"], verbose=0), axis=1)
    if model_name == "CNN 1D":
        return np.argmax(trained_models[model_name].predict(data[f"X_{split}_cnn"], verbose=0), axis=1)
    raise ValueError(f"Modelo no soportado para inferencia final: {model_name}")


def save_artifacts(bundle: dict, val_summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle["trained_models"]["RandomForest + features"], output_dir / "random_forest_ecg.joblib")
    bundle["trained_models"]["MLP"].save(output_dir / "mlp.keras")
    bundle["trained_models"]["CNN 1D"].save(output_dir / "cnn_1d.keras")
    joblib.dump(bundle["data"]["scaler"], output_dir / "scaler.joblib")

    rows = []
    for _, row in val_summary.iterrows():
        rows.append(row.to_dict())

    for model_name in bundle["trained_models"]:
        y_test_pred = predict_with_model_name(model_name, bundle, split="test")
        rows.append(
            {
                "modelo": model_name,
                "split": "test",
                **compute_metrics(bundle["data"]["y_test"], y_test_pred),
            }
        )

    metrics_summary = pd.DataFrame(rows)
    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-run", action="store_true", help="Entrenamiento reducido para generar artefactos mas rapido")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models"

    bundle, summary = compare_models(fast_run=args.fast_run or FAST_RUN)
    best_model_name = summary.iloc[0]["modelo"]
    test_pred = predict_with_model_name(best_model_name, bundle, split="test")
    test_metrics = compute_metrics(bundle["data"]["y_test"], test_pred)
    metrics_summary = save_artifacts(bundle, summary, model_dir)

    print("Comparativa en validacion:")
    print(summary.round(4).to_string(index=False))
    print()
    print(f"Mejor modelo: {best_model_name}")
    print(pd.DataFrame([{"modelo": best_model_name, "split": "test", **test_metrics}]).round(4).to_string(index=False))
    print()
    print(f"Etiquetas disponibles: {LABEL_MAP}")
    print()
    print("Artefactos guardados:")
    print("- models/random_forest_ecg.joblib")
    print("- models/mlp.keras")
    print("- models/cnn_1d.keras")
    print("- models/scaler.joblib")
    print("- models/metrics_summary.csv")
    print()
    print(metrics_summary.round(4).to_string(index=False))
