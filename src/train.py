from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from dotenv import load_dotenv

load_dotenv()
from src.config import FAST_RUN, LABEL_MAP, SEED
from src.data import prepare_splits
from src.evaluation import build_results_table, compute_metrics
from src.features import add_feature_views
from src.models import train_cnn1d, train_dummy, train_logreg, train_mlp, train_random_forest

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class _WandbEpochLogger(tf.keras.callbacks.Callback):
    """Keras callback que registra métricas por época en un run activo de W&B."""

    def __init__(self, run, prefix: str):
        super().__init__()
        self._run = run
        self._prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        if logs and self._run:
            self._run.log({
                f"{self._prefix}/train_loss": logs.get("loss"),
                f"{self._prefix}/val_loss": logs.get("val_loss"),
                f"{self._prefix}/val_accuracy": logs.get("val_accuracy"),
                "epoch": epoch,
            })


_ALL_MODELS = ["dummy", "logreg", "random_forest", "mlp", "cnn"]


def compare_models(
    seed: int = SEED,
    fast_run: bool = FAST_RUN,
    wandb_run=None,
    active_models: list[str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    enabled = set(active_models if active_models is not None else _ALL_MODELS)
    data = add_feature_views(prepare_splits(random_state=seed, fast_run=fast_run))
    results = []
    trained_models = {}

    if "dummy" in enabled:
        y_pred_dummy = train_dummy(data, seed)
        _m = compute_metrics(data["y_val"], y_pred_dummy)
        results.append({"modelo": "Dummy - clase mayoritaria", "split": "val", **_m})
        if wandb_run:
            wandb_run.log({"dummy/val_macro_f1": _m["macro_f1"], "dummy/val_accuracy": _m["accuracy"]})

    if "logreg" in enabled:
        y_pred_logreg = train_logreg(data, seed)
        _m = compute_metrics(data["y_val"], y_pred_logreg)
        results.append({"modelo": "LogReg multinomial", "split": "val", **_m})
        if wandb_run:
            wandb_run.log({"logreg/val_macro_f1": _m["macro_f1"], "logreg/val_accuracy": _m["accuracy"]})

    if "random_forest" in enabled:
        rf_model, y_pred_rf = train_random_forest(data, seed, fast_run=fast_run)
        trained_models["RandomForest + features"] = rf_model
        _m = compute_metrics(data["y_val"], y_pred_rf)
        results.append({"modelo": "RandomForest + features", "split": "val", **_m})
        if wandb_run:
            wandb_run.log({"random_forest/val_macro_f1": _m["macro_f1"], "random_forest/val_accuracy": _m["accuracy"]})

    if "mlp" in enabled:
        mlp_cb = [_WandbEpochLogger(wandb_run, "mlp")] if wandb_run else None
        mlp_model, y_pred_mlp = train_mlp(data, seed, fast_run=fast_run, extra_callbacks=mlp_cb)
        trained_models["MLP"] = mlp_model
        _m = compute_metrics(data["y_val"], y_pred_mlp)
        results.append({"modelo": "MLP", "split": "val", **_m})
        if wandb_run:
            wandb_run.log({"mlp/val_macro_f1": _m["macro_f1"], "mlp/val_accuracy": _m["accuracy"]})

    if "cnn" in enabled:
        cnn_cb = [_WandbEpochLogger(wandb_run, "cnn")] if wandb_run else None
        cnn_model, y_pred_cnn = train_cnn1d(data, seed, fast_run=fast_run, extra_callbacks=cnn_cb)
        trained_models["CNN 1D"] = cnn_model
        _m = compute_metrics(data["y_val"], y_pred_cnn)
        results.append({"modelo": "CNN 1D", "split": "val", **_m})
        if wandb_run:
            wandb_run.log({"cnn/val_macro_f1": _m["macro_f1"], "cnn/val_accuracy": _m["accuracy"]})

    summary = build_results_table(results, split="val")

    if wandb_run:
        cols = list(summary.columns)
        table = wandb.Table(data=summary.values.tolist(), columns=cols)
        wandb_run.log({"val_comparison_table": table})

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


def save_artifacts(bundle: dict, val_summary: pd.DataFrame, output_dir: Path, wandb_run=None) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_map = {
        "RandomForest + features": lambda m: joblib.dump(m, output_dir / "random_forest_ecg.joblib"),
        "MLP": lambda m: m.save(output_dir / "mlp.keras"),
        "CNN 1D": lambda m: m.save(output_dir / "cnn_1d.keras"),
    }
    saved_files = [output_dir / "scaler.joblib"]
    for name, save_fn in _save_map.items():
        if name in bundle["trained_models"]:
            save_fn(bundle["trained_models"][name])
            artifact_path = {
                "RandomForest + features": output_dir / "random_forest_ecg.joblib",
                "MLP": output_dir / "mlp.keras",
                "CNN 1D": output_dir / "cnn_1d.keras",
            }[name]
            saved_files.append(artifact_path)
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

    if wandb_run:
        artifact = wandb.Artifact(
            name="ecg_models",
            type="model",
            description="Modelos ECG entrenados con scaler",
        )
        for fpath in saved_files:
            if fpath.exists():
                artifact.add_file(str(fpath))
        wandb_run.log_artifact(artifact)

        cols = list(metrics_summary.columns)
        test_table = wandb.Table(data=metrics_summary.values.tolist(), columns=cols)
        wandb_run.log({"test_metrics_summary": test_table})

    return metrics_summary


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-run", action="store_true", help="Entrenamiento reducido para generar artefactos mas rapido")
    parser.add_argument("--config", default=None, help="Ruta al fichero de configuracion YAML (ej: config/config.yaml)")
    parser.add_argument("--wandb-project", default="ecg-classification", help="Nombre del proyecto en W&B")
    parser.add_argument("--no-wandb", action="store_true", help="Desactivar logging con W&B")
    args = parser.parse_args()

    cfg = _load_config(args.config) if args.config else {}
    seed = int(cfg.get("seed", SEED))
    fast_run = bool(cfg.get("fast_run", False)) or args.fast_run or FAST_RUN
    active_models = cfg.get("models", None)  # None -> todos

    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models"

    wandb_run = None
    if not args.no_wandb and _WANDB_AVAILABLE:
        models_tag = "-".join(active_models) if active_models else "all"
        lr = cfg.get("mlp", {}).get("learning_rate", 0.001)
        epochs = cfg.get("mlp", {}).get("epochs", 30)
        run_name = f"{models_tag}-lr{lr}-e{epochs}-seed{seed}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={"seed": seed, "fast_run": fast_run, **cfg},
            job_type="training",
        )

    bundle, summary = compare_models(seed=seed, fast_run=fast_run, wandb_run=wandb_run, active_models=active_models)
    best_model_name = summary.iloc[0]["modelo"]
    test_pred = predict_with_model_name(best_model_name, bundle, split="test")
    test_metrics = compute_metrics(bundle["data"]["y_test"], test_pred)
    metrics_summary = save_artifacts(bundle, summary, model_dir, wandb_run=wandb_run)

    if wandb_run:
        wandb_run.summary["best_model"] = best_model_name
        wandb_run.summary["best_test_macro_f1"] = test_metrics["macro_f1"]
        wandb_run.summary["best_test_accuracy"] = test_metrics["accuracy"]
        wandb_run.finish()

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
