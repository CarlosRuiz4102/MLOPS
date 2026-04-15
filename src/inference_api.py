from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import LABEL_MAP
from src.features import extract_time_features


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATHS = {
    "random_forest": MODELS_DIR / "random_forest_ecg.joblib",
    "mlp": MODELS_DIR / "mlp.keras",
    "cnn_1d": MODELS_DIR / "cnn_1d.keras",
}
SCALER_PATH = MODELS_DIR / "scaler.joblib"

app = FastAPI(title="ECG Inference API", version="1.0.0")


class ECGRequest(BaseModel):
    signal: list[float] = Field(..., min_length=187, max_length=187)
    model_name: str = "random_forest"


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            "No se encuentran los artefactos entrenados. Ejecuta primero `python src/train.py`."
        )
    return path


def load_model(model_name: str):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Modelo no soportado: {model_name}")

    path = require_path(MODEL_PATHS[model_name])
    if model_name == "random_forest":
        return joblib.load(path)
    return tf.keras.models.load_model(path)


def load_scaler():
    return joblib.load(require_path(SCALER_PATH))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: ECGRequest) -> dict:
    model = load_model(request.model_name)
    signal = np.asarray([request.signal], dtype=np.float32)

    if request.model_name == "random_forest":
        inputs = extract_time_features(signal)
        pred = int(model.predict(inputs)[0])
    elif request.model_name == "mlp":
        scaler = load_scaler()
        inputs = scaler.transform(signal)
        pred = int(np.argmax(model.predict(inputs, verbose=0), axis=1)[0])
    elif request.model_name == "cnn_1d":
        inputs = signal[..., np.newaxis]
        pred = int(np.argmax(model.predict(inputs, verbose=0), axis=1)[0])
    else:
        raise ValueError(f"Modelo no soportado: {request.model_name}")

    return {
        "model_name": request.model_name,
        "predicted_class": pred,
        "predicted_label": LABEL_MAP[pred],
    }
