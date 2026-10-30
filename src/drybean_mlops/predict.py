"""Carga del modelo (una sola vez) y predicción."""
import json
from functools import lru_cache

import joblib
import pandas as pd

from . import config


class ModelNotFound(FileNotFoundError):
    pass


@lru_cache(maxsize=4)
def load_model(path=config.MODEL_PATH):
    if not path.exists():
        raise ModelNotFound(f"No existe {path}. Ejecuta: uv run entrenar")
    return joblib.load(path)


def model_info(path=config.METRICS_PATH) -> dict:
    if not path.exists():
        raise ModelNotFound(f"No existe {path}. Ejecuta: uv run entrenar")
    return json.loads(path.read_text())


def predict(features: dict, model_path=config.MODEL_PATH) -> dict:
    model = load_model(model_path)
    row = pd.DataFrame([features])[config.FEATURES]
    proba = model.predict_proba(row)[0]
    best = int(proba.argmax())
    return {
        "variety": str(model.classes_[best]),
        "confidence": round(float(proba[best]), 4),
        "probabilities": {str(c): round(float(p), 4) for c, p in zip(model.classes_, proba)},
    }
