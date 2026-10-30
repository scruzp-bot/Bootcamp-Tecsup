"""Entrenamiento reproducible con puerta de calidad y ficha del modelo."""
import json
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, data


class QualityGateError(RuntimeError):
    """El modelo no alcanza la métrica mínima y no se guarda."""


def train(df: pd.DataFrame, model_path=config.MODEL_PATH,
          metrics_path=config.METRICS_PATH, min_f1=config.MIN_F1_MACRO) -> dict:
    df = data.validate(df)
    X, y = df[config.FEATURES], df[config.TARGET].str.upper()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=config.SEED, **config.MODEL_PARAMS)),
    ])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "f1_macro": round(float(f1_score(y_te, pred, average="macro")), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
    }
    card = {
        "model": "LogisticRegression (multinomial) + StandardScaler",
        "version": config.MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config.SEED,
        "params": config.MODEL_PARAMS,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "features": config.FEATURES,
        "classes": list(model.classes_),
        "metric_main": "f1_macro",
        "min_f1_macro": min_f1,
        "metrics": metrics,
        "dataset": {"source": "UCI Dry Bean (id 602)", "doi": config.DATA_DOI},
    }
    if metrics["f1_macro"] < min_f1:
        raise QualityGateError(
            f"F1 macro {metrics['f1_macro']} < mínimo {min_f1}: modelo NO guardado")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(card, indent=2, ensure_ascii=False))
    return card


def main() -> None:
    try:
        card = train(data.load())
    except QualityGateError as exc:
        print(f"[puerta de calidad] {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(card["metrics"], indent=2))
    print(f"Modelo guardado en {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
