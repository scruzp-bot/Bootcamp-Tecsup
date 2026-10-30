"""Configuración central: rutas, semilla, parámetros y esquema de datos."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_PATH = DATA_DIR / "dry_bean.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Fuente: UCI ML Repository, dataset 602 (CC BY 4.0)
DATA_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
DATA_DOI = "10.24432/C50S4B"

SEED = 42
TEST_SIZE = 0.2
MODEL_VERSION = "1.0.0"
MODEL_PARAMS = {"C": 1.0, "max_iter": 2000}

# Puerta de calidad: no se guarda un modelo con F1 macro menor a este valor
MIN_F1_MACRO = 0.88

TARGET = "Class"
CLASSES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]

# Nombres canónicos de columnas (el original trae 'AspectRation' y 'roundness')
FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2",
    "ShapeFactor3", "ShapeFactor4",
]

# Rangos válidos (min, max) usados en la validación de datos y en la API
RANGES = {
    "Area": (1_000, 500_000),
    "Perimeter": (100, 3_000),
    "MajorAxisLength": (50, 1_000),
    "MinorAxisLength": (50, 1_000),
    "AspectRation": (1.0, 5.0),
    "Eccentricity": (0.0, 1.0),
    "ConvexArea": (1_000, 500_000),
    "EquivDiameter": (30, 900),
    "Extent": (0.0, 1.0),
    "Solidity": (0.0, 1.0),
    "roundness": (0.0, 1.0),
    "Compactness": (0.0, 1.0),
    "ShapeFactor1": (0.0, 0.1),
    "ShapeFactor2": (0.0, 0.01),
    "ShapeFactor3": (0.0, 1.0),
    "ShapeFactor4": (0.0, 1.0),
}
