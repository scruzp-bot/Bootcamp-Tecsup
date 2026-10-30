"""Descarga, carga y validación de datos del Dry Bean Dataset (UCI 602)."""
import io
import zipfile

import pandas as pd
import requests

from . import config


def download(dest=config.DATA_PATH) -> "pd.DataFrame":
    """Descarga el zip de UCI, lee el .xlsx y lo guarda como CSV."""
    resp = requests.get(config.DATA_URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        with zf.open(name) as f:
            df = pd.read_excel(f)
    df = normalize_columns(df)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica mayúsculas/minúsculas con los nombres canónicos de config."""
    canon = {c.lower(): c for c in config.FEATURES + [config.TARGET]}
    return df.rename(columns={c: canon.get(str(c).lower().strip(), c) for c in df.columns})


def load(path=config.DATA_PATH) -> pd.DataFrame:
    """Carga el CSV local; si no existe lo descarga."""
    if not path.exists():
        download(path)
    return normalize_columns(pd.read_csv(path))


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Valida columnas, nulos, tipos, rangos y clases. Lanza ValueError."""
    missing = [c for c in config.FEATURES + [config.TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    if df[config.FEATURES + [config.TARGET]].isnull().any().any():
        raise ValueError("El dataset contiene valores nulos")
    for col in config.FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"La columna {col} no es numérica")
        lo, hi = config.RANGES[col]
        bad = ((df[col] < lo) | (df[col] > hi)).sum()
        if bad:
            raise ValueError(f"{col}: {bad} valores fuera del rango [{lo}, {hi}]")
    unknown = set(df[config.TARGET].str.upper()) - set(config.CLASSES)
    if unknown:
        raise ValueError(f"Clases desconocidas: {sorted(unknown)}")
    return df


def main() -> None:
    df = validate(download())
    print(f"OK: {len(df)} registros, {df[config.TARGET].nunique()} clases -> {config.DATA_PATH}")
    print(df[config.TARGET].value_counts().to_string())


if __name__ == "__main__":
    main()
