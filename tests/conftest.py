import numpy as np
import pandas as pd
import pytest

from drybean_mlops import config


def make_df(n_per_class=120, seed=0) -> pd.DataFrame:
    """Datos sintéticos válidos con 7 clases separables (solo para pruebas)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, cls in enumerate(config.CLASSES):
        center = {n: lo + (hi - lo) * (0.2 + 0.1 * k) for n, (lo, hi) in config.RANGES.items()}
        block = pd.DataFrame({
            n: np.clip(rng.normal(c, (config.RANGES[n][1] - config.RANGES[n][0]) * 0.01, n_per_class),
                       *config.RANGES[n]) for n, c in center.items()})
        block[config.TARGET] = cls
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def df():
    return make_df()


@pytest.fixture(scope="session")
def trained(tmp_path_factory, df):
    from drybean_mlops import train
    d = tmp_path_factory.mktemp("models")
    card = train.train(df, d / "model.joblib", d / "metrics.json")
    return d, card
