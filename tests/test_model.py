import json

import pytest

from drybean_mlops import config, predict, train


def test_metrics_card(trained):
    d, card = trained
    saved = json.loads((d / "metrics.json").read_text())
    for key in ("version", "trained_at", "params", "metrics", "seed"):
        assert key in saved
    assert saved["metrics"]["f1_macro"] >= config.MIN_F1_MACRO


def test_prediction_shape(trained, df):
    d, _ = trained
    row = df.iloc[0][config.FEATURES].to_dict()
    out = predict.predict(row, d / "model.joblib")
    assert out["variety"] in config.CLASSES
    assert 0 <= out["confidence"] <= 1
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-2


def test_quality_gate_blocks_bad_model(tmp_path, df):
    with pytest.raises(train.QualityGateError):
        train.train(df, tmp_path / "m.joblib", tmp_path / "m.json", min_f1=1.01)
    assert not (tmp_path / "m.joblib").exists()
