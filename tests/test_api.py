import pytest
from fastapi.testclient import TestClient

from drybean_mlops import api, config, predict


@pytest.fixture()
def client(trained, monkeypatch):
    d, _ = trained
    monkeypatch.setattr(config, "MODEL_PATH", d / "model.joblib")
    monkeypatch.setattr(config, "METRICS_PATH", d / "metrics.json")
    real_predict, real_info = predict.predict, predict.model_info
    monkeypatch.setattr(predict, "predict", lambda f, p=d / "model.joblib": real_predict(f, p))
    monkeypatch.setattr(predict, "model_info", lambda p=d / "metrics.json": real_info(p))
    return TestClient(api.app)


def valid(df):
    return df.iloc[0][config.FEATURES].to_dict()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    assert "metrics" in client.get("/model-info").json()


def test_predict_ok(client, df):
    r = client.post("/predict", json=valid(df))
    assert r.status_code == 200
    assert r.json()["variety"] in config.CLASSES


def test_predict_invalid_422(client, df):
    bad = valid(df); bad["Extent"] = 7
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_missing_field_422(client, df):
    bad = valid(df); bad.pop("Area")
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_model_503(tmp_path, monkeypatch, df):
    monkeypatch.setattr(predict, "predict",
                        lambda f, p=tmp_path / "x.joblib": predict.load_model.__wrapped__(p))
    r = TestClient(api.app).post("/predict", json=valid(df))
    assert r.status_code == 503
