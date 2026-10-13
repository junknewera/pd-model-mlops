import joblib
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.models.pipeline import build_pipeline


@pytest.fixture
def api(clients, tmp_path, monkeypatch):
    model = build_pipeline("logreg").fit(clients, (clients["PAY_1"] > 1).astype(int))
    joblib.dump(model, tmp_path / "model.joblib")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def payload(clients):
    return clients.head(1).to_dict(orient="records")[0]


def test_health(api):
    assert api.get("/health").json() == {"status": "ok"}


def test_predict(api, payload):
    resp = api.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["default"] in (0, 1)
    assert 0 <= body["probability"] <= 1
    assert body["default"] == int(body["probability"] >= 0.5)


def test_predict_rejects_bad_value(api, payload):
    payload["AGE"] = 200
    assert api.post("/predict", json=payload).status_code == 422


def test_predict_rejects_missing_field(api, payload):
    del payload["LIMIT_BAL"]
    assert api.post("/predict", json=payload).status_code == 422
