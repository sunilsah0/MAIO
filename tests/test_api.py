import json
import pytest
from app.api import app

@pytest.fixture
def client():
    return app.test_client()

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    j = resp.get_json()
    assert "status" in j and j["status"] == "ok"
    assert "model_version" in j

def test_predict_success(client, monkeypatch):
    # monkeypatch model.predict to a deterministic value
    from app.api import model
    monkeypatch.setattr(model, "predict", lambda features: 99.9)
    payload = {k: 0.0 for k in ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    j = resp.get_json()
    assert "prediction" in j
    assert j["prediction"] == 99.9

def test_predict_error(client):
    resp = client.post("/predict", json={"age": 0.0})
    assert resp.status_code == 400
    j = resp.get_json()
    assert "error" in j
