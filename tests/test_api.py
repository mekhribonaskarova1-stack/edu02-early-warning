"""
Run directly (from the project root, after training a model):
    python -m pytest -v
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app

SAMPLE_STUDENT = {
    "code_module": "BBB",
    "code_presentation": "2013J",
    "gender": "F",
    "region": "East Anglian Region",
    "highest_education": "A Level or Equivalent",
    "imd_band": "30-40%",
    "age_band": "0-35",
    "disability": "N",
    "num_of_prev_attempts": 0,
    "studied_credits": 60,
    "date_registration": -45,
    "withdrawn_before_cutoff": 0,
    "sum_click_total": 820,
    "n_active_days": 27,
    "n_early_assessments": 2,
    "mean_early_score": 68.5,
    "min_early_score": 55,
}


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager triggers FastAPI's lifespan
    # (startup/shutdown) events, which is what actually loads the model.
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_expected_shape(client):
    response = client.post("/predict", json=SAMPLE_STUDENT)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["at_risk_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["risk_band"] in ("Low", "Medium", "High")


def test_predict_rejects_missing_field(client):
    bad_payload = dict(SAMPLE_STUDENT)
    del bad_payload["gender"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
