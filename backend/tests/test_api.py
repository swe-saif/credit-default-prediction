"""API tests using Flask's test client (no running server needed). Run: python -m pytest"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app import FEATURES, SCHEMA, app  # noqa: E402

PAYLOAD = json.loads((ROOT / "backend/test_payload.json").read_text(encoding="utf-8"))


@pytest.fixture
def client():
    return app.test_client()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["features"] == len(FEATURES)


def test_predict_valid(client):
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability"] <= 1
    assert body["prediction"] == int(body["probability"] >= body["threshold"])


def test_predict_missing_feature(client):
    incomplete = dict(PAYLOAD)
    incomplete.pop(FEATURES[0])
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 400


def test_predict_invalid_json(client):
    response = client.post("/predict", data="not-json", content_type="application/json")
    assert response.status_code == 400


def test_predict_unknown_feature(client):
    response = client.post("/predict", json={**PAYLOAD, "NOT_A_FEATURE": 1})
    assert response.status_code == 400
    assert response.get_json()["extra"] == ["NOT_A_FEATURE"]


def test_optional_inputs_may_be_blank(client):
    optional = [name for name, field in SCHEMA.items() if not field["required"]]
    response = client.post("/predict", json={**PAYLOAD, **{name: None for name in optional}})
    assert response.status_code == 200, response.get_json()


def test_required_input_may_not_be_blank(client):
    required = next(name for name, field in SCHEMA.items() if field["required"])
    response = client.post("/predict", json={**PAYLOAD, required: None})
    assert response.status_code == 400


def test_out_of_range_and_unknown_category_rejected(client):
    number = next(name for name, field in SCHEMA.items() if field["kind"] == "number")
    category = next(name for name, field in SCHEMA.items() if field["kind"] == "category")
    too_big = SCHEMA[number]["max"] + 1
    assert client.post("/predict", json={**PAYLOAD, number: too_big}).status_code == 400
    assert client.post("/predict", json={**PAYLOAD, category: "Not a real option"}).status_code == 400
    assert client.post("/predict", json={**PAYLOAD, number: "12"}).status_code == 400  # text is not a number
