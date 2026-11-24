import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.delenv("PREDICTION_LOG_TABLE", raising=False)
    from churn.serve.app import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_version"]


def test_metadata(client):
    body = client.get("/metadata").json()
    assert body["decision_threshold"] == 0.55
    assert "tenure" in body["features"]


def test_predict_single(client, sample_customer):
    r = client.post("/predict", json={"instances": [sample_customer]})
    assert r.status_code == 200
    pred = r.json()["predictions"][0]
    assert pred["customer_id"] == "c-1"
    assert 0 <= pred["churn_probability"] <= 1
    assert pred["risk_band"] in {"low", "medium", "high"}
    assert "X-Process-Time-Ms" in r.headers


def test_predict_batch_preserves_order(client, sample_customer):
    batch = [{**sample_customer, "customer_id": f"c-{i}"} for i in range(25)]
    r = client.post("/predict", json={"instances": batch})
    ids = [p["customer_id"] for p in r.json()["predictions"]]
    assert ids == [f"c-{i}" for i in range(25)]


def test_risk_ordering(client, sample_customer):
    loyal = {
        **sample_customer,
        "customer_id": "loyal",
        "contract": "Two year",
        "tenure": 70,
        "total_charges": 6200.0,
        "payment_method": "Credit card (automatic)",
        "internet_service": "DSL",
    }
    r = client.post("/predict", json={"instances": [sample_customer, loyal]}).json()
    scores = {p["customer_id"]: p["churn_probability"] for p in r["predictions"]}
    assert scores["c-1"] > scores["loyal"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("contract", "Monthly"),
        ("tenure", -1),
        ("monthly_charges", "abc"),
        ("internet_service", None),
    ],
)
def test_rejects_bad_input(client, sample_customer, field, value):
    r = client.post("/predict", json={"instances": [{**sample_customer, field: value}]})
    assert r.status_code == 422


def test_rejects_swapped_charges(client, sample_customer):
    bad = {**sample_customer, "tenure": 30, "monthly_charges": 90.0, "total_charges": 12.0}
    assert client.post("/predict", json={"instances": [bad]}).status_code == 422


def test_rejects_empty_batch(client):
    assert client.post("/predict", json={"instances": []}).status_code == 422


def test_prediction_rows_are_logged(client, sample_customer):
    captured = []

    class Capture:
        def write(self, rows):
            captured.extend(rows)

        def close(self):
            pass

    client.app.state.sink = Capture()
    client.post("/predict", json={"instances": [sample_customer]})
    assert len(captured) == 1
    row = captured[0]
    assert row["customer_id"] == "c-1"
    assert {"churn_probability", "model_version", "prediction_ts", "contract"} <= row.keys()
