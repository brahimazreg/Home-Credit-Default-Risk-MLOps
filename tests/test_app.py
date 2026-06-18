from fastapi.testclient import TestClient
from src.api import app
from tests.test_predict import get_sample_input

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict():
    df = get_sample_input()

    payload = df.to_dict(orient="records")[0]

    response = client.post("/predict", json={"data": payload})

    assert response.status_code == 200
    assert "prediction" in response.json()