from fastapi.testclient import TestClient
from src.api import app
import pandas as pd
from src.config import DATA_PATH
import numpy as np

from tests.test_predict import get_sample_input


client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict():
    df = get_sample_input()
    
    """ sample = (
        df.head(1)
        .drop(columns=["SK_ID_CURR", "TARGET"])
        .replace([np.nan], 0)
        .to_dict(orient="records")[0]
    ) """

    response = client.post("/predict", json={"data": df})
    #response = client.post("/predict", json={"data": sample})
    assert response.status_code == 200
    assert "prediction" in response.json()