from src.predict import predict
import pandas as pd

def test_schema():
    sample = pd.DataFrame([{
        "feature1": 0,
        "feature2": 1
    }])

    output = predict(sample)
    assert output is not None