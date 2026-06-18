""" from joblib import load
import pandas as pd
from src.config import default_model

model = load(default_model)

df = pd.read_csv("data/raw/application_train.csv")
X = df.drop(columns=["SK_ID_CURR", "TARGET"])

def test_predict():
    sample = X.head(1)
    pred = model.predict(sample)

    assert len(pred) == 1
    print("OK - prediction works")

if __name__=='__main__':
    test_predict() """

from src.predict import predict
import pandas as pd

def test_schema():
    sample = pd.DataFrame([{
        "feature1": 0,
        "feature2": 1
    }])

    output = predict(sample)
    assert output is not None