
from src.predict import predict
from tests.sample_input import get_sample_input

def test_predict():
    df = get_sample_input()
    output = predict(df)
    assert output is not None
