from src.predict import predict
from tests.sample_input import get_sample_input

def test_schema():
    sample = get_sample_input()
    output = predict(sample)
    assert output is not None
