from pathlib import Path

TEST_SIZE=0.2
RANDOM_STATE=42
DATA_PATH=(Path(__file__).resolve().parent.parent / "data" / "raw" /"application_train.csv" )

MODEL_PATH = Path(__file__).resolve().parent.parent / "models"

default_model = MODEL_PATH / "logistic_regression.joblib"
