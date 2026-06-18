from joblib import load
import pandas as pd
from src.config import default_model

model = load(default_model)

def predict(df: pd.DataFrame):
    return model.predict(df)

if __name__ == "__main__":
    sample = pd.read_csv("data/raw/application_train.csv").drop(columns=["SK_ID_CURR", "TARGET"]).head(5)
    print(predict(sample))
    