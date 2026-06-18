from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
from src.config import default_model
import math
from src.config import DATA_PATH


app = FastAPI()

model = load(default_model)

class InputData(BaseModel):
    data: dict


@app.post("/predict")
def predict(input: InputData):
    df = pd.DataFrame([input.data])

    # safety cleaning
    df = df.replace([float("inf"), float("-inf")], 0)
    df = df.fillna(0)

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(0 if math.isnan(proba) else proba)
    }

@app.get("/health")
def health():
    return {"status":"Api is running"}

@app.get("/sample")
def sample():
    df = pd.read_csv(DATA_PATH)

    sample = (
        df.head(1)
          .drop(columns=["SK_ID_CURR", "TARGET"])
          .fillna(0)
          .to_dict(orient="records")[0]
    )

    return {"data": sample}


