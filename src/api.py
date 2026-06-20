from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timezone
import os

from src.config import default_model

app = FastAPI()

# load model ONLY (no dataset dependency)
model = load(default_model)

# load feature schema ONLY
EXPECTED_FEATURES = load("models/features.pkl")


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.jsonl"


class InputData(BaseModel):
    data: dict


def align(df):
    return df.reindex(columns=EXPECTED_FEATURES, fill_value=0)


@app.post("/predict")
def predict(input: InputData):

    df = pd.DataFrame([input.data])
    df = df.replace([float("inf"), float("-inf")], 0).fillna(0)
    df = align(df)

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input.data,
        "prediction": int(pred),
        "probability": float(proba)
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "prediction": int(pred),
        "probability": float(proba)
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sample")
def sample():
    return {
        "data": {
            "AMT_INCOME_TOTAL": 202500,
            "AMT_CREDIT": 406597.5,
            "CNT_CHILDREN": 0,
            "CODE_GENDER": "M",
            "FLAG_OWN_CAR": "N",
            "FLAG_OWN_REALTY": "Y",
            "NAME_CONTRACT_TYPE": "Cash loans",
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Secondary / secondary special",
            "NAME_FAMILY_STATUS": "Single / not married",
            "NAME_HOUSING_TYPE": "House / apartment",
            "DAYS_BIRTH": -9461,
            "DAYS_EMPLOYED": -637
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)