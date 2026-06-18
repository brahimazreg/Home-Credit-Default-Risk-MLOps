from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
from src.config import default_model, DATA_PATH
import os
import math
import json
from pathlib import Path
from datetime import datetime, timezone

app = FastAPI()

model = load(default_model)
EXPECTED_FEATURES = model.feature_names_in_

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.jsonl"


class InputData(BaseModel):
    data: dict


def align(df):
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df[EXPECTED_FEATURES]


@app.post("/predict")
def predict(input: InputData):
    try:
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

    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sample")
def sample():
    try:
        df = pd.read_csv(DATA_PATH)
        sample = df.head(1).drop(columns=["SK_ID_CURR", "TARGET"]).fillna(0).to_dict(orient="records")[0]
        return {"data": sample}
    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)