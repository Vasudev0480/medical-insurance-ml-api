from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Medical Insurance Premium Predictor")

# Always resolve paths relative to this file (api/main.py)
BASE_DIR = Path(__file__).resolve().parent  # api/
MODEL_PATH = (BASE_DIR / "../ml/models/premium_predictor.joblib").resolve()
COLS_PATH  = (BASE_DIR / "../ml/models/feature_columns.json").resolve()

# Load artifacts once at startup
model = joblib.load(MODEL_PATH)
feature_columns = json.loads(COLS_PATH.read_text(encoding="utf-8"))

class PredictRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: str
    bmi: float = Field(..., ge=10, le=80)
    children: int = Field(..., ge=0, le=20)
    smoker: str
    region: str

def normalize_text(s: str) -> str:
    return s.strip().lower()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        sex = normalize_text(req.sex)
        smoker = normalize_text(req.smoker)
        region = normalize_text(req.region)

        if sex not in {"male", "female"}:
            raise HTTPException(status_code=400, detail="sex must be 'male' or 'female'")
        if smoker not in {"yes", "no"}:
            raise HTTPException(status_code=400, detail="smoker must be 'yes' or 'no'")
        if region not in {"southwest", "southeast", "northwest", "northeast"}:
            raise HTTPException(status_code=400, detail="region must be one of: southwest, southeast, northwest, northeast")

        raw = pd.DataFrame([{
            "age": req.age,
            "sex": sex,
            "bmi": req.bmi,
            "children": req.children,
            "smoker": smoker,
            "region": region
        }])

        encoded = pd.get_dummies(raw, drop_first=True)
        encoded = encoded.reindex(columns=feature_columns, fill_value=0)

        pred = model.predict(encoded)[0]
        return {"predicted_premium": float(pred), "currency": "CAD"}

    except HTTPException:
        raise
    except Exception as e:
        # This makes debugging easier (no silent 500s)
        raise HTTPException(status_code=500, detail=f"Server error: {type(e).__name__}: {e}")
