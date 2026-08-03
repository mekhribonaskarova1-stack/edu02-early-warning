"""
FastAPI inference service for the day-60 early-warning model.
Run (from the project root): uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.inference_schema import RiskPrediction, StudentSnapshot

_model_holder: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.MODEL_PATH.exists():
        raise RuntimeError(
            f"No trained model found at {config.MODEL_PATH}. Run 'python -m src.train' first."
        )
    _model_holder["model"] = joblib.load(config.MODEL_PATH)
    yield
    _model_holder.clear()


app = FastAPI(
    title="EDU-02 Student Early-Warning API",
    description="Predicts day-60 academic risk (Fail/Withdrawn vs Pass/Distinction).",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _risk_band(probability: float) -> str:
    for low, high, label in config.RISK_BANDS:
        if low <= probability < high:
            return label
    return "High"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _model_holder}


@app.post("/predict", response_model=RiskPrediction)
def predict(snapshot: StudentSnapshot):
    model = _model_holder.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([snapshot.model_dump()])
    row = row[config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES]

    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)

    return RiskPrediction(
        at_risk_probability=round(probability, 4),
        prediction=prediction,
        risk_band=_risk_band(probability),
    )