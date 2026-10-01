import os

import pandas as pd
import pandera.errors
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, field_validator

from drift_monitor import DriftMonitor
from feature_extraction import FEATURE_NAMES
from model_loader import Classifier
from schemas import FEATURE_SCHEMA

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))
DRIFT_ALERT_THRESHOLD = float(os.environ.get("DRIFT_ALERT_THRESHOLD", "0.5"))

app = FastAPI(title="ecoecho inference-api")
app.mount("/metrics", make_asgi_app())

classifier = Classifier()
drift_monitor = DriftMonitor()

predictions_total = Counter(
    "inference_predictions_total", "Predictions served, by predicted species", ["species"]
)
low_confidence_total = Counter(
    "inference_low_confidence_total", "Predictions below CONFIDENCE_THRESHOLD"
)
confidence_score = Histogram(
    "inference_confidence_score", "Distribution of prediction confidence scores"
)
drift_score_gauge = Gauge(
    "inference_drift_score",
    "Share of features drifted vs. the training reference, over the last DRIFT_WINDOW_SIZE requests",
)
drift_detected_gauge = Gauge(
    "inference_drift_detected", "Whether drift_score is currently over DRIFT_ALERT_THRESHOLD (0/1)"
)


class PredictRequest(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def check_length(cls, v: list[float]) -> list[float]:
        if len(v) != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {len(v)}")
        return v


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    low_confidence: bool
    model_version: str
    drift_score: float
    drift_detected: bool


@app.get("/health")
def health():
    return {"status": "ok", "model_version": str(classifier.version)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    df = pd.DataFrame([request.features], columns=FEATURE_NAMES)
    try:
        df = FEATURE_SCHEMA.validate(df)
    except pandera.errors.SchemaError as e:
        raise HTTPException(status_code=422, detail=str(e))

    label, confidence = classifier.predict(df)
    flagged = confidence < CONFIDENCE_THRESHOLD

    predictions_total.labels(species=label).inc()
    confidence_score.observe(confidence)
    if flagged:
        low_confidence_total.inc()

    fresh_score = drift_monitor.observe(request.features)
    if fresh_score is not None:
        drift_score_gauge.set(fresh_score)
        drift_detected_gauge.set(1 if fresh_score >= DRIFT_ALERT_THRESHOLD else 0)

    return PredictResponse(
        prediction=label,
        confidence=confidence,
        low_confidence=flagged,
        model_version=str(classifier.version),
        drift_score=drift_monitor.drift_score,
        drift_detected=drift_monitor.drift_score >= DRIFT_ALERT_THRESHOLD,
    )
