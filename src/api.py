"""
FastAPI Backend for Hybrid NIDS
===============================
Serves predictions, SHAP explanations, and model metadata
to the Streamlit dashboard.
"""

import os
import time
import logging
import warnings

# SHAP calls into scikit-learn in a way that emits a UserWarning on every
# single prediction. It is harmless, but it floods the console during a live
# demo, so silence just that one warning. Everything else still surfaces.
warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")

from datetime import datetime, timezone
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nids-api")

# ── CICIDS2017 Feature Names (78 features) ────────────────────
CICIDS_FEATURE_NAMES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std",
    "Idle Max", "Idle Min",
]


# ── Autoencoder architecture (must match training) ──────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Global state ────────────────────────────────────────────────
class AppState:
    rf_model = None
    ae_model = None
    scaler = None
    explainer = None          # lazy-loaded on first /predict
    feature_names: list[str] = []
    classes: list[str] = []
    ae_threshold: float = 0.0
    input_dim: int = 0
    scrutiny: str = "medium"  # low / medium / high / custom
    custom_threshold: float | None = None
    start_time: float = 0.0
    alert_log: list[dict] = []

state = AppState()

# ── Scrutiny multipliers ────────────────────────────────────────
SCRUTINY_MULTIPLIERS = {
    "low": 1.5,       # fewer alerts, higher threshold
    "medium": 1.0,    # default (95th percentile)
    "high": 0.6,      # more alerts, lower threshold
}

def effective_threshold() -> float:
    if state.scrutiny == "custom" and state.custom_threshold is not None:
        return state.custom_threshold
    return state.ae_threshold * SCRUTINY_MULTIPLIERS.get(state.scrutiny, 1.0)


# ── Model loading ───────────────────────────────────────────────
def _find(name: str, dirs: list[str]) -> str | None:
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None

def load_models():
    search = ["models", "model"]  # teammate put models in model/

    rf_path = _find("random_forest.pkl", search)
    sc_path = _find("scaler.pkl", search)
    ae_path = _find("autoencoder.pth", search)

    if rf_path is None:
        raise RuntimeError("random_forest.pkl not found in models/ or model/")

    state.rf_model = joblib.load(rf_path)
    state.classes = list(state.rf_model.classes_)
    state.input_dim = state.rf_model.n_features_in_

    # Use real CICIDS2017 feature names if feature count matches
    if state.input_dim == len(CICIDS_FEATURE_NAMES):
        state.feature_names = list(CICIDS_FEATURE_NAMES)
    else:
        state.feature_names = [f"feature_{i}" for i in range(state.input_dim)]

    logger.info("RF loaded from %s  (%d features, %d classes)", rf_path, state.input_dim, len(state.classes))

    if sc_path:
        state.scaler = joblib.load(sc_path)
        logger.info("Scaler loaded from %s", sc_path)

    if ae_path:
        ckpt = torch.load(ae_path, map_location="cpu", weights_only=False)
        ae_dim = ckpt.get("input_dim", state.input_dim)
        ae = Autoencoder(ae_dim)
        ae.load_state_dict(ckpt["model_state_dict"])
        ae.eval()
        state.ae_model = ae
        state.ae_threshold = ckpt.get("threshold_95", 0.5)
        logger.info("Autoencoder loaded from %s  (threshold=%.4f)", ae_path, state.ae_threshold)
    else:
        logger.warning("Autoencoder not found — anomaly scoring disabled")

    state.start_time = time.time()


# ── Lifespan ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

app = FastAPI(
    title="Hybrid NIDS API",
    version="1.0.0",
    description="Network Intrusion Detection System — RF + Autoencoder + SHAP",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────
class FlowInput(BaseModel):
    features: list[float] = Field(..., description="Feature vector")

class BatchInput(BaseModel):
    flows: list[list[float]] = Field(..., max_length=500)

class SHAPFeature(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str

class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    is_anomaly: bool
    anomaly_score: float
    threshold_used: float
    scrutiny_level: str
    shap_top_features: list[SHAPFeature]
    class_probabilities: dict[str, float]
    timestamp: str

class SettingsInput(BaseModel):
    scrutiny: str = Field(..., pattern="^(low|medium|high|custom)$")
    custom_threshold: Optional[float] = None

class SettingsResponse(BaseModel):
    scrutiny: str
    effective_threshold: float
    base_threshold: float
    custom_threshold: Optional[float]


# ── Helpers ─────────────────────────────────────────────────────
def _get_explainer():
    if state.explainer is None:
        from src.explainability import SHAPExplainer
        # find the model path
        for d in ["models", "model"]:
            p = os.path.join(d, "random_forest.pkl")
            if os.path.isfile(p):
                state.explainer = SHAPExplainer(p)
                break
        if state.explainer is None:
            state.explainer = SHAPExplainer("models/random_forest.pkl")
    return state.explainer


def _predict_single(features: list[float]) -> dict:
    n = state.input_dim
    if len(features) != n:
        raise HTTPException(422, f"Expected {n} features, got {len(features)}")

    X = np.array(features, dtype=np.float64).reshape(1, -1)

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise HTTPException(400, "Input contains NaN or Infinity values")

    # RF prediction
    prediction = state.rf_model.predict(X)[0]
    proba = state.rf_model.predict_proba(X)[0]
    confidence = float(proba.max())

    # Autoencoder anomaly score
    anomaly_score = 0.0
    is_anomaly = False
    thresh = effective_threshold()

    if state.ae_model is not None and state.scaler is not None:
        X_scaled = state.scaler.transform(X)
        t = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            recon = state.ae_model(t)
            anomaly_score = float(torch.mean((t - recon) ** 2).item())
        is_anomaly = anomaly_score > thresh

    # SHAP
    try:
        exp = _get_explainer()
        shap_result = exp.explain(X, state.feature_names, top_n=5)
        shap_top = shap_result["top_features"]
    except Exception as e:
        logger.warning("SHAP explanation failed: %s", e)
        shap_top = []

    ts = datetime.now(timezone.utc).isoformat()

    result = {
        "prediction": str(prediction),
        "confidence": confidence,
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "threshold_used": thresh,
        "scrutiny_level": state.scrutiny,
        "shap_top_features": shap_top,
        "class_probabilities": {str(c): float(p) for c, p in zip(state.classes, proba)},
        "timestamp": ts,
    }

    # Log alerts
    if prediction != "BENIGN" or is_anomaly:
        state.alert_log.append({
            "timestamp": ts,
            "prediction": str(prediction),
            "confidence": confidence,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
        })
        # keep last 500 alerts
        if len(state.alert_log) > 500:
            state.alert_log = state.alert_log[-500:]

    return result


# ── Endpoints ───────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResult)
async def predict(flow: FlowInput):
    return _predict_single(flow.features)


@app.post("/predict/batch")
async def predict_batch(batch: BatchInput):
    results = []
    for flow in batch.flows:
        try:
            results.append(_predict_single(flow))
        except HTTPException as e:
            results.append({"error": e.detail})
    return {"results": results, "total": len(batch.flows)}


@app.get("/model/info")
async def model_info():
    return {
        "model_type": "RandomForest + Autoencoder (Hybrid)",
        "n_features": state.input_dim,
        "feature_names": state.feature_names,
        "classes": state.classes,
        "n_classes": len(state.classes),
        "ae_base_threshold": state.ae_threshold,
        "ae_effective_threshold": effective_threshold(),
        "scrutiny_level": state.scrutiny,
        "has_autoencoder": state.ae_model is not None,
        "has_scaler": state.scaler is not None,
    }


@app.get("/health")
async def health():
    uptime = time.time() - state.start_time
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 1),
        "models_loaded": {
            "random_forest": state.rf_model is not None,
            "autoencoder": state.ae_model is not None,
            "scaler": state.scaler is not None,
        },
        "total_alerts": len(state.alert_log),
    }


@app.get("/settings", response_model=SettingsResponse)
async def get_settings():
    return {
        "scrutiny": state.scrutiny,
        "effective_threshold": effective_threshold(),
        "base_threshold": state.ae_threshold,
        "custom_threshold": state.custom_threshold,
    }


@app.post("/settings", response_model=SettingsResponse)
async def update_settings(s: SettingsInput):
    state.scrutiny = s.scrutiny
    if s.scrutiny == "custom":
        if s.custom_threshold is None:
            raise HTTPException(400, "custom_threshold required when scrutiny=custom")
        state.custom_threshold = s.custom_threshold
    return {
        "scrutiny": state.scrutiny,
        "effective_threshold": effective_threshold(),
        "base_threshold": state.ae_threshold,
        "custom_threshold": state.custom_threshold,
    }


@app.get("/alerts")
async def get_alerts(limit: int = 50):
    return {
        "alerts": state.alert_log[-limit:][::-1],
        "total": len(state.alert_log),
    }


# ── Serve dashboard ────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")

@app.get("/")
async def serve_dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
