"""Loads the active classifier.

Phase 5: on startup, tries MLflow's model registry first (the "champion"
alias — see ml/retrain.py, which only promotes a challenger there if it
beats the current champion on a fixed validation set). A background thread
polls for a new champion version and hot-swaps it into serving without a
restart. Falls back to the local joblib files Phase 0 (ml/train_baseline.py)
produced if the registry has no champion yet (e.g. before the first retrain
has ever run) or MLflow is unreachable — this keeps the service usable even
before Phase 4's pipeline exists, which is why that fallback is Phase 1's
whole design.
"""

import os
import threading
import time

import joblib
import mlflow
import pandas as pd
from prometheus_client import Gauge

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_REGISTRY_NAME = os.environ.get("MODEL_REGISTRY_NAME", "ecoecho-classifier")
CHAMPION_ALIAS = "champion"
MODEL_POLL_INTERVAL_SECONDS = float(os.environ.get("MODEL_POLL_INTERVAL_SECONDS", "30"))

active_model_version_gauge = Gauge(
    "inference_active_model_version",
    "MLflow registry version of the model currently serving (0 if running the local fallback)",
)
active_model_is_registry_gauge = Gauge(
    "inference_active_model_is_registry",
    "Whether the active model came from the MLflow registry (1) vs. the local fallback (0)",
)


class Classifier:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.version = "0"
        self.source = "none"
        self.model = None
        self.label_encoder = None
        self._lock = threading.Lock()

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self._client = mlflow.MlflowClient()

        if not self._try_load_from_registry():
            self._load_local_fallback()

        threading.Thread(target=self._poll_loop, daemon=True).start()

    def _load_local_fallback(self):
        model_path = os.path.join(self.model_dir, "baseline_model.joblib")
        encoder_path = os.path.join(self.model_dir, "label_encoder.joblib")
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"No model found in {self.model_dir} and no registry champion available. "
                "Run ml/train_baseline.py first and make sure ./models is mounted into the container."
            )
        model = joblib.load(model_path, mmap_mode="r")
        label_encoder = joblib.load(encoder_path)
        version = str(os.path.getmtime(model_path))

        with self._lock:
            self.model, self.label_encoder = model, label_encoder
            self.version, self.source = version, "local-fallback"
        active_model_version_gauge.set(0)
        active_model_is_registry_gauge.set(0)
        print(f"[model_loader] loaded local fallback model (mtime={version})")

    def _try_load_from_registry(self) -> bool:
        try:
            mv = self._client.get_model_version_by_alias(MODEL_REGISTRY_NAME, CHAMPION_ALIAS)
        except Exception:
            return False

        if self.source == "registry" and mv.version == self.version:
            return True  # already serving this version, nothing to do

        model = mlflow.sklearn.load_model(f"models:/{MODEL_REGISTRY_NAME}@{CHAMPION_ALIAS}")
        encoder_path = mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path="label_encoder.joblib"
        )
        label_encoder = joblib.load(encoder_path)

        with self._lock:
            self.model, self.label_encoder = model, label_encoder
            self.version, self.source = mv.version, "registry"
        active_model_version_gauge.set(float(mv.version))
        active_model_is_registry_gauge.set(1)
        print(f"[model_loader] hot-loaded registry champion (version={mv.version})")
        return True

    def _poll_loop(self):
        while True:
            time.sleep(MODEL_POLL_INTERVAL_SECONDS)
            try:
                self._try_load_from_registry()
            except Exception as e:
                print(f"[model_loader] registry poll failed: {e}")

    def predict(self, features: pd.DataFrame) -> tuple[str, float]:
        with self._lock:
            model, label_encoder = self.model, self.label_encoder
        probs = model.predict_proba(features)[0]
        best_idx = probs.argmax()
        label = label_encoder.inverse_transform([best_idx])[0]
        return label, float(probs[best_idx])
