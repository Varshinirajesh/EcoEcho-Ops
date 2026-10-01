"""Aggregates Prometheus + MLflow + buoy-simulator state into one feed for
the custom Overview page (services/dashboard-frontend). Polls every
POLL_INTERVAL_SECONDS in a background thread and derives a rolling
log/alert feed from real state transitions it observes — nothing here is
synthesized display data, it's the same numbers Grafana shows, reshaped
onto a single page.
"""

"""Aggregates Prometheus + MLflow + buoy-simulator state into one feed for
the custom Overview page (services/dashboard-frontend).

Polls every POLL_INTERVAL_SECONDS in a background thread and derives a
rolling log/alert feed from real state transitions it observes.

The API never exposes NaN or Infinity values because strict JSON encoding
rejects those values. Non-finite metrics are represented as None/null.
"""

import os
import threading
import time
import math
from collections import deque
from datetime import datetime, timezone

import json as jsonlib

import mlflow
import requests as http
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from minio import Minio


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROMETHEUS_URL = os.environ.get(
    "PROMETHEUS_URL",
    "http://prometheus:9090",
)

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",
)

BUOY_CONTROL_URL = os.environ.get(
    "BUOY_CONTROL_URL",
    "http://buoy-simulator:8002",
)

MODEL_REGISTRY_NAME = os.environ.get(
    "MODEL_REGISTRY_NAME",
    "ecoecho-classifier",
)

POLL_INTERVAL_SECONDS = float(
    os.environ.get("POLL_INTERVAL_SECONDS", "5")
)

DRIFT_ALERT_THRESHOLD = float(
    os.environ.get("DRIFT_ALERT_THRESHOLD", "0.5")
)

MINIO_ENDPOINT = os.environ.get(
    "MINIO_ENDPOINT",
    "minio:9000",
)

MINIO_ROOT_USER = os.environ.get(
    "MINIO_ROOT_USER",
    "minioadmin",
)

MINIO_ROOT_PASSWORD = os.environ.get(
    "MINIO_ROOT_PASSWORD",
    "minioadmin",
)

MINIO_BUCKET = os.environ.get(
    "MINIO_BUCKET",
    "ecoecho-uplink",
)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow_client = mlflow.MlflowClient()

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ROOT_USER,
    secret_key=MINIO_ROOT_PASSWORD,
    secure=False,
)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="ecoecho dashboard-api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()

_state: dict = {}

_events: deque = deque(maxlen=50)

_prev: dict = {}


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------

def _json_safe(value):
    """Recursively convert non-JSON-safe numeric values to None.

    JSON does not officially support NaN or Infinity. FastAPI/Starlette
    therefore raises an error when these values reach JSON serialization.

    This function preserves normal values and converts:

        NaN
        +Infinity
        -Infinity

    into Python None, which becomes JSON null.
    """

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {
            key: _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _json_safe(item)
            for item in value
        ]

    return value


# ---------------------------------------------------------------------------
# Prometheus helpers
# ---------------------------------------------------------------------------

def _prom_instant(expr: str):
    """Execute an instant Prometheus query.

    Returns:
        float  - finite metric value
        None   - no result, query failure, NaN, or Infinity
    """

    try:
        r = http.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": expr},
            timeout=5,
        )

        r.raise_for_status()

        payload = r.json()

        result = payload["data"]["result"]

        if not result:
            return None

        value = float(result[0]["value"][1])

        if not math.isfinite(value):
            return None

        return value

    except Exception:
        return None


def _prom_range(
    expr: str,
    minutes: int = 60,
    step: str = "60s",
):
    """Execute a Prometheus range query.

    Non-finite samples are discarded so they cannot break JSON
    serialization in the dashboard API.
    """

    try:
        now = time.time()

        r = http.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={
                "query": expr,
                "start": now - minutes * 60,
                "end": now,
                "step": step,
            },
            timeout=5,
        )

        r.raise_for_status()

        payload = r.json()

        result = payload["data"]["result"]

        if not result:
            return []

        values = []

        for timestamp, raw_value in result[0]["values"]:
            try:
                timestamp = float(timestamp)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue

            if not math.isfinite(timestamp):
                continue

            if not math.isfinite(value):
                continue

            values.append([
                timestamp,
                value,
            ])

        return values

    except Exception:
        return []


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------

def _log(level: str, message: str):
    """Add an event to the dashboard event feed."""

    _events.appendleft(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
        }
    )


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def _get_champion_and_previous():
    """Return current Champion and previous model information."""

    try:
        champion_mv = mlflow_client.get_model_version_by_alias(
            MODEL_REGISTRY_NAME,
            "champion",
        )
    except Exception:
        return None, None

    try:
        champion_run = mlflow_client.get_run(
            champion_mv.run_id
        )

        champion = {
            "version": champion_mv.version,
            "metrics": _json_safe(
                champion_run.data.metrics
            ),
            "params": champion_run.data.params,
            "created_at": champion_mv.creation_timestamp / 1000,
            "promoted": champion_run.data.tags.get("promoted"),
        }
    except Exception:
        return None, None

    previous = None

    try:
        prev_version = int(champion_mv.version) - 1

        if prev_version >= 1:
            prev_mv = mlflow_client.get_model_version(
                MODEL_REGISTRY_NAME,
                str(prev_version),
            )

            prev_run = mlflow_client.get_run(
                prev_mv.run_id
            )

            previous = {
                "version": prev_mv.version,
                "metrics": _json_safe(
                    prev_run.data.metrics
                ),
            }

    except Exception:
        pass

    return champion, previous


def _get_last_retrain_run():
    """Return the latest retraining run from MLflow."""

    try:
        experiment = mlflow_client.get_experiment_by_name(
            "ecoecho-retrain"
        )

        if experiment is None:
            return None

        runs = mlflow_client.search_runs(
            [experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            return None

        run = runs[0]

        return {
            "run_name": run.info.run_name,
            "start_time": run.info.start_time / 1000,
            "metrics": _json_safe(
                run.data.metrics
            ),
            "params": run.data.params,
            "promoted": run.data.tags.get("promoted"),
        }

    except Exception:
        return None


# ---------------------------------------------------------------------------
# MinIO / flagged clips
# ---------------------------------------------------------------------------

def _get_recent_flagged_clips(limit: int = 6):
    """Return recent low-confidence/drift clips from MinIO.

    Only JSON metadata objects are read. Audio files are not downloaded.
    """

    try:
        clips = []

        for prefix, action in (
            ("batches/", "Pending"),
            ("batches-processed/", "Retrained on"),
        ):
            for obj in minio_client.list_objects(
                MINIO_BUCKET,
                prefix=prefix,
                recursive=True,
            ):
                if obj.object_name.endswith(".json"):
                    clips.append(
                        (
                            obj.object_name,
                            obj.last_modified,
                            action,
                        )
                    )

        clips.sort(
            key=lambda c: c[1],
            reverse=True,
        )

        result = []

        for object_name, _, action in clips[:limit]:

            resp = None

            try:
                resp = minio_client.get_object(
                    MINIO_BUCKET,
                    object_name,
                )

                meta = jsonlib.loads(
                    resp.read()
                )

                result.append(
                    {
                        "time": meta.get("captured_at"),
                        "species": meta.get("predicted_species"),
                        "true_species": meta.get("true_species"),
                        "confidence": _json_safe(
                            meta.get("confidence")
                        ),
                        "action": action,
                    }
                )

            finally:
                if resp is not None:
                    try:
                        resp.close()
                        resp.release_conn()
                    except Exception:
                        pass

        return result

    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main polling logic
# ---------------------------------------------------------------------------

def _poll_once():

    # -----------------------------------------------------------------------
    # Prometheus metrics
    # -----------------------------------------------------------------------

    accuracy = _prom_instant(
        'sum(rate(buoy_prediction_match_total{result="correct"}[5m])) / '
        'sum(rate(buoy_prediction_match_total[5m]))'
    )

    drift_score = _prom_instant(
        "inference_drift_score"
    )

    if drift_score is None:
        drift_score = 0.0

    drift_detected_value = _prom_instant(
        "inference_drift_detected"
    )

    storm_mode_value = _prom_instant(
        "buoy_storm_mode_enabled"
    )

    drift_detected = bool(
        drift_detected_value
    )

    storm_mode = bool(
        storm_mode_value
    )

    active_version = _prom_instant(
        "inference_active_model_version"
    )

    is_registry_value = _prom_instant(
        "inference_active_model_is_registry"
    )

    is_registry = bool(
        is_registry_value
    )

    # A 15-minute window is used because Prometheus increase() can
    # extrapolate when a series has not existed for the full window.
    low_conf_recent = _prom_instant(
        "increase(inference_low_confidence_total[15m])"
    )

    if low_conf_recent is None:
        low_conf_recent = 0.0

    pending_clips = _prom_instant(
        "buoy_cache_pending_clips"
    )

    if pending_clips is None:
        pending_clips = 0.0

    uplink_batches = _prom_instant(
        "buoy_uplink_batches_total"
    )

    if uplink_batches is None:
        uplink_batches = 0.0

    uplink_clips = _prom_instant(
        "buoy_uplink_clips_uploaded_total"
    )

    if uplink_clips is None:
        uplink_clips = 0.0

    # -----------------------------------------------------------------------
    # Historical metrics
    # -----------------------------------------------------------------------

    drift_history = _prom_range(
        "inference_drift_score"
    )

    accuracy_history = _prom_range(
        'sum(rate(buoy_prediction_match_total{result="correct"}[5m])) / '
        'sum(rate(buoy_prediction_match_total[5m]))'
    )

    # -----------------------------------------------------------------------
    # Buoy state
    # -----------------------------------------------------------------------

    try:
        buoy_info = http.get(
            f"{BUOY_CONTROL_URL}/buoy-info",
            timeout=3,
        ).json()
    except Exception:
        buoy_info = None

    try:
        latest_prediction = http.get(
            f"{BUOY_CONTROL_URL}/latest-prediction",
            timeout=3,
        ).json()
    except Exception:
        latest_prediction = None

    # -----------------------------------------------------------------------
    # MLflow state
    # -----------------------------------------------------------------------

    champion, previous = _get_champion_and_previous()

    last_retrain = _get_last_retrain_run()

    recent_clips = _get_recent_flagged_clips()

    # System health should reflect the actual buoy endpoints.
    system_healthy = (
        buoy_info is not None
        and latest_prediction is not None
    )

    # -----------------------------------------------------------------------
    # State transition logging
    # -----------------------------------------------------------------------

    if drift_detected and not _prev.get("drift_detected"):
        _prev["drift_detected_at"] = time.time()

    if (
        _prev.get("drift_detected") is not None
        and drift_detected != _prev.get("drift_detected")
    ):
        if drift_detected:
            _log(
                "warn",
                (
                    f"Drift detected: score={drift_score:.2f} "
                    f"(threshold={DRIFT_ALERT_THRESHOLD})"
                ),
            )
        else:
            _log(
                "info",
                "Drift cleared",
            )

    if (
        _prev.get("storm_mode") is not None
        and storm_mode != _prev.get("storm_mode")
    ):
        _log(
            "info",
            (
                "Storm mode turned "
                f"{'on' if storm_mode else 'off'}"
            ),
        )

    if (
        _prev.get("active_version") is not None
        and active_version != _prev.get("active_version")
    ):
        _log(
            "success",
            (
                f"Model hot-swapped to v{int(active_version)}"
                if active_version is not None
                else "Model reverted to local fallback"
            ),
        )

    if (
        latest_prediction
        and _prev.get("latest_ts")
        != latest_prediction.get("timestamp")
    ):
        prediction = latest_prediction

        prediction_name = prediction.get(
            "prediction",
            "unknown",
        )

        confidence = prediction.get(
            "confidence"
        )

        if isinstance(confidence, (int, float)):
            confidence_text = f"{confidence:.2f}"
        else:
            confidence_text = "n/a"

        _log(
            "warn"
            if prediction.get("low_confidence")
            else "info",
            (
                f"Prediction: {prediction_name} "
                f"({confidence_text})"
                + (
                    " [low confidence, saved]"
                    if prediction.get("low_confidence")
                    else ""
                )
            ),
        )

    if (
        last_retrain
        and _prev.get("last_retrain_start")
        != last_retrain.get("start_time")
    ):
        _log(
            "success"
            if last_retrain.get("promoted") == "true"
            else "info",
            (
                f"Retrain completed "
                f"({last_retrain['run_name']}): "
                + (
                    "promoted"
                    if last_retrain.get("promoted") == "true"
                    else "not promoted"
                )
            ),
        )

    # -----------------------------------------------------------------------
    # Previous state
    # -----------------------------------------------------------------------

    _prev.update(
        {
            "drift_detected": drift_detected,
            "storm_mode": storm_mode,
            "active_version": active_version,
            "latest_ts": (
                latest_prediction.get("timestamp")
                if latest_prediction
                else None
            ),
            "last_retrain_start": (
                last_retrain.get("start_time")
                if last_retrain
                else None
            ),
        }
    )

    # -----------------------------------------------------------------------
    # Store current state
    # -----------------------------------------------------------------------

    new_state = {
        "accuracy": accuracy,
        "drift_score": drift_score,
        "drift_detected": drift_detected,
        "drift_history": drift_history,
        "accuracy_history": accuracy_history,
        "storm_mode": storm_mode,
        "active_version": active_version,
        "is_registry": is_registry,
        "low_confidence_recent": low_conf_recent,
        "pending_clips": pending_clips,
        "uplink_batches": uplink_batches,
        "uplink_clips": uplink_clips,
        "buoy_info": buoy_info,
        "latest_prediction": latest_prediction,
        "champion": champion,
        "previous_champion": previous,
        "last_retrain": last_retrain,
        "recent_clips": recent_clips,
        "system_healthy": system_healthy,
        "drift_detected_at": _prev.get(
            "drift_detected_at"
        ),
        "events": list(_events),
        "updated_at": time.time(),
    }

    # Final defensive cleanup before putting the state into the shared
    # object. This means even values originating from MLflow, MinIO, or
    # another API cannot introduce NaN/Infinity.
    new_state = _json_safe(new_state)

    with _state_lock:
        _state.clear()
        _state.update(new_state)


# ---------------------------------------------------------------------------
# Background polling
# ---------------------------------------------------------------------------

def poll_loop():
    while True:
        try:
            _poll_once()
        except Exception as e:
            _log(
                "error",
                f"dashboard-api poll failed: {e}",
            )

        time.sleep(
            POLL_INTERVAL_SECONDS
        )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/overview")
def overview():
    """Return the current dashboard state.

    A final JSON-safety pass prevents NaN/Infinity values from causing
    HTTP 500 responses.
    """

    with _state_lock:
        return _json_safe(
            dict(_state)
        )


@app.get("/api/spectrogram.png")
def spectrogram():
    """Proxy the current buoy spectrogram."""

    try:
        r = http.get(
            f"{BUOY_CONTROL_URL}/spectrogram.png",
            timeout=5,
        )

        if r.status_code != 200:
            return Response(
                status_code=404
            )

        return Response(
            content=r.content,
            media_type="image/png",
        )

    except Exception:
        return Response(
            status_code=404
        )


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ---------------------------------------------------------------------------
# Start background polling
# ---------------------------------------------------------------------------

threading.Thread(
    target=poll_loop,
    daemon=True,
).start()