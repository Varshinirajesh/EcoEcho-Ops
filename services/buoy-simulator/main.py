"""Simulates a hydrophone buoy streaming recorded clips like a live feed.

Phase 1: play clips back on an interval, extract features, call
inference-api, expose metrics. Phase 2 added storm mode: synthetic ocean
storm noise mixed into clips before feature extraction, toggled either at
startup (STORM_MODE_ENABLED) or live via the control endpoints below.
Phase 3 adds caching flagged clips locally and shipping them to MinIO in
batches (see uplink.py).
"""

import itertools
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from feature_extraction import extract_features_from_signal, load_audio
from spectrogram_image import render_spectrogram_png
from storm_noise import inject_storm_noise
from uplink import cache_clip, start_uplink_thread

DATA_DIR = os.environ.get("DATA_DIR", "/app/data/audio_stream")
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://inference-api:8000/predict")
STREAM_INTERVAL_SECONDS = float(os.environ.get("STREAM_INTERVAL_SECONDS", "2"))
METRICS_PORT = int(os.environ.get("METRICS_PORT", "8001"))
CONTROL_PORT = int(os.environ.get("CONTROL_PORT", "8002"))
STORM_MODE_ENABLED = os.environ.get("STORM_MODE_ENABLED", "false").lower() == "true"
STORM_NOISE_INTENSITY = float(os.environ.get("STORM_NOISE_INTENSITY", "0.6"))
BUOY_ID = os.environ.get("BUOY_ID", "BUOY-07")
# Fixed simulated deployment position — this project has no real hardware,
# so there's no GPS to read; kept as an explicit constant (not fabricated
# per-request) so it's honestly a fixed simulated value, not moving noise.
BUOY_LOCATION = {"lat": 42.3601, "lon": -70.9, "depth_m": 45}  # Stellwagen Bank, Gulf of Maine

clips_streamed_total = Counter("buoy_clips_streamed_total", "Clips played back to inference-api")
inference_errors_total = Counter("buoy_inference_errors_total", "Failed calls to inference-api")
inference_request_seconds = Histogram(
    "buoy_inference_request_seconds", "Round-trip latency of inference-api calls"
)
prediction_match_total = Counter(
    "buoy_prediction_match_total",
    "Predicted species vs. the clip's known true species (sanity check only — "
    "a real buoy would not have this ground truth)",
    ["result"],
)
storm_mode_gauge = Gauge(
    "buoy_storm_mode_enabled", "Whether synthetic storm noise injection is currently on (0/1)"
)

storm_mode = threading.Event()
if STORM_MODE_ENABLED:
    storm_mode.set()
storm_mode_gauge.set(1 if storm_mode.is_set() else 0)

# Latest clip's audio (for on-demand spectrogram rendering) and latest
# prediction result, shared with the dashboard-api's polling — real state
# from the live stream, not synthesized for display.
_latest_lock = threading.Lock()
_latest_audio: dict = {"y": None, "sr": None}
_latest_prediction: dict = {}


class StormControlHandler(BaseHTTPRequestHandler):
    def _reply(self, status: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _status_body(self):
        return {"storm_mode_enabled": storm_mode.is_set(), "intensity": STORM_NOISE_INTENSITY}

    def _reply_bytes(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/storm":
            self._reply(200, self._status_body())
        elif self.path == "/buoy-info":
            self._reply(200, {"id": BUOY_ID, "status": "online", **BUOY_LOCATION})
        elif self.path == "/latest-prediction":
            with _latest_lock:
                prediction = dict(_latest_prediction)
            if not prediction:
                self._reply(404, {"error": "no predictions yet"})
            else:
                self._reply(200, prediction)
        elif self.path == "/spectrogram.png":
            with _latest_lock:
                y, sr = _latest_audio["y"], _latest_audio["sr"]
            if y is None:
                self._reply(404, {"error": "no clip streamed yet"})
            else:
                self._reply_bytes(200, "image/png", render_spectrogram_png(y, sr))
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/storm/on":
            storm_mode.set()
        elif self.path == "/storm/off":
            storm_mode.clear()
        else:
            self._reply(404, {"error": "not found"})
            return
        storm_mode_gauge.set(1 if storm_mode.is_set() else 0)
        self._reply(200, self._status_body())

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass  # keep container logs to the stream_clip prints


def iter_clips(data_dir: str):
    """Cycle through <species>/*.wav clips forever, pairing each with its true label."""
    species_dirs = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    if not species_dirs:
        raise FileNotFoundError(
            f"No species subfolders found in {data_dir}. "
            "Expected data/audio_stream/<species>/*.wav"
        )

    clips = [
        (os.path.join(data_dir, species, clip), species)
        for species in species_dirs
        for clip in sorted(os.listdir(os.path.join(data_dir, species)))
        if clip.lower().endswith(".wav")
    ]
    if not clips:
        raise FileNotFoundError(f"No .wav clips found under {data_dir}")

    # Shuffled once so any window of consecutive clips is a reasonable species
    # mix — built species-by-species above, a window of a few dozen clips
    # would otherwise be a near-single-species block, which reads as drift
    # against the reference (which spans all species) even with storm mode off.
    random.shuffle(clips)
    return itertools.cycle(clips)


def stream_clip(path: str, true_species: str):
    y, sr = load_audio(path)
    stormy = storm_mode.is_set()
    if stormy:
        y = inject_storm_noise(y, sr, intensity=STORM_NOISE_INTENSITY)
    features = extract_features_from_signal(y, sr)

    start = time.monotonic()
    try:
        response = requests.post(
            INFERENCE_API_URL, json={"features": features.tolist()}, timeout=10
        )
        response.raise_for_status()
    except requests.RequestException as e:
        inference_errors_total.inc()
        print(f"[buoy-simulator] inference-api call failed for {path}: {e}")
        return
    finally:
        inference_request_seconds.observe(time.monotonic() - start)

    result = response.json()
    clips_streamed_total.inc()
    match = "correct" if result["prediction"] == true_species else "incorrect"
    prediction_match_total.labels(result=match).inc()
    flagged = result.get("low_confidence") or result.get("drift_detected")

    with _latest_lock:
        _latest_audio["y"], _latest_audio["sr"] = y, sr
        _latest_prediction.clear()
        _latest_prediction.update(
            {
                "clip": os.path.basename(path),
                "true_species": true_species,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "match": match,
                "storm_mode": stormy,
                "low_confidence": result.get("low_confidence"),
                "drift_score": result.get("drift_score"),
                "drift_detected": result.get("drift_detected"),
                "timestamp": time.time(),
            }
        )

    print(
        f"[buoy-simulator] {os.path.basename(path)} true={true_species} "
        f"predicted={result['prediction']} confidence={result['confidence']:.3f} "
        f"({match}){' [STORM]' if stormy else ''}{' [CACHED]' if flagged else ''}"
    )

    if flagged:
        cache_clip(
            path,
            {
                "true_species": true_species,
                "predicted_species": result["prediction"],
                "confidence": result["confidence"],
                "low_confidence": result["low_confidence"],
                "drift_score": result.get("drift_score"),
                "drift_detected": result.get("drift_detected"),
                "storm_mode": stormy,
                "captured_at": time.time(),
            },
        )


def main():
    threading.Thread(target=start_http_server, args=(METRICS_PORT,), daemon=True).start()
    control_server = ThreadingHTTPServer(("0.0.0.0", CONTROL_PORT), StormControlHandler)
    threading.Thread(target=control_server.serve_forever, daemon=True).start()
    start_uplink_thread()
    print(
        f"[buoy-simulator] metrics on :{METRICS_PORT}, storm control on :{CONTROL_PORT}, "
        f"streaming from {DATA_DIR} (storm_mode={'on' if storm_mode.is_set() else 'off'})"
    )

    for path, species in iter_clips(DATA_DIR):
        stream_clip(path, species)
        time.sleep(STREAM_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
