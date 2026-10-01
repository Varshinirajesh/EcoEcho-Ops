"""Evidently-based drift monitoring.

Keeps a rolling window of recently-seen feature vectors and, once the
window fills, compares it against the reference distribution captured by
ml/train_baseline.py (models/reference_features.csv) using Evidently's
DriftedColumnsCount metric. A single-request comparison isn't statistically
meaningful, so this batches — the window is the unit of comparison, not
each /predict call.
"""

import os
import threading
from collections import deque

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from feature_extraction import FEATURE_NAMES

DRIFT_WINDOW_SIZE = int(os.environ.get("DRIFT_WINDOW_SIZE", "30"))
REFERENCE_PATH = os.environ.get(
    "DRIFT_REFERENCE_PATH", "/app/models/reference_features.csv"
)


class DriftMonitor:
    def __init__(self, reference_path: str = REFERENCE_PATH, window_size: int = DRIFT_WINDOW_SIZE):
        self.window_size = window_size
        self._window: deque[list[float]] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self.drift_score = 0.0
        self.last_checked_count = 0

        self._data_definition = DataDefinition()
        if os.path.exists(reference_path):
            self._reference = pd.read_csv(reference_path)[FEATURE_NAMES]
        else:
            self._reference = None
            print(f"[drift_monitor] no reference dataset at {reference_path}, drift checks disabled")

    def observe(self, features: list[float]) -> float | None:
        """Record one prediction's features; returns a fresh drift score when the window fills."""
        if self._reference is None:
            return None

        with self._lock:
            self._window.append(features)
            if len(self._window) < self.window_size:
                return None
            current = pd.DataFrame(list(self._window), columns=FEATURE_NAMES)

        score = self._compute_drift(current)
        self.drift_score = score
        return score

    def _compute_drift(self, current: pd.DataFrame) -> float:
        current_ds = Dataset.from_pandas(current, data_definition=self._data_definition)
        reference_ds = Dataset.from_pandas(self._reference, data_definition=self._data_definition)
        report = Report([DataDriftPreset()])
        result = report.run(current_ds, reference_ds).dict()

        for metric in result.get("metrics", []):
            if metric.get("config", {}).get("type") == "evidently:metric_v2:DriftedColumnsCount":
                return float(metric["value"]["share"])
        return 0.0
