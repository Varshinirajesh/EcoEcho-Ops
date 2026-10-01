"""Local caching + batched satellite-uplink stand-in for flagged clips.

Phase 3 scope: a real buoy has limited bandwidth, so clips flagged by
inference-api (low confidence or drift) are cached to local disk first and
shipped to the cloud landing zone (MinIO here) in batches rather than one
at a time. Batches flush on whichever comes first: UPLINK_BATCH_SIZE
pending clips, or UPLINK_INTERVAL_SECONDS of wall clock time.

Each cached clip keeps its true species label alongside it (see CLAUDE.md's
"where retraining labels come from" — this is the synthetic-storm-noise
simplification, not a solved general labeling problem) so Phase 4's
retrain job has ground truth without a human labeling step.
"""

import json
import os
import shutil
import threading
import time
import uuid

from minio import Minio
from minio.error import S3Error
from prometheus_client import Counter, Gauge

CACHE_DIR = os.environ.get("UPLINK_CACHE_DIR", "/app/cache/uplink")
PENDING_DIR = os.path.join(CACHE_DIR, "pending")
UPLOADED_DIR = os.path.join(CACHE_DIR, "uploaded")

UPLINK_BATCH_SIZE = int(os.environ.get("UPLINK_BATCH_SIZE", "20"))
UPLINK_INTERVAL_SECONDS = float(os.environ.get("UPLINK_INTERVAL_SECONDS", "300"))
UPLINK_CHECK_INTERVAL_SECONDS = float(os.environ.get("UPLINK_CHECK_INTERVAL_SECONDS", "10"))

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio:9000")
MINIO_ROOT_USER = os.environ.get("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD = os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "ecoecho-uplink")

clips_cached_total = Counter(
    "buoy_clips_cached_total", "Clips written to the local uplink cache (low-confidence or drift)"
)
uplink_batches_total = Counter(
    "buoy_uplink_batches_total", "Batches successfully uploaded to MinIO"
)
uplink_clips_uploaded_total = Counter(
    "buoy_uplink_clips_uploaded_total", "Individual clips successfully uploaded to MinIO"
)
uplink_errors_total = Counter("buoy_uplink_errors_total", "Failed MinIO batch upload attempts")
cache_pending_gauge = Gauge(
    "buoy_cache_pending_clips", "Clips currently sitting in the local cache, not yet uploaded"
)


def _ensure_dirs():
    os.makedirs(PENDING_DIR, exist_ok=True)
    os.makedirs(UPLOADED_DIR, exist_ok=True)


def cache_clip(clip_path: str, metadata: dict) -> None:
    """Copy a flagged clip + its metadata into the local pending cache."""
    _ensure_dirs()
    stem = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    dest_audio = os.path.join(PENDING_DIR, f"{stem}.wav")
    dest_meta = os.path.join(PENDING_DIR, f"{stem}.json")

    shutil.copyfile(clip_path, dest_audio)
    with open(dest_meta, "w") as f:
        json.dump(metadata, f)

    clips_cached_total.inc()
    cache_pending_gauge.set(_pending_count())


def _pending_count() -> int:
    if not os.path.isdir(PENDING_DIR):
        return 0
    return len([f for f in os.listdir(PENDING_DIR) if f.endswith(".wav")])


def _make_client() -> Minio:
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ROOT_USER, secret_key=MINIO_ROOT_PASSWORD, secure=False)


def _flush_batch(client: Minio) -> None:
    _ensure_dirs()
    stems = sorted(f[:-4] for f in os.listdir(PENDING_DIR) if f.endswith(".wav"))
    if not stems:
        return

    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)

    batch_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    uploaded = []
    for stem in stems:
        audio_path = os.path.join(PENDING_DIR, f"{stem}.wav")
        meta_path = os.path.join(PENDING_DIR, f"{stem}.json")
        if not os.path.exists(meta_path):
            continue
        client.fput_object(MINIO_BUCKET, f"batches/{batch_id}/{stem}.wav", audio_path)
        client.fput_object(MINIO_BUCKET, f"batches/{batch_id}/{stem}.json", meta_path)
        uploaded.append(stem)

    for stem in uploaded:
        for ext in (".wav", ".json"):
            src = os.path.join(PENDING_DIR, f"{stem}{ext}")
            if os.path.exists(src):
                shutil.move(src, os.path.join(UPLOADED_DIR, f"{stem}{ext}"))

    uplink_batches_total.inc()
    uplink_clips_uploaded_total.inc(len(uploaded))
    cache_pending_gauge.set(_pending_count())
    print(f"[uplink] batch {batch_id}: uploaded {len(uploaded)} clip(s) to s3://{MINIO_BUCKET}/batches/{batch_id}/")


def uplink_loop():
    """Background thread: flush whenever the batch fills or the interval elapses."""
    _ensure_dirs()
    client = _make_client()
    last_flush = time.monotonic()

    while True:
        time.sleep(UPLINK_CHECK_INTERVAL_SECONDS)
        pending = _pending_count()
        cache_pending_gauge.set(pending)
        if pending == 0:
            continue

        due_by_size = pending >= UPLINK_BATCH_SIZE
        due_by_time = (time.monotonic() - last_flush) >= UPLINK_INTERVAL_SECONDS
        if not (due_by_size or due_by_time):
            continue

        try:
            _flush_batch(client)
        except S3Error as e:
            uplink_errors_total.inc()
            print(f"[uplink] batch upload failed: {e}")
        finally:
            last_flush = time.monotonic()


def start_uplink_thread():
    threading.Thread(target=uplink_loop, daemon=True).start()
