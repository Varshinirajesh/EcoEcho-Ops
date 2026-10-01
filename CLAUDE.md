# CLAUDE.md

This file gives Claude (via Claude Code or any Claude session working in this repo) the context needed to work on EcoEcho-Ops effectively. Read this before making changes.

## Project summary

EcoEcho-Ops is a self-healing edge MLOps pipeline for autonomous acoustic marine telemetry. Hydrophone buoys classify whale sounds at the edge. Ocean storms inject noise that causes data drift and degrades the classifier. The system detects that drift, caches the low-confidence clips locally, ships them to a cloud landing zone in a bandwidth-light batch, automatically retrains and evaluates a new model, and redeploys it to the edge with no manual intervention.

The whole thing runs on Docker Compose. No paid cloud services — "cloud" components (object storage, orchestrator, model registry) also run locally in containers, standing in for what would be real cloud services in a production deployment.

## Architecture (end to end)

```
buoy-simulator --(features)--> inference-api --(prediction+confidence)--> [caller]
      |                              |
      |                    Evidently drift check
      |                              |
      +---- low-confidence/drift clips cached locally ----+
                                     |
                        batched upload to MinIO (satellite-uplink stand-in)
                                     |
                        Airflow DAG: sensor -> retrain -> evaluate -> register
                                     |
                        MLflow tracking + model registry
                                     |
                        inference-api polls registry, hot-reloads new model
```

Prometheus scrapes metrics from `buoy-simulator` and `inference-api` the whole way through; Grafana visualizes them (drift score, confidence, active model version).

## Repo structure

```
docker-compose.yml            # core stack: buoy-simulator, inference-api, prometheus, grafana, minio, mlflow, dashboard-api, dashboard-frontend
services/
  buoy-simulator/              # simulates the edge device streaming audio
  inference-api/                # FastAPI + Pandera validation + model serving, polls MLflow registry
  mlflow/                       # MLflow tracking server
  retrain-worker/                # image the Airflow DAG launches on-demand to run ml/retrain.py
  dashboard-api/                 # aggregates Prometheus + MLflow + buoy state into one JSON feed
  dashboard-frontend/            # static custom Overview page (nginx), polls dashboard-api
airflow/dags/retrain_pipeline.py  # cloud orchestrator DAG (kept in its own docker-compose, see below)
ml/
  feature_extraction.py         # audio clip -> feature vector (MFCC, spectral centroid, RMS, ZCR)
  storm_noise.py                 # synthetic ocean storm noise injection
  spectrogram_image.py           # renders a live spectrogram PNG for the custom dashboard
  train_baseline.py             # Phase 0: offline baseline model, logged to MLflow + uploads reference/validation sets
  retrain.py                    # run inside retrain-worker by the Airflow DAG: retrain, evaluate, promote-or-reject
monitoring/
  prometheus.yml
  grafana/dashboards, grafana/datasources
data/audio_stream/              # source clips, played back like a live feed
cache/uplink/                   # local cache of flagged clips before upload
storage/minio/, storage/mlflow/ # persisted volumes, gitignored
models/                         # local fallback model cache for inference-api
```

Airflow is intentionally NOT in the main `docker-compose.yml`. Use the official Airflow docker-compose (pulled fresh from the Airflow docs for whichever version is installed — the 3.x compose layout changes across releases, don't hand-roll it from memory). Join it to this stack with the external `ecoecho-net` Docker network so its DAG can reach `http://minio:9000` and `http://mlflow:5000` by service name.

## Key design decision: where retraining labels come from

A real buoy that flags a low-confidence clip does not know the true label. To keep the automated retrain loop honest and demonstrable, this project uses labeled data with **synthetic storm noise injected** on top: noisy clips sent to the buoy are derived from clips whose true label is already known on the cloud side, so retraining has real ground truth without requiring a human-in-the-loop labeling step. This is a deliberate simplification — document it as such in any report or defense, don't present it as a solved general labeling problem. A human-in-the-loop labeling step is the natural "future work" extension.

## Build phases (see project roadmap for full detail)

0. Offline baseline model (no Docker) — `ml/train_baseline.py`
1. Edge stack — `services/inference-api`, `services/buoy-simulator`, Pandera validation
2. Drift injection + monitoring — storm mode, Evidently, Prometheus, Grafana
3. Alert -> cache -> uplink — low-confidence caching, batched MinIO upload
4. Cloud orchestrator — Airflow DAG, retrain, evaluate challenger vs. champion
5. Close the loop — inference-api polls MLflow registry, hot-reloads new model
6. Dashboard + demo polish

Current status: **All phases (0-6) done and verified, plus a custom Overview dashboard built afterward (see below) — `http://localhost:3001`.**

Phase 0: dataset is the Watkins Marine Mammal Sound DB "Best of" cuts (downloaded via the Internet Archive mirror `watkins_best_of_whales_202008`, since cis.whoi.edu was down for maintenance) — 629 clips across 12 species in `data/audio_stream/<species>/`. `python ml/train_baseline.py` runs clean and gets **86% test accuracy** (RandomForest, 200 trees); weakest classes are sperm_whale (67% F1) and long_finned_pilot_whale (71% F1) — worth knowing since those are the classes most likely to drive low-confidence/drift activity once storm noise is added in Phase 2. Model artifacts are in `models/`. One real bug fixed along the way: MLflow 3.x put the plain file-based tracking store into maintenance mode (raises on use) — `train_baseline.py` now defaults `MLFLOW_TRACKING_URI` to a local sqlite file (`ml/mlflow.db`) instead.

Phase 1: `services/inference-api` (FastAPI + Pandera + prometheus-client, serves the Phase 0 model), `services/buoy-simulator` (streams `data/audio_stream/` clips through inference-api on a loop), `services/mlflow` (tracking server), `docker-compose.yml`, `monitoring/prometheus.yml` + Grafana provisioning — `docker compose up --build -d` runs clean, all six containers (`inference-api`, `buoy-simulator`, `mlflow`, `prometheus`, `grafana`, `minio`) reach a healthy/running state. Verified end-to-end: buoy-simulator streams clips through inference-api (16/16 correct predictions in the first smoke test, some flagged low-confidence as expected), Prometheus scrapes both `buoy-simulator:8001` and `inference-api:8000` targets (both `up`), and the Grafana "EcoEcho-Ops Overview" dashboard is provisioned and loads. Two real bugs fixed getting here:
- `services/inference-api/requirements.txt` was missing `librosa`/`soundfile`. It imports the shared `ml/feature_extraction.py` module (for `FEATURE_NAMES`) but that module does `import librosa` at load time even though inference-api never calls the audio-processing functions — container crash-looped with `ModuleNotFoundError` until the deps were added.
- First `docker compose build` hit `pip` read timeouts on a slow connection (large wheels like scikit-learn/mlflow deps). Added `--timeout 120 --retries 5` to the `pip install` line in all three service Dockerfiles.

Also: the repo had zero git commits until this pass — first commit now covers the Phase 0 + Phase 1 work.

Docker/WSL2 unblocked: contrary to the earlier note below, Docker Desktop started fine with no `wsl --install` or reboot needed — WSL2 core was already enabled on this machine, Docker Desktop just hadn't been launched yet. Leaving the note below struck through for history; `docker compose up --build` now just works.

~~Blocked on: this dev machine had no WSL2, so Docker Desktop's engine can't start yet (`docker` CLI is installed and confirmed at v29.6.1, but the daemon needs WSL2). Needs an elevated `wsl --install` + reboot + one Docker Desktop launch, which only a human with admin rights on the machine can do — resume with `docker compose up --build` once that's done.~~

Also noted: MinIO's official Docker Hub repo was archived April 2026 (read-only); compose pins the last official tag (`RELEASE.2025-09-07T16-13-09Z`) — revisit before final submission. Watkins data is licensed for personal/academic, non-commercial use only — credit "Watkins Marine Mammal Sound Database, Woods Hole Oceanographic Institution and the New Bedford Whaling Museum" in any report.

Phase 2: storm noise injection (`ml/storm_noise.py` — broadband + low-frequency rumble mixed in at a tunable RMS-relative intensity) and Evidently-based drift monitoring (`services/inference-api/drift_monitor.py`), wired end-to-end and verified live:
- `ml/train_baseline.py` now also saves `models/reference_features.csv` (the full training feature set) as the Evidently reference distribution.
- `buoy-simulator` mixes storm noise into a clip's waveform before feature extraction when storm mode is on. Storm mode has a startup default (`STORM_MODE_ENABLED`) but is also toggleable live without a restart via a small stdlib `http.server` control endpoint on port 8002 (`GET /storm`, `POST /storm/on`, `POST /storm/off`) — needed since a demo has to show the "storm arrives -> drift detected -> (later) retrain" story without rebuilding containers.
- `inference-api` keeps a rolling window (`DRIFT_WINDOW_SIZE`, default 30) of recent feature vectors and, once full, runs Evidently's `DataDriftPreset` against the reference set, exposing `inference_drift_score` (share of drifted columns) and `inference_drift_detected` (vs. `DRIFT_ALERT_THRESHOLD`) as Prometheus gauges; also returned per-response for the future Phase 3 caching decision.
- Grafana dashboard got 3 new panels: drift score vs. threshold, storm mode indicator, drift-detected indicator.

Verified live: with storm mode off, `inference_drift_score` sits near 0 (0.0-0.03 observed). Toggling `POST :8002/storm/on` degrades prediction quality immediately (low-confidence rate jumped from ~14% baseline to 48% during the run) and `inference_drift_score` climbs smoothly to 0.84-0.88 as the rolling window fills with storm clips, crossing the 0.5 `DRIFT_ALERT_THRESHOLD` and flipping `inference_drift_detected` to 1. Turning storm mode back off reverses the trend as clean clips flush the window. One real bug fixed here: `iter_clips` originally built the clip list grouped by species folder, so any 30-clip window was a near-single-species block and read as "drift" against the all-species reference even with storm mode off (baseline read ~0.6, not near 0) — fixed by shuffling the clip order once before cycling.

Evidently's API changed significantly across versions (old `Report(metrics=[...]).run(current_data=, reference_data=)` docs are stale) — pinned `evidently==0.7.21`, confirmed the real `Report([DataDriftPreset()]).run(current_ds, reference_ds).dict()` shape and the `DriftedColumnsCount` metric's `value.share` field by running it in a throwaway venv before writing the integration, rather than trusting docs alone.

Phase 3: flagged-clip caching + batched MinIO uplink (`services/buoy-simulator/uplink.py`), verified live end-to-end:
- When a `/predict` response comes back `low_confidence` or `drift_detected`, buoy-simulator copies the raw clip plus a metadata JSON (true species, predicted species, confidence, drift score, storm-mode flag, timestamp) into `cache/uplink/pending/`. The true species label is what Phase 4's retrain job will use as ground truth — see "where retraining labels come from" above.
- A background thread flushes whichever comes first: `UPLINK_BATCH_SIZE` pending clips or `UPLINK_INTERVAL_SECONDS` of wall time, uploading each pair to MinIO under `s3://ecoecho-uplink/batches/<batch_id>/` (bucket auto-created on first flush) and moving the local copies to `cache/uplink/uploaded/` so they aren't re-sent.
- New Prometheus metrics (`buoy_clips_cached_total`, `buoy_cache_pending_clips`, `buoy_uplink_batches_total`, `buoy_uplink_clips_uploaded_total`, `buoy_uplink_errors_total`) and 2 new Grafana panels.

Verified live (temporarily dropped `UPLINK_BATCH_SIZE`/`UPLINK_INTERVAL_SECONDS` to 3/20s to speed up the check, then restored the `.env.example` defaults of 20/300 after): with storm mode on, flagged clips accumulated in `cache/uplink/pending/`, batches flushed automatically, zero `buoy_uplink_errors_total`. Confirmed via `mc ls` that objects actually landed in MinIO (82 objects across ~10 batches in the test run) and `mc cat` on a metadata file showed the expected fields, e.g. `{"true_species": "northern_right_whale", "predicted_species": "melon_headed_whale", "confidence": 0.446, "low_confidence": true, "drift_score": 0.0, "drift_detected": false, "storm_mode": true, ...}`.

Also noticed while testing: `.env` (the real, gitignored file, as opposed to `.env.example`) had drifted out of sync — missing `STORM_NOISE_INTENSITY` and `DRIFT_WINDOW_SIZE` from Phase 2. It fell back to code defaults so nothing broke, but keep `.env` in sync with `.env.example` when adding new tunables.

Phase 4: cloud orchestrator (`airflow/`, `ml/retrain.py`, `services/retrain-worker/`), verified end-to-end including the auto-rollback safety net specifically:
- Airflow 3.3.0's official `docker-compose.yaml` (pulled fresh per the warning above, not hand-rolled) lives in `airflow/` as its own stack, joined to `ecoecho-net` only on `airflow-worker` (plus the host's `docker.sock` mounted there) so its tasks can reach `mlflow`/`minio` and launch the retrain-worker container via `DockerOperator`. **Not part of `docker-compose.yml`** and not meant to run continuously alongside it — see "resource note" below.
- DAG (`airflow/dags/retrain_pipeline.py`): a `ShortCircuitOperator` sensor checks MinIO for any pending `batches/` objects; if there are none, the run short-circuits. Otherwise a `DockerOperator` runs `ecoecho-retrain-worker:latest` (build via `docker build -t ecoecho-retrain-worker:latest -f services/retrain-worker/Dockerfile .`), which does retrain + evaluate + register as one atomic step (not three separate tasks) so a mid-pipeline failure can't leave a half-registered model.
- `ml/retrain.py`: pulls `reference_features.csv` + `validation_set.csv` from `s3://ecoecho-uplink/reference/` (uploaded by `train_baseline.py`, see below), pulls any pending flagged clips from `batches/`, trains a challenger RandomForest, evaluates both the challenger and the current MLflow-registry "champion" (alias-based, not the old stage-based API — see MLflow note below) on the *same fixed validation set* every time, and **only promotes the challenger if it actually scores higher**. Processed batches move to `batches-processed/` so they aren't reused.
- `ml/train_baseline.py` now also uploads `reference_features.csv`/`validation_set.csv` (features + species labels, not just features) to `s3://ecoecho-uplink/reference/` — best-effort, skipped silently if MinIO isn't reachable, so Phase 0's "no Docker" invariant still holds. `load_dataset` now sorts clips within each species folder (`os.listdir` order isn't guaranteed) so the train/validation split is reproducible run to run — the auto-rollback comparison is meaningless if the "fixed" validation set silently isn't fixed.

**Auto-rollback safety net, actually verified, not assumed:** bootstrapped a champion (v1, 87.3% validation accuracy) from real data, then deliberately trained a crippled challenger (`N_ESTIMATORS=1` env override, added specifically to make this test possible) which scored 65.9%. Confirmed in both the log output and by querying the registry directly that the weak challenger was logged as v2 but the `champion` alias stayed on v1 — it was evaluated and correctly rejected, not just skipped. Then retrained normally again with more data (v3, 90.5%), confirmed it correctly beat v1 and was promoted.

Three real bugs found and fixed getting Phase 4 working, two of them the kind that would have silently broken the demo without ever throwing an error a casual test would catch:
- **MLflow rejected requests from any hostname but `localhost`.** MLflow 3.5+ added DNS-rebinding protection (`--allowed-hosts`) that isn't documented as a breaking change but blocks every container-to-container call by default (`http://mlflow:5000` gets a 403 "possible DNS rebinding attack"). This had been latent since Phase 1 — nothing before Phase 4 ever actually called MLflow from another container. Fixed by adding `--allowed-hosts` to `services/mlflow/Dockerfile`'s CMD.
- **Registered models existed but couldn't be loaded remotely.** `mlflow server --default-artifact-root /mlflow/artifacts` returns a bare local filesystem path as the artifact location — readable only from inside the mlflow container itself. Any other container calling `mlflow.sklearn.load_model()` got `No such artifact: ''`. The registry metadata (versions, aliases) all worked fine, which made this easy to miss — it only breaks at the "load the actual model" step. Fixed by switching to `--artifacts-destination`, which proxies artifact downloads through MLflow's HTTP API instead. **This required wiping `storage/mlflow/` and re-seeding from scratch** — existing experiments keep their original (broken) artifact location even after the server flag changes; only new experiments get the proxied `mlflow-artifacts:/` scheme.
- Airflow's own compose network is isolated from `ecoecho-net` by default — the sensor task (runs inline in `airflow-worker`, not via `DockerOperator`) couldn't resolve `minio` until `airflow-worker` was explicitly joined to both networks.

**Resource note (matters for anyone re-running this):** running the main stack (6 containers) and Airflow's full CeleryExecutor stack (7 more — Postgres, Redis, scheduler, api-server, dag-processor, worker, triggerer) at the same time crashed Docker Desktop entirely, twice, on this dev machine. Airflow is intentionally not part of `docker-compose.yml` for exactly this reason (see architecture note above) — bring it up only when triggering a retrain (`cd airflow && docker compose up -d`, `airflow dags trigger ecoecho_retrain_pipeline`), then `docker compose down` it again rather than leaving both stacks running.

Also: Grafana's admin password can drift from `.env`'s `GRAFANA_ADMIN_PASSWORD` — it's only applied on the container's very first boot (stored in its own sqlite db after that), so it doesn't reset just because the env var still says `admin`. If login fails, `docker exec claudewhale-grafana-1 grafana cli admin reset-admin-password admin` forces it back in line with `.env` regardless of history.

Phase 5: closing the loop — `services/inference-api/model_loader.py` now tries the MLflow registry's `champion` alias first on startup, falling back to the local Phase 0 joblib files only if the registry has no champion yet or MLflow is unreachable (keeps the service usable before Phase 4 has ever run). A background thread polls every `MODEL_POLL_INTERVAL_SECONDS` (default 30) and hot-swaps in a new champion version — model + label encoder (logged as an MLflow run artifact by `retrain.py` specifically so this works) — under a lock, with no restart. New Prometheus gauges `inference_active_model_version` / `inference_active_model_is_registry` (0 = running the local fallback) replace the old static `inference_active_model_loaded_timestamp`, which couldn't represent a version that changes at runtime.

Verified live: confirmed via `docker inspect` that `inference-api`'s `RestartCount` stayed at 0 throughout, while `/health`'s `model_version` field flipped from `"1"` to `"3"` within 5 seconds of a new champion being promoted, and a subsequent `/predict` call correctly used the new model. This is the actual "self-healing" claim — verified end-to-end, not just that each piece works in isolation.

Phase 6: dashboard polish. Reorganized the Grafana dashboard into three labeled rows (edge inference / drift & storm detection / cloud retraining pipeline) plus an intro text panel, and color-coded the status panels (storm mode, drift detected, active model version, low-confidence rate) with Grafana's theme-aware thresholds so problem states read visually rather than requiring the viewer to parse numbers.

Also fixed while doing this: Grafana's admin password had no durable storage — it lives in `/var/lib/grafana`, which had no volume mount, so it silently reset to the `.env` default every time the container got recreated (which happened repeatedly during the Docker Desktop crash recovery earlier in this session — see the resource note above). Added `./storage/grafana:/var/lib/grafana`, matching the existing pattern for `storage/minio` and `storage/mlflow`. If login ever fails after a crash/recreate anyway, `docker exec claudewhale-grafana-1 grafana cli admin reset-admin-password admin` forces it back to the `.env` value.

## Custom Overview dashboard (post-Phase-6, not in the original roadmap)

Built at the user's request, after seeing a mockup screenshot of a polished custom ops dashboard and asking for it. Scoped down from the mockup deliberately: 5 pages were built, not the ~10 implied by the mockup's sidebar (Reports, Settings, Data Explorer don't exist), and every number on every page is real, not fabricated to match the mockup's placeholder values.

- `services/dashboard-api/` (FastAPI, port 8003): polls Prometheus, MLflow, buoy-simulator, and MinIO every `POLL_INTERVAL_SECONDS` (default 5) and exposes one aggregated `GET /api/overview` JSON payload plus `GET /api/spectrogram.png` (proxies the buoy's current spectrogram). Every page below reads from this same payload. Also derives a rolling system-log/alerts feed by detecting real state transitions across polls (drift detected/cleared, storm mode toggled, model hot-swapped, new prediction, retrain completed) — this is not a real log-aggregation pipeline, it's synthesized from the same metrics Grafana already shows, which is why it's a separate concern from an actual logging stack.
- `services/dashboard-frontend/` (nginx, port 3001): plain HTML/CSS/JS, no build tooling/framework/CDN dependency. `style.css` and `common.js` (auth guard, sidebar nav, fetch/format helpers, hand-drawn SVG drift chart — no charting library) are shared across pages:
  - `index.html` — sign-in gate (see below)
  - `overview.html` — slim summary (6 stat cards) + a topic card linking to each page below, once the user asked for separate pages instead of one dense page
  - `live.html` — Live Monitoring: spectrogram, current prediction, recent-predictions log
  - `drift.html` — Drift Detection: drift score chart, threshold, storm mode
  - `registry.html` — Model Registry: retraining pipeline, champion-vs-previous performance bars, flagged clips table
  - `health.html` — System Health: full event log, alerts, buoy deployment status, uplink counters
- Sign-in gate (`index.html`, whale-themed design supplied by the user): demo-only client-side check (hardcoded credential matching the one shown on the card, `sessionStorage` as the "session") — not real backend auth. Explicitly appropriate only because this is a single-machine local demo with no backend user system and nothing sensitive behind it; every other dashboard page redirects back to it if the session flag isn't set.
- `ml/spectrogram_image.py`: renders a real mel-spectrogram PNG (librosa + matplotlib, `Agg` backend) of buoy-simulator's most-recently-streamed clip. `buoy-simulator` keeps the latest clip's raw audio in memory (see `main.py`'s `_latest_audio`) and exposes it at `GET :8002/spectrogram.png`, `GET :8002/buoy-info`, `GET :8002/latest-prediction`.
- The mockup showed a buoy GPS map and per-clip "duration" — this project has no real hardware, so there's no real GPS or per-clip duration to show. `buoy-simulator`'s location (`BUOY_LOCATION` in `main.py`) is an explicit fixed constant (Stellwagen Bank, Gulf of Maine), labeled in the UI as simulated, not live-tracked — chosen over a fake live-updating map, which would have been actively misleading.
- `ml/retrain.py` now logs `val_precision`/`val_recall`/`val_f1` (macro-averaged) alongside `val_accuracy`, so the model-comparison bars have real data for all four metrics — added specifically because the mockup wanted a multi-metric comparison and only accuracy existed before. Model versions registered before this change only have `val_accuracy` logged and can't be retroactively backfilled (immutable MLflow runs) — the frontend handles this by rendering "N/A" rather than a fake 0 or omitting the row silently.
- One real bug caught building this: Prometheus's `increase()` over a 24h window extrapolates when the series hasn't existed that long, which produced a wildly inflated "90 low-confidence clips" reading minutes after a restart. Switched the "low-confidence clips" stat to a 15-minute window instead of 24h specifically to avoid this — a demo environment that doesn't run continuously should not show a stat implying 24h of continuous history.

## Commands

```bash
# core stack
docker compose up --build

# core stack, rebuild a single service after code changes
docker compose up --build inference-api

# tear down
docker compose down

# offline baseline training (Phase 0, before Docker is involved)
python ml/train_baseline.py
```

## Conventions

- Python, one virtualenv per service if running outside Docker; keep `requirements.txt` per service minimal and pinned once the project stabilizes.
- Config via environment variables (see `.env.example`), not hardcoded values — thresholds like `DRIFT_ALERT_THRESHOLD` and `CONFIDENCE_THRESHOLD` should stay tunable without a code change.
- Every service that emits metrics should expose them in Prometheus format (use `prometheus-client`) rather than logging-only.
- Don't pin Docker image tags to `latest` once the demo is stable — pin exact versions for reproducibility before submission/defense.

## Things to be careful about

- Don't let the retrain loop auto-promote a worse model — always compare challenger vs. champion on a fixed validation set before registering as "Production" (this is the auto-rollback safety net; implement it before relying on the demo).
- Keep `storage/`, `cache/`, and `data/audio_stream/` out of git (already in `.gitignore`) — these can get large.
- MLflow moved to a 3.x API generation; check current docs rather than assuming 2.x tracking/registry syntax still applies.
