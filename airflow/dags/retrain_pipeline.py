"""Phase 4: cloud orchestrator DAG.

sensor -> retrain_evaluate_register

The sensor checks MinIO (ecoecho-uplink bucket) for at least one pending
uplink batch under batches/ (buoy-simulator writes these — see
services/buoy-simulator/uplink.py) and short-circuits the run if there's
nothing new. The actual retrain/evaluate/register work runs as ONE task in
its own container (ecoecho-retrain-worker, see services/retrain-worker/ and
ml/retrain.py) via DockerOperator on the shared ecoecho-net network — kept
as a single atomic step rather than three tasks passing model state through
XCom, so a mid-pipeline failure can't leave a half-registered model.

This DAG only talks to MinIO and MLflow, both reachable at their ecoecho-net
service names — it has no access to (and doesn't need) the buoy's local
disk or the main docker-compose.yml stack directly.
"""

from datetime import datetime

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.python import ShortCircuitOperator
from airflow.sdk import DAG

MINIO_ENDPOINT = "minio:9000"
MINIO_ROOT_USER = "minioadmin"
MINIO_ROOT_PASSWORD = "minioadmin"
MINIO_BUCKET = "ecoecho-uplink"


def _has_pending_batches() -> bool:
    from minio import Minio

    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ROOT_USER, secret_key=MINIO_ROOT_PASSWORD, secure=False)
    if not client.bucket_exists(MINIO_BUCKET):
        return False
    objects = client.list_objects(MINIO_BUCKET, prefix="batches/", recursive=False)
    return any(True for _ in objects)


with DAG(
    dag_id="ecoecho_retrain_pipeline",
    description="Self-healing retrain loop: sensor -> retrain/evaluate/register challenger vs. champion",
    schedule="*/10 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ecoecho"],
) as dag:
    sensor = ShortCircuitOperator(
        task_id="sensor_pending_batches",
        python_callable=_has_pending_batches,
    )

    retrain_evaluate_register = DockerOperator(
        task_id="retrain_evaluate_register",
        image="ecoecho-retrain-worker:latest",
        api_version="auto",
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        network_mode="ecoecho-net",
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "MINIO_ENDPOINT": MINIO_ENDPOINT,
            "MINIO_ROOT_USER": MINIO_ROOT_USER,
            "MINIO_ROOT_PASSWORD": MINIO_ROOT_PASSWORD,
            "MINIO_BUCKET": MINIO_BUCKET,
            "MODEL_REGISTRY_NAME": "ecoecho-classifier",
        },
        mount_tmp_dir=False,
    )

    sensor >> retrain_evaluate_register
