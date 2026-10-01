# MinIO in this stack backs the satellite-uplink clip storage, not MLflow
# artifacts — the tracking server owns its own volume and every client
# (train_baseline.py, retrain.py, inference-api) talks to it over HTTP.
#
# --artifacts-destination (not --default-artifact-root) is what actually
# makes that true: it makes new experiments get mlflow-artifacts:/ run URIs
# that proxy through the server's HTTP API, so a remote client (a different
# container, or retrain-worker) can load a registered model without direct
# filesystem access to /mlflow/artifacts. --default-artifact-root hands back
# the server's own local filesystem path, which is unreachable from anywhere
# else — silently broke every remote mlflow.sklearn.load_model() call.
FROM python:3.11.9-slim-bookworm

WORKDIR /mlflow

COPY services/mlflow/requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--allowed-hosts", "localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,mlflow,mlflow:5000"]
