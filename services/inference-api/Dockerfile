# build context is the repo root (see docker-compose.yml) so this can
# pull in ml/feature_extraction.py, the shared feature-contract module.
FROM python:3.11.9-slim-bookworm

WORKDIR /app

COPY services/inference-api/requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

COPY services/inference-api/main.py services/inference-api/model_loader.py services/inference-api/schemas.py services/inference-api/drift_monitor.py ./
COPY ml/feature_extraction.py .

ENV MODEL_DIR=/app/models
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
