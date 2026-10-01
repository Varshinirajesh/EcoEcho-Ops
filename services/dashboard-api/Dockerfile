# build context is the repo root (see docker-compose.yml).
# Aggregates Prometheus + MLflow + buoy-simulator into one JSON feed for
# the custom Overview page (services/dashboard-frontend) — read-only,
# talks to nothing the rest of the stack doesn't already expose.
FROM python:3.11.9-slim-bookworm

WORKDIR /app

COPY services/dashboard-api/requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

COPY services/dashboard-api/main.py .

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
