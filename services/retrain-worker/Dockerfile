# build context is the repo root (see the retrain-worker build command in
# CLAUDE.md) so this can pull in ml/feature_extraction.py and ml/retrain.py.
# Launched on-demand by the Airflow DAG via DockerOperator — not part of the
# main docker-compose.yml, since it isn't a long-running service.
FROM python:3.11.9-slim-bookworm

# Use HTTPS for Debian repositories.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources

# libsndfile1 is required by soundfile/librosa to decode .wav clips
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY services/retrain-worker/requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --no-cache-dir \
        --timeout 300 \
        --retries 10 \
        --index-url https://pypi.org/simple \
        -r requirements.txt
COPY ml/retrain.py ml/feature_extraction.py ./

CMD ["python", "retrain.py"]
