# build context is the repo root (see docker-compose.yml) so this can
# pull in ml/feature_extraction.py, ml/storm_noise.py, and
# ml/spectrogram_image.py, the shared audio-processing modules.
FROM python:3.11.9-slim-bookworm

# libsndfile1 is required by soundfile/librosa to decode .wav clips
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY services/buoy-simulator/requirements.txt .
RUN pip install --no-cache-dir --timeout 600 --retries 10 -r requirements.txt
COPY services/buoy-simulator/main.py services/buoy-simulator/uplink.py ./
COPY ml/feature_extraction.py ml/storm_noise.py ml/spectrogram_image.py ./

EXPOSE 8001 8002

CMD ["python", "main.py"]
