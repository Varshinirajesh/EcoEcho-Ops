# EcoEcho-Ops

**An MLOps pipeline for underwater audio species classification and intelligent buoy monitoring.**

EcoEcho-Ops uses machine learning to classify underwater audio recordings and supports a model lifecycle with experiment tracking, automated retraining, model versioning, monitoring, and a live dashboard.

## Key Features

* **Underwater Species Classification** — Extracts audio features using MFCCs, spectral centroid, RMS energy, and zero-crossing rate.
* **Machine Learning Pipeline** — Trains and evaluates a baseline species classification model.
* **Model Tracking** — Uses MLflow to track experiments and manage registered model versions.
* **Automated Retraining** — Uses Apache Airflow to orchestrate retraining workflows.
* **Model Promotion** — Evaluates challenger models against the current champion before promotion.
* **Object Storage** — Uses MinIO to store reference and validation datasets.
* **Inference API** — Serves predictions through a FastAPI service.
* **Buoy Simulation** — Simulates incoming underwater audio monitoring data.
* **Monitoring and Metrics** — Integrates Prometheus and Grafana.
* **Live Dashboard** — Provides a web interface for monitoring classification results.

## Architecture

```text
Underwater Audio Dataset
          |
          v
   Feature Extraction
          |
          v
   Baseline Model Training
          |
          v
     MLflow Tracking
          |
          v
      Inference API <------ Buoy Simulator
          |                       |
          v                       v
    Species Predictions     Live Audio Events
          |
          v
    Dashboard API
          |
          v
    Dashboard Frontend

Airflow ---> Automated Retraining ---> MLflow Model Registry
                    |
                    v
             MinIO Reference Data

Prometheus ---> Metrics ---> Grafana
```

## Technology Stack

| Component                   | Technology                     |
| --------------------------- | ------------------------------ |
| Machine Learning            | Python, scikit-learn, librosa  |
| Model Tracking and Registry | MLflow                         |
| Workflow Orchestration      | Apache Airflow                 |
| Object Storage              | MinIO                          |
| API                         | FastAPI                        |
| Containers                  | Docker, Docker Compose         |
| Metrics                     | Prometheus                     |
| Visualization               | Grafana                        |
| Dashboard                   | Web frontend and dashboard API |

## Project Structure

```text
EcoEcho-Ops/
├── airflow/
│   ├── dags/
│   │   └── retrain_pipeline.py
│   └── docker-compose.yaml
├── deploy/
│   └── Caddyfile
├── ml/
│   ├── feature_extraction.py
│   ├── train_baseline.py
│   └── retrain.py
├── models/
│   └── .gitkeep
├── monitoring/
├── services/
│   ├── inference-api/
│   ├── buoy-simulator/
│   └── dashboard/
├── docker-compose.yml
├── docker-compose.prod.yml
├── .env.example
├── .gitignore
├── DEPLOY.md
└── README.md
```

## Dataset

The project uses underwater audio recordings organized by species.

The development dataset contains approximately **1,697 audio clips across 32 species**.

Audio features include:

* 13 Mel-frequency cepstral coefficients (MFCCs), using mean and standard deviation
* Spectral centroid mean and standard deviation
* RMS energy mean and standard deviation
* Zero-crossing rate mean and standard deviation

The extracted feature vector contains 32 features per audio clip.

**Note:** The original audio dataset and generated model artifacts are not included in the Git repository. Configure your local dataset and model artifacts before running the training pipeline.

## Model Training

From the project root, activate Docker Desktop and run the baseline training script in your configured Python environment:

```powershell
python ml/train_baseline.py
```

The baseline training pipeline extracts features, trains a classifier, and evaluates its performance.

The initial evaluation achieved approximately **84.71% validation accuracy** and a macro F1-score of **0.85** on the development dataset. Results may vary when the dataset or training configuration changes.

## Getting Started

### Prerequisites

Install and configure:

* Git
* Docker Desktop with Docker Compose
* Python 3.10 or a compatible version for local ML scripts
* The underwater audio dataset

### 1. Clone the repository

```powershell
git clone https://github.com/Varshinirajesh/EcoEcho-Ops.git
cd EcoEcho-Ops
```

### 2. Configure environment variables

Create your local environment file from the example:

```powershell
Copy-Item .env.example .env
```

Review `.env` and configure the required credentials, service settings, and storage parameters for your environment. Never commit real secrets or passwords.

### 3. Start the services

From the project root:

```powershell
docker compose up -d --build
```

Check service status:

```powershell
docker compose ps
```

If you want to run the Airflow stack, follow the setup instructions in `airflow/docker-compose.yaml` and `DEPLOY.md`.

### 4. Open the interfaces

When the corresponding services are running, use these local URLs:

| Service                     | URL                        |
| --------------------------- | -------------------------- |
| Dashboard                   | http://localhost:3001      |
| Inference API               | http://localhost:8000      |
| Inference API documentation | http://localhost:8000/docs |
| Dashboard API               | http://localhost:8003      |
| MLflow                      | http://localhost:5000      |
| MinIO Console               | http://localhost:9001      |
| Prometheus                  | http://localhost:9090      |
| Grafana                     | http://localhost:3000      |
| Airflow                     | http://localhost:8080      |

Some services may require separate startup or configuration, depending on the Compose file used.

## Automated Retraining

The retraining pipeline uses Apache Airflow to coordinate model retraining.

The workflow is designed to:

1. Retrieve reference and validation data from object storage.
2. Validate the feature schema and species labels.
3. Train a challenger model.
4. Log the training run and model artifacts in MLflow.
5. Compare challenger performance with the current champion.
6. Promote the challenger when it meets the configured promotion condition.

This workflow supports reproducible model evaluation and controlled model updates.

## Monitoring

EcoEcho-Ops includes monitoring components for observing service and model-related activity:

* **Prometheus** collects configured metrics.
* **Grafana** visualizes monitoring data.
* **MLflow** records experiments and model versions.
* **Dashboard** displays available monitoring and classification information.

The exact metrics and panels depend on the active service and monitoring configuration.

## Security and Data Management

* Keep `.env` files and credentials out of version control.
* Do not commit large audio datasets, generated models, runtime databases, or local logs.
* Use `.env.example` to document required environment variables without including real credentials.
* Configure access controls and strong credentials before exposing services outside your local machine.

## Future Improvements

Potential extensions include:

* Expanding the underwater species dataset.
* Evaluating additional classification algorithms.
* Adding model drift detection and alerting.
* Improving real-time inference and dashboard visualizations.
* Adding automated tests and continuous integration.
* Deploying the stack to a cloud or edge environment.

## Author

**Varshini Rajesh**

GitHub: [Varshinirajesh](https://github.com/Varshinirajesh)

## License

Add a license file if you intend to distribute this project publicly. Until a license is specified, reuse and redistribution are subject to applicable copyright law.
