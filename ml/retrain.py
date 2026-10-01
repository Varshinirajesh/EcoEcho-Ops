"""Phase 4: cloud-side retrain job, triggered by the Airflow DAG.

Runs in its own container (services/retrain-worker), not on the Airflow
host, and only talks to MinIO (the landing zone) and MLflow — it never
reaches into the buoy's local disk. Steps:

1. Pull the labeled reference dataset (models/reference_features.csv +
   validation_set.csv, as uploaded by train_baseline.py) from MinIO.
2. Pull any new flagged clips uploaded by buoy-simulator's uplink batches,
   extract features, pair with their true_species (see CLAUDE.md's "where
   retraining labels come from" — the synthetic-storm-noise simplification).
3. Train a challenger RandomForest on reference + new clips.
4. Evaluate the challenger AND the current "champion" (if one exists) on
   the fixed validation set persisted by train_baseline.py, then only
   promote the challenger if it's actually better — the auto-rollback
   safety net CLAUDE.md calls out as required before relying on the demo.
5. Move processed batches to batches-processed/ so they aren't reused.
"""


import json
import os
import shutil
import tempfile

import joblib
import mlflow
import pandas as pd
from minio import Minio
from minio.commonconfig import CopySource
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

from feature_extraction import FEATURE_NAMES, extract_features


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",
)

MLFLOW_EXPERIMENT = "ecoecho-retrain"

MODEL_NAME = os.environ.get(
    "MODEL_REGISTRY_NAME",
    "ecoecho-classifier",
)

CHAMPION_ALIAS = "champion"

MINIO_ENDPOINT = os.environ.get(
    "MINIO_ENDPOINT",
    "minio:9000",
)

MINIO_ROOT_USER = os.environ.get(
    "MINIO_ROOT_USER",
    "minioadmin",
)

MINIO_ROOT_PASSWORD = os.environ.get(
    "MINIO_ROOT_PASSWORD",
    "minioadmin",
)

MINIO_BUCKET = os.environ.get(
    "MINIO_BUCKET",
    "ecoecho-uplink",
)

# Default is the same 200-tree Random Forest used by the retraining design.
# Can be overridden through the environment when deliberately testing
# rollback behavior.
N_ESTIMATORS = int(
    os.environ.get("N_ESTIMATORS", "200")
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _eval_metrics(y_true, y_pred) -> dict[str, float]:
    """
    Calculate macro-averaged classification metrics.

    zero_division=0 prevents warnings/errors when a class is absent from
    predictions for a validation run.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ---------------------------------------------------------------------------
# MinIO
# ---------------------------------------------------------------------------

def _minio_client() -> Minio:
    """Create a non-TLS MinIO client for the Docker network."""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ROOT_USER,
        secret_key=MINIO_ROOT_PASSWORD,
        secure=False,
    )


# ---------------------------------------------------------------------------
# Reference / validation data
# ---------------------------------------------------------------------------

def _load_reference_data(
    client: Minio,
    tmp_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download the fixed reference and validation datasets from MinIO.
    """
    ref_path = os.path.join(
        tmp_dir,
        "reference_features.csv",
    )

    val_path = os.path.join(
        tmp_dir,
        "validation_set.csv",
    )

    try:
        client.fget_object(
            MINIO_BUCKET,
            "reference/reference_features.csv",
            ref_path,
        )

        client.fget_object(
            MINIO_BUCKET,
            "reference/validation_set.csv",
            val_path,
        )

    except Exception as e:
        raise RuntimeError(
            "No reference/validation dataset found in MinIO under "
            "reference/. Run ml/train_baseline.py with MinIO reachable "
            "at least once first."
        ) from e

    reference_df = pd.read_csv(ref_path)
    validation_df = pd.read_csv(val_path)

    # -----------------------------------------------------------------------
    # Feature-contract validation
    # -----------------------------------------------------------------------

    missing_reference_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in reference_df.columns
    ]

    missing_validation_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in validation_df.columns
    ]

    if missing_reference_features:
        raise RuntimeError(
            "Reference dataset is missing required features: "
            + ", ".join(missing_reference_features)
        )

    if missing_validation_features:
        raise RuntimeError(
            "Validation dataset is missing required features: "
            + ", ".join(missing_validation_features)
        )

    if "species" not in reference_df.columns:
        raise RuntimeError(
            "reference_features.csv is missing the required 'species' column."
        )

    if "species" not in validation_df.columns:
        raise RuntimeError(
            "validation_set.csv is missing the required 'species' column."
        )

    if reference_df.empty:
        raise RuntimeError("reference_features.csv contains no rows.")

    if validation_df.empty:
        raise RuntimeError("validation_set.csv contains no rows.")

    return reference_df, validation_df


# ---------------------------------------------------------------------------
# Pending uplink batches
# ---------------------------------------------------------------------------

def _pending_batch_ids(client: Minio) -> list[str]:
    """
    Return all pending batch IDs under batches/<batch_id>/.
    """
    ids = set()

    for obj in client.list_objects(
        MINIO_BUCKET,
        prefix="batches/",
        recursive=False,
    ):
        parts = obj.object_name.strip("/").split("/")

        if len(parts) >= 2:
            ids.add(parts[1])

    return sorted(ids)


# ---------------------------------------------------------------------------
# Load flagged clips from MinIO
# ---------------------------------------------------------------------------

def _load_batch_clips(
    client: Minio,
    batch_ids: list[str],
    tmp_dir: str,
) -> pd.DataFrame:
    """
    Download WAV + JSON metadata from each pending batch and extract the
    exact same 32-feature contract used by the baseline model.
    """
    rows = []

    for batch_id in batch_ids:
        prefix = f"batches/{batch_id}/"

        objects = list(
            client.list_objects(
                MINIO_BUCKET,
                prefix=prefix,
                recursive=True,
            )
        )

        stems = sorted(
            {
                o.object_name[len(prefix):].rsplit(".", 1)[0]
                for o in objects
                if o.object_name.lower().endswith((".wav", ".json"))
            }
        )

        for stem in stems:
            wav_key = f"{prefix}{stem}.wav"
            json_key = f"{prefix}{stem}.json"

            wav_path = os.path.join(
                tmp_dir,
                f"{batch_id}_{stem}.wav",
            )

            json_path = os.path.join(
                tmp_dir,
                f"{batch_id}_{stem}.json",
            )

            # Make sure both files actually exist before attempting to read
            # them. This prevents a partially uploaded clip from silently
            # breaking the retraining job later.
            try:
                client.fget_object(
                    MINIO_BUCKET,
                    wav_key,
                    wav_path,
                )

                client.fget_object(
                    MINIO_BUCKET,
                    json_key,
                    json_path,
                )

            except Exception as e:
                raise RuntimeError(
                    f"Failed to download complete clip '{stem}' "
                    f"from batch '{batch_id}'."
                ) from e

            with open(
                json_path,
                "r",
                encoding="utf-8",
            ) as f:
                meta = json.load(f)

            if "true_species" not in meta:
                raise RuntimeError(
                    f"Metadata file '{json_key}' is missing "
                    "'true_species'."
                )

            true_species = meta["true_species"]

            if not true_species:
                raise RuntimeError(
                    f"Metadata file '{json_key}' contains an empty "
                    "'true_species'."
                )

            features = extract_features(wav_path)

            if len(features) != len(FEATURE_NAMES):
                raise RuntimeError(
                    f"Feature extraction returned {len(features)} values "
                    f"for '{stem}', but the model contract requires "
                    f"{len(FEATURE_NAMES)} features."
                )

            rows.append(
                {
                    **dict(
                        zip(
                            FEATURE_NAMES,
                            features,
                        )
                    ),
                    "species": true_species,
                }
            )

    if not rows:
        raise RuntimeError(
            "Pending batches were found, but no valid clips could be loaded."
        )

    return pd.DataFrame(
        rows,
        columns=FEATURE_NAMES + ["species"],
    )


# ---------------------------------------------------------------------------
# Mark processed batches
# ---------------------------------------------------------------------------

def _mark_batches_processed(
    client: Minio,
    batch_ids: list[str],
) -> None:
    """
    Move successfully processed batches from batches/ to
    batches-processed/.
    """
    for batch_id in batch_ids:
        prefix = f"batches/{batch_id}/"

        for obj in list(
            client.list_objects(
                MINIO_BUCKET,
                prefix=prefix,
                recursive=True,
            )
        ):
            new_key = obj.object_name.replace(
                "batches/",
                "batches-processed/",
                1,
            )

            client.copy_object(
                MINIO_BUCKET,
                new_key,
                CopySource(
                    MINIO_BUCKET,
                    obj.object_name,
                ),
            )

            client.remove_object(
                MINIO_BUCKET,
                obj.object_name,
            )


# ---------------------------------------------------------------------------
# Main retraining pipeline
# ---------------------------------------------------------------------------

def main():
    print("============================================================")
    print("EcoEcho retraining pipeline")
    print("============================================================")

    client = _minio_client()

    # Configure MLflow before creating the client or starting a run.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    mlflow_client = mlflow.MlflowClient()

    # -----------------------------------------------------------------------
    # Find pending data
    # -----------------------------------------------------------------------

    batch_ids = _pending_batch_ids(client)

    if not batch_ids:
        print(
            "No new uplink batches pending, nothing to retrain on. Exiting."
        )
        return

    print(
        f"Pending batches: {len(batch_ids)}"
    )

    for batch_id in batch_ids:
        print(f"  - {batch_id}")

    # -----------------------------------------------------------------------
    # Download reference + validation and load new flagged clips
    # -----------------------------------------------------------------------

    with tempfile.TemporaryDirectory() as tmp_dir:
        reference_df, validation_df = _load_reference_data(
            client,
            tmp_dir,
        )

        new_df = _load_batch_clips(
            client,
            batch_ids,
            tmp_dir,
        )

    print(
        f"Reference clips: {len(reference_df)}, "
        f"new flagged clips: {len(new_df)} "
        f"(from {len(batch_ids)} batch(es)), "
        f"validation clips: {len(validation_df)}"
    )

    # -----------------------------------------------------------------------
    # Combine reference + newly flagged clips
    # -----------------------------------------------------------------------

    train_df = pd.concat(
        [
            reference_df,
            new_df,
        ],
        ignore_index=True,
    )

    # Ensure there are no missing labels.
    if train_df["species"].isna().any():
        raise RuntimeError(
            "Training data contains rows with missing species labels."
        )

    if validation_df["species"].isna().any():
        raise RuntimeError(
            "Validation data contains rows with missing species labels."
        )

    # -----------------------------------------------------------------------
    # Validate feature values
    # -----------------------------------------------------------------------

    X_train_df = train_df[FEATURE_NAMES]
    X_val_df = validation_df[FEATURE_NAMES]

    if X_train_df.isna().any().any():
        raise RuntimeError(
            "Training features contain NaN values."
        )

    if X_val_df.isna().any().any():
        raise RuntimeError(
            "Validation features contain NaN values."
        )

    X_train = X_train_df.values
    X_val = X_val_df.values

    # -----------------------------------------------------------------------
    # Label encoding
    # -----------------------------------------------------------------------

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(
        train_df["species"]
    )

    # Every validation class must exist in the training label encoder.
    unknown_validation_classes = sorted(
        set(validation_df["species"])
        - set(label_encoder.classes_)
    )

    if unknown_validation_classes:
        raise RuntimeError(
            "Validation contains species not present in the training data: "
            + ", ".join(unknown_validation_classes)
        )

    y_val = label_encoder.transform(
        validation_df["species"]
    )

    print(
        f"Training classes: {len(label_encoder.classes_)}"
    )

    print(
        "Classes:"
    )

    for species in label_encoder.classes_:
        print(f"  - {species}")

    # -----------------------------------------------------------------------
    # MLflow run
    # -----------------------------------------------------------------------

    run_name = f"retrain-{batch_ids[-1]}"

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        # ---------------------------------------------------------------
        # Train challenger
        # ---------------------------------------------------------------

        challenger = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=42,
            class_weight="balanced",
        )

        challenger.fit(
            X_train,
            y_train,
        )

        challenger_predictions = challenger.predict(
            X_val
        )

        challenger_metrics = _eval_metrics(
            y_val,
            challenger_predictions,
        )

        # ---------------------------------------------------------------
        # Log parameters
        # ---------------------------------------------------------------

        mlflow.log_param(
            "n_estimators",
            N_ESTIMATORS,
        )

        mlflow.log_param(
            "n_train_samples",
            len(train_df),
        )

        mlflow.log_param(
            "n_reference_samples",
            len(reference_df),
        )

        mlflow.log_param(
            "n_new_clips",
            len(new_df),
        )

        mlflow.log_param(
            "n_batches",
            len(batch_ids),
        )

        mlflow.log_param(
            "n_validation_samples",
            len(validation_df),
        )

        mlflow.log_param(
            "n_classes",
            len(label_encoder.classes_),
        )

        mlflow.log_param(
            "feature_count",
            len(FEATURE_NAMES),
        )

        mlflow.log_param(
            "feature_contract",
            ",".join(FEATURE_NAMES),
        )

        # ---------------------------------------------------------------
        # Log validation metrics
        # ---------------------------------------------------------------

        for name, value in challenger_metrics.items():
            mlflow.log_metric(
                f"val_{name}",
                value,
            )

        print(
            f"Challenger validation metrics: "
            f"{challenger_metrics}"
        )

        # ---------------------------------------------------------------
        # Save label encoder
        # ---------------------------------------------------------------

        label_encoder_path = os.path.join(
            tempfile.gettempdir(),
            f"label_encoder_{run.info.run_id}.joblib",
        )

        joblib.dump(
            label_encoder,
            label_encoder_path,
        )

        mlflow.log_artifact(
            label_encoder_path,
            artifact_path="artifacts",
        )

        # ---------------------------------------------------------------
        # Log + register challenger model
        #
        # IMPORTANT:
        # MLflow 3.x uses skops for sklearn model serialization.
        # RandomForest contains sklearn.tree._tree.Tree objects, which
        # skops treats as untrusted unless explicitly allowed.
        #
        # This model was created locally in this trusted retraining
        # container, so the Tree type is explicitly trusted.
        # ---------------------------------------------------------------

        model_info = mlflow.sklearn.log_model(
            challenger,
            name="model",
            registered_model_name=MODEL_NAME,
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        new_version = model_info.registered_model_version

        if new_version is None:
            raise RuntimeError(
                "MLflow logged the model but did not return a registered "
                "model version."
            )

        print(
            f"Registered challenger model version: {new_version}"
        )

        # ---------------------------------------------------------------
        # Evaluate existing champion
        # ---------------------------------------------------------------

        champion_metrics = None
        champion_version = None

        try:
            champion_mv = (
                mlflow_client.get_model_version_by_alias(
                    MODEL_NAME,
                    CHAMPION_ALIAS,
                )
            )

            champion_version = champion_mv.version

            champion_model = mlflow.sklearn.load_model(
                f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
            )

            champion_predictions = champion_model.predict(
                X_val
            )

            champion_metrics = _eval_metrics(
                y_val,
                champion_predictions,
            )

            print(
                f"Current champion "
                f"(v{champion_version}) validation metrics: "
                f"{champion_metrics}"
            )

        except Exception as e:
            print(
                "No existing champion — this is the first registered "
                "version."
            )

            print(
                f"Champion lookup detail: {type(e).__name__}: {e}"
            )

        # ---------------------------------------------------------------
        # Challenger vs champion
        # ---------------------------------------------------------------

        promote = (
            champion_metrics is None
            or challenger_metrics["accuracy"]
            > champion_metrics["accuracy"]
        )

        mlflow.set_tag(
            "promoted",
            str(promote).lower(),
        )

        if champion_version is not None:
            mlflow.set_tag(
                "champion_version_at_eval",
                str(champion_version),
            )

        if champion_metrics is not None:
            mlflow.set_tag(
                "champion_accuracy_at_eval",
                f"{champion_metrics['accuracy']:.4f}",
            )

        mlflow.set_tag(
            "challenger_version",
            str(new_version),
        )

        # ---------------------------------------------------------------
        # Promote if challenger wins
        # ---------------------------------------------------------------

        if promote:
            mlflow_client.set_registered_model_alias(
                MODEL_NAME,
                CHAMPION_ALIAS,
                new_version,
            )

            print(
                f"Promoted challenger v{new_version} "
                f"to '{CHAMPION_ALIAS}'."
            )

        else:
            print(
                f"Challenger v{new_version} "
                f"({challenger_metrics['accuracy']:.4f}) "
                f"did not beat champion "
                f"({champion_metrics['accuracy']:.4f}) — "
                f"keeping current champion, not promoting."
            )

    # -----------------------------------------------------------------------
    # Only mark batches processed after the MLflow operation succeeds.
    # -----------------------------------------------------------------------

    _mark_batches_processed(
        client,
        batch_ids,
    )

    print(
        f"Marked {len(batch_ids)} batch(es) as processed."
    )

    print("============================================================")
    print("Retraining pipeline completed successfully.")
    print("============================================================")


if __name__ == "__main__":
    main()

