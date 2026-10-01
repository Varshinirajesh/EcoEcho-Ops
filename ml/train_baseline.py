"""Phase 0: offline baseline classifier, logged to MLflow.

Expects data/audio_stream/<species_label>/*.wav (one subfolder per class,
matching how the Watkins "Best of" cuts are organized per species). Run
from the repo root: python ml/train_baseline.py
"""
import os
import glob
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from feature_extraction import extract_features, FEATURE_NAMES


# =========================================================
# Paths
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "audio_stream"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# Load Dataset
# =========================================================

def load_dataset():

    print(
        f"Loading clips from {DATA_DIR} ..."
    )

    X = []
    y = []

    species_folders = [
        folder
        for folder in glob.glob(
            os.path.join(DATA_DIR, "*")
        )
        if os.path.isdir(folder)
    ]

    if not species_folders:
        raise FileNotFoundError(
            f"No species subfolders found in {DATA_DIR}. "
            "Expected data/audio_stream/<species>/*.wav"
        )

    for species_folder in species_folders:

        species = os.path.basename(
            species_folder
        )

        wav_files = glob.glob(
            os.path.join(
                species_folder,
                "*.wav"
            )
        )

        for wav_file in wav_files:

            try:

                features = extract_features(
                    wav_file
                )

                if features is None:
                    continue

                X.append(features)
                y.append(species)

            except Exception as e:

                print(
                    f"Skipping {wav_file}: {e}"
                )

    if not X:
        raise RuntimeError(
            "No audio features were extracted."
        )

    X = np.array(X)
    y = np.array(y)

    print(
        f"Loaded {len(X)} clips across "
        f"{len(np.unique(y))} species."
    )

    return X, y


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # Load audio features
    # -----------------------------------------------------

    X, y = load_dataset()

    # -----------------------------------------------------
    # Verify feature count
    # -----------------------------------------------------

    if X.shape[1] != len(FEATURE_NAMES):

        raise ValueError(
            f"Feature count mismatch! "
            f"Extracted {X.shape[1]} features, "
            f"but FEATURE_NAMES contains "
            f"{len(FEATURE_NAMES)} names."
        )

    print(
        f"Feature vector size: {X.shape[1]}"
    )

    # -----------------------------------------------------
    # Check for invalid feature values
    # -----------------------------------------------------

    if not np.isfinite(X).all():

        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    # -----------------------------------------------------
    # Encode labels
    # -----------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(
        y
    )

    print(
        f"Number of species: "
        f"{len(label_encoder.classes_)}"
    )

    print(
        "Species:"
    )

    print(
        list(label_encoder.classes_)
    )

    # -----------------------------------------------------
    # Train / Test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Validation samples: {len(X_test)}"
    )

    # -----------------------------------------------------
    # MLflow
    # -----------------------------------------------------

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "sqlite:///mlflow.db"
        )
    )

    mlflow.set_experiment(
        "ecoecho-baseline"
    )

    # -----------------------------------------------------
    # Train model
    # -----------------------------------------------------

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

        print(
            "\nTraining Random Forest..."
        )

        model.fit(
            X_train,
            y_train
        )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        # -------------------------------------------------
        # Accuracy
        # -------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        print(
            f"\nValidation Accuracy: "
            f"{accuracy:.4f}"
        )

        # -------------------------------------------------
        # Classification report
        # -------------------------------------------------

        present_labels = np.unique(
            np.concatenate(
                [y_test, y_pred]
            )
        )

        present_target_names = (
            label_encoder.inverse_transform(
                present_labels
            )
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=present_labels,
            target_names=present_target_names,
            output_dict=True,
            zero_division=0
        )

        print(
            "\nClassification Report:"
        )

        print(
            classification_report(
                y_test,
                y_pred,
                labels=present_labels,
                target_names=present_target_names,
                zero_division=0
            )
        )

        # -------------------------------------------------
        # MLflow parameters
        # -------------------------------------------------

        mlflow.log_param(
            "n_estimators",
            200
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "feature_count",
            len(FEATURE_NAMES)
        )

        mlflow.log_param(
            "species_count",
            len(label_encoder.classes_)
        )

        # -------------------------------------------------
        # MLflow metrics
        # -------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        if "macro avg" in report:

            mlflow.log_metric(
                "macro_f1",
                report["macro avg"]["f1-score"]
            )

        if "weighted avg" in report:

            mlflow.log_metric(
                "weighted_f1",
                report["weighted avg"]["f1-score"]
            )

        # -------------------------------------------------
        # Log model to MLflow
        # -------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        # -------------------------------------------------
        # Save model
        # -------------------------------------------------

        model_path = os.path.join(
            MODEL_DIR,
            "baseline_model.joblib"
        )

        encoder_path = os.path.join(
            MODEL_DIR,
            "label_encoder.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        joblib.dump(
            label_encoder,
            encoder_path
        )

        # -------------------------------------------------
        # Artifact paths
        # -------------------------------------------------

        reference_path = os.path.join(
            MODEL_DIR,
            "reference_features.csv"
        )

        validation_path = os.path.join(
            MODEL_DIR,
            "validation_set.csv"
        )

        # -------------------------------------------------
        # Reference dataset
        #
        # IMPORTANT:
        #
        # The retraining pipeline requires the reference
        # dataset to contain the true species label.
        #
        # Therefore:
        #
        # FEATURE_NAMES + species
        #
        # are stored in reference_features.csv.
        # -------------------------------------------------

        reference_df = pd.DataFrame(
            X_train,
            columns=FEATURE_NAMES
        )

        reference_df["species"] = (
            label_encoder.inverse_transform(
                y_train
            )
        )

        # -------------------------------------------------
        # Validation dataset
        #
        # Contains:
        #
        # FEATURE_NAMES + species
        # -------------------------------------------------

        validation_df = pd.DataFrame(
            X_test,
            columns=FEATURE_NAMES
        )

        validation_df["species"] = (
            label_encoder.inverse_transform(
                y_test
            )
        )

        # -------------------------------------------------
        # Expected artifact schemas
        # -------------------------------------------------

        expected_reference_columns = (
            FEATURE_NAMES + ["species"]
        )

        expected_validation_columns = (
            FEATURE_NAMES + ["species"]
        )

        # -------------------------------------------------
        # Validate reference schema
        # -------------------------------------------------

        reference_columns = (
            reference_df.columns.tolist()
        )

        if reference_columns != expected_reference_columns:

            raise ValueError(
                "Reference feature schema mismatch!\n"
                f"Expected: {expected_reference_columns}\n"
                f"Actual:   {reference_columns}"
            )

        # -------------------------------------------------
        # Validate validation schema
        # -------------------------------------------------

        validation_columns = (
            validation_df.columns.tolist()
        )

        if validation_columns != expected_validation_columns:

            raise ValueError(
                "Validation dataset schema mismatch!\n"
                f"Expected: {expected_validation_columns}\n"
                f"Actual:   {validation_columns}"
            )

        # -------------------------------------------------
        # Validate reference labels
        # -------------------------------------------------

        if reference_df["species"].isna().any():

            raise ValueError(
                "Reference dataset contains missing "
                "species labels."
            )

        # -------------------------------------------------
        # Validate validation labels
        # -------------------------------------------------

        if validation_df["species"].isna().any():

            raise ValueError(
                "Validation dataset contains missing "
                "species labels."
            )

        # -------------------------------------------------
        # Validate row counts
        # -------------------------------------------------

        if len(reference_df) != len(X_train):

            raise ValueError(
                "Reference dataset row count mismatch!"
            )

        if len(validation_df) != len(X_test):

            raise ValueError(
                "Validation dataset row count mismatch!"
            )

        # -------------------------------------------------
        # Validate species names
        # -------------------------------------------------

        expected_species = set(
            label_encoder.classes_
        )

        reference_species = set(
            reference_df["species"].unique()
        )

        validation_species = set(
            validation_df["species"].unique()
        )

        if not reference_species.issubset(
            expected_species
        ):

            raise ValueError(
                "Reference dataset contains unknown species."
            )

        if not validation_species.issubset(
            expected_species
        ):

            raise ValueError(
                "Validation dataset contains unknown species."
            )

        # -------------------------------------------------
        # Save CSV artifacts
        # -------------------------------------------------

        reference_df.to_csv(
            reference_path,
            index=False
        )

        validation_df.to_csv(
            validation_path,
            index=False
        )

        # -------------------------------------------------
        # Verify saved files by reading them back
        # -------------------------------------------------

        saved_reference_df = pd.read_csv(
            reference_path
        )

        saved_validation_df = pd.read_csv(
            validation_path
        )

        if (
            saved_reference_df.columns.tolist()
            != expected_reference_columns
        ):

            raise ValueError(
                "Saved reference_features.csv schema "
                "does not match expected schema."
            )

        if (
            saved_validation_df.columns.tolist()
            != expected_validation_columns
        ):

            raise ValueError(
                "Saved validation_set.csv schema "
                "does not match expected schema."
            )

        if (
            len(saved_reference_df)
            != len(X_train)
        ):

            raise ValueError(
                "Saved reference_features.csv row count "
                "does not match training data."
            )

        if (
            len(saved_validation_df)
            != len(X_test)
        ):

            raise ValueError(
                "Saved validation_set.csv row count "
                "does not match validation data."
            )

        # -------------------------------------------------
        # Print saved files
        # -------------------------------------------------

        print(
            "\nModel saved:"
        )

        print(
            model_path
        )

        print(
            "\nLabel encoder saved:"
        )

        print(
            encoder_path
        )

        print(
            "\nReference features saved:"
        )

        print(
            reference_path
        )

        print(
            "\nValidation set saved:"
        )

        print(
            validation_path
        )

        print(
            "\nReference feature columns:"
        )

        print(
            saved_reference_df.columns.tolist()
        )

        print(
            "\nValidation columns:"
        )

        print(
            saved_validation_df.columns.tolist()
        )

        print(
            "\nReference species count:"
        )

        print(
            saved_reference_df["species"].nunique()
        )

        print(
            "\nValidation species count:"
        )

        print(
            saved_validation_df["species"].nunique()
        )

        print(
            "\nReference rows:"
        )

        print(
            len(saved_reference_df)
        )

        print(
            "\nValidation rows:"
        )

        print(
            len(saved_validation_df)
        )

        print(
            "\nValidation Accuracy:"
        )

        print(
            f"{accuracy:.4f}"
        )

        print(
            "\nTraining completed successfully!"
        )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()