"""Audio clip -> fixed-length feature vector.

Same extraction path is used offline (train_baseline.py, retrain.py) and
at inference time (inference-api), so the feature contract must stay
identical across both — changing it here requires retraining.
"""

import librosa
import numpy as np

SAMPLE_RATE = 22050
N_MFCC = 13

FEATURE_NAMES = (
    [f"mfcc_{i}_mean" for i in range(N_MFCC)]
    + [f"mfcc_{i}_std" for i in range(N_MFCC)]
    + ["spectral_centroid_mean", "spectral_centroid_std"]
    + ["rms_mean", "rms_std"]
    + ["zcr_mean", "zcr_std"]
)


def load_audio(path: str) -> tuple[np.ndarray, int]:
    return librosa.load(path, sr=SAMPLE_RATE, mono=True)


def extract_features(path: str) -> np.ndarray:
    y, sr = load_audio(path)
    return extract_features_from_signal(y, sr)


def extract_features_from_signal(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)

    return np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            [centroid.mean(), centroid.std()],
            [rms.mean(), rms.std()],
            [zcr.mean(), zcr.std()],
        ]
    )
