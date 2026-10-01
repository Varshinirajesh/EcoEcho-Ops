"""Renders a mel-spectrogram PNG for the dashboard's live-prediction panel.

Only buoy-simulator uses this (the "what is the buoy hearing right now"
view) — kept in ml/ alongside feature_extraction.py and storm_noise.py
since it's audio-processing code that could be reused offline too.
"""

import io

import matplotlib

matplotlib.use("Agg")  # headless: no display available in a container

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def render_spectrogram_png(y: np.ndarray, sr: int) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_axis_off()
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="#0b0f19")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
