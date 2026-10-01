"""Synthetic ocean storm noise injection.

Simulates the acoustic effect of a storm on a hydrophone recording: broadband
noise (wind/rain/wave slap at the surface) layered with a low-frequency
rumble (swell/wave action) on top of the clean signal. Used live by
buoy-simulator to degrade predictions and trigger drift, and will be reused
offline by the Phase 4 retrain job to generate labeled noisy training data —
kept as its own module so both callers apply the identical corruption.
"""

import numpy as np


def inject_storm_noise(
    y: np.ndarray,
    sr: int,
    intensity: float = 0.6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mix storm noise into a clean waveform.

    intensity is the noise RMS as a fraction of the signal RMS: 0 leaves the
    clip untouched, ~0.5-0.8 is a rough storm, 1.0+ is severe enough that the
    original call is barely recoverable.
    """
    if intensity <= 0:
        return y

    rng = rng or np.random.default_rng()
    dtype = y.dtype

    white = rng.normal(0.0, 1.0, size=y.shape)

    t = np.arange(len(y)) / sr
    rumble = np.zeros_like(y, dtype=np.float64)
    for freq in rng.uniform(20.0, 80.0, size=3):
        rumble += rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    rumble /= np.max(np.abs(rumble)) + 1e-9

    noise = 0.7 * white + 0.3 * rumble
    noise /= np.sqrt(np.mean(noise**2)) + 1e-9  # unit RMS

    signal_rms = np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-9
    noisy = y.astype(np.float64) + intensity * signal_rms * noise

    peak = np.max(np.abs(noisy))
    if peak > 1.0:
        noisy /= peak

    return noisy.astype(dtype)
