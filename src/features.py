"""
Baseline hand-crafted features:

  • per-channel stats: mean, std, max, min, median
  • magnitude stats for acc / gyro / mag
  • FFT band power + dominant frequency
  • jerk stats (d a/dt)
  • wavelet energy (db4, level-3) on sternum |a|
"""

from __future__ import annotations
import numpy as np
from scipy.fft import rfft, rfftfreq
import pywt

_STATS = [np.mean, np.std, np.max, np.min, np.median]


def _magnitude(arr: np.ndarray):
    return np.linalg.norm(arr, axis=1)


def _band_power(y, fs, low, high):
    f = rfftfreq(len(y), 1 / fs)
    Y = np.abs(rfft(y))
    return Y[(f >= low) & (f < high)].mean()


def _dominant_freq(y, fs):
    f = rfftfreq(len(y), 1 / fs)
    return f[np.argmax(np.abs(rfft(y)))]


def extract_feature_matrix(pairs, fs: int = 100):
    feats, labels = [], []
    for w, lbl in pairs:                   # w (T,63)
        f = []

        # 1) basic stats
        for fn in _STATS:
            f.append(fn(w, axis=0))

        # 2) magnitude features per sensor
        for sensor in range(7):
            acc = w[:, sensor * 9 : sensor * 9 + 3]
            gyro = w[:, sensor * 9 + 3 : sensor * 9 + 6]
            mag = w[:, sensor * 9 + 6 : sensor * 9 + 9]
            for arr in (acc, gyro, mag):
                m = _magnitude(arr)
                f.append([fn(m) for fn in _STATS])
                f.append(
                    [
                        _dominant_freq(m, fs),
                        _band_power(m, fs, 0, 2),
                        _band_power(m, fs, 2, 5),
                        _band_power(m, fs, 5, 10),
                    ]
                )

        # 3) jerk stats (derivada acel)
        jerk = np.diff(w[:, :21], axis=0) * fs
        f.append([fn(jerk, axis=0) for fn in _STATS])

        # 4) wavelet energy sternum |a|
        stern_mag = _magnitude(w[:, 5 * 9 : 5 * 9 + 3])
        coeffs = pywt.wavedec(stern_mag, "db4", level=3)
        f.append([np.sum(c**2) for c in coeffs])

        feats.append(np.concatenate([np.ravel(x) for x in f]))
        labels.append(lbl)

    return np.stack(feats).astype(np.float32), np.array(labels)
