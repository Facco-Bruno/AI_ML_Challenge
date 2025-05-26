"""
Hand-crafted features v2
────────────────────────
  • per-channel stats
  • per-sensor magnitude stats + dominant f + band-power
  • jerk stats
  • wavelet energy (sternum |a|)
  • **NEW** 10 delta-correlation features (pairs that change the most ADL→Fall)
"""

from __future__ import annotations
import numpy as np, pandas as pd, pywt
from scipy.fft import rfft, rfftfreq

_STATS = [np.mean, np.std, np.max, np.min, np.median]

# -------- 10 channel pairs with |Δr| > 0.5 (from EDA step-9) ----------
DELTA_PAIRS = [
    ("waist_acceleration_x_m/s^2",  "r.thigh_acceleration_x_m/s^2"),
    ("waist_acceleration_y_m/s^2",  "r.thigh_acceleration_y_m/s^2"),
    ("waist_acceleration_z_m/s^2",  "r.thigh_acceleration_z_m/s^2"),
    ("l.thigh_acceleration_x_m/s^2","r.thigh_acceleration_x_m/s^2"),
    ("head_acceleration_x_m/s^2",   "waist_acceleration_x_m/s^2"),
    ("head_acceleration_y_m/s^2",   "waist_acceleration_y_m/s^2"),
    ("head_acceleration_z_m/s^2",   "waist_acceleration_z_m/s^2"),
    ("sternum_acceleration_x_m/s^2","waist_acceleration_x_m/s^2"),
    ("sternum_acceleration_y_m/s^2","waist_acceleration_y_m/s^2"),
    ("sternum_acceleration_z_m/s^2","waist_acceleration_z_m/s^2"),
]

# full list of 63 column names in correct order
COLNAMES = [
    "r.ankle_acceleration_x_m/s^2", "r.ankle_acceleration_y_m/s^2", "r.ankle_acceleration_z_m/s^2",
    "r.ankle_angular_velocity_x_rad/s", "r.ankle_angular_velocity_y_rad/s", "r.ankle_angular_velocity_z_rad/s",
    "r.ankle_magnetic_field_x_ut", "r.ankle_magnetic_field_y_ut", "r.ankle_magnetic_field_z_ut",
    "l.ankle_acceleration_x_m/s^2", "l.ankle_acceleration_y_m/s^2", "l.ankle_acceleration_z_m/s^2",
    "l.ankle_angular_velocity_x_rad/s", "l.ankle_angular_velocity_y_rad/s", "l.ankle_angular_velocity_z_rad/s",
    "l.ankle_magnetic_field_x_ut", "l.ankle_magnetic_field_y_ut", "l.ankle_magnetic_field_z_ut",
    "r.thigh_acceleration_x_m/s^2", "r.thigh_acceleration_y_m/s^2", "r.thigh_acceleration_z_m/s^2",
    "r.thigh_angular_velocity_x_rad/s", "r.thigh_angular_velocity_y_rad/s", "r.thigh_angular_velocity_z_rad/s",
    "r.thigh_magnetic_field_x_ut", "r.thigh_magnetic_field_y_ut", "r.thigh_magnetic_field_z_ut",
    "l.thigh_acceleration_x_m/s^2", "l.thigh_acceleration_y_m/s^2", "l.thigh_acceleration_z_m/s^2",
    "l.thigh_angular_velocity_x_rad/s", "l.thigh_angular_velocity_y_rad/s", "l.thigh_angular_velocity_z_rad/s",
    "l.thigh_magnetic_field_x_ut", "l.thigh_magnetic_field_y_ut", "l.thigh_magnetic_field_z_ut",
    "head_acceleration_x_m/s^2", "head_acceleration_y_m/s^2", "head_acceleration_z_m/s^2",
    "head_angular_velocity_x_rad/s", "head_angular_velocity_y_rad/s", "head_angular_velocity_z_rad/s",
    "head_magnetic_field_x_ut", "head_magnetic_field_y_ut", "head_magnetic_field_z_ut",
    "sternum_acceleration_x_m/s^2", "sternum_acceleration_y_m/s^2", "sternum_acceleration_z_m/s^2",
    "sternum_angular_velocity_x_rad/s", "sternum_angular_velocity_y_rad/s", "sternum_angular_velocity_z_rad/s",
    "sternum_magnetic_field_x_ut", "sternum_magnetic_field_y_ut", "sternum_magnetic_field_z_ut",
    "waist_acceleration_x_m/s^2", "waist_acceleration_y_m/s^2", "waist_acceleration_z_m/s^2",
    "waist_angular_velocity_x_rad/s", "waist_angular_velocity_y_rad/s", "waist_angular_velocity_z_rad/s",
    "waist_magnetic_field_x_ut", "waist_magnetic_field_y_ut", "waist_magnetic_field_z_ut",
]

# -------------------------------------------------------------------- #
def _magnitude(arr): return np.linalg.norm(arr, axis=1)

def _band_power(y, fs, low, high):
    f = rfftfreq(len(y), 1/fs)
    Y = np.abs(rfft(y))
    return Y[(f>=low)&(f<high)].mean()

def _dominant_freq(y, fs):
    f = rfftfreq(len(y), 1/fs)
    return f[np.argmax(np.abs(rfft(y)))]

def _delta_corr(win_arr):
    df = pd.DataFrame(win_arr, columns=COLNAMES)
    C  = df.corr()
    return np.array([C.loc[a, b] for a, b in DELTA_PAIRS])

# -------------------------------------------------------------------- #
def extract_feature_matrix(pairs, fs: int = 100):
    feats, labels = [], []
    for w, lbl in pairs:      # w shape (T,63)
        f = []
        # 1) per-channel stats
        for fn in _STATS:
            f.append(fn(w, axis=0))
        # 2) per-sensor magnitude + freq stats
        for s in range(7):
            acc  = w[:, s*9     : s*9+3]
            gyro = w[:, s*9+3   : s*9+6]
            mag  = w[:, s*9+6   : s*9+9]
            for arr in (acc, gyro, mag):
                m = _magnitude(arr)
                f.append([fn(m) for fn in _STATS])
                f.append([_dominant_freq(m, fs),
                          _band_power(m, fs,0,2),
                          _band_power(m, fs,2,5),
                          _band_power(m, fs,5,10)])
        # 3) jerk stats (first derivative accel)
        jerk = np.diff(w[:, :21], axis=0) * fs
        f.append([fn(jerk, axis=0) for fn in _STATS])
        # 4) wavelet energy (sternum |a|)
        stern = _magnitude(w[:, 5*9 : 5*9+3])
        f.append([np.sum(c**2) for c in pywt.wavedec(stern, 'db4', level=3)])
        # 5) delta-corr
        f.append(_delta_corr(w))
        feats.append(np.concatenate([np.ravel(x) for x in f]))
        labels.append(lbl)
    return np.vstack(feats).astype(np.float32), np.array(labels)
