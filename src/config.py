"""Global configuration constants – tweak here only."""

from pathlib import Path

DATA_DIR = Path(r"C:\Users\Bruno\OneDrive\Área de Trabalho\AI_ML_Challenge\data")
PARQUET_DIR = Path(r"C:\Users\Bruno\OneDrive\Área de Trabalho\AI_ML_Challenge\data_parquet")
TARGET_HZ       = 100                   # re‑sampling rate
WINDOW_SEC      = 2.0
STEP_SEC        = 0.5
SEED            = 42
CHECKPOINT_DIR  = Path(r"C:\Users\Bruno\OneDrive\Área de Trabalho\AI_ML_Challenge\src\checkpoints")

# Label map – extend if near‑fall must be split
LABELS = {"ADLs": 0, "Near_Falls": 1, "Falls": 2}