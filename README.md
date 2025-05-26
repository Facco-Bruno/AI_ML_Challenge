# Fall‑Detection IMU Pipeline

End‑to‑end, reproducible codebase for detecting **falls, near‑falls, and
ADLs** from seven APDM‑Opal IMU sensors.  It includes
pre‑processing → feature extraction → model training → cross‑validation →
checkpoints + result logs.

---
## 1 · Repository Structure

```
fall‑detection/
├── data/               # original .xlsx (downloaded manually)
├── data_parquet/       # auto‑generated (xlsx → parquet)
├── data_resampled/     # cached 100 Hz re‑sampled dfs
├── model_results/      # *.json metrics per model/fold
├── checkpoints/        # .pt or .pkl weights per LOSO fold
├── notebooks/
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── features.py
│   ├── metrics.py
│   ├── train.py        # <-- main entry‑point
│   └── models/
│       ├── rf.py
│       ├── xgb_lgbm.py
│       ├── cnn_lstm.py
│       └── sensor_transformer.py
└── README.md           # (this file)
```

---
## 2 · Quick Start

```bash
# clone or unzip repo
cd fall-detection

# (Conda) recommended
conda env create -f environment.yml
conda activate falldet
```

### 2.1 Download raw data
```
mkdir data && cd data
# manually download → https://drive.google.com/drive/folders/1Rr5eI8btUAKqDjmDc2vxRyu0C0yRX1Xl
# put sub1 … sub8 folders here
```

### 2.2 Pre‑process

```bash
# converts every .xlsx once → data_parquet/
python -m src.data_utils --convert
```

### 2.3 Train & validate (LOSO)

| Model | Command |
|-------|---------|
| Random‑Forest | `python -m src.train --model rf --out rf.json` |
| XGBoost       | `python -m src.train --model xgb --out xgb.json` |
| LightGBM      | `python -m src.train --model lgbm --out lgbm.json` |
| CNN+BiLSTM    | `python -m src.train --model cnn --epochs 40 --out cnn.json` |
| Transformer   | `python -m src.train --model transformer --epochs 60 --out tf.json` |

*Metrics JSON* → `model_results/<name>.json`  
*Checkpoints*  → `checkpoints/<model>_subX.(pkl|pt)`

---
## 3 · Design Decisions (interview‑ready)

### 3.1 Data layer
* `.xlsx → Parquet (snappy)` for 10× faster I/O.  
* Re‑sample all 63 channels to **100 Hz**, cached in `data_resampled/`.
* Fixed windows **2 s** length, **0.5 s** stride (75 % overlap).

### 3.2 Feature Engineering (classical models)
* Per‑channel stats, SMA, peak‑count, zero‑cross.  
* Magnitude stats per sensor with dominant‑frequency & band‑power.  
* Jerk stats + wavelet energy + time‑to‑peak.

### 3.3 Model zoo
* **Random‑Forest:** 600 trees, `class_weight="balanced"`.  
* **XGBoost:** 600 trees, depth 6, early‑stop 50.  
* **LightGBM:** 1000 rounds, leaves 31, early‑stop 100.  
* **CNN + BiLSTM:** 3 conv blocks → BiLSTM 256×2, dropout 0.4.  
* **Sensor‑Transformer:** `d_model=256`, 4 layers, focal‑loss.

### 3.4 Validation strategy
* **LOSO** (Leave‑One‑Subject‑Out).  Guarantees generalisation to unseen users.
* Metrics: **macro‑F1** (headline) + recall‑Fall.

---

---
## 4 · Extending / Tuning

* Add more features in `src/features.py` (vector auto‑consumed).  
* Tune hyper‑params with `optuna` – see example `tune_xgb.py`.  
* Change window size via `src/config.py` (`WINDOW_SEC`, `STEP_SEC`).

---
## 5 · Inference Example

```python
import pandas as pd, torch, numpy as np
from pathlib import Path
from src.data_utils import create_windows
from src.models.sensor_transformer import SensorTransformer

# load one trial (already parquet)
df = pd.read_parquet('data_parquet/sub1/Falls/trial1.parquet')
win, _ = next(create_windows(df, label=0))  # first window

model = SensorTransformer()
model.load_state_dict(torch.load('checkpoints/transformer_sub2.pt'))
model.eval()
with torch.no_grad():
    pred = model(torch.from_numpy(win).unsqueeze(0)).argmax(1).item()
print(['ADL','NearFall','Fall'][pred])
```

---
