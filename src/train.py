# === file: src/train.py ===
"""
LOSO cross-validation trainer
─────────────────────────────
Model choices:
  rf | xgb | lgbm | cnn | msgru | transformer
Outputs:
  • JSON de métricas  →  ./model_results/<out>
  • Checkpoints       →  ./checkpoints/<model>_subX.(pkl|pt)
"""

from __future__ import annotations
import argparse, json, random, joblib
from pathlib import Path
from collections import defaultdict

import numpy as np, torch
from src.config        import SEED, CHECKPOINT_DIR
from src.data_utils    import window_generator
from src.features      import extract_feature_matrix
from src.metrics       import compute_metrics
from src.models import (
    RandomForestFallDetector,
    WindowTensorDataset,
    train_cnn_bilstm,
    train_mscnn_gru,
    train_sensor_transformer,     # <- Transformer trainer
    train_xgboost,
    train_lightgbm,
)

# reproducibilidade
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

RESULT_DIR = Path("model_results"); RESULT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ───────────────────────── helpers ─────────────────────────
def _split_by_subject():
    g = defaultdict(list)
    for subj, win, lbl in window_generator():
        g[subj].append((win, lbl))
    return g


def _save(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "state_dict"):          # PyTorch
        torch.save(model.state_dict(), path.with_suffix(".pt"))
    else:                                     # sklearn / xgb / lgbm
        joblib.dump(model, path.with_suffix(".pkl"))


def _build_ds(tr_p, vl_p, te_p):
    ws_tr, ys_tr = zip(*tr_p)
    tmp = WindowTensorDataset(ws_tr, ys_tr)   # calc mean/std
    m, s = tmp.mean, tmp.std
    tr = WindowTensorDataset(ws_tr, ys_tr, m, s, augment=True)
    vl = WindowTensorDataset(*zip(*vl_p), m, s)
    te = WindowTensorDataset(*zip(*te_p), m, s)
    return tr, vl, te


# ───────────────────── fold runner ────────────────────────
def _run_fold(model_name, tr_p, vl_p, te_p, epochs):
    # classical models -------------------------------------
    if model_name in ("rf", "xgb", "lgbm"):
        X_tr, y_tr = extract_feature_matrix(tr_p)
        X_vl, y_vl = extract_feature_matrix(vl_p)
        X_te, y_te = extract_feature_matrix(te_p)

        if model_name == "rf":
            model = RandomForestFallDetector().fit(X_tr, y_tr)
        elif model_name == "xgb":
            model = train_xgboost(X_tr, y_tr, X_vl, y_vl)
        else:
            model = train_lightgbm(X_tr, y_tr, X_vl, y_vl)

        y_pred = model.predict(X_te)
        if y_pred.ndim > 1:                    # LGBM proba → class idx
            y_pred = y_pred.argmax(1)
        return model, y_te, y_pred

    # deep models ------------------------------------------
    tr_ds, vl_ds, te_ds = _build_ds(tr_p, vl_p, te_p)
    if model_name == "cnn":
        model = train_cnn_bilstm(tr_ds, vl_ds, epochs)
    elif model_name == "msgru":
        model = train_mscnn_gru(tr_ds, vl_ds, epochs)
    else:                                       # transformer
        model = train_sensor_transformer(tr_ds, vl_ds, epochs)

    loader = torch.utils.data.DataLoader(te_ds, batch_size=128)
    model.eval(); preds, gts = [], []
    dev = next(model.parameters()).device
    with torch.no_grad():
        for X, y in loader:
            preds += model(X.to(dev)).argmax(1).cpu().tolist()
            gts   += y.tolist()
    return model, gts, preds


# ─────────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    choices=["rf","xgb","lgbm","cnn","msgru","transformer"],
                    default="transformer")
    ap.add_argument("--epochs", type=int, default=60,
                    help="epochs for CNN/MSGRU/Transformer")
    ap.add_argument("--out", default="results.json",
                    help="filename inside ./model_results/")
    ap.add_argument("--ckpt_dir", default=str(CHECKPOINT_DIR))
    args = ap.parse_args()

    groups = _split_by_subject(); res = {}
    subjects = sorted(groups)

    for i, test_sub in enumerate(subjects):
        val_sub   = subjects[(i+1) % len(subjects)]
        train_sub = [s for s in subjects if s not in (test_sub, val_sub)]

        tr_p = [p for s in train_sub for p in groups[s]]
        vl_p, te_p = groups[val_sub], groups[test_sub]

        print(f"Fold {i+1}/{len(subjects)}  test={test_sub}  val={val_sub}")
        mdl, y_true, y_pred = _run_fold(args.model, tr_p, vl_p, te_p, args.epochs)
        metrics = compute_metrics(y_true, y_pred)
        res[test_sub] = metrics
        print(metrics)

        _save(mdl, Path(args.ckpt_dir) / f"{args.model}_{test_sub}")

    out_path = RESULT_DIR / args.out
    out_path.write_text(json.dumps(res, indent=2))
    print(f"✓ Metrics saved to {out_path}")

if __name__ == "__main__":
    main()

