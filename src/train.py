# === file: src/train.py ===
"""
LOSO cross-validation trainer for:
    Random-Forest · XGBoost · LightGBM · CNN · Transformer

Key points
──────────
• Classical models use hand-crafted features.
• Deep models use raw 200×63 tensors with z-score + augmentation.
• Metrics of each fold are written to  JSON files under ./model_results/.
• Checkpoints:  .pkl for scikit/XGB/LGBM  ·  .pt for PyTorch.
"""

from __future__ import annotations
import argparse, json, random, joblib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from .config import SEED, CHECKPOINT_DIR
from .data_utils import window_generator
from .features import extract_feature_matrix
from .metrics import compute_metrics
from .models import (
    RandomForestFallDetector,
    WindowTensorDataset,
    train_cnn_bilstm,
    train_sensor_transformer,
    train_xgboost,
    train_lightgbm,
)

# -------------------- reproducibility --------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------- paths ------------------------------
MODEL_RESULTS_DIR = Path("model_results")
MODEL_RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# -------------------- helper functions ------------------
def _split_by_subject():
    groups = defaultdict(list)
    for subj, win, lbl in window_generator():
        groups[subj].append((win, lbl))
    return groups


def _save(model, path: Path):
    """Save .pt for PyTorch models, .pkl for scikit/XGB/LGBM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "state_dict"):
        torch.save(model.state_dict(), path.with_suffix(".pt"))
    else:
        joblib.dump(model, path.with_suffix(".pkl"))


def _build_ds(train_p, val_p, test_p):
    ws_tr, ys_tr = zip(*train_p)
    tmp = WindowTensorDataset(ws_tr, ys_tr)  # computes mean/std
    mean, std = tmp.mean, tmp.std
    tr = WindowTensorDataset(ws_tr, ys_tr, mean, std, augment=True)
    vl = WindowTensorDataset(*zip(*val_p), mean, std)
    te = WindowTensorDataset(*zip(*test_p), mean, std)
    return tr, vl, te


# -------------------- fold runner -----------------------
def _run_fold(name: str, tr_p, vl_p, te_p, epochs: int):
    """Return (model, y_true, y_pred)."""

    # --- classical -------------------------------------------------
    if name in ("rf", "xgb", "lgbm"):
        X_tr, y_tr = extract_feature_matrix(tr_p)
        X_vl, y_vl = extract_feature_matrix(vl_p)
        X_te, y_te = extract_feature_matrix(te_p)

        if name == "rf":
            model = RandomForestFallDetector().fit(X_tr, y_tr)
        elif name == "xgb":
            model = train_xgboost(X_tr, y_tr, X_vl, y_vl)
        else:
            model = train_lightgbm(X_tr, y_tr, X_vl, y_vl)

        y_pred = model.predict(X_te)
        if y_pred.ndim > 1:  # LightGBM returns (N, 3)
            y_pred = y_pred.argmax(1)
        return model, y_te, y_pred

    # --- deep ------------------------------------------------------
    tr_ds, vl_ds, te_ds = _build_ds(tr_p, vl_p, te_p)
    model = (
        train_cnn_bilstm(tr_ds, vl_ds, epochs)
        if name == "cnn"
        else train_sensor_transformer(tr_ds, vl_ds, epochs)
    )

    loader = torch.utils.data.DataLoader(te_ds, batch_size=128)
    model.eval()
    preds, gts = [], []
    dev = next(model.parameters()).device
    with torch.no_grad():
        for X, y in loader:
            preds += model(X.to(dev)).argmax(1).cpu().tolist()
            gts += y.tolist()
    return model, gts, preds


# ------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        choices=["rf", "xgb", "lgbm", "cnn", "transformer"],
        default="transformer",
    )
    ap.add_argument("--epochs", type=int, default=60,
                    help="epochs for CNN / Transformer")
    ap.add_argument("--out", default="results.json",
                    help="filename inside ./model_results/")
    ap.add_argument("--ckpt_dir", default=str(CHECKPOINT_DIR))
    args = ap.parse_args()

    groups = _split_by_subject()
    subjects = sorted(groups)
    results = {}

    for i, test_sub in enumerate(subjects):
        val_sub = subjects[(i + 1) % len(subjects)]
        train_subs = [s for s in subjects if s not in (test_sub, val_sub)]

        tr_p = [p for s in train_subs for p in groups[s]]
        vl_p = groups[val_sub]
        te_p = groups[test_sub]

        print(f"\nFold {i+1}/{len(subjects)}  test={test_sub}  val={val_sub}")
        mdl, y_true, y_pred = _run_fold(args.model, tr_p, vl_p, te_p, args.epochs)
        metrics = compute_metrics(y_true, y_pred)
        results[test_sub] = metrics
        print(metrics)

        _save(mdl, Path(args.ckpt_dir) / f"{args.model}_{test_sub}")

    out_path = MODEL_RESULTS_DIR / args.out
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
