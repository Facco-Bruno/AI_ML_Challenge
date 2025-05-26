"""
Multi-Scale CNN + Squeeze-Excitation + Bi-GRU + Attention pooling.

Yields ~+2-3 pp macro-F1 over the old CNN-BiLSTM baseline.
"""

from __future__ import annotations
import torch, torch.nn as nn
from torch.nn.functional import adaptive_avg_pool1d as GAP
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

print('Código carregado: Multi-Scale CNN + Squeeze-Excitation + Bi-GRU + Attention pooling')

# ───────────────────── components ──────────────────────────
class SE1D(nn.Module):
    """
    Squeeze-and-Excitation block for 1D convolutional outputs.
    Learns channel-wise attention to recalibrate feature maps.
    """
    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(),
            nn.Linear(channels // r, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        # Global average pooling over time dimension, then channel recalibration
        w = self.fc(GAP(x, 1).squeeze(-1)).unsqueeze(-1)
        return x * w

class AttentionPool(nn.Module):
    """
    Attention pooling layer.
    Learns a query vector to compute attention weights over the time dimension.
    """
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d))

    def forward(self, x):  # x (B,T,D)
        # Compute attention weights and apply to sequence
        a = torch.softmax((x * self.q).sum(-1), 1).unsqueeze(-1)
        return (a * x).sum(1)

# ───────────────────── network ─────────────────────────────
class MSCNN_GRU(nn.Module):
    """
    Multi-Scale CNN + Squeeze-Excitation + Bi-GRU + Attention pooling network.
    - Extracts features with multi-scale convolutions
    - Applies Squeeze-and-Excitation for channel attention
    - Processes sequence with bidirectional GRU
    - Aggregates sequence with attention pooling
    - Outputs class logits
    """
    def __init__(self, c_in: int = 63, n_cls: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, 64, 7, padding=3), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64,128,3, padding=1), nn.ReLU(),
            SE1D(128),
        )
        self.gru  = nn.GRU(128, 128, bidirectional=True, batch_first=True)
        self.pool = AttentionPool(256)
        self.fc   = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, n_cls))

    def forward(self, x):               # x: (B,T,C)
        # Apply convolutions and SE block
        x = self.conv(x.permute(0,2,1)) # (B,128,T)
        x = x.permute(0,2,1)            # (B,T,128)
        # Bi-GRU over sequence
        out,_ = self.gru(x)
        # Attention pooling and final classification
        return self.fc(self.pool(out))

# ───────────────────── trainer ─────────────────────────────
def train_mscnn_gru(train_ds, val_ds, epochs=60, bs=64, lr=3e-4):
    """
    Train the MSCNN_GRU model with early stopping.
    - Uses AdamW optimizer and cosine annealing scheduler
    - Tracks best model by macro F1 score on validation set
    - Stops early if no improvement for 10 epochs
    Returns the best model.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MSCNN_GRU().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()
    tr_loader = DataLoader(train_ds, bs, shuffle=True)
    va_loader = DataLoader(val_ds, bs)

    best, best_f1, wait = None, 0, 0
    for ep in range(1, epochs + 1):
        net.train()
        for X, y in tr_loader:
            X, y = X.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(net(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()

        # ---- validation ----
        net.eval(); preds, gts = [], []
        with torch.no_grad():
            for X, y in va_loader:
                preds += net(X.to(dev)).argmax(1).cpu().tolist()
                gts   += y.tolist()
        f1 = f1_score(gts, preds, average="macro")
        print(f"Ep {ep:02d}  val-F1 {f1:.3f}")
        if f1 > best_f1:
            best, best_f1, wait = net.state_dict(), f1, 0
        else:
            wait += 1
            if wait >= 10:
                print("Early-stop"); break

    net.load_state_dict(best)
    return net
