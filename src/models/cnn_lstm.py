"""Enhanced CNN + BiLSTM with extra conv layer, pooling and early‑stop hook."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

__all__ = ["CNNBiLSTM", "WindowTensorDataset", "train_cnn_bilstm"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Código carregado: CNN + BiLSTM com camada convolucional extra, pooling e early‑stop hook')

class CNNBiLSTM(nn.Module):
    def __init__(self, in_channels: int = 63, n_classes: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.rnn = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, n_classes))
    def forward(self, x):  # x (B, T, C)
        x = self.conv(x.permute(0,2,1))            # (B, 256, T/2)
        x = x.permute(0,2,1)
        _, (h, _) = self.rnn(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.classifier(h)

# ───────────────────────────────────────────────────────────────
# Dataset with normalisation + augmentation
# ───────────────────────────────────────────────────────────────
class WindowTensorDataset(Dataset):
    def __init__(self, windows, labels, mean=None, std=None, augment=False):
        X = np.stack(windows).astype(np.float32)   # (N, T, C)
        self.mean = mean if mean is not None else X.mean(axis=(0,1), keepdims=True)
        self.std  = std  if std  is not None else X.std(axis=(0,1), keepdims=True) + 1e-5
        self.X = (X - self.mean) / self.std
        self.y = np.array(labels, dtype=np.int64)
        self.augment = augment
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            # jitter + scaling (only on ADL majority)
            if self.y[idx] == 0:
                if np.random.rand() < 0.5:
                    x += np.random.normal(0, 0.02, x.shape)
                if np.random.rand() < 0.5:
                    x *= np.random.uniform(0.9, 1.1)
        return torch.from_numpy(x), torch.tensor(self.y[idx])

# ───────────────────────────────────────────────────────────────
# Training loop with early stopping
# ───────────────────────────────────────────────────────────────
from sklearn.metrics import f1_score

def train_cnn_bilstm(train_ds, val_ds, epochs=60, batch_size=64, lr=1e-3, patience=10):
    net = CNNBiLSTM().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()
    tr_loader = DataLoader(train_ds, batch_size, shuffle=True)
    va_loader = DataLoader(val_ds,   batch_size)

    best_state, best_f1, wait = None, 0.0, 0
    for ep in range(1, epochs+1):
        net.train()
        for X,y in tqdm(tr_loader, desc=f"Ep{ep}"):
            X,y = X.to(device), y.to(device)
            opt.zero_grad(); crit(net(X), y).backward(); opt.step()
        # validation F1
        net.eval(); preds, gts = [], []
        with torch.no_grad():
            for X,y in va_loader:
                preds += net(X.to(device)).argmax(1).cpu().tolist()
                gts   += y.tolist()
        f1 = f1_score(gts, preds, average="macro")
        print(f"Epoch {ep}: val F1={f1:.3f}")
        if f1 > best_f1:
            best_state, best_f1, wait = net.state_dict(), f1, 0
        else:
            wait += 1
            if wait >= patience:
                print("Early‑stopping")
                break
    net.load_state_dict(best_state)
    return net