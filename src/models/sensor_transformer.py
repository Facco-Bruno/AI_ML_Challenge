"""Sensor‑Transformer with focal loss and configurable depth."""
import math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

__all__ = ["SensorTransformer", "train_sensor_transformer"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2)*-(math.log(10000.0)/d))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self,x):
        return x + self.pe[:,:x.size(1)]

class SensorTransformer(nn.Module):
    def __init__(self, C=63, d_model=256, layers=4, nhead=8, classes=3):
        super().__init__()
        self.proj = nn.Linear(C, d_model)
        self.cls  = nn.Parameter(torch.zeros(1,1,d_model))
        self.pos  = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc_layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, classes))
    def forward(self,x):
        B=x.size(0)
        x=self.proj(x); x=torch.cat([self.cls.repeat(B,1,1),x],1)
        x=self.pos(x); x=self.enc(x)
        return self.head(x[:,0])

# Focal loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__(); self.alpha=alpha; self.gamma=gamma
    def forward(self, logits, y):
        ce = nn.functional.cross_entropy(logits, y, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

def train_sensor_transformer(train_ds, val_ds, epochs=60, bs=64, lr=5e-4, patience=10):
    net=SensorTransformer().to(device)
    crit=FocalLoss(alpha=torch.tensor([0.3,0.3,0.4],device=device))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr=DataLoader(train_ds,bs,shuffle=True); va=DataLoader(val_ds,bs)
    best,best_f1,wait=None,0,0
    for ep in range(1,epochs+1):
        net.train()
        for X,y in tqdm(tr,desc=f"Ep{ep}"):
            opt.zero_grad(); loss=crit(net(X.to(device)), y.to(device)); loss.backward(); opt.step()
        sch.step()
        # val
        net.eval(); preds,gts=[],[]
        with torch.no_grad():
            for X,y in va:
                preds+=net(X.to(device)).argmax(1).cpu().tolist(); gts+=y.tolist()
        f1=f1_score(gts,preds,average="macro")
        print(f"Epoch{ep}: F1={f1:.3f}")
        if f1>best_f1: best,best_f1,wait=net.state_dict(),f1,0
        else: wait+=1
        if wait>=patience:
            print("Early stop"); break
    net.load_state_dict(best)
    return net