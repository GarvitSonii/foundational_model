from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models import UNetSmall

import multiprocessing

import pickle


torch.set_num_threads(multiprocessing.cpu_count())  # e.g., 16
torch.set_num_interop_threads(4)
torch.set_float32_matmul_precision('medium')

with open("train_val_split.pkl", "rb") as f:
    data = pickle.load(f)
train_samples, val_samples = data["train"], data["val"]

print(len(train_samples), len(val_samples))


class Cloud95Dataset(Dataset):
    
    def __init__(self, items, tilesize=None, augment=False):
        self.items = items
        self.tilesize = tilesize
        self.augment = augment

    def __len__(self): 
        return len(self.items)

    def _read_band(self, p: Path):
        arr = np.array(Image.open(p), dtype=np.uint8).astype(np.float32)
        return arr / 255.0

    def __getitem__(self, i):
        rec = self.items[i]
        R = self._read_band(rec["r"])
        G = self._read_band(rec["g"])
        B = self._read_band(rec["b"])
        N = self._read_band(rec["n"])
        M = np.array(Image.open(rec["m"]).convert("L")).astype(np.uint8)
        M = (M > 0).astype(np.float32)

        # Optional random 512 crop (most are 512 already; keep robust)
        H, W = R.shape
        if self.tilesize and (H >= self.tilesize and W >= self.tilesize):
            s = self.tilesize
            # x = 0 if W == s else random.randint(0, W - s)
            # y = 0 if H == s else random.randint(0, H - s)
            # R, G, B, N, M = R[y:y+s, x:x+s], G[y:y+s, x:x+s], B[y:y+s, x:x+s], N[y:y+s, x:x+s], M[y:y+s, x:x+s]

            if self.augment:
                x = 0 if W == s else random.randint(0, W - s)
                y = 0 if H == s else random.randint(0, H - s)
            else:
                x = (W - s) // 2
                y = (H - s) // 2
                
            R, G, B, N, M = R[y:y+s, x:x+s], G[y:y+s, x:x+s], B[y:y+s, x:x+s], N[y:y+s, x:x+s], M[y:y+s, x:x+s]
        
        
        # Simple flips as light augmentation
        if self.augment and random.random() < 0.5:
            R, G, B, N, M = np.fliplr(R), np.fliplr(G), np.fliplr(B), np.fliplr(N), np.fliplr(M)
        if self.augment and random.random() < 0.5:
            R, G, B, N, M = np.flipud(R), np.flipud(G), np.flipud(B), np.flipud(N), np.flipud(M)

        img = np.stack([R,G,B,N], axis=0).astype(np.float32)        # (4,H,W)
        msk = M[None, ...].astype(np.float32)                        # (1,H,W)

        
        return torch.from_numpy(img), torch.from_numpy(msk)


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs*targets).sum(dim=(1,2,3))
        den = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + self.eps
        return (1 - (num + self.eps)/(den + self.eps)).mean()

def iou_binary(logits, targets, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p*targets).sum(dim=(1,2,3))
    union = (p + targets - p*targets).sum(dim=(1,2,3))
    return ((inter+1e-6)/(union+1e-6)).mean().item()


tilesize = 128 
train_ds = Cloud95Dataset(
    train_samples, 
    tilesize=tilesize, 
    augment=True
)
train_dl = DataLoader(
    train_ds,
    batch_size=8,            # smaller batch to start
    shuffle=True,
    num_workers=0,           # <-- key
    pin_memory=False,        # pinning helps GPU, can slow CPU-only
    persistent_workers=False # <-- key
)
val_ds = Cloud95Dataset(
    val_samples, 
    tilesize=tilesize, 
    augment=False
)
val_dl = DataLoader(
    val_ds,
    batch_size=8,            # smaller batch to start
    shuffle=False,
    num_workers=0,           # <-- key
    pin_memory=False,        # pinning helps GPU, can slow CPU-only
    persistent_workers=False # <-- key
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetSmall(in_ch=4, out_ch=1).to(device)

bce, dice = nn.BCEWithLogitsLoss(), DiceLoss()

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

from tqdm.auto import tqdm

epochs = 300
print('begin training')

for epoch in range(1, epochs+1):
    model.train()
    train_loss = 0.0
    n_batches = 0
    pbar = tqdm(train_dl, desc=f'Epoch {epoch}/{epochs} - train', leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = 0.5*bce(logits, masks) + 0.5*dice(logits, masks)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': f'{train_loss / n_batches:.4f}', 'lr': opt.param_groups[0]['lr']})

    sched.step()

    # validation
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    n_val = 0
    with torch.no_grad():
        pbarv = tqdm(val_dl, desc=f'Epoch {epoch}/{epochs} - val', leave=False)
        for imgs, masks in pbarv:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = 0.5*bce(logits, masks) + 0.5*dice(logits, masks)
            val_loss += loss.item()
            val_iou += iou_binary(logits, masks)
            n_val += 1
            pbarv.set_postfix({'val_loss': f'{val_loss / n_val:.4f}', 'val_iou': f'{val_iou / n_val:.4f}'})

    avg_train_loss = train_loss / max(1, n_batches)
    avg_val_loss = val_loss / max(1, n_val)
    avg_val_iou = val_iou / max(1, n_val)

    print(f'Epoch {epoch}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_iou={avg_val_iou:.4f}')

    # save best model
    # save model for this epoch
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    ep_path = models_dir / f"model_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'iou': avg_val_iou,
        'model_state': model.state_dict(),
        'optimizer_state': opt.state_dict(),
    }, ep_path)
    print(f'Saved epoch model: {ep_path}')

