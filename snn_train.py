from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# <-- Import your saved SNN model here (no change to data loading)
from SNN_Unet import SpikingUNetSmall

import multiprocessing

import pickle

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

if __name__ == '__main__':

    torch.set_num_threads(multiprocessing.cpu_count())  # e.g., 16
    torch.set_num_interop_threads(4)
    torch.set_float32_matmul_precision('medium')

    with open("train_val_split.pkl", "rb") as f:
        data = pickle.load(f)
    train_samples, val_samples = data["train"], data["val"]

    print(len(train_samples), len(val_samples))
    tilesize = 128  #512
    train_ds = Cloud95Dataset(
        train_samples, 
        tilesize=tilesize, 
        augment=True
    )
    # NOTE: SNNs run for multiple timesteps, so batch size is often reduced to save memory.
    train_dl = DataLoader(
        train_ds,
        batch_size=16,            # reduced from 16 to account for time dimension; change back if you prefer
        shuffle=True,
        num_workers=0,           # <-- keep as you had
        pin_memory=False,        # pinning helps GPU, can slow CPU-only
        persistent_workers=False # <-- keep as you had
    )
    val_ds = Cloud95Dataset(
        val_samples, 
        tilesize=tilesize, 
        augment=False
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=16,            # match train batch size
        shuffle=False,
        num_workers=0,           # <-- keep as you had
        pin_memory=False,        # pinning helps GPU, can slow CPU-only
        persistent_workers=False # <-- keep as you had
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Replace ANN with your SNN model (class comes from SNN_Unet.py)
    TIMESTEPS = 6   # how many timesteps to simulate; increase for better accuracy (costs runtime)
    model = SpikingUNetSmall(in_ch=4, out_ch=1, timesteps=TIMESTEPS).to(device)

    bce, dice = nn.BCEWithLogitsLoss(), DiceLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    # ==== RESUME FROM CHECKPOINT ====
    start_epoch = 1
    ckpt_path = Path("models/model_snn_5.pt")

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        prev_epoch = ckpt.get("epoch", 0)
        start_epoch = prev_epoch + 1
        print(f"Resuming from epoch {prev_epoch}, starting at epoch {start_epoch}")

        # Optional: advance LR scheduler so LR is consistent
        for _ in range(prev_epoch):
            sched.step()
    # ================================
    from tqdm.auto import tqdm

    epochs = 300
    print('begin training (SNN)')

    for epoch in range(start_epoch, epochs+1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_dl, desc=f'Epoch {epoch}/{epochs} - train', leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            # Run SNN for TIMESTEPS and get time-averaged logits (SpikingUNetSmall.forward handles reset)
            logits = model(imgs, timesteps=TIMESTEPS, return_last=False, reset=True)

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

                # time-averaged logits from SNN
                logits = model(imgs, timesteps=TIMESTEPS, return_last=False, reset=True)
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
        ep_path = models_dir / f"model_snn_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'iou': avg_val_iou,
            'model_state': model.state_dict(),
            'optimizer_state': opt.state_dict(),
        }, ep_path)
        print(f'Saved epoch model: {ep_path}')
