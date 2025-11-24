from pathlib import Path
import random                       
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

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
