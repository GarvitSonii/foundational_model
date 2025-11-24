print('importing dependencies')
from pathlib import Path
import re, random, os, math
import numpy as np
from PIL import Image
import pickle

print('saving data path to root')
root = Path(r"C:\Users\garvi\.cache\kagglehub\datasets\sorour\38cloud-cloud-segmentation-in-satellite-images\versions\4\38-Cloud_training")

print('extracting channels')
RED   = root / "train_red"
GREEN = root / "train_green"
BLUE  = root / "train_blue"
NIR   = root / "train_nir"
GT    = root / "train_gt"

IMG_EXT = (".tif",".tiff",".png",".jpg",".jpeg")
MSK_EXT = (".tif",".tiff",".png")

assert RED.exists() and GREEN.exists() and BLUE.exists() and NIR.exists() and GT.exists(), "One or more band/GT folders missing."

print('making dictionary by removing name of channel')
_token = re.compile(r"(?:^|[_\-])(red|green|blue|nir|gt|mask|label)(?:$|[_\-])", re.I)

def norm_stem(p: Path) -> str:
    s = p.stem
    s = _token.sub("_", s)
    s = re.sub(r"[_\-]+", "_", s).strip("_-")
    return s.lower()

def list_files(folder, exts):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]

def index(files):
    d = {}
    for p in files:
        d.setdefault(norm_stem(p), []).append(p)
    return d

R = list_files(RED, IMG_EXT)
G = list_files(GREEN, IMG_EXT)
B = list_files(BLUE, IMG_EXT)
N = list_files(NIR, IMG_EXT)
M = list_files(GT,  MSK_EXT)

R_idx, G_idx, B_idx, N_idx, M_idx = map(index, (R,G,B,N,M))

print('taking common keys')
keys = set(R_idx) & set(G_idx) & set(B_idx) & set(N_idx) & set(M_idx)

keys = sorted(list(keys))

print(f"Found {len(keys)} paired chips.")

pairs = []
for k in keys:
    pairs.append({
        "r": R_idx[k][0], "g": G_idx[k][0], "b": B_idx[k][0], "n": N_idx[k][0], "m": M_idx[k][0]
    })

print('making train and val sets')
random.seed(42)
random.shuffle(pairs)
val_frac = 0.2
n_val = int(len(pairs)*val_frac)

val_samples = pairs[:n_val]
train_samples = pairs[n_val:]

print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

print('saving to train_val_split.pkl')
with open("train_val_split.pkl", "wb") as f:
    pickle.dump({"train": train_samples, "val": val_samples}, f)
    