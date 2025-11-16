# preprocess_cache.py
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm   
# from data_loader import norm_stem  # or copy helper functions

OUT_DIR = Path('cached_dataset')
OUT_DIR.mkdir(exist_ok=True)
with open('train_val_split.pkl','rb') as f:
    data = pickle.load(f)
all_samples = data['train'] + data['val']

def read_img(p):
    im = Image.open(p)
    arr = np.array(im, dtype=np.uint8).astype(np.float32) / 255.0
    return arr

for idx, rec in enumerate(tqdm(all_samples, desc='caching')):
    # stack channels in order R,G,B,N => shape (4,H,W)
    r = read_img(rec['r'])
    g = read_img(rec['g'])
    b = read_img(rec['b'])
    n = read_img(rec['n'])
    img = np.stack([r,g,b,n], axis=0).astype(np.float32)
    # mask: single channel binary
    m = np.array(Image.open(rec['m']).convert('L'), dtype=np.uint8)
    m = (m>0).astype(np.uint8)[None,...]  # (1,H,W)
    torch.save({'img': torch.from_numpy(img), 'mask': torch.from_numpy(m)}, OUT_DIR / f'{idx:06d}.pt')