# in data_loader.py or a new file cached_dataset.py
from torch.utils.data import Dataset
import torch, glob
from pathlib import Path

class CachedCloud95Dataset(Dataset):
    def __init__(self, cache_dir, index_list=None, tilesize=None, augment=False):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(list(self.cache_dir.glob('*.pt')))
        if index_list is not None:
            # optionally filter by indices in train/val split
            self.files = [self.files[i] for i in index_list]
        self.tilesize = tilesize
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i])
        img = d['img'].float()    # (4,H,W)
        mask = d['mask'].float()  # (1,H,W)
        # optional crop/augment here (on tensors)
        return img, mask