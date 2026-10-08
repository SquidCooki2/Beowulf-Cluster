"""NIH malaria cell images (parasitized vs uninfected).

The ~27.5k PNGs are packed once into a single .npz inside `data_dir` (which must be
on scratch on the HPC, to avoid the inode quota), then served from memory.

Splits are a fixed 80/10/10 permutation (seed 0) so the HPC trainer and every
engine stage see identical train/val/test sets without exchanging any data.
"""
import os
import urllib.request
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset

URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def packed_path(data_dir, size=128):
    return os.path.join(data_dir, f"malaria_{size}.npz")


def _pack(data_dir, size, packed):
    from PIL import Image
    zpath = os.path.join(data_dir, "cell_images.zip")
    if not os.path.exists(zpath):
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, zpath)
    images, labels = [], []
    with zipfile.ZipFile(zpath) as z:  # read straight from the zip: no loose files
        for name in sorted(z.namelist()):
            if not name.lower().endswith(".png"):
                continue
            label = 0 if "Parasitized" in name else 1
            with z.open(name) as f:
                img = Image.open(f).convert("RGB").resize((size, size))
            images.append(np.asarray(img, dtype=np.uint8))
            labels.append(label)
    np.savez(packed, x=np.stack(images), y=np.array(labels, dtype=np.int64))


def load_arrays(data_dir, size=128, images=True):
    """Return (x_uint8 NHWC or None, y). Pass images=False to read only the labels."""
    os.makedirs(data_dir, exist_ok=True)
    packed = packed_path(data_dir, size)
    if not os.path.exists(packed):
        _pack(data_dir, size, packed)
    d = np.load(packed)
    return (d["x"] if images else None), d["y"]


def split_indices(n, seed=0, limit=None):
    """Fixed 80/10/10 (train, val, test) index arrays."""
    idx = np.random.default_rng(seed).permutation(n)
    if limit:
        idx = idx[:limit]
    m = len(idx)
    a, b = int(0.8 * m), int(0.9 * m)
    return idx[:a], idx[a:b], idx[b:]


def epoch_order(train_idx, epoch, seed=0):
    """Deterministic shuffle of the training indices for one epoch (same on every node)."""
    return train_idx[np.random.default_rng(seed + 1 + epoch).permutation(len(train_idx))]


def to_tensor(x_uint8, augment=False, rng=None):
    """uint8 NHWC batch -> normalized float NCHW tensor (optionally random flips)."""
    t = torch.from_numpy(np.ascontiguousarray(x_uint8)).permute(0, 3, 1, 2).float() / 255.0
    if augment:
        rng = rng or np.random.default_rng()
        for i in range(t.shape[0]):
            if rng.random() < 0.5:
                t[i] = t[i].flip(2)
            if rng.random() < 0.5:
                t[i] = t[i].flip(1)
    return (t - MEAN) / STD


class MalariaDataset(Dataset):
    def __init__(self, x, y, train):
        self.x, self.y, self.train = x, y, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = torch.from_numpy(self.x[i]).permute(2, 0, 1).float() / 255.0
        if self.train:
            if torch.rand(1) < 0.5:
                img = img.flip(2)
            if torch.rand(1) < 0.5:
                img = img.flip(1)
        return (img - MEAN) / STD, int(self.y[i])


def load(data_dir, limit=None, size=128, seed=0):
    """Return (train, val, test) datasets with the fixed 80/10/10 split."""
    x, y = load_arrays(data_dir, size)
    tr, va, te = split_indices(len(y), seed, limit)
    mk = lambda ids, train: MalariaDataset(x[ids], y[ids], train)
    return mk(tr, True), mk(va, False), mk(te, False)
