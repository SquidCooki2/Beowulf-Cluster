"""ScienceQA multimodal dataset (image + multiple-choice question -> 5-way answer A..E).

Image-bearing samples from `derek-thomas/ScienceQA` are packed once into a single
`scienceqa_128.npz` inside `data_dir` (images as uint8 NHWC + left-padded SmolVLM
token IDs ending with "Answer:"), then served from memory.

Splits are a fixed 80/10/10 permutation (seed 0) so the HPC trainer and every
engine stage see identical train/val/test sets without exchanging any data.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DATASET_ID = "derek-thomas/ScienceQA"
CHOICE_LETTERS = ("A", "B", "C", "D", "E")
NUM_CHOICES = 5
MAX_TEXT_LEN = 60

MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)


class PackedScienceQA:
    """Indexable holder for (images_uint8, tokens_int32) supporting numpy fancy indexing."""

    def __init__(self, images: np.ndarray, tokens: np.ndarray):
        self.images = images
        self.tokens = tokens
        self.shape = (len(images), 4, images.shape[1], images.shape[2])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return PackedScienceQA(self.images[idx], self.tokens[idx])


def packed_path(data_dir, size=128):
    return os.path.join(data_dir, f"scienceqa_{size}.npz")


def _format_prompt(row) -> str:
    q = (row.get("question") or "").strip()
    hint = (row.get("hint") or "").strip()
    choices = row.get("choices") or []
    choice_str = " ".join(
        f"{CHOICE_LETTERS[i]}) {str(c).strip()}"
        for i, c in enumerate(choices[:NUM_CHOICES])
    )
    prefix = f"Context: {hint} " if hint else ""
    return f"{prefix}Question: {q} Options: {choice_str}"


def _encode_prompt(tokenizer, prompt_body: str, suffix_ids: list, pad_id: int) -> np.ndarray:
    budget = MAX_TEXT_LEN - len(suffix_ids)
    body_ids = tokenizer.encode(prompt_body, add_special_tokens=False)[:budget]
    ids = body_ids + suffix_ids
    # Left-pad so the final "Answer:" token always sits at position MAX_TEXT_LEN - 1
    if len(ids) < MAX_TEXT_LEN:
        ids = [pad_id] * (MAX_TEXT_LEN - len(ids)) + ids
    return np.asarray(ids[-MAX_TEXT_LEN:], dtype=np.int32)


def _pack(data_dir, size, packed):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Downloading and packing {DATASET_ID} (size={size}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    suffix_ids = tokenizer.encode(" Answer:", add_special_tokens=False)

    ds_dict = load_dataset(DATASET_ID)
    images, tokens, labels = [], [], []
    for split_name in ("train", "validation", "test"):
        if split_name not in ds_dict:
            continue
        for row in ds_dict[split_name]:
            img = row.get("image")
            ans = row.get("answer")
            if img is None or ans is None or not (0 <= int(ans) < NUM_CHOICES):
                continue
            rgb = img.convert("RGB").resize((size, size))
            images.append(np.asarray(rgb, dtype=np.uint8))
            tokens.append(_encode_prompt(tokenizer, _format_prompt(row), suffix_ids, pad_id))
            labels.append(int(ans))

    np.savez_compressed(
        packed,
        images=np.stack(images),
        tokens=np.stack(tokens),
        y=np.asarray(labels, dtype=np.int64),
    )
    print(f"Saved {len(labels)} multimodal ScienceQA samples to {packed}")


def load_arrays(data_dir, size=128, images=True):
    """Return (PackedScienceQA or None, y). Pass images=False to read only the labels."""
    os.makedirs(data_dir, exist_ok=True)
    packed = packed_path(data_dir, size)
    if not os.path.exists(packed):
        _pack(data_dir, size, packed)
    d = np.load(packed)
    if not images:
        return None, d["y"]
    return PackedScienceQA(d["images"], d["tokens"]), d["y"]


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


def to_tensor(batch: PackedScienceQA, augment=False, rng=None) -> torch.Tensor:
    """PackedScienceQA batch -> (B, 4, H, W) float32 tensor (RGB normalized + token IDs in ch 3)."""
    imgs = torch.from_numpy(np.ascontiguousarray(batch.images)).permute(0, 3, 1, 2).float() / 255.0
    if augment:
        rng = rng or np.random.default_rng()
        for i in range(imgs.shape[0]):
            if rng.random() < 0.5:
                imgs[i] = imgs[i].flip(2)
    imgs = (imgs - MEAN) / STD

    b, _, h, w = imgs.shape
    out = torch.zeros((b, 4, h, w), dtype=torch.float32)
    out[:, :3, :, :] = imgs
    toks = torch.from_numpy(np.ascontiguousarray(batch.tokens)).float()
    out[:, 3, 0, :toks.shape[1]] = toks
    return out


class ScienceQADataset(Dataset):
    def __init__(self, packed_x: PackedScienceQA, y: np.ndarray, train: bool):
        self.x, self.y, self.train = packed_x, y, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = torch.from_numpy(self.x.images[i]).permute(2, 0, 1).float() / 255.0
        if self.train and torch.rand(1) < 0.5:
            img = img.flip(2)
        img = (img - MEAN) / STD
        _, h, w = img.shape
        out = torch.zeros((4, h, w), dtype=torch.float32)
        out[:3, :, :] = img
        toks = torch.from_numpy(self.x.tokens[i]).float()
        out[3, 0, :len(toks)] = toks
        return out, int(self.y[i])


def load(data_dir, limit=None, size=128, seed=0):
    """Return (train, val, test) datasets with the fixed 80/10/10 split."""
    x, y = load_arrays(data_dir, size)
    tr, va, te = split_indices(len(y), seed, limit)
    mk = lambda ids, train: ScienceQADataset(x[ids], y[ids], train)
    return mk(tr, True), mk(va, False), mk(te, False)
