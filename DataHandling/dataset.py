"""
dataset.py
----------
Dataset wrapper over the memory-mapped window files produced by prepare_dataset.py.
Memory-mapping keeps RAM use flat: the OS pages in only the windows each batch
needs, instead of loading tens of millions of windows at once.

    from dataset import load_data
    train_loader, val_loader, test_loader = load_data()
"""

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# config.py lives in the sibling ShipTransformer/ folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ShipTransformer")))
from config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Memory-mapped Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MemmapWindowDataset(Dataset):
    """
    Dataset backed by a memory-mapped float32 binary file.

    The file contains N windows of shape (window_len, n_features).
    We load only the requested index from disk on each __getitem__ call,
    so memory use is O(batch_size) rather than O(N).
    """

    def __init__(self, bin_path: str, n_windows: int, window_len: int, n_features: int):
        self.data = np.memmap(
            bin_path, dtype="float32", mode="r",
            shape=(n_windows, window_len, n_features),
        )
        self.seq_enc = cfg.seq_len_enc

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # .copy() materialis into  numpy array
        # torch.from_numpy doesn't hold a ref to the memmap.
        window = self.data[idx].copy()
        src = torch.from_numpy(window[: self.seq_enc])
        tgt = torch.from_numpy(window[self.seq_enc :])
        return src, tgt


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Read metadata written by prepare_dataset.py, open the memory-mapped
    window files, and return train / val / test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    with open(cfg.meta_path) as f:
        meta = json.load(f)

    window_len = meta["window_len"]
    n_features = meta["n_features"]
    expected_window_len = cfg.seq_len_enc + cfg.seq_len_dec
    if window_len != expected_window_len:
        raise ValueError(
            f"dataset_meta.json window_len={window_len} but cfg expects "
            f"seq_len_enc + seq_len_dec = {expected_window_len}. "
            f"Regenerate the cached windows or restore matching config values."
        )

    splits = {
        "train": (cfg.train_windows, meta["n_train"]),
        "val":   (cfg.val_windows,   meta["n_val"]),
        "test":  (cfg.test_windows,  meta["n_test"]),
    }

    loaders = {}
    for split, (path, n_windows) in splits.items():
        if n_windows == 0:
            loaders[split] = None
            print(f"  {split}: 0 windows (skipped)")
            continue
        dataset = MemmapWindowDataset(path, n_windows, window_len, n_features)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size  = cfg.batch_size,
            shuffle     = shuffle,
            num_workers = cfg.num_workers,
            pin_memory  = cfg.device == "cuda",
            persistent_workers = cfg.num_workers > 0,
        )
        print(f"  {split}: {n_windows:,} windows")

    return loaders["train"], loaders["val"], loaders["test"]
