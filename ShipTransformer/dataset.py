"""
dataset.py
----------
Thin Dataset wrapper over the pre-processed memory-mapped window files
produced by prepare_dataset.py.

Why memory-mapped files?
    Loading every sliding window into RAM at once (previous approach) fails for
    large datasets: a year of DMA data produces tens of millions of windows,
    each 40 × 5 float32 = 800 bytes → tens of GB.  Memory-mapped files let the
    OS page in only what each training batch needs, so RAM use stays flat
    regardless of dataset size.

Usage
-----
    Run prepare_dataset.py first to build the .bin files, then call load_data()
    here which returns ready-to-use DataLoaders.

    from dataset import load_data
    train_loader, val_loader, test_loader = load_data()
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
        # .copy() materialises the slice from disk into a regular numpy array
        # so torch.from_numpy doesn't hold a reference to the memmap.
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
