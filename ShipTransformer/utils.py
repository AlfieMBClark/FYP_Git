"""
utils.py
--------
Shared helper functions used across dataset, training, and prediction.
"""

import math
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Distance metric
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between two points on Earth in kilometres.

    Uses the Haversine formula which accounts for Earth's spherical shape.
    This is the standard metric for evaluating trajectory prediction accuracy.

    Inputs can be scalars or numpy arrays (vectorised).
    All angles in DEGREES (converted internally to radians).
    """
    R = 6371.0  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def haversine_tensor(pred, target):
    """
    Haversine distance for batched torch tensors.

    pred, target : (..., 2)  where last dim is [lat, lon] in degrees.
    Returns a tensor of the same leading shape containing distances in km.
    """
    R = 6371.0
    pred   = torch.deg2rad(pred)
    target = torch.deg2rad(target)

    dlat = target[..., 0] - pred[..., 0]
    dlon = target[..., 1] - pred[..., 1]

    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(pred[..., 0]) * torch.cos(target[..., 0]) * torch.sin(dlon / 2) ** 2
    )
    return 2 * R * torch.asin(torch.sqrt(a.clamp(0, 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

class MinMaxScaler:
    """
    Scales each feature column independently to the range [0, 1].

    fit()       — compute min and max from training data
    transform() — apply the scaling
    inverse_transform() — undo the scaling (needed to read predictions)

    We normalise inputs because neural networks train better when values
    are small and centred.  Without this, the raw lat/lon numbers (~50, ~10)
    and COG (~0–360) are on very different scales, making training unstable.
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, data: np.ndarray):
        """data : (N, n_features)"""
        self.min_ = data.min(axis=0)
        self.max_ = data.max(axis=0)
        # Avoid division by zero for constant features
        self.range_ = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min_) / self.range_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.range_ + self.min_

    def inverse_transform_tensor(self, t: torch.Tensor) -> np.ndarray:
        """Convenience: accepts a torch tensor, returns numpy array."""
        arr = t.detach().cpu().numpy()
        return self.inverse_transform(arr)

    def save(self, path: str):
        np.savez(path, min_=self.min_, max_=self.max_, range_=self.range_)

    @classmethod
    def load(cls, path: str) -> "MinMaxScaler":
        d = np.load(path)
        scaler = cls()
        scaler.min_   = d["min_"]
        scaler.max_   = d["max_"]
        scaler.range_ = d["range_"]
        return scaler


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
