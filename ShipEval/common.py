"""
common.py
---------
Shared helpers for the evaluation scripts.

Everything here is read-only with respect to the existing project: models are
loaded through ShipDashboard/model_registry.py (the same registry the dashboard
uses, so a model evaluated here is exactly the model a marker can click on in
the UI), and windows are cut with ShipTransformer/prepare_dataset.py's own
split_and_filter(), so the evaluation set is cleaned identically to the
training set.

The autoregressive decode below is the batched version of
predict.py::predict_autoregressive — no teacher forcing, the decoder is fed its
own output, which is how the model runs at inference. All four registry models
share the forward(src, dec_input) -> (mu, log_var) interface and all were
trained with predict_deltas, so one decode loop drives every one of them.
"""

import os
import sys

import numpy as np
import torch

# ── path bootstrap ────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ALFIE       = os.path.abspath(os.path.join(_HERE, ".."))
_TRANSFORMER = os.path.join(_ALFIE, "ShipTransformer")
_DASHBOARD   = os.path.join(_ALFIE, "ShipDashboard")
_DATAHANDLING = os.path.join(_ALFIE, "DataHandling")

for _p in (_TRANSFORMER, _DASHBOARD, _DATAHANDLING):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import cfg                      # noqa: E402  ShipTransformer/config.py
import model_registry                       # noqa: E402  ShipDashboard/model_registry.py

RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── shapes / features ─────────────────────────────────────────────────────────
NF         = cfg.n_features            # 14
N_DEC      = cfg.n_dec_features        # 5   LAT, LON, SOG, COG_SIN, COG_COS
N_DEC_IN   = cfg.n_dec_input_features  # 6   + DT
SEQ_ENC    = cfg.seq_len_enc           # 90
SEQ_DEC    = cfg.seq_len_dec           # 10
WINDOW_LEN = SEQ_ENC + SEQ_DEC         # 100

IDX_LAT, IDX_LON, IDX_SOG, IDX_CSIN, IDX_CCOS = 0, 1, 2, 3, 4
IDX_DT, IDX_TYPE = 5, 6
IDX_DLAT, IDX_DLON, IDX_DCOG, IDX_ROT = 7, 8, 9, 10
IDX_HSIN, IDX_HCOS, IDX_NAV = 11, 12, 13

GROUP_NAMES = [
    "Unknown", "Cargo", "Tanker", "Passenger",
    "Fishing", "Tug/Service", "Pleasure/Sail", "Other",
]

# ── normalisation (identical bounds to training) ──────────────────────────────
FEAT = cfg.feature_cols
_LO  = np.array([cfg.norm_bounds[f][0] for f in FEAT], dtype=np.float32)
_HI  = np.array([cfg.norm_bounds[f][1] for f in FEAT], dtype=np.float32)
_RNG = _HI - _LO

LAT_LO, LAT_HI = cfg.norm_bounds["LAT"]
LON_LO, LON_HI = cfg.norm_bounds["LON"]
LAT_RNG = LAT_HI - LAT_LO
LON_RNG = LON_HI - LON_LO

DLAT_LO, DLAT_HI = cfg.norm_bounds["dLAT"]
DLON_LO, DLON_HI = cfg.norm_bounds["dLON"]
DLAT_RNG = DLAT_HI - DLAT_LO
DLON_RNG = DLON_HI - DLON_LO

SOG_LO, SOG_HI = cfg.norm_bounds["SOG"]
SOG_RNG = SOG_HI - SOG_LO

PREDICT_DELTAS = cfg.predict_deltas


def normalise(arr: np.ndarray) -> np.ndarray:
    """Physical units -> [0, 1], full 14-feature array."""
    return np.clip((arr - _LO) / _RNG, 0.0, 1.0).astype(np.float32)


def denormalise(arr: np.ndarray) -> np.ndarray:
    """[0, 1] -> physical units, full 14-feature array."""
    return (np.clip(arr, 0.0, 1.0) * _RNG + _LO).astype(np.float32)


def denorm_latlon(norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(..., >=2) normalised -> (lat, lon) in degrees."""
    lat = np.clip(norm[..., IDX_LAT], 0.0, 1.0) * LAT_RNG + LAT_LO
    lon = np.clip(norm[..., IDX_LON], 0.0, 1.0) * LON_RNG + LON_LO
    return lat, lon


def group_of(window_norm: np.ndarray) -> int:
    """Ship-type group (0-7) stored in feature 6 of a normalised window."""
    return int(round(float(window_norm[0, IDX_TYPE]) * 7.0))


# ── geometry ──────────────────────────────────────────────────────────────────

EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance, degrees in / km out."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi   = p2 - p1
    dlmb   = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def cog_from_sincos(sin_v, cos_v):
    """COG in degrees [0, 360) from the sin/cos feature pair (physical units)."""
    return np.degrees(np.arctan2(sin_v, cos_v)) % 360.0


def wrap180(deg):
    """Wrap an angle difference into [-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


# ── models ────────────────────────────────────────────────────────────────────

def available_models(keys=None):
    """Registry entries whose checkpoint actually exists on disk.

    Returns a list of (key, meta) preserving registry order.
    """
    wanted = list(model_registry.MODELS) if keys in (None, "all") else list(keys)
    out = []
    for k in wanted:
        if k not in model_registry.MODELS:
            raise SystemExit(f"Unknown model '{k}'. "
                             f"Choices: {', '.join(model_registry.MODELS)}")
        meta = model_registry.MODELS[k]
        if not os.path.exists(meta["ckpt"]):
            print(f"  ! skipping '{k}' — checkpoint missing: {meta['ckpt']}")
            continue
        out.append((k, meta))
    if not out:
        raise SystemExit("No usable checkpoints found.")
    return out


def load_model(key: str, device: str):
    meta = model_registry.MODELS[key]
    return meta["loader"](meta["ckpt"], device)


# ── autoregressive decode ─────────────────────────────────────────────────────

@torch.no_grad()
def ar_predict(model, enc: np.ndarray, device: str, batch_size: int = 256,
               progress: bool = False):
    """Autoregressive rollout for SEQ_DEC steps over a batch of encoder windows.

    enc : (N, SEQ_ENC, NF) normalised float32
    Returns mu, sigma — each (N, SEQ_DEC, N_DEC) in normalised *absolute*
    position space, so they are directly comparable with the ground-truth
    window's first N_DEC features.

    In delta mode the head predicts dLAT/dLON per step: the mean is accumulated
    onto the running position, and the *variance* is summed (independent steps)
    then converted from delta-normalised units (4 deg range) back to
    absolute-normalised units (LAT 16 deg, LON 25 deg). Skipping that unit
    conversion silently shrinks sigma ~100x and no anomaly ever fires.
    """
    model.eval()
    N   = len(enc)
    mus, sigmas = [], []

    for s in range(0, N, batch_size):
        chunk = enc[s : s + batch_size]
        B     = len(chunk)
        src   = torch.from_numpy(np.ascontiguousarray(chunk)).to(device)

        last_dt     = src[:, -1:, N_DEC:N_DEC_IN]     # reuse last observed interval
        dec_input   = src[:, -1:, :N_DEC_IN]
        prev_motion = src[:, -1:, :N_DEC].float()

        var_lat = torch.zeros((B, 1, 1), device=device)
        var_lon = torch.zeros((B, 1, 1), device=device)
        mu_acc, sig_acc = [], []

        for _ in range(SEQ_DEC):
            mu_raw, log_var = model(src, dec_input)
            mu_last  = mu_raw[:, -1:, :].float()
            std_last = (log_var[:, -1:, :].float() * 0.5).exp()

            if PREDICT_DELTAS:
                prev_lat = prev_motion[:, :, 0:1] * LAT_RNG + LAT_LO
                prev_lon = prev_motion[:, :, 1:2] * LON_RNG + LON_LO
                dlat     = mu_last[:, :, 0:1] * DLAT_RNG + DLAT_LO
                dlon     = mu_last[:, :, 1:2] * DLON_RNG + DLON_LO
                abs_lat  = ((prev_lat + dlat - LAT_LO) / LAT_RNG).clamp(0.0, 1.0)
                abs_lon  = ((prev_lon + dlon - LON_LO) / LON_RNG).clamp(0.0, 1.0)
                nxt      = torch.cat([abs_lat, abs_lon, mu_last[:, :, 2:].clamp(0, 1)], -1)
            else:
                nxt = mu_last.clamp(0.0, 1.0)

            # keep (COG_SIN, COG_COS) on the unit circle
            sin_raw = nxt[:, :, 3] * 2.0 - 1.0
            cos_raw = nxt[:, :, 4] * 2.0 - 1.0
            mag     = (sin_raw.pow(2) + cos_raw.pow(2)).sqrt().clamp(min=1e-8)
            nxt = nxt.clone()
            nxt[:, :, 3] = (sin_raw / mag + 1.0) * 0.5
            nxt[:, :, 4] = (cos_raw / mag + 1.0) * 0.5

            prev_motion = nxt

            if PREDICT_DELTAS:
                var_lat = var_lat + (std_last[:, :, 0:1] * DLAT_RNG) ** 2
                var_lon = var_lon + (std_last[:, :, 1:2] * DLON_RNG) ** 2
                std_abs = torch.cat([
                    var_lat.sqrt() / LAT_RNG,
                    var_lon.sqrt() / LON_RNG,
                    std_last[:, :, 2:],
                ], dim=-1)
            else:
                std_abs = std_last

            mu_acc.append(nxt.cpu().numpy())
            sig_acc.append(std_abs.cpu().numpy())
            dec_input = torch.cat([dec_input, torch.cat([nxt, last_dt], -1)], dim=1)

        mus.append(np.concatenate(mu_acc, axis=1))
        sigmas.append(np.concatenate(sig_acc, axis=1))

        if progress and (s // batch_size) % 20 == 0:
            print(f"    {min(s + batch_size, N):,}/{N:,} windows", flush=True)

    return np.concatenate(mus, 0), np.concatenate(sigmas, 0)


def step_errors_km(actual_norm: np.ndarray, mu_norm: np.ndarray) -> np.ndarray:
    """Per-step great-circle error. Both (N, SEQ_DEC, >=2) normalised.
    Returns (N, SEQ_DEC) in km."""
    a_lat, a_lon = denorm_latlon(actual_norm)
    p_lat, p_lon = denorm_latlon(mu_norm)
    return haversine_km(a_lat, a_lon, p_lat, p_lon)


# ── plotting ──────────────────────────────────────────────────────────────────

def use_report_style():
    """Matplotlib defaults that read well printed in greyscale as well as colour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi":       130,
        "savefig.dpi":      200,
        "savefig.bbox":     "tight",
        "font.size":        10,
        "axes.grid":        True,
        "grid.alpha":       0.25,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "legend.frameon":   False,
    })
    return plt


# Colour-blind-safe, distinct in greyscale; one colour per model, stable across
# every figure so a reader can track a model between plots.
MODEL_COLOURS = {
    "gru":            "#4C72B0",
    "tcn":            "#DD8452",
    "transformer":    "#55A868",
    "transformer_6m": "#C44E52",
}


def colour_for(key: str) -> str:
    return MODEL_COLOURS.get(key, "#7F7F7F")
