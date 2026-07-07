"""
train.py
--------
Two-phase training for the probabilistic Ship Trajectory Transformer.

Phase 1 — TF Convergence (cfg.phase1_epochs)
    Pure teacher forcing throughout.  No scheduled sampling.  Optimises core
    predictive accuracy and saves the best checkpoint by TF ADE.

Phase 2 — AR Fine-tuning (cfg.phase2_epochs)
    Loads the Phase 1 best checkpoint.  teacher_prob=0 from epoch 1 — every
    decoder input token is the model's own previous prediction, exactly as at
    inference.  Uses a fresh CosineAnnealingLR starting at cfg.phase2_lr so the
    model has a stable, meaningful gradient signal throughout (unlike a single-
    phase run where OneCycleLR dies off just as SS gets most aggressive).
    Saves the best checkpoint by AR ADE.

Note: Phase 2 training is ~10x slower per epoch than Phase 1 because each batch
requires 9 extra no-grad forward passes to build the fully-AR decoder input.

All output is tee'd to a timestamped log file in logs/.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from config  import cfg
from model   import ShipTrajectoryTransformer
from dataset import load_data
from utils   import set_seed, haversine_tensor, load_water_mask

N_ENC    = cfg.n_enc_features
N_DEC    = cfg.n_dec_features
N_DEC_IN = cfg.n_dec_input_features

_LAT_LO, _LAT_HI = cfg.norm_bounds["LAT"]
_LON_LO, _LON_HI = cfg.norm_bounds["LON"]
_LAT_RNG = _LAT_HI - _LAT_LO
_LON_RNG = _LON_HI - _LON_LO

_DLAT_LO, _DLAT_HI = cfg.norm_bounds["dLAT"]
_DLON_LO, _DLON_HI = cfg.norm_bounds["dLON"]
_DLAT_RNG = _DLAT_HI - _DLAT_LO   # 4.0
_DLON_RNG = _DLON_HI - _DLON_LO   # 4.0


# ── Logging ───────────────────────────────────────────────────────────────────

class _Tee:
    """Write to both the original stdout and a log file simultaneously."""
    def __init__(self, stream, logfile):
        self._stream  = stream
        self._logfile = logfile

    def write(self, data):
        self._stream.write(data)
        self._logfile.write(data)

    def flush(self):
        self._stream.flush()
        self._logfile.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return self._stream.isatty()


# ── Land mask ─────────────────────────────────────────────────────────────────

_LAND_RASTER: torch.Tensor | None = None


def _init_land_raster(erosion_cells: int = 2) -> None:
    global _LAND_RASTER
    try:
        water_np, _, _ = load_water_mask(_LAT_LO, _LAT_HI, _LON_LO, _LON_HI)
    except RuntimeError as exc:
        print(f"  [land mask] {exc}\n  land penalty disabled.")
        return
    land_bool = ~water_np   # True = land
    if erosion_cells > 0:
        from scipy.ndimage import binary_erosion
        # Shrink land inward by erosion_cells (~1 km/cell at 0.01°).
        # Coastal raster cells that straddle the shoreline are reclassified as
        # water so predictions just offshore are not penalised.
        land_bool = binary_erosion(land_bool, iterations=erosion_cells)
    land_np      = land_bool.astype(np.float32)
    _LAND_RASTER = torch.from_numpy(land_np).to(cfg.device)
    print(f"  [land mask] Raster ready "
          f"({_LAND_RASTER.shape[0]}×{_LAND_RASTER.shape[1]} cells, 0.01° res, "
          f"OSM-aware, eroded {erosion_cells} cell)")


def _land_penalty(mu: torch.Tensor) -> torch.Tensor:
    if _LAND_RASTER is None:
        return mu.new_zeros(())
    B, T, _ = mu.shape
    gx     = mu[:, :, 1] * 2.0 - 1.0
    gy     = mu[:, :, 0] * 2.0 - 1.0
    grid   = torch.stack([gx, gy], dim=-1).view(B, 1, T, 2)
    raster = _LAND_RASTER.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

    # Nearest-neighbour gives a hard binary land/water mask with no gradient
    # bleeding into adjacent water cells.  Detach so the mask itself carries
    # no gradient — only its gating effect on land_p matters.
    on_land = F.grid_sample(
        raster, grid, mode="nearest", align_corners=True, padding_mode="border",
    ).view(B, T).detach()

    # Bilinear gives smooth gradients for backprop, but only on cells where
    # the nearest cell is actually land.  Water predictions — even right at
    # the coastline — get zero penalty and zero gradient.
    land_p = F.grid_sample(
        raster, grid, mode="bilinear", align_corners=True, padding_mode="border",
    ).view(B, T)

    return (on_land * land_p).mean()


# ── Scheduled sampling ────────────────────────────────────────────────────────

@torch.no_grad()
def _build_scheduled_input(model, src, tgt_dec, tgt_dec_in, teacher_prob, use_amp):
    """Build decoder input mixing ground truth and model predictions.

    At teacher_prob=0 (Phase 2) every token is a model prediction — exactly
    the AR regime seen at inference.  At teacher_prob=1 (Phase 1) every token
    is ground truth — pure TF.
    """
    B, T, _ = tgt_dec.shape
    dec_input = src[:, -1:, :N_DEC_IN]   # seed: last encoder step

    for step in range(T - 1):
        with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=use_amp):
            mu, _ = model(src, dec_input)
        mu_last = mu[:, -1:, :].nan_to_num(nan=0.5).float()

        if cfg.predict_deltas:
            # Convert predicted delta to absolute position for the next decoder input.
            # dec_input is always absolute — only the loss target changes to deltas.
            prev_lat = dec_input[:, -1:, 0].float() * _LAT_RNG + _LAT_LO  # (B, 1)
            prev_lon = dec_input[:, -1:, 1].float() * _LON_RNG + _LON_LO
            dlat     = mu_last[:, :, 0:1] * _DLAT_RNG + _DLAT_LO           # (B, 1, 1)
            dlon     = mu_last[:, :, 1:2] * _DLON_RNG + _DLON_LO
            next_lat = ((prev_lat.unsqueeze(-1) + dlat - _LAT_LO) / _LAT_RNG).clamp(0, 1)
            next_lon = ((prev_lon.unsqueeze(-1) + dlon - _LON_LO) / _LON_RNG).clamp(0, 1)
            pred     = torch.cat([
                next_lat,
                next_lon,
                mu_last[:, :, 2:N_DEC].clamp(0, 1),  # SOG, COG_SIN, COG_COS remain absolute
            ], dim=-1)
        else:
            pred = mu_last.clamp(0.0, 1.0)

        sin_raw = pred[:, :, 3] * 2.0 - 1.0
        cos_raw = pred[:, :, 4] * 2.0 - 1.0
        mag     = (sin_raw.pow(2) + cos_raw.pow(2)).sqrt().clamp(min=1e-8)
        pred[:, :, 3] = (sin_raw / mag + 1.0) * 0.5
        pred[:, :, 4] = (cos_raw / mag + 1.0) * 0.5

        gt_motion  = tgt_dec[:, step:step + 1, :].float()   # always absolute
        gt_dt      = tgt_dec_in[:, step:step + 1, N_DEC:N_DEC_IN].float()
        mask       = (torch.rand(B, 1, 1, device=src.device) < teacher_prob).float()
        new_motion = mask * gt_motion + (1.0 - mask) * pred
        new_token  = torch.cat([new_motion, gt_dt], dim=-1)
        dec_input  = torch.cat([dec_input, new_token], dim=1)

    return dec_input.detach()


def _ar_rollout_with_grad(
    model, src: torch.Tensor, tgt_dec_in: torch.Tensor, use_amp: bool,
    position_noise_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AR rollout with BPTT — gradients flow through the full 10-step prediction chain.

    Unlike _build_scheduled_input (no_grad + single final pass), every step is
    differentiable.  The backward pass propagates loss from step T back through
    steps T-1, T-2 … 1, teaching the model that an error at step k compounds
    into larger errors at steps k+1 … T.

    position_noise_std: if > 0, adds zero-mean Gaussian noise (detached) to the
    lat/lon decoder input at each step.  This exposes the model to realistic
    position drift during training, closing the gap between BPTT (where
    accumulated errors are small) and true inference (where they compound freely).
    Noise is detached — BPTT gradients still flow through delta accumulation.
    """
    B, T, _ = tgt_dec_in.shape
    dec_input = src[:, -1:, :N_DEC_IN].float()   # seed: last encoder step

    # Access the original (uncompiled) module so we can call _encode/_decode separately.
    # This lets us run the encoder ONCE and cache its output for all 10 decoder steps,
    # reducing activation memory from O(10 × encoder) to O(1 × encoder).
    orig = getattr(model, '_orig_mod', model)

    with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=use_amp):
        enc_output = orig._encode(src)   # single encoder pass, gradient retained

    all_mu: list[torch.Tensor] = []
    all_lv: list[torch.Tensor] = []

    for step in range(T):
        with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=use_amp):
            dec_out = orig._decode(dec_input, enc_output)
            mu  = orig.mu_proj(dec_out)
            lv  = orig.log_var_proj(dec_out).clamp(-4.0, 4.0)
        mu_f = mu.float()
        lv_f = lv.float()

        all_mu.append(mu_f[:, -1:, :])
        all_lv.append(lv_f[:, -1:, :])

        if step < T - 1:
            if cfg.predict_deltas:
                # Accumulate predicted delta onto the current absolute position.
                # dec_input is always absolute; next token must also be absolute.
                prev_lat = dec_input[:, -1:, 0] * _LAT_RNG + _LAT_LO  # (B, 1)
                prev_lon = dec_input[:, -1:, 1] * _LON_RNG + _LON_LO
                dlat     = mu_f[:, -1:, 0] * _DLAT_RNG + _DLAT_LO      # (B, 1)
                dlon     = mu_f[:, -1:, 1] * _DLON_RNG + _DLON_LO
                next_lat = ((prev_lat + dlat - _LAT_LO) / _LAT_RNG).clamp(0, 1).unsqueeze(-1)  # (B, 1, 1)
                next_lon = ((prev_lon + dlon - _LON_LO) / _LON_RNG).clamp(0, 1).unsqueeze(-1)
                sog      = mu_f[:, -1:, 2:3].clamp(0, 1)
                sin_raw  = mu_f[:, -1:, 3] * 2.0 - 1.0
                cos_raw  = mu_f[:, -1:, 4] * 2.0 - 1.0
                mag      = (sin_raw.pow(2) + cos_raw.pow(2)).sqrt().clamp(min=1e-8)
                # Use cat instead of in-place to keep the gradient graph intact
                next_motion = torch.cat([
                    next_lat,
                    next_lon,
                    sog,
                    ((sin_raw / mag + 1.0) * 0.5).unsqueeze(-1),
                    ((cos_raw / mag + 1.0) * 0.5).unsqueeze(-1),
                ], dim=-1)
            else:
                next_motion = mu_f[:, -1:, :N_DEC].clamp(0.0, 1.0)
                sin_raw = next_motion[:, :, 3] * 2.0 - 1.0
                cos_raw = next_motion[:, :, 4] * 2.0 - 1.0
                mag     = (sin_raw.pow(2) + cos_raw.pow(2)).sqrt().clamp(min=1e-8)
                next_motion = torch.cat([
                    next_motion[:, :, :3],
                    ((sin_raw / mag + 1.0) * 0.5).unsqueeze(-1),
                    ((cos_raw / mag + 1.0) * 0.5).unsqueeze(-1),
                ], dim=-1)

            # Inject position noise so the model trains against realistic drift.
            # Detached noise shifts where the model looks next without blocking
            # BPTT gradients from flowing back through the delta accumulation.
            if position_noise_std > 0.0:
                noise = (torch.randn(B, 1, 2, device=next_motion.device,
                                     dtype=next_motion.dtype) * position_noise_std).detach()
                next_motion = torch.cat([
                    (next_motion[:, :, 0:1] + noise[:, :, 0:1]).clamp(0.0, 1.0),
                    (next_motion[:, :, 1:2] + noise[:, :, 1:2]).clamp(0.0, 1.0),
                    next_motion[:, :, 2:],
                ], dim=-1)

            gt_dt      = tgt_dec_in[:, step:step + 1, N_DEC:N_DEC_IN].float()
            next_token = torch.cat([next_motion, gt_dt], dim=-1)
            dec_input  = torch.cat([dec_input, next_token], dim=1)

    # Return dec_input alongside mu/lv so the caller can use it for delta-mode
    # haversine: pred_lat_t = dec_input_lat_t + delta_pred_t
    return torch.cat(all_mu, dim=1), torch.cat(all_lv, dim=1), dec_input


@torch.no_grad()
def compute_ar_val_ade(model, loader, use_amp: bool, subsample: float = 1.0) -> float:
    """ADE from fully autoregressive decoding — matches true inference performance."""
    model.eval()
    total_ade   = 0.0
    n_samples   = 0
    max_batches = max(1, int(len(loader) * subsample))

    for i, (src, tgt) in enumerate(loader):
        if i >= max_batches:
            break
        src = src.to(cfg.device)
        tgt = tgt.to(cfg.device)
        tgt_dec_in = tgt[:, :, :N_DEC_IN].nan_to_num(0.5)
        tgt_dec    = tgt_dec_in[:, :, :N_DEC]

        B, T, D = tgt_dec_in.shape
        last_past = src[:, -1:, :D]
        dec_input = torch.cat([last_past, tgt_dec_in[:, :-1, :]], dim=1).clone()

        for t in range(T - 1):
            with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=use_amp):
                mu_t, _ = model(src, dec_input)

            if cfg.predict_deltas:
                # Accumulate delta to absolute for the next decoder input token
                delta_lat = mu_t[:, t, 0].float() * _DLAT_RNG + _DLAT_LO  # (B,)
                delta_lon = mu_t[:, t, 1].float() * _DLON_RNG + _DLON_LO
                prev_lat  = dec_input[:, t, 0].float() * _LAT_RNG + _LAT_LO
                prev_lon  = dec_input[:, t, 1].float() * _LON_RNG + _LON_LO
                next_lat  = ((prev_lat + delta_lat - _LAT_LO) / _LAT_RNG).clamp(0, 1)
                next_lon  = ((prev_lon + delta_lon - _LON_LO) / _LON_RNG).clamp(0, 1)
                pred      = torch.stack([next_lat, next_lon], dim=-1)
                pred      = torch.cat([pred, mu_t[:, t, 2:N_DEC].float().clamp(0, 1)], dim=-1)
            else:
                pred = mu_t[:, t, :].float().clamp(0.0, 1.0)

            sin_raw = pred[:, 3] * 2.0 - 1.0
            cos_raw = pred[:, 4] * 2.0 - 1.0
            cog_mag = (sin_raw.pow(2) + cos_raw.pow(2)).sqrt().clamp(min=1e-8)
            pred[:, 3] = (sin_raw / cog_mag + 1.0) * 0.5
            pred[:, 4] = (cos_raw / cog_mag + 1.0) * 0.5
            dec_input[:, t + 1, :N_DEC] = pred.to(dec_input.dtype)

        with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=use_amp):
            mu_ar, _ = model(src, dec_input)

        # In delta mode: pred_lat_t = dec_input_lat_t + delta_pred_t
        # dec_input[:, t, :] holds the accumulated absolute position at step t,
        # which is exactly the context used to predict delta_t.
        sum_ade, _, bs = _compute_ade_fde(
            mu_ar.float(), tgt_dec,
            dec_input=dec_input if cfg.predict_deltas else None,
        )
        total_ade += sum_ade
        n_samples  += bs

    return total_ade / n_samples if n_samples > 0 else 0.0


# ── Loss ──────────────────────────────────────────────────────────────────────

def nll_loss(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    w   = torch.tensor(cfg.loss_feature_weights, dtype=mu.dtype, device=mu.device)
    nll = 0.5 * (log_var + (target - mu).pow(2) / log_var.exp())
    return (nll * w).mean()


def _to_delta_targets(src: torch.Tensor, tgt_dec_abs: torch.Tensor) -> torch.Tensor:
    """Replace LAT/LON targets with normalised position deltas (dLAT, dLON).

    The decoder *input* always carries absolute positions so the model retains
    geographic context.  Only the loss target changes.  SOG/COG remain absolute.

    delta_0 = pos_0 - last_encoder_pos
    delta_t = pos_t - pos_{t-1}  for t >= 1

    Deltas are normalised using dLAT/dLON bounds (-2.0, 2.0) → [0, 1].
    Zero movement normalises to 0.5 (matches the mu_proj bias initialisation).
    """
    tgt_lat = tgt_dec_abs[:, :, 0] * _LAT_RNG + _LAT_LO   # (B, T) degrees
    tgt_lon = tgt_dec_abs[:, :, 1] * _LON_RNG + _LON_LO
    last_lat = src[:, -1, 0] * _LAT_RNG + _LAT_LO          # (B,)
    last_lon = src[:, -1, 1] * _LON_RNG + _LON_LO
    prev_lat = torch.cat([last_lat[:, None], tgt_lat[:, :-1]], dim=1)  # (B, T)
    prev_lon = torch.cat([last_lon[:, None], tgt_lon[:, :-1]], dim=1)
    dlat_norm = ((tgt_lat - prev_lat) - _DLAT_LO) / _DLAT_RNG
    dlon_norm = ((tgt_lon - prev_lon) - _DLON_LO) / _DLON_RNG
    out = tgt_dec_abs.clone()
    out[:, :, 0] = dlat_norm
    out[:, :, 1] = dlon_norm
    return out


def _mu_to_abs_latlon(
    mu: torch.Tensor,
    dec_input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert model output to physical (lat, lon) in degrees.

    Delta mode:   pred_lat_t = dec_input_lat_t_phys + delta_pred_t_phys
    Absolute mode: pred_lat_t = denorm(mu_lat_t)

    dec_input_lat_t is the absolute position the model *saw* at step t, so
    adding the predicted delta gives the absolute predicted position at step t.
    """
    if cfg.predict_deltas:
        prev_lat = dec_input[:, :, 0] * _LAT_RNG + _LAT_LO
        prev_lon = dec_input[:, :, 1] * _LON_RNG + _LON_LO
        pred_lat = prev_lat + mu[:, :, 0] * _DLAT_RNG + _DLAT_LO
        pred_lon = prev_lon + mu[:, :, 1] * _DLON_RNG + _DLON_LO
    else:
        pred_lat = mu[:, :, 0].clamp(0, 1) * _LAT_RNG + _LAT_LO
        pred_lon = mu[:, :, 1].clamp(0, 1) * _LON_RNG + _LON_LO
    return pred_lat, pred_lon


def _position_haversine_loss(mu: torch.Tensor, tgt_dec: torch.Tensor,
                              dec_input: torch.Tensor | None = None) -> torch.Tensor:
    """tgt_dec must always be absolute positions. dec_input required in delta mode."""
    if cfg.predict_deltas and dec_input is not None:
        pred_lat, pred_lon = _mu_to_abs_latlon(mu, dec_input)
    else:
        mu_c = mu.clamp(0.0, 1.0)
        pred_lat = mu_c[:, :, 0] * _LAT_RNG + _LAT_LO
        pred_lon = mu_c[:, :, 1] * _LON_RNG + _LON_LO
    true_lat = tgt_dec[:, :, 0] * _LAT_RNG + _LAT_LO
    true_lon = tgt_dec[:, :, 1] * _LON_RNG + _LON_LO
    pred_coords = torch.stack([pred_lat, pred_lon], dim=-1)
    true_coords = torch.stack([true_lat, true_lon], dim=-1)
    return haversine_tensor(pred_coords, true_coords).mean()


# ── ADE / FDE ─────────────────────────────────────────────────────────────────

def _compute_ade_fde(
    mu: torch.Tensor,
    tgt_dec: torch.Tensor,                  # always absolute positions
    dec_input: torch.Tensor | None = None,  # required when cfg.predict_deltas
) -> tuple[float, float, int]:
    if cfg.predict_deltas and dec_input is not None:
        pred_lat, pred_lon = _mu_to_abs_latlon(mu.detach(), dec_input)
    else:
        pred_lat = mu[:, :, 0].detach() * _LAT_RNG + _LAT_LO
        pred_lon = mu[:, :, 1].detach() * _LON_RNG + _LON_LO
    true_lat = tgt_dec[:, :, 0] * _LAT_RNG + _LAT_LO
    true_lon = tgt_dec[:, :, 1] * _LON_RNG + _LON_LO

    pred_coords = torch.stack([pred_lat, pred_lon], dim=-1)
    true_coords = torch.stack([true_lat, true_lon], dim=-1)
    dists = haversine_tensor(pred_coords, true_coords)

    return (
        dists.mean(dim=1).sum().item(),
        dists[:, -1].sum().item(),
        mu.size(0),
    )


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model() -> ShipTrajectoryTransformer:
    model = ShipTrajectoryTransformer(
        n_features           = cfg.n_features,
        d_model              = cfg.d_model,
        num_heads            = cfg.num_heads,
        num_layers           = cfg.num_layers,
        d_ff                 = cfg.d_ff,
        max_seq_length       = cfg.max_seq_length,
        dropout              = cfg.dropout,
        n_enc_features       = cfg.n_enc_features,
        n_dec_features       = cfg.n_dec_features,
        n_dec_input_features = cfg.n_dec_input_features,
    ).to(cfg.device)

    if cfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("  torch.compile: enabled")
        except Exception as e:
            print(f"  torch.compile: skipped ({e})")

    def _zero_nan_grad(grad: torch.Tensor) -> torch.Tensor:
        return grad.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(_zero_nan_grad)

    return model


# ── Training / evaluation loop ────────────────────────────────────────────────

def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    scheduler,      # stepped per optimizer step (OneCycleLR) or None (step externally)
    train: bool,
    teacher_prob: float = 1.0,
    use_land_penalty: bool = False,
    use_aux_haversine: bool = False,
    haversine_weight: float | None = None,  # None → use cfg.aux_haversine_weight
    use_rollout: bool = False,
) -> tuple[float, float, float]:
    model.train(train)
    total_loss = total_ade = total_fde = 0.0
    n_samples  = 0
    n_batches  = len(loader)
    print_every = max(1, n_batches // 10)

    use_amp = cfg.use_amp and cfg.device == "cuda"
    accum   = cfg.grad_accumulation

    if train:
        optimizer.zero_grad()

    with torch.set_grad_enabled(train):
        for batch_idx, (src, tgt) in enumerate(loader):
            src = src.to(cfg.device)
            tgt = tgt.to(cfg.device)

            tgt_dec_in = tgt[:, :, :N_DEC_IN]
            if torch.isnan(tgt_dec_in).any():
                nan_wins = torch.isnan(tgt_dec_in).any(dim=-1).any(dim=-1)
                print(f"  [NaN-data] batch={batch_idx} has {nan_wins.sum().item()} windows with NaN — replacing with 0.5")
            tgt_dec_in  = tgt_dec_in.nan_to_num(0.5)
            tgt_dec_abs = tgt_dec_in[:, :, :N_DEC]   # absolute positions — used for haversine/ADE
            # Loss target: delta-normalised LAT/LON when predict_deltas=True, absolute otherwise
            tgt_dec_loss = _to_delta_targets(src, tgt_dec_abs) if cfg.predict_deltas else tgt_dec_abs

            if train and use_rollout and teacher_prob < 1e-6:
                # BPTT rollout: gradients flow through the full AR prediction chain.
                # Returns dec_input (accumulated absolute positions) alongside mu/lv
                # so the haversine aux loss can be computed correctly in delta mode.
                mu_f, lv_f, dec_input_aux = _ar_rollout_with_grad(
                    model, src, tgt_dec_in, use_amp,
                    position_noise_std=cfg.phase2_position_noise_std,
                )
            else:
                if train and teacher_prob < 1.0:
                    # _build_scheduled_input always takes absolute tgt_dec
                    tgt_input = _build_scheduled_input(
                        model, src, tgt_dec_abs, tgt_dec_in, teacher_prob, use_amp
                    )
                else:
                    last_past = src[:, -1:, :N_DEC_IN]
                    tgt_input = torch.cat([last_past, tgt_dec_in[:, :-1, :]], dim=1)
                with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=use_amp):
                    mu, log_var = model(src, tgt_input)
                mu_f = mu.float()
                lv_f = log_var.float()
                dec_input_aux = tgt_input   # absolute positions for delta-mode aux losses

            if not torch.isfinite(mu_f).all():
                mu_f = mu_f.nan_to_num(nan=0.5, posinf=0.5, neginf=0.5)
            if not torch.isfinite(lv_f).all():
                lv_f = lv_f.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

            _hw  = haversine_weight if haversine_weight is not None else cfg.aux_haversine_weight
            _di  = dec_input_aux if cfg.predict_deltas else None   # delta-mode accumulation context
            loss = nll_loss(mu_f, lv_f, tgt_dec_loss.float())
            if use_aux_haversine and _hw > 0:
                loss = loss + _hw * _position_haversine_loss(mu_f, tgt_dec_abs.float(), dec_input=_di)
            loss = loss / accum

            if train and use_land_penalty and cfg.land_penalty_weight > 0:
                # In delta mode mu_f holds dLAT/dLON offsets, not absolute positions.
                # Use dec_input_aux (accumulated absolute LAT/LON) so the raster is
                # sampled at the actual predicted ship position, not delta-space.
                pen_pos = dec_input_aux[:, :, :N_DEC] if cfg.predict_deltas else mu_f
                loss = loss + cfg.land_penalty_weight * _land_penalty(pen_pos) / accum

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                is_last = (
                    (batch_idx + 1) % accum == 0
                    or (batch_idx + 1) == n_batches
                )
                if is_last:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    if torch.isfinite(grad_norm):
                        if scaler is not None:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                    else:
                        print(f"  [NaN-guard] batch {batch_idx}: "
                              f"non-finite grad_norm ({grad_norm:.2e}) — skipping update", flush=True)
                    if scaler is not None:
                        scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
            else:
                sum_ade, sum_fde, bs = _compute_ade_fde(mu_f, tgt_dec_abs, dec_input=_di)
                total_ade += sum_ade
                total_fde += sum_fde
                n_samples  += bs

            total_loss += loss.item() * accum

            if train and (batch_idx + 1) % print_every == 0:
                print(f"  batch {batch_idx + 1}/{n_batches}  loss={total_loss / (batch_idx + 1):.6f}", flush=True)

    mean_ade = total_ade / n_samples if n_samples > 0 else 0.0
    mean_fde = total_fde / n_samples if n_samples > 0 else 0.0
    return total_loss / n_batches, mean_ade, mean_fde


def _checkpoint_config() -> dict:
    return {
        "n_features":           cfg.n_features,
        "n_enc_features":       cfg.n_enc_features,
        "n_dec_features":       cfg.n_dec_features,
        "n_dec_input_features": cfg.n_dec_input_features,
        "d_model":              cfg.d_model,
        "num_heads":            cfg.num_heads,
        "num_layers":           cfg.num_layers,
        "d_ff":                 cfg.d_ff,
        "max_seq_length":       cfg.max_seq_length,
        "dropout":              cfg.dropout,
    }


# ── Phase 1: TF convergence ───────────────────────────────────────────────────

def run_phase1(model, train_loader, val_loader, use_amp, amp_scaler) -> None:
    print("\n" + "=" * 60)
    print("PHASE 1 — Teacher-Forced Convergence")
    print(f"  epochs={cfg.phase1_epochs}  lr={cfg.phase1_lr}  teacher=1.00 (fixed)")
    print("=" * 60 + "\n")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.phase1_lr, betas=(0.9, 0.98), eps=1e-9
    )
    total_steps = (
        (len(train_loader) + cfg.grad_accumulation - 1) // cfg.grad_accumulation
    ) * cfg.phase1_epochs + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = cfg.phase1_lr,
        total_steps     = total_steps,
        pct_start       = cfg.phase1_lr_pct_start,
        anneal_strategy = "cos",
    )

    best_val_ade_tf = float("inf")

    for epoch in range(1, cfg.phase1_epochs + 1):
        use_late_hav = (
            cfg.phase1_haversine_start_epoch > 0
            and epoch >= cfg.phase1_haversine_start_epoch
        )
        train_loss, _, _ = run_epoch(
            model, train_loader, optimizer, amp_scaler, scheduler,
            train=True, teacher_prob=1.0,
            use_aux_haversine=use_late_hav,
            haversine_weight=cfg.phase1_haversine_weight if use_late_hav else None,
        )
        val_loss, val_ade_tf, _ = run_epoch(
            model, val_loader, optimizer, amp_scaler, None,
            train=False,
        )

        run_ar = (epoch % cfg.phase1_ar_val_every == 0) or (epoch == cfg.phase1_epochs)
        val_ade_ar = (
            compute_ar_val_ade(model, val_loader, use_amp, cfg.phase1_ar_subsample)
            if run_ar else None
        )

        ar_str = f"{val_ade_ar:.3f} km(AR)" if val_ade_ar is not None else "-.--- km(AR)"
        print(
            f"[P1] Epoch {epoch:3d}/{cfg.phase1_epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"tf_ade={val_ade_tf:.3f} km  {ar_str}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        if val_ade_tf < best_val_ade_tf:
            best_val_ade_tf = val_ade_tf
            torch.save(
                {
                    "epoch":         epoch,
                    "model_state":   model.state_dict(),
                    "val_loss":      val_loss,
                    "val_ade_tf_km": val_ade_tf,
                    "phase":         1,
                    "config":        _checkpoint_config(),
                },
                cfg.phase1_checkpoint_path,
            )
            print(f"  ✓ [P1] saved best (tf_ade={val_ade_tf:.3f} km)")

    print(f"\nPhase 1 complete.  Best TF ADE: {best_val_ade_tf:.3f} km\n")


# ── Phase 2: AR fine-tuning ───────────────────────────────────────────────────

def run_phase2(model, train_loader, val_loader, use_amp, amp_scaler) -> None:
    rollout_str = (
        f"  BPTT rollout from epoch {cfg.phase2_teacher_anneal_epochs + 1} "
        f"(teacher_prob=0)"
        if cfg.phase2_use_rollout else "  scheduled-sampling only"
    )
    print("\n" + "=" * 60)
    print("PHASE 2 — Autoregressive Fine-tuning")
    print(f"  epochs={cfg.phase2_epochs}  lr={cfg.phase2_lr}  "
          f"teacher={cfg.phase2_teacher_start:.2f}→0.00 over "
          f"{cfg.phase2_teacher_anneal_epochs} epochs")
    print(rollout_str)
    print(f"  Loading: {cfg.phase1_checkpoint_path}")
    print("=" * 60 + "\n")

    ckpt = torch.load(cfg.phase1_checkpoint_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded P1 checkpoint — epoch={ckpt['epoch']}  tf_ade={ckpt['val_ade_tf_km']:.3f} km\n")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.phase2_lr, betas=(0.9, 0.98), eps=1e-9
    )
    # CosineAnnealingWarmRestarts resets to phase2_lr every T_0 epochs so the
    # BPTT rollout (active from epoch 11 onward) gets a fresh high-LR cycle
    # rather than training on near-zero gradient for most of Phase 2.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.phase2_warmrestart_t0, T_mult=1, eta_min=1e-6,
    )

    best_val_ade_ar = float("inf")

    for epoch in range(1, cfg.phase2_epochs + 1):
        anneal_progress = min(1.0, (epoch - 1) / cfg.phase2_teacher_anneal_epochs)
        teacher_prob    = cfg.phase2_teacher_start * (1.0 - anneal_progress)

        train_loss, _, _ = run_epoch(
            model, train_loader, optimizer, amp_scaler, None,
            train=True, teacher_prob=teacher_prob,
            use_land_penalty=True, use_aux_haversine=True,
            use_rollout=cfg.phase2_use_rollout,
        )
        val_loss, val_ade_tf, _ = run_epoch(
            model, val_loader, optimizer, amp_scaler, None,
            train=False,
        )

        run_ar = (epoch % cfg.phase2_ar_val_every == 0) or (epoch == cfg.phase2_epochs)
        val_ade_ar = (
            compute_ar_val_ade(model, val_loader, use_amp, cfg.ar_val_subsample)
            if run_ar else None
        )

        scheduler.step()

        ar_str = f"{val_ade_ar:.3f} km(AR)" if val_ade_ar is not None else "-.--- km(AR)"
        print(
            f"[P2] Epoch {epoch:3d}/{cfg.phase2_epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"tf_ade={val_ade_tf:.3f} km  {ar_str}  "
            f"teacher={teacher_prob:.2f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        if val_ade_ar is not None and val_ade_ar < best_val_ade_ar:
            best_val_ade_ar = val_ade_ar
            torch.save(
                {
                    "epoch":         epoch,
                    "model_state":   model.state_dict(),
                    "val_loss":      val_loss,
                    "val_ade_tf_km": val_ade_tf,
                    "val_ade_ar_km": val_ade_ar,
                    "phase":         2,
                    "config":        _checkpoint_config(),
                },
                cfg.checkpoint_path,
            )
            print(f"  ✓ [P2] saved best (ar_ade={val_ade_ar:.3f} km)")

    print(f"\nPhase 2 complete.  Best AR ADE: {best_val_ade_ar:.3f} km\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    _orig_stdout = sys.stdout
    log_path     = os.path.join(
        "logs", f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    _log_file  = open(log_path, "w")
    sys.stdout = _Tee(_orig_stdout, _log_file)
    print(f"Logging to {log_path}\n")

    train_loader, val_loader, _ = load_data()
    model = build_model()
    _init_land_raster()

    use_amp    = cfg.use_amp and cfg.device == "cuda"
    amp_scaler = torch.amp.GradScaler(
        "cuda", init_scale=2 ** 14, growth_interval=1_000_000,
    ) if use_amp else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")
    print(f"Device: {cfg.device}  AMP: {'enabled' if use_amp else 'disabled'}")
    print(f"Effective batch size: {cfg.batch_size * cfg.grad_accumulation}")
    print(f"Encoder features: {N_ENC}  Decoder features: {N_DEC}\n")

    if cfg.skip_phase1 and os.path.exists(cfg.phase1_checkpoint_path):
        print(f"Phase 1 skipped — reusing checkpoint: {cfg.phase1_checkpoint_path}\n")
    else:
        if cfg.skip_phase1:
            print(f"  [warning] skip_phase1=True but {cfg.phase1_checkpoint_path} "
                  f"not found — running Phase 1 anyway\n")
        run_phase1(model, train_loader, val_loader, use_amp, amp_scaler)

    run_phase2(model, train_loader, val_loader, use_amp, amp_scaler)

    skipped = cfg.skip_phase1 and os.path.exists(cfg.phase1_checkpoint_path)
    total   = (0 if skipped else cfg.phase1_epochs) + cfg.phase2_epochs
    print(f"Training complete ({total} epochs run).  Log saved to {log_path}")
    _log_file.close()
    sys.stdout = _orig_stdout


if __name__ == "__main__":
    main()
