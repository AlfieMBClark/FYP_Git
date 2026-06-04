"""
train.py
--------
Training loop for the probabilistic Ship Trajectory Transformer.

Usage
-----
    python train.py

Requires the .bin window files produced by prepare_dataset.py.  Run that
script once (and again whenever config.py filter/normalisation settings change)
before calling train.py.

Key design choices
------------------
Pure Gaussian NLL loss
    Penalises both inaccurate means (large residuals) and miscalibrated
    variance simultaneously.  No auxiliary MSE or regularisation terms.

Per-feature loss weighting
    LAT and LON receive 5× the weight of SOG and COG features so the loss
    surface aligns with geographic accuracy.

DT in decoder input
    The encoder sees all 7 features [LAT, LON, SOG, COG_SIN, COG_COS, DT,
    SHIP_TYPE].  The decoder receives 6 features [LAT, LON, SOG, COG_SIN,
    COG_COS, DT] so it knows the time elapsed since the previous ping.
    DT is always taken from ground truth (never from the model's predictions)
    because the model predicts position/motion, not ping timing.
    SHIP_TYPE remains encoder-only static context.

Scheduled sampling
    After ss_start_epoch epochs of pure teacher forcing the decoder is
    increasingly fed its own predictions (up to ss_max_prob), bridging the
    train/inference gap.  DT is still ground-truth during scheduled sampling.
    COG_SIN/COG_COS are renormalised to the unit circle before feedback to
    prevent heading drift across autoregressive steps.

ADE / FDE validation and checkpointing
    Each validation epoch computes ADE (Average Displacement Error, km/step)
    and FDE (Final Displacement Error, km) via haversine distance.  The best
    checkpoint is saved by ADE so the saved model maximises geographic accuracy.

Mixed precision, gradient accumulation, OneCycleLR, optional torch.compile.
"""

import os
import torch
import torch.nn as nn
from config  import cfg
from model   import ShipTrajectoryTransformer
from dataset import load_data
from utils   import set_seed, haversine_tensor

N_ENC    = cfg.n_enc_features        # 14 — full feature set for encoder (adds dLAT/dLON/dCOG/ROT/HDG/NAV)
N_DEC    = cfg.n_dec_features        # 5  — decoder output: LAT, LON, SOG, COG_SIN, COG_COS
N_DEC_IN = cfg.n_dec_input_features  # 6  — decoder input: adds DT at index 5

# Normalisation constants for ADE/FDE computation (LAT / LON channels)
_LAT_LO,  _LAT_HI  = cfg.norm_bounds["LAT"]
_LON_LO,  _LON_HI  = cfg.norm_bounds["LON"]
_LAT_RNG = _LAT_HI - _LAT_LO
_LON_RNG = _LON_HI - _LON_LO


# ── Scheduled sampling ───────────────────────────────────────────────────────

@torch.no_grad()
def _build_scheduled_input(model, src, tgt_dec, tgt_dec_in, teacher_prob, use_amp):
    """Build decoder input mixing ground truth and model predictions.

    src        : (B, enc_len, N_ENC)    — full encoder features
    tgt_dec    : (B, dec_len, N_DEC)    — motion-only targets for GT mixing
    tgt_dec_in : (B, dec_len, N_DEC_IN) — full decoder input including DT

    DT (index 5 of tgt_dec_in) is always taken from ground truth — the model
    predicts position/motion but not when the next ping will arrive.
    """
    B, T, _ = tgt_dec.shape
    dec_input = src[:, -1:, :N_DEC_IN]   # seed: last encoder step (motion + DT)

    _diag_printed = False
    for step in range(T - 1):
        with torch.autocast(device_type=cfg.device, enabled=use_amp):
            mu, _ = model(src, dec_input)
        if not _diag_printed and torch.isnan(mu).any():
            print(f"  [NaN-diag] step={step} dec_input: min={dec_input.min():.4f} max={dec_input.max():.4f} has_nan={torch.isnan(dec_input).any().item()}")
            print(f"  [NaN-diag] src:       min={src.min():.4f} max={src.max():.4f} has_nan={torch.isnan(src).any().item()}")
            _mu_valid = mu[~torch.isnan(mu)]
            if _mu_valid.numel() > 0:
                print(f"  [NaN-diag] mu:        min={_mu_valid.min():.4f} max={_mu_valid.max():.4f} nan_count={torch.isnan(mu).sum().item()}")
            else:
                print(f"  [NaN-diag] mu:        ALL NaN — nan_count={torch.isnan(mu).sum().item()} shape={list(mu.shape)}")
            _diag_printed = True
        # Cast to fp32: clamp(min=1e-8) is a no-op in fp16 (underflows to 0),
        # which would make mag=0 and cause 0/0=NaN in the COG renorm below.
        pred = mu[:, -1:, :].nan_to_num(nan=0.5).clamp(0.0, 1.0).float()   # (B, 1, N_DEC)

        # COG_SIN (index 3) and COG_COS (index 4) are normalised from [-1,1]
        # to [0,1].  Denormalise, project to unit circle, re-normalise so the
        # autoregressive feedback is always a valid unit-circle heading.
        sin_raw = pred[:, :, 3] * 2.0 - 1.0
        cos_raw = pred[:, :, 4] * 2.0 - 1.0
        mag     = (sin_raw.pow(2) + cos_raw.pow(2)).sqrt().clamp(min=1e-8)
        pred[:, :, 3] = (sin_raw / mag + 1.0) * 0.5
        pred[:, :, 4] = (cos_raw / mag + 1.0) * 0.5

        gt_motion = tgt_dec[:, step:step + 1, :].float()                    # (B, 1, N_DEC)
        gt_dt     = tgt_dec_in[:, step:step + 1, N_DEC:N_DEC_IN].float()    # (B, 1, 1) — always GT
        mask      = (torch.rand(B, 1, 1, device=src.device) < teacher_prob).float()
        new_motion = mask * gt_motion + (1.0 - mask) * pred                 # (B, 1, N_DEC)
        new_token  = torch.cat([new_motion, gt_dt], dim=-1)                 # (B, 1, N_DEC_IN)
        dec_input  = torch.cat([dec_input, new_token], dim=1)

    return dec_input.detach()


# ── Loss ─────────────────────────────────────────────────────────────────────

def nll_loss(
    mu:      torch.Tensor,
    log_var: torch.Tensor,
    target:  torch.Tensor,
) -> torch.Tensor:
    """
    Pure weighted Gaussian NLL.

    NLL per feature = 0.5 * (log_var + (target - mu)^2 / exp(log_var))

    Each feature is multiplied by cfg.loss_feature_weights before averaging so
    LAT and LON errors dominate the loss surface, pushing the model to optimise
    geographic accuracy over less critical features like SOG or SHIP_TYPE.

    No MSE or variance-regularisation terms: the old combined loss had
    conflicting gradients whose relative scale depended on the current log_var
    state and could cause training instability.
    """
    w   = torch.tensor(cfg.loss_feature_weights, dtype=mu.dtype, device=mu.device)
    nll = 0.5 * (log_var + (target - mu).pow(2) / log_var.exp())
    return (nll * w).mean()


# ── ADE / FDE ─────────────────────────────────────────────────────────────────

def _compute_ade_fde(mu: torch.Tensor, tgt_dec: torch.Tensor) -> tuple[float, float, int]:
    """
    Compute haversine ADE and FDE (km) for one batch without gradients.

    Returns (sum_ade_km, sum_fde_km, batch_size) so the caller can accumulate
    across batches and divide by total samples at the end.
    """
    # Denormalise LAT and LON channels to degrees
    pred_lat = mu[:, :, 0].detach() * _LAT_RNG + _LAT_LO   # (B, T)
    pred_lon = mu[:, :, 1].detach() * _LON_RNG + _LON_LO
    true_lat = tgt_dec[:, :, 0]     * _LAT_RNG + _LAT_LO
    true_lon = tgt_dec[:, :, 1]     * _LON_RNG + _LON_LO

    pred_coords = torch.stack([pred_lat, pred_lon], dim=-1)   # (B, T, 2)
    true_coords = torch.stack([true_lat, true_lon], dim=-1)
    dists = haversine_tensor(pred_coords, true_coords)         # (B, T) km

    return (
        dists.mean(dim=1).sum().item(),   # sum of per-sample ADEs
        dists[:, -1].sum().item(),         # sum of per-sample FDEs
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

    return model


# ── Training / evaluation loop ────────────────────────────────────────────────

def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    scheduler,
    train: bool,
    teacher_prob: float = 1.0,
) -> tuple[float, float, float]:
    """
    One full pass over a DataLoader.

    Returns (mean_loss, mean_ade_km, mean_fde_km).
    ADE and FDE are 0.0 when train=True (not computed during training to avoid
    the overhead of denormalisation on every batch).
    """
    model.train(train)
    total_loss  = 0.0
    total_ade   = 0.0
    total_fde   = 0.0
    n_samples   = 0
    n_batches   = len(loader)
    print_every = max(1, n_batches // 10)

    use_amp = cfg.use_amp and cfg.device == "cuda"
    accum   = cfg.grad_accumulation

    if train:
        optimizer.zero_grad()

    with torch.set_grad_enabled(train):
        for batch_idx, (src, tgt) in enumerate(loader):
            src = src.to(cfg.device)   # (B, enc_len, N_ENC)
            tgt = tgt.to(cfg.device)   # (B, dec_len, N_ENC)

            # Decoder inputs and targets
            tgt_dec_in = tgt[:, :, :N_DEC_IN]   # (B, dec_len, N_DEC_IN) — motion + DT

            if torch.isnan(tgt_dec_in).any():
                nan_wins = torch.isnan(tgt_dec_in).any(dim=-1).any(dim=-1)
                print(f"  [NaN-data] batch={batch_idx} has {nan_wins.sum().item()} windows with NaN — replacing with 0.5")
            tgt_dec_in = tgt_dec_in.nan_to_num(0.5)
            tgt_dec    = tgt_dec_in[:, :, :N_DEC]   # (B, dec_len, N_DEC) — motion only, for loss

            # Build decoder input
            if train and teacher_prob < 1.0:
                tgt_input = _build_scheduled_input(model, src, tgt_dec, tgt_dec_in, teacher_prob, use_amp)
            else:
                last_past = src[:, -1:, :N_DEC_IN]   # last encoder step, motion + DT
                tgt_input = torch.cat([last_past, tgt_dec_in[:, :-1, :]], dim=1)

            # ── Forward pass ──────────────────────────────────────────────────
            with torch.autocast(device_type=cfg.device, enabled=use_amp):
                mu, log_var = model(src, tgt_input)
            loss = nll_loss(mu.float(), log_var.float(), tgt_dec.float()) / accum

            if train and torch.isnan(loss) and batch_idx < 5:
                print(f"  [NaN-diag] batch={batch_idx} tgt_input has_nan={torch.isnan(tgt_input).any().item()} mu has_nan={torch.isnan(mu).any().item()} log_var has_nan={torch.isnan(log_var).any().item()} loss={loss.item()}")

            # ── Backward ──────────────────────────────────────────────────────
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                is_last_in_batch = (
                    (batch_idx + 1) % accum == 0
                    or (batch_idx + 1) == n_batches
                )
                if is_last_in_batch:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                # ADE / FDE only during validation (no grad context)
                sum_ade, sum_fde, bs = _compute_ade_fde(mu, tgt_dec)
                total_ade += sum_ade
                total_fde += sum_fde
                n_samples  += bs

            total_loss += loss.item() * accum

            if train and (batch_idx + 1) % print_every == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"  batch {batch_idx + 1}/{n_batches}  loss={avg:.6f}", flush=True)

    mean_ade = total_ade / n_samples if n_samples > 0 else 0.0
    mean_fde = total_fde / n_samples if n_samples > 0 else 0.0
    return total_loss / n_batches, mean_ade, mean_fde


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    os.makedirs("checkpoints", exist_ok=True)

    train_loader, val_loader, _ = load_data()

    model     = build_model()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9
    )

    total_steps = (
        (len(train_loader) + cfg.grad_accumulation - 1) // cfg.grad_accumulation
    ) * cfg.epochs + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = cfg.lr,
        total_steps     = total_steps,
        pct_start       = 0.1,
        anneal_strategy = "cos",
    )

    use_amp    = cfg.use_amp and cfg.device == "cuda"
    amp_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")
    print(f"Device: {cfg.device}")
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")
    print(f"Effective batch size: {cfg.batch_size * cfg.grad_accumulation}")
    print(f"Encoder features: {N_ENC}  Decoder features: {N_DEC}\n")

    best_val_ade = float("inf")
    ramp_epochs  = max(1, cfg.epochs - cfg.ss_start_epoch)

    for epoch in range(1, cfg.epochs + 1):
        if epoch <= cfg.ss_start_epoch:
            teacher_prob = 1.0
        else:
            progress     = (epoch - cfg.ss_start_epoch) / ramp_epochs
            teacher_prob = 1.0 - cfg.ss_max_prob * min(1.0, progress)

        train_loss, _, _ = run_epoch(
            model, train_loader, optimizer, amp_scaler, scheduler,
            train=True, teacher_prob=teacher_prob,
        )
        val_loss, val_ade, val_fde = run_epoch(
            model, val_loader, optimizer, amp_scaler, scheduler,
            train=False,
        )

        if epoch % cfg.log_every == 0:
            print(
                f"Epoch {epoch:3d}/{cfg.epochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                f"val_ade={val_ade:.3f} km  val_fde={val_fde:.3f} km  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"teacher={teacher_prob:.2f}"
            )

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_loss":    val_loss,
                    "val_ade_km":  val_ade,
                    "val_fde_km":  val_fde,
                    "config": {
                        "n_features":          cfg.n_features,
                        "n_enc_features":      cfg.n_enc_features,
                        "n_dec_features":      cfg.n_dec_features,
                        "n_dec_input_features": cfg.n_dec_input_features,
                        "d_model":             cfg.d_model,
                        "num_heads":           cfg.num_heads,
                        "num_layers":          cfg.num_layers,
                        "d_ff":                cfg.d_ff,
                        "max_seq_length":      cfg.max_seq_length,
                        "dropout":             cfg.dropout,
                    },
                },
                cfg.checkpoint_path,
            )
            print(f"  ✓ saved best model (val_ade={val_ade:.3f} km/step)")

    print(f"\nTraining complete.  Best val ADE: {best_val_ade:.3f} km/step")


if __name__ == "__main__":
    main()
