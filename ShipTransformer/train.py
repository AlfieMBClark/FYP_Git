"""
train.py
--------
Training loop for the probabilistic Ship Trajectory Transformer.

Efficiency improvements over the original version
--------------------------------------------------
Mixed precision (AMP)
    torch.autocast runs eligible ops in float16, roughly doubling GPU
    throughput and halving memory use.  GradScaler compensates for
    the reduced dynamic range when scaling gradients.

Gradient accumulation
    We step the optimiser every `grad_accumulation` mini-batches, giving an
    effective batch size of batch_size × grad_accumulation without extra
    GPU memory.

OneCycleLR scheduler
    Warms the learning rate up from lr/25 to lr over the first ~30 % of
    training, then cosine-anneals back to lr/1e4.  This typically reaches
    lower loss in fewer epochs than a fixed learning rate.

torch.compile (PyTorch ≥ 2.0)
    Traces the model into a fused kernel graph.  Gives 20–40 % throughput
    improvement with no changes to model code.

Loss: Gaussian NLL instead of MSE
    The model now outputs (mu, log_var) per timestep.  NLL is:
        0.5 × (log_var + (target - mu)² / exp(log_var))
    This trains the model to output calibrated uncertainty alongside its
    mean prediction, which is used directly for anomaly scoring.
"""

import os
import torch
import torch.nn as nn
from config import cfg
from model  import ShipTrajectoryTransformer
from dataset import load_data
from utils  import set_seed

SEQ_DEC = cfg.seq_len_dec


# ── Scheduled sampling ───────────────────────────────────────────────────────

@torch.no_grad()
def _build_scheduled_input(model, src, tgt, teacher_prob, use_amp):
    """Build decoder input mixing ground truth and model predictions.

    teacher_prob=1.0 → pure teacher forcing (same as normal training).
    teacher_prob=0.5 → each decoder step independently uses the model's own
                       prediction ~50% of the time, bridging the train/inference gap.

    Runs without gradients; the returned tensor is detached so that the
    subsequent forward pass (with grads) is the only differentiated call.
    """
    B, T, _ = tgt.shape
    dec_input = src[:, -1:, :]          # seed: last encoder observation
    for step in range(T - 1):
        with torch.autocast(device_type=cfg.device, enabled=use_amp):
            mu, _ = model(src, dec_input)
        pred = mu[:, -1:, :].clamp(0.0, 1.0)  # clamp before feeding back
        gt   = tgt[:, step:step+1, :]
        mask = (torch.rand(B, 1, 1, device=src.device) < teacher_prob).to(pred.dtype)
        dec_input = torch.cat([dec_input, mask * gt + (1 - mask) * pred], dim=1)
    return dec_input.detach()


# ── Loss ─────────────────────────────────────────────────────────────────────

def nll_loss(
    mu:      torch.Tensor,
    log_var: torch.Tensor,
    target:  torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood averaged over all elements.

    log_var is expected to already be clamped (done inside the model).
    """
    return 0.5 * (log_var + (target - mu).pow(2) / log_var.exp()).mean()


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model() -> ShipTrajectoryTransformer:
    model = ShipTrajectoryTransformer(
        n_features     = cfg.n_features,
        d_model        = cfg.d_model,
        num_heads      = cfg.num_heads,
        num_layers     = cfg.num_layers,
        d_ff           = cfg.d_ff,
        max_seq_length = cfg.max_seq_length,
        dropout        = cfg.dropout,
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
    scaler,         # torch.cuda.amp.GradScaler (or None on CPU)
    scheduler,
    train: bool,
    teacher_prob: float = 1.0,
) -> float:
    """
    One full pass over a DataLoader.

    Gradient accumulation: gradients are accumulated over `grad_accumulation`
    mini-batches before an optimiser step so that the effective batch size
    scales without extra GPU memory.
    """
    model.train(train)
    total_loss  = 0.0
    n_batches   = len(loader)
    print_every = max(1, n_batches // 10)

    use_amp = cfg.use_amp and cfg.device == "cuda"
    accum   = cfg.grad_accumulation

    if train:
        optimizer.zero_grad()

    with torch.set_grad_enabled(train):
        for batch_idx, (src, tgt) in enumerate(loader):
            src = src.to(cfg.device)
            tgt = tgt.to(cfg.device)

            # Build decoder input: pure teacher forcing, or scheduled sampling
            if train and teacher_prob < 1.0:
                tgt_input = _build_scheduled_input(model, src, tgt, teacher_prob, use_amp)
            else:
                last_past = src[:, -1:, :]
                tgt_input = torch.cat([last_past, tgt[:, :-1, :]], dim=1)

            # ── Forward pass (with optional mixed precision) ───────────────
            with torch.autocast(device_type=cfg.device, enabled=use_amp):
                mu, log_var = model(src, tgt_input)
                # Divide loss by accumulation steps so gradients are averaged,
                # not summed, across the effective batch.
                loss = nll_loss(mu, log_var, tgt) / accum

            # ── Backward ──────────────────────────────────────────────────
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

            # loss was divided by accum for gradient purposes; undo for logging
            total_loss += loss.item() * accum

            if train and (batch_idx + 1) % print_every == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"  batch {batch_idx + 1}/{n_batches}  loss={avg:.6f}", flush=True)

    return total_loss / n_batches


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    os.makedirs("checkpoints", exist_ok=True)

    train_loader, val_loader, _ = load_data()

    model     = build_model()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9
    )

    # OneCycleLR: warm-up then cosine decay within each epoch.
    # total_steps counts every optimizer step (accounting for accumulation).
    # Each epoch has ceil(batches / accumulation) optimizer steps.
    # +1 buffer guards against an off-by-one when len(train_loader) is one
    # fewer than the batches actually yielded (last partial batch edge case).
    total_steps = ((len(train_loader) + cfg.grad_accumulation - 1) // cfg.grad_accumulation) * cfg.epochs + 1
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = cfg.lr,
        total_steps     = total_steps,
        pct_start       = 0.1,    # 10 % warm-up
        anneal_strategy = "cos",
    )

    use_amp = cfg.use_amp and cfg.device == "cuda"
    amp_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")
    print(f"Device: {cfg.device}")
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")
    print(f"Effective batch size: {cfg.batch_size * cfg.grad_accumulation}\n")

    best_val_loss = float("inf")

    ramp_epochs = max(1, cfg.epochs - cfg.ss_start_epoch)

    for epoch in range(1, cfg.epochs + 1):
        # Linear ramp: pure teacher forcing until ss_start_epoch, then
        # gradually increase the fraction of model-predicted decoder inputs.
        if epoch <= cfg.ss_start_epoch:
            teacher_prob = 1.0
        else:
            progress     = (epoch - cfg.ss_start_epoch) / ramp_epochs
            teacher_prob = 1.0 - cfg.ss_max_prob * min(1.0, progress)

        train_loss = run_epoch(model, train_loader, optimizer, amp_scaler, scheduler,
                               train=True, teacher_prob=teacher_prob)
        val_loss   = run_epoch(model, val_loader,   optimizer, amp_scaler, scheduler,
                               train=False)

        if epoch % cfg.log_every == 0:
            print(
                f"Epoch {epoch:3d}/{cfg.epochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"teacher={teacher_prob:.2f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_loss":    val_loss,
                    "config": {
                        "n_features":     cfg.n_features,
                        "d_model":        cfg.d_model,
                        "num_heads":      cfg.num_heads,
                        "num_layers":     cfg.num_layers,
                        "d_ff":           cfg.d_ff,
                        "max_seq_length": cfg.max_seq_length,
                        "dropout":        cfg.dropout,
                    },
                },
                cfg.checkpoint_path,
            )
            print(f"  ✓ saved best model (val_loss={val_loss:.6f})")

    print(f"\nTraining complete.  Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
