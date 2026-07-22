"""
train_gru.py
-----------------
Single-phase training for ShipGRUBaseline.
Same data, same loss, same evaluation pipeline as the Transformer.
Architecture is the only difference.

Usage:
    python train_gru.py               # full 50 epochs
    python train_gru.py --smoke-test  # 1 epoch to verify pipeline
    python train_gru.py --epochs 10   # custom epoch count
"""

import csv
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE        = os.path.dirname(os.path.abspath(__file__))
_TRANSFORMER = os.path.abspath(os.path.join(_HERE, "..", "ShipTransformer"))
_DATAHANDLING = os.path.abspath(os.path.join(_HERE, "..", "DataHandling"))
sys.path.insert(0, _HERE)
sys.path.insert(0, _TRANSFORMER)
sys.path.insert(0, _DATAHANDLING)  # dataset.py lives here
os.chdir(_TRANSFORMER)  # config.py checkpoint/log paths are relative to here
from config         import cfg
from dataset        import load_data
from utils          import set_seed, haversine_tensor, load_water_mask
from gru_model import ShipGRUBaseline

N_DEC    = cfg.n_dec_features        # 5  — LAT, LON, SOG, COG_SIN, COG_COS
N_DEC_IN = cfg.n_dec_input_features  # 6  — above + DT

_LAT_LO, _LAT_HI = cfg.norm_bounds["LAT"]
_LON_LO, _LON_HI = cfg.norm_bounds["LON"]
_LAT_RNG = _LAT_HI - _LAT_LO
_LON_RNG = _LON_HI - _LON_LO

_DLAT_LO, _DLAT_HI = cfg.norm_bounds["dLAT"]
_DLON_LO, _DLON_HI = cfg.norm_bounds["dLON"]
_DLAT_RNG = _DLAT_HI - _DLAT_LO
_DLON_RNG = _DLON_HI - _DLON_LO

GRU_HIDDEN    = 256
GRU_LAYERS    = 2
GRU_DROPOUT   = 0.2
GRU_EPOCHS    = 50
CKPT_PATH     = os.path.join(_HERE, "checkpoints", "gru_model.pt")
LOG_PATH      = os.path.join(_HERE, "logs", "gru_training.log")
CSV_PATH      = os.path.join(_HERE, "logs", "gru_history.csv")
PNG_PATH      = os.path.join(_HERE, "logs", "gru_curves.png")


# ── Training curves ────────────────────────────────────────────────────────────

class TrainingCurves:
    """Per-epoch metrics: appends to a CSV and re-renders a PNG after every epoch.

    The plot is written to disk rather than shown in a window (matplotlib's Agg
    backend), so this works fine on a headless box — open the PNG and it refreshes
    as training goes.  The CSV keeps the raw numbers for writing up results.
    """

    FIELDS = ["epoch", "train_loss", "val_loss", "val_ade", "val_fde", "lr"]

    def __init__(self, csv_path: str, png_path: str):
        self.csv_path = csv_path
        self.png_path = png_path
        self.rows: list[dict] = []

    def log(self, **metrics) -> None:
        self.rows.append({f: metrics.get(f) for f in self.FIELDS})
        self._write_csv()
        self._save_png()

    def _write_csv(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def _column(self, key: str) -> list:
        return [r[key] for r in self.rows]

    def best_epoch(self):
        """Epoch with the lowest val ADE — the checkpoint that gets kept."""
        ades = self._column("val_ade")
        return self._column("epoch")[ades.index(min(ades))] if ades else None

    def _save_png(self) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")          # headless: render to file, never to a window
            import matplotlib.pyplot as plt
        except ImportError:
            return   # no matplotlib — the CSV is still written

        epochs = self._column("epoch")
        fig, (ax_loss, ax_err, ax_lr) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        for key, label in [("train_loss", "train"), ("val_loss", "validation")]:
            ax_loss.plot(epochs, self._column(key), marker="o", ms=3, lw=1.5, label=label)
        ax_loss.set_title("Loss", loc="left", fontsize=10)
        ax_loss.set_ylabel("weighted NLL")

        for key, label in [("val_ade", "ADE"), ("val_fde", "FDE")]:
            ax_err.plot(epochs, self._column(key), marker="o", ms=3, lw=1.5, label=label)
        ax_err.set_title("Validation position error", loc="left", fontsize=10)
        ax_err.set_ylabel("km")

        ax_lr.plot(epochs, self._column("lr"), marker="o", ms=3, lw=1.5, label="learning rate")
        ax_lr.set_yscale("log")
        ax_lr.set_title("Learning rate", loc="left", fontsize=10)
        ax_lr.set_ylabel("lr")
        ax_lr.set_xlabel("epoch")

        best = self.best_epoch()
        for ax in (ax_loss, ax_err, ax_lr):
            if best is not None:
                ax.axvline(best, color="grey", ls=":", lw=1)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc="best")

        title = f"GRU baseline — {len(self.rows)} epochs"
        if best is not None:
            title += f"  |  best ADE = {min(self._column('val_ade')):.3f} km @ epoch {best}"
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(self.png_path, dpi=120)
        plt.close(fig)


# ── Land mask ──────────────────────────────────────────────────────────────────
# Differentiable land-avoidance penalty (mirrors ShipTransformer/train.py):
# predicted positions on land are penalised against an OSM-derived water raster.

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
        # Shrink land inward so coastal cells straddling the shoreline are treated
        # as water and near-shore predictions are not penalised.
        land_bool = binary_erosion(land_bool, iterations=erosion_cells)
    land_np      = land_bool.astype(np.float32)
    _LAND_RASTER = torch.from_numpy(land_np).to(cfg.device)
    print(f"  [land mask] Raster ready "
          f"({_LAND_RASTER.shape[0]}×{_LAND_RASTER.shape[1]} cells, 0.005° res, "
          f"OSM-aware, eroded {erosion_cells} cell)")


def _land_penalty(pos: torch.Tensor) -> torch.Tensor:
    """pos: (B, T, >=2) with normalised absolute LAT/LON in columns 0, 1."""
    if _LAND_RASTER is None:
        return pos.new_zeros(())
    B, T, _ = pos.shape
    gx     = pos[:, :, 1] * 2.0 - 1.0
    gy     = pos[:, :, 0] * 2.0 - 1.0
    grid   = torch.stack([gx, gy], dim=-1).view(B, 1, T, 2)
    raster = _LAND_RASTER.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

    # Nearest-neighbour → hard binary land/water gate, detached (no gradient).
    on_land = F.grid_sample(
        raster, grid, mode="nearest", align_corners=True, padding_mode="border",
    ).view(B, T).detach()

    # Bilinear → smooth gradient, but only where the nearest cell is land.
    land_p = F.grid_sample(
        raster, grid, mode="bilinear", align_corners=True, padding_mode="border",
    ).view(B, T)

    return (on_land * land_p).mean()


def _pred_abs_positions(mu: torch.Tensor, dec_input: torch.Tensor) -> torch.Tensor:
    """Absolute normalised LAT/LON (B, T, 2) = previous position + predicted delta,
    so the land raster is sampled at the real predicted position, not in delta-space."""
    prev_lat = dec_input[:, :, 0] * _LAT_RNG + _LAT_LO
    prev_lon = dec_input[:, :, 1] * _LON_RNG + _LON_LO
    dlat     = mu[:, :, 0] * _DLAT_RNG + _DLAT_LO
    dlon     = mu[:, :, 1] * _DLON_RNG + _DLON_LO
    pred_lat = ((prev_lat + dlat - _LAT_LO) / _LAT_RNG).clamp(0.0, 1.0)
    pred_lon = ((prev_lon + dlon - _LON_LO) / _LON_RNG).clamp(0.0, 1.0)
    return torch.stack([pred_lat, pred_lon], dim=-1)


# ── Loss ──────────────────────────────────────────────────────────────────────

def nll_loss(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL with per-feature weights — identical to Transformer training."""
    w   = torch.tensor(cfg.loss_feature_weights, dtype=mu.dtype, device=mu.device)
    nll = 0.5 * (log_var + (target - mu).pow(2) * (-log_var).exp())
    return (nll * w).mean()


def _to_delta_targets(src: torch.Tensor, tgt_dec_abs: torch.Tensor) -> torch.Tensor:
    """Replace LAT/LON columns with normalised dLAT/dLON. SOG/COG stay absolute."""
    tgt_lat  = tgt_dec_abs[:, :, 0] * _LAT_RNG + _LAT_LO
    tgt_lon  = tgt_dec_abs[:, :, 1] * _LON_RNG + _LON_LO
    last_lat = src[:, -1, 0] * _LAT_RNG + _LAT_LO
    last_lon = src[:, -1, 1] * _LON_RNG + _LON_LO
    prev_lat = torch.cat([last_lat[:, None], tgt_lat[:, :-1]], dim=1)
    prev_lon = torch.cat([last_lon[:, None], tgt_lon[:, :-1]], dim=1)
    dlat_norm = ((tgt_lat - prev_lat) - _DLAT_LO) / _DLAT_RNG
    dlon_norm = ((tgt_lon - prev_lon) - _DLON_LO) / _DLON_RNG
    out = tgt_dec_abs.clone()
    out[:, :, 0] = dlat_norm
    out[:, :, 1] = dlon_norm
    return out


# ── ADE / FDE ─────────────────────────────────────────────────────────────────

def _compute_ade_fde(mu: torch.Tensor, tgt_dec: torch.Tensor) -> tuple[float, float, int]:
    """Haversine ADE/FDE in km. mu must be in absolute normalised [0,1] space."""
    pred_lat = mu[:, :, 0].detach() * _LAT_RNG + _LAT_LO
    pred_lon = mu[:, :, 1].detach() * _LON_RNG + _LON_LO
    true_lat = tgt_dec[:, :, 0] * _LAT_RNG + _LAT_LO
    true_lon = tgt_dec[:, :, 1] * _LON_RNG + _LON_LO

    pred_coords = torch.stack([pred_lat, pred_lon], dim=-1)
    true_coords = torch.stack([true_lat, true_lon], dim=-1)
    dists = haversine_tensor(pred_coords, true_coords)  # (B, T)

    return dists.mean(dim=1).sum().item(), dists[:, -1].sum().item(), mu.size(0)


# ── Training / evaluation loop ─────────────────────────────────────────────────

def run_epoch(
    model:     ShipGRUBaseline,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    scheduler,
    train:     bool,
) -> dict:
    """
    One epoch of training or evaluation.
    - Training:   teacher-forced forward via model.forward()
    - Validation: autoregressive forward via model.predict()
    Returns {"loss": float, "ade": float, "fde": float}
    """
    model.train(train)
    total_loss = 0.0
    total_ade  = 0.0
    total_fde  = 0.0
    n_samples  = 0
    n_batches  = len(loader)

    use_amp = cfg.use_amp and cfg.device == "cuda"
    accum   = cfg.grad_accumulation

    if train:
        optimizer.zero_grad()

    with torch.set_grad_enabled(train):
        for batch_idx, (src, tgt) in enumerate(loader):
            src = src.to(cfg.device)
            tgt = tgt.to(cfg.device)

            # tgt has 14 features; first 6 are [LAT, LON, SOG, COG_SIN, COG_COS, DT]
            tgt_dec_in   = tgt[:, :, :N_DEC_IN].nan_to_num(0.5)  # (B, 10, 6)
            tgt_dec      = tgt_dec_in[:, :, :N_DEC]               # (B, 10, 5) — absolute positions
            tgt_dec_loss = _to_delta_targets(src, tgt_dec)         # (B, 10, 5) — delta targets

            if train:
                # Teacher-forced: shift tgt right, seed from last encoder step
                last_past = src[:, -1:, :N_DEC_IN]                               # (B, 1, 6)
                tgt_input = torch.cat([last_past, tgt_dec_in[:, :-1, :]], dim=1)  # (B, 10, 6)

                with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=use_amp):
                    mu, log_var = model(src, tgt_input)

                mu_f  = mu.float()
                lv_f  = log_var.float()
                loss  = nll_loss(mu_f, lv_f, tgt_dec_loss.float()) / accum

                # Penalise predictions that land on land.
                if cfg.land_penalty_weight > 0:
                    pred_pos = _pred_abs_positions(mu_f, tgt_input)
                    loss = loss + cfg.land_penalty_weight * _land_penalty(pred_pos) / accum

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                is_last_accum = (
                    (batch_idx + 1) % accum == 0
                    or (batch_idx + 1) == n_batches
                )
                if is_last_accum:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accum

            else:
                # Autoregressive validation: decode step by step, accumulating the
                # predicted deltas into absolute positions for the next input.
                # (Not model.predict(), which would feed deltas back as absolutes.)
                _, dec_hidden = model.encode(src)
                prev_abs = src[:, -1:, :N_DEC].float()  # (B, 1, 5) absolute normalised
                mu_list, lv_list, abs_list = [], [], []

                for t in range(tgt_dec_in.size(1)):
                    dt_t   = tgt_dec_in[:, t:t+1, N_DEC:N_DEC_IN].float()  # (B, 1, 1)
                    dec_in = torch.cat([prev_abs, dt_t], dim=-1)             # (B, 1, 6)

                    with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=use_amp):
                        mu_s, lv_s, _, dec_hidden = model.decode_step(dec_in, dec_hidden)

                    mu_s = mu_s.float()
                    lv_s = lv_s.float()

                    # Accumulate dLAT/dLON → next absolute position
                    prev_lat = prev_abs[:, :, 0:1] * _LAT_RNG + _LAT_LO
                    prev_lon = prev_abs[:, :, 1:2] * _LON_RNG + _LON_LO
                    dlat = mu_s[:, :, 0:1] * _DLAT_RNG + _DLAT_LO
                    dlon = mu_s[:, :, 1:2] * _DLON_RNG + _DLON_LO
                    abs_lat = ((prev_lat + dlat - _LAT_LO) / _LAT_RNG).clamp(0.0, 1.0)
                    abs_lon = ((prev_lon + dlon - _LON_LO) / _LON_RNG).clamp(0.0, 1.0)
                    sog_cog = mu_s[:, :, 2:].clamp(0.0, 1.0)
                    prev_abs = torch.cat([abs_lat, abs_lon, sog_cog], dim=-1)

                    mu_list.append(mu_s)
                    lv_list.append(lv_s)
                    abs_list.append(prev_abs)

                mu_f     = torch.cat(mu_list,  dim=1)   # (B, T, 5) delta outputs — for loss
                lv_f     = torch.cat(lv_list,  dim=1)
                abs_mu_f = torch.cat(abs_list, dim=1)   # (B, T, 5) absolute — for ADE

                loss = nll_loss(mu_f, lv_f, tgt_dec_loss.float())

                sum_ade, sum_fde, bs = _compute_ade_fde(abs_mu_f, tgt_dec)
                total_ade += sum_ade
                total_fde += sum_fde
                n_samples += bs
                total_loss += loss.item()

    mean_ade = total_ade / n_samples if n_samples > 0 else 0.0
    mean_fde = total_fde / n_samples if n_samples > 0 else 0.0
    return {"loss": total_loss / n_batches, "ade": mean_ade, "fde": mean_fde}


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, epoch, val_ade, val_fde, val_loss):
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "val_ade":     val_ade,
        "val_fde":     val_fde,
        "val_loss":    val_loss,
        "config": {
            "hidden_size":  GRU_HIDDEN,
            "num_layers":   GRU_LAYERS,
            "dropout":      GRU_DROPOUT,
            "n_features":   cfg.n_features,
            "dec_features": cfg.n_dec_input_features,
            "out_features": cfg.n_dec_features,
        },
    }, CKPT_PATH)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GRU baseline for ship trajectory prediction.")
    parser.add_argument("--epochs",     type=int, default=GRU_EPOCHS)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 1 training batch + 1 val batch to verify the pipeline.")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(os.path.join(_HERE, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(_HERE, "logs"), exist_ok=True)

    log_file = open(LOG_PATH, "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\n=== GRU Baseline — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # Rewritten every epoch, so the PNG can be watched while the run is going.
    history = TrainingCurves(CSV_PATH, PNG_PATH)

    print("Loading data ...")
    train_loader, val_loader, _ = load_data()

    if args.smoke_test:
        from torch.utils.data import DataLoader, Subset
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(min(cfg.batch_size * 2, len(train_loader.dataset)))),
            batch_size=cfg.batch_size, shuffle=False, num_workers=0,
        )
        val_loader = DataLoader(
            Subset(val_loader.dataset, range(min(cfg.batch_size, len(val_loader.dataset)))),
            batch_size=cfg.batch_size, shuffle=False, num_workers=0,
        )
        log("  [smoke-test] Using 2 train batches + 1 val batch.")

    model = ShipGRUBaseline(
        n_features   = cfg.n_features,
        dec_features = cfg.n_dec_input_features,
        out_features = cfg.n_dec_features,
        hidden_size  = GRU_HIDDEN,
        num_layers   = GRU_LAYERS,
        dropout      = GRU_DROPOUT,
    ).to(cfg.device)

    _init_land_raster()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model: {n_params:,} trainable parameters")
    log(f"Device: {cfg.device}  AMP: {'enabled' if cfg.use_amp and cfg.device == 'cuda' else 'disabled'}")
    log(f"Effective batch size: {cfg.batch_size * cfg.grad_accumulation}")
    log(f"Encoder sequence length: {cfg.seq_len_enc}  Land penalty weight: {cfg.land_penalty_weight}")

    use_amp    = cfg.use_amp and cfg.device == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", init_scale=2**14, growth_interval=1_000_000) if use_amp else None

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.phase1_lr, betas=(0.9, 0.98), eps=1e-9,
    )

    epochs = 1 if args.smoke_test else args.epochs
    total_steps = (
        (len(train_loader) + cfg.grad_accumulation - 1) // cfg.grad_accumulation
    ) * epochs + 1

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = cfg.phase1_lr,
        total_steps     = total_steps,
        pct_start       = 0.1,
        anneal_strategy = "cos",
    )

    best_val_ade = float("inf")
    log("")

    for epoch in range(1, epochs + 1):
        train_m = run_epoch(model, train_loader, optimizer, amp_scaler, scheduler, train=True)
        val_m   = run_epoch(model, val_loader,   optimizer, amp_scaler, None,      train=False)

        lr  = scheduler.get_last_lr()[0]
        msg = (
            f"Epoch {epoch:>3}/{epochs}  "
            f"train_loss={train_m['loss']:>8.3f}  "
            f"val_loss={val_m['loss']:>8.3f}  "
            f"val_ade={val_m['ade']:>6.2f}  "
            f"val_fde={val_m['fde']:>6.2f}  "
            f"lr={lr:.2e}"
        )
        log(msg)

        history.log(
            epoch      = epoch,
            train_loss = train_m["loss"],
            val_loss   = val_m["loss"],
            val_ade    = val_m["ade"],
            val_fde    = val_m["fde"],
            lr         = lr,
        )

        if val_m["ade"] < best_val_ade:
            best_val_ade = val_m["ade"]
            save_checkpoint(model, epoch, val_m["ade"], val_m["fde"], val_m["loss"])
            log(f"  ✓ Saved best checkpoint (val_ade={best_val_ade:.2f} km)")

    log(f"\nTraining complete.  Best val ADE: {best_val_ade:.2f} km")
    log(f"Checkpoint: {CKPT_PATH}")
    log(f"Log:        {LOG_PATH}")
    log(f"Curves:     {PNG_PATH}")
    log(f"History:    {CSV_PATH}")
    log_file.close()


if __name__ == "__main__":
    main()
