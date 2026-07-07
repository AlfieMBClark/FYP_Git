"""
config.py
---------
All hyperparameters and paths in one place.
"""

import torch


class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    # Source databases — kept separate so train and test never share vessels.
    # DMA data (2 years of dense Danish/North Sea AIS) is used for training.
    # A second DMA database (2023) is used only for evaluation and prediction.
    train_db_path = "data/dma.db"
    test_db_path  = "data2/2023.db"

    # Pre-processed window files produced by prepare_dataset.py.
    # The .bin files are raw float32 memory-maps; shapes are in dataset_meta.json.
    train_windows   = "data/train_windows.bin"
    val_windows     = "data/val_windows.bin"
    test_windows    = "data2/test_windows.bin"     # WorldwideAIS 2023 — separate from DMA training data
    anomaly_windows = "data2/anomaly_windows.bin"  # FLAGS != 0 rows from WorldwideAIS test set
    meta_path       = "data/dataset_meta.json"

    # ── Sequence lengths ───────────────────────────────────────────────────────
    seq_len_enc = 90    # past pings seen by encoder
    seq_len_dec = 10   # future pings predicted by decoder — must match dataset_meta.json

    # ── Train / val split (DMA DB only — test set is WorldwideAIS) ───────────
    train_split = 0.90
    val_split   = 0.10

    # ── Geographic region filter ──────────────────────────────────────────────
    region_bounds = (50.0, 66.0, -5.0, 20.0)

    # ── Track-level filters (applied in prepare_dataset.py) ───────────────────
    gap_max_seconds     = 7_200   # 2 hours
    min_voyage_points   = 20
    max_sog_minimum     = 1.0    # knots — exclude near-stationary voyages from training
    low_speed_threshold = 2.0    # knots
    low_speed_fraction  = 0.60   # default — drop voyages where >60% of pings are below threshold
    # Per-group overrides: ship types that legitimately operate slowly get a relaxed limit.
    # Keys are group indices from ship_type_groups; missing groups fall back to low_speed_fraction.
    low_speed_fraction_by_group = {
        5: 0.90,   # Tug/Service — slow manoeuvring is normal operational behaviour
        4: 0.80,   # Fishing — hauling, netting, slow trawling
        6: 0.85,   # Pleasure/Sail — becalmed or motoring slowly
    }
    clean_flags_only    = True
    window_stride       = 10

    # ── Stratified sampling (applied in prepare_dataset.py) ───────────────────
    max_windows_per_mmsi       = 300
    max_windows_per_type_group = 70_000

    # ── Ship-type semantic groups ─────────────────────────────────────────────
    ship_type_groups = {
        **{c: 0 for c in [0, 38, 39]},
        **{c: 1 for c in range(70, 80)},
        **{c: 2 for c in range(80, 90)},
        **{c: 3 for c in range(60, 70)},
        30: 4,
        **{c: 5 for c in [31, 32, 33, 34, 50, 51, 52, 53, 54, 55, 56, 57, 58]},
        **{c: 6 for c in [36, 37]},
        **{c: 7 for c in [*range(20, 30), 35, *range(40, 50), 59, *range(90, 100)]},
    }

    # ── Features ──────────────────────────────────────────────────────────────
    # DT is placed before SHIP_TYPE so the decoder input slice [:N_DEC_IN]
    # naturally captures [LAT, LON, SOG, COG_SIN, COG_COS, DT] without SHIP_TYPE.
    # dLAT, dLON, dCOG are derived delta features appended after SHIP_TYPE.
    # They are encoder-only: the decoder input/output (indices 0-5) is unchanged.
    feature_cols         = ["LAT", "LON", "SOG", "COG_SIN", "COG_COS", "DT", "SHIP_TYPE",
                            "dLAT", "dLON", "dCOG", "ROT", "HDG_SIN", "HDG_COS", "NAV_STATUS"]
    n_features           = 14  # total features stored per ping in .bin files
    n_enc_features       = 14  # encoder input: full feature set including derived + AIS fields
    n_dec_features       = 5   # decoder output: LAT, LON, SOG, COG_SIN, COG_COS
    n_dec_input_features = 6   # decoder input: adds DT (index 5); all others encoder-only

    # ── Normalisation (fixed physical bounds — no fitting pass needed) ─────────
    norm_bounds = {
        "LAT":       (50.0,   66.0),   # Danish/North Sea actual range
        "LON":       (-5.0,   20.0),   # Danish/North Sea actual range
        "SOG":        (0.0,    30.0),
        "COG_SIN":   (-1.0,    1.0),
        "COG_COS":   (-1.0,    1.0),
        "DT":         (0.0, 7200.0),   # seconds since previous ping; first ping = 0 (matches gap_max_seconds)
        "SHIP_TYPE":  (0.0,    7.0),
        "dLAT":      (-2.0,    2.0),   # degrees latitude change per ping (~30 kts × 2 h max)
        "dLON":      (-2.0,    2.0),   # degrees longitude change per ping
        "dCOG":    (-180.0,  180.0),   # heading change per ping (degrees, wrap-corrected)
        "ROT":     (-127.0,  127.0),   # rate of turn °/min; 0 = no turn or Class B default
        "HDG_SIN":   (-1.0,    1.0),   # sin(true heading); defaults to COG when unavailable
        "HDG_COS":   (-1.0,    1.0),   # cos(true heading)
        "NAV_STATUS": (0.0,    8.0),   # navigational status: 0=underway…8=sailing
    }

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Per-feature weights for Gaussian NLL applied to the n_dec_features outputs
    # (order: LAT, LON, SOG, COG_SIN, COG_COS).  Higher weight on position
    # features so the loss surface aligns with geographic accuracy.
    loss_feature_weights = [15.0, 15.0, 1.0, 1.0, 1.0]

    # Small auxiliary term that directly penalises physical position error in km.
    # This sharpens the AR training signal without replacing the probabilistic loss.
    aux_haversine_weight = 0.02

    # Weight applied to the differentiable land-avoidance penalty during training.
    # Set to 0.0 to disable.  Tune upward if land crossings persist after training.
    land_penalty_weight = 0.1

    # ── Model architecture ────────────────────────────────────────────────────
    d_model        = 512
    num_heads      = 8
    num_layers     = 5
    d_ff           = 2048
    dropout        = 0.1
    max_seq_length = 200

    # ── AR validation subsample (shared across phases) ────────────────────────
    ar_val_subsample = 0.2   # fraction of val batches used for AR

    # ── Phase 1: TF convergence ────────────────────────────────────────────────
    # Pure teacher forcing throughout — no scheduled sampling.  Optimises core
    # predictive accuracy; checkpoint saved by TF ADE.
    # Set skip_phase1=True to reuse an existing phase1_checkpoint_path and go
    # straight to Phase 2.  Phase 1 is deterministic (seed=42) so the checkpoint
    # never needs to be retrained unless hyperparameters or data change.
    # Decoder output target: predict dLAT/dLON offsets from the previous position
    # instead of absolute LAT/LON.  Decoder *input* stays absolute so the model
    # retains geographic context.  At inference, positions are accumulated:
    #   LAT_t = LAT_{t-1} + dLAT_t.
    # dLAT/dLON bounds (-2.0, 2.0) are already in norm_bounds.
    # REQUIRES full retrain — set skip_phase1=False when enabling this.
    predict_deltas       = True
    skip_phase1          = False  # seq_len_enc changed 60→90; window shape incompatible with old P1 checkpoint
    phase1_epochs        = 60
    phase1_lr            = 1e-3
    phase1_lr_pct_start  = 0.15
    phase1_ar_val_every  = 5    # AR monitoring in phase 1 (informational only)
    phase1_ar_subsample  = 0.2
    # After NLL plateaus, add a tiny haversine nudge to sharpen position accuracy.
    # Set phase1_haversine_start_epoch=0 to disable.  Only active when skip_phase1=False.
    phase1_haversine_start_epoch = 50
    phase1_haversine_weight      = 0.005

    # ── Phase 2: AR fine-tuning ────────────────────────────────────────────────
    # Loaded from phase 1 best checkpoint.  teacher_prob anneals from
    # phase2_teacher_start down to 0.0 over phase2_teacher_anneal_epochs, then
    # stays at 0.0 (pure AR) for the remainder — avoids the hard TF->AR jump.
    # CosineAnnealingLR decays from phase2_lr to 1e-6.
    # phase2_lr was raised 10x from the original 1e-5: at 1e-5, AR ADE was
    # flat (4.06-4.19 km, no trend) for all 40 epochs of the BestRun26 run —
    # the optimiser was barely moving the weights.  1e-4 is still 10x below
    # phase1_lr's peak, conservative enough to preserve phase 1 features.
    # Note: each training batch requires 9 extra no-grad forward passes to build
    # the AR input, so phase 2 is ~10x slower per epoch than phase 1.
    phase2_epochs               = 60   # 3 full restart cycles of 20 epochs; cycles 4+ showed degradation
    phase2_lr                   = 5e-5
    phase2_teacher_start        = 1.0
    phase2_teacher_anneal_epochs = 10   # decay to 0.0 over first 10 epochs; 80 epochs of pure AR
    phase2_ar_val_every         = 2    # AR every 2 epochs — this is what we're optimising
    # When teacher_prob reaches 0.0, switch from no-grad build + single forward pass
    # to a full BPTT rollout: each of the 10 AR steps runs with gradients so the
    # loss propagates back through the compounding error chain.
    phase2_use_rollout          = True
    # CosineAnnealingWarmRestarts: LR resets to phase2_lr every T0 epochs.
    # Prevents LR from dying before BPTT has had enough cycles.
    # T0=20 gives 3 restart cycles over 60 epochs — all in pure-AR BPTT territory.
    phase2_warmrestart_t0       = 20
    # Gaussian noise std injected into the lat/lon decoder input during BPTT rollout.
    # Calibrated to the model's typical inference position error (~1.4 km at 0.001
    # normalised units in a 16°×25° region ≈ 1.8 km).  Trains the model to remain
    # accurate even when accumulated position errors are present — closing the
    # training/inference distribution gap (exposure bias).  Set to 0.0 to disable.
    phase2_position_noise_std   = 0.00

    # ── Training (shared) ─────────────────────────────────────────────────────
    batch_size        = 256
    num_workers       = 4
    grad_clip         = 1.0
    grad_accumulation = 4
    log_every         = 1
    use_amp           = True   # float16 AMP with GradScaler
    compile_model     = True   # torch.compile via Triton; supported on Linux

    # ── Checkpointing ─────────────────────────────────────────────────────────
    phase1_checkpoint_path = "checkpoints/best_model_phase1.pt"
    checkpoint_path        = "checkpoints/best_model.pt"   # phase 2 best AR ADE

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
