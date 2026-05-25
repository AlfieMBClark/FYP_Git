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
    # WorldwideAIS data is used only for evaluation and prediction.
    train_db_path = "data/dma.db"
    test_db_path  = "data/worldwide.db"

    # Pre-processed window files produced by prepare_dataset.py.
    # The .bin files are raw float32 memory-maps; shapes are in dataset_meta.json.
    train_windows   = "data/train_windows.bin"
    val_windows     = "data/val_windows.bin"
    test_windows    = "data/test_windows.bin"
    anomaly_windows = "data/anomaly_windows.bin"   # FLAGS != 0 rows from test MMSIs
    meta_path       = "data/dataset_meta.json"

    # ── Sequence lengths ───────────────────────────────────────────────────────
    seq_len_enc = 60    # past pings seen by encoder (increased for better context)
    seq_len_dec = 10    # future pings predicted by decoder

    # ── Train / val split (DMA DB only — test set is WorldwideAIS) ───────────
    train_split = 0.90
    val_split   = 0.10

    # ── Geographic region filter ──────────────────────────────────────────────
    region_bounds = (10.0, 75.0, -30.0, 65.0)

    # ── Track-level filters (applied in prepare_dataset.py) ───────────────────
    gap_max_seconds     = 7_200   # 2 hours
    min_voyage_points   = 20
    max_sog_minimum     = 1.0    # knots
    low_speed_threshold = 2.0    # knots
    low_speed_fraction  = 0.80
    clean_flags_only    = True
    window_stride       = 10

    # ── Stratified sampling (applied in prepare_dataset.py) ───────────────────
    max_windows_per_mmsi       = 200
    max_windows_per_type_group = 50_000

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
    # Encoder sees all 7 features (including SHIP_TYPE as context and DT for
    # temporal irregularity).  The decoder only predicts the 5 dynamic movement
    # features; SHIP_TYPE is held constant as context and DT is encoder-only.
    feature_cols   = ["LAT", "LON", "SOG", "COG_SIN", "COG_COS", "SHIP_TYPE", "DT"]
    n_features     = 7      # total features stored per ping in .bin files
    n_enc_features = 7      # encoder input: full feature set
    n_dec_features = 5      # decoder input/output: LAT, LON, SOG, COG_SIN, COG_COS only

    # ── Normalisation (fixed physical bounds — no fitting pass needed) ─────────
    norm_bounds = {
        "LAT":       (-90.0,   90.0),
        "LON":      (-180.0,  180.0),
        "SOG":        (0.0,    30.0),
        "COG_SIN":   (-1.0,    1.0),
        "COG_COS":   (-1.0,    1.0),
        "SHIP_TYPE":  (0.0,    7.0),
        "DT":         (0.0, 3600.0),   # seconds since previous ping; first ping = 0
    }

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Per-feature weights for Gaussian NLL applied to the n_dec_features outputs
    # (order: LAT, LON, SOG, COG_SIN, COG_COS).  Higher weight on position
    # features so the loss surface aligns with geographic accuracy.
    loss_feature_weights = [5.0, 5.0, 1.0, 1.0, 1.0]

    # ── Model architecture ────────────────────────────────────────────────────
    d_model        = 512
    num_heads      = 8
    num_layers     = 5
    d_ff           = 1024
    dropout        = 0.1
    max_seq_length = 200

    # ── Scheduled sampling ────────────────────────────────────────────────────
    # Bridges the train/inference gap (exposure bias): after ss_start_epoch,
    # the decoder is fed the model's own predictions instead of ground truth
    # with linearly increasing probability, reaching ss_max_prob at the final epoch.
    ss_start_epoch = 2    # epochs of pure teacher forcing before sampling begins
    ss_max_prob    = 0.8  # peak probability of using model prediction as next input

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size        = 256
    lr                = 1e-3
    epochs            = 20
    num_workers       = 4
    grad_clip         = 1.0
    grad_accumulation = 4
    log_every         = 1
    use_amp           = True
    compile_model     = False  # torch.compile requires Triton, which is Linux-only

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_path = "checkpoints/best_model.pt"

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
