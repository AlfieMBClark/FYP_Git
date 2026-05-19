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
    # Real held-out evaluation uses predict.py against worldwide.db, so we
    # dedicate all DMA vessels to train/val rather than wasting 10% on a
    # redundant test split from the same source.
    train_split = 0.90
    val_split   = 0.10
    # test = 0.0 — evaluation is done against worldwide.db via predict.py

    # ── Geographic region filter ──────────────────────────────────────────────
    # Bounding box for Europe, North Africa, and the Middle East.
    # Applied at ingestion (Data_Processing.py --bbox default) and again when
    # building training windows (prepare_dataset.py / Data_Processing.py Phase 3)
    # so that existing databases are also cleaned on next prepare run.
    # Format: (lat_min, lat_max, lon_min, lon_max)
    region_bounds = (10.0, 75.0, -30.0, 65.0)

    # ── Track-level filters (applied in prepare_dataset.py) ───────────────────
    # Split a track into separate voyages when there is a gap longer than this.
    gap_max_seconds     = 7_200   # 2 hours

    # Discard voyage segments shorter than this many pings.
    min_voyage_points   = 20

    # Discard a voyage if its peak SOG never exceeds this value (stationary trip).
    max_sog_minimum     = 1.0    # knots

    # A ping is considered "low speed" if SOG is below this threshold.
    low_speed_threshold = 2.0    # knots

    # Discard a voyage if more than this fraction of its pings are low speed.
    low_speed_fraction  = 0.80

    # If the FLAGS column exists in the database, only include rows where FLAGS=0
    # (no anomaly flags set) in the clean training data.
    clean_flags_only    = True

    # Step between consecutive sliding windows within a voyage.
    # 1 = fully overlapping (many windows but slower to prepare).
    # 10 = ~10× fewer windows, good balance for large datasets.
    window_stride       = 10

    # ── Stratified sampling (applied in prepare_dataset.py) ───────────────────
    # Maximum windows taken from any single MMSI.  Prevents a single busy vessel
    # (e.g. a ferry with 100k pings) from dominating the training distribution.
    max_windows_per_mmsi       = 200

    # Maximum total windows per semantic ship-type group within one split.
    # Once a group hits this limit, further MMSIs in that group are skipped.
    # Setting this equal across groups gives balanced type coverage.
    max_windows_per_type_group = 50_000

    # ── Ship-type semantic groups ─────────────────────────────────────────────
    # Raw ITU R M.1371 SHIP_TYPE codes (0–99) → group index (0–7).
    # The group index replaces the raw code as the SHIP_TYPE feature so the
    # model sees 8 meaningful categories rather than near-arbitrary integers.
    #   0 Unknown   1 Cargo    2 Tanker   3 Passenger
    #   4 Fishing   5 Tug/Service  6 Pleasure/Sailing  7 Other
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

    # ── Normalisation (fixed physical bounds — no fitting pass needed) ─────────
    # Each feature is clipped to its physical range then scaled to [0, 1].
    # Using fixed bounds means the scaler is deterministic across datasets.
    norm_bounds = {
        "LAT":       (-90.0,   90.0),
        "LON":      (-180.0,  180.0),
        "SOG":        (0.0,    30.0),   # values above 30 kts are clamped, not dropped
        "COG_SIN":   (-1.0,    1.0),    # sin(COG) — circular encoding avoids 359°/1° discontinuity
        "COG_COS":   (-1.0,    1.0),    # cos(COG)
        "SHIP_TYPE":  (0.0,    7.0),    # group index 0–7 (see ship_type_groups)
    }

    # ── Features ──────────────────────────────────────────────────────────────
    feature_cols = ["LAT", "LON", "SOG", "COG_SIN", "COG_COS", "SHIP_TYPE"]
    n_features   = len(feature_cols)    # 6

    # ── Model architecture ────────────────────────────────────────────────────
    d_model        = 256
    num_heads      = 8
    num_layers     = 4
    d_ff           = 1024
    dropout        = 0.1
    max_seq_length = 200

    # ── Scheduled sampling ────────────────────────────────────────────────────
    # Bridges the train/inference gap (exposure bias): after ss_start_epoch,
    # the decoder is fed the model's own predictions instead of ground truth
    # with linearly increasing probability, reaching ss_max_prob at the final epoch.
    ss_start_epoch = 5    # epochs of pure teacher forcing before sampling begins
    ss_max_prob    = 0.5  # peak probability of using model prediction as next input

    # ── Training ──────────────────────────────────────────────────────────────
    # Larger batch size than before because mixed precision halves GPU memory use.
    batch_size        = 256

    # Peak learning rate for OneCycleLR. The scheduler warms up from lr/div_factor
    # then cosine-anneals back down, giving faster convergence than a fixed rate.
    lr                = 1e-3

    epochs            = 20
    num_workers       = 4
    grad_clip         = 1.0

    # Effective batch size = batch_size × grad_accumulation.
    # Accumulating gradients over multiple forward passes before stepping the
    # optimiser approximates a larger batch without extra GPU memory.
    grad_accumulation = 4

    log_every         = 1

    # torch.cuda.amp automatic mixed precision (float16 ops where safe).
    use_amp           = True

    # torch.compile gives ~20–40 % throughput improvement on PyTorch ≥ 2.0
    # with no code changes to the model itself.
    compile_model     = False  # torch.compile requires Triton, which is Linux-only

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_path = "checkpoints/best_model.pt"

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
