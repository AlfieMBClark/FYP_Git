"""
config.py
---------
All hyperparameters and paths in one place.
"""

import torch


class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    data_path     = "data/ais.db"

    # Pre-processed window files produced by prepare_dataset.py.
    # The .bin files are raw float32 memory-maps; shapes are in dataset_meta.json.
    train_windows   = "data/train_windows.bin"
    val_windows     = "data/val_windows.bin"
    test_windows    = "data/test_windows.bin"
    anomaly_windows = "data/anomaly_windows.bin"   # FLAGS != 0 rows from test MMSIs
    meta_path       = "data/dataset_meta.json"

    # ── Sequence lengths ───────────────────────────────────────────────────────
    seq_len_enc = 30    # past pings seen by encoder
    seq_len_dec = 10    # future pings predicted by decoder

    # ── Train / val / test split (done at MMSI level, not window level) ────────
    # Splitting by MMSI prevents the same vessel's pings appearing in both train
    # and test, which would inflate test metrics due to track memorisation.
    train_split = 0.80
    val_split   = 0.10
    # test = 1.0 - train_split - val_split = 0.10

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
        "COG":        (0.0,   360.0),
        "SHIP_TYPE":  (0.0,    7.0),    # group index 0–7 (see ship_type_groups)
    }

    # ── Features ──────────────────────────────────────────────────────────────
    feature_cols = ["LAT", "LON", "SOG", "COG", "SHIP_TYPE"]
    n_features   = len(feature_cols)    # 5

    # ── Model architecture ────────────────────────────────────────────────────
    d_model        = 128
    num_heads      = 8
    num_layers     = 3
    d_ff           = 512
    dropout        = 0.1
    max_seq_length = 200

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
    compile_model     = True

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_path = "checkpoints/best_model.pt"

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
