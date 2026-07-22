"""
config.py
---------
All hyperparameters and paths in one place.
"""

import os

import torch

# Data lives in the sibling DataHandling/ folder. Anchor absolutely off this
# file so the paths resolve no matter what the current working directory is.
_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DataHandling"))


class Config:
    # ----- Paths -----
    #Data root
    data_root = _DATA

    #Separate databases
    train_db_path = os.path.join(_DATA, "training", "dma.db")    # training + validation
    test_db_path  = os.path.join(_DATA, "testing",  "2023.db")   # held-out year for evaluation

    # Pre-processed window files -----
    train_windows   = os.path.join(_DATA, "training", "train_windows.bin")
    val_windows     = os.path.join(_DATA, "training", "val_windows.bin")
    test_windows    = os.path.join(_DATA, "testing",  "test_windows.bin")
    anomaly_windows = os.path.join(_DATA, "testing",  "anomaly_windows.bin")
    meta_path       = os.path.join(_DATA, "training", "dataset_meta.json")

    # -----Sequence lengths -----
    seq_len_enc = 90 # past pings seen by encoder
    seq_len_dec = 10 # future pings predicted by decoder — match dataset_meta.json

    # ----- Train / val split ----
    train_split = 0.90
    val_split   = 0.10

    # ----- Geographic filter -----
    region_bounds = (50.0, 66.0, -5.0, 20.0)

    # ----- Track-level filters (prepare_dataset.py) -----
    gap_max_seconds     = 7_200 # 2 hours
    min_voyage_points   = 20
    max_sog_minimum     = 1.0   # knots - exclude near-stationary voyages from training
    low_speed_threshold = 2.0    #knots
    low_speed_fraction  = 0.60   #drop >60% of pings are below threshold
    # Per-group overrides: ship types expected to operate slowly get a relaxed limit
    low_speed_fraction_by_group = {
        5: 0.90,   # Tug/Service
        4: 0.80,   # Fishing
        6: 0.85,   # Pleasure/Sail
    }
    clean_flags_only = True
    window_stride = 10

    # ----- Stratified sampling  -----
    #caps stop abundant classes (Cargo, Tanker) from swamping rare ones
    max_windows_per_mmsi       = 300
    max_windows_per_type_group = 125_000

    # ----- Ship-type semantic groups -----
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

    # -----Features -----
    # Feature order matters: the decoder input is the first 6 columns
    # (LAT, LON, SOG, COG_SIN, COG_COS, DT) and its output the first 5.(SHIP_TYPE, deltas, ROT, heading, nav status) are encoder-only.
    feature_cols = ["LAT", "LON", "SOG", "COG_SIN", "COG_COS", "DT", "SHIP_TYPE", "dLAT", "dLON", "dCOG", "ROT", "HDG_SIN", "HDG_COS", "NAV_STATUS"]
    n_features = 14  # features stored per ping in the .bin files
    n_enc_features = 14  # encoder input
    n_dec_features = 5   # decoder output: LAT, LON, SOG, COG_SIN, COG_COS
    n_dec_input_features = 6   

    # ----- Normalisation  -----
    norm_bounds = {
        "LAT":       (50.0,   66.0),   # Danish/North sea
        "LON":       (-5.0,   20.0),   # Danish/North sea
        "SOG":        (0.0,    30.0),
        "COG_SIN":   (-1.0,    1.0),
        "COG_COS":   (-1.0,    1.0),
        "DT":         (0.0, 7200.0),   # seconds since previous ping; first ping = 0 (match gap_max_seconds)
        "SHIP_TYPE":  (0.0,    7.0),
        "dLAT":      (-2.0,    2.0),   # degrees lat change per ping
        "dLON":      (-2.0,    2.0),   # degrees long change per ping
        "dCOG":    (-180.0,  180.0),   # heading change per ping
        "ROT":     (-127.0,  127.0),   # rate of turn °/min;
        "HDG_SIN":   (-1.0,    1.0),   # sin(true heading); defaults to COG when unavailable
        "HDG_COS":   (-1.0,    1.0),   # cos(true heading)
        "NAV_STATUS": (0.0,    8.0),   # navigational status;
    }

    # ----- Loss -----
    # Per-feature NLL weights (LAT, LON, SOG, COG_SIN, COG_COS)
    loss_feature_weights = [15.0, 15.0, 1.0, 1.0, 1.0]

    aux_haversine_weight = 0.02   # position error penalty in km
    land_penalty_weight  = 0.2    # land-avoidance penalty weight

    # ----- Model architecture -----
    d_model        = 128 
    num_heads      = 4
    num_layers     = 5
    d_ff           = 512
    dropout        = 0.1
    max_seq_length = 200


    # ----- AR validation subsample  -----
    ar_val_subsample = 0.2   # fraction of val batches used for AR

    # ----- Phase 1: teacher-forced convergence -----
    predict_deltas       = True #T predicts position offsets 
    skip_phase1          = False
    phase1_epochs        = 60
    phase1_lr            = 1e-3
    phase1_lr_pct_start  = 0.15
    phase1_ar_val_every  = 5 #AR validation
    phase1_ar_subsample  = 0.2
    #NLL loss plateaus, add a small haversine term
    phase1_haversine_start_epoch = 50
    phase1_haversine_weight      = 0.005

    # ----- Phase 2: autoregressive fine-tuning -----
    # Starts from the Phase 1 checkpoint; teacher_prob anneals to 0
    phase2_epochs                = 60
    phase2_lr                    = 5e-5
    phase2_teacher_start         = 1.0
    phase2_teacher_anneal_epochs = 10 #decay to 0.0 over the first 10 epochs
    phase2_ar_val_every          = 2
    # teacher_prob hits 0, unroll decode with gradients (BPTT) so loss propagates back
    phase2_use_rollout           = True
    phase2_warmrestart_t0        = 20
    phase2_position_noise_std    = 0.00

    # ----- Training -----
    batch_size        = 256
    num_workers       = 4
    grad_clip         = 1.0
    grad_accumulation = 4
    log_every         = 1
    use_amp           = True   # float16 AMP with GradScaler
    compile_model     = True   # torch.compile via Triton - Linux only

    # ----- Checkpointing -----
    phase1_checkpoint_path = "checkpoints/transModel_phase1.pt"
    checkpoint_path        = "checkpoints/transformer_model.pt"   # phase 2 best AR ADE

    # ----- Device -----
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
