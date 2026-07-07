"""
prepare_dataset.py
------------------
python3 prepare_dataset.py --db data/dma.db --test-db data2/2023.db

One-time preprocessing step: reads the AIS SQLite database, applies all
track-level filters, creates sliding windows, and writes three memory-mapped
binary files (train / val / test) that train.py streams from.

Run once before training (and again whenever config.py changes):
    python prepare_dataset.py
    python prepare_dataset.py --db data/dma.db --out data/

Pipeline
--------
1.  Load all distinct MMSIs and shuffle them with a fixed seed.
2.  Split MMSIs 90 / 10 into train / val (configured by train_split / val_split).
    Splitting at MMSI level prevents the same vessel appearing in both splits.
    Held-out test evaluation uses WorldwideAIS via predict.py, not a DMA split.
3.  For each split, MMSIs are processed in parallel (multiprocessing.Pool):
      Each worker fetches a batch of MMSIs in a single SQL query, runs
      split_and_filter, and returns (voyages, dominant_group) per MMSI.
    Cap accounting (per-MMSI and per-type-group limits) is applied serially
    in the original MMSI order so output is deterministic.
    Voyages are cached in memory, so no second DB pass is needed.
4.  Build an anomaly evaluation set from FLAGS != 0 rows in the test MMSIs.
5.  Save a metadata JSON with shapes used by dataset.py.

Track-level filters (configured in config.py)
----------------------------------------------
  FLAGS filter      : only FLAGS=0 rows used for train/val/test clean sets.
                      FLAGS != 0 rows go into anomaly_windows.bin for eval.
  Gap splitting     : split a track where the gap between consecutive pings
                      exceeds gap_max_seconds (default 2 h).
  Minimum length    : drop voyage segments with fewer than min_voyage_points pings.
  Stationary check  : drop a voyage whose peak SOG never exceeds max_sog_minimum.
  Low-speed filter  : drop a voyage where > low_speed_fraction of pings have
                      SOG below low_speed_threshold.

Stratified sampling (configured in config.py)
----------------------------------------------
  Per-MMSI cap      : at most max_windows_per_mmsi windows per vessel.
                      Stops a single busy ferry dominating training.
  Per-group cap     : at most max_windows_per_type_group windows per semantic
                      ship-type group (Cargo, Tanker, Fishing, …).
                      Ensures all vessel categories are represented even if
                      one category is far more common in the raw data.

SHIP_TYPE feature
-----------------
Raw ITU codes (0–99) are mapped to 8 semantic group indices (0–7) via
cfg.ship_type_groups before normalisation.  This gives the model a
meaningful categorical signal instead of near-arbitrary integers.
"""

import argparse
import json
import math
import multiprocessing
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from config import cfg


# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_LEN  = cfg.seq_len_enc + cfg.seq_len_dec
FEAT_COLS   = cfg.feature_cols
N_FEATURES  = cfg.n_features
NORM_BOUNDS = cfg.norm_bounds

_NORM_LO  = np.array([NORM_BOUNDS[c][0] for c in FEAT_COLS], dtype=np.float32)
_NORM_HI  = np.array([NORM_BOUNDS[c][1] for c in FEAT_COLS], dtype=np.float32)
_NORM_RNG = _NORM_HI - _NORM_LO

GROUP_NAMES = [
    "Unknown", "Cargo", "Tanker", "Passenger",
    "Fishing", "Tug/Service", "Pleasure/Sail", "Other",
]
N_GROUPS = len(GROUP_NAMES)

# SQLite expression that converts a text TIMESTAMP column to a unix integer
# directly in the DB engine — avoids Python datetime.strptime per ping.
_TS_EXPR = "CAST(strftime('%s', TIMESTAMP) AS INTEGER)"

# Parallelism: one SQLite connection per worker, one batch query per call.
_N_WORKERS  = min(multiprocessing.cpu_count(), 8)
_BATCH_SIZE = 64   # MMSIs per worker call


# ── Ship-type helpers ──────────────────────────────────────────────────────────

def _itu_to_group(code: int) -> int:
    return cfg.ship_type_groups.get(code, 7)


def _dominant_group(rows) -> int:
    """Most common non-zero ship-type group across a set of DB rows."""
    counts = Counter(
        _itu_to_group(int(r[5])) for r in rows if int(r[5]) != 0
    )
    return counts.most_common(1)[0][0] if counts else 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize(arr: np.ndarray) -> np.ndarray:
    return np.clip((arr - _NORM_LO) / _NORM_RNG, 0.0, 1.0)


# ── Database queries ──────────────────────────────────────────────────────────

def has_flags_column(conn: sqlite3.Connection) -> bool:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(ais)")}
    return "FLAGS" in cols


def fetch_mmsi_track(
    conn: sqlite3.Connection, mmsi: int, use_flags: bool
) -> list[tuple]:
    """
    Return clean rows for one MMSI ordered by time.
    Each row: (unix_ts, LAT, LON, SOG, COG, SHIP_TYPE, ROT, HEADING, NAV_STATUS).
    Timestamp is returned as a unix integer via SQLite strftime.
    """
    if use_flags and cfg.clean_flags_only:
        q = (f"SELECT {_TS_EXPR}, LAT, LON, SOG, COG, SHIP_TYPE, ROT, HEADING, NAV_STATUS "
             "FROM ais WHERE MMSI = ? AND FLAGS = 0 ORDER BY TIMESTAMP")
    else:
        q = (f"SELECT {_TS_EXPR}, LAT, LON, SOG, COG, SHIP_TYPE, ROT, HEADING, NAV_STATUS "
             "FROM ais WHERE MMSI = ? ORDER BY TIMESTAMP")
    return conn.execute(q, (mmsi,)).fetchall()


def fetch_mmsi_track_anomalous(
    conn: sqlite3.Connection, mmsi: int
) -> list[tuple]:
    """Return only FLAGS != 0 rows for one MMSI, ordered by time."""
    return conn.execute(
        f"SELECT {_TS_EXPR}, LAT, LON, SOG, COG, SHIP_TYPE, ROT, HEADING, NAV_STATUS "
        "FROM ais WHERE MMSI = ? AND FLAGS != 0 ORDER BY TIMESTAMP",
        (mmsi,),
    ).fetchall()


# ── Track-level processing ────────────────────────────────────────────────────

_RB_LAT_MIN, _RB_LAT_MAX, _RB_LON_MIN, _RB_LON_MAX = cfg.region_bounds


def split_and_filter(rows: list[tuple]) -> list[np.ndarray]:
    """
    Convert raw DB rows for one MMSI into a list of normalised voyage arrays.

    SHIP_TYPE is mapped from its raw ITU code to a group index (0–7) before
    normalisation.  Each returned array has shape (L, N_FEATURES) where
    L >= min_voyage_points.
    Pings outside the configured region_bounds are skipped; any resulting
    gaps are handled by the existing gap-splitting logic.

    DT (7th feature) is the time in seconds since the previous ping within the
    same gap-free segment.  The first ping of each segment gets DT = 0.

    row[0] must be a unix timestamp (integer or float) as returned by the
    _TS_EXPR SQL expression.
    """
    if not rows:
        return []

    # Determine the vessel's dominant ship-type group so we can apply the
    # appropriate low_speed_fraction limit.  Tugs, fishing, and leisure vessels
    # legitimately spend most of their time below the speed threshold, so they
    # get a relaxed limit from low_speed_fraction_by_group.
    dom_group = _dominant_group(rows)
    low_speed_limit = cfg.low_speed_fraction_by_group.get(dom_group, cfg.low_speed_fraction)

    timestamps: list[float] = []
    features:   list[list]  = []

    for row in rows:
        try:
            t    = float(row[0])   # unix timestamp from SQLite strftime
            lat  = float(row[1])
            lon  = float(row[2])
            sog  = float(row[3])
            cog  = float(row[4])
            grp  = float(_itu_to_group(int(row[5])))
            rot  = float(row[6]) if row[6] is not None else 0.0
            rot  = max(-127.0, min(127.0, rot))
            hdg  = float(row[7]) if row[7] is not None else cog
            nav  = float(row[8]) if row[8] is not None else 0.0
        except (ValueError, TypeError):
            continue
        if not (_RB_LAT_MIN <= lat <= _RB_LAT_MAX and _RB_LON_MIN <= lon <= _RB_LON_MAX):
            continue
        timestamps.append(t)
        cog_rad = math.radians(cog)
        hdg_rad = math.radians(hdg)
        # Indices 0-5: model features.
        # Indices 6-8: raw lat/lon/cog kept temporarily for delta computation.
        # Indices 9-12: ROT, HDG_SIN, HDG_COS, NAV_STATUS.
        features.append([lat, lon, sog, math.sin(cog_rad), math.cos(cog_rad), grp,
                         lat, lon, cog,
                         rot, math.sin(hdg_rad), math.cos(hdg_rad), nav])

    if len(features) < cfg.min_voyage_points:
        return []

    ts_arr   = np.array(timestamps, dtype=np.float64)
    feat_arr = np.array(features,   dtype=np.float32)

    # Gap splitting — keep ts_arr and feat_arr in sync
    split_at  = np.where(ts_arr[1:] - ts_arr[:-1] > cfg.gap_max_seconds)[0] + 1
    feat_segs = np.split(feat_arr, split_at)
    ts_segs   = np.split(ts_arr,   split_at)

    voyages = []
    for seg, ts_seg in zip(feat_segs, ts_segs):
        if len(seg) < cfg.min_voyage_points:
            continue
        sog_col = seg[:, 2]
        if sog_col.max() < cfg.max_sog_minimum:
            continue
        if (sog_col < cfg.low_speed_threshold).mean() > low_speed_limit:
            continue

        # DT: seconds since previous ping; 0 for the first ping in the segment.
        dt = np.zeros(len(seg), dtype=np.float32)
        dt[1:] = np.diff(ts_seg).astype(np.float32)

        # dLAT / dLON: position displacement in degrees per ping.
        dlat = np.zeros(len(seg), dtype=np.float32)
        dlon = np.zeros(len(seg), dtype=np.float32)
        dlat[1:] = np.diff(seg[:, 6]).astype(np.float32)
        dlon[1:] = np.diff(seg[:, 7]).astype(np.float32)

        # dCOG: wrap-aware heading change in degrees per ping [-180, 180].
        dcog = np.zeros(len(seg), dtype=np.float32)
        raw_diff = np.diff(seg[:, 8])
        dcog[1:] = ((raw_diff + 180.0) % 360.0 - 180.0).astype(np.float32)

        # Final layout:
        # [LAT, LON, SOG, COG_SIN, COG_COS, DT, SHIP_TYPE,
        #  dLAT, dLON, dCOG, ROT, HDG_SIN, HDG_COS, NAV_STATUS]
        # Decoder input/output still uses only indices 0-5 / 0-4 — unchanged.
        seg_final = np.concatenate([
            seg[:, :5],            # LAT, LON, SOG, COG_SIN, COG_COS  (0-4)
            dt[:, np.newaxis],     # DT        (5)
            seg[:, 5:6],           # SHIP_TYPE (6)
            dlat[:, np.newaxis],   # dLAT      (7)
            dlon[:, np.newaxis],   # dLON      (8)
            dcog[:, np.newaxis],   # dCOG      (9)
            seg[:, 9:10],          # ROT       (10)
            seg[:, 10:12],         # HDG_SIN, HDG_COS (11-12)
            seg[:, 12:13],         # NAV_STATUS (13)
        ], axis=1)

        normed = normalize(seg_final)
        if np.isnan(normed).any():
            continue  # discard voyage segments containing NaN features
        voyages.append(normed)

    return voyages


def count_windows_for_voyages(voyages: list[np.ndarray]) -> int:
    return sum(
        (len(v) - WINDOW_LEN) // cfg.window_stride + 1
        for v in voyages if len(v) >= WINDOW_LEN
    )


def _write_mmsi_windows(
    fp: np.memmap, idx: int, voyages: list[np.ndarray], max_w: int
) -> int:
    """Write the first max_w windows from voyages into fp[idx:]. Returns count written."""
    written = 0
    for v in voyages:
        n = len(v)
        for start in range(0, n - WINDOW_LEN + 1, cfg.window_stride):
            if written >= max_w:
                return written
            fp[idx + written] = v[start : start + WINDOW_LEN]
            written += 1
    return written


# ── Parallel worker functions (module-level so they are picklable) ─────────────

def _worker_process_mmsis(args: tuple) -> list[tuple]:
    """
    Fetch and process a batch of MMSIs with a single SQL query.

    Opens its own SQLite connection (required — connections are not picklable).
    Returns a list of (voyages, dominant_group) tuples in the same order as
    the input mmsis list, so cap accounting in the caller stays deterministic.
    """
    db_path, mmsis, use_flags = args
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size=-32000")   # 32 MB read cache per worker
    conn.execute("PRAGMA mmap_size=4294967296")  # memory-mapped I/O up to 4 GB

    placeholders = ','.join('?' * len(mmsis))
    if use_flags and cfg.clean_flags_only:
        q = (f"SELECT MMSI, {_TS_EXPR}, LAT, LON, SOG, COG, SHIP_TYPE, ROT, HEADING, NAV_STATUS "
             f"FROM ais WHERE MMSI IN ({placeholders}) AND FLAGS = 0 ORDER BY MMSI, TIMESTAMP")
    else:
        q = (f"SELECT MMSI, {_TS_EXPR}, LAT, LON, SOG, COG, SHIP_TYPE, ROT, HEADING, NAV_STATUS "
             f"FROM ais WHERE MMSI IN ({placeholders}) ORDER BY MMSI, TIMESTAMP")

    all_rows = conn.execute(q, mmsis).fetchall()
    conn.close()

    # Group rows by MMSI (already ordered by MMSI, TIMESTAMP from SQL).
    mmsi_rows: dict[int, list] = {m: [] for m in mmsis}
    for row in all_rows:
        mmsi_rows[row[0]].append(row[1:])  # strip MMSI prefix; row[1] is now unix ts

    return [
        (split_and_filter(mmsi_rows.get(m, [])), _dominant_group(mmsi_rows.get(m, [])))
        for m in mmsis  # preserve submission order
    ]


def _worker_anomaly_mmsis(args: tuple) -> list[list]:
    """
    Fetch and process anomalous (FLAGS != 0) pings for a batch of MMSIs.

    Returns a list of voyage-lists (one per MMSI) in submission order.
    """
    db_path, mmsis = args
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size=-32000")
    conn.execute("PRAGMA mmap_size=4294967296")

    placeholders = ','.join('?' * len(mmsis))
    q = (f"SELECT MMSI, {_TS_EXPR}, LAT, LON, SOG, COG, SHIP_TYPE, ROT, HEADING, NAV_STATUS "
         f"FROM ais WHERE MMSI IN ({placeholders}) AND FLAGS != 0 ORDER BY MMSI, TIMESTAMP")

    all_rows = conn.execute(q, mmsis).fetchall()
    conn.close()

    mmsi_rows: dict[int, list] = {m: [] for m in mmsis}
    for row in all_rows:
        mmsi_rows[row[0]].append(row[1:])

    return [split_and_filter(mmsi_rows.get(m, [])) for m in mmsis]


# ── Stratified writer ──────────────────────────────────────────────────────────

def process_split(
    db_path:   str,
    mmsis:     list[int],
    use_flags: bool,
    bin_path:  str,
    label:     str,
) -> int:
    """
    Parallel MMSI processing with stratified window caps.

    MMSIs are processed in parallel batches (_BATCH_SIZE each, _N_WORKERS
    processes).  pool.imap returns results in submission order, so cap
    accounting proceeds in the original MMSI order — output is identical to
    the original serial implementation.

    Processed voyages are cached in memory after the parallel step, so no
    second database pass is needed.
    """
    print(f"  [{label}] Processing {len(mmsis):,} ships "
          f"({_N_WORKERS} workers, batch {_BATCH_SIZE}) ...")

    batches     = [mmsis[i : i + _BATCH_SIZE] for i in range(0, len(mmsis), _BATCH_SIZE)]
    worker_args = [(db_path, batch, use_flags) for batch in batches]

    with multiprocessing.Pool(processes=_N_WORKERS) as pool:
        batch_results = list(pool.imap(_worker_process_mmsis, worker_args))

    # Flatten in MMSI submission order.
    all_results = [item for batch in batch_results for item in batch]

    # Apply per-MMSI and per-type-group caps serially (must be in MMSI order).
    group_remaining = {g: cfg.max_windows_per_type_group for g in range(N_GROUPS)}
    group_counts    = {g: 0 for g in range(N_GROUPS)}
    cached: list[tuple] = []   # (voyages, max_windows_to_write)
    total = 0

    for voyages, tg in all_results:
        n = min(
            count_windows_for_voyages(voyages),
            cfg.max_windows_per_mmsi,
            group_remaining[tg],
        )
        group_remaining[tg] -= n
        group_counts[tg]    += n
        cached.append((voyages, n))
        total += n

    gb = total * WINDOW_LEN * N_FEATURES * 4 / 1e9
    print(f"  [{label}] {total:,} windows ({gb:.2f} GB)")
    print(f"  [{label}] Type-group breakdown:")
    for g, name in enumerate(GROUP_NAMES):
        cap_note = "  *** capped ***" if group_remaining[g] == 0 else ""
        print(f"             {name:16s}: {group_counts[g]:>8,}{cap_note}")

    if total == 0:
        print(f"  [{label}] WARNING: no windows produced — check filters.")
        return 0

    # ── Write ─────────────────────────────────────────────────────────────────
    Path(bin_path).parent.mkdir(parents=True, exist_ok=True)
    fp = np.memmap(bin_path, dtype="float32", mode="w+",
                   shape=(total, WINDOW_LEN, N_FEATURES))

    print(f"  [{label}] Writing windows ...")
    idx = 0
    for voyages, max_w in cached:
        written = _write_mmsi_windows(fp, idx, voyages, max_w)
        idx += written

    del fp
    print(f"  [{label}] Done — {idx:,} windows written to {bin_path}")
    return idx


# ── Anomaly evaluation set ────────────────────────────────────────────────────

def build_anomaly_set(
    db_path:  str,
    mmsis:    list[int],
    bin_path: str,
) -> int:
    """
    Build a separate evaluation set from FLAGS != 0 rows in the test MMSIs.

    These windows contain actual anomalous AIS pings (position jumps, speed
    anomalies, COG/SOG mismatches, stationary drift) and can be used to
    measure the model's anomaly detection performance separately from the
    clean test set.

    The same voyage-level filters (gap splitting, minimum length) are applied
    so windows have consistent length and structure, but only anomalous rows
    are used as input.
    """
    print("  [anomaly] Processing anomalous windows ...")

    batches     = [mmsis[i : i + _BATCH_SIZE] for i in range(0, len(mmsis), _BATCH_SIZE)]
    worker_args = [(db_path, batch) for batch in batches]

    with multiprocessing.Pool(processes=_N_WORKERS) as pool:
        batch_results = list(pool.imap(_worker_anomaly_mmsis, worker_args))

    all_voyages = [vs for batch in batch_results for vs in batch]

    total = sum(count_windows_for_voyages(vs) for vs in all_voyages)

    if total == 0:
        print("  [anomaly] No anomalous windows found "
              "(FLAGS column absent or no flagged rows in test set).")
        return 0

    print(f"  [anomaly] {total:,} anomalous windows")
    Path(bin_path).parent.mkdir(parents=True, exist_ok=True)
    fp = np.memmap(bin_path, dtype="float32", mode="w+",
                   shape=(total, WINDOW_LEN, N_FEATURES))

    print("  [anomaly] Writing anomalous windows ...")
    idx = 0
    for voyages in all_voyages:
        for v in voyages:
            n = len(v)
            for start in range(0, n - WINDOW_LEN + 1, cfg.window_stride):
                fp[idx] = v[start : start + WINDOW_LEN]
                idx += 1

    del fp
    print(f"  [anomaly] Done — {idx:,} windows written to {bin_path}")
    return idx


# ── Test-only DB processing ───────────────────────────────────────────────────

def build_test_from_db(db_path: str, out_dir: str, meta_path: str) -> dict:
    """
    Process a separate database as pure test data — all MMSIs go into the
    test set with no train/val split.

    Writes test_windows.bin and (if FLAGS present) anomaly_windows.bin to
    out_dir, then updates the counts in an existing meta_path JSON.

    Called when --test-db is supplied to main(), or directly when a second
    DMA source should be held out for evaluation only.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    test_path    = str(out / "test_windows.bin")
    anomaly_path = str(out / "anomaly_windows.bin")

    print(f"\n[test-db] {db_path}")
    with sqlite3.connect(db_path) as conn:
        use_flags  = has_flags_column(conn)
        flag_note  = ("FLAGS=0 only" if cfg.clean_flags_only else "all FLAGS") if use_flags else "none"
        print(f"  FLAGS column: {'present' if use_flags else 'absent'} — using {flag_note} rows\n")
        all_mmsis = [r[0] for r in conn.execute("SELECT DISTINCT MMSI FROM ais ORDER BY MMSI")]

    print(f"  Total MMSIs (all → test): {len(all_mmsis):,}")

    n_test_w    = process_split(db_path, all_mmsis, use_flags, test_path, "test")
    print()
    n_anomaly_w = build_anomaly_set(db_path, all_mmsis, anomaly_path) if use_flags else 0

    # Merge counts into the existing meta JSON (created by run_prepare_phase).
    if Path(meta_path).exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {
            "window_len":       WINDOW_LEN,
            "n_features":       N_FEATURES,
            "feature_cols":     FEAT_COLS,
            "norm_bounds":      {k: list(v) for k, v in NORM_BOUNDS.items()},
            "seq_len_enc":      cfg.seq_len_enc,
            "seq_len_dec":      cfg.seq_len_dec,
            "n_train":          0,
            "n_val":            0,
            "ship_type_groups": {str(k): v for k, v in cfg.ship_type_groups.items()},
            "group_names":      GROUP_NAMES,
        }

    meta["n_test"]    = n_test_w
    meta["n_anomaly"] = n_anomaly_w
    meta["test_db"]   = db_path

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[test-db] meta updated at {meta_path}")
    return {"n_test": n_test_w, "n_anomaly": n_anomaly_w, "meta_path": meta_path}


# ── Public API (also called by Data_Processing.py) ───────────────────────────

def run_prepare_phase(db_path: str, out_dir: str) -> dict:
    """
    Build train/val/test/anomaly window files from a SQLite AIS database.

    Returns a dict with keys: n_train, n_val, n_test, n_anomaly, meta_path.
    Called both by this script's main() and by Data_Processing.py's Phase 3
    so that both pipelines use identical preprocessing logic.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path   = str(out / "train_windows.bin")
    val_path     = str(out / "val_windows.bin")
    test_path    = str(out / "test_windows.bin")
    anomaly_path = str(out / "anomaly_windows.bin")
    meta_path    = str(out / "dataset_meta.json")

    print(f"  Window   : {WINDOW_LEN} steps ({cfg.seq_len_enc} enc + {cfg.seq_len_dec} dec)")
    print(f"  Stride   : {cfg.window_stride}")
    print(f"  MMSI cap : {cfg.max_windows_per_mmsi} windows/vessel")
    print(f"  Group cap: {cfg.max_windows_per_type_group:,} windows/type-group\n")

    with sqlite3.connect(db_path) as conn:
        use_flags = has_flags_column(conn)
        flag_note = ("FLAGS=0 only" if cfg.clean_flags_only else "all FLAGS") if use_flags else "none"
        print(f"  FLAGS column: {'present' if use_flags else 'absent'} — using {flag_note} rows\n")
        all_mmsis = np.array([r[0] for r in conn.execute("SELECT DISTINCT MMSI FROM ais ORDER BY MMSI")])

    print(f"  Total MMSIs: {len(all_mmsis):,}")
    np.random.default_rng(seed=42).shuffle(all_mmsis)
    n_train = int(len(all_mmsis) * cfg.train_split)
    n_val   = int(len(all_mmsis) * cfg.val_split)
    train_mmsis = all_mmsis[:n_train].tolist()
    val_mmsis   = all_mmsis[n_train : n_train + n_val].tolist()
    test_mmsis  = all_mmsis[n_train + n_val :].tolist()
    print(f"  MMSI split — train:{len(train_mmsis):,}  val:{len(val_mmsis):,}  test:{len(test_mmsis):,}\n")

    n_train_w   = process_split(db_path, train_mmsis, use_flags, train_path,   "train")
    print()
    n_val_w     = process_split(db_path, val_mmsis,   use_flags, val_path,     "val")
    print()
    n_test_w    = process_split(db_path, test_mmsis,  use_flags, test_path,    "test")
    print()
    n_anomaly_w = build_anomaly_set(db_path, test_mmsis, anomaly_path) if use_flags else 0

    meta = {
        "window_len":       WINDOW_LEN,
        "n_features":       N_FEATURES,
        "feature_cols":     FEAT_COLS,
        "norm_bounds":      {k: list(v) for k, v in NORM_BOUNDS.items()},
        "seq_len_enc":      cfg.seq_len_enc,
        "seq_len_dec":      cfg.seq_len_dec,
        "n_train":          n_train_w,
        "n_val":            n_val_w,
        "n_test":           n_test_w,
        "n_anomaly":        n_anomaly_w,
        "ship_type_groups": {str(k): v for k, v in cfg.ship_type_groups.items()},
        "group_names":      GROUP_NAMES,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "n_train":   n_train_w,
        "n_val":     n_val_w,
        "n_test":    n_test_w,
        "n_anomaly": n_anomaly_w,
        "meta_path": meta_path,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-process AIS SQLite database into training window files."
    )
    parser.add_argument("--db",  default=cfg.train_db_path,
                        help="Path to the training SQLite database (default: %(default)s)")
    parser.add_argument("--out", default="data/",
                        help="Output directory for .bin files and metadata (default: %(default)s)")
    parser.add_argument("--test-db", default=None,
                        help="Path to a separate database whose MMSIs are all used as test data "
                             "(default: %(default)s — uses holdout MMSIs from --db instead). "
                             f"Config default: {cfg.test_db_path}")
    parser.add_argument("--test-out", default=None,
                        help="Output directory for test-DB windows (default: same as --out). "
                             "Pass a separate path (e.g. data2/) to keep WorldwideAIS windows "
                             "isolated from the DMA training data.")
    args = parser.parse_args()

    print(f"Database : {args.db}")
    print(f"Output   : {args.out}")
    if args.test_db:
        test_out = args.test_out or args.out
        print(f"Test DB  : {args.test_db}")
        print(f"Test out : {test_out}")

    counts = run_prepare_phase(args.db, args.out)

    if args.test_db:
        meta_path   = str(Path(args.out) / "dataset_meta.json")
        test_counts = build_test_from_db(args.test_db, test_out, meta_path)
        counts["n_test"]    = test_counts["n_test"]
        counts["n_anomaly"] = test_counts["n_anomaly"]

    print(f"\nMetadata saved to {counts['meta_path']}")
    print(f"\nSummary:")
    print(f"  train   : {counts['n_train']:,} windows")
    print(f"  val     : {counts['n_val']:,} windows")
    print(f"  test    : {counts['n_test']:,} windows")
    print(f"  anomaly : {counts['n_anomaly']:,} windows  (FLAGS != 0, for eval only)")
    print(f"  total   : {counts['n_train'] + counts['n_val'] + counts['n_test']:,} clean windows")


if __name__ == "__main__":
    main()
