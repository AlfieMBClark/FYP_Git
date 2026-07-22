"""
prepare_dataset.py
------------------
One-time preprocessing: read the AIS SQLite database, apply the track-level
filters and stratified caps from config.py, cut sliding windows, and write the
memory-mapped train/val/test/anomaly .bin files (plus a metadata JSON) that
train.py streams from.

    python prepare_dataset.py --db data/dma.db --test-db data2/2023.db

    python prepare_dataset.py --db data/dma.db --out data/ --test-db data2/2023.db --test-out data2/

MMSIs are split at the vessel level (so no ship appears in two splits), processed
in parallel, then capped serially in MMSI order for deterministic output. Raw ITU
ship-type codes are mapped to 8 semantic groups before normalisation. Re-run
whenever config.py changes.
"""

import argparse
import json
import math
import multiprocessing
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# config.py lives in the sibling ShipTransformer/ folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ShipTransformer")))
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

# Force the MMSI index on every per-vessel read. Do not remove this.
#
# Every query here filters on both MMSI and FLAGS. Left to itself SQLite plans
# `SEARCH ais USING INDEX idx_flags (FLAGS=?)` — catastrophic, because 99.8% of
# rows have FLAGS=0, so it walks ~370M index entries and fetches every row just
# to find one batch of 64 vessels. Measured on the 65 GB database:
#
#     idx_flags (planner's choice) : 744 s per batch  → ~21 h for a full run
#     idx_mmsi  (forced)           : 3.9 s per batch  → ~7 min
#
# ANALYZE does not rescue it (the planner still prefers idx_flags), so the index
# is pinned explicitly. MMSI is highly selective (~58k distinct values); FLAGS is
# not selective at all and is only ever used as a post-filter.
_BY_MMSI = "INDEXED BY idx_mmsi"

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
             f"FROM ais {_BY_MMSI} WHERE MMSI = ? AND FLAGS = 0 ORDER BY TIMESTAMP")
    else:
        q = (f"SELECT {_TS_EXPR}, LAT, LON, SOG, COG, SHIP_TYPE, ROT, HEADING, NAV_STATUS "
             f"FROM ais {_BY_MMSI} WHERE MMSI = ? ORDER BY TIMESTAMP")
    return conn.execute(q, (mmsi,)).fetchall()


# ── Track-level processing ────────────────────────────────────────────────────

_RB_LAT_MIN, _RB_LAT_MAX, _RB_LON_MIN, _RB_LON_MAX = cfg.region_bounds


def split_and_filter(rows: list[tuple]) -> list[np.ndarray]:
    """Convert raw DB rows for one MMSI into a list of normalised voyage arrays.

    Splits the track on time gaps, drops out-of-region pings and voyages that fail
    the length/speed filters, then normalises. Each array is (L, N_FEATURES) with
    L >= min_voyage_points. row[0] must be a unix timestamp (from _TS_EXPR).
    """
    if not rows:
        return []

    # Slow ship types (tugs, fishing, leisure) get a relaxed low-speed limit.
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
        # Indices 6-8: raw lat/lon/cog temporarily for delta
        # Indices 9-12: ROT, HDG_SIN, HDG_COS, NAV_STATUS.
        features.append([lat, lon, sog, math.sin(cog_rad), math.cos(cog_rad), grp,
                         lat, lon, cog,
                         rot, math.sin(hdg_rad), math.cos(hdg_rad), nav])

    if len(features) < cfg.min_voyage_points:
        return []

    ts_arr   = np.array(timestamps, dtype=np.float64)
    feat_arr = np.array(features,   dtype=np.float32)

    # Gap splitting - keep ts_arr and feat_arr in sync
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

        # dCOG:heading change in degrees per ping [-180, 180].
        dcog = np.zeros(len(seg), dtype=np.float32)
        raw_diff = np.diff(seg[:, 8])
        dcog[1:] = ((raw_diff + 180.0) % 360.0 - 180.0).astype(np.float32)

        # Final:
        # [LAT, LON, SOG, COG_SIN, COG_COS, DT, SHIP_TYPE, dLAT, dLON, dCOG, ROT, HDG_SIN, HDG_COS, NAV_STATUS]
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


def _stream_mmsi_windows(fh, voyages: list[np.ndarray], max_w: int) -> int:
    """Append up to max_w windows from voyages straight to an open file handle.

    A .bin file is nothing but raw float32 windows back to back — dataset.py
    memmaps it using the count recorded in dataset_meta.json rather than reading
    any header. So windows can simply be appended as they are produced, which is
    what lets process_split() stream instead of buffering an entire split in RAM.
    """
    written = 0
    for v in voyages:
        for start in range(0, len(v) - WINDOW_LEN + 1, cfg.window_stride):
            if written >= max_w:
                return written
            w = np.ascontiguousarray(v[start : start + WINDOW_LEN], dtype=np.float32)
            fh.write(w.tobytes())
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
             f"FROM ais {_BY_MMSI} WHERE MMSI IN ({placeholders}) AND FLAGS = 0 "
             f"ORDER BY MMSI, TIMESTAMP")
    else:
        q = (f"SELECT MMSI, {_TS_EXPR}, LAT, LON, SOG, COG, SHIP_TYPE, ROT, HEADING, NAV_STATUS "
             f"FROM ais {_BY_MMSI} WHERE MMSI IN ({placeholders}) ORDER BY MMSI, TIMESTAMP")

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
         f"FROM ais {_BY_MMSI} WHERE MMSI IN ({placeholders}) AND FLAGS != 0 "
         f"ORDER BY MMSI, TIMESTAMP")

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
    """Process MMSIs in parallel and stream their windows straight to disk.

    Results are consumed in submission order (pool.imap preserves it), so the
    stratified cap accounting stays deterministic — byte-identical output to the
    previous buffered implementation.

    Why streaming: the old version did `list(pool.imap(...))`, holding every
    ship's voyages in RAM and writing only once all of them were done. On the
    65 GB database that meant hours of total silence, no partial output, nothing
    on disk until the very end, and a memory footprint that grew with the
    dataset. Each batch is now written and released as it arrives, so memory is
    O(batch) and progress is visible.
    """
    batches     = [mmsis[i : i + _BATCH_SIZE] for i in range(0, len(mmsis), _BATCH_SIZE)]
    worker_args = [(db_path, batch, use_flags) for batch in batches]

    print(f"  [{label}] Processing {len(mmsis):,} ships in {len(batches):,} batches "
          f"({_N_WORKERS} workers, batch {_BATCH_SIZE}) ...", flush=True)

    group_remaining = {g: cfg.max_windows_per_type_group for g in range(N_GROUPS)}
    group_counts    = {g: 0 for g in range(N_GROUPS)}
    total = 0
    t0    = time.time()

    Path(bin_path).parent.mkdir(parents=True, exist_ok=True)
    with multiprocessing.Pool(processes=_N_WORKERS) as pool, open(bin_path, "wb") as fh:
        for i, batch in enumerate(pool.imap(_worker_process_mmsis, worker_args), 1):
            for voyages, tg in batch:
                n = min(
                    count_windows_for_voyages(voyages),
                    cfg.max_windows_per_mmsi,
                    group_remaining[tg],
                )
                if n <= 0:
                    continue          # this type-group is already full
                written = _stream_mmsi_windows(fh, voyages, n)
                group_remaining[tg] -= written
                group_counts[tg]    += written
                total               += written

            if i % 10 == 0 or i == len(batches):
                el  = time.time() - t0
                eta = (el / i) * (len(batches) - i) / 60
                print(f"  [{label}] batch {i:,}/{len(batches):,}  |  "
                      f"{total:,} windows  |  {el/60:.0f} min elapsed, "
                      f"~{eta:.0f} min left", flush=True)

    gb = total * WINDOW_LEN * N_FEATURES * 4 / 1e9
    print(f"  [{label}] Type-group breakdown:")
    for g, name in enumerate(GROUP_NAMES):
        cap_note = "  *** capped ***" if group_remaining[g] == 0 else ""
        print(f"             {name:16s}: {group_counts[g]:>8,}{cap_note}")

    if total == 0:
        print(f"  [{label}] WARNING: no windows produced — check filters.")
        return 0

    print(f"  [{label}] Done — {total:,} windows ({gb:.2f} GB) written to {bin_path}")
    return total


# ── Anomaly evaluation set ────────────────────────────────────────────────────

def build_anomaly_set(
    db_path:  str,
    mmsis:    list[int],
    bin_path: str,
) -> int:
    """Build an evaluation set from the FLAGS != 0 (anomalous) rows of the test MMSIs.

    Uses the same voyage filters as the clean set so windows match in shape, but
    keeps only flagged pings — for measuring anomaly detection separately.
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
    parser.add_argument("--out", default=os.path.join(cfg.data_root, "training"),
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
