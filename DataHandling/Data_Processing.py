"""
Data_Processing.py
------------------
End-to-end pipeline: converts raw AIS data from both sources into a single
SQLite database, then preprocesses it into memory-mapped training windows.

Usage — full pipeline (ingest both sources + build training windows):
    python Data_Processing.py --dma   "C:/TriAIS/DMA/downloads/2024" --jsonl "C:/TriAIS/Worldwide AIS Network" --dma-output data/dma.db --jsonl-output data/worldwide.db --out-dir data/

Usage — DMA only (no WorldwideAIS, still builds training windows):
    python Data_Processing.py \
        --dma "C:/TriAIS/DMA/downloads/2024" \
        --dma-output data/dma.db --out-dir data/

Usage — ingestion only (skip window preparation):
    python Data_Processing.py \
        --dma "C:/TriAIS/DMA/downloads/2024" \
        --dma-output data/dma.db --db-only

Usage — prepare windows only (DBs already built):
    python Data_Processing.py --prepare-only --dma-output data/dma.db --out-dir data/

Speed & size options:
    --sample-rate 1  random sample of N% of vessels (default 1.0 = keep all)
    --min-interval 300  Keep at most one ping per MMSI per N seconds 
    --workers 4         parallel file-parsing processes

Other filters (apply to both sources):
    --bbox        lat_min lat_max lon_min lon_max
    --ship-types  70 71 72 79
    --date-from   2024-01-01
    --date-to     2024-06-30
    --limit       N   (process at most N files per source — useful for testing)

Phases
------
  Phase 1 — DMA conversion    : DMA zip files → SQLite rows
  Phase 2 — JSONL conversion  : Worldwide AIS .jsonl.gz files → SQLite rows
  Phase 3 — Prepare dataset   : SQLite → stratified memory-mapped window files
"""

import argparse
import csv
import gzip
import io
import json
import math
import os
import signal
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path

# config.py lives in the sibling ShipTransformer/ folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ShipTransformer")))
from config import cfg


# ══════════════════════════════════════════════════════════════════════════════
# Shared constants & helpers
# ══════════════════════════════════════════════════════════════════════════════

FLAG_SPEED_ANOMALY    = 1
FLAG_POSITION_JUMP    = 2
FLAG_COG_SOG_MISMATCH = 4
FLAG_STATIONARY_DRIFT = 8

_INVALID_MMSIS = {0, 111_111_111, 123_456_789, 999_999_999}
BATCH_SIZE     = 100_000


def is_valid_mmsi(mmsi: int) -> bool:
    return 100_000_000 <= mmsi <= 999_999_999 and mmsi not in _INVALID_MMSIS

def is_null_island(lat: float, lon: float) -> bool:
    return abs(lat) < 0.5 and abs(lon) < 0.5

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))

def implied_speed_kts(lat1, lon1, lat2, lon2, dt_s: float) -> float:
    return 0.0 if dt_s <= 0 else haversine_km(lat1, lon1, lat2, lon2) / (dt_s / 3600) * 0.539957

def compute_flags(mmsi, lat, lon, sog, cog, unix_ts, last_pos) -> int:
    """Compute anomaly FLAGS bitmask; always updates last_pos even if row is skipped."""
    flags = 0
    if sog > 30.0:
        flags |= FLAG_SPEED_ANOMALY
    if cog in (0.0, 360.0) and sog > 5.0:
        flags |= FLAG_COG_SOG_MISMATCH
    if mmsi in last_pos:
        plat, plon, pts = last_pos[mmsi]
        dt = unix_ts - pts
        if dt > 0:
            if implied_speed_kts(plat, plon, lat, lon, dt) > 60.0:
                flags |= FLAG_POSITION_JUMP
            if sog < 0.5 and haversine_km(plat, plon, lat, lon) > 1.0:
                flags |= FLAG_STATIONARY_DRIFT
    last_pos[mmsi] = (lat, lon, unix_ts)
    return flags

def in_bbox(lat, lon, bbox) -> bool:
    if bbox is None:
        return True
    return bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]

def in_date_range(ts, date_from, date_to) -> bool:
    if date_from and ts < date_from:
        return False
    if date_to   and ts > date_to:
        return False
    return True


# ── Database ───────────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous  = OFF;
        PRAGMA cache_size   = -65536;
        PRAGMA temp_store   = MEMORY;
        PRAGMA mmap_size    = 268435456;

        CREATE TABLE IF NOT EXISTS ais (
            MMSI       INTEGER NOT NULL,
            TIMESTAMP  TEXT    NOT NULL,
            LAT        REAL    NOT NULL,
            LON        REAL    NOT NULL,
            SOG        REAL    NOT NULL,
            COG        REAL    NOT NULL,
            SHIP_TYPE  INTEGER NOT NULL DEFAULT 0,
            FLAGS      INTEGER NOT NULL DEFAULT 0,
            ROT        REAL    NOT NULL DEFAULT 0.0,
            HEADING    REAL    NOT NULL DEFAULT 0.0,
            NAV_STATUS INTEGER NOT NULL DEFAULT 0,
            UNIQUE (MMSI, TIMESTAMP)
        );
    """)
    conn.commit()


def finalise_db(conn: sqlite3.Connection) -> None:
    """Switch back to safe sync mode and build indexes after bulk loading."""
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.commit()
    print("  Building indexes ...")
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_mmsi      ON ais (MMSI);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON ais (TIMESTAMP);
        CREATE INDEX IF NOT EXISTS idx_shiptype  ON ais (SHIP_TYPE);
        CREATE INDEX IF NOT EXISTS idx_flags     ON ais (FLAGS);
    """)
    conn.commit()


def flush_batch(conn: sqlite3.Connection, batch: list) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO ais "
        "(MMSI,TIMESTAMP,LAT,LON,SOG,COG,SHIP_TYPE,FLAGS,ROT,HEADING,NAV_STATUS) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        batch,
    )
    conn.commit()


def print_db_summary(conn: sqlite3.Connection) -> None:
    rows  = conn.execute("SELECT COUNT(*) FROM ais").fetchone()[0]
    ships = conn.execute("SELECT COUNT(DISTINCT MMSI) FROM ais").fetchone()[0]
    tr    = conn.execute("SELECT MIN(TIMESTAMP), MAX(TIMESTAMP) FROM ais").fetchone()
    fc    = conn.execute("""
        SELECT SUM(FLAGS&1>0),SUM(FLAGS&2>0),SUM(FLAGS&4>0),SUM(FLAGS&8>0) FROM ais
    """).fetchone()
    print(f"\n  Rows in database   : {rows:,}")
    print(f"  Unique ships       : {ships:,}")
    print(f"  Time range         : {tr[0]} to {tr[1]}")
    print(f"\n  FLAGS breakdown:")
    print(f"    SPEED_ANOMALY    : {fc[0] or 0:,}")
    print(f"    POSITION_JUMP    : {fc[1] or 0:,}")
    print(f"    COG_SOG_MISMATCH : {fc[2] or 0:,}")
    print(f"    STATIONARY_DRIFT : {fc[3] or 0:,}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — DMA conversion
# Worker functions must be module-level for multiprocessing pickling on Windows.
# ══════════════════════════════════════════════════════════════════════════════

DMA_SHIP_TYPE_MAP = {
    "Undefined": 0,  "Reserved": 0,  "Spare 1": 0,  "DB": 0,
    "Fishing": 30,   "Towing": 31,   "Towing long/wide": 32,
    "Dredging": 33,  "Diving": 34,   "Military": 35,
    "Sailing": 36,   "Pleasure": 37, "HSC": 40,
    "Pilot": 50,     "SAR": 51,      "Tug": 52,
    "Port tender": 53, "Anti-pollution": 54, "Law enforcement": 55,
    "Medical": 58,   "Not party to conflict": 59,
    "Passenger": 60, "Cargo": 70,    "Tanker": 80, "Other": 90,
}
DMA_NAV_STATUS_MAP = {
    "Under way using engine":     0,
    "At anchor":                  1,
    "Not under command":          2,
    "Restricted manoeuvrability": 3,
    "Constrained by her draught": 4,
    "Moored":                     5,
    "Aground":                    6,
    "Engaged in fishing":         7,
    "Under way sailing":          8,
}
DMA_KEEP_MOBILE                        = {"Class A", "Class B"}
DMA_COL_TS, DMA_COL_MOB, DMA_COL_MMSI = 0, 1, 2
DMA_COL_LAT, DMA_COL_LON               = 3, 4
DMA_COL_NAV                            = 5
DMA_COL_ROT                            = 6
DMA_COL_SOG, DMA_COL_COG, DMA_COL_ST  = 7, 8, 13
DMA_COL_HDG                            = 9


def _parse_dma_ts(s: str):
    try:
        dt = datetime.strptime(s.strip(), "%d/%m/%Y %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S"), dt.timestamp()
    except ValueError:
        return None, None


def _pf(s: str):
    s = s.strip()
    if not s or s.lower() in ("unknown", "undefined", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _worker_init():
    """Workers ignore SIGINT — main process handles Ctrl+C and terminates pool."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _should_keep_mmsi(mmsi: int, sample_rate: float) -> bool:
    """Deterministic MMSI sampling — same vessel always included/excluded."""
    if sample_rate >= 1.0:
        return True
    return (mmsi % 10_000) < int(sample_rate * 10_000)


def _dma_worker(task: tuple):
    """
    Parse one DMA zip file.  Returns (rows, {}, n_inserted, n_skipped).

    last_pos is local to this worker (one file = one day, so cross-day
    position-jump detection is not needed within a single worker call).
    Temporal downsampling: only one row per MMSI per min_interval seconds
    is kept, but last_pos is updated on every ping so FLAGS stay accurate.
    """
    zip_path, min_interval, sample_rate, bbox, ship_types, date_from, date_to = task
    print(f"  >> {os.path.basename(zip_path)}", flush=True)

    rows: list      = []
    last_pos: dict  = {}   # for FLAGS
    last_ins_ts: dict = {} # for temporal downsampling
    inserted = skipped = 0

    name = os.path.basename(zip_path)
    try:
        zf_ctx = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile:
        return name, rows, {}, 0, 0

    with zf_ctx as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            return name, rows, {}, 0, 0

        for csv_name in csv_names:
            with zf.open(csv_name) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
                next(reader, None)

                for row in reader:
                    if len(row) <= DMA_COL_ST:
                        skipped += 1; continue
                    if row[DMA_COL_MOB].strip() not in DMA_KEEP_MOBILE:
                        skipped += 1; continue

                    ts, uts = _parse_dma_ts(row[DMA_COL_TS])
                    if ts is None or not in_date_range(ts, date_from, date_to):
                        skipped += 1; continue

                    try:
                        mmsi = int(row[DMA_COL_MMSI].strip())
                    except ValueError:
                        skipped += 1; continue
                    if not is_valid_mmsi(mmsi):
                        skipped += 1; continue
                    if not _should_keep_mmsi(mmsi, sample_rate):
                        skipped += 1; continue

                    lat, lon = _pf(row[DMA_COL_LAT]), _pf(row[DMA_COL_LON])
                    if lat is None or lon is None:
                        skipped += 1; continue
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        skipped += 1; continue
                    if is_null_island(lat, lon) or not in_bbox(lat, lon, bbox):
                        skipped += 1; continue

                    sog, cog = _pf(row[DMA_COL_SOG]), _pf(row[DMA_COL_COG])
                    if sog is None or cog is None:
                        skipped += 1; continue
                    if not (0 <= sog <= 102 and 0 <= cog <= 360):
                        skipped += 1; continue

                    ship_type = DMA_SHIP_TYPE_MAP.get(row[DMA_COL_ST].strip(), 0)
                    if ship_types and ship_type != 0 and ship_type not in ship_types:
                        skipped += 1; continue

                    # ROT — blank for Class B; clamp to AIS range [-127, 127]
                    rot_raw = _pf(row[DMA_COL_ROT]) if len(row) > DMA_COL_ROT else None
                    rot = max(-127.0, min(127.0, rot_raw)) if rot_raw is not None else 0.0

                    # Heading — 511 means "not available"; default to COG
                    hdg_raw = _pf(row[DMA_COL_HDG]) if len(row) > DMA_COL_HDG else None
                    heading = hdg_raw if (hdg_raw is not None and hdg_raw != 511.0 and 0.0 <= hdg_raw <= 359.0) else cog

                    # Navigational status — text field mapped to int
                    nav_str = row[DMA_COL_NAV].strip() if len(row) > DMA_COL_NAV else ""
                    nav_status = DMA_NAV_STATUS_MAP.get(nav_str, 0)

                    # FLAGS always computed (updates last_pos on every ping)
                    flags = compute_flags(mmsi, lat, lon, sog, cog, uts, last_pos)

                    # Temporal downsampling — skip insert but keep last_pos updated
                    if min_interval > 0 and uts - last_ins_ts.get(mmsi, 0) < min_interval:
                        skipped += 1; continue

                    last_ins_ts[mmsi] = uts
                    rows.append((mmsi, ts, lat, lon, round(sog, 1), round(cog, 1),
                                 ship_type, flags, rot, round(heading, 1), nav_status))
                    inserted += 1

    return os.path.basename(zip_path), rows, {}, inserted, skipped


def run_dma_phase(conn, args) -> int:
    files = []
    for dma_path in args.dma:
        p = Path(dma_path)
        if p.is_file():
            files.append(str(p))
        else:
            files.extend(sorted(str(f) for f in p.rglob("*.zip")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("  No .zip files found — skipping DMA phase.")
        return 0

    n_workers = min(args.workers, len(files))
    print(f"  Files    : {len(files)}  |  Workers: {n_workers}  |  "
          f"Sample rate: {args.sample_rate:.0%}  |  Min interval: {args.min_interval}s")
    width = len(str(len(files)))
    tasks = [(fp, args.min_interval, args.sample_rate, args.bbox, args.ship_types, args.date_from, args.date_to)
             for fp in files]

    total_ins = total_skp = 0
    pool = Pool(n_workers, initializer=_worker_init)
    try:
        for i, (name, rows, _, ins, skp) in enumerate(pool.imap_unordered(_dma_worker, tasks, chunksize=1), 1):
            print(f"  [{i:{width}}/{len(files)}]  {name}  "
                  f"parsed={ins:,}  skipped={skp:,}  inserting ...", end="", flush=True)
            if rows:
                flush_batch(conn, rows)
            total_ins += ins
            total_skp += skp
            print(f"  done")
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()   # kill workers immediately — don't wait for them to finish
        print(f"\n\n  Interrupted — {total_ins:,} DMA rows committed so far.")
        raise

    print(f"\n  DMA total: inserted={total_ins:,}  skipped={total_skp:,}")
    return total_ins


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Worldwide AIS Network (JSONL.gz) conversion
# ══════════════════════════════════════════════════════════════════════════════

JSONL_POS_TYPES    = {1, 2, 3, 18, 19, 27}
JSONL_STATIC_TYPES = {5, 24}
SOG_NA, COG_NA     = 1023, 3600
T27_SOG_NA, T27_COG_NA = 63, 360


def _parse_server_time(s: str):
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S"), dt.timestamp()


def _jsonl_worker(task: tuple):
    """
    Parse one JSONL.gz file.
    Returns (rows, static_lookup, n_pos, n_static).
    """
    gz_path, min_interval, sample_rate, bbox, ship_types, date_from, date_to = task
    print(f"  >> {os.path.basename(gz_path)}", flush=True)

    rows: list          = []
    static_lookup: dict = {}
    last_pos: dict      = {}
    last_ins_ts: dict   = {}
    pos_added = static_added = 0

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            mtype = msg.get("type")
            mmsi  = msg.get("mmsi")
            if mmsi is None:
                continue

            if mtype in JSONL_STATIC_TYPES:
                st = msg.get("shipType")
                if st is not None and mmsi not in static_lookup:
                    static_lookup[mmsi] = int(st)
                    static_added += 1
                continue

            if mtype not in JSONL_POS_TYPES:
                continue

            mmsi_int = int(mmsi)
            if not is_valid_mmsi(mmsi_int):
                continue
            if not _should_keep_mmsi(mmsi_int, sample_rate):
                continue

            lat    = msg.get("latitude")
            lon    = msg.get("longitude")
            ts_raw = rec.get("server_time")
            if lat is None or lon is None or ts_raw is None:
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            if is_null_island(lat, lon) or not in_bbox(lat, lon, bbox):
                continue

            sr = msg.get("speedOverGroundRaw")
            cr = msg.get("courseOverGroundRaw")
            if sr is None or cr is None:
                continue
            if mtype == 27:
                if sr >= T27_SOG_NA or cr >= T27_COG_NA:
                    continue
                sog, cog = float(sr), float(cr)
            else:
                if sr == SOG_NA or cr == COG_NA:
                    continue
                sog, cog = sr / 10.0, cr / 10.0
            if sog > 102 or cog > 360:
                continue

            try:
                ts, uts = _parse_server_time(ts_raw)
            except (ValueError, TypeError):
                continue
            if not in_date_range(ts, date_from, date_to):
                continue

            if mtype == 19:
                emb = msg.get("shipType")
                if emb is not None and mmsi_int not in static_lookup:
                    static_lookup[mmsi_int] = int(emb)

            if ship_types:
                st = static_lookup.get(mmsi_int, 0)
                if st != 0 and st not in ship_types:
                    continue

            flags = compute_flags(mmsi_int, lat, lon, sog, cog, uts, last_pos)

            if min_interval > 0 and uts - last_ins_ts.get(mmsi_int, 0) < min_interval:
                continue

            last_ins_ts[mmsi_int] = uts
            rows.append((mmsi_int, ts, lat, lon, round(sog, 1), round(cog, 1),
                         0, flags, 0.0, round(cog, 1), 0))
            pos_added += 1

    return os.path.basename(gz_path), rows, static_lookup, pos_added, static_added


def run_jsonl_phase(conn, args) -> int:
    p = Path(args.jsonl)
    files = [str(p)] if p.is_file() else sorted(str(f) for f in p.rglob("*.jsonl.gz"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("  No .jsonl.gz files found — skipping JSONL phase.")
        return 0

    n_workers = min(args.workers, len(files))
    print(f"  Files    : {len(files)}  |  Workers: {n_workers}  |  "
          f"Sample rate: {args.sample_rate:.0%}  |  Min interval: {args.min_interval}s")
    width = len(str(len(files)))
    tasks = [(fp, args.min_interval, args.sample_rate, args.bbox, args.ship_types, args.date_from, args.date_to)
             for fp in files]

    all_static: dict = {}
    total_pos = total_sta = 0

    pool = Pool(n_workers, initializer=_worker_init)
    try:
        for i, (name, rows, slookup, pos, sta) in enumerate(pool.imap_unordered(_jsonl_worker, tasks, chunksize=1), 1):
            print(f"  [{i:{width}}/{len(files)}]  {name}  "
                  f"pos={pos:,}  static={sta:,}  inserting ...", end="", flush=True)
            if rows:
                flush_batch(conn, rows)
            all_static.update(slookup)
            total_pos += pos
            total_sta += sta
            print(f"  done")
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()   # kill workers immediately — don't wait for them to finish
        print(f"\n\n  Interrupted — {total_pos:,} JSONL rows committed so far.")
        raise

    print(f"\n  Updating SHIP_TYPE for {len(all_static):,} ships ...")
    conn.executemany(
        "UPDATE ais SET SHIP_TYPE = ? WHERE MMSI = ? AND SHIP_TYPE = 0",
        [(v, k) for k, v in all_static.items()],
    )
    conn.commit()

    if args.ship_types:
        tl = ", ".join(str(t) for t in args.ship_types)
        deleted = conn.execute(
            f"DELETE FROM ais WHERE SHIP_TYPE != 0 AND SHIP_TYPE NOT IN ({tl})"
        ).rowcount
        conn.commit()
        print(f"  Ship type filter: removed {deleted:,} unwanted rows")

    print(f"\n  JSONL total: pos={total_pos:,}  static={total_sta:,}")
    return total_pos


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Prepare dataset  (delegated to prepare_dataset.py)
# ══════════════════════════════════════════════════════════════════════════════

from prepare_dataset import run_prepare_phase  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _banner(text: str) -> None:
    print(f"\n{'═' * 60}\n  {text}\n{'═' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "AIS pipeline: DMA → dma.db (training), WorldwideAIS → worldwide.db (testing).\n"
            "Each source is written to its own database so train and test vessels never overlap.\n\n"
            "Full pipeline:\n"
            "  python Data_Processing.py --dma <dir> --jsonl <dir>\n\n"
            "DB ingestion only (skip prepare):\n"
            "  python Data_Processing.py --dma <dir> --jsonl <dir> --db-only\n\n"
            "Prepare windows only (DBs already built):\n"
            "  python Data_Processing.py --prepare-only"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dma",   default=None, nargs="+",
                        help="One or more DMA zip directories (training source). "
                             "E.g. --dma path/2024 path/2025")
    parser.add_argument("--jsonl", default=None,
                        help="Path to WorldwideAIS .jsonl.gz directory (test source).")
    parser.add_argument("--dma-output",   default=cfg.train_db_path, dest="dma_output",
                        help=f"Output DB for DMA data (default: {cfg.train_db_path}).")
    parser.add_argument("--jsonl-output", default=cfg.test_db_path,  dest="jsonl_output",
                        help=f"Output DB for WorldwideAIS data (default: {cfg.test_db_path}).")
    parser.add_argument("--out-dir", default=os.path.join(cfg.data_root, "training"), dest="out_dir",
                        help="Output directory for training window .bin files (default: %(default)s).")
    parser.add_argument("--bbox", type=float, nargs=4,
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                        default=list(cfg.region_bounds),
                        help="Bounding box filter applied to both sources "
                             f"(default: Europe/N.Africa/Middle East {cfg.region_bounds}).")
    parser.add_argument("--ship-types", type=int, nargs="+", dest="ship_types")
    parser.add_argument("--date-from",  dest="date_from")
    parser.add_argument("--date-to",    dest="date_to")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Max files per source (for testing).")
    parser.add_argument("--sample-rate", type=float, default=1.0, dest="sample_rate",
                        help="Fraction of vessels to keep (default 1.0 = all).")
    parser.add_argument("--min-interval", type=int, default=None, dest="min_interval",
                        help="Seconds between kept pings per MMSI. "
                             "Defaults to 0 when --sample-rate < 1, else 300.")
    parser.add_argument("--workers", type=int,
                        default=max(1, cpu_count() // 2),
                        help="Parallel file-parsing processes (default: half CPU cores).")
    parser.add_argument("--db-only",      action="store_true",
                        help="Ingest to DBs only — skip prepare phase.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Build training windows from existing dma.db — skip ingestion.")
    args = parser.parse_args()

    if not args.prepare_only and args.dma is None and args.jsonl is None:
        parser.error("Provide --dma and/or --jsonl, or use --prepare-only.")

    if args.min_interval is None:
        args.min_interval = 0 if args.sample_rate < 1.0 else 300

    dma_dirs = ", ".join(args.dma) if args.dma else "none"
    print(f"DMA sources  : {dma_dirs}")
    print(f"Train DB     : {args.dma_output}  (DMA → training windows)")
    print(f"Test DB      : {args.jsonl_output}  (WorldwideAIS → predict.py evaluation)")
    print(f"Out dir      : {args.out_dir}")
    print(f"Sample rate  : {args.sample_rate:.0%} of vessels")
    print(f"Min interval : {args.min_interval}s per MMSI")
    print(f"Workers      : {args.workers}")
    if args.bbox:       print(f"BBox         : lat {args.bbox[0]}-{args.bbox[1]}  lon {args.bbox[2]}-{args.bbox[3]}")
    if args.ship_types: print(f"Ship types   : {args.ship_types}")
    if args.date_from:  print(f"Date from    : {args.date_from}")
    if args.date_to:    print(f"Date to      : {args.date_to}")
    if args.limit:      print(f"Limit        : {args.limit} files/source")

    # ── Phase 1: DMA → dma.db ─────────────────────────────────────────────────
    if args.dma and not args.prepare_only:
        _banner("Phase 1 — DMA Conversion  →  training DB")
        os.makedirs(os.path.dirname(args.dma_output) or ".", exist_ok=True)
        with sqlite3.connect(args.dma_output) as conn:
            init_db(conn)
            run_dma_phase(conn, args)
            finalise_db(conn)
            print_db_summary(conn)

    # ── Phase 2: WorldwideAIS → worldwide.db ──────────────────────────────────
    if args.jsonl and not args.prepare_only:
        _banner("Phase 2 — Worldwide AIS Conversion  →  test DB")
        os.makedirs(os.path.dirname(args.jsonl_output) or ".", exist_ok=True)
        with sqlite3.connect(args.jsonl_output) as conn:
            init_db(conn)
            run_jsonl_phase(conn, args)
            finalise_db(conn)
            print_db_summary(conn)

    if args.db_only:
        print("\nDone (--db-only: skipping prepare phase).")
        return

    # ── Phase 3: build training windows from DMA DB ───────────────────────────
    _banner("Phase 3 — Prepare Training Windows  (source: DMA DB)")
    counts = run_prepare_phase(args.dma_output, args.out_dir)

    print(f"\n{'═' * 60}\n  Pipeline complete.\n{'═' * 60}")
    print(f"  train   : {counts['n_train']:,} windows  ← from {args.dma_output}")
    print(f"  val     : {counts['n_val']:,} windows  ← from {args.dma_output}")
    print(f"  test    : {counts['n_test']:,} windows  ← from {args.dma_output} (DMA holdout)")
    print(f"  anomaly : {counts['n_anomaly']:,} windows  (FLAGS != 0, eval only)")
    print(f"  total   : {counts['n_train'] + counts['n_val'] + counts['n_test']:,} clean windows")
    print(f"  meta    : {counts['meta_path']}")
    print(f"\n  Real evaluation → python predict.py --db {args.jsonl_output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutdown complete. Any rows inserted before the interrupt are saved.")
        sys.exit(0)
