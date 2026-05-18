"""
Convert_to_db.py
----------------
Converts Worldwide AIS Network JSONL.gz files into the ShipTransformer
SQLite database.

Usage — single file:
    python Convert_to_db.py --input file.jsonl.gz --output data/ais.db

Usage — whole directory (recursive):
    python Convert_to_db.py --input "C:/path/to/folder" --output data/ais.db

Optional filters:
    --bbox        lat_min lat_max lon_min lon_max
    --ship-types  70 71 72 79        (ITU codes; keep only these)
    --date-from   2026-04-01
    --date-to     2026-04-30
    --limit       N                  (process at most N files, useful for testing)

Output schema (table 'ais')
---------------------------
    MMSI       INTEGER  — ship identifier (validated: 9-digit, non-test)
    TIMESTAMP  TEXT     — YYYY-MM-DDTHH:MM:SS UTC
    LAT        REAL     — latitude
    LON        REAL     — longitude
    SOG        REAL     — speed over ground (knots)
    COG        REAL     — course over ground (degrees)
    SHIP_TYPE  INTEGER  — ITU type code (0 = unknown)
    FLAGS      INTEGER  — bitmask; 0 = clean (see FLAG_* constants below)

FLAGS bitmask
-------------
    1  SPEED_ANOMALY     SOG > 30 knots (unusual, retained for anomaly research)
    2  POSITION_JUMP     Implied speed from last known position > 60 knots
                         (strong spoofing indicator)
    4  COG_SOG_MISMATCH  SOG > 5 kts but COG is exactly 0 or 360 (likely corrupt)
    8  STATIONARY_DRIFT  SOG < 0.5 kts but vessel moved > 1 km since last ping
                         (position error or low-quality spoof)

Rows flagged here are kept in the database so they can be used as anomaly
examples.  prepare_dataset.py excludes them from clean training data unless
cfg.clean_flags_only is set to False.
"""

import argparse
import gzip
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# ── AIS sentinel values ────────────────────────────────────────────────────────
SOG_RAW_NOT_AVAILABLE    = 1023
COG_RAW_NOT_AVAILABLE    = 3600
TYPE27_SOG_NOT_AVAILABLE = 63
TYPE27_COG_NOT_AVAILABLE = 360

POSITION_MSG_TYPES = {1, 2, 3, 18, 19, 27}
STATIC_MSG_TYPES   = {5, 24}

# ── FLAGS bitmask constants ────────────────────────────────────────────────────
FLAG_SPEED_ANOMALY    = 1
FLAG_POSITION_JUMP    = 2
FLAG_COG_SOG_MISMATCH = 4
FLAG_STATIONARY_DRIFT = 8

# ── MMSI validity ──────────────────────────────────────────────────────────────
_INVALID_MMSIS = {0, 111_111_111, 123_456_789, 999_999_999}

def is_valid_mmsi(mmsi: int) -> bool:
    return 100_000_000 <= mmsi <= 999_999_999 and mmsi not in _INVALID_MMSIS

# ── Null Island check ──────────────────────────────────────────────────────────
def is_null_island(lat: float, lon: float) -> bool:
    return abs(lat) < 0.5 and abs(lon) < 0.5

# ── Geometry ───────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))

def implied_speed_kts(lat1, lon1, lat2, lon2, dt_seconds: float) -> float:
    if dt_seconds <= 0:
        return 0.0
    return haversine_km(lat1, lon1, lat2, lon2) / (dt_seconds / 3600) * 0.539957

BATCH_SIZE = 50_000


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_server_time(ts_str: str) -> tuple[str, float]:
    """Return (ISO string, unix timestamp)."""
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S"), dt.timestamp()


def collect_gz_files(input_path: str, limit=None) -> list[str]:
    p = Path(input_path)
    if p.is_file():
        files = [str(p)]
    elif p.is_dir():
        files = sorted(p.rglob("*.jsonl.gz"))
        if not files:
            raise FileNotFoundError(f"No .jsonl.gz files found under {input_path}")
        files = [str(f) for f in files]
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return files[:limit] if limit else files


def date_str_to_ts(date_str: str, end: bool = False) -> str:
    """Convert 'YYYY-MM-DD' to a sortable timestamp string."""
    suffix = "T23:59:59" if end else "T00:00:00"
    return date_str + suffix


# ── Database ───────────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous  = NORMAL;

        CREATE TABLE IF NOT EXISTS ais (
            MMSI       INTEGER NOT NULL,
            TIMESTAMP  TEXT    NOT NULL,
            LAT        REAL    NOT NULL,
            LON        REAL    NOT NULL,
            SOG        REAL    NOT NULL,
            COG        REAL    NOT NULL,
            SHIP_TYPE  INTEGER NOT NULL DEFAULT 0,
            FLAGS      INTEGER NOT NULL DEFAULT 0,
            UNIQUE (MMSI, TIMESTAMP)
        );
    """)
    conn.commit()


def create_indexes(conn: sqlite3.Connection) -> None:
    print("  Creating indexes ...")
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_mmsi      ON ais (MMSI);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON ais (TIMESTAMP);
        CREATE INDEX IF NOT EXISTS idx_shiptype  ON ais (SHIP_TYPE);
        CREATE INDEX IF NOT EXISTS idx_flags     ON ais (FLAGS);
    """)
    conn.commit()


# ── Per-file processor ─────────────────────────────────────────────────────────

def process_file(
    gz_path:       str,
    conn:          sqlite3.Connection,
    static_lookup: dict,
    last_pos:      dict,     # {mmsi: (lat, lon, unix_ts)} — persists across files
    args,
) -> tuple[int, int]:
    batch: list[tuple] = []
    pos_added    = 0
    static_added = 0

    date_from = args.date_from + "T00:00:00" if args.date_from else None
    date_to   = args.date_to   + "T23:59:59" if args.date_to   else None

    def flush(b):
        conn.executemany(
            "INSERT OR IGNORE INTO ais (MMSI, TIMESTAMP, LAT, LON, SOG, COG, FLAGS) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            b,
        )
        conn.commit()

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

            msg_type = msg.get("type")
            mmsi     = msg.get("mmsi")
            if mmsi is None:
                continue

            # ── Static messages ───────────────────────────────────────────
            if msg_type in STATIC_MSG_TYPES:
                ship_type = msg.get("shipType")
                if ship_type is not None and mmsi not in static_lookup:
                    static_lookup[mmsi] = int(ship_type)
                    static_added += 1
                continue

            if msg_type not in POSITION_MSG_TYPES:
                continue

            # ── Validate MMSI ─────────────────────────────────────────────
            if not is_valid_mmsi(int(mmsi)):
                continue

            lat    = msg.get("latitude")
            lon    = msg.get("longitude")
            ts_str = rec.get("server_time")

            if lat is None or lon is None or ts_str is None:
                continue
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                continue

            # ── Null Island ───────────────────────────────────────────────
            if is_null_island(lat, lon):
                continue

            # ── SOG / COG ─────────────────────────────────────────────────
            sog_raw = msg.get("speedOverGroundRaw")
            cog_raw = msg.get("courseOverGroundRaw")
            if sog_raw is None or cog_raw is None:
                continue

            if msg_type == 27:
                if sog_raw >= TYPE27_SOG_NOT_AVAILABLE or cog_raw >= TYPE27_COG_NOT_AVAILABLE:
                    continue
                sog, cog = float(sog_raw), float(cog_raw)
            else:
                if sog_raw == SOG_RAW_NOT_AVAILABLE or cog_raw == COG_RAW_NOT_AVAILABLE:
                    continue
                sog, cog = sog_raw / 10.0, cog_raw / 10.0

            if sog > 102 or cog > 360:
                continue

            # ── Timestamp & date filter ───────────────────────────────────
            try:
                timestamp, unix_ts = parse_server_time(ts_str)
            except (ValueError, TypeError):
                continue

            if date_from and timestamp < date_from:
                continue
            if date_to   and timestamp > date_to:
                continue

            # ── Bounding box ──────────────────────────────────────────────
            if args.bbox:
                lat_min, lat_max, lon_min, lon_max = args.bbox
                if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                    continue

            # ── Ship type filter ──────────────────────────────────────────
            if args.ship_types:
                # At this point we might not know the ship type yet (comes from
                # static messages).  Apply the filter after bulk-update instead;
                # for now we insert and the post-processing DELETE handles it.
                pass

            # ── Compute FLAGS ─────────────────────────────────────────────
            flags = 0
            mmsi_int = int(mmsi)

            if sog > 30.0:
                flags |= FLAG_SPEED_ANOMALY

            if cog in (0.0, 360.0) and sog > 5.0:
                flags |= FLAG_COG_SOG_MISMATCH

            if mmsi_int in last_pos:
                prev_lat, prev_lon, prev_ts = last_pos[mmsi_int]
                dt = unix_ts - prev_ts
                if dt > 0:
                    imp_spd = implied_speed_kts(prev_lat, prev_lon, lat, lon, dt)
                    if imp_spd > 60.0:
                        flags |= FLAG_POSITION_JUMP
                    if sog < 0.5 and haversine_km(prev_lat, prev_lon, lat, lon) > 1.0:
                        flags |= FLAG_STATIONARY_DRIFT

            last_pos[mmsi_int] = (lat, lon, unix_ts)

            # Type 19 carries embedded ship type
            if msg_type == 19:
                embedded = msg.get("shipType")
                if embedded is not None and mmsi_int not in static_lookup:
                    static_lookup[mmsi_int] = int(embedded)

            batch.append((mmsi_int, timestamp, lat, lon, round(sog, 1), round(cog, 1), flags))
            pos_added += 1

            if len(batch) >= BATCH_SIZE:
                flush(batch)
                batch.clear()

    if batch:
        flush(batch)

    return pos_added, static_added


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert AIS JSONL.gz files to a ShipTransformer SQLite database."
    )
    parser.add_argument("--input",  required=True,
                        help="Path to a single .jsonl.gz file or directory to scan recursively.")
    parser.add_argument("--output", default="data/ais.db",
                        help="Output SQLite database path (default: %(default)s).")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Process at most N files (quick test).")
    parser.add_argument("--bbox",   type=float, nargs=4,
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                        help="Only keep pings within this geographic bounding box.")
    parser.add_argument("--ship-types", type=int, nargs="+", dest="ship_types",
                        help="Keep only these ITU ship type codes, e.g. --ship-types 70 71 72 79.")
    parser.add_argument("--date-from", dest="date_from",
                        help="Keep pings on or after this date (YYYY-MM-DD).")
    parser.add_argument("--date-to",   dest="date_to",
                        help="Keep pings on or before this date (YYYY-MM-DD).")
    args = parser.parse_args()

    gz_files = collect_gz_files(args.input, args.limit)
    print(f"Found {len(gz_files)} file(s) to process")
    print(f"Output database: {args.output}")
    if args.bbox:
        print(f"Bounding box: lat [{args.bbox[0]}, {args.bbox[1]}]  "
              f"lon [{args.bbox[2]}, {args.bbox[3]}]")
    if args.date_from or args.date_to:
        print(f"Date range: {args.date_from or '(any)'} → {args.date_to or '(any)'}")
    print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    static_lookup: dict = {}
    last_pos:      dict = {}
    total_pos      = 0
    total_static   = 0
    width          = len(str(len(gz_files)))

    with sqlite3.connect(args.output) as conn:
        init_db(conn)

        for i, gz_path in enumerate(gz_files, 1):
            pos, sta = process_file(gz_path, conn, static_lookup, last_pos, args)
            total_pos    += pos
            total_static += sta
            print(f"  [{i:{width}}/{len(gz_files)}]  {os.path.basename(gz_path)}"
                  f"  pos={pos:,}  static={sta:,}")

        # ── Bulk update SHIP_TYPE ──────────────────────────────────────────
        print(f"\nUpdating SHIP_TYPE for {len(static_lookup):,} ships ...")
        conn.executemany(
            "UPDATE ais SET SHIP_TYPE = ? WHERE MMSI = ?",
            [(v, k) for k, v in static_lookup.items()],
        )

        # ── Ship type filter: remove rows for unwanted types ───────────────
        if args.ship_types:
            type_list = ", ".join(str(t) for t in args.ship_types)
            deleted = conn.execute(
                f"DELETE FROM ais WHERE SHIP_TYPE != 0 AND SHIP_TYPE NOT IN ({type_list})"
            ).rowcount
            print(f"Ship type filter: removed {deleted:,} rows not in {args.ship_types}")

        conn.commit()
        create_indexes(conn)

        row_count   = conn.execute("SELECT COUNT(*) FROM ais").fetchone()[0]
        ship_count  = conn.execute("SELECT COUNT(DISTINCT MMSI) FROM ais").fetchone()[0]
        flag_counts = conn.execute("""
            SELECT
                SUM(CASE WHEN FLAGS & 1 > 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN FLAGS & 2 > 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN FLAGS & 4 > 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN FLAGS & 8 > 0 THEN 1 ELSE 0 END)
            FROM ais
        """).fetchone()
        time_range  = conn.execute("SELECT MIN(TIMESTAMP), MAX(TIMESTAMP) FROM ais").fetchone()

    print(f"\nDone.")
    print(f"  Rows in database   : {row_count:,}")
    print(f"  Unique ships       : {ship_count:,}")
    print(f"  Time range         : {time_range[0]} → {time_range[1]}")
    print(f"\nFLAGS breakdown:")
    print(f"  SPEED_ANOMALY      : {flag_counts[0] or 0:,}")
    print(f"  POSITION_JUMP      : {flag_counts[1] or 0:,}")
    print(f"  COG_SOG_MISMATCH   : {flag_counts[2] or 0:,}")
    print(f"  STATIONARY_DRIFT   : {flag_counts[3] or 0:,}")


if __name__ == "__main__":
    main()
