"""
filter_danish_waters.py
-----------------------
Filters an existing AIS SQLite database to only retain pings within
Danish waters (North Sea coast, Kattegat, Danish Straits, Baltic entrance).

Operates in-place on the database — make a backup first if needed.

Usage:
    python filter_danish_waters.py
    python filter_danish_waters.py --db data/worldwide.db
    python filter_danish_waters.py --db data/worldwide.db --min-pings 70
"""

import argparse
import sqlite3
import os

# ── Danish waters bounding box ────────────────────────────────────────────────
# Covers: Jutland, Funen, Zealand, Bornholm, Kattegat, Danish Straits,
#         nearby North Sea and Baltic Sea entrance.
LAT_MIN = 54.0
LAT_MAX = 58.5
LON_MIN =  7.5
LON_MAX = 16.0


def main():
    parser = argparse.ArgumentParser(
        description="Filter AIS database to Danish waters only (in-place)."
    )
    parser.add_argument("--db", default="data/worldwide.db",
                        help="Path to the AIS SQLite database (default: data/worldwide.db).")
    parser.add_argument("--min-pings", type=int, default=70, dest="min_pings",
                        help="Remove any MMSI with fewer than this many pings remaining "
                             "after the geographic filter (default: 70 = one full window).")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        return

    print(f"Database     : {args.db}")
    print(f"Region       : lat {LAT_MIN}–{LAT_MAX}  lon {LON_MIN}–{LON_MAX}")
    print(f"Min pings    : {args.min_pings}")
    print()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous  = OFF")
    conn.execute("PRAGMA cache_size   = -65536")

    # ── Before stats ──────────────────────────────────────────────────────────
    rows_before  = conn.execute("SELECT COUNT(*) FROM ais").fetchone()[0]
    ships_before = conn.execute("SELECT COUNT(DISTINCT MMSI) FROM ais").fetchone()[0]
    print(f"Before  :  {rows_before:>12,} pings   {ships_before:>8,} ships")

    # ── Step 1: delete pings outside Danish waters ────────────────────────────
    print("Step 1  : removing out-of-region pings ...", end="", flush=True)
    cur = conn.execute(
        "DELETE FROM ais WHERE LAT NOT BETWEEN ? AND ? OR LON NOT BETWEEN ? AND ?",
        (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX),
    )
    conn.commit()
    print(f"  removed {cur.rowcount:,} pings")

    # ── Step 2: drop MMSIs with too few remaining pings ───────────────────────
    print("Step 2  : removing ships with too few pings ...", end="", flush=True)
    cur = conn.execute(
        """
        DELETE FROM ais WHERE MMSI IN (
            SELECT MMSI FROM ais
            GROUP BY MMSI
            HAVING COUNT(*) < ?
        )
        """,
        (args.min_pings,),
    )
    conn.commit()
    print(f"  removed {cur.rowcount:,} pings")

    # ── After stats ───────────────────────────────────────────────────────────
    rows_after  = conn.execute("SELECT COUNT(*) FROM ais").fetchone()[0]
    ships_after = conn.execute("SELECT COUNT(DISTINCT MMSI) FROM ais").fetchone()[0]
    tr = conn.execute("SELECT MIN(TIMESTAMP), MAX(TIMESTAMP) FROM ais").fetchone()
    print()
    print(f"After   :  {rows_after:>12,} pings   {ships_after:>8,} ships")
    print(f"Removed :  {rows_before - rows_after:>12,} pings   {ships_before - ships_after:>8,} ships")
    if tr[0]:
        print(f"Time range : {tr[0]} to {tr[1]}")

    # ── Reclaim disk space ────────────────────────────────────────────────────
    size_before = os.path.getsize(args.db)
    print(f"\nVACUUM  : reclaiming disk space ...", end="", flush=True)
    conn.execute("VACUUM")
    conn.close()
    size_after = os.path.getsize(args.db)
    print(f"  {size_before / 1e9:.2f} GB → {size_after / 1e9:.2f} GB")
    print("\nDone.")


if __name__ == "__main__":
    main()
