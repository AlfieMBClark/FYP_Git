"""
build_eval_set.py
-----------------
Usage
-----
python build_eval_set.py                       # 600 vessels, <=6 windows each
python build_eval_set.py --n-vessels 1000 --max-windows-per-vessel 4
"""

import argparse
import os
import random
import sqlite3
import sys
import time

import numpy as np

from common import (
    _TRANSFORMER, _DATAHANDLING, GROUP_NAMES, SEQ_ENC, WINDOW_LEN, group_of, denormalise,
    IDX_SOG, denorm_latlon, haversine_km,
)

sys.path.insert(0, _TRANSFORMER)
sys.path.insert(0, _DATAHANDLING)
from config import cfg                                     
from prepare_dataset import (                              
    fetch_mmsi_track, split_and_filter, has_flags_column,
)

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.npz")


def train_mmsis(train_db: str) -> set:
    """Every MMSI in the training database — used to mark test vessels as seen."""
    if not os.path.exists(train_db):
        print(f"  ! training DB not found ({train_db}) — seen/unseen split disabled")
        return set()
    t0 = time.time()
    con = sqlite3.connect(train_db)
    s = {r[0] for r in con.execute("SELECT DISTINCT MMSI FROM ais")}
    con.close()
    print(f"  Training vessels: {len(s):,}  ({time.time() - t0:.0f}s)")
    return s


def main():
    ap = argparse.ArgumentParser(description="Build a cached evaluation window set.")
    ap.add_argument("--db",       default=os.path.join(_TRANSFORMER, cfg.test_db_path),
                    help="Held-out AIS database (default: the 2023 hold-out).")
    ap.add_argument("--train-db", default=os.path.join(_TRANSFORMER, cfg.train_db_path),
                    help="Training database — only read to tag vessels as seen/unseen.")
    ap.add_argument("--n-vessels", type=int, default=600)
    ap.add_argument("--max-windows-per-vessel", type=int, default=6,
                    help="Cap per vessel so a few chatty ships can't dominate the set.")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--out",      default=DEFAULT_OUT)
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"Database not found: {args.db}")

    rng = random.Random(args.seed)
    print(f"\n  Test DB : {args.db}")
    seen = train_mmsis(args.train_db)

    con = sqlite3.connect(args.db)
    con.execute("PRAGMA cache_size=-64000")
    con.execute("PRAGMA mmap_size=4294967296")
    use_flags = has_flags_column(con)

    all_mmsis = [r[0] for r in con.execute("SELECT DISTINCT MMSI FROM ais")]
    rng.shuffle(all_mmsis)
    print(f"  Test vessels: {len(all_mmsis):,}  "
          f"({sum(m in seen for m in all_mmsis):,} also in training)")

    windows, mmsis = [], []
    kept_vessels = 0
    t0 = time.time()

    for mmsi in all_mmsis:
        if kept_vessels >= args.n_vessels:
            break
        rows = fetch_mmsi_track(con, mmsi, use_flags)
        if len(rows) < WINDOW_LEN:
            continue
        voyages = split_and_filter(rows)     # same cleaning as the training set
        cand = []
        for v in voyages:
            for start in range(0, len(v) - WINDOW_LEN + 1, cfg.window_stride):
                cand.append(v[start : start + WINDOW_LEN])
        if not cand:
            continue
        rng.shuffle(cand)
        for w in cand[: args.max_windows_per_vessel]:
            windows.append(w.astype(np.float32))
            mmsis.append(mmsi)
        kept_vessels += 1
        if kept_vessels % 100 == 0:
            print(f"    {kept_vessels}/{args.n_vessels} vessels  "
                  f"({len(windows):,} windows, {time.time() - t0:.0f}s)", flush=True)

    con.close()

    if not windows:
        raise SystemExit("No usable windows found — check the database and filters.")

    W     = np.stack(windows)                       # (N, 100, 14) normalised
    mmsi  = np.asarray(mmsis, dtype=np.int64)
    group = np.asarray([group_of(w) for w in W], dtype=np.int8)
    seen_f = np.asarray([m in seen for m in mmsi], dtype=bool)

    phys   = np.stack([denormalise(w) for w in W])
    fut    = phys[:, SEQ_ENC:, :]
    sog    = fut[:, :, IDX_SOG].mean(axis=1)
    lat, lon = denorm_latlon(W[:, :, :2])
    disp   = haversine_km(lat[:, SEQ_ENC - 1], lon[:, SEQ_ENC - 1],
                          lat[:, -1],          lon[:, -1])

    np.savez_compressed(
        args.out,
        windows=W, mmsi=mmsi, group=group, seen=seen_f,
        future_mean_sog=sog.astype(np.float32),
        future_disp_km=disp.astype(np.float32),
        seed=args.seed, db=args.db,
    )

    print(f"\n  ── Evaluation set ──")
    print(f"  Windows        : {len(W):,} from {kept_vessels:,} vessels")
    print(f"  Seen in train  : {seen_f.sum():,} windows "
          f"({100 * seen_f.mean():.1f}%)   Unseen: {(~seen_f).sum():,}")
    print(f"  Future 10-step displacement: mean {disp.mean():.2f} km, "
          f"median {np.median(disp):.2f} km")
    print(f"  Future mean SOG            : mean {sog.mean():.1f} kn, "
          f"median {np.median(sog):.1f} kn")
    print("  Ship-type groups:")
    for g in range(len(GROUP_NAMES)):
        n = int((group == g).sum())
        if n:
            print(f"    {GROUP_NAMES[g]:<14} {n:>6,}  ({100 * n / len(W):.1f}%)")
    print(f"\n  Saved -> {args.out}\n")


if __name__ == "__main__":
    main()
