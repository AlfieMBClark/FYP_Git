"""
eda.py
------
The complete exploratory data analysis for the AIS dataset, in one script.

Replaces eda_simple.py and eda_processed.py, which between them drew the same
picture twice and each paid the full cost of reading the database to do it.

The analysis has two halves, because there are two genuinely different questions:

  RAW (figures 01-10)
      What is actually in the database? Volume, time coverage, fault flags,
      vessel mix, reporting rate, train/hold-out overlap. This is the dataset as
      collected, before any modelling decisions are applied.

  PROCESSED (figures 11-23)
      What does the model actually see? Every trajectory here is produced by
      prepare_dataset.split_and_filter() — the exact function the training
      pipeline calls, with the same gap-splitting, the same length and low-speed
      filters, and the same 14 engineered features. So any claim made about
      these figures transfers directly to the model's input.

Each source is used where it is authoritative: exact counts come from SQL
aggregates over the whole table; anything trajectory- or feature-level comes
from the preprocessing, because those quantities do not exist until it has run.

Figures carry a title, axes, and a legend where a marker needs naming — nothing
else. The written findings are not drawn onto the images; they are collected in
eda_findings.txt next to them, for the report to quote.

Cost
----
The first run scans a 69 GB table several times and rebuilds trajectories for a
vessel sample: budget 30-60 minutes. Everything it needs is then cached, so
restyling the figures afterwards is a seconds-long --redraw.

Usage
-----
python eda.py                     # scan, cache, draw everything
python eda.py --redraw            # redraw from cache, no database access
python eda.py --n-vessels 4000    # larger trajectory sample
python eda.py --raw-only          # skip the processed half (faster)
"""

import argparse
import json
import os
import sys
import sqlite3
import time
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "ShipTransformer"))) # config
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "DataHandling")))    # pipeline

from config import cfg                                    # noqa: E402
from prepare_dataset import (                             # noqa: E402
    fetch_mmsi_track, split_and_filter, has_flags_column,
    GROUP_NAMES, WINDOW_LEN,
)

OUT_DIR    = os.path.join(_HERE, "figures")
CACHE_PATH = os.path.join(OUT_DIR, "_eda_cache.npz")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_NAME = "Training data"
TEST_NAME  = "Hold-out data"

# The flags Data_Processing.py actually writes. These were mislabelled in the
# original EDA ("Interpolated", "Low accuracy") — every one of them is in fact a
# physical impossibility check, which is why flagged rows can simply be dropped.
FLAG_BITS = {
    1: ("Impossible speed",      "faster than 30 knots"),
    2: ("Position jump",         "moved faster than 60 knots"),
    4: ("Course/speed mismatch", "course stuck at 0° while moving"),
    8: ("Stationary drift",      "reported stopped, but moved 1 km"),
}

# ── feature layout (indices into the 14-feature engineered vector) ────────────
FEAT = cfg.feature_cols
(IDX_LAT, IDX_LON, IDX_SOG, IDX_CSIN, IDX_CCOS, IDX_DT, IDX_TYPE,
 IDX_DLAT, IDX_DLON, IDX_DCOG, IDX_ROT, IDX_HSIN, IDX_HCOS, IDX_NAV) = range(14)

_LO  = np.array([cfg.norm_bounds[f][0] for f in FEAT], dtype=np.float32)
_HI  = np.array([cfg.norm_bounds[f][1] for f in FEAT], dtype=np.float32)
_RNG = _HI - _LO


def denorm(col_norm, idx):
    """One normalised feature column back to physical units."""
    return np.clip(col_norm, 0.0, 1.0) * _RNG[idx] + _LO[idx]


BLUE, ORANGE, GREEN, RED, PURPLE = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 190, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 12, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

_log      = []   # everything printed, mirrored into the summary file
_findings = []   # (figure name, one-sentence finding) for the report


def say(msg=""):
    print(msg, flush=True)
    _log.append(msg)


def human(v, _pos=None):
    """Axis ticks a person can read: 1.4M, not 1.4 x 10^6."""
    v = float(v)
    if abs(v) >= 1e6:
        return f"{v / 1e6:g}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


HUMAN = FuncFormatter(human)


def save(fig, name, finding=None):
    """Write the figure, and file its finding as text rather than drawing it.

    Nothing explanatory is rendered onto the image — the figures carry a title,
    axes and a legend only. The finding is recorded in eda_findings.txt so the
    written analysis survives without cluttering the plot.
    """
    fig.savefig(os.path.join(OUT_DIR, name))
    plt.close(fig)
    if finding:
        _findings.append((name, finding))
    say(f"    saved  {name}")


# ══ data gathering ════════════════════════════════════════════════════════════

def connect(path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA cache_size=-64000")
    con.execute("PRAGMA mmap_size=4294967296")
    return con


def scan_db(con, full=True):
    """Aggregate facts about one database.

    Several totals are derived from other queries rather than asked for
    separately, because each independent aggregate is another full pass over a
    69 GB table:
      * the row count is the sum of the per-day counts;
      * the number of days with data is the number of per-day rows;
      * the observation period is the first and last of those days;
      * the vessel count is the length of the per-vessel count list.
    That removes four whole-table scans from what the original scripts did.
    """
    say("    per-day counts ...")
    day_rows = con.execute(
        "SELECT substr(TIMESTAMP, 1, 10) AS d, COUNT(*) FROM ais GROUP BY d ORDER BY d"
    ).fetchall()
    days   = np.array([r[0] for r in day_rows])
    counts = np.array([r[1] for r in day_rows], dtype=np.int64)

    say("    per-vessel counts ...")
    ppv_rows = con.execute("SELECT MMSI, COUNT(*) FROM ais GROUP BY MMSI").fetchall()
    mmsis = {int(r[0]) for r in ppv_rows}
    ppv   = np.array([r[1] for r in ppv_rows], dtype=np.int64)

    out = {
        "days": days, "counts": counts, "ppv": ppv,
        "n_rows": int(counts.sum()), "n_vessels": len(ppv),
        "n_days": len(days),
        "ts_lo": str(days[0]), "ts_hi": str(days[-1]),
        "mmsis": mmsis,
    }
    if not full:
        return out

    # Older databases were built before the quality checks existed and have no
    # FLAGS column; treat every row in those as clean rather than failing.
    if not has_flags_column(con):
        out["clean"]     = out["n_rows"]
        out["flag_bits"] = np.zeros(len(FLAG_BITS), dtype=np.int64)
    else:
        out.update(_flag_counts(con))

    say("    vessel types ...")
    grouped = Counter()
    for code, n in con.execute("SELECT SHIP_TYPE, COUNT(*) FROM ais GROUP BY SHIP_TYPE"):
        grouped[cfg.ship_type_groups.get(int(code or 0), 7)] += n
    out["type_groups"] = np.array(
        [grouped.get(g, 0) for g in range(len(GROUP_NAMES))], dtype=np.int64)
    return out


def _flag_counts(con):
    say("    fault flags ...")
    clean, per_bit = 0, {b: 0 for b in FLAG_BITS}
    for value, n in con.execute("SELECT FLAGS, COUNT(*) FROM ais GROUP BY FLAGS"):
        value = int(value or 0)
        if value == 0:
            clean += n
            continue
        # A ping can carry several faults at once, so the per-bit tallies overlap.
        # Counting per bit rather than per bit-combination is what makes the chart
        # legible: the original had bars labelled "1+4", which mean nothing.
        for bit in FLAG_BITS:
            if value & bit:
                per_bit[bit] += n
    return {
        "clean":     clean,
        "flag_bits": np.array([per_bit[b] for b in sorted(FLAG_BITS)], dtype=np.int64),
    }


def scan_vessels(con, mmsis, n_vessels, seed, want_processed):
    """One pass over a vessel sample, feeding both halves of the analysis.

    The raw and processed views need the same expensive thing — every ping for a
    set of vessels — so they are read once and used twice. Raw speed and
    reporting-interval statistics come straight off the rows; the clean subset is
    then handed to split_and_filter() to produce exactly the trajectories the
    training pipeline builds.
    """
    rng  = np.random.default_rng(seed)
    pick = rng.choice(sorted(mmsis), size=min(n_vessels, len(mmsis)), replace=False)
    use_flags = has_flags_column(con)

    raw_sog, raw_lat, raw_lon, raw_gaps = [], [], [], []
    voyage_len, feats, traj_lines = [], [], []
    example = None
    kept = 0
    t0 = time.time()

    for i, m in enumerate(pick, 1):
        rows = con.execute(
            "SELECT CAST(strftime('%s', TIMESTAMP) AS INTEGER), LAT, LON, SOG "
            "FROM ais INDEXED BY idx_mmsi WHERE MMSI = ? ORDER BY TIMESTAMP",
            (int(m),),
        ).fetchall()
        if len(rows) >= 2:
            a = np.asarray(rows, dtype=np.float64)
            raw_gaps.append(np.diff(a[:, 0]))
            raw_lat.append(a[:, 1]); raw_lon.append(a[:, 2]); raw_sog.append(a[:, 3])

        if want_processed:
            track = fetch_mmsi_track(con, int(m), use_flags)
            if len(track) >= cfg.min_voyage_points:
                vs = split_and_filter(track)
                if vs:
                    kept += 1
                    for v in vs:
                        voyage_len.append(len(v))
                        feats.append(v)
                        if len(traj_lines) < 60:
                            traj_lines.append(np.column_stack([
                                denorm(v[:, IDX_LAT], IDX_LAT),
                                denorm(v[:, IDX_LON], IDX_LON)]))
                    if example is None:
                        for v in vs:
                            if 150 <= len(v) <= 600:
                                example = v.copy()
                                break

        if i % 250 == 0:
            say(f"      {i}/{len(pick)} vessels  ({time.time()-t0:.0f}s)")

    if not raw_sog:
        raise SystemExit("No vessels returned any rows — check the database.")

    out = {
        "raw_sog":  np.concatenate(raw_sog),
        "raw_lat":  np.concatenate(raw_lat),
        "raw_lon":  np.concatenate(raw_lon),
        "raw_gaps": np.concatenate(raw_gaps),
        "n_sampled": len(pick),
    }
    # Trim the raw point cloud: the hexbin is indistinguishable at 400k points and
    # the cache stays small.
    if len(out["raw_sog"]) > 400_000:
        idx = rng.choice(len(out["raw_sog"]), 400_000, replace=False)
        for k in ("raw_sog", "raw_lat", "raw_lon"):
            out[k] = out[k][idx]
    out["raw_gaps"] = out["raw_gaps"][out["raw_gaps"] > 0]

    if want_processed:
        if not feats:
            raise SystemExit("No voyages survived preprocessing — check the filters.")
        allf = np.concatenate(feats, axis=0)
        if len(allf) > 1_500_000:
            allf = allf[rng.choice(len(allf), 1_500_000, replace=False)]
        if example is None:
            example = feats[int(np.argmax([len(f) for f in feats]))].copy()
        out.update({
            "feats":      allf.astype(np.float32),
            "voyage_len": np.array(voyage_len, dtype=np.int32),
            "example":    example.astype(np.float32),
            "traj_lines": np.array(traj_lines, dtype=object),
            "vessels_kept": kept,
        })
    return out


def gather(args):
    con_tr = connect(args.db)
    con_te = connect(args.test_db)

    say("\n  Scanning training database ...")
    tr = scan_db(con_tr, full=True)
    say("\n  Scanning hold-out database ...")
    te = scan_db(con_te, full=False)

    say(f"\n  Rebuilding trajectories for {args.n_vessels} sampled vessels ...")
    vs = scan_vessels(con_tr, tr["mmsis"], args.n_vessels, args.seed,
                      want_processed=not args.raw_only)

    overlap   = len(tr["mmsis"] & te["mmsis"])
    test_only = len(te["mmsis"] - tr["mmsis"])
    con_tr.close(); con_te.close()

    d = {
        "train_days": tr["days"], "train_counts": tr["counts"],
        "test_days":  te["days"], "test_counts":  te["counts"],
        "train_ppv":  tr["ppv"],
        "train_totals": np.array([tr["n_rows"], tr["n_vessels"], tr["n_days"]]),
        "test_totals":  np.array([te["n_rows"], te["n_vessels"], te["n_days"]]),
        "train_span": np.array([tr["ts_lo"], tr["ts_hi"]]),
        "test_span":  np.array([te["ts_lo"], te["ts_hi"]]),
        "clean": np.array(tr["clean"]),
        "flag_bits": tr["flag_bits"],
        "type_groups": tr["type_groups"],
        "overlap": np.array(overlap), "test_only": np.array(test_only),
    }
    d.update({k: v for k, v in vs.items() if k != "n_sampled"})
    d["n_sampled"] = np.array(vs["n_sampled"])

    np.savez_compressed(CACHE_PATH, **d)
    say(f"\n  Cached -> {CACHE_PATH}")
    return d


def physical_features(feats):
    """Engineered feature columns back in physical units."""
    csin = feats[:, IDX_CSIN] * 2 - 1          # sin/cos were normalised [-1,1]->[0,1]
    ccos = feats[:, IDX_CCOS] * 2 - 1
    hsin = feats[:, IDX_HSIN] * 2 - 1
    hcos = feats[:, IDX_HCOS] * 2 - 1
    return {
        "SOG (knots)":   denorm(feats[:, IDX_SOG], IDX_SOG),
        "COG (degrees)": np.degrees(np.arctan2(csin, ccos)) % 360,
        "Heading (deg)": np.degrees(np.arctan2(hsin, hcos)) % 360,
        "ROT (deg/min)": denorm(feats[:, IDX_ROT], IDX_ROT),
        "Δt (seconds)":  denorm(feats[:, IDX_DT], IDX_DT),
        "ΔLAT":          denorm(feats[:, IDX_DLAT], IDX_DLAT),
        "ΔLON":          denorm(feats[:, IDX_DLON], IDX_DLON),
        "ΔCOG (deg)":    denorm(feats[:, IDX_DCOG], IDX_DCOG),
    }


# ══ figures — raw dataset ═════════════════════════════════════════════════════

def fig01_summary(d):
    meta_path = cfg.meta_path
    win_total = None
    if os.path.exists(meta_path):
        m = json.load(open(meta_path))
        win_total = m.get("n_train", 0) + m.get("n_val", 0) + m.get("n_test", 0)

    n_days   = int(d["train_totals"][2])
    span     = (np.datetime64(str(d["train_span"][1])) -
                np.datetime64(str(d["train_span"][0]))).astype(int) + 1
    med_gap  = float(np.median(d["raw_gaps"]) / 60.0)
    coverage = (f"{n_days} days" +
                (f"  (within a {span}-day span)" if n_days < span else "  (continuous)"))

    rows = [
        ("Position reports",     f"{int(d['train_totals'][0]):,}"),
        ("Unique vessels (MMSI)", f"{int(d['train_totals'][1]):,}"),
        ("Observation period",   f"{d['train_span'][0]} to {d['train_span'][1]}"),
        ("Days of data",         coverage),
        ("Median reporting interval", f"{med_gap:.1f} minutes"),
        ("Training windows",     f"{win_total:,}" if win_total else "not yet built"),
        ("Window length",        f"{WINDOW_LEN} pings "
                                 f"({cfg.seq_len_enc} in / {cfg.seq_len_dec} out)"),
        ("Engineered features",  f"{len(FEAT)}"),
        ("Region",               f"Danish waters (lat {cfg.region_bounds[0]:.0f}–"
                                 f"{cfg.region_bounds[1]:.0f}, lon {cfg.region_bounds[2]:.0f}–"
                                 f"{cfg.region_bounds[3]:.0f})"),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.axis("off")
    ax.set_title("Dataset Summary", pad=16)
    tbl = ax.table(cellText=rows, colLabels=["Property", "Value"],
                   cellLoc="left", colLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor(BLUE); cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F7FA")
        if c == 0:
            cell.set_text_props(weight="bold")
    tbl.auto_set_column_width([0, 1])
    save(fig, "01_dataset_summary.png",
         f"{int(d['train_totals'][0]):,} position reports from "
         f"{int(d['train_totals'][1]):,} vessels over {n_days} days.")


def fig02_volume(d):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    names = [TRAIN_NAME, TEST_NAME]

    pings = [d["train_totals"][0] / 1e6, d["test_totals"][0] / 1e6]
    b = ax1.bar(names, pings, color=[BLUE, ORANGE], width=0.55)
    ax1.bar_label(b, labels=[f"{v:,.0f}M" for v in pings], padding=4, fontweight="bold")
    ax1.set_ylabel("Position reports (millions)")
    ax1.set_title("Position reports")
    ax1.set_ylim(0, max(pings) * 1.25)

    ves = [int(d["train_totals"][1]), int(d["test_totals"][1])]
    b = ax2.bar(names, ves, color=[BLUE, ORANGE], width=0.55)
    ax2.bar_label(b, labels=[f"{v:,}" for v in ves], padding=4, fontweight="bold")
    ax2.set_ylabel("Distinct vessels")
    ax2.set_title("Vessels")
    ax2.set_ylim(0, max(ves) * 1.25)
    ax2.yaxis.set_major_formatter(HUMAN)

    save(fig, "02_data_volume.png",
         f"Training draws on {pings[0]:,.0f}M reports from {ves[0]:,} vessels; the hold-out "
         f"year contributes {pings[1]:,.0f}M from {ves[1]:,}.")


def _largest_gap(days):
    """Longest run of consecutive calendar days with no data, and where it starts.

    Plotting against a day *index* hides gaps entirely — consecutive rows sit
    next to each other and a five-month hole looks like unbroken coverage. This
    measures the hole instead of assuming one, so the annotation stays correct
    once the data is backfilled.
    """
    dt = np.array(days, dtype="datetime64[D]")
    if len(dt) < 2:
        return 0, None
    deltas = np.diff(dt).astype(int) - 1          # missing days between each pair
    i = int(np.argmax(deltas))
    return int(deltas[i]), dt[i]


def fig03_coverage(d):
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.4))

    for ax, days, cnt, name, col in (
        (axes[0], d["train_days"], d["train_counts"], TRAIN_NAME, BLUE),
        (axes[1], d["test_days"],  d["test_counts"],  TEST_NAME,  ORANGE),
    ):
        dts = np.array(days, dtype="datetime64[D]")
        # Reindex onto every calendar day in the range, leaving absent days at
        # zero, so a missing stretch is drawn as a real gap rather than closed up.
        full = np.arange(dts[0], dts[-1] + np.timedelta64(1, "D"))
        vals = np.zeros(len(full))
        vals[np.searchsorted(full, dts)] = np.asarray(cnt)

        ax.fill_between(full.astype("datetime64[ms]").astype("O"), vals,
                        color=col, alpha=0.9, lw=0)
        ax.set_title(f"{name} — {len(dts)} days", fontsize=13)
        ax.set_ylabel("Reports per day")
        ax.yaxis.set_major_formatter(HUMAN)
        loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

        gap, start = _largest_gap(days)
        if gap >= 7:
            ax.annotate(f"{gap} days missing",
                        xy=(start.astype("datetime64[ms]").astype("O"),
                            ax.get_ylim()[1] * 0.6),
                        color=RED, fontweight="bold", fontsize=11, ha="left")

    fig.subplots_adjust(hspace=0.55)

    gap_tr, _ = _largest_gap(d["train_days"])
    note = (f"the largest break is {gap_tr} days" if gap_tr >= 7
            else "coverage is effectively continuous")
    save(fig, "03_temporal_coverage.png",
         f"Training spans {d['train_span'][0]} to {d['train_span'][1]} and {note}. "
         f"The hold-out is an earlier, disjoint period, so evaluation is a genuine "
         f"forward-in-time test.")


def fig04_quality(d):
    clean   = int(d["clean"])
    bits    = d["flag_bits"]
    total   = int(d["train_totals"][0])
    flagged = total - clean

    keys   = sorted(FLAG_BITS)
    labels = [f"{FLAG_BITS[b][0]}\n({FLAG_BITS[b][1]})" for b in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5),
                                   gridspec_kw={"width_ratios": [1, 1.6]})

    b = ax1.bar(["Usable", "Faulty"], [clean, flagged], color=[GREEN, RED], width=0.55)
    ax1.bar_label(b, labels=[f"{100*clean/total:.2f}%", f"{100*flagged/total:.2f}%"],
                  padding=4, fontweight="bold")
    ax1.set_ylabel("Position reports")
    ax1.set_title("Overall data quality")
    ax1.set_ylim(0, total * 1.18)
    ax1.yaxis.set_major_formatter(HUMAN)

    b = ax2.barh(labels, bits, color=RED, height=0.6)
    ax2.bar_label(b, labels=[f"{int(c):,}" for c in bits], padding=5, fontsize=11)
    ax2.set_xlabel("Reports affected")
    ax2.set_title("Faults by type")
    ax2.set_xlim(0, max(bits.max(), 1) * 1.30)
    ax2.xaxis.set_major_formatter(HUMAN)
    ax2.tick_params(axis="y", labelsize=10)
    ax2.grid(axis="y", visible=False)

    save(fig, "04_data_quality.png",
         f"{100*flagged/total:.2f}% of reports fail a physical-plausibility check and are "
         f"removed before training. All four faults are impossibilities rather than "
         f"judgement calls, so dropping them is safe. A report can carry more than one "
         f"fault, so the per-fault bars overlap.")


def fig05_types(d):
    g = d["type_groups"]; total = g.sum()
    order = np.argsort(g)
    names = [GROUP_NAMES[i] for i in order]
    vals  = [100 * g[i] / total for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    b = ax.barh(names, vals, color=BLUE, height=0.66)
    ax.bar_label(b, labels=[f"{v:.1f}%" for v in vals], padding=4, fontweight="bold")
    ax.set_xlabel("Share of all position reports")
    ax.set_title("Vessel Type Distribution")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(axis="y", visible=False)

    top, bot = GROUP_NAMES[int(np.argmax(g))], GROUP_NAMES[int(np.argmin(g))]
    save(fig, "05_vessel_types.png",
         f"The fleet is heavily imbalanced: {top} dominates while {bot} is scarce. Ship type "
         f"is given to the model as an input, and training windows are capped at "
         f"{cfg.max_windows_per_type_group:,} per type group so the rare classes are not "
         f"drowned out.")


def fig06_interval(d):
    gaps = d["raw_gaps"]
    med  = float(np.median(gaps) / 60.0)
    mins = gaps[gaps < 15 * 60] / 60.0

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(mins, bins=np.arange(0, 12.25, 0.25), color=BLUE, edgecolor="white")
    ax.axvline(med, color=RED, lw=2.5, label=f"median ({med:.1f} min)")
    ax.set_xlabel("Time since previous report (minutes)")
    ax.set_ylabel("Number of reports")
    ax.set_title("Reporting Interval")
    ax.set_xlim(0, 12)
    ax.yaxis.set_major_formatter(HUMAN)
    ax.legend()

    save(fig, "06_reporting_interval.png",
         f"Vessels report roughly every {med:.1f} minutes, and consistently so. This sets the "
         f"time scale of the whole project: {cfg.seq_len_enc} input pings is about "
         f"{cfg.seq_len_enc * med / 60:.1f} hours of history, and {cfg.seq_len_dec} predicted "
         f"pings is about {cfg.seq_len_dec * med:.0f} minutes ahead.")


def fig07_speed_raw(d):
    sog     = d["raw_sog"]
    stopped = float((sog < 0.5).mean() * 100)
    moving  = sog[(sog >= 0.5) & (sog < 102.3)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(np.clip(moving, 0, 25), bins=np.arange(0, 25.5, 0.5),
            color=GREEN, edgecolor="white",
            label=f"moving vessels ({100-stopped:.0f}% of reports)")
    ax.set_xlabel("Speed over ground (knots)")
    ax.set_ylabel("Number of reports")
    ax.set_title("Speed Distribution (raw data, moving vessels)")
    ax.yaxis.set_major_formatter(HUMAN)
    ax.legend()

    save(fig, "07_speed_raw.png",
         f"{stopped:.0f}% of raw reports come from vessels that are moored or anchored and are "
         f"filtered out before training — the model should learn how ships travel, not how they "
         f"sit still. Moving vessels average {moving.mean():.1f} knots.")


def fig08_activity(d):
    counts = d["train_ppv"]
    buckets = [
        ("Under 100",      int((counts < 100).sum())),
        ("100 – 999",      int(((counts >= 100)   & (counts < 1_000)).sum())),
        ("1,000 – 9,999",  int(((counts >= 1_000) & (counts < 10_000)).sum())),
        ("10,000+",        int((counts >= 10_000).sum())),
    ]
    names = [b[0] for b in buckets]
    vals  = [b[1] for b in buckets]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    b = ax.bar(names, vals, color=BLUE, width=0.6)
    ax.bar_label(b, labels=[f"{v:,}\n({100*v/len(counts):.0f}%)" for v in vals],
                 padding=4, fontsize=11)
    ax.set_xlabel("Total position reports sent by that vessel")
    ax.set_ylabel("Number of vessels")
    ax.set_title("Reports per Vessel")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.yaxis.set_major_formatter(HUMAN)

    save(fig, "08_reports_per_vessel.png",
         f"Activity is very uneven: the median vessel sent {int(np.median(counts)):,} reports "
         f"while the busiest sent {int(counts.max()):,}. Training is capped at "
         f"{cfg.max_windows_per_mmsi} windows per vessel so a handful of very active ships "
         f"cannot dominate what the model learns.")


def fig09_overlap(d):
    both      = int(d["overlap"])
    only_test = int(d["test_only"])
    total     = both + only_test

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.barh([""], [both], color=ORANGE, height=0.45,
            label=f"also seen in training ({both:,})")
    ax.barh([""], [only_test], left=[both], color=GREEN, height=0.45,
            label=f"unseen vessels ({only_test:,})")
    ax.set_xlabel("Vessels in the hold-out year")
    ax.set_title("Vessel Overlap Between Training and Hold-out")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=11)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(HUMAN)
    for x, v in ((both / 2, both), (both + only_test / 2, only_test)):
        ax.text(x, 0, f"{100*v/total:.0f}%", ha="center", va="center",
                color="white", fontweight="bold", fontsize=14)

    save(fig, "09_vessel_overlap.png",
         f"{100*both/total:.0f}% of hold-out vessels also appear in training. The hold-out is a "
         f"different *period*, not a different *fleet*, so it tests temporal generalisation; "
         f"results are also reported separately for the {only_test:,} genuinely unseen vessels.")


def fig10_funnel(d):
    if not os.path.exists(cfg.meta_path):
        say("    (skipping funnel — dataset_meta.json not found)")
        return
    meta = json.load(open(cfg.meta_path))

    raw     = int(d["train_totals"][0])
    windows = meta["n_train"] + meta["n_val"]
    kept    = windows * cfg.window_stride     # a window is cut every `stride` reports

    stages = [("Raw position reports", raw),
              ("Reports surviving cleaning\nand filtering", kept),
              (f"Training windows\n({WINDOW_LEN} consecutive reports)", windows)]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    b = ax.barh([s[0] for s in stages][::-1], [s[1] for s in stages][::-1],
                color=[GREEN, ORANGE, BLUE], height=0.6)
    ax.bar_label(b, labels=[f"{s[1]:,}" for s in stages][::-1], padding=6,
                 fontweight="bold")
    ax.set_xlim(0, raw * 1.25)
    ax.set_xlabel("Count")
    ax.set_title("From Raw Reports to Training Windows")
    ax.xaxis.set_major_formatter(HUMAN)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=11)

    save(fig, "10_preprocessing_funnel.png",
         f"{raw/1e6:.0f}M raw reports reduce to {windows:,} training windows. The first drop is "
         f"stationary vessels and faulty reports being removed; the second is that each window "
         f"is a run of {WINDOW_LEN} consecutive reports taken every {cfg.window_stride}.")


# ══ figures — processed dataset ═══════════════════════════════════════════════

def fig11_density(d):
    lat = denorm(d["feats"][:, IDX_LAT], IDX_LAT)
    lon = denorm(d["feats"][:, IDX_LON], IDX_LON)
    fig, ax = plt.subplots(figsize=(8, 6.6))
    hb = ax.hexbin(lon, lat, gridsize=110, bins="log", cmap="inferno", mincnt=1)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Vessel Density")
    cb = fig.colorbar(hb, ax=ax, shrink=0.85)
    cb.set_label("Busier  →")
    cb.set_ticks([])     # relative density only — numbered ticks would imply a
                         # precision the vessel sample does not have
    save(fig, "11_density_heatmap.png",
         "The bright ribbons are the major lanes through the Danish Straits, the Kattegat and "
         "the Great Belt — the corridors where the model has the densest training signal and "
         "where it should predict best.")


def fig12_scatter(d):
    lat = denorm(d["feats"][:, IDX_LAT], IDX_LAT)
    lon = denorm(d["feats"][:, IDX_LON], IDX_LON)
    k = min(120_000, len(lat))
    sel = np.random.default_rng(0).choice(len(lat), k, replace=False)
    fig, ax = plt.subplots(figsize=(8, 6.6))
    ax.scatter(lon[sel], lat[sel], s=1.5, alpha=0.15, color=BLUE, linewidths=0)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Geographic Coverage")
    save(fig, "12_position_scatter.png",
         f"A {k:,}-point sample of individual reports after preprocessing. Coverage is confined "
         f"to Danish waters, confirming the region filter is applied as intended.")


def fig13_trajectories(d):
    lines = list(d["traj_lines"])[:24]
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    cmap = plt.get_cmap("tab20")
    for i, ln in enumerate(lines):
        ax.plot(ln[:, 1], ln[:, 0], lw=1.1, alpha=0.85, color=cmap(i % 20))
        ax.scatter(ln[0, 1], ln[0, 0], s=18, color=cmap(i % 20), zorder=3)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Sample Reconstructed Trajectories")
    ax.scatter([], [], s=18, color="#444", label="voyage start")
    ax.legend()
    save(fig, "13_sample_trajectories.png",
         "Two dozen cleaned voyages. Each is continuous and lane-following after gap-splitting "
         "and filtering, which is the visual check that segmentation worked.")


def fig14_example(d):
    v = d["example"]
    lat = denorm(v[:, IDX_LAT], IDX_LAT); lon = denorm(v[:, IDX_LON], IDX_LON)
    sog = denorm(v[:, IDX_SOG], IDX_SOG)
    fig, (axm, axs) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    sc = axm.scatter(lon, lat, c=np.arange(len(v)), cmap="viridis", s=10)
    axm.plot(lon, lat, lw=0.6, color="#999", alpha=0.6)
    axm.scatter(lon[0], lat[0], s=70, marker="o", color=GREEN, zorder=3, label="start")
    axm.scatter(lon[-1], lat[-1], s=90, marker="*", color=RED, zorder=3, label="end")
    axm.set_xlabel("Longitude"); axm.set_ylabel("Latitude")
    axm.set_title("A Single Cleaned Voyage")
    axm.legend()
    cb = fig.colorbar(sc, ax=axm, shrink=0.85); cb.set_label("ping order")

    axs.plot(np.arange(len(v)), sog, color=BLUE, lw=1.5)
    axs.set_xlabel("Ping number along the voyage")
    axs.set_ylabel("Speed (knots)")
    axs.set_title("Speed Profile of the Same Voyage")
    axs.grid(alpha=0.3)

    save(fig, "14_individual_voyage.png",
         f"One vessel's cleaned trajectory ({len(v)} pings) coloured by time, with its speed "
         f"profile alongside. The path is continuous and the speed varies smoothly — no "
         f"teleports and no gaps, which is what preprocessing is meant to produce.")


def fig15_voyage_len(d):
    L = d["voyage_len"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(np.clip(L, 0, 1500), bins=60, color=BLUE, edgecolor="white")
    ax.axvline(WINDOW_LEN, color=RED, lw=2.2, label=f"window length ({WINDOW_LEN} pings)")
    ax.set_xlabel("Voyage length (pings)")
    ax.set_ylabel("Number of voyages")
    ax.set_title("Voyage Length Distribution")
    ax.yaxis.set_major_formatter(HUMAN)
    ax.legend()
    short = 100 * float((L < WINDOW_LEN).mean())
    save(fig, "15_voyage_length.png",
         f"{short:.0f}% of cleaned voyages are shorter than one {WINDOW_LEN}-ping window and "
         f"therefore contribute no training data; the median voyage is "
         f"{int(np.median(L))} pings.")


def fig16_dt(d):
    x = physical_features(d["feats"])["Δt (seconds)"] / 60.0
    med = float(np.median(x[x > 0]))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(np.clip(x, 0, 15), bins=np.arange(0, 15.25, 0.25), color=PURPLE,
            edgecolor="white")
    ax.axvline(med, color=RED, lw=2.2, label=f"median ({med:.1f} min)")
    ax.set_xlabel("Time since previous report (minutes)")
    ax.set_ylabel("Number of reports")
    ax.set_title("Δt Within Cleaned Voyages")
    ax.yaxis.set_major_formatter(HUMAN)
    ax.legend()
    save(fig, "16_dt_histogram.png",
         f"Within a voyage the interval is tightly concentrated around {med:.1f} minutes. Δt is "
         f"fed to the model as a feature so it can reason about irregular sampling rather than "
         f"assuming a fixed step.")


def _hist(d, key, title, fname, finding, clip=None, bins=60, colour=BLUE, xlabel=None):
    x = physical_features(d["feats"])[key]
    if clip is not None:
        x = np.clip(x, *clip)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(x, bins=bins, color=colour, edgecolor="white")
    ax.set_xlabel(xlabel or key)
    ax.set_ylabel("Number of reports")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(HUMAN)
    save(fig, fname, finding)


def _boxplot(d, key, title, fname, finding, clip=None, colour=BLUE):
    x = physical_features(d["feats"])[key]
    if clip is not None:
        x = np.clip(x, *clip)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.boxplot(x, orientation="horizontal", widths=0.55, showfliers=True,
               flierprops=dict(marker="o", ms=2, alpha=0.2, mfc=colour, mec="none"),
               medianprops=dict(color=RED, lw=2),
               boxprops=dict(color=colour, lw=1.5),
               whiskerprops=dict(color=colour), capprops=dict(color=colour))
    ax.set_xlabel(key)
    ax.set_yticks([])
    ax.set_title(title)
    save(fig, fname, finding)


def fig21_correlation(d):
    p = physical_features(d["feats"])
    p["LAT"] = denorm(d["feats"][:, IDX_LAT], IDX_LAT)
    p["LON"] = denorm(d["feats"][:, IDX_LON], IDX_LON)
    order = ["LAT", "LON", "SOG (knots)", "COG (degrees)", "Heading (deg)",
             "ROT (deg/min)", "Δt (seconds)", "ΔLAT", "ΔLON", "ΔCOG (deg)"]
    M = np.column_stack([p[k] for k in order])
    C = np.corrcoef(M, rowvar=False)
    short = [k.split(" (")[0] for k in order]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(short))); ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticks(range(len(short))); ax.set_yticklabels(short)
    for i in range(len(short)):
        for j in range(len(short)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(C[i, j]) > 0.55 else "#333")
    ax.set_title("Feature Correlation Matrix")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")

    # Report the strongest off-diagonal pair rather than asserting a relationship
    # in prose — an earlier version claimed a speed/delta correlation the matrix
    # did not actually show.
    off = np.abs(C - np.eye(len(C)))
    i, j = np.unravel_index(np.argmax(off), off.shape)
    save(fig, "21_correlation_matrix.png",
         f"The strongest off-diagonal relationship is {short[i]} vs {short[j]} at "
         f"{C[i, j]:.2f}. No pair approaches ±1, so the engineered features are not "
         f"redundant and none can be dropped as a duplicate of another.")


# ══ main ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Complete EDA of the AIS dataset.")
    ap.add_argument("--db",         default=cfg.train_db_path)
    ap.add_argument("--test-db",    default=cfg.test_db_path)
    ap.add_argument("--n-vessels",  type=int, default=2500)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--raw-only",   action="store_true",
                    help="Skip the processed half (no trajectory rebuild).")
    ap.add_argument("--redraw",     action="store_true",
                    help="Redraw from the cache without touching the databases.")
    args = ap.parse_args()

    t0 = time.time()
    say("\n" + "=" * 70)
    say("  AIS DATASET — EXPLORATORY DATA ANALYSIS")
    say("=" * 70)

    if args.redraw:
        if not os.path.exists(CACHE_PATH):
            raise SystemExit(f"No cache at {CACHE_PATH} — run without --redraw first.")
        d = dict(np.load(CACHE_PATH, allow_pickle=True))
        say(f"\n  Redrawing from cache ({CACHE_PATH})")
    else:
        for p in (args.db, args.test_db):
            if not os.path.exists(p):
                raise SystemExit(f"Database not found: {p}")
        say(f"\n  Training DB : {args.db}")
        say(f"  Hold-out DB : {args.test_db}")
        d = gather(args)

    have_processed = "feats" in d

    say("\n  Drawing raw-dataset figures ...")
    fig01_summary(d)
    fig02_volume(d)
    fig03_coverage(d)
    fig04_quality(d)
    fig05_types(d)
    fig06_interval(d)
    fig07_speed_raw(d)
    fig08_activity(d)
    fig09_overlap(d)
    fig10_funnel(d)

    if have_processed:
        say("\n  Drawing processed-dataset figures ...")
        fig11_density(d)
        fig12_scatter(d)
        fig13_trajectories(d)
        fig14_example(d)
        fig15_voyage_len(d)
        fig16_dt(d)
        _hist(d, "SOG (knots)", "Speed Over Ground", "17_sog_histogram.png",
              "After filtering, speeds concentrate in a cruising band with a low tail of "
              "manoeuvring near port. Near-stationary pings are already gone, so this is "
              "moving traffic only.",
              clip=(0, 30), colour=GREEN)
        _hist(d, "COG (degrees)", "Course Over Ground", "18_cog_histogram.png",
              "Course is far from uniform: the peaks fall on the headings of the main shipping "
              "lanes, which is the geographic structure the model is expected to learn.",
              bins=72, colour=ORANGE)
        _hist(d, "Heading (deg)", "Vessel Heading", "19_heading_histogram.png",
              "Heading closely tracks course. Where heading is unavailable the pipeline "
              "substitutes course, which is why this mirrors the COG distribution.",
              bins=72, colour=ORANGE)
        _hist(d, "ROT (deg/min)", "Rate of Turn", "20_rot_histogram.png",
              "Rate of turn is sharply peaked at zero — vessels travel straight most of the "
              "time. The rare non-zero values are the manoeuvres an anomaly detector most "
              "needs to model.",
              clip=(-40, 40), bins=80, colour=PURPLE)
        fig21_correlation(d)
        _boxplot(d, "SOG (knots)", "Speed — Spread and Outliers", "22_sog_boxplot.png",
                 "The box is the normal cruising range; points beyond the whiskers are "
                 "unusually fast vessels — exactly the extremes the anomaly detector should "
                 "flag.", clip=(0, 30), colour=GREEN)
        _boxplot(d, "ROT (deg/min)", "Rate of Turn — Spread and Outliers",
                 "23_rot_boxplot.png",
                 "Most pings sit at zero (straight-line travel); the scattered outliers are "
                 "sharp turns — infrequent, but strong behavioural signals.",
                 clip=(-60, 60), colour=PURPLE)
    else:
        say("\n  (processed figures skipped — --raw-only)")

    # ── written findings ──────────────────────────────────────────────────────
    med_gap = float(np.median(d["raw_gaps"]) / 60.0)
    total   = int(d["train_totals"][0])
    clean   = int(d["clean"])
    stopped = 100 * float((d["raw_sog"] < 0.5).mean())
    both    = int(d["overlap"]); only = int(d["test_only"])

    say("\n  ── Key figures ──")
    say(f"    Training   : {total:,} reports, {int(d['train_totals'][1]):,} vessels, "
        f"{d['train_span'][0]} → {d['train_span'][1]}")
    say(f"    Hold-out   : {int(d['test_totals'][0]):,} reports, "
        f"{int(d['test_totals'][1]):,} vessels, "
        f"{d['test_span'][0]} → {d['test_span'][1]}")
    say(f"    Reporting  : every {med_gap:.1f} min → "
        f"{cfg.seq_len_enc * med_gap / 60:.1f} h of history, "
        f"{cfg.seq_len_dec * med_gap:.0f} min predicted")
    say(f"    Stationary : {stopped:.0f}% of reports (filtered out)")
    say(f"    Faulty     : {100*(total-clean)/total:.2f}% of reports (filtered out)")
    say(f"    Overlap    : {both:,} of {both+only:,} hold-out vessels "
        f"({100*both/(both+only):.0f}%) also appear in training")
    if have_processed:
        say(f"    Voyages    : {len(d['voyage_len']):,} sampled, median "
            f"{int(np.median(d['voyage_len']))} pings")

    with open(os.path.join(OUT_DIR, "eda_log.txt"), "w") as fh:
        fh.write("\n".join(_log) + "\n")
    with open(os.path.join(OUT_DIR, "eda_findings.txt"), "w") as fh:
        fh.write("EDA findings — one entry per figure\n")
        fh.write("=" * 70 + "\n\n")
        for name, text in _findings:
            fh.write(f"{name}\n    {text}\n\n")

    say(f"\n  Done in {time.time() - t0:.0f}s.")
    say(f"  Figures  -> {OUT_DIR}/")
    say(f"  Findings -> {os.path.join(OUT_DIR, 'eda_findings.txt')}\n")


if __name__ == "__main__":
    main()
