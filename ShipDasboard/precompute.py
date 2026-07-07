"""
precompute.py
-------------
Precompute Replay-mode data (static/data/ships.json) for the dashboard.

v3 — Replay now mirrors the Live Sim's overlapping-prediction behaviour:
for each vessel a longer trajectory is used (SEQ_ENC history + --n-future
ground-truth future steps, both taken from the AIS database), the encoder
window slides forward in strides of --predict-every, and an autoregressive
SEQ_DEC-step prediction is issued at every stride. Each ship therefore
carries several overlapping predictions (issued at steps 90, 95, 100, …)
that the frontend reveals as the replay tick passes their issued_step,
exactly like predictions streaming in during a live sim.

Output schema (per ship) — old fields kept for backward compatibility:
    enc          [[lat,lon] x SEQ_ENC]       encoder history track
    future       [[lat,lon] x n_future]      full ground-truth future track
    actual       first SEQ_DEC of future     (legacy)
    pred         coords of first prediction  (legacy)
    predictions  [{pred_id, issued_step, coords, sogs, sigma, ade_km}]
                 `coords` is the denormalised prediction mean (mu);
                 `sigma` is the per-step [lat, lon] std-dev in degrees.
Top-level additions: predict_every, n_future_steps, window_len.

Usage:
    python precompute.py                          # defaults
    python precompute.py --n-ships 200 --predict-every 5 --n-future 50
    python precompute.py --checkpoint checkpoints/1476kmll01/best_model.pt
"""

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch

_HERE        = Path(__file__).resolve().parent
_TRANSFORMER = _HERE.parent / "ShipTransformer"
sys.path.insert(0, str(_TRANSFORMER))

from config  import cfg                                                # noqa: E402
from predict import (load_model, fetch_clean_track, rows_to_array,     # noqa: E402
                     denormalise_dec)

LAT_LO, LAT_HI = cfg.norm_bounds["LAT"]
LON_LO, LON_HI = cfg.norm_bounds["LON"]
LAT_RNG = LAT_HI - LAT_LO
LON_RNG = LON_HI - LON_LO
SOG_HI  = cfg.norm_bounds["SOG"][1]

N_DEC    = cfg.n_dec_features
N_DEC_IN = cfg.n_dec_input_features
SEQ_ENC  = cfg.seq_len_enc
SEQ_DEC  = cfg.seq_len_dec

_PREDICT_DELTAS = getattr(cfg, "predict_deltas", False)
if _PREDICT_DELTAS:
    _DLAT_LO, _DLAT_HI = cfg.norm_bounds["dLAT"]
    _DLON_LO, _DLON_HI = cfg.norm_bounds["dLON"]
    _DLAT_RNG = _DLAT_HI - _DLAT_LO
    _DLON_RNG = _DLON_HI - _DLON_LO

GROUP_NAMES = [
    "Unknown", "Cargo", "Tanker", "Passenger",
    "Fishing", "Tug/Service", "Pleasure/Sail", "Other",
]

DEFAULT_CKPT = str(_TRANSFORMER / "checkpoints/best_model.pt")
DEFAULT_DB   = str(_TRANSFORMER / "data2" / "2023.db")
OUT_PATH     = _HERE / "static/data/ships.json"

_REGION_PARAMS = tuple(cfg.region_bounds)   # (lat_min, lat_max, lon_min, lon_max)


# ── vessel selection ──────────────────────────────────────────────────────────

def select_vessels(db_path, n_wanted, min_pings, rng):
    """Return candidate MMSIs with enough clean pings, moving preferred.
    Uses the dashboard_vessel_stats cache table (built by sim_engine) when
    available; otherwise falls back to a one-off full GROUP BY."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'"
                "  AND name='dashboard_vessel_stats'")
    if cur.fetchone():
        cur.execute(
            "SELECT MMSI FROM dashboard_vessel_stats"
            "  WHERE uniq_pos >= 10 AND cnt >= ? ORDER BY cnt DESC LIMIT ?",
            (min_pings, n_wanted * 4),
        )
    else:
        print("  (no stats cache table — running full-table scan once)")
        cur.execute(
            "SELECT MMSI FROM ("
            "  SELECT MMSI, COUNT(*) AS cnt,"
            "    COUNT(DISTINCT ROUND(LAT,2)||','||ROUND(LON,2)) AS uniq_pos"
            "  FROM ais"
            "  WHERE LAT BETWEEN ? AND ? AND LON BETWEEN ? AND ? AND FLAGS=0"
            "  GROUP BY MMSI HAVING uniq_pos >= 10 AND cnt >= ?"
            "  ORDER BY cnt DESC LIMIT ?)",
            (*_REGION_PARAMS, min_pings, n_wanted * 4),
        )
    mmsis = [r[0] for r in cur.fetchall()]
    conn.close()
    rng.shuffle(mmsis)
    return mmsis


def pick_segment(track_norm, seg_len):
    """Pick the start index of the most-moving seg_len window of the track
    (maximum summed |dLAT|+|dLON|) so replay ships aren't sitting at anchor."""
    n = len(track_norm)
    if n < seg_len:
        return None
    step = np.abs(np.diff(track_norm[:, 0])) + np.abs(np.diff(track_norm[:, 1]))
    csum = np.concatenate([[0.0], np.cumsum(step)])
    move = csum[seg_len - 1:] - csum[:n - seg_len + 1]   # movement per window start
    return int(np.argmax(move))


# ── batched autoregressive inference (mirrors sim_engine._batch_ar_predict) ──

@torch.no_grad()
def batch_ar_predict(model, enc_arrays, device):
    """Batched AR decode for SEQ_DEC steps, handling cfg.predict_deltas.
    Reuses the last observed DT for every predicted step — the same policy as
    the live sim, so Replay and Live Sim show the same model behaviour.
    Returns (mu, sigma) per item, each (SEQ_DEC, N_DEC) normalised numpy."""
    src = torch.from_numpy(np.stack(enc_arrays)).to(device)
    last_dt     = src[:, -1:, N_DEC:N_DEC_IN]
    dec_input   = src[:, -1:, :N_DEC_IN]
    prev_motion = src[:, -1:, :N_DEC].float()

    mu_acc, sig_acc = [], []
    # Delta mode: accumulate delta variance into absolute-position sigma
    # (delta-normalised → absolute-normalised units), matching sim_engine.
    var_lat_deg2 = torch.zeros((len(enc_arrays), 1, 1), device=src.device)
    var_lon_deg2 = torch.zeros((len(enc_arrays), 1, 1), device=src.device)
    for _ in range(SEQ_DEC):
        mu_raw, log_var = model(src, dec_input)
        mu_last  = mu_raw[:, -1:, :].float()
        std_last = (log_var[:, -1:, :].float() * 0.5).exp()

        if _PREDICT_DELTAS:
            prev_lat = prev_motion[:, :, 0:1] * LAT_RNG + LAT_LO
            prev_lon = prev_motion[:, :, 1:2] * LON_RNG + LON_LO
            dlat = mu_last[:, :, 0:1] * _DLAT_RNG + _DLAT_LO
            dlon = mu_last[:, :, 1:2] * _DLON_RNG + _DLON_LO
            abs_lat = ((prev_lat + dlat - LAT_LO) / LAT_RNG).clamp(0.0, 1.0)
            abs_lon = ((prev_lon + dlon - LON_LO) / LON_RNG).clamp(0.0, 1.0)
            sog_cog = mu_last[:, :, 2:].clamp(0.0, 1.0)
            next_motion = torch.cat([abs_lat, abs_lon, sog_cog], dim=-1)
        else:
            next_motion = mu_last.clamp(0.0, 1.0)

        sin_raw = next_motion[:, :, 3] * 2.0 - 1.0
        cos_raw = next_motion[:, :, 4] * 2.0 - 1.0
        mag     = (sin_raw.pow(2) + cos_raw.pow(2)).sqrt().clamp(min=1e-8)
        next_motion = next_motion.clone()
        next_motion[:, :, 3] = (sin_raw / mag + 1.0) * 0.5
        next_motion[:, :, 4] = (cos_raw / mag + 1.0) * 0.5

        prev_motion = next_motion
        dec_input   = torch.cat(
            [dec_input, torch.cat([next_motion, last_dt], dim=-1)], dim=1)

        if _PREDICT_DELTAS:
            var_lat_deg2 = var_lat_deg2 + (std_last[:, :, 0:1] * _DLAT_RNG) ** 2
            var_lon_deg2 = var_lon_deg2 + (std_last[:, :, 1:2] * _DLON_RNG) ** 2
            std_abs = torch.cat([
                var_lat_deg2.sqrt() / LAT_RNG,
                var_lon_deg2.sqrt() / LON_RNG,
                std_last[:, :, 2:],
            ], dim=-1)
        else:
            std_abs = std_last

        mu_acc.append(next_motion.cpu().numpy())
        sig_acc.append(std_abs.cpu().numpy())

    mu_b  = np.concatenate(mu_acc,  axis=1)
    sig_b = np.concatenate(sig_acc, axis=1)
    return [(mu_b[i], sig_b[i]) for i in range(len(enc_arrays))]


# ── metrics / labelling ───────────────────────────────────────────────────────

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    """Vectorised haversine over numpy arrays (degrees in, km out)."""
    R = 6371.0
    p1, p2 = np.radians(a_lat), np.radians(b_lat)
    dphi   = p2 - p1
    dlmb   = np.radians(b_lon - a_lon)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))


def anomaly_reason(enc_sogs_norm, enc_csin, enc_ccos, ade_km):
    early_sog = enc_sogs_norm[:20].mean() * SOG_HI
    late_sog  = enc_sogs_norm[-20:].mean() * SOG_HI
    mean_sog  = enc_sogs_norm.mean() * SOG_HI

    cog_e = math.atan2(float(enc_csin[:10].mean()), float(enc_ccos[:10].mean()))
    cog_l = math.atan2(float(enc_csin[-10:].mean()), float(enc_ccos[-10:].mean()))
    dcog  = abs((math.degrees(cog_l - cog_e) + 180) % 360 - 180)

    parts = []
    if ade_km > 8:
        parts.append(f"Extreme trajectory deviation ({ade_km:.1f} km error)")
    elif ade_km > 3.5:
        parts.append(f"Significant trajectory deviation ({ade_km:.1f} km error)")
    elif ade_km > 1.5:
        parts.append(f"Moderate trajectory deviation ({ade_km:.1f} km error)")

    if early_sog > 5 and late_sog < 1.5:
        parts.append(f"Deceleration: {early_sog:.0f}→{late_sog:.0f} kn")
    elif late_sog > 5 and early_sog < 1.5:
        parts.append("Acceleration from near-stop")
    elif mean_sog < 2:
        parts.append(f"Low-speed vessel ({mean_sog:.1f} kn mean)")

    if dcog > 130:
        parts.append(f"Course reversal ({dcog:.0f}°)")
    elif dcog > 60:
        parts.append(f"Sharp course change ({dcog:.0f}°)")

    return "; ".join(parts) if parts else "Within expected trajectory range"


def dn_coords(seg):
    """Denormalise [:,0:2] of a normalised segment to [[lat,lon], ...]."""
    return [[round(float(r[0] * LAT_RNG + LAT_LO), 5),
             round(float(r[1] * LON_RNG + LON_LO), 5)] for r in seg]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ships",       type=int, default=150,
                    help="Target number of ships in ships.json")
    ap.add_argument("--checkpoint",    default=DEFAULT_CKPT)
    ap.add_argument("--db",            default=DEFAULT_DB,
                    help="AIS SQLite database to draw trajectories from")
    ap.add_argument("--predict-every", type=int, default=5,
                    help="Stride (steps) between overlapping predictions")
    ap.add_argument("--n-future",      type=int, default=50,
                    help="Ground-truth future steps per ship (40–60 typical)")
    ap.add_argument("--threshold",     type=float, default=3.0,
                    help="Anomaly σ threshold — match the live sim default (3.0)")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--batch",         type=int, default=32,
                    help="Inference batch size across (ship, stride) pairs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng    = np.random.default_rng(args.seed)
    n_fut  = max(SEQ_DEC, args.n_future)
    pe     = max(1, args.predict_every)
    seg_len = SEQ_ENC + n_fut

    print(f"Device       : {device}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"DB           : {args.db}")
    print(f"Window       : {SEQ_ENC} history + {n_fut} future, predict every {pe}")
    model = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded       : {n_params:,} parameters\n")

    candidates = select_vessels(args.db, args.n_ships, seg_len + 10,
                                np.random.default_rng(args.seed))
    print(f"Candidates   : {len(candidates)} vessels")

    # ── collect segments ──────────────────────────────────────────────────────
    ships = []      # per ship: dict with seg (normalised) + mmsi
    for mmsi in candidates:
        if len(ships) >= args.n_ships:
            break
        rows = fetch_clean_track(args.db, mmsi)
        if len(rows) < seg_len:
            continue
        track_norm = rows_to_array(rows)
        start = pick_segment(track_norm, seg_len)
        if start is None:
            continue
        seg = track_norm[start : start + seg_len]
        # Skip vessels that barely move even in their best window (<0.5 km)
        d_km = haversine_km(seg[0, 0] * LAT_RNG + LAT_LO, seg[0, 1] * LON_RNG + LON_LO,
                            seg[-1, 0] * LAT_RNG + LAT_LO, seg[-1, 1] * LON_RNG + LON_LO)
        if d_km < 0.5:
            continue
        ships.append({"mmsi": int(mmsi), "seg": seg})
        if len(ships) % 25 == 0:
            print(f"  segments: {len(ships)}/{args.n_ships}", end="\r")
    print(f"\nSegments     : {len(ships)} usable vessels")
    if not ships:
        sys.exit("No vessels with long enough moving tracks — lower --n-future?")

    # ── build strided inference tasks and run them batched ──────────────────
    issued_steps = list(range(SEQ_ENC, seg_len - SEQ_DEC + 1, pe))
    tasks = [(si, s) for si in range(len(ships)) for s in issued_steps]
    print(f"Predictions  : {len(issued_steps)} per ship × {len(ships)} ships "
          f"= {len(tasks)} AR inferences")

    results = {}   # (ship_idx, issued_step) -> (mu, sigma)
    for i in range(0, len(tasks), args.batch):
        chunk = tasks[i : i + args.batch]
        encs  = [ships[si]["seg"][s - SEQ_ENC : s] for si, s in chunk]
        outs  = batch_ar_predict(model, encs, device)
        for (si, s), out in zip(chunk, outs):
            results[(si, s)] = out
        print(f"  inference: {min(i + args.batch, len(tasks))}/{len(tasks)}", end="\r")
    print()

    # ── assemble output ───────────────────────────────────────────────────────
    # Per-step z-score of the ground-truth position against each prediction's
    # mean/sigma — the *same* quantity the live sim thresholds on. Anomaly
    # severity is then driven by z vs the σ threshold (matching the sim),
    # not by ADE percentiles.
    all_raw_z = []
    ships_raw = []
    for si, sh in enumerate(ships):
        seg  = sh["seg"]
        preds = []
        ship_raw_z = []
        for k, s in enumerate(issued_steps):
            mu, sigma = results[(si, s)]
            dec_raw = denormalise_dec(mu)                       # (SEQ_DEC, N_DEC)
            gt      = seg[s : s + SEQ_DEC]
            gt_lat  = gt[:, 0] * LAT_RNG + LAT_LO
            gt_lon  = gt[:, 1] * LON_RNG + LON_LO
            ade = float(haversine_km(dec_raw[:, 0], dec_raw[:, 1], gt_lat, gt_lon).mean())

            sig_lat = sigma[:, 0] * LAT_RNG                     # degrees
            sig_lon = sigma[:, 1] * LON_RNG
            z_lat   = np.abs(gt_lat - dec_raw[:, 0]) / (sig_lat + 1e-6)
            z_lon   = np.abs(gt_lon - dec_raw[:, 1]) / (sig_lon + 1e-6)
            raw_z   = 0.5 * (z_lat + z_lon)                     # (SEQ_DEC,) unitless
            ship_raw_z.extend(raw_z.tolist())
            all_raw_z.extend(raw_z.tolist())

            preds.append({
                "pred_id":     k,
                "issued_step": s,
                "coords": [[round(float(dec_raw[i, 0]), 5),
                            round(float(dec_raw[i, 1]), 5)] for i in range(SEQ_DEC)],
                "sogs":   [round(float(dec_raw[i, 2]), 2) for i in range(SEQ_DEC)],
                "sigma":  [[round(float(sigma[i, 0] * LAT_RNG), 4),
                            round(float(sigma[i, 1] * LON_RNG), 4)]
                           for i in range(SEQ_DEC)],
                "ade_km": round(ade, 3),
                "z_peak": round(float(raw_z.max()), 3),
            })

        ship_type = int(np.clip(round(float(seg[:SEQ_ENC, 6].mean()) * 7), 0, 7))
        ships_raw.append({
            "mmsi":       sh["mmsi"],
            "ship_type":  ship_type,
            "seg":        seg,
            "preds":      preds,
            "ade_km":     float(np.mean([p["ade_km"] for p in preds])),
            "raw_z_peak": float(np.max(ship_raw_z)) if ship_raw_z else 0.0,
        })

    # Calibrate exactly like SimEngine._calibrate_z: rescale by the fleet's p90
    # so calibrated z is a unit-normal-like deviate and the same σ threshold
    # means the same thing here as in the live sim.
    thr    = float(args.threshold)
    z_p90  = float(np.percentile(all_raw_z, 90)) if all_raw_z else 0.0
    z_scale = (z_p90 / 1.2816) if z_p90 > 1e-9 else 1.0

    def z_level(peak_cal):
        if   peak_cal <  thr:          return "none"
        elif peak_cal <  1.5 * thr:    return "mild"
        elif peak_cal <  2.5 * thr:    return "moderate"
        else:                          return "severe"

    ades = np.array([s["ade_km"] for s in ships_raw])
    p25, p75, p90 = (float(np.percentile(ades, q)) for q in (25, 75, 90))
    print(f"ADE          : p25={p25:.2f}  p75={p75:.2f}  p90={p90:.2f} km")
    print(f"Threshold    : {thr:.1f}σ   (z p90={z_p90:.2f}, scale={z_scale:.3f})")

    ships_out = []
    for s in ships_raw:
        seg = s["seg"]
        ade = s["ade_km"]
        peak_cal = s["raw_z_peak"] / z_scale
        level  = z_level(peak_cal)
        reason = anomaly_reason(seg[:SEQ_ENC, 2], seg[:SEQ_ENC, 3],
                                seg[:SEQ_ENC, 4], ade)
        first = s["preds"][0]
        ships_out.append({
            "id":          0,   # assigned after sort
            "mmsi":        s["mmsi"],
            "type":        s["ship_type"],
            "type_name":   GROUP_NAMES[s["ship_type"]],
            "enc":         dn_coords(seg[:SEQ_ENC]),
            "future":      dn_coords(seg[SEQ_ENC:]),
            "actual":      dn_coords(seg[SEQ_ENC : SEQ_ENC + SEQ_DEC]),  # legacy
            "pred":        first["coords"],                              # legacy
            "sogs":        [round(float(v * SOG_HI), 2) for v in seg[:SEQ_ENC, 2]],
            "predictions": s["preds"],
            "ade_km":      round(ade, 2),
            "z_peak":      round(peak_cal, 2),
            "level":       level,
            "reason":      reason,
        })

    # Most-anomalous first (by calibrated z, the classifier — matches the sim).
    ships_out.sort(key=lambda x: x["z_peak"], reverse=True)
    for i, s in enumerate(ships_out):
        s["id"] = i

    level_counts, type_counts = {}, {}
    for s in ships_out:
        level_counts[s["level"]]    = level_counts.get(s["level"], 0) + 1
        type_counts[s["type_name"]] = type_counts.get(s["type_name"], 0) + 1

    output = {
        "n_ships":        len(ships_out),
        "seq_enc":        SEQ_ENC,
        "seq_dec":        SEQ_DEC,
        "predict_every":  pe,
        "n_future_steps": n_fut,
        "window_len":     seg_len,
        "anomaly_threshold": round(thr, 2),
        "z_scale":        round(z_scale, 4),
        "ade_p25":        round(p25, 2),
        "ade_p75":        round(p75, 2),
        "ade_p90":        round(p90, 2),
        "level_counts":   level_counts,
        "type_counts":    type_counts,
        "ships":          ships_out,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f)

    print(f"Saved {len(ships_out)} ships → {OUT_PATH}")
    print(f"Levels : {level_counts}")
    print(f"Types  : {type_counts}")


if __name__ == "__main__":
    main()
