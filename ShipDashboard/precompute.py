"""
precompute.py
-------------
Precompute Replay-mode data (static/data/ships.json) for the dashboard.
Usage:
    python precompute.py 
    python precompute.py --n-ships 200 --predict-every 5 --n-future 50
    python precompute.py --checkpoint checkpoints/1476kmll01/best_model.pt
"""

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_HERE        = Path(__file__).resolve().parent
_TRANSFORMER = _HERE.parent / "ShipTransformer"
_DATAHANDLING = _HERE.parent / "DataHandling"
sys.path.insert(0, str(_TRANSFORMER))

from config  import cfg                                               
from predict import (load_model, fetch_clean_track, rows_to_array,    
                     denormalise_dec, _parse_ts)
import model_registry          # (GRU/TCN/Transformer loaders + ckpts)
import inject_replay         # (labelled synthetic anomaly injection)

try:                        # land mask 
    from global_land_mask import globe as _land_globe
    _LAND_OK = True
except Exception:
    _land_globe = None
    _LAND_OK = False

#accuracy scoreboard, in display order.
SCOREBOARD_MODELS = ["gru", "tcn", "transformer"]

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

DEFAULT_CKPT = str(_TRANSFORMER / "checkpoints/transformer_model.pt")
DEFAULT_DB   = str(_DATAHANDLING / "testing" / "2023.db")
OUT_PATH     = _HERE / "static/data/ships.json"

_REGION_PARAMS = tuple(cfg.region_bounds)   # (lat_min, lat_max, lon_min, lon_max)
_REGION_SQL    = "LAT BETWEEN ? AND ? AND LON BETWEEN ? AND ?"

# Data-integrity fault bits written by Data_Processing.py (same bitmask in both DBs).
_FLAG_NAMES = {1: "impossible speed", 2: "position jump",
               4: "course/speed mismatch", 8: "stationary drift"}
#impossible-speed and position-jump are anomaly.
_SUSPICIOUS_FLAGS = 1 | 2


#--- vessel selection ---

def select_faulty_vessels(db_path, n_wanted, rng):
    #MMSIs carrying(FLAGS != 0) filtered
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    # FLAGS values with bit 1 or 2 set 
    susp = [v for v in range(1, 16) if v & _SUSPICIOUS_FLAGS]
    cur.execute(
        f"SELECT MMSI FROM ais WHERE FLAGS IN ({','.join('?' * len(susp))}) AND {_REGION_SQL} "
        f"GROUP BY MMSI ORDER BY COUNT(*) DESC LIMIT ?",
        (*susp, *_REGION_PARAMS, max(n_wanted * 20, 200)),
    )
    mmsis = [r[0] for r in cur.fetchall()]
    conn.close()
    rng.shuffle(mmsis)
    return mmsis


def fetch_full_track(db_path, mmsi):
    """Full track for one MMSI including faulty pings (no FLAGS filter), region-
    filtered, time-ordered. Returns (normalised (N,14) array, FLAGS (N,) array)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute(f"SELECT * FROM ais WHERE MMSI=? AND {_REGION_SQL} ORDER BY TIMESTAMP",
                (mmsi, *_REGION_PARAMS))
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return None, None
    flags = np.array([int(r["FLAGS"]) if r["FLAGS"] is not None else 0 for r in rows], dtype=np.int32)
    ts    = np.array([_parse_ts(r["TIMESTAMP"]) for r in rows], dtype=float)
    return rows_to_array(rows), flags, ts


# Replay fleet mix (per ship-type group)
_CLASS_TARGETS_BASE = {1: 45, 2: 45, 3: 30, 4: 20, 5: 20, 6: 5, 0: 5}   # 170


def _class_targets(n_ships):
    total = sum(_CLASS_TARGETS_BASE.values())
    if n_ships == total:
        return dict(_CLASS_TARGETS_BASE)
    t = {g: max(0, int(round(c / total * n_ships))) for g, c in _CLASS_TARGETS_BASE.items()}
    t[1] += n_ships - sum(t.values())      # rounding remainder onto Cargo
    return t


def select_vessels_typed(db_path, n_ships, min_pings, rng):
    """Candidate MMSIs bucketed by ship-type group so the replay fleet follows
    _CLASS_TARGETS_BASE. One GROUP BY over the ship-type index gives every typed
    vessel's dominant class; a second query picks up the Unknown (no static type)
    vessels. Returns (cand_by_group {group: [mmsi,...]}, targets {group: count})."""
    targets = _class_targets(n_ships)
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute(
        f"SELECT MMSI, SHIP_TYPE, COUNT(*) c FROM ais "
        f"WHERE SHIP_TYPE > 0 AND {_REGION_SQL} AND FLAGS = 0 "
        f"GROUP BY MMSI, SHIP_TYPE", _REGION_PARAMS)
    tot, best = {}, {}
    for m, st, cnt in cur.fetchall():
        tot[m] = tot.get(m, 0) + cnt
        if cnt > best.get(m, (0, 0))[0]:
            best[m] = (cnt, int(st))
    cand = {g: [] for g in targets}
    for m, (cnt, st) in best.items():
        if tot[m] < min_pings:
            continue
        g = cfg.ship_type_groups.get(st, 7)
        if g in cand:
            cand[g].append(m)
    # Unknown (group 0) - vessels with no usable static ship type (SHIP_TYPE 0).
    if targets.get(0):
        cur.execute(
            f"SELECT MMSI, COUNT(*) c FROM ais "
            f"WHERE SHIP_TYPE = 0 AND {_REGION_SQL} AND FLAGS = 0 "
            f"GROUP BY MMSI HAVING c >= ?", (*_REGION_PARAMS, min_pings))
        typed = set(best)
        cand[0] = [m for m, _ in cur.fetchall() if m not in typed]
    conn.close()
    for g in cand:
        rng.shuffle(cand[g])
    return cand, targets


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


#--- batched autoregressive inference (mirrors sim_engine._batch_ar_predict) ---

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
    # Delta mode: delta variance to absolute-position sigma
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


# --- metrics / labelling ----

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    #Vectorised haversine over numpy arrays (degrees in, km out)
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


# -- physical rule-based detectors (ported from sim_engine, adapted to Replay) --
_JUMP_MIN_KM, _JUMP_MAX_GAP_SEC, _JUMP_MAX_KN = 5.0, 900.0, 80.0
# on_land for Replay's coarse (~1 km) land mask: 
_ONLAND_MIN_SOG, _ONLAND_MIN_RUN = 2.0, 3
# loitering: a *transiting* vessel confined + near-stopped.
_LOITER_RADIUS_KM, _LOITER_MIN_RUN = 0.75, 6
_LOITER_SHIP_TYPES = (1, 2, 3)    # Cargo, Tanker, Passenger
_LOITER_COAST_DEG  = 0.045       # ~5 km ring; any land inside treat as port


def _open_sea(la, lo):
    """True when no land sits within ~5 km — i.e. genuinely offshore, so a stop
    here isn't a routine port call. Coarse (uses the same land mask as on_land)."""
    if not _LAND_OK:
        return True
    d = _LOITER_COAST_DEG
    for dla in (-d, 0.0, d):
        for dlo in (-d, 0.0, d):
            if bool(_land_globe.is_land(la + dla, lo + dlo)):
                return False
    return True

def rule_detectors(seg, ship_type):
    """Physical alerts over the replayed future track (rows >= SEQ_ENC).
    Returns a list of {kind, tick, lat, lon, reason}, at most one per kind."""
    lat = seg[:, 0] * LAT_RNG + LAT_LO
    lon = seg[:, 1] * LON_RNG + LON_LO
    sog = seg[:, 2] * SOG_HI
    dt  = seg[:, 5] * 7200.0
    alerts, fired = [], set()
    onland_run = 0
    loiter_lat = loiter_lon = None
    loiter_run = 0
    for t in range(SEQ_ENC, len(seg)):
        la, lo, sg = float(lat[t]), float(lon[t]), float(sog[t])

        # speed_jump — implausible distance/time between consecutive pings.
        if "speed_jump" not in fired and t > 0:
            d = float(haversine_km(lat[t - 1], lon[t - 1], la, lo))
            tsec = float(dt[t])
            if d >= _JUMP_MIN_KM and 0 < tsec <= _JUMP_MAX_GAP_SEC:
                implied = (d / 1.852) / (tsec / 3600.0)
                if implied > _JUMP_MAX_KN:
                    alerts.append({"kind": "speed_jump", "tick": t,
                                   "lat": round(la, 5), "lon": round(lo, 5),
                                   "reason": f"Jumped {d:.0f} km in {tsec/60:.0f} min "
                                             f"(~{implied:.0f} kn — impossible)"})
                    fired.add("speed_jump")

        # on_land — moving at transit speed over land for a sustained run.
        if _LAND_OK and sg > _ONLAND_MIN_SOG:
            on = bool(_land_globe.is_land(la, lo))
            onland_run = onland_run + 1 if on else 0
            if on and onland_run >= _ONLAND_MIN_RUN and "on_land" not in fired:
                alerts.append({"kind": "on_land", "tick": t,
                               "lat": round(la, 5), "lon": round(lo, 5),
                               "reason": f"Tracking over land at {sg:.0f} kn"})
                fired.add("on_land")
        else:
            onland_run = 0

        # loitering - transiting vessel confined + near-stopped for a run of steps.
        if ship_type in _LOITER_SHIP_TYPES:
            if loiter_lat is None or float(haversine_km(loiter_lat, loiter_lon, la, lo)) > _LOITER_RADIUS_KM:
                loiter_lat, loiter_lon, loiter_run = la, lo, 0
            elif sg < 1.0:
                loiter_run += 1
                if loiter_run >= _LOITER_MIN_RUN and "loitering" not in fired \
                        and _open_sea(la, lo):
                    alerts.append({"kind": "loitering", "tick": t,
                                   "lat": round(la, 5), "lon": round(lo, 5),
                                   "reason": f"Loitering offshore (~{loiter_run} steps stopped)"})
                    fired.add("loitering")
            else:
                loiter_run = 0
    return alerts


# --- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ships",       type=int, default=185,
                    help="Target number of ships (split across classes per "
                         "_CLASS_TARGETS_BASE; default 185 = 49C/49T/33P/22F/22Tug/5Pl/5Unk)")
    ap.add_argument("--checkpoint",    default=DEFAULT_CKPT)
    ap.add_argument("--db",            default=DEFAULT_DB,
                    help="AIS SQLite database to draw trajectories from")
    ap.add_argument("--predict-every", type=int, default=5,
                    help="Stride (steps) between overlapping predictions")
    ap.add_argument("--n-future",      type=int, default=50,
                    help="Ground-truth future steps per ship (40–60 typical)")
    ap.add_argument("--threshold",     type=float, default=None,
                    help="Flagging threshold on the per-class-calibrated km deviation. "
                         "Omit to use the F1-optimal point found by the sweep (the km "
                         "scale is data-dependent, so there's no fixed default).")
    ap.add_argument("--seed",          type=int, default=7)
    ap.add_argument("--batch",         type=int, default=32,
                    help="Inference batch size across (ship, stride) pairs")
    ap.add_argument("--inject-frac",   type=float, default=0.2,
                    help="Fraction of vessels to inject a labelled synthetic "
                         "anomaly into (0 = none). Injected vessels get a "
                         "synthetic MMSI 9000000xx and an 'injected' flag.")
    ap.add_argument("--inject-seed",   type=int, default=21,
                    help="Seed for anomaly type/severity/target selection")
    ap.add_argument("--persistence",   type=int, default=3,
                    help="Anomaly must persist this many consecutive steps (K-step "
                         "moving average of the consensus deviation) before flagging "
                         "(default 3)")
    ap.add_argument("--consensus",     choices=("min", "mean", "max"), default="mean",
                    help="How to fuse the overlapping forecasts covering each true "
                         "step: min = surprise every forecast (max precision, least "
                         "recall), mean = surprise on average (balanced, default), "
                         "max = any forecast (max recall, most false alarms)")
    ap.add_argument("--n-faulty",      type=int, default=0,
                    help="Include this many vessels carrying REAL data-integrity "
                         "faults (FLAGS != 0) — normally filtered out — to simulate "
                         "noisy real-world conditions and give real anomaly labels.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_fut  = max(SEQ_DEC, args.n_future)
    pe     = max(1, args.predict_every)
    seg_len = SEQ_ENC + n_fut

    print(f"Device       : {device}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"DB           : {args.db}")
    print(f"Window       : {SEQ_ENC} history + {n_fut} future, predict every {pe}")

    # Load all three models.
    models = {
        "gru":         model_registry.MODELS["gru"]["loader"](model_registry.MODELS["gru"]["ckpt"], device),
        "tcn":         model_registry.MODELS["tcn"]["loader"](model_registry.MODELS["tcn"]["ckpt"], device),
        "transformer": load_model(args.checkpoint, device),
    }
    for mk in SCOREBOARD_MODELS:
        n_params = sum(p.numel() for p in models[mk].parameters())
        print(f"Loaded {mk:12}: {n_params:,} parameters")
    print()

    cand_by_group, targets = select_vessels_typed(
        args.db, args.n_ships, seg_len + 10, np.random.default_rng(args.seed))
    print("Candidates   : " + ", ".join(
        f"{GROUP_NAMES[g]}:{len(cand_by_group[g])}" for g in sorted(cand_by_group)))
    print("Class targets: " + ", ".join(
        f"{GROUP_NAMES[g]}:{targets[g]}" for g in sorted(targets)))

    # --- collect segments per class up to its target ---
    def _collect_one(mmsi):
        rows = fetch_clean_track(args.db, mmsi)
        if len(rows) < seg_len:
            return None
        track_norm = rows_to_array(rows)
        start = pick_segment(track_norm, seg_len)
        if start is None:
            return None
        seg = track_norm[start : start + seg_len]
        seg_ts = np.array([_parse_ts(rows[i]["TIMESTAMP"])
                           for i in range(start, start + seg_len)], dtype=float)
        # Skip vessels that barely move even in their best window (<0.5 km)
        d_km = haversine_km(seg[0, 0] * LAT_RNG + LAT_LO, seg[0, 1] * LON_RNG + LON_LO,
                            seg[-1, 0] * LAT_RNG + LAT_LO, seg[-1, 1] * LON_RNG + LON_LO)
        if d_km < 0.5:
            return None
        return {"mmsi": int(mmsi), "seg": seg, "seg_ts": seg_ts}

    ships = []      # per ship: dict with seg (normalised) + mmsi
    got = {}
    for g in sorted(targets):
        n = 0
        for mmsi in cand_by_group.get(g, []):
            if n >= targets[g]:
                break
            rec = _collect_one(mmsi)
            if rec is not None:
                ships.append(rec); n += 1
        got[g] = n
    print("Segments     : " + ", ".join(
        f"{GROUP_NAMES[g]}:{got.get(g,0)}" for g in sorted(targets)) +
        f"  (total {len(ships)})")
    if not ships:
        sys.exit("No vessels with long enough moving tracks — lower --n-future?")

    # --- include real data-integrity faults (FLAGS != 0) ---
    # normally filtered out; including them simulates noisy real-world conditions for real anomaly
    n_faulty = 0
    if args.n_faulty > 0:
        f_rng = np.random.default_rng(args.seed + 1)
        for mmsi in select_faulty_vessels(args.db, args.n_faulty, f_rng):
            if n_faulty >= args.n_faulty:
                break
            arr, flags, ts = fetch_full_track(args.db, mmsi)
            if arr is None or len(arr) < seg_len:
                continue
            start = None
            for fi in np.where((flags & _SUSPICIOUS_FLAGS) != 0)[0]:
                st = int(fi) - SEQ_ENC - 3
                if st >= 0 and st + seg_len <= len(arr) \
                        and ((flags[st + SEQ_ENC: st + seg_len] & _SUSPICIOUS_FLAGS) != 0).any():
                    start = st
                    break
            if start is None:
                continue
            seg_flags = flags[start: start + seg_len]
            bits = int(np.bitwise_or.reduce(seg_flags[SEQ_ENC:])) & _SUSPICIOUS_FLAGS
            if bits == 0:
                continue
            ships.append({
                "mmsi": int(mmsi), "seg": arr[start: start + seg_len],
                "seg_ts": ts[start: start + seg_len],
                "real_anomaly": True, "seg_flags": seg_flags,
                "fault_kinds": [_FLAG_NAMES[b] for b in (1, 2) if bits & b],
            })
            n_faulty += 1
        print(f"Faulty       : {n_faulty} real FLAGS-anomaly vessels included")

    # --- inject labelled synthetic anomalies into a subset ---
    def _stype(sh):
        return int(np.clip(round(float(sh["seg"][:SEQ_ENC, 6].mean()) * 7), 0, 7))
    eligible = [i for i, sh in enumerate(ships)
                if not sh.get("real_anomaly")                             # never inject into a real-fault vessel
                and float(sh["seg"][:SEQ_ENC, 2].mean()) * SOG_HI >= 3.0  # underway (mean SOG ≥ 3 kn)
                and _stype(sh) in (1, 2, 3, 4, 5, 6)]                 # named type only
    inj_rng  = np.random.default_rng(args.inject_seed)
    budget = min(int(round(len(ships) * max(0.0, min(1.0, args.inject_frac)))), len(eligible))
    pred   = [i for i in eligible if _stype(ships[i]) in (1, 2, 3)]   # cargo/tanker/passenger
    rest   = [i for i in eligible if _stype(ships[i]) in (4, 5, 6)]   # fishing/tug/pleasure
    inj_rng.shuffle(pred); inj_rng.shuffle(rest)
    n_rest = min(len(rest), max(3, round(budget * 0.25)))            # keep 5%
    n_pred = min(len(pred), budget - n_rest)
    inj_idx = set(pred[:n_pred] + rest[:n_rest])
    n_inject = len(inj_idx)
    for i, sh in enumerate(ships):
        sh.setdefault("real_anomaly", False)
        sh.setdefault("seg_flags", None)
        sh.setdefault("fault_kinds", None)
        if i in inj_idx:
            detectable = True
            atype_pool = inject_replay._DEVIATION_TYPES if _stype(sh) in (4, 5, 6) else None
            atype, severity, unit = inject_replay.pick_anomaly(
                inj_rng, detectable=detectable, types=atype_pool)
            sh["seg"]           = inject_replay.inject_segment(sh["seg"], atype, severity, inj_rng)
            sh["injected"]      = True
            sh["anomaly_type"]  = atype
            sh["severity"]      = severity
            sh["severity_unit"] = unit
            sh["mmsi"]          = 900000000 + i
        else:
            sh["injected"]      = False
            sh["anomaly_type"]  = None
            sh["severity"]      = None
            sh["severity_unit"] = None
    n_clean = len(ships) - n_inject - n_faulty
    print(f"Injected     : {n_inject} vessels  ({n_faulty} real-fault, {n_clean} clean)")

    # --- build strided inference tasks and run them batched ---
    issued_steps = list(range(SEQ_ENC, seg_len - SEQ_DEC + 1, pe))
    tasks = [(si, s) for si in range(len(ships)) for s in issued_steps]
    print(f"Predictions  : {len(issued_steps)} per ship × {len(ships)} ships "
          f"= {len(tasks)} AR inferences")

    # model - (ship_idx, issued_step) - (mu, sigma)
    results = {mk: {} for mk in SCOREBOARD_MODELS}
    for mk in SCOREBOARD_MODELS:
        for i in range(0, len(tasks), args.batch):
            chunk = tasks[i : i + args.batch]
            encs  = [ships[si]["seg"][s - SEQ_ENC : s] for si, s in chunk]
            outs  = batch_ar_predict(models[mk], encs, device)
            for (si, s), out in zip(chunk, outs):
                results[mk][(si, s)] = out
            done = min(i + args.batch, len(tasks))
            print(f"  inference [{mk:11}]: {done}/{len(tasks)}", end="\r")
        print()

    # --- assemble output ---
    K = max(1, int(args.persistence))
    def _persist(z):
        return np.convolve(z, np.ones(K) / K, mode="valid") if (K > 1 and len(z) >= K) else z
    _fuse = {"min": min, "mean": lambda v: sum(v) / len(v), "max": max}[args.consensus]

    clean_wz_by_group = {}          # ship-type group- windowed clean z values
    ships_raw = []
    for si, sh in enumerate(ships):
        seg  = sh["seg"]
        ship_type = int(np.clip(round(float(seg[:SEQ_ENC, 6].mean()) * 7), 0, 7))
        preds = []
        per_step_z = defaultdict(list)   # absolute step km deviation from every covering prediction
        for k, s in enumerate(issued_steps):
            mu, sigma = results["transformer"][(si, s)]
            dec_raw = denormalise_dec(mu)     # (SEQ_DEC, N_DEC)
            gt      = seg[s : s + SEQ_DEC]
            gt_lat  = gt[:, 0] * LAT_RNG + LAT_LO
            gt_lon  = gt[:, 1] * LON_RNG + LON_LO
            err_km  = haversine_km(dec_raw[:, 0], dec_raw[:, 1], gt_lat, gt_lon)  # (SEQ_DEC,)
            ade = float(err_km.mean())

            # Per-model ADE/FDE against the identical ground truth

            model_ade, model_fde, model_coords = {}, {}, {}
            for mk in SCOREBOARD_MODELS:
                dec_mk = denormalise_dec(results[mk][(si, s)][0])
                err_mk = haversine_km(dec_mk[:, 0], dec_mk[:, 1], gt_lat, gt_lon)
                model_ade[mk] = round(float(err_mk.mean()), 3)
                model_fde[mk] = round(float(err_mk[-1]),   3)
                if mk != "transformer":
                    model_coords[mk] = [[round(float(dec_mk[i, 0]), 5),
                                         round(float(dec_mk[i, 1]), 5)] for i in range(SEQ_DEC)]

            # Anomaly score is the raw km deviation (actual vs predicted)
            for i in range(SEQ_DEC):   # index this forecast by true step
                per_step_z[s + i].append(float(err_km[i]))

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
                "z_peak": round(float(err_km.max()), 3),  # per-pred peak km deviation
                "model_ade": model_ade,
                "model_fde": model_fde,
                "model_coords": model_coords,       # gru/tcn paths (transformer = coords)
            })

        # --consensus across overlapping forecasts, then persistence --
        #Per true step, fuse the z of every forecast that covered it (_fuse, per--consensus),
        #then smooth the resulting timeline with the K-step
        #persistence window and take the peak as this vessel's anomaly score.
        steps_sorted = sorted(per_step_z)
        zc = np.array([_fuse(per_step_z[t]) for t in steps_sorted]) if steps_sorted \
             else np.array([])
        wz = _persist(zc)
        if len(wz):
            wi        = int(np.argmax(wz))
            peak_step = steps_sorted[min(wi + (K - 1) // 2, len(steps_sorted) - 1)]
            trip_tick = int(peak_step)
            trip_lat  = round(float(seg[peak_step, 0] * LAT_RNG + LAT_LO), 5)
            trip_lon  = round(float(seg[peak_step, 1] * LON_RNG + LON_LO), 5)
        else:
            trip_tick, trip_lat, trip_lon = None, None, None
        # Anomaly SCORE is the vessel's mean km deviation (robust to transient clean
        # spikes that a peak would catch); the peak is kept only for the trip marker.
        mean_dev = float(np.mean([p["ade_km"] for p in preds]))
        if not sh["injected"] and not sh["real_anomaly"]:
            clean_wz_by_group.setdefault(ship_type, []).append(mean_dev)

        ships_raw.append({
            "mmsi":       sh["mmsi"],
            "ship_type":  ship_type,
            "seg":        seg,
            "seg_ts":     sh.get("seg_ts"),
            "preds":      preds,
            "ade_km":     mean_dev,
            "raw_z_peak": mean_dev,          # score = mean deviation (km)
            "trip_tick":  trip_tick,
            "trip_lat":   trip_lat,
            "trip_lon":   trip_lon,
            "rule_alerts": rule_detectors(seg, ship_type),
            "injected":      sh["injected"],
            "anomaly_type":  sh["anomaly_type"],
            "severity":      sh["severity"],
            "severity_unit": sh["severity_unit"],
            "real_anomaly":  sh["real_anomaly"],
            "fault_kinds":   sh["fault_kinds"],
        })

    #---- per-ship-type calibration (per-class km-deviation scale) ---
    # Each class's clean per-VESSEL peak-deviation p90 sets its scale (p90 -> 1.0)
    all_clean_wz = [v for lst in clean_wz_by_group.values() for v in lst]
    z_p90_global = float(np.median(all_clean_wz)) if all_clean_wz else 0.0
    z_scale_global = z_p90_global if z_p90_global > 1e-9 else 1.0
    MIN_GROUP = 8       # clean vessels needed to trust a class's own median
    z_scale_by_group = {}
    for g in range(8):
        vals = clean_wz_by_group.get(g, [])
        med  = float(np.median(vals)) if len(vals) >= MIN_GROUP else z_p90_global
        z_scale_by_group[g] = med if med > 1e-9 else 1.0

    for s in ships_raw:
        s["peak_cal"] = s["raw_z_peak"] / z_scale_by_group[s["ship_type"]]

    #Display threshold for the MAP/feed levels 
    #(2.5× each class's typical deviation), 
    #per-model detection metrics below use per-class F1-optimal σ instead.
    thr = float(args.threshold) if args.threshold is not None else 2.5

    def z_level(peak_cal):
        if   peak_cal <  thr:          return "none"
        elif peak_cal <  1.5 * thr:    return "mild"
        elif peak_cal <  2.5 * thr:    return "moderate"
        else:                          return "severe"

    ades = np.array([s["ade_km"] for s in ships_raw])
    p25, p75, p90 = (float(np.percentile(ades, q)) for q in (25, 75, 90))
    print(f"ADE          : p25={p25:.2f}  p75={p75:.2f}  p90={p90:.2f} km")
    print(f"Persistence  : K={K} steps")
    print(f"Display thr  : {thr:.2f}× class median  (map/feed levels, transformer)")

    ships_out = []
    for s in ships_raw:
        seg = s["seg"]
        ade = s["ade_km"]
        peak_cal = s["peak_cal"]
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
            "times":       ([None if (t != t) else int(t) for t in s["seg_ts"]]
                            if s.get("seg_ts") is not None else None),  # unix s per ping (NaN→null)
            "predictions": s["preds"],
            "ade_km":      round(ade, 2),
            "z_peak":      round(peak_cal, 2),
            "level":       level,
            "reason":      reason,
            "trip_tick":   s["trip_tick"],
            "trip_lat":    s["trip_lat"],
            "trip_lon":    s["trip_lon"],
            "rule_alerts": s["rule_alerts"],
            "injected":      s["injected"],
            "anomaly_type":  s["anomaly_type"],
            "severity":      s["severity"],
            "severity_unit": s["severity_unit"],
            "real_anomaly":  s["real_anomaly"],
            "fault_kinds":   s["fault_kinds"],
        })

    # Most-anomalous first (by calibrated z, the classifier).
    ships_out.sort(key=lambda x: x["z_peak"], reverse=True)
    for i, s in enumerate(ships_out):
        s["id"] = i

    level_counts, type_counts = {}, {}
    for s in ships_out:
        level_counts[s["level"]]    = level_counts.get(s["level"], 0) + 1
        type_counts[s["type_name"]] = type_counts.get(s["type_name"], 0) + 1

    # Fleet-wide aggregate scoreboard (final numbers over every prediction)
    # Severe-anomaly and injected (tampered) vessels are excluded so the board
    def _in_scoreboard(s):
        return s["level"] != "severe" and not s["injected"] and not s["real_anomaly"]
    scored_ships = [s for s in ships_out if _in_scoreboard(s)]
    n_excluded   = len(ships_out) - len(scored_ships)
    model_scores = {}
    for mk in SCOREBOARD_MODELS:
        ades = [p["model_ade"][mk] for s in scored_ships for p in s["predictions"]]
        fdes = [p["model_fde"][mk] for s in scored_ships for p in s["predictions"]]
        model_scores[mk] = {
            "ade": round(float(np.mean(ades)), 3) if ades else None,
            "fde": round(float(np.mean(fdes)), 3) if fdes else None,
            "n":   len(ades),
        }
    print("Scoreboard   : " + "  ".join(
        f"{mk}={model_scores[mk]['ade']}/{model_scores[mk]['fde']}km"
        for mk in SCOREBOARD_MODELS) + f"  (ADE/FDE, {n_excluded} excluded)")

    # --- anomaly-detection metrics, computed PER MODEL ---
    # Each model (GRU/TCN/Transformer) is a separate detector: its per-vessel mean
    # km deviation, calibrated per class, thresholded at each class's F1-optimal σ,
    # OR a (model-independent) physical rule. Two views per model:
    #   SET 1 (per ship type): each class at its own F1-optimal σ, model only,
    #     injected-vs-clean — the model's best-case per-class skill.
    #   SET 2 (combined): injected + real FLAGS anomalies vs clean, opt σ OR rule.
    # A vessel's deviation under model mk is the mean of its predictions' model_ade.
    inj   = [s for s in ships_out if s["injected"]]
    real  = [s for s in ships_out if s["real_anomaly"]]
    clean = [s for s in ships_out if not s["injected"] and not s["real_anomaly"]]

    def _metrics(pos, neg, flag):
        tp = sum(1 for s in pos if flag(s)); fn = len(pos) - tp
        fp = sum(1 for s in neg if flag(s)); tn = len(neg) - fp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        far  = fp / (fp + tn) if fp + tn else 0.0
        return {
            "n_pos": len(pos), "n_clean": len(neg),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3), "false_alarm_rate": round(far, 3),
        }

    def _model_detection(mk):
        # per-vessel deviation (km) under this model
        dev = {s["id"]: float(np.mean([p["model_ade"][mk] for p in s["predictions"]]))
               for s in ships_out}
        # calibrate: scale = clean median deviation per class (fallback global)
        by_g = {}
        for s in clean:
            by_g.setdefault(s["type"], []).append(dev[s["id"]])
        gmed  = float(np.median([v for l in by_g.values() for v in l])) if by_g else 1.0
        scale = {}
        for g in range(8):
            vals = by_g.get(g, [])
            m = float(np.median(vals)) if len(vals) >= 8 else gmed
            scale[g] = m if m > 1e-9 else 1.0
        cal = {s["id"]: dev[s["id"]] / scale[s["type"]] for s in ships_out}

        def best_thr(pos, neg):
            cand = sorted(set(cal[s["id"]] for s in pos + neg))
            best = (-1.0, cand[-1] if cand else 1.0)
            for t in cand:
                tp = sum(1 for s in pos if cal[s["id"]] >= t)
                fp = sum(1 for s in neg if cal[s["id"]] >= t)
                fn = len(pos) - tp
                pr = tp / (tp + fp) if tp + fp else 0.0
                rc = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
                if f1 > best[0] + 1e-9:
                    best = (f1, float(t))
            return best[1]

        thr_g = best_thr(inj, clean) if inj and clean else 2.5
        thr_by_g = {}
        for g in range(8):
            gi = [s for s in inj   if s["type"] == g]
            gc = [s for s in clean if s["type"] == g]
            thr_by_g[g] = best_thr(gi, gc) if gi and gc else thr_g

        # Manual σ floors
        thr_by_g[1] = max(thr_by_g[1], 3.0)  # Cargo   (group 1) floor high
        thr_by_g[4] = max(thr_by_g[4], 0.5)  # Fishing (group 4) low safety rail

        def flag_model(s): return cal[s["id"]] >= thr_by_g.get(s["type"], thr_g)
        def flag_opt(s):   return flag_model(s) or bool(s["rule_alerts"])

        # Per-vessel feed record for THIS model, from the SAME calibrated deviation
        # (cal) and per-class σ (thr_by_g) the metrics above use
        feed = {}
        for s in ships_out:
            sig = thr_by_g.get(s["type"], thr_g)
            r   = cal[s["id"]] / sig if sig > 1e-9 else 0.0
            lvl = ("severe"   if r >= 2.5 else
                   "moderate" if r >= 1.5 else
                   "mild"     if r >= 1.0 else "none")
            feed[s["id"]] = {"level": lvl, "sigma": round(cal[s["id"]], 2)}

        by_type = {}
        for g in range(8):
            gi = [s for s in inj   if s["type"] == g]
            gc = [s for s in clean if s["type"] == g]
            if not gi:
                continue
            d = _metrics(gi, gc, flag_model)
            d["opt_sigma"] = round(thr_by_g[g], 2)
            by_type[GROUP_NAMES[g]] = d

        combined = _metrics(inj + real, clean, flag_opt)
        combined["n_injected"] = len(inj)
        combined["n_real"]     = len(real)
        rf = sum(1 for s in real if flag_opt(s))
        return by_type, combined, {
            "n_real": len(real), "flagged": rf,
            "recall": round(rf / len(real), 3) if real else None,
        }, feed

    detection_by_type, detection_combined, real_detection = {}, {}, {}
    detection_feed = {}
    for mk in SCOREBOARD_MODELS:
        bt, cb, rd, fd = _model_detection(mk)
        detection_by_type[mk] = bt
        detection_combined[mk] = cb
        real_detection[mk] = rd
        detection_feed[mk] = fd

    # Attach each model's feed level + shown σ to every vessel, so the frontend
    # anomaly feed and map markers switch with the selected model AND match that
    # model's scoreline σ.
    for s in ships_out:
        s["model_levels"] = {mk: detection_feed[mk][s["id"]]["level"] for mk in SCOREBOARD_MODELS}
        s["model_sigma"]  = {mk: detection_feed[mk][s["id"]]["sigma"] for mk in SCOREBOARD_MODELS}

    if inj:
        for mk in SCOREBOARD_MODELS:
            c = detection_combined[mk]
            print(f"Detection {mk:11}: combined P{c['precision']:.2f}/R{c['recall']:.2f}/"
                  f"F1{c['f1']:.2f}  (inj={len(inj)}+real={len(real)} vs clean={len(clean)})")

    output = {
        "n_ships":        len(ships_out),
        "seq_enc":        SEQ_ENC,
        "seq_dec":        SEQ_DEC,
        "predict_every":  pe,
        "n_future_steps": n_fut,
        "window_len":     seg_len,
        "anomaly_threshold": round(thr, 2),
        "z_scale":        round(z_scale_global, 4),
        "persistence":    K,
        "consensus":      args.consensus,
        "per_type_calibrated": True,
        "n_faulty":       len(real),
        "real_detection": real_detection,
        "ade_p25":        round(p25, 2),
        "ade_p75":        round(p75, 2),
        "ade_p90":        round(p90, 2),
        "level_counts":   level_counts,
        "type_counts":    type_counts,
        "model_scores":   model_scores,
        "scoreboard_models": SCOREBOARD_MODELS,
        "scoreboard_excluded": n_excluded,
        "n_injected":     len(inj),
        "detection_by_type": detection_by_type,     # SET 1: per class at optimal σ, injected only
        "detection_combined": detection_combined,   # SET 2: injected + real anomalies vs clean
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
