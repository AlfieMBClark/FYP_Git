"""
inject_anomalies.py
-------------------
Build a *labelled* anomaly evaluation set by injecting synthetic suspicious
behaviour into real held-out trajectories.

Why this is necessary
---------------------
The FLAGS bitmask written at ingest (speed > 30 kn, implied speed > 60 kn,
COG/SOG mismatch) labels corrupt *pings*, not suspicious *behaviour*, and
prepare_dataset.py filters those rows out when it builds clean tracks — which
is why the 2023 hold-out contains exactly one flagged window. There is no
usable ground truth, so precision/recall cannot be measured against it. Nor
would it be meaningful: scoring a deep model against three hand-written
if-statements only asks whether it can rediscover our own thresholds.

This is the standard response to a known gap in the field — Wolsing et al.
(2022) note the absence of any standardised labelled AIS anomaly dataset.

How the injection works
-----------------------
The 90-ping encoder context is left completely untouched; only the 10-ping
future — the segment the anomaly score is computed against — is rewritten. That
has a useful consequence: the model sees an *identical* input for the clean and
the injected version of a window, so its prediction (mu, sigma) is the same for
both. Clean and injected are therefore a matched pair differing only in what
actually happened next, which is exactly the comparison an anomaly detector
should be judged on.

Crucially the future is not re-simulated from scratch — it is built by
*transforming the real displacement vectors* between consecutive true positions
(rotate them, scale them, or add an offset). The injected track therefore keeps
the same GPS noise and reporting cadence as genuine data. Re-simulating would
have left a reconstruction fingerprint that the detector could latch onto,
inflating recall for the wrong reason.

Injected behaviours (mapped to the taxonomy in the literature review, and to
what the interviewees described as operationally significant):

  course_deviation  route deviation                   theta = 15/30/45/60/90 deg
  u_turn            turning anomaly / course reversal theta = 120/150/180 deg
  speed_drop        speed anomaly (deceleration)      factor = 0.5/0.25/0.1/0.0
  speed_surge       speed anomaly (acceleration)      factor = 1.5/2/3/4
  loitering         loitering anomaly                 4/6/8/10 stopped steps
  position_jump     spoof-like location tampering     0.5/1/2/5/10 km

Usage
-----
python inject_anomalies.py                       # 4000 clean + 4000 injected
python inject_anomalies.py --n-sources 8000 --variants-per-source all
"""

import argparse
import os

import numpy as np

from common import (
    SEQ_ENC, WINDOW_LEN,
    IDX_LAT, IDX_LON, IDX_SOG, IDX_CSIN, IDX_CCOS, IDX_DT,
    IDX_DLAT, IDX_DLON, IDX_DCOG, IDX_ROT, IDX_HSIN, IDX_HCOS,
    normalise, denormalise, cog_from_sincos, wrap180, haversine_km,
)

_HERE       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN  = os.path.join(_HERE, "eval_set.npz")
DEFAULT_OUT = os.path.join(_HERE, "injected_set.npz")

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320   # scaled by cos(lat) at use

# type -> (severity levels, human-readable unit)
ANOMALY_TYPES = {
    "course_deviation": ([15.0, 30.0, 45.0, 60.0, 90.0], "deg off course"),
    "u_turn":           ([120.0, 150.0, 180.0],          "deg reversal"),
    "speed_drop":       ([0.5, 0.25, 0.1, 0.0],          "x speed"),
    "speed_surge":      ([1.5, 2.0, 3.0, 4.0],           "x speed"),
    "loitering":        ([4.0, 6.0, 8.0, 10.0],          "stopped steps"),
    "position_jump":    ([0.5, 1.0, 2.0, 5.0, 10.0],     "km jump"),
}


# ── displacement-vector helpers ───────────────────────────────────────────────

def to_km(dlat, dlon, lat):
    """Degree displacement -> local (east, north) km."""
    dx = dlon * KM_PER_DEG_LON * np.cos(np.radians(lat))
    dy = dlat * KM_PER_DEG_LAT
    return dx, dy


def to_deg(dx, dy, lat):
    """Local (east, north) km -> degree displacement."""
    dlon = dx / (KM_PER_DEG_LON * max(np.cos(np.radians(lat)), 1e-6))
    dlat = dy / KM_PER_DEG_LAT
    return dlat, dlon


def rebuild_future(win_phys, lats, lons, sogs, cogs):
    """Write a new future (positions, speeds, courses) into a physical-units
    window, recomputing every derived feature so the window stays internally
    consistent with how prepare_dataset.py would have built it.

    win_phys : (100, 14) physical units — modified in place and returned.
    lats/lons/sogs/cogs : length-10 arrays for the future pings.
    """
    w = win_phys.copy()
    prev_lat = w[SEQ_ENC - 1, IDX_LAT]
    prev_lon = w[SEQ_ENC - 1, IDX_LON]
    prev_cog = cog_from_sincos(w[SEQ_ENC - 1, IDX_CSIN], w[SEQ_ENC - 1, IDX_CCOS])

    for i in range(WINDOW_LEN - SEQ_ENC):
        r   = SEQ_ENC + i
        lat, lon = float(lats[i]), float(lons[i])
        sog = float(np.clip(sogs[i], 0.0, 30.0))
        cog = float(cogs[i] % 360.0)
        dt  = float(w[r, IDX_DT])
        rad = np.radians(cog)

        dcog = wrap180(cog - prev_cog)
        rot  = np.clip(dcog / (dt / 60.0), -127.0, 127.0) if dt > 0 else 0.0

        w[r, IDX_LAT]  = lat
        w[r, IDX_LON]  = lon
        w[r, IDX_SOG]  = sog
        w[r, IDX_CSIN] = np.sin(rad)
        w[r, IDX_CCOS] = np.cos(rad)
        # IDX_DT, IDX_TYPE, IDX_NAV: reporting cadence, vessel class and the
        # (crew-entered, unreliable) nav status are properties of the vessel,
        # not of the manoeuvre — an anomalous ship still reports on schedule.
        w[r, IDX_DLAT] = lat - prev_lat
        w[r, IDX_DLON] = lon - prev_lon
        w[r, IDX_DCOG] = dcog
        w[r, IDX_ROT]  = rot
        w[r, IDX_HSIN] = np.sin(rad)
        w[r, IDX_HCOS] = np.cos(rad)

        prev_lat, prev_lon, prev_cog = lat, lon, cog

    return w


def real_future_deltas(win_phys):
    """Displacement of each future ping from the one before it (the last encoder
    ping anchors the first step). Returns anchor lat/lon plus per-step
    (dlat, dlon), sog and cog arrays taken from the genuine data."""
    anchor_lat = float(win_phys[SEQ_ENC - 1, IDX_LAT])
    anchor_lon = float(win_phys[SEQ_ENC - 1, IDX_LON])

    fut  = win_phys[SEQ_ENC:]
    lats = np.concatenate([[anchor_lat], fut[:, IDX_LAT]])
    lons = np.concatenate([[anchor_lon], fut[:, IDX_LON]])

    dlat = np.diff(lats)
    dlon = np.diff(lons)
    sog  = fut[:, IDX_SOG].astype(np.float64).copy()
    cog  = cog_from_sincos(fut[:, IDX_CSIN], fut[:, IDX_CCOS]).astype(np.float64)
    return anchor_lat, anchor_lon, dlat, dlon, sog, cog


def accumulate(anchor_lat, anchor_lon, dlat, dlon):
    """Integrate per-step degree displacements from the anchor."""
    lats = anchor_lat + np.cumsum(dlat)
    lons = anchor_lon + np.cumsum(dlon)
    return lats, lons


# ── the injections ────────────────────────────────────────────────────────────

def inject(win_phys, atype, severity, rng):
    """Return a new physical-units window with an anomaly written into the
    future segment. The encoder segment is never touched."""
    a_lat, a_lon, dlat, dlon, sog, cog = real_future_deltas(win_phys)
    n = len(sog)

    if atype in ("course_deviation", "u_turn"):
        # Rotate every real displacement vector by theta in the local tangent
        # plane. Speed profile and GPS noise survive untouched; only the
        # direction of travel changes.
        theta = np.radians(severity * (1 if rng.random() < 0.5 else -1))
        c, s  = np.cos(theta), np.sin(theta)
        new_dlat, new_dlon = np.empty(n), np.empty(n)
        lat_cursor = a_lat
        for i in range(n):
            dx, dy = to_km(dlat[i], dlon[i], lat_cursor)
            rx, ry = c * dx - s * dy, s * dx + c * dy      # 2-D rotation
            new_dlat[i], new_dlon[i] = to_deg(rx, ry, lat_cursor)
            lat_cursor += new_dlat[i]
        lats, lons = accumulate(a_lat, a_lon, new_dlat, new_dlon)
        new_sog = sog                                       # unchanged
        new_cog = cog + np.degrees(theta)                   # course follows

    elif atype in ("speed_drop", "speed_surge"):
        # Scale the magnitude of each displacement, and the reported SOG with
        # it, so kinematics and positions stay mutually consistent.
        #
        # SOG saturates at the 30 kn feature bound. Scaling the displacement by
        # the requested factor while the reported speed clips would leave a
        # track that moves faster than it claims to — an inconsistency the
        # detector could exploit without learning anything about behaviour. So
        # the displacement follows the *effective* per-step factor that survives
        # the clip, and a surge on an already-fast ship simply saturates.
        new_sog = np.clip(sog * severity, 0.0, 30.0)
        f_eff   = np.where(sog > 0.1, new_sog / np.maximum(sog, 1e-6), severity)
        lats, lons = accumulate(a_lat, a_lon, dlat * f_eff, dlon * f_eff)
        new_cog = cog                                       # heading unchanged

    elif atype == "loitering":
        # Vessel stops: displacement collapses to metre-scale GPS jitter, speed
        # falls to a drift, heading wanders as it swings on the spot.
        k = int(severity)                                   # steps spent stopped
        new_dlat, new_dlon = dlat.copy(), dlon.copy()
        new_sog, new_cog   = sog.copy(), cog.copy()
        heading = float(cog[0])
        for i in range(n - k, n):
            jitter_km = rng.normal(0.0, 0.01, size=2)       # ~10 m
            new_dlat[i], new_dlon[i] = to_deg(jitter_km[0], jitter_km[1], a_lat)
            new_sog[i] = abs(rng.normal(0.0, 0.2))          # drifting, ~0 kn
            heading   += rng.normal(0.0, 25.0)              # swinging at anchor
            new_cog[i] = heading
        lats, lons = accumulate(a_lat, a_lon, new_dlat, new_dlon)

    elif atype == "position_jump":
        # Spoof-like teleport: one step is displaced by `severity` km on a
        # random bearing, then the vessel carries on with its real motion.
        # SOG/COG are left reporting the original values — the self-reported
        # kinematics no longer explain the position, which is the signature
        # both Owen and Lily described.
        j       = rng.integers(2, n - 1)
        bearing = rng.uniform(0, 2 * np.pi)
        off_km  = severity
        new_dlat, new_dlon = dlat.copy(), dlon.copy()
        jlat, jlon = to_deg(off_km * np.sin(bearing), off_km * np.cos(bearing), a_lat)
        new_dlat[j] += jlat
        new_dlon[j] += jlon
        lats, lons = accumulate(a_lat, a_lon, new_dlat, new_dlon)
        new_sog, new_cog = sog, cog

    else:
        raise ValueError(f"unknown anomaly type: {atype}")

    return rebuild_future(win_phys, lats, lons, new_sog, new_cog)


# ── physical-consistency self-check ───────────────────────────────────────────

def implied_vs_reported(win_phys):
    """Mean |implied SOG - reported SOG| over the future, in knots.

    Implied SOG is derived from the distance actually covered between pings.
    For a physically consistent track the two agree to within GPS/ping noise.
    A large gap means the reported kinematics do not explain the movement —
    which is deliberately true for position_jump and should be true for
    nothing else. This is the check that stops us from accidentally building
    anomalies that are trivially detectable for the wrong reason."""
    fut = win_phys[SEQ_ENC:]
    lat = np.concatenate([[win_phys[SEQ_ENC - 1, IDX_LAT]], fut[:, IDX_LAT]])
    lon = np.concatenate([[win_phys[SEQ_ENC - 1, IDX_LON]], fut[:, IDX_LON]])
    dt  = fut[:, IDX_DT]
    dist_km = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    ok = dt > 0
    if not ok.any():
        return np.nan
    implied_kn = dist_km[ok] / (dt[ok] / 3600.0) * 0.539957
    return float(np.abs(implied_kn - fut[ok, IDX_SOG]).mean())


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Inject synthetic anomalies into held-out windows.")
    ap.add_argument("--eval-set",  default=DEFAULT_IN)
    ap.add_argument("--out",       default=DEFAULT_OUT)
    ap.add_argument("--n-sources", type=int, default=4000,
                    help="Source windows. Each yields 1 clean + N injected variants.")
    ap.add_argument("--variants-per-source", default="1",
                    help="'1' (round-robin over types, gives a balanced 1:1 set) "
                         "or 'all' (every type per source, 6:1 — report PR-AUC, not accuracy).")
    ap.add_argument("--min-sog", type=float, default=3.0,
                    help="Only inject into windows whose real future averages at least "
                         "this speed. Slowing down a ship that is already stopped is "
                         "not an anomaly, and would poison the labels.")
    ap.add_argument("--min-disp-km", type=float, default=0.3,
                    help="Minimum real 10-step displacement, same reasoning.")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not os.path.exists(args.eval_set):
        raise SystemExit(f"Missing {args.eval_set} — run build_eval_set.py first.")

    d      = np.load(args.eval_set, allow_pickle=True)
    W      = d["windows"]
    mmsi   = d["mmsi"]
    group  = d["group"]
    seen   = d["seen"]
    sog_f  = d["future_mean_sog"]
    disp_f = d["future_disp_km"]

    eligible = np.where((sog_f >= args.min_sog) & (disp_f >= args.min_disp_km))[0]
    print(f"\n  Loaded {len(W):,} windows; {len(eligible):,} are under way "
          f"(>= {args.min_sog} kn and >= {args.min_disp_km} km moved) and can host "
          f"a kinematic anomaly.")
    if len(eligible) == 0:
        raise SystemExit("No eligible source windows.")

    rng = np.random.default_rng(args.seed)
    n_src = min(args.n_sources, len(eligible))
    src_idx = rng.choice(eligible, size=n_src, replace=False)

    types = list(ANOMALY_TYPES)
    out_w, out_lbl, out_type, out_sev, out_src, out_mmsi, out_grp, out_seen = \
        [], [], [], [], [], [], [], []

    consistency = {"clean": []}

    for si, wi in enumerate(src_idx):
        phys = denormalise(W[wi])

        # the negative: the real, untouched window
        out_w.append(W[wi]); out_lbl.append(0); out_type.append("clean")
        out_sev.append(0.0); out_src.append(si)
        out_mmsi.append(mmsi[wi]); out_grp.append(group[wi]); out_seen.append(seen[wi])
        consistency["clean"].append(implied_vs_reported(phys))

        chosen = types if args.variants_per_source == "all" else [types[si % len(types)]]
        for atype in chosen:
            levels, _ = ANOMALY_TYPES[atype]
            sev = float(rng.choice(levels))
            inj = inject(phys, atype, sev, rng)

            inj_norm = normalise(inj)
            # The physical round trip (denormalise -> inject -> normalise) is
            # lossy in the last float bits. Restore the encoder verbatim from
            # the source so the clean and injected variants are bit-identical
            # in the context the model actually reads — evaluate_anomaly.py
            # relies on that to reuse one inference across a source's variants,
            # and asserts it.
            inj_norm[:SEQ_ENC] = W[wi][:SEQ_ENC]

            out_w.append(inj_norm); out_lbl.append(1); out_type.append(atype)
            out_sev.append(sev); out_src.append(si)
            out_mmsi.append(mmsi[wi]); out_grp.append(group[wi]); out_seen.append(seen[wi])
            consistency.setdefault(atype, []).append(implied_vs_reported(inj))

        if (si + 1) % 1000 == 0:
            print(f"    {si + 1:,}/{n_src:,} sources", flush=True)

    Wout = np.stack(out_w).astype(np.float32)
    lbl  = np.asarray(out_lbl,  dtype=np.int8)
    typ  = np.asarray(out_type)
    sev  = np.asarray(out_sev,  dtype=np.float32)
    src  = np.asarray(out_src,  dtype=np.int32)

    np.savez_compressed(
        args.out,
        windows=Wout, label=lbl, atype=typ, severity=sev, source_id=src,
        mmsi=np.asarray(out_mmsi, dtype=np.int64),
        group=np.asarray(out_grp, dtype=np.int8),
        seen=np.asarray(out_seen, dtype=bool),
        seed=args.seed,
    )

    print(f"\n  ── Injected set ──")
    print(f"  Windows : {len(Wout):,}   clean {int((lbl == 0).sum()):,}  "
          f"anomalous {int((lbl == 1).sum()):,}")
    for t in types:
        n = int((typ == t).sum())
        if n:
            _, unit = ANOMALY_TYPES[t]
            print(f"    {t:<18} {n:>6,}   severity in {ANOMALY_TYPES[t][0]} ({unit})")

    print(f"\n  ── Physical-consistency check ──")
    print(f"  Mean |implied speed - reported speed| over the future segment.")
    print(f"  Clean data sets the noise floor. Every injected type except")
    print(f"  position_jump should sit near that floor — if one does not, the")
    print(f"  injection has left an artefact the detector could cheat on.")
    floor = float(np.nanmean(consistency['clean']))
    for t, vals in consistency.items():
        v = float(np.nanmean(vals))
        note = ""
        if t == "position_jump":
            note = "  <- inconsistent by design (the spoof signature)"
        elif t != "clean" and v > floor * 3 + 1.0:
            note = "  <- WARNING: artefact, investigate before trusting recall"
        print(f"    {t:<18} {v:6.2f} kn{note}")

    print(f"\n  Saved -> {args.out}\n")


if __name__ == "__main__":
    main()
