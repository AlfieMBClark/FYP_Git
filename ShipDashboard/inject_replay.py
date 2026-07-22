"""
inject_replay.py
----------------
Inject a labelled synthetic anomaly into a Replay vessel segment, for
precompute.py. 
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ShipTransformer")))
from config import cfg   # noqa: E402

# --- feature indices (match cfg.feature_cols) ---
IDX_LAT, IDX_LON, IDX_SOG, IDX_CSIN, IDX_CCOS = 0, 1, 2, 3, 4
IDX_DT, IDX_TYPE = 5, 6
IDX_DLAT, IDX_DLON, IDX_DCOG, IDX_ROT = 7, 8, 9, 10
IDX_HSIN, IDX_HCOS, IDX_NAV = 11, 12, 13

SEQ_ENC = cfg.seq_len_enc

# Physical <-> normalised bounds (per feature) from the single source of truth.
_LAT_LO, _LAT_HI = cfg.norm_bounds["LAT"];  _LAT_RNG = _LAT_HI - _LAT_LO
_LON_LO, _LON_HI = cfg.norm_bounds["LON"];  _LON_RNG = _LON_HI - _LON_LO
_SOG_LO, _SOG_HI = cfg.norm_bounds["SOG"];  _SOG_RNG = _SOG_HI - _SOG_LO
_DLAT_LO, _DLAT_HI = cfg.norm_bounds["dLAT"]; _DLAT_RNG = _DLAT_HI - _DLAT_LO
_DLON_LO, _DLON_HI = cfg.norm_bounds["dLON"]; _DLON_RNG = _DLON_HI - _DLON_LO
_DCOG_LO, _DCOG_HI = cfg.norm_bounds["dCOG"]; _DCOG_RNG = _DCOG_HI - _DCOG_LO
_ROT_LO, _ROT_HI = cfg.norm_bounds["ROT"];  _ROT_RNG = _ROT_HI - _ROT_LO
_DT_LO, _DT_HI = cfg.norm_bounds["DT"];      _DT_RNG = _DT_HI - _DT_LO

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320  #scaled by cos(lat) at use

# type -> (severity levels, human-readable unit) — copied from ShipEval so Replay
# and the offline eval inject the same behaviours.
ANOMALY_TYPES = {
    "course_deviation": ([15.0, 30.0, 45.0, 60.0, 90.0], "deg off course"),
    "u_turn":           ([120.0, 150.0, 180.0],          "deg reversal"),
    "speed_drop":       ([0.5, 0.25, 0.1, 0.0],          "x speed"),
    "speed_surge":      ([1.5, 2.0, 3.0, 4.0],           "x speed"),
    "loitering":        ([4.0, 6.0, 8.0, 10.0],          "stopped steps"),
    "position_jump":    ([0.5, 1.0, 2.0, 5.0, 10.0],     "km jump"),
}


#-- geometry helpers (from ShipEval/inject_anomalies.py) ---

def _to_km(dlat, dlon, lat):
    dx = dlon * KM_PER_DEG_LON * np.cos(np.radians(lat))
    dy = dlat * KM_PER_DEG_LAT
    return dx, dy


def _to_deg(dx, dy, lat):
    dlon = dx / (KM_PER_DEG_LON * max(np.cos(np.radians(lat)), 1e-6))
    dlat = dy / KM_PER_DEG_LAT
    return dlat, dlon


def _wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _cog_deg(sin_norm, cos_norm):
    """COG in degrees from normalised sin/cos features (each [0,1] -> [-1,1])."""
    s = sin_norm * 2.0 - 1.0
    c = cos_norm * 2.0 - 1.0
    return np.degrees(np.arctan2(s, c))


#---core injection ---

def _apply(atype, severity, a_lat, a_lon, dlat, dlon, sog, cog, rng):
    """Return (lats, lons, new_sog, new_cog) physical future arrays with the
    anomaly written in. Mirrors ShipEval/inject_anomalies.inject()."""
    n = len(sog)

    if atype in ("course_deviation", "u_turn"):
        theta = np.radians(severity * (1 if rng.random() < 0.5 else -1))
        c, s = np.cos(theta), np.sin(theta)
        new_dlat, new_dlon = np.empty(n), np.empty(n)
        lat_cursor = a_lat
        for i in range(n):
            dx, dy = _to_km(dlat[i], dlon[i], lat_cursor)
            rx, ry = c * dx - s * dy, s * dx + c * dy
            new_dlat[i], new_dlon[i] = _to_deg(rx, ry, lat_cursor)
            lat_cursor += new_dlat[i]
        lats = a_lat + np.cumsum(new_dlat)
        lons = a_lon + np.cumsum(new_dlon)
        return lats, lons, sog, cog + np.degrees(theta)

    if atype in ("speed_drop", "speed_surge"):
        new_sog = np.clip(sog * severity, 0.0, 30.0)
        f_eff = np.where(sog > 0.1, new_sog / np.maximum(sog, 1e-6), severity)
        lats = a_lat + np.cumsum(dlat * f_eff)
        lons = a_lon + np.cumsum(dlon * f_eff)
        return lats, lons, new_sog, cog

    if atype == "loitering":
        k = int(severity)
        new_dlat, new_dlon = dlat.copy(), dlon.copy()
        new_sog, new_cog = sog.copy(), cog.copy()
        heading = float(cog[0])
        for i in range(max(0, n - k), n):
            jit = rng.normal(0.0, 0.01, size=2)      # ~10 m
            new_dlat[i], new_dlon[i] = _to_deg(jit[0], jit[1], a_lat)
            new_sog[i] = abs(rng.normal(0.0, 0.2))   # drifting ~0 kn
            heading += rng.normal(0.0, 25.0)         # swinging at anchor
            new_cog[i] = heading
        lats = a_lat + np.cumsum(new_dlat)
        lons = a_lon + np.cumsum(new_dlon)
        return lats, lons, new_sog, new_cog

    if atype == "position_jump":
        j = int(rng.integers(2, max(3, n - 1)))
        bearing = rng.uniform(0, 2 * np.pi)
        new_dlat, new_dlon = dlat.copy(), dlon.copy()
        jlat, jlon = _to_deg(severity * np.sin(bearing), severity * np.cos(bearing), a_lat)
        new_dlat[j] += jlat
        new_dlon[j] += jlon
        lats = a_lat + np.cumsum(new_dlat)
        lons = a_lon + np.cumsum(new_dlon)
        return lats, lons, sog, cog   # SOG/COG keep reporting original (spoof signature)

    raise ValueError(f"unknown anomaly type: {atype}")


def inject_segment(seg_norm, atype, severity, rng):
    """Return a copy of `seg_norm` with `atype`/`severity` written into the
    future (rows >= SEQ_ENC). The encoder history is never touched."""
    w = seg_norm.copy()
    fut = w[SEQ_ENC:]

    #Denormalise the future to physical units, plus the anchor (last enc ping).
    a_lat = float(w[SEQ_ENC - 1, IDX_LAT]) * _LAT_RNG + _LAT_LO
    a_lon = float(w[SEQ_ENC - 1, IDX_LON]) * _LON_RNG + _LON_LO
    a_cog = float(_cog_deg(w[SEQ_ENC - 1, IDX_CSIN], w[SEQ_ENC - 1, IDX_CCOS]))

    lat = fut[:, IDX_LAT] * _LAT_RNG + _LAT_LO
    lon = fut[:, IDX_LON] * _LON_RNG + _LON_LO
    sog = (fut[:, IDX_SOG] * _SOG_RNG + _SOG_LO).astype(np.float64)
    cog = _cog_deg(fut[:, IDX_CSIN], fut[:, IDX_CCOS]).astype(np.float64)

    #Per-step displacement then apply anomaly.
    lats_full = np.concatenate([[a_lat], lat])
    lons_full = np.concatenate([[a_lon], lon])
    dlat = np.diff(lats_full)
    dlon = np.diff(lons_full)

    lats, lons, new_sog, new_cog = _apply(atype, severity, a_lat, a_lon,
                                          dlat, dlon, sog, cog, rng)

    # Rebuild the future rows
    prev_lat, prev_lon, prev_cog = a_lat, a_lon, a_cog
    for i in range(len(lats)):
        r = SEQ_ENC + i
        la, lo = float(lats[i]), float(lons[i])
        so = float(np.clip(new_sog[i], 0.0, 30.0))
        co = float(new_cog[i] % 360.0)
        rad = np.radians(co)
        dt = float(fut[i, IDX_DT]) * _DT_RNG + _DT_LO
        dcog = _wrap180(co - prev_cog)
        rot = float(np.clip(dcog / (dt / 60.0), -127.0, 127.0)) if dt > 0 else 0.0

        w[r, IDX_LAT]  = np.clip((la - _LAT_LO) / _LAT_RNG, 0.0, 1.0)
        w[r, IDX_LON]  = np.clip((lo - _LON_LO) / _LON_RNG, 0.0, 1.0)
        w[r, IDX_SOG]  = np.clip((so - _SOG_LO) / _SOG_RNG, 0.0, 1.0)
        w[r, IDX_CSIN] = (np.sin(rad) + 1.0) * 0.5
        w[r, IDX_CCOS] = (np.cos(rad) + 1.0) * 0.5
        w[r, IDX_DLAT] = np.clip(((la - prev_lat) - _DLAT_LO) / _DLAT_RNG, 0.0, 1.0)
        w[r, IDX_DLON] = np.clip(((lo - prev_lon) - _DLON_LO) / _DLON_RNG, 0.0, 1.0)
        w[r, IDX_DCOG] = np.clip((dcog - _DCOG_LO) / _DCOG_RNG, 0.0, 1.0)
        w[r, IDX_ROT]  = np.clip((rot - _ROT_LO) / _ROT_RNG, 0.0, 1.0)
        w[r, IDX_HSIN] = (np.sin(rad) + 1.0) * 0.5
        w[r, IDX_HCOS] = (np.cos(rad) + 1.0) * 0.5
        # IDX_DT, IDX_TYPE, IDX_NAV: reporting cadence + vessel class are
        prev_lat, prev_lon, prev_cog = la, lo, co

    return w


_DEVIATION_TYPES = ("course_deviation", "u_turn", "speed_surge", "position_jump")


def pick_anomaly(rng, detectable=False, types=None):
    keys = list(types) if types else list(ANOMALY_TYPES.keys())
    atype = str(rng.choice(keys))
    severities, unit = ANOMALY_TYPES[atype]
    if detectable and len(severities) > 1:
        exp = float(os.environ.get("INJECT_DETECT_EXP", "3"))
        w = np.arange(1, len(severities) + 1, dtype=float) ** exp
        severity = float(rng.choice(severities, p=w / w.sum()))
    else:
        severity = float(rng.choice(severities))
    return atype, severity, unit
