"""
predict.py — inference and anomaly scoring against a SQLite AIS database.

Usage
-----
# Random vessel from the default test DB (data/worldwide.db)
python predict.py --random

# 10 random vessels on one map
python predict.py --random --count 10

# Specific vessel, window 5, flag if z-score exceeds 2.0
python predict.py --mmsi 219123456 --window 5 --anomaly-threshold 2.0

# All windows for a vessel (aggregate ADE/FDE stats)
python predict.py --mmsi 219123456 --all-windows

# Save a matplotlib PNG alongside the terminal output
python predict.py --random --plot --plot-out my_plot.png

# Run against the DMA training DB instead of WorldwideAIS
python predict.py --random --db data/dma.db

# Use a specific checkpoint and run on CPU
python predict.py --random --checkpoint checkpoints/best_model.pt --device cpu
"""
import argparse, math, os, random, sqlite3, sys
from collections import Counter
from datetime import datetime, timezone
import numpy as np
import torch

try:
    from global_land_mask import globe as _globe
    _LAND_MASK_AVAILABLE = True
except ImportError:
    _LAND_MASK_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from model  import ShipTrajectoryTransformer

FEAT     = cfg.feature_cols          # ["LAT","LON","SOG","COG_SIN","COG_COS","DT","SHIP_TYPE"]
NF       = cfg.n_features            # 7
N_ENC    = cfg.n_enc_features        # 7 — encoder receives all features
N_DEC    = cfg.n_dec_features        # 5 — decoder outputs LAT, LON, SOG, COG_SIN, COG_COS
N_DEC_IN = cfg.n_dec_input_features  # 6 — decoder input: motion + DT (index 5)
SEQ_ENC  = cfg.seq_len_enc
SEQ_DEC  = cfg.seq_len_dec
WINDOW   = SEQ_ENC + SEQ_DEC

# Normalisation arrays for the full 7-feature ping vector
_LO  = np.array([cfg.norm_bounds[f][0] for f in FEAT], dtype=np.float32)
_HI  = np.array([cfg.norm_bounds[f][1] for f in FEAT], dtype=np.float32)
_RNG = _HI - _LO

# Normalisation arrays for the 5 decoder output features (LAT…COG_COS)
_DEC_LO  = _LO[:N_DEC]
_DEC_RNG = _RNG[:N_DEC]

# One distinct colour per vessel slot (up to 10)
VESSEL_COLORS = [
    "#1f77b4", "#d62728", "#9467bd", "#ff7f0e",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf",
    "#2ca02c", "#7f7f7f",
]

# Bounding box: Europe + North Africa + Middle East
_REGION_BOUNDS = [[10, -30], [75, 65]]


def _itu_to_group(code):
    return cfg.ship_type_groups.get(int(code), 7)


def _dominant_group(rows):
    counts = Counter(_itu_to_group(r["SHIP_TYPE"]) for r in rows)
    return counts.most_common(1)[0][0]


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def normalise(arr):
    """Normalise a full NF-feature array to [0, 1]."""
    return np.clip((arr - _LO) / _RNG, 0.0, 1.0)


def denormalise(arr):
    """Denormalise a full NF-feature array from [0, 1] to physical units."""
    return np.clip(arr, 0.0, 1.0) * _RNG + _LO


def denormalise_dec(arr):
    """Denormalise a N_DEC-feature decoder prediction from [0, 1] to physical units."""
    return np.clip(arr, 0.0, 1.0) * _DEC_RNG + _DEC_LO


def _snap_to_water(lat: float, lon: float, max_radius_deg: float = 2.0) -> tuple[float, float]:
    """If (lat, lon) is on land, search outward in concentric rings and return
    the nearest water point. Falls back to the original position if none is
    found within max_radius_deg or if global_land_mask is not installed."""
    if not _LAND_MASK_AVAILABLE or not _globe.is_land(lat, lon):
        return lat, lon
    angles = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    radius = 0.1
    while radius <= max_radius_deg:
        for angle in angles:
            clat = lat + radius * math.cos(angle)
            clon = lon + radius * math.sin(angle)
            if not _globe.is_land(clat, clon):
                return clat, clon
        radius += 0.1
    return lat, lon


def load_model(checkpoint_path, device):
    """Load model using architecture hyperparameters saved in the checkpoint.

    Using the checkpoint's own config avoids an architecture mismatch when
    cfg.py is updated between the training run and an inference run.
    """
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # Fall back to current cfg values for keys absent in older checkpoints
    c = ckpt.get("config", {})
    model = ShipTrajectoryTransformer(
        n_features           = c.get("n_features",          NF),
        d_model              = c.get("d_model",             cfg.d_model),
        num_heads            = c.get("num_heads",           cfg.num_heads),
        num_layers           = c.get("num_layers",          cfg.num_layers),
        d_ff                 = c.get("d_ff",                cfg.d_ff),
        dropout              = 0.0,
        max_seq_length       = c.get("max_seq_length",      cfg.max_seq_length),
        n_enc_features       = c.get("n_enc_features",      N_ENC),
        n_dec_features       = c.get("n_dec_features",      N_DEC),
        n_dec_input_features = c.get("n_dec_input_features", N_DEC_IN),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


_LAT_MIN, _LAT_MAX, _LON_MIN, _LON_MAX = cfg.region_bounds

_REGION_SQL    = "LAT BETWEEN ? AND ? AND LON BETWEEN ? AND ?"
_REGION_PARAMS = (_LAT_MIN, _LAT_MAX, _LON_MIN, _LON_MAX)


def fetch_clean_track(db_path, mmsi):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute("PRAGMA table_info(ais)")
    cols      = {r["name"] for r in cur.fetchall()}
    has_flags = "FLAGS" in cols
    if has_flags:
        cur.execute(
            f"SELECT * FROM ais WHERE MMSI=? AND FLAGS=0 AND {_REGION_SQL} ORDER BY TIMESTAMP",
            (mmsi, *_REGION_PARAMS),
        )
    else:
        cur.execute(
            f"SELECT * FROM ais WHERE MMSI=? AND {_REGION_SQL} ORDER BY TIMESTAMP",
            (mmsi, *_REGION_PARAMS),
        )
    rows = cur.fetchall()
    conn.close()
    return rows


def list_mmsis(db_path):
    conn  = sqlite3.connect(db_path)
    cur   = conn.cursor()
    cur.execute(
        f"SELECT DISTINCT MMSI FROM ais WHERE {_REGION_SQL}",
        _REGION_PARAMS,
    )
    mmsis = [r[0] for r in cur.fetchall()]
    conn.close()
    return mmsis


def _cog_from_sincos(sin_val, cos_val):
    return math.degrees(math.atan2(float(sin_val), float(cos_val))) % 360


def _parse_ts(ts_str) -> float:
    try:
        return datetime.strptime(str(ts_str), "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc
        ).timestamp()
    except (ValueError, TypeError):
        return 0.0


def rows_to_array(rows):
    """Convert DB rows to a normalised (N, NF) array including all derived features."""
    group     = _dominant_group(rows)
    has_extra = len(rows) > 0 and "ROT" in rows[0].keys()
    out       = []
    prev_t = prev_lat = prev_lon = prev_cog = None

    for r in rows:
        lat = float(r["LAT"])
        lon = float(r["LON"])
        cog = float(r["COG"])
        t   = _parse_ts(r["TIMESTAMP"])

        dt   = 0.0 if prev_t   is None else max(0.0, t - prev_t)
        dlat = 0.0 if prev_lat is None else lat - prev_lat
        dlon = 0.0 if prev_lon is None else lon - prev_lon
        dcog = 0.0 if prev_cog is None else ((cog - prev_cog + 180.0) % 360.0) - 180.0

        prev_t = t; prev_lat = lat; prev_lon = lon; prev_cog = cog

        if has_extra:
            rot = float(r["ROT"]) if r["ROT"] is not None else 0.0
            rot = max(-127.0, min(127.0, rot))
            hdg = float(r["HEADING"]) if r["HEADING"] is not None else cog
            nav = float(r["NAV_STATUS"]) if r["NAV_STATUS"] is not None else 0.0
        else:
            rot, hdg, nav = 0.0, cog, 0.0

        cog_rad = math.radians(cog)
        hdg_rad = math.radians(hdg)
        out.append([
            lat, lon, float(r["SOG"]),
            math.sin(cog_rad), math.cos(cog_rad),
            dt, float(group),
            dlat, dlon, dcog,
            rot, math.sin(hdg_rad), math.cos(hdg_rad), nav,
        ])
    return normalise(np.array(out, dtype=np.float32))


def extract_windows(track_norm):
    windows = []
    for start in range(0, len(track_norm) - WINDOW + 1, cfg.window_stride):
        enc = track_norm[start          : start + SEQ_ENC]
        dec = track_norm[start + SEQ_ENC: start + WINDOW]
        windows.append((enc, dec))
    return windows


@torch.no_grad()
def predict_autoregressive(model, src_np, device):
    """Autoregressive decode for SEQ_DEC steps.

    Decoder input has N_DEC_IN=6 features [LAT, LON, SOG, COG_SIN, COG_COS, DT].
    The model predicts N_DEC=5 motion features; DT is always appended from the
    last observed inter-ping interval (src[-1, DT_IDX]) as an inference estimate.
    COG_SIN/COG_COS are renormalised to the unit circle before feedback.
    """
    src      = torch.from_numpy(src_np).unsqueeze(0).to(device)   # (1, SEQ_ENC, N_ENC)
    last_dt  = src[:, -1:, N_DEC:N_DEC_IN]                        # (1, 1, 1) — reuse last DT
    dec_input = src[:, -1:, :N_DEC_IN]                             # (1, 1, N_DEC_IN)
    mu_steps, sigma_steps = [], []

    for _ in range(SEQ_DEC):
        mu, log_var = model(src, dec_input)
        mu_last  = mu[:, -1:, :]                                   # (1, 1, N_DEC)
        std_last = (log_var[:, -1:, :] * 0.5).exp()

        # Clamp to valid normalised range then fix COG unit circle
        next_motion = mu_last.clamp(0.0, 1.0).clone()
        sin_raw     = next_motion[:, :, 3] * 2.0 - 1.0
        cos_raw     = next_motion[:, :, 4] * 2.0 - 1.0
        mag         = (sin_raw.pow(2) + cos_raw.pow(2)).sqrt().clamp(min=1e-8)
        next_motion[:, :, 3] = (sin_raw / mag + 1.0) * 0.5
        next_motion[:, :, 4] = (cos_raw / mag + 1.0) * 0.5

        # Snap to nearest water point if prediction landed on land
        lat_phys = float(next_motion[0, 0, 0]) * _DEC_RNG[0] + _DEC_LO[0]
        lon_phys = float(next_motion[0, 0, 1]) * _DEC_RNG[1] + _DEC_LO[1]
        lat_snapped, lon_snapped = _snap_to_water(lat_phys, lon_phys)
        next_motion[0, 0, 0] = float((lat_snapped - _DEC_LO[0]) / _DEC_RNG[0])
        next_motion[0, 0, 1] = float((lon_snapped - _DEC_LO[1]) / _DEC_RNG[1])

        next_step = torch.cat([next_motion, last_dt], dim=-1)      # (1, 1, N_DEC_IN)

        mu_steps.append(next_motion.squeeze(0).cpu().numpy())
        sigma_steps.append(std_last.squeeze(0).cpu().numpy())
        dec_input = torch.cat([dec_input, next_step], dim=1)

    return np.concatenate(mu_steps, axis=0), np.concatenate(sigma_steps, axis=0)


def _z_score(actual_norm_dec, mu_pred, sigma_pred):
    """Mean absolute z-score per timestep.

    actual_norm_dec : (T, N_DEC)  — first N_DEC features of the actual window
    mu_pred         : (T, N_DEC)
    sigma_pred      : (T, N_DEC)
    """
    return (np.abs(actual_norm_dec - mu_pred) / (sigma_pred + 1e-8)).mean(axis=1)


def print_window_result(dec_actual_norm, mu_pred, sigma_pred, threshold):
    # dec_actual_norm has NF features; predictions have N_DEC features
    dec_actual_raw = denormalise(dec_actual_norm)          # (T, NF)
    dec_pred_raw   = denormalise_dec(mu_pred)              # (T, N_DEC)
    dec_sigma_raw  = sigma_pred * _DEC_RNG                 # (T, N_DEC) in physical units
    z              = _z_score(dec_actual_norm[:, :N_DEC], mu_pred, sigma_pred)

    hdr = ("Step   Act LAT    Act LON    Pred LAT   Pred LON"
           "  Sigma LAT  Sigma LON  Dist km  z-score  Flag")
    print(hdr)
    print("-" * len(hdr))
    total_dist, flagged = 0.0, 0
    for i in range(SEQ_DEC):
        a_lat, a_lon = dec_actual_raw[i, 0], dec_actual_raw[i, 1]
        p_lat, p_lon = dec_pred_raw[i, 0],   dec_pred_raw[i, 1]
        s_lat, s_lon = dec_sigma_raw[i, 0],  dec_sigma_raw[i, 1]
        dist         = _haversine_km(a_lat, a_lon, p_lat, p_lon)
        total_dist  += dist
        flag         = "  <<" if z[i] > threshold else ""
        flagged     += bool(flag)
        print(f"{i+1:>4}  {a_lat:>9.4f} {a_lon:>10.4f}  "
              f"{p_lat:>9.4f} {p_lon:>10.4f}  "
              f"{s_lat:>9.4f} {s_lon:>9.4f}  "
              f"{dist:>7.3f}  {z[i]:>7.3f}{flag}")
    print()
    print(f"  Mean error : {total_dist/SEQ_DEC:.3f} km/step"
          f"   Total : {total_dist:.3f} km"
          f"   Flagged : {flagged}/{SEQ_DEC}")


def _add_vessel_to_map(m, full_track_raw, enc_actual_raw, dec_actual_norm, mu_pred, mmsi, color):
    """Add one vessel's tracks and markers to an existing folium map."""
    import folium  # type: ignore[import-untyped]

    dec_actual_raw = denormalise(dec_actual_norm)   # (T, NF)
    dec_pred_raw   = denormalise_dec(mu_pred)        # (T, N_DEC)

    def coords(arr):
        return [[float(r[0]), float(r[1])] for r in arr]

    folium.PolyLine(
        coords(full_track_raw),
        color="grey", weight=1.5, opacity=0.35,
        tooltip=f"MMSI {mmsi} — history",
    ).add_to(m)

    folium.PolyLine(
        coords(enc_actual_raw),
        color=color, weight=3, opacity=0.9,
        tooltip=f"MMSI {mmsi} — past (encoder)",
    ).add_to(m)
    for i, row in enumerate(enc_actual_raw):
        folium.CircleMarker(
            location=[float(row[0]), float(row[1])],
            radius=3, color=color, fill=True, fill_opacity=0.6,
            tooltip=f"MMSI {mmsi} — Encoder step {i+1}<br>Lat {row[0]:.4f}  Lon {row[1]:.4f}"
                    f"<br>SOG {row[2]:.1f} kn  COG {_cog_from_sincos(row[3], row[4]):.0f}°",
        ).add_to(m)

    last_enc   = enc_actual_raw[-1:]
    pred_route = [coords([last_enc[0]])[0]] + coords(dec_pred_raw)

    folium.PolyLine(
        pred_route,
        color=color, weight=3, opacity=0.9,
        dash_array="8",
        tooltip=f"MMSI {mmsi} — predicted",
    ).add_to(m)
    for i, row in enumerate(dec_pred_raw):
        dist = _haversine_km(
            dec_actual_raw[i, 0], dec_actual_raw[i, 1],
            row[0], row[1],
        )
        folium.CircleMarker(
            location=[float(row[0]), float(row[1])],
            radius=6, color=color, fill=True, fill_color="white",
            fill_opacity=0.85, weight=2,
            tooltip=(f"<b>MMSI {mmsi} — Predicted step {i+1}</b><br>"
                     f"Lat {row[0]:.4f}  Lon {row[1]:.4f}<br>"
                     f"SOG {row[2]:.1f} kn  COG {_cog_from_sincos(row[3], row[4]):.0f}°<br>"
                     f"Error vs actual: {dist:.3f} km"),
        ).add_to(m)

    actual_route = [coords([last_enc[0]])[0]] + coords(dec_actual_raw)

    folium.PolyLine(
        actual_route,
        color=color, weight=2, opacity=0.6,
        dash_array="2 6",
        tooltip=f"MMSI {mmsi} — ground truth",
    ).add_to(m)
    for i, row in enumerate(dec_actual_raw):
        dist = _haversine_km(
            row[0], row[1],
            dec_pred_raw[i, 0], dec_pred_raw[i, 1],
        )
        folium.CircleMarker(
            location=[float(row[0]), float(row[1])],
            radius=5, color=color, fill=True, fill_color=color,
            fill_opacity=0.4, weight=1,
            tooltip=(f"<b>MMSI {mmsi} — Actual step {i+1}</b><br>"
                     f"Lat {row[0]:.4f}  Lon {row[1]:.4f}<br>"
                     f"SOG {row[2]:.1f} kn  COG {_cog_from_sincos(row[3], row[4]):.0f}°<br>"
                     f"Error vs predicted: {dist:.3f} km"),
        ).add_to(m)

    folium.CircleMarker(
        location=[float(enc_actual_raw[0, 0]), float(enc_actual_raw[0, 1])],
        radius=8, color=color, fill=True, fill_color=color, fill_opacity=1.0,
        tooltip=f"MMSI {mmsi} — Encoder start",
    ).add_to(m)
    folium.CircleMarker(
        location=[float(dec_pred_raw[-1, 0]), float(dec_pred_raw[-1, 1])],
        radius=8, color=color, fill=True, fill_color="white", fill_opacity=0.9, weight=3,
        tooltip=f"MMSI {mmsi} — Predicted end",
    ).add_to(m)
    folium.CircleMarker(
        location=[float(dec_actual_raw[-1, 0]), float(dec_actual_raw[-1, 1])],
        radius=8, color=color, fill=True, fill_color=color, fill_opacity=0.5, weight=3,
        tooltip=f"MMSI {mmsi} — Actual end",
    ).add_to(m)


def save_folium_map(vessels, out_path):
    """Save a folium map for one or more vessels.

    vessels: list of (full_track_raw, enc_actual_raw, dec_actual_norm, mu_pred, mmsi)
    """
    try:
        import folium  # type: ignore[import-untyped]
    except ImportError:
        print("folium not installed -- run: pip install folium")
        return

    m = folium.Map(location=[42, 20], zoom_start=4, tiles="OpenStreetMap")
    m.fit_bounds(_REGION_BOUNDS)

    legend_items = ""
    for i, (_, _, _, _, mmsi) in enumerate(vessels):
        color = VESSEL_COLORS[i % len(VESSEL_COLORS)]
        legend_items += (
            f'<div style="display:flex;align-items:center;margin-bottom:4px">'
            f'<div style="width:20px;height:4px;background:{color};margin-right:6px"></div>'
            f'<span>MMSI {mmsi}</span></div>'
        )
    legend_html = (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
        'background:white;padding:10px 14px;border-radius:6px;'
        'box-shadow:0 1px 5px rgba(0,0,0,.4);font-family:sans-serif;font-size:13px">'
        f'<b>Vessels ({len(vessels)})</b><br><br>'
        f'{legend_items}'
        '<br><span style="font-size:11px;color:#555">'
        '&#9135; encoder &nbsp; - - predicted &nbsp; &#xB7;&#xB7; actual</span>'
        '</div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    for i, (full_track_raw, enc_actual_raw, dec_actual_norm, mu_pred, mmsi) in enumerate(vessels):
        color = VESSEL_COLORS[i % len(VESSEL_COLORS)]
        _add_vessel_to_map(m, full_track_raw, enc_actual_raw, dec_actual_norm, mu_pred, mmsi, color)

    m.save(out_path)
    print(f"  Map saved to {out_path}")

    import webbrowser
    webbrowser.open(f"file:///{os.path.abspath(out_path)}")


def save_plot(enc_actual_raw, dec_actual_norm, mu_pred, sigma_pred, mmsi, out_path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed -- skipping plot.")
        return
    dec_actual_raw = denormalise(dec_actual_norm)     # (T, NF)
    dec_pred_raw   = denormalise_dec(mu_pred)          # (T, N_DEC)
    dec_sigma_raw  = sigma_pred * _DEC_RNG
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(enc_actual_raw[:, 1], enc_actual_raw[:, 0],
            "b-o", ms=3, lw=1.5, label="Encoder (actual)", zorder=3)
    ax.plot(dec_actual_raw[:, 1], dec_actual_raw[:, 0],
            "g-o", ms=5, lw=2,   label="Actual future",    zorder=4)
    ax.plot(dec_pred_raw[:, 1],   dec_pred_raw[:, 0],
            "r--s", ms=5, lw=2,  label="Predicted future", zorder=5)
    for i in range(SEQ_DEC):
        ell = mpatches.Ellipse(
            (dec_pred_raw[i, 1], dec_pred_raw[i, 0]),
            width=2 * dec_sigma_raw[i, 1],
            height=2 * dec_sigma_raw[i, 0],
            color="red", alpha=0.12, zorder=2,
        )
        ax.add_patch(ell)
    for col, arr in [("g", dec_actual_raw), ("r", dec_pred_raw)]:
        ax.plot(
            [enc_actual_raw[-1, 1], arr[0, 1]],
            [enc_actual_raw[-1, 0], arr[0, 0]],
            color=col, ls="--", lw=1, alpha=0.5,
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Trajectory comparison -- MMSI {mmsi}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Plot saved to {out_path}")
    plt.close(fig)


def _process_vessel(mmsi, args, model):
    """Fetch, predict, and print results for one vessel. Returns map data tuple or None."""
    rows = fetch_clean_track(args.db, mmsi)
    if len(rows) < WINDOW:
        print(f"  Skipping MMSI {mmsi}: only {len(rows)} clean pings (need {WINDOW}).")
        return None

    track_norm = rows_to_array(rows)
    windows    = extract_windows(track_norm)
    if not windows:
        print(f"  Skipping MMSI {mmsi}: no complete windows ({len(rows)} pings).")
        return None

    print(f"\n{'='*60}")
    print(f"  MMSI {mmsi}  |  {len(rows)} pings  |  {len(windows)} windows")

    w_idx = args.window
    if w_idx >= len(windows):
        w_idx = 0
    enc, dec   = windows[w_idx]
    start_ping = w_idx * cfg.window_stride
    print(f"  Window {w_idx}  |  encoder pings {start_ping} - {start_ping + SEQ_ENC - 1}"
          f"  |  predicting next {SEQ_DEC} steps\n")

    mu_pred, sigma_pred = predict_autoregressive(model, enc, args.device)
    enc_raw        = denormalise(enc)
    full_track_raw = denormalise(track_norm)

    print_window_result(dec, mu_pred, sigma_pred, args.anomaly_threshold)

    return (full_track_raw, enc_raw, dec, mu_pred, mmsi)


def main():
    parser = argparse.ArgumentParser(description="Predict and compare AIS trajectories.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--mmsi",   type=int)
    src.add_argument("--random", action="store_true")
    parser.add_argument("--count",              type=int,   default=1,
                        help="Number of random vessels to process (use with --random).")
    parser.add_argument("--window",            type=int,   default=0)
    parser.add_argument("--all-windows",       action="store_true")
    parser.add_argument("--plot",              action="store_true")
    parser.add_argument("--plot-out",          type=str,   default="trajectory.png")
    parser.add_argument("--anomaly-threshold", type=float, default=3.0)
    parser.add_argument("--db",                type=str,   default=cfg.test_db_path,
                        help="AIS database to predict against "
                             f"(default: {cfg.test_db_path} — WorldwideAIS test set).")
    parser.add_argument("--checkpoint",        type=str,   default=cfg.checkpoint_path)
    parser.add_argument("--device",            type=str,   default=cfg.device)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    print(f"  Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, args.device)
    print(f"  Device: {args.device}")

    # ------------------------------------------------------------------ multi-vessel
    if args.random and args.count > 1:
        all_mmsis = list_mmsis(args.db)
        if not all_mmsis:
            sys.exit("No MMSIs found in database.")
        n = min(args.count, len(all_mmsis))
        selected = random.sample(all_mmsis, n)
        print(f"  Selected {n} random MMSIs: {selected}")

        vessels = []
        for mmsi in selected:
            result = _process_vessel(mmsi, args, model)
            if result is not None:
                vessels.append(result)

        if not vessels:
            sys.exit("No vessels had enough data to process.")

        print(f"\n{'='*60}")
        print(f"  Saving map for {len(vessels)} vessel(s) ...")
        save_folium_map(vessels, "prediction.html")
        return

    # ------------------------------------------------------------------ single vessel
    if args.random:
        mmsis = list_mmsis(args.db)
        if not mmsis:
            sys.exit("No MMSIs found in database.")
        args.mmsi = random.choice(mmsis)
        print(f"  Selected MMSI: {args.mmsi}")

    rows = fetch_clean_track(args.db, args.mmsi)
    if len(rows) < WINDOW:
        sys.exit(f"MMSI {args.mmsi} has only {len(rows)} clean pings (need {WINDOW}).")

    track_norm = rows_to_array(rows)
    windows    = extract_windows(track_norm)
    if not windows:
        sys.exit(f"No complete windows for MMSI {args.mmsi} ({len(rows)} pings).")

    print(f"  MMSI {args.mmsi}  |  {len(rows)} pings  |  {len(windows)} windows")

    if args.all_windows:
        total_dist = total_z = 0.0
        total_steps = flagged_tot = 0
        for wi, (enc, dec) in enumerate(windows):
            mu_pred, sigma_pred = predict_autoregressive(model, enc, args.device)
            z        = _z_score(dec[:, :N_DEC], mu_pred, sigma_pred)
            dec_raw  = denormalise(dec)
            pred_raw = denormalise_dec(mu_pred)
            for i in range(SEQ_DEC):
                dist         = _haversine_km(
                    dec_raw[i, 0], dec_raw[i, 1],
                    pred_raw[i, 0], pred_raw[i, 1],
                )
                total_dist  += dist
                total_z     += z[i]
                total_steps += 1
                flagged_tot += int(z[i] > args.anomaly_threshold)
            if (wi + 1) % 10 == 0:
                print(f"    {wi+1}/{len(windows)} windows ...", flush=True)
        mean_dist = total_dist / total_steps
        mean_z    = total_z    / total_steps
        flag_pct  = 100.0 * flagged_tot / total_steps
        print(f"\n  === Aggregate over {len(windows)} windows ({total_steps} steps) ===")
        print(f"  Mean positional error : {mean_dist:.3f} km/step")
        print(f"  Mean z-score          : {mean_z:.3f}")
        print(f"  Flagged steps         : {flagged_tot} / {total_steps}"
              f"  ({flag_pct:.1f}%,  threshold={args.anomaly_threshold})")
        return

    w_idx = args.window
    if w_idx >= len(windows):
        sys.exit(f"Window index {w_idx} out of range (0 - {len(windows)-1}).")

    enc, dec   = windows[w_idx]
    start_ping = w_idx * cfg.window_stride
    print(f"\n  Window {w_idx}  |  encoder pings {start_ping} - {start_ping + SEQ_ENC - 1}"
          f"  |  predicting next {SEQ_DEC} steps\n")

    mu_pred, sigma_pred = predict_autoregressive(model, enc, args.device)
    enc_raw        = denormalise(enc)
    full_track_raw = denormalise(track_norm)

    print_window_result(dec, mu_pred, sigma_pred, args.anomaly_threshold)

    save_folium_map([(full_track_raw, enc_raw, dec, mu_pred, args.mmsi)], "prediction.html")

    if args.plot:
        save_plot(enc_raw, dec, mu_pred, sigma_pred, args.mmsi, args.plot_out)


if __name__ == "__main__":
    main()
