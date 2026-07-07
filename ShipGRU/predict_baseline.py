"""
predict_baseline.py — inference and anomaly scoring for ShipGRUBaseline.
Mirrors predict.py interface; adds --compare mode for side-by-side Transformer vs GRU.

Usage
-----
python predict_baseline.py --random --plot
python predict_baseline.py --mmsi 219123456 --window 0
python predict_baseline.py --random --count 10
python predict_baseline.py --mmsi 219123456 --all-windows
python predict_baseline.py --compare                     # same window, both models
python predict_baseline.py --compare --random --plot     # random vessel, both models, map
"""

import argparse
import math
import os
import random
import sys
from collections import Counter

import numpy as np
import torch

_HERE        = os.path.dirname(os.path.abspath(__file__))
_TRANSFORMER = os.path.abspath(os.path.join(_HERE, "..", "ShipTransformer"))
_DEFAULT_MAP_OUT     = os.path.join(_HERE, "prediction.html")
_DEFAULT_COMPARE_OUT = os.path.join(_HERE, "comparison.html")
sys.path.insert(0, _HERE)
sys.path.insert(0, _TRANSFORMER)
os.chdir(_TRANSFORMER)  # config.py / dataset.py use relative paths rooted here
from config         import cfg
from baseline_model import ShipGRUBaseline

FEAT     = cfg.feature_cols
NF       = cfg.n_features            # 14
N_DEC    = cfg.n_dec_features        # 5
N_DEC_IN = cfg.n_dec_input_features  # 6
SEQ_ENC  = cfg.seq_len_enc           # 90
SEQ_DEC  = cfg.seq_len_dec           # 10
WINDOW   = SEQ_ENC + SEQ_DEC         # 100

_LO  = np.array([cfg.norm_bounds[f][0] for f in FEAT], dtype=np.float32)
_HI  = np.array([cfg.norm_bounds[f][1] for f in FEAT], dtype=np.float32)
_RNG = _HI - _LO

_DEC_LO  = _LO[:N_DEC]
_DEC_RNG = _RNG[:N_DEC]

VESSEL_COLORS = [
    "#1f77b4", "#d62728", "#9467bd", "#ff7f0e",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf",
    "#2ca02c", "#7f7f7f",
]

_REGION_BOUNDS = [[50, -5], [66, 20]]


# ── DB helpers (shared with predict.py) ───────────────────────────────────────

import sqlite3
from datetime import datetime, timezone


def _itu_to_group(code):
    return cfg.ship_type_groups.get(int(code), 7)


def _dominant_group(rows):
    counts = Counter(_itu_to_group(r["SHIP_TYPE"]) for r in rows)
    return counts.most_common(1)[0][0]


def _parse_ts(ts_str) -> float:
    try:
        return datetime.strptime(str(ts_str), "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc
        ).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _cog_from_sincos(sin_val, cos_val):
    return math.degrees(math.atan2(float(sin_val), float(cos_val))) % 360


def normalise(arr):
    return np.clip((arr - _LO) / _RNG, 0.0, 1.0)


def denormalise(arr):
    return np.clip(arr, 0.0, 1.0) * _RNG + _LO


def denormalise_dec(arr):
    return np.clip(arr, 0.0, 1.0) * _DEC_RNG + _DEC_LO


_LAT_MIN, _LAT_MAX, _LON_MIN, _LON_MAX = cfg.region_bounds
_REGION_SQL    = "LAT BETWEEN ? AND ? AND LON BETWEEN ? AND ?"
_REGION_PARAMS = (_LAT_MIN, _LAT_MAX, _LON_MIN, _LON_MAX)


def fetch_clean_track(db_path, mmsi):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
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
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute(f"SELECT DISTINCT MMSI FROM ais WHERE {_REGION_SQL}", _REGION_PARAMS)
    mmsis = [r[0] for r in cur.fetchall()]
    conn.close()
    return mmsis


def rows_to_array(rows):
    """Convert DB rows to normalised (N, NF) array including all derived features."""
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
        enc = track_norm[start:          start + SEQ_ENC]
        dec = track_norm[start + SEQ_ENC: start + WINDOW]
        windows.append((enc, dec))
    return windows


# ── GRU model loader ──────────────────────────────────────────────────────────

def load_gru_model(checkpoint_path, device):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    c     = ckpt.get("config", {})
    model = ShipGRUBaseline(
        n_features   = c.get("n_features",   NF),
        dec_features = c.get("dec_features", N_DEC_IN),
        out_features = c.get("out_features", N_DEC),
        hidden_size  = c.get("hidden_size",  256),
        num_layers   = c.get("num_layers",   2),
        dropout      = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ── Transformer model loader (mirrors predict.py) ────────────────────────────

def load_transformer_model(checkpoint_path, device):
    from model import ShipTrajectoryTransformer
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    c     = ckpt.get("config", {})
    model = ShipTrajectoryTransformer(
        n_features           = c.get("n_features",           NF),
        d_model              = c.get("d_model",              cfg.d_model),
        num_heads            = c.get("num_heads",            cfg.num_heads),
        num_layers           = c.get("num_layers",           cfg.num_layers),
        d_ff                 = c.get("d_ff",                 cfg.d_ff),
        dropout              = 0.0,
        max_seq_length       = c.get("max_seq_length",       cfg.max_seq_length),
        n_enc_features       = c.get("n_enc_features",       NF),
        n_dec_features       = c.get("n_dec_features",       N_DEC),
        n_dec_input_features = c.get("n_dec_input_features", N_DEC_IN),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_gru(model: ShipGRUBaseline, src_np: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """Autoregressive GRU inference; reuses last observed DT for all future steps."""
    src     = torch.from_numpy(src_np).unsqueeze(0).to(device)   # (1, SEQ_ENC, NF)
    seed    = src[:, -1:, :N_DEC]                                 # (1, 1, 5)
    last_dt = src[:, -1:, N_DEC:N_DEC_IN]                         # (1, 1, 1)
    tgt_dt  = last_dt.expand(-1, SEQ_DEC, -1)                     # (1, 10, 1) — replicate DT

    mu, log_var = model.predict(src, tgt_dt, seed)
    sigma = (log_var * 0.5).exp()
    return mu.squeeze(0).cpu().numpy(), sigma.squeeze(0).cpu().numpy()


@torch.no_grad()
def predict_transformer(model, src_np: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """Autoregressive Transformer inference (mirrors predict.py behaviour)."""
    src       = torch.from_numpy(src_np).unsqueeze(0).to(device)
    last_dt   = src[:, -1:, N_DEC:N_DEC_IN]
    dec_input = src[:, -1:, :N_DEC_IN]
    mu_steps, sigma_steps = [], []

    for _ in range(SEQ_DEC):
        mu, log_var = model(src, dec_input)
        mu_last  = mu[:, -1:, :]
        std_last = (log_var[:, -1:, :] * 0.5).exp()

        next_motion = mu_last.clamp(0.0, 1.0).clone()
        sin_raw = next_motion[:, :, 3] * 2.0 - 1.0
        cos_raw = next_motion[:, :, 4] * 2.0 - 1.0
        mag     = (sin_raw.pow(2) + cos_raw.pow(2)).sqrt().clamp(min=1e-8)
        next_motion[:, :, 3] = (sin_raw / mag + 1.0) * 0.5
        next_motion[:, :, 4] = (cos_raw / mag + 1.0) * 0.5

        next_step = torch.cat([next_motion, last_dt], dim=-1)
        mu_steps.append(next_motion.squeeze(0).cpu().numpy())
        sigma_steps.append(std_last.squeeze(0).cpu().numpy())
        dec_input = torch.cat([dec_input, next_step], dim=1)

    return np.concatenate(mu_steps, axis=0), np.concatenate(sigma_steps, axis=0)


# ── Scoring & printing ────────────────────────────────────────────────────────

def _z_score(actual_norm, mu_pred, sigma_pred):
    """Mean absolute z-score per timestep. (T,)"""
    return (np.abs(actual_norm[:, :N_DEC] - mu_pred) / (sigma_pred + 1e-8)).mean(axis=1)


def print_window_result(dec_actual_norm, mu_pred, sigma_pred, threshold):
    dec_actual_raw = denormalise(dec_actual_norm)
    dec_pred_raw   = denormalise_dec(mu_pred)
    dec_sigma_raw  = sigma_pred * _DEC_RNG
    z              = _z_score(dec_actual_norm, mu_pred, sigma_pred)

    hdr = ("Step   Act LAT    Act LON    Pred LAT   Pred LON"
           "  Sigma LAT  Sigma LON  Dist km  z-score  Flag")
    print(hdr)
    print("-" * len(hdr))
    total_dist = flagged = 0
    for i in range(SEQ_DEC):
        a_lat, a_lon = dec_actual_raw[i, 0], dec_actual_raw[i, 1]
        p_lat, p_lon = dec_pred_raw[i,  0], dec_pred_raw[i,  1]
        s_lat, s_lon = dec_sigma_raw[i, 0], dec_sigma_raw[i, 1]
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


# ── Compare mode ──────────────────────────────────────────────────────────────

def print_compare_table(mmsi, w_idx, dec_actual_norm,
                        tf_mu, tf_sigma,
                        gru_mu, gru_sigma,
                        threshold):
    """Side-by-side Transformer vs GRU comparison table."""
    tf_pred  = denormalise_dec(tf_mu)
    gru_pred = denormalise_dec(gru_mu)
    actual   = denormalise(dec_actual_norm)

    tf_z     = _z_score(dec_actual_norm, tf_mu,  tf_sigma)
    gru_z    = _z_score(dec_actual_norm, gru_mu, gru_sigma)

    tf_dists  = [_haversine_km(actual[i, 0], actual[i, 1], tf_pred[i,  0], tf_pred[i,  1]) for i in range(SEQ_DEC)]
    gru_dists = [_haversine_km(actual[i, 0], actual[i, 1], gru_pred[i, 0], gru_pred[i, 1]) for i in range(SEQ_DEC)]

    tf_ade  = float(np.mean(tf_dists))
    gru_ade = float(np.mean(gru_dists))
    tf_fde  = tf_dists[-1]
    gru_fde = gru_dists[-1]
    tf_mz   = float(tf_z.mean())
    gru_mz  = float(gru_z.mean())

    tf_verdict  = "SUSPICIOUS" if tf_mz  > threshold else "NORMAL"
    gru_verdict = "SUSPICIOUS" if gru_mz > threshold else "NORMAL"

    sep = "─" * 49
    print(f"\nMMSI: {mmsi}  Window: {w_idx}")
    print(sep)
    print(f"{'':20}{'Transformer':>14}      {'GRU Baseline':>14}")
    print(sep)
    print(f"{'Mean ADE':<20}{tf_ade:>11.2f} km   {gru_ade:>11.2f} km")
    print(f"{'Final FDE':<20}{tf_fde:>11.2f} km   {gru_fde:>11.2f} km")
    print(f"{'Mean Z-score':<20}{tf_mz:>14.2f}   {gru_mz:>14.2f}")
    print(f"{'Verdict':<20}{tf_verdict:>14}   {gru_verdict:>14}")
    print(sep)
    for i in range(SEQ_DEC):
        print(f"Step {i+1:>2}             {tf_dists[i]:>8.1f} km   {gru_dists[i]:>8.1f} km")
    print(sep)


# ── Map / plot helpers ────────────────────────────────────────────────────────

def save_compare_map(enc_raw, dec_actual_norm, tf_mu, gru_mu, mmsi, out_path):
    """Folium map: red=Transformer, orange=GRU, green=actual."""
    try:
        import folium
    except ImportError:
        print("folium not installed — run: pip install folium")
        return

    dec_actual_raw = denormalise(dec_actual_norm)
    tf_pred_raw    = denormalise_dec(tf_mu)
    gru_pred_raw   = denormalise_dec(gru_mu)

    centre_lat = float(enc_raw[-1, 0])
    centre_lon = float(enc_raw[-1, 1])
    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=8, tiles="OpenStreetMap")

    def coords(arr):
        return [[float(r[0]), float(r[1])] for r in arr]

    # Encoder history
    folium.PolyLine(coords(enc_raw), color="blue", weight=3,
                    tooltip=f"MMSI {mmsi} — encoder").add_to(m)

    # Ground truth
    actual_route = [coords(enc_raw[-1:])[0]] + coords(dec_actual_raw)
    folium.PolyLine(actual_route, color="green", weight=2,
                    dash_array="4 6", tooltip=f"MMSI {mmsi} — actual").add_to(m)
    for i, row in enumerate(dec_actual_raw):
        folium.CircleMarker(
            [float(row[0]), float(row[1])], radius=5, color="green",
            fill=True, fill_opacity=0.5,
            tooltip=f"Actual step {i+1}  Lat {row[0]:.4f}  Lon {row[1]:.4f}",
        ).add_to(m)

    # Transformer prediction
    tf_route = [coords(enc_raw[-1:])[0]] + coords(tf_pred_raw)
    folium.PolyLine(tf_route, color="red", weight=2, dash_array="8",
                    tooltip="Transformer").add_to(m)
    for i, row in enumerate(tf_pred_raw):
        folium.CircleMarker(
            [float(row[0]), float(row[1])], radius=6, color="red",
            fill=True, fill_color="white", fill_opacity=0.9,
            tooltip=f"Transformer step {i+1}  Lat {row[0]:.4f}  Lon {row[1]:.4f}",
        ).add_to(m)

    # GRU prediction
    gru_route = [coords(enc_raw[-1:])[0]] + coords(gru_pred_raw)
    folium.PolyLine(gru_route, color="orange", weight=2, dash_array="8",
                    tooltip="GRU Baseline").add_to(m)
    for i, row in enumerate(gru_pred_raw):
        folium.CircleMarker(
            [float(row[0]), float(row[1])], radius=6, color="orange",
            fill=True, fill_color="white", fill_opacity=0.9,
            tooltip=f"GRU step {i+1}  Lat {row[0]:.4f}  Lon {row[1]:.4f}",
        ).add_to(m)

    legend_html = (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
        'background:white;padding:10px 14px;border-radius:6px;'
        'box-shadow:0 1px 5px rgba(0,0,0,.4);font-family:sans-serif;font-size:13px">'
        f'<b>MMSI {mmsi}</b><br><br>'
        '<div style="color:blue">&#9135; Encoder history</div>'
        '<div style="color:green">&#xB7;&#xB7; Ground truth</div>'
        '<div style="color:red">- - Transformer predicted</div>'
        '<div style="color:orange">- - GRU Baseline predicted</div>'
        '</div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(out_path)
    print(f"  Map saved to {out_path}")

    import webbrowser
    webbrowser.open(f"file:///{os.path.abspath(out_path)}")


def _add_vessel_to_map(m, full_track_raw, enc_actual_raw, dec_actual_norm, mu_pred, mmsi, color):
    """Add one vessel's tracks and markers to an existing folium map."""
    import folium

    dec_actual_raw = denormalise(dec_actual_norm)
    dec_pred_raw   = denormalise_dec(mu_pred)

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
        color=color, weight=3, opacity=0.9, dash_array="8",
        tooltip=f"MMSI {mmsi} — GRU predicted",
    ).add_to(m)
    for i, row in enumerate(dec_pred_raw):
        dist = _haversine_km(
            dec_actual_raw[i, 0], dec_actual_raw[i, 1], row[0], row[1],
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
        color=color, weight=2, opacity=0.6, dash_array="2 6",
        tooltip=f"MMSI {mmsi} — ground truth",
    ).add_to(m)
    for i, row in enumerate(dec_actual_raw):
        dist = _haversine_km(
            row[0], row[1], dec_pred_raw[i, 0], dec_pred_raw[i, 1],
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
        import folium
    except ImportError:
        print("folium not installed — run: pip install folium")
        return

    m = folium.Map(location=[58, 10], zoom_start=5, tiles="OpenStreetMap")
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
        f'<b>GRU Baseline — Vessels ({len(vessels)})</b><br><br>'
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


# ── Per-vessel helper ─────────────────────────────────────────────────────────

def _process_vessel(mmsi, args, gru_model):
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
    print(f"  Window {w_idx}  |  encoder pings {start_ping} – {start_ping + SEQ_ENC - 1}"
          f"  |  predicting next {SEQ_DEC} steps\n")

    gru_mu, gru_sigma = predict_gru(gru_model, enc, args.device)
    enc_raw        = denormalise(enc)
    full_track_raw = denormalise(track_norm)

    print_window_result(dec, gru_mu, gru_sigma, args.anomaly_threshold)

    return (full_track_raw, enc_raw, dec, gru_mu, mmsi)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict with GRU baseline or compare with Transformer.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--mmsi",   type=int)
    src.add_argument("--random", action="store_true")

    parser.add_argument("--count",              type=int,   default=1)
    parser.add_argument("--window",             type=int,   default=0)
    parser.add_argument("--all-windows",        action="store_true")
    parser.add_argument("--plot",               action="store_true")
    parser.add_argument("--plot-out",           type=str,   default=_DEFAULT_MAP_OUT)
    parser.add_argument("--anomaly-threshold",  type=float, default=3.0)
    parser.add_argument("--compare",            action="store_true",
                        help="Run both Transformer and GRU on same window, side-by-side.")
    _default_gru_ckpt = os.path.join(_HERE, "checkpoints", "baseline_model.pt")
    _default_tf_ckpt  = os.path.join(_TRANSFORMER, cfg.checkpoint_path)
    parser.add_argument("--db",                 type=str,   default=cfg.test_db_path)
    parser.add_argument("--checkpoint",         type=str,   default=_default_gru_ckpt)
    parser.add_argument("--tf-checkpoint",      type=str,   default=_default_tf_ckpt,
                        help="Transformer checkpoint for --compare mode.")
    parser.add_argument("--device",             type=str,   default=cfg.device)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"GRU checkpoint not found: {args.checkpoint}")

    print(f"  Loading GRU model from {args.checkpoint} ...")
    gru_model = load_gru_model(args.checkpoint, args.device)

    tf_model = None
    if args.compare:
        if not os.path.exists(args.tf_checkpoint):
            sys.exit(f"Transformer checkpoint not found: {args.tf_checkpoint}")
        print(f"  Loading Transformer model from {args.tf_checkpoint} ...")
        tf_model = load_transformer_model(args.tf_checkpoint, args.device)

    # ── multi-vessel (--random --count N) ────────────────────────────────────
    if args.random and args.count > 1:
        all_mmsis = list_mmsis(args.db)
        if not all_mmsis:
            sys.exit("No MMSIs found in database.")
        n        = min(args.count, len(all_mmsis))
        selected = random.sample(all_mmsis, n)
        print(f"  Selected {n} random MMSIs: {selected}")

        vessels = []
        for mmsi in selected:
            result = _process_vessel(mmsi, args, gru_model)
            if result is not None:
                vessels.append(result)

        if not vessels:
            sys.exit("No vessels had enough data to process.")

        print(f"\n{'='*60}")
        print(f"  Saving map for {len(vessels)} vessel(s) ...")
        save_folium_map(vessels, args.plot_out)
        return

    # ── resolve single MMSI ───────────────────────────────────────────────────
    if args.random:
        all_mmsis = list_mmsis(args.db)
        if not all_mmsis:
            sys.exit("No MMSIs found in database.")
        args.mmsi = random.choice(all_mmsis)
        print(f"  Selected MMSI: {args.mmsi}")

    rows = fetch_clean_track(args.db, args.mmsi)
    if len(rows) < WINDOW:
        sys.exit(f"MMSI {args.mmsi} has only {len(rows)} clean pings (need {WINDOW}).")

    track_norm = rows_to_array(rows)
    windows    = extract_windows(track_norm)
    if not windows:
        sys.exit(f"No complete windows for MMSI {args.mmsi} ({len(rows)} pings).")

    print(f"  MMSI {args.mmsi}  |  {len(rows)} pings  |  {len(windows)} windows")

    # ── --all-windows aggregate stats ─────────────────────────────────────────
    if args.all_windows:
        total_dist = total_z = 0.0
        total_steps = flagged_tot = 0
        for wi, (enc, dec) in enumerate(windows):
            mu_pred, sigma_pred = predict_gru(gru_model, enc, args.device)
            z        = _z_score(dec, mu_pred, sigma_pred)
            dec_raw  = denormalise(dec)
            pred_raw = denormalise_dec(mu_pred)
            for i in range(SEQ_DEC):
                dist         = _haversine_km(
                    dec_raw[i, 0], dec_raw[i, 1], pred_raw[i, 0], pred_raw[i, 1]
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
        print(f"\n  === GRU Aggregate over {len(windows)} windows ({total_steps} steps) ===")
        print(f"  Mean positional error : {mean_dist:.3f} km/step")
        print(f"  Mean z-score          : {mean_z:.3f}")
        print(f"  Flagged steps         : {flagged_tot} / {total_steps}  ({flag_pct:.1f}%)")
        return

    # ── single window ─────────────────────────────────────────────────────────
    w_idx = args.window
    if w_idx >= len(windows):
        w_idx = 0
    enc, dec   = windows[w_idx]
    start_ping = w_idx * cfg.window_stride
    print(f"\n  Window {w_idx}  |  encoder pings {start_ping} – {start_ping + SEQ_ENC - 1}"
          f"  |  predicting next {SEQ_DEC} steps\n")

    gru_mu, gru_sigma = predict_gru(gru_model, enc, args.device)
    enc_raw        = denormalise(enc)
    full_track_raw = denormalise(track_norm)

    if args.compare and tf_model is not None:
        tf_mu, tf_sigma = predict_transformer(tf_model, enc, args.device)
        print_compare_table(args.mmsi, w_idx, dec,
                            tf_mu, tf_sigma, gru_mu, gru_sigma,
                            args.anomaly_threshold)
        if args.plot:
            out = args.plot_out if args.plot_out != _DEFAULT_MAP_OUT else _DEFAULT_COMPARE_OUT
            save_compare_map(enc_raw, dec, tf_mu, gru_mu, args.mmsi, out)
    else:
        print("GRU Baseline:")
        print_window_result(dec, gru_mu, gru_sigma, args.anomaly_threshold)
        save_folium_map([(full_track_raw, enc_raw, dec, gru_mu, args.mmsi)], args.plot_out)


if __name__ == "__main__":
    main()
