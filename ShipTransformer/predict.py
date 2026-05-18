import argparse, math, os, random, sqlite3, sys
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from model import AISTransformer

FEAT    = cfg.feature_cols
NF      = cfg.n_features
SEQ_ENC = cfg.seq_len_enc
SEQ_DEC = cfg.seq_len_dec
WINDOW  = SEQ_ENC + SEQ_DEC

_LO  = np.array([cfg.norm_bounds[f][0] for f in FEAT], dtype=np.float32)
_HI  = np.array([cfg.norm_bounds[f][1] for f in FEAT], dtype=np.float32)
_RNG = _HI - _LO


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
    return np.clip((arr - _LO) / _RNG, 0.0, 1.0)


def denormalise(arr):
    return arr * _RNG + _LO


def load_model(checkpoint_path, device):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model = AISTransformer(
        n_features=NF,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        dropout=0.0,
        max_seq_length=cfg.max_seq_length,
        seq_len_enc=SEQ_ENC,
        seq_len_dec=SEQ_DEC,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def fetch_clean_track(db_path, mmsi):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute("PRAGMA table_info(ais_data)")
    cols      = {r["name"] for r in cur.fetchall()}
    has_flags = "FLAGS" in cols
    if has_flags:
        cur.execute(
            "SELECT * FROM ais_data WHERE MMSI=? AND FLAGS=0 ORDER BY TIMESTAMP",
            (mmsi,),
        )
    else:
        cur.execute(
            "SELECT * FROM ais_data WHERE MMSI=? ORDER BY TIMESTAMP",
            (mmsi,),
        )
    rows = cur.fetchall()
    conn.close()
    return rows


def list_mmsis(db_path):
    conn  = sqlite3.connect(db_path)
    cur   = conn.cursor()
    cur.execute("SELECT DISTINCT MMSI FROM ais_data")
    mmsis = [r[0] for r in cur.fetchall()]
    conn.close()
    return mmsis


def rows_to_array(rows):
    group = _dominant_group(rows)
    out   = [[r["LAT"], r["LON"], r["SOG"], r["COG"], float(group)] for r in rows]
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
    src       = torch.from_numpy(src_np).unsqueeze(0).to(device)
    dec_input = src[:, -1:, :]
    mu_steps, sigma_steps = [], []
    for _ in range(SEQ_DEC):
        mu, log_var = model(src, dec_input)
        mu_last     = mu[:, -1:, :]
        std_last    = (log_var[:, -1:, :] * 0.5).exp()
        mu_steps.append(mu_last.squeeze(0).cpu().numpy())
        sigma_steps.append(std_last.squeeze(0).cpu().numpy())
        dec_input = torch.cat([dec_input, mu_last], dim=1)
    return np.concatenate(mu_steps, axis=0), np.concatenate(sigma_steps, axis=0)


def _z_score(actual_norm, mu_pred, sigma_pred):
    return (np.abs(actual_norm - mu_pred) / (sigma_pred + 1e-8)).mean(axis=1)


def print_window_result(enc_actual_raw, dec_actual_norm, mu_pred, sigma_pred, threshold):
    dec_actual_raw = denormalise(dec_actual_norm)
    dec_pred_raw   = denormalise(mu_pred)
    dec_sigma_raw  = sigma_pred * _RNG
    z              = _z_score(dec_actual_norm, mu_pred, sigma_pred)
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


def save_plot(enc_actual_raw, dec_actual_norm, mu_pred, sigma_pred, mmsi, out_path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed -- skipping plot.")
        return
    dec_actual_raw = denormalise(dec_actual_norm)
    dec_pred_raw   = denormalise(mu_pred)
    dec_sigma_raw  = sigma_pred * _RNG
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


def main():
    parser = argparse.ArgumentParser(description="Predict and compare AIS trajectories.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--mmsi",   type=int)
    src.add_argument("--random", action="store_true")
    parser.add_argument("--window",            type=int,   default=0)
    parser.add_argument("--all-windows",       action="store_true")
    parser.add_argument("--plot",              action="store_true")
    parser.add_argument("--plot-out",          type=str,   default="trajectory.png")
    parser.add_argument("--anomaly-threshold", type=float, default=3.0)
    parser.add_argument("--db",                type=str,   default=cfg.data_path)
    parser.add_argument("--checkpoint",        type=str,   default=cfg.checkpoint_path)
    parser.add_argument("--device",            type=str,   default=cfg.device)
    args = parser.parse_args()

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

    if not os.path.exists(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    print(f"  Loading model from {args.checkpoint} ...")
    model  = load_model(args.checkpoint, args.device)
    print(f"  Device: {args.device}")

    if args.all_windows:
        total_dist = total_z = 0.0
        total_steps = flagged_tot = 0
        for wi, (enc, dec) in enumerate(windows):
            mu_pred, sigma_pred = predict_autoregressive(model, enc, args.device)
            z        = _z_score(dec, mu_pred, sigma_pred)
            dec_raw  = denormalise(dec)
            pred_raw = denormalise(mu_pred)
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
    enc_raw = denormalise(enc)
    print_window_result(enc_raw, dec, mu_pred, sigma_pred, args.anomaly_threshold)

    if args.plot:
        save_plot(enc_raw, dec, mu_pred, sigma_pred, args.mmsi, args.plot_out)


if __name__ == "__main__":
    main()
