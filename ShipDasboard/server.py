"""
server.py
---------
Flask server for the maritime anomaly dashboard.
http://localhost:8050

Usage:
    python server.py            # port 8050
    python server.py --port 9000

Exposing it publicly (via Tailscale Funnel — nothing to install):
    # 1) One time: publish port 8050 to the internet, persistently.
    tailscale funnel --bg 8050        # approve the enable-link if prompted (owner only)
    # 2) Run the server WITH a password so it isn't wide open:
    DASH_USER=admin DASH_PASS='fypais' python server.py
    # → live at https://<this-node>.<tailnet>.ts.net  (e.g. tclarkserver.tail1c7371.ts.net)
    # Stop exposing:  tailscale funnel off

Auth: set DASH_PASS to require HTTP Basic Auth on every request (DASH_USER
defaults to "admin"). Leave it unset for an open server on a trusted LAN/tailnet.
"""

import argparse
import hmac
import json
import os
import queue as _queue
import time
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory, Response

HERE       = Path(__file__).resolve().parent
SHIPS_PATH = HERE / "static/data/ships.json"

_TRANSFORMER  = HERE.parent / "ShipTransformer"
_DB_PATH      = str(_TRANSFORMER / "data2" / "2023.db")
_CKPT_PATH    = str(_TRANSFORMER / "checkpoints" / "best_model.pt")
_DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="")

# ── optional HTTP Basic Auth ───────────────────────────────────────────────────
# Gates every request when DASH_PASS is set in the environment. Leaving it unset
# keeps the dashboard open (fine on a trusted tailnet/LAN); set it before
# exposing the server publicly (e.g. via `tailscale funnel`). Username defaults
# to "admin" and can be overridden with DASH_USER.
#     DASH_USER=alice DASH_PASS='a-long-random-password' python server.py
_AUTH_USER = os.environ.get("DASH_USER", "admin")
_AUTH_PASS = os.environ.get("DASH_PASS")   # auth active only when this is set


def _auth_ok(auth) -> bool:
    if not auth or auth.username is None or auth.password is None:
        return False
    # Constant-time compares so a wrong username can't be timed against a right one.
    user_ok = hmac.compare_digest(auth.username, _AUTH_USER)
    pass_ok = hmac.compare_digest(auth.password, _AUTH_PASS)
    return user_ok and pass_ok


@app.before_request
def _require_auth():
    if not _AUTH_PASS:
        return  # auth disabled
    if not _auth_ok(request.authorization):
        return Response(
            "Authentication required.", 401,
            {"WWW-Authenticate": 'Basic realm="Maritime Dashboard"'},
        )


# ── simulation engine ─────────────────────────────────────────────────────────
from sim_engine import SimEngine   # noqa: E402
sim = SimEngine(_DB_PATH, _CKPT_PATH, _DEVICE)
# Load model + discover data span at startup so the first Start is instant.
sim._load_model()


# ── static routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/ships")
def ships():
    if not SHIPS_PATH.exists():
        return jsonify({"error": "ships.json not found — run precompute.py first"}), 404
    with open(SHIPS_PATH) as f:
        data = json.load(f)
    return jsonify(data)


# ── simulation SSE stream ──────────────────────────────────────────────────────

@app.route("/api/sim/events")
def sim_events():
    """Server-Sent Events stream. Each message is a JSON array of event objects."""
    client_q = sim.subscribe()

    def generate():
        try:
            while True:
                time.sleep(0.05)              # 20 fps max drain
                batch = []
                while True:
                    try:
                        batch.extend(client_q.get_nowait())
                    except _queue.Empty:
                        break
                if batch:
                    yield f"data: {json.dumps(batch)}\n\n"
                else:
                    yield ": hb\n\n"           # keep-alive comment
        except GeneratorExit:
            pass
        finally:
            sim.unsubscribe(client_q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── simulation control endpoints ──────────────────────────────────────────────

@app.route("/api/sim/start", methods=["POST"])
def sim_start():
    sim.start()
    return jsonify(sim.status())


@app.route("/api/sim/stop", methods=["POST"])
def sim_stop():
    sim.stop()
    return jsonify(sim.status())


@app.route("/api/sim/pause", methods=["POST"])
def sim_pause():
    sim.pause()
    return jsonify(sim.status())


@app.route("/api/sim/resume", methods=["POST"])
def sim_resume():
    sim.resume()
    return jsonify(sim.status())


@app.route("/api/sim/reset", methods=["POST"])
def sim_reset():
    sim.reset()
    return jsonify(sim.status())


@app.route("/api/sim/seek", methods=["POST"])
def sim_seek():
    data = request.get_json(force=True, silent=True) or {}
    ts = data.get("sim_ts")
    frac = data.get("fraction")
    # Allow seeking by absolute unix ts or by fraction of the data span.
    if ts is None and frac is not None and sim.data_start_ts and sim.data_end_ts:
        span = sim.data_end_ts - sim.data_start_ts
        ts = sim.data_start_ts + float(frac) * span
    if ts is not None:
        # If engine is stopped, start it first so the seek has a loop to act on.
        if sim._state in ("stopped", "done"):
            sim.start()
        sim.seek(ts)
    return jsonify(sim.status())


@app.route("/api/sim/configure", methods=["POST"])
def sim_configure():
    data = request.get_json(force=True, silent=True) or {}
    sim.configure(
        speed             = data.get("speed"),
        max_vessels       = data.get("max_vessels"),
        anomaly_threshold = data.get("anomaly_threshold"),
        predict_every     = data.get("predict_every"),
    )
    return jsonify(sim.status())


@app.route("/api/sim/status")
def sim_status():
    return jsonify(sim.status())


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    print(f"Dashboard : http://{args.host}:{args.port}")
    print(f"Ships JSON: {SHIPS_PATH}")
    print(f"Sim DB    : {_DB_PATH}")
    print(f"Checkpoint: {_CKPT_PATH}")
    print(f"Device    : {_DEVICE}")
    if _AUTH_PASS:
        print(f"Auth      : Basic Auth ENABLED (user '{_AUTH_USER}')")
    else:
        print("Auth      : DISABLED — set DASH_PASS before exposing publicly")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
