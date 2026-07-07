#!/usr/bin/env bash
#
# run-dashboard.sh — start the Maritime Anomaly Dashboard (Flask + GPU) on :8050.
#
# The Cloudflare tunnel (systemd service `cloudflared-tunnel`) is always running and
# forwards https://fyp.meridian-ais.online to this server on localhost:8050.
# So: run this while you want the dashboard reachable; Ctrl+C to take it offline.
#
#   ./run-dashboard.sh              # default port 8050
#   ./run-dashboard.sh --port 9000  # (also update the tunnel config if you change this)
#
# Access is gated by Cloudflare Access at the edge, so no app password is needed.
# If you ever want in-app HTTP Basic Auth as well, set DASH_PASS before running:
#   DASH_USER=admin DASH_PASS='some-long-secret' ./run-dashboard.sh
#
set -euo pipefail

PROJECT_DIR="/home/aclark/alfie/ShipDashboard"
VENV_PYTHON="/home/aclark/alfie/venv/bin/python"

cd "$PROJECT_DIR"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "error: venv python not found at $VENV_PYTHON" >&2
    exit 1
fi

echo "Starting dashboard  ->  http://localhost:8050"
echo "Public (when up)    ->  https://fyp.meridian-ais.online"
echo "Press Ctrl+C to stop (this takes the site offline; the tunnel stays up)."
echo

# exec so Ctrl+C reaches Python directly and any --port/--host args pass through.
exec "$VENV_PYTHON" server.py "$@"
