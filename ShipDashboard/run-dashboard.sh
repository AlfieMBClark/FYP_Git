#!/usr/bin/env bash
#
# run-dashboard.sh — start the Maritime Anomaly Dashboard (Flask) on :8050.
#
#   ./run-dashboard.sh              # default port 8050
#   ./run-dashboard.sh --port 9000
#   PYTHON=/path/to/venv/bin/python ./run-dashboard.sh   # pick the interpreter
#
# Optional in-app HTTP Basic Auth: set DASH_USER / DASH_PASS before running.
#
set -euo pipefail

# Resolve the dashboard dir from this script's own location so it runs from any
# machine / working directory.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Interpreter: honour $PYTHON, else a venv at the repo root (../../venv), else
# python3/python on PATH. Activate your environment or set PYTHON to override.
if [[ -n "${PYTHON:-}" ]]; then
    VENV_PYTHON="$PYTHON"
elif [[ -x "$PROJECT_DIR/../../venv/bin/python" ]]; then
    VENV_PYTHON="$PROJECT_DIR/../../venv/bin/python"
else
    VENV_PYTHON="$(command -v python3 || command -v python || true)"
fi

if [[ -z "$VENV_PYTHON" || ! -x "$VENV_PYTHON" ]]; then
    echo "error: no python interpreter found — set PYTHON=/path/to/python" >&2
    exit 1
fi

cd "$PROJECT_DIR"

echo "Starting dashboard  ->  http://localhost:8050"
echo "Press Ctrl+C to stop."
echo

# exec so Ctrl+C reaches Python directly and any --port/--host args pass through.
exec "$VENV_PYTHON" server.py "$@"
