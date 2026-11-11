#!/usr/bin/env bash
# Start the staging annotation page on this workstation (127.0.0.1 only).
# Usage: scripts/annotate.sh [WORKSPACE] [PORT]
# Runs in the background and keeps going if the terminal closes. The log is
# WORKSPACE/server.log. Stop it with: pkill -f staging.annotate.server
set -euo pipefail
cd "$(dirname "$0")/.."
WS="${1:-outputs/staging-annotation-2026-11-06-01}"
PORT="${2:-8765}"
if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  echo "Already running. Open http://127.0.0.1:${PORT}/"
  xdg-open "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 || true
  exit 0
fi
nohup .venv/bin/python -m eyespy_spermatogenesis.staging.annotate.server "$WS" \
  --port "$PORT" >>"$WS/server.log" 2>&1 &
echo "Started (PID $!). Open http://127.0.0.1:${PORT}/  Log: $WS/server.log"
