#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"

DEFAULT_HAMER_DIR="$SCRIPT_DIR/.external/hamer"
HAMER_DIR="${HAMER_DIR:-$DEFAULT_HAMER_DIR}"

export PORT="${PORT:-8080}"
export HOST="${HOST:-127.0.0.1}"
export KILL_OLD_PROCESSES="${KILL_OLD_PROCESSES:-1}"
export HAMER_DIR
export HAMER_CHECKPOINT="${HAMER_CHECKPOINT:-$HAMER_DIR/_DATA/hamer_ckpts/checkpoints/hamer.ckpt}"
export HAMER_BODY_DETECTOR="${HAMER_BODY_DETECTOR:-vitdet}"
export REQUIRE_HAMER_ASSETS="${REQUIRE_HAMER_ASSETS:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/novaplan-matplotlib}"
export PYTHONPATH="$HAMER_DIR${HAMER_EXTRA_PYTHONPATH:+:$HAMER_EXTRA_PYTHONPATH}"
mkdir -p "$MPLCONFIGDIR"

kill_listening_port() {
  local port="$1"
  local pids
  pids="$(ss -ltnp "sport = :$port" 2>/dev/null | sed -n "s/.*pid=\([0-9]\+\).*/\1/p" | sort -u)"
  if [[ -n "$pids" ]]; then
    echo "Killing process(es) listening on port $port: $pids"
    kill $pids || true
  fi
}

kill_hand_flow_server_processes() {
  local pids
  pids="$(pgrep -f 'remote_hand_flow/hand_flow_server.py' || true)"
  if [[ -n "$pids" ]]; then
    echo "Killing existing hand-flow service process(es): $pids"
    kill $pids || true
    sleep 2
    local remaining
    remaining="$(pgrep -f 'remote_hand_flow/hand_flow_server.py' || true)"
    if [[ -n "$remaining" ]]; then
      echo "Force killing remaining hand-flow service process(es): $remaining"
      kill -9 $remaining || true
    fi
  fi
}

if [[ ! -d "$HAMER_DIR/hamer" ]]; then
  echo "HAMER_DIR does not point to an installed HaMeR checkout: $HAMER_DIR" >&2
  echo "Run: MANO_RIGHT_SOURCE=/path/to/MANO_RIGHT.pkl ./remote_hand_flow/bootstrap_hand_flow_host.sh" >&2
  exit 1
fi

if [[ "$REQUIRE_HAMER_ASSETS" == "1" ]]; then
  if [[ ! -s "$HAMER_CHECKPOINT" ]]; then
    echo "Missing HaMeR checkpoint: $HAMER_CHECKPOINT" >&2
    echo "Run: DOWNLOAD_HAMER_MODELS=1 pixi run -e hand-flow-host setup-hand-flow-host" >&2
    exit 1
  fi
  if [[ ! -s "$HAMER_DIR/_DATA/data/mano/MANO_RIGHT.pkl" ]]; then
    echo "Missing MANO_RIGHT.pkl: $HAMER_DIR/_DATA/data/mano/MANO_RIGHT.pkl" >&2
    echo "Run: MANO_RIGHT_SOURCE=/path/to/MANO_RIGHT.pkl pixi run -e hand-flow-host setup-hand-flow-host" >&2
    exit 1
  fi
fi

if [[ "$KILL_OLD_PROCESSES" == "1" ]]; then
  kill_hand_flow_server_processes
  kill_listening_port "$PORT"
  sleep 1
fi

log_path="$REPO_ROOT/logs/hand_flow_server.log"
echo "Launching NovaPlan hand-flow service (HaMeR backend) on ${HOST}:${PORT}"
(
  cd "$HAMER_DIR"
  setsid python "$REPO_ROOT/remote_hand_flow/hand_flow_server.py" --host "$HOST" --port "$PORT" > "$log_path" 2>&1 < /dev/null &
  server_pid="$!"
  disown "$server_pid" || true
  echo "Server started. PID: $server_pid. Log: $log_path"
)
